import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple
from ..builder import AGENTS
from ..custom import AgentBase
import random
from collections import namedtuple
from trademaster.utils import get_attr, GeneralReplayBuffer, get_optim_param

class Transition(NamedTuple): # 使用 typing.NamedTuple 以支持類型提示
    state: Tensor
    action: Tensor
    reward: Tensor
    undone: Tensor
    next_state: Tensor
    s_market: Tensor         # 新增: [B, d_model]
    next_s_market: Tensor    # 新增: [B, d_model]
    # 可選：也可以直接存儲 MarketNet 的預測結果 (regime_signal)
    # market_regime_signal_pred: Optional[Tensor]
    # next_market_regime_signal_pred: Optional[Tensor]
    # 但目前計劃是從 s_market 即時推斷 regime_signal

@AGENTS.register_module()
class PortfolioManagementEIIE(AgentBase):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIE, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_step = get_attr(kwargs, "max_step",
                                 12345)  # the max step number of an episode. 'set as 12345 in default.
        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)
        self.time_steps = get_attr(kwargs, "time_steps", 10)

        '''Arguments for reward shaping'''
        self.gamma = get_attr(kwargs, "gamma", 0.99)  # discount factor of future rewards
        self.reward_scale = get_attr(kwargs, "reward_scale",
                                     2 ** 0)  # an approximate target reward usually be closed to 256
        self.repeat_times = get_attr(kwargs, "repeat_times", 1.0)  # repeatedly update network using ReplayBuffer
        self.batch_size = int(get_attr(kwargs, "batch_size", 64))
        self.clip_grad_norm = get_attr(kwargs, "clip_grad_norm", 3.0)  # clip the gradient after normalization
        self.soft_update_tau = get_attr(kwargs, "soft_update_tau",
                                        0)  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = get_attr(kwargs, "state_value_tau", 5e-3)  # the tau of normalize for value and state

        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None).to(self.device)
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)
        # <<< 新增 MarketNet 初始化 >>>
        self.market_net = get_attr(kwargs, "market_net", None) # SimpleMarketNetBN instance
        if self.market_net is not None:
            self.market_net = self.market_net.to(self.device)
            self.market_net.eval() # MarketNet 在 Agent 中主要用於推斷，默認 eval 模式
        else:
            # 允許 MarketNet 為 None，此時 Actor/Critic 的 market_regime_signal 將為 None
            print("Info: MarketNet is not provided to the EIIE Agent. Regime-based logic in Actor/Critic will use defaults or handle None.")
        # <<< 新增結束 >>>

        # 獲取 s_market 的維度 (應等於 EIIEConv 的 d_model)
        if hasattr(self.act, 'd_model'):
            self.d_model_for_s_market = self.act.d_model
        else:
            # 嘗試從配置中獲取，或使用一個合理的默認值
            # 這個值對於定義 ReplayBuffer 中 s_market 的形狀至關重要
            actor_config = get_attr(kwargs, "cfg", {}).get("act", {}) # 假設 cfg 傳入了 kwargs
            self.d_model_for_s_market = actor_config.get("d_model", 128) # 例如從act配置讀取，或默認128
            print(f"Warning: self.act.d_model not directly found. Inferring/defaulting d_model_for_s_market to {self.d_model_for_s_market}. Ensure this matches EIIEConv's d_model.")

        
        self.criterion = get_attr(kwargs, "criterion", None)

        # self.transition = get_attr(kwargs, "transition", namedtuple("Transition", ['state','action','reward','undone','next_state']))
        self.transition = Transition 
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)

        self.explore_noise_std = get_attr(kwargs, "explore_noise_std", 0.05) # 引入可配置的噪聲標準差


    def get_save(self):
        models = {
            "act":self.act,
            "cri":self.cri,
            "market_net":self.market_net
        }
        optimizers = {
            "act_optimizer":self.act_optimizer,
            "cri_optimizer":self.cri_optimizer
        }
        res = {
            "models":models,
            "optimizers":optimizers
        }
        return res
    
    @torch.no_grad()
    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        # 初始化存儲張量
        # 狀態 state: [horizon_len, num_envs, N, T, F_input_actor]
        # s_market:    [horizon_len, num_envs, d_model]
        # 動作 action: [horizon_len, num_envs, N+1] (股票權重 + 現金權重)

        states = torch.zeros((horizon_len,
                              self.num_envs,
                              self.action_dim,
                              self.time_steps,
                              self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim + 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32, device=self.device)
        undones = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32, device=self.device) # undone (1-done)
        next_states = torch.zeros_like(states)

        # <<< 新增 s_market 存儲 >>>
        s_markets = torch.zeros(
            (horizon_len, self.num_envs, self.d_model_for_s_market), # d_model_for_s_market 應為 EIIEConv 的 d_model
            dtype=torch.float32, device=self.device
        )
        next_s_markets = torch.zeros_like(s_markets)
        # <<< 新增結束 >>>

        current_env_state_tensor  = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        # 確保 Actor 和 MarketNet 在評估模式
        # 
        
        # train for no noise
        # self.act.train()
        
        # train for noise
        self.act.eval()
        
        #--------------
        
        if self.market_net:
            self.market_net.eval()
            
        for t in range(horizon_len):
            # 1. 從當前環境狀態 current_env_state_tensor 提取 s_market
            current_s_market_tensor = self.act.extract_market_features(current_env_state_tensor) # [B, d_model]

            # 2. 使用 MarketNet 預測市場狀態信號
            market_regime_signal_tensor: Optional[torch.Tensor] = None
            if self.market_net:
                market_logits = self.market_net(current_s_market_tensor) # [B, num_classes]
                market_regime_signal_tensor = torch.argmax(market_logits, dim=1) # [B], 值為 0 (牛) 或 1 (熊)
            else:
                # 如果沒有 MarketNet，Actor 的 forward 應能處理 signal=None
                # 或者我們可以提供一個默認值，例如總是“牛市”(0)
                # market_regime_signal_tensor = torch.zeros(current_env_state_tensor.size(0), dtype=torch.long, device=self.device) # 默認為牛市
                pass # 讓 market_regime_signal_tensor 保持 None，由 Actor 內部處理

            # 3. Actor 根據環境狀態和市場狀態信號決定動作
            # Actor 的 forward 方法需要修改為 def forward(self, x, market_regime_signal)
            action_probs_tensor = self.act(current_env_state_tensor, market_regime_signal_tensor) # [B, N+1]

            # --- 添加探索噪聲 (這是關鍵!) ---
            action_probs_tensor_with_noise = action_probs_tensor + torch.randn_like(action_probs_tensor) * self.explore_noise_std
            action_probs_tensor_with_noise = torch.clamp(action_probs_tensor_with_noise, min=0.0)
            sum_of_weights = action_probs_tensor_with_noise.sum(dim=1, keepdim=True)
            sum_of_weights = torch.where(sum_of_weights == 0, torch.ones_like(sum_of_weights), sum_of_weights)
            action_probs_tensor = action_probs_tensor_with_noise / sum_of_weights
            # ---------------------------------
            
            states[t] = current_env_state_tensor
            s_markets[t] = current_s_market_tensor
            actions[t] = action_probs_tensor # 存儲 agent 輸出的原始動作概率/權重
            action_np = action_probs_tensor.detach().cpu().numpy()

            # 與環境交互 (假設是單環境 self.num_envs = 1)
            action_to_env = action_np[0] # 取出第一個 (也是唯一一個) 動作
            next_env_state_np, reward_val, done_val, info_dict = env.step(action_to_env)
            # 如果 done，env.reset() 返回的是 numpy array
            # 如果 not done，next_env_state_np 是 numpy array
            current_env_state_np = env.reset() if done_val else next_env_state_np
            current_env_state_tensor = torch.as_tensor(current_env_state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
           
            rewards[t, 0] = reward_val # 假設單環境
            undones[t, 0] = (1.0 - done_val) # undone = 1.0 if not done else 0.0
            next_states[t] = current_env_state_tensor # 存儲交互後的下一個狀態

            # 4. 為 next_state 提取 next_s_market
            next_s_market_tensor = self.act.extract_market_features(current_env_state_tensor)
            next_s_markets[t] = next_s_market_tensor

        self.last_state = current_env_state_tensor.detach() # 保存最後的狀態以備下次調用

        rewards_scaled = rewards * self.reward_scale # 縮放獎勵

        return self.transition(states, actions, rewards_scaled, undones, next_states,
                               s_markets, next_s_markets) # 返回包含 s_market 的 Transition


    def update_net(self, buffer: GeneralReplayBuffer):
        obj_critics = 0.0
        obj_actors = 0.0
        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer: GeneralReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        transition = buffer.sample(batch_size) # 從 buffer 中採樣一個批次的 Transition

        state_batch = transition.state    # [B, N, T, F_actor_input]
        action_batch = transition.action  # [B, N+1] (actor 產生的原始動作)
        reward_batch = transition.reward  # [B, 1] or [B]
        undone_batch = transition.undone  # [B, 1] or [B]
        next_state_batch = transition.next_state # [B, N, T, F_actor_input]
        
        s_market_batch = transition.s_market # [B, d_model]
        next_s_market_batch = transition.next_s_market # [B, d_model]

        # 確保 MarketNet 在評估模式
        if self.market_net:
            self.market_net.eval()

        # 1. 從 s_market 推斷當前市場狀態信號
        market_regime_signal_curr_tensor: Optional[torch.Tensor] = None
        if self.market_net:
            with torch.no_grad(): # MarketNet 不參與此處的梯度更新
                market_logits_curr = self.market_net(s_market_batch)
                market_regime_signal_curr_tensor = torch.argmax(market_logits_curr, dim=1)
        
        # Actor Loss: -E[Q(s, mu(s|regime))]
        # Actor 網絡設為訓練模式以計算梯度
        self.act.train()
        # Critic 網絡在計算 Actor Loss 時，其參數是固定的，但它本身也參與損失計算並更新，所以也應該是 train 模式
        # 或者，如果用的是 target critic，則 target critic 是 eval 模式
        # 你的 EIIE 似乎沒有 target networks，所以 cri 是 train 模式
        self.cri.train()

        # Actor 根據 state_batch 和推斷出的 market_regime_signal_curr_tensor 產生新動作
        # 注意：這裡的 state_batch 是從 buffer 來的，action_batch 也是從 buffer 來的 (是過去的動作)
        # 計算 Actor loss 時，我們需要 Actor 對 state_batch 產生 *新* 的動作
        current_actions_from_actor = self.act(state_batch, market_regime_signal_curr_tensor)
        
        # Critic 評估這些新動作的 Q 值
        # Critic 的 forward 方法需要修改為 forward(self, state, action, market_regime_signal)
        q_values_for_actor_loss = self.cri(state_batch, current_actions_from_actor, market_regime_signal_curr_tensor)
        actor_loss = -q_values_for_actor_loss.mean()

        self.act_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.act.parameters(), self.clip_grad_norm)
        self.act_optimizer.step()

        # Critic Loss: E[(r + gamma * Q_target(s', mu'(s'|regime')) - Q(s,a|regime))^2]
        # 計算 Q_target 時，Actor 和 Critic (以及 MarketNet) 都應處於 eval 模式，且不計算梯度
        self.act.eval() 
        # self.cri.eval() # 計算 target Q 值時，critic 應為 eval (或使用 target critic)
                        # 如果沒有 target critic，這裡用 eval 的 cri 可能會導致不穩定
                        # DDPG 標準做法是有 target_actor 和 target_critic
                        # 你的 EIIE 沒有 soft_update_tau，可能意味著沒有 target networks
                        # 如果是這樣，計算 Q_target 時的 a_ 和 Q_ 仍然用當前的 act 和 cri，但用 no_grad()
        # 從 next_s_market 推斷下一市場狀態信號
        market_regime_signal_next_tensor: Optional[torch.Tensor] = None
        if self.market_net:
            with torch.no_grad():
                market_logits_next = self.market_net(next_s_market_batch)
                market_regime_signal_next_tensor = torch.argmax(market_logits_next, dim=1)

        with torch.no_grad(): # Target Q 值計算不應產生梯度
            next_actions_from_actor = self.act(next_state_batch, market_regime_signal_next_tensor)
            q_values_next = self.cri(next_state_batch, next_actions_from_actor, market_regime_signal_next_tensor)
            
            if reward_batch.dim() == 1: reward_batch = reward_batch.unsqueeze(-1) # [B,1]
            if undone_batch.dim() == 1: undone_batch = undone_batch.unsqueeze(-1) # [B,1]
            
            q_target = reward_batch + self.gamma * undone_batch * q_values_next # [B,1]
        
        # Critic 網絡恢復訓練模式以計算梯度
        self.cri.train()
        # Critic 評估 (state_batch, action_batch 来自 buffer, market_regime_signal_curr_tensor) 的 Q 值
        # action_batch 是實際執行的歷史動作
        q_eval = self.cri(state_batch, action_batch.detach(), market_regime_signal_curr_tensor) # [B,1]
        
        critic_loss = self.criterion(q_eval, q_target.detach()) # MSELoss

        self.cri_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.cri.parameters(), self.clip_grad_norm)
        self.cri_optimizer.step()
        
        # Actor 恢復訓練模式，為下一次 explore_env 做準備 (如果 explore_env 會切換模式)
        # self.act.train() # 通常在 explore_env 或 train_and_valid 循環開始時設置

        return critic_loss, q_target # 返回 critic_loss 和 q_target (用於日誌)