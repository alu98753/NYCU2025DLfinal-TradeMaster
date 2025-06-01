import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

import torch
from torch import Tensor
from typing import Tuple
from ..builder import AGENTS
from ..custom import AgentBase
import random
from collections import namedtuple
from trademaster.utils import get_attr, GeneralReplayBuffer, get_optim_param
import wandb


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
        self.clip_grad_norm = get_attr(kwargs, "clip_grad_norm", 5.0)
        self.soft_update_tau = get_attr(kwargs, "soft_update_tau",
                                          0)
        self.state_value_tau = get_attr(kwargs, "state_value_tau", 5e-3)

        self.last_state = None
        
        # MarketNet, Actor, Critic 初始化
        self.market = get_attr(kwargs, "market", None)
        if self.market is not None:
            self.market = self.market.to(self.device)
            self.market.eval() # <--- 修改點1: 初始化後，如果 market 存在，預設為 eval 模式
        assert self.market is not None, "配置文件里必须给出 market: {type='MarketNet', market_lr=...}" # 如果允許 market 為 None，則此 assert 需調整

        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None).to(self.device)
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)
        
        # MarketNet Optimizer 和 Criterion
        # 這些可能會在 Trainer 中被重新初始化或使用，如果 MarketNet 需要重新訓練
        # MarketNet Optimizer 和 Criterion (Trainer 會使用這些)
        if self.market is not None and hasattr(self.market, 'market_lr'):
            # 初始優化器，Trainer 可能會基於 MarketNet 是否解凍來重新配置它
            self.market_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.market.parameters()), # 只優化需要梯度的參數
                lr=self.market.market_lr 
            )
        else:
            self.market_optimizer = None # 如果 market 不存在或沒有 market_lr，則不初始化
        
        self.market_criterion = torch.nn.CrossEntropyLoss() # 即使 market 為 None，也先初始化，Trainer 中可能用到
        # --- LR Scheduler 和 Warmup 初始化 ---
        self.use_lr_scheduler = get_attr(kwargs, "use_lr_scheduler", False)
        self.warmup_steps = int(get_attr(kwargs, "warmup_steps", 0))
        
        self.initial_lr_actor = self.act_optimizer.param_groups[0]['lr'] if self.act_optimizer else 0
        self.initial_lr_critic = self.cri_optimizer.param_groups[0]['lr'] if self.cri_optimizer else 0
        
        self.optimizer_steps = 0

        self.act_lr_scheduler = None
        self.cri_lr_scheduler = None

        if self.use_lr_scheduler and self.act_optimizer and self.cri_optimizer:
            pass

        self.criterion = get_attr(kwargs, "criterion", None) # Actor-Critic 的主損失函數 (MSELoss for TD error)
        self.transition = namedtuple("Transition", [
            'state',          # [horizon_len, num_envs, action_dim, time_steps, state_dim]
            'action',         # [horizon_len, num_envs, action_dim + 1]
            'reward',         # [horizon_len, num_envs]
            'undone',         # [horizon_len, num_envs]
            'next_state',     # [horizon_len, num_envs, action_dim, time_steps, state_dim]
            's_market',       # [horizon_len, num_envs, s_market_dim]
            'next_s_market',  # [horizon_len, num_envs, s_market_dim]
            'regime',         # [horizon_len, num_envs] —— regime_t
            'next_regime'     # [horizon_len, num_envs] —— regime_{t+1}
        ])
        
    def _adjust_learning_rate(self, optimizer, initial_lr):
        if self.optimizer_steps < self.warmup_steps:
            lr_scale = float(self.optimizer_steps + 1) / float(self.warmup_steps)
            current_lr = initial_lr * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif self.optimizer_steps == self.warmup_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr

    def get_save(self):
        models = {
            "act": self.act,
            "cri": self.cri,
            "market": self.market # 如果需要保存 MarketNet
        }
        optimizers = {
            "act_optimizer": self.act_optimizer,
            "cri_optimizer": self.cri_optimizer,
            "market_optimizer": self.market_optimizer # 如果需要保存 MarketNet 優化器狀態
        }
        # 如果 market 也保存，models 和 optimizers 中應加入
        if self.market:
            models["market"] = self.market
        if self.market_optimizer:
            optimizers["market_optimizer"] = self.market_optimizer
            
        res = {
            "models": models,
            "optimizers": optimizers
        }
        return res

    def explore_env(self, env, horizon_len: int, global_step) -> Tuple[Tensor, ...]:
        D_s_market = self.act.s_market_dim

        s_markets = torch.zeros((horizon_len, self.num_envs, D_s_market), dtype=torch.float32).to(self.device)
        next_s_markets = torch.zeros((horizon_len, self.num_envs, D_s_market), dtype=torch.float32).to(self.device)
        states = torch.zeros((horizon_len, self.num_envs, self.action_dim, self.time_steps, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim + 1), dtype=torch.float32).to(self.device) # 通常 action 是 float
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        next_states = torch.zeros((horizon_len, self.num_envs, self.action_dim, self.time_steps, self.state_dim), dtype=torch.float32).to(self.device)
        regimes = torch.zeros((horizon_len, self.num_envs), dtype=torch.long, device=self.device)
        next_regimes = torch.zeros((horizon_len, self.num_envs), dtype=torch.long, device=self.device)
    
        state = self.last_state
        
        # --- 修改點2: 在 explore_env 循環前設置 market 為 eval 模式 ---
        if self.market:
            self.market.eval()

        for t in range(horizon_len):
            with torch.no_grad(): # 所有推斷操作都在 no_grad 上下文中
                current_s_market = self.act.s_market_stock_pool(self.act.temporal_feature_extractor(state, global_step=None))
                
                if self.market:
                    regime_probs, _ = self.market(current_s_market)
                else:
                    num_regimes_actor_expects = self.act.num_regimes if hasattr(self.act, 'num_regimes') else 2
                    if num_regimes_actor_expects > 0:
                        regime_probs = torch.ones(current_s_market.size(0), num_regimes_actor_expects, device=self.device) / num_regimes_actor_expects
                    else:
                        regime_probs = torch.empty(current_s_market.size(0), 0, device=self.device)
            
                regime_probs = regime_probs.detach() 
                action_probs, _ = self.act(state, regime_probs, global_step=global_step) # Actor 可能仍需要 global_step for its own logging/logic
            
                regime_probs = regime_probs.detach() # 確保 regime_probs 無梯度
                action_probs, _ = self.act(state, regime_probs, global_step=global_step)
            
            states[t] = state
            s_markets[t] = current_s_market.detach()

            ary_action = action_probs.detach().cpu().numpy()
            if self.num_envs == 1:
                ary_action = ary_action[0]

            next_state_ary, reward_val, done_val, info = env.step(ary_action) # 重命名以避免與 tensor rewards 衝突
            
            regimes[t, :] = torch.tensor(info["regime_t"], device=self.device, dtype=torch.long)
            next_regimes[t, :] = torch.tensor(info["regime_next"], device=self.device, dtype=torch.long)

            if self.num_envs == 1:
                if done_val:
                    state = torch.as_tensor(env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    state = torch.as_tensor(next_state_ary, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                raise NotImplementedError("Multi-environment S_market handling in explore_env not fully detailed here.")

            with torch.no_grad(): # 推斷過程不計算梯度
                 next_s_market_val = self.act.s_market_stock_pool(self.act.temporal_feature_extractor(state, global_step=None))

            actions[t] = action_probs # action_probs 是 float Tensor
            rewards[t] = reward_val
            dones[t] = done_val
            next_states[t] = state
            next_s_markets[t] = next_s_market_val.detach()

        self.last_state = state.detach()
        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)

        transition = self.transition(
            state=states, action=actions, reward=rewards, undone=undones, next_state=next_states,
            s_market=s_markets, next_s_market=next_s_markets, regime=regimes, next_regime=next_regimes
        )
        # print(f">>> explore_env collected {horizon_len} transitions; s_markets[0]={s_markets[0 if horizon_len > 0 else None]}")
        return transition

    def update_net(self, buffer: GeneralReplayBuffer, global_step: int):
        obj_critics = 0.0
        obj_actors = 0.0
        update_times = int(buffer.add_size * self.repeat_times) # buffer.add_size 可能是0
        assert update_times >= 1, f"update_times is {update_times}, buffer.add_size is {buffer.add_size}, buffer.size is {buffer.size}"
        for _ in range(update_times):
            if self.use_lr_scheduler:
                if self.optimizer_steps % 100 == 0:
                    if self.act_optimizer:
                        self._adjust_learning_rate(self.act_optimizer, self.initial_lr_actor)
                        wandb.log({"Learning_Rate/Actor_LR": self.act_optimizer.param_groups[0]['lr'], 
                                "agent_step": self.optimizer_steps}) # 移除 step=...
                    if self.cri_optimizer:
                        self._adjust_learning_rate(self.cri_optimizer, self.initial_lr_critic)
                        wandb.log({"Learning_Rate/Critic_LR": self.cri_optimizer.param_groups[0]['lr'],
                                "agent_step": self.optimizer_steps}) # 移除 step=...
            
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size, self.optimizer_steps)
            # --- 在優化器 step 之後，如果配置了衰減調度器，則調用 scheduler.step() ---

            self.optimizer_steps += 1
            
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item() # q_value 是 q_target，是一個 tensor
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer: GeneralReplayBuffer, batch_size: int, global_step: int) -> Tuple[Tensor, Tensor]:
        """
        1) 先用 MarketNet 更新市场分类损失（CrossEntropyLoss），并记录 loss & accuracy。
        2) 再用 Actor+Critic 做 DDPG 的更新。
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        # print(f">>> get_obj_critic called, buffer size = {buffer.size}, add_size = {buffer.add_size}, global_step = {self.optimizer_steps}")
        # sample = buffer.sample(1) # Debugging sample
        # print(">>> One sampled transition from get_obj_critic:",
        #       "state shape=", sample.state.shape,
        #       "reward=", sample.reward[0],
        #       "s_market shape=", sample.s_market.shape,
        #       "regime=", sample.regime[0])

        transition = buffer.sample(batch_size) # 使用傳入的 batch_size

        state = transition.state
        action = transition.action
        reward = transition.reward
        undone = transition.undone
        next_state = transition.next_state
        s_market = transition.s_market
        next_s_market = transition.next_s_market
        regime_labels = transition.regime
        
        # --- 修改點4: 在 get_obj_critic 中推斷 market 前設置為 eval 模式 (如果 market 不在此處訓練) ---
        # ======= 【Step A: MarketNet 推理 (如果存在且不在此處訓練) 或 MarketNet 訓練 (如果 Trainer 觸發) 】 ======= #
        if self.market:
            self.market.eval() # 確保在推斷時是 eval 模式
            with torch.no_grad():
                s_market_flat = s_market.view(s_market.size(0), -1) # 使用 s_market.size(0) 作為 batch size
                regime_probs_curr, logits = self.market(s_market_flat)
                
                # 計算準確率 (用於監控)
                preds = torch.argmax(logits, dim=1)
                total_correct = (preds == regime_labels).float().sum()
                overall_acc = total_correct / float(preds.size(0))
                
                per_class_acc = {}
                class_names = ["Bull", "Bear"] # 假設 MarketNet 輸出是2分類
                num_actual_regimes = logits.size(1)
                for cls_id in range(num_actual_regimes):
                    cls_name_key = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
                    mask = (regime_labels == cls_id)
                    num_cls_samples = mask.sum().item()
                    if num_cls_samples > 0:
                        correct_cls = ((preds == regime_labels) & mask).float().sum().item()
                        per_class_acc[cls_name_key] = correct_cls / num_cls_samples
                    else:
                        per_class_acc[cls_name_key] = float("nan")
                if self.optimizer_steps % 100 == 0:

                    log_dict_market = {
                        # "Market/Market_CE_Inference": self.market_criterion(logits, regime_labels).item(), # 可選：記錄推斷時的CE
                        "Market/Market_OverallAcc_Inference": overall_acc.item(),
                        "agent_step": self.optimizer_steps
                    }
                    for cls_name_key, acc_val in per_class_acc.items():
                        log_dict_market[f"Market/Acc_{cls_name_key}_Inference"] = acc_val
                    wandb.log(log_dict_market)
        else: # self.market is None
            num_regimes_actor_expects = self.act.num_regimes if hasattr(self.act, 'num_regimes') else 2
            if num_regimes_actor_expects > 0:
                 regime_probs_curr = torch.ones(state.size(0), num_regimes_actor_expects, device=self.device) / num_regimes_actor_expects
            else:
                regime_probs_curr = torch.empty(state.size(0), 0, device=self.device)
            wandb.log({"Market/Market_OverallAcc_Inference": float("nan"), "agent_step": self.optimizer_steps})
        
        regime_probs_curr = regime_probs_curr.detach()

        # —— Step B: Actor 更新 —— #
        a, _ = self.act(state, regime_probs_curr, global_step=self.optimizer_steps)
        if self.optimizer_steps % 1000  == 0:
            action_weights_detached = a.detach().cpu().numpy()

            wandb.log({
                "Action/Portfolio_Weights": wandb.Histogram(action_weights_detached),
                "Action/Cash_Weight": action_weights_detached[-1, 0] if action_weights_detached.ndim == 2 else action_weights_detached[-1], # 假設現金權重在第一個位置
                "Action/stock1_Weight": action_weights_detached[1, 0] if action_weights_detached.ndim == 2 else action_weights_detached[0], # 假設現金權重在第一個位置
                "Action/stock2_Weight": action_weights_detached[2, 0] if action_weights_detached.ndim == 2 else action_weights_detached[0], # 假設現金權重在第一個位置
                "agent_step": self.optimizer_steps
            }) # 移除 step=...
        
        # Critic 輸入的 s_market 應該與 state 對應
        q = self.cri(state, a, regime_probs_curr.detach()) # 確保 cri 的 s_market 輸入與 state 匹配
        a_loss = -torch.mean(q)

        self.act_optimizer.zero_grad()
        a_loss.backward()
        # ... (Log Actor Gradients and clip) ...
        if self.optimizer_steps % 1000  == 0: # Log every 100 optimizer steps
            actor_grad_abs_means = {}
            actor_grad_stds = {}
            # Log for HCAR_Actor.temporal_feature_extractor.start_conv.weight (example)
            # You need to access the specific layer you're interested in.
            # For start_linear in RelationalContextIntegrator:
            start_linear_layer = self.act.relational_context_integrator.start_linear
            if start_linear_layer.weight.grad is not None:
                actor_grad_abs_means["HCAR_Actor/3B_Relational/StartLinear_Weight_grad_AbsMean"] = start_linear_layer.weight.grad.abs().mean().item()
                actor_grad_stds["HCAR_Actor/3B_Relational/StartLinear_Weight_grad_Std"] = start_linear_layer.weight.grad.std().item() # Using .std() directly, not .abs().std()
            else:
                actor_grad_abs_means["HCAR_Actor/3B_Relational/StartLinear_Weight_grad_AbsMean"] = 0
                actor_grad_stds["HCAR_Actor/3B_Relational/StartLinear_Weight_grad_Std"] = 0

            # Log overall Actor gradient norm (as you had before)
            total_norm_act_before_clip = 0
            for p in self.act.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_act_before_clip += param_norm.item() ** 2
            total_norm_act_before_clip = total_norm_act_before_clip ** 0.5
            actor_grad_abs_means["Gradients/Actor_Grad_Norm_Before_Clip"] = total_norm_act_before_clip

            # You can add more specific layer gradients here if needed
            # Example: For the first TCN conv layer in TemporalFeatureExtractor
            # first_tcn_conv = self.act.temporal_feature_extractor.tcn_blocks[0] # Assuming direct access
            # if first_tcn_conv.weight.grad is not None:
            #     actor_grad_abs_means["HCAR_Actor/3A_Temporal/TCN0_Conv_grad_AbsMean"] = first_tcn_conv.weight.grad.abs().mean().item()
            if self.optimizer_steps % 100 == 0:
                wandb.log({**actor_grad_abs_means, **actor_grad_stds, "agent_step": self.optimizer_steps})

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), self.clip_grad_norm)
            # Optionally log Actor_Grad_Norm_After_Clip here        
            total_norm_act_after_clip = 0
            for p in self.act.parameters():
                if p.grad is not None: # 裁剪後梯度仍然存在
                    param_norm = p.grad.data.norm(2)
                    total_norm_act_after_clip += param_norm.item() ** 2
            total_norm_act_after_clip = total_norm_act_after_clip ** 0.5
            if self.optimizer_steps % 100 == 0:
                wandb.log({"Gradients/Actor_Grad_Norm_After_Clip": total_norm_act_after_clip, "agent_step": self.optimizer_steps})
  
        self.act_optimizer.step()

        # —— Step C: Critic 更新 —— #
        if self.market:
            # 再次確保是 eval 模式，如果 MarketNet 只是推斷
            # self.market.eval() # 如果 Trainer 沒有在訓練它，這裡應為 eval
            with torch.no_grad():
                if next_s_market.ndim == 3 and next_s_market.size(1) == 1:
                    next_s_market_flat = next_s_market.squeeze(1)
                else:
                    next_s_market_flat = next_s_market.view(state.size(0), -1) # 使用 state.size(0) 獲取實際 batch size
                regime_probs_next, _ = self.market(next_s_market_flat)
        else: # self.market is None
            num_regimes_actor_expects = self.act.num_regimes if hasattr(self.act, 'num_regimes') else 2
            if num_regimes_actor_expects > 0:
                 regime_probs_next = torch.ones(state.size(0), num_regimes_actor_expects, device=self.device) / num_regimes_actor_expects
            else:
                regime_probs_next = torch.empty(state.size(0), 0, device=self.device)

        regime_probs_next = regime_probs_next.detach()

        with torch.no_grad(): # Target networks or inputs for target should not have grads
            a_, _ = self.act(next_state, regime_probs_next, global_step=self.optimizer_steps)
            # next_s_market 應與 next_state 對應
            q_ = self.cri(next_state, a_.detach(), regime_probs_next.detach())
            q_target = reward + undone * self.gamma * q_ # undone 應該在這裡使用
        
        q_eval = self.cri(state, action.detach(), regime_probs_curr.detach()) # s_market 應與 state 對應
        td_error = self.criterion(q_eval, q_target.detach()) # 通常是 (q_eval, q_target.detach())

        self.cri_optimizer.zero_grad()
        td_error.backward()
        # ... (Log Critic Gradients and clip) ...
        if self.cri_optimizer and self.optimizer_steps % 1000  == 0 : # 檢查是否存在且定期記錄
            total_norm_cri_before_clip = 0
            for p in self.cri.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_cri_before_clip += param_norm.item() ** 2
            total_norm_cri_before_clip = total_norm_cri_before_clip ** 0.5
            if self.optimizer_steps % 100 == 0:
                wandb.log({"Gradients/Critic_Grad_Norm_Before_Clip": total_norm_cri_before_clip, "agent_step": self.optimizer_steps})

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.cri.parameters(), self.clip_grad_norm)
                total_norm_cri_after_clip = 0
                for p in self.cri.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_cri_after_clip += param_norm.item() ** 2
                total_norm_cri_after_clip = total_norm_cri_after_clip ** 0.5
                if self.optimizer_steps % 100 == 0:
                    wandb.log({"Gradients/Critic_Grad_Norm_After_Clip": total_norm_cri_after_clip, "agent_step": self.optimizer_steps})
     

        self.cri_optimizer.step()
        if self.optimizer_steps % 100 == 0:

            wandb.log({
                "Loss/Actor_Loss": a_loss.item(), # a_loss 已經是 -torch.mean(q)
                "Loss/Critic_Loss": td_error.item(),
                "Values/Mean_Q_Value_from_buffer_action": q_eval.mean().item(),
                "Values/Mean_Q_Value_from_current_policy_action": q.mean().item(),
                "agent_step": self.optimizer_steps
            })

        return td_error, q_target # q_target 是一個 tensor