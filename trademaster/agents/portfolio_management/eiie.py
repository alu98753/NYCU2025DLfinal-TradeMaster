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
        self.clip_grad_norm = get_attr(kwargs, "clip_grad_norm", 5.0)  # clip the gradient after normalization # origin: 3.0 # deeptrader 100.0
        self.soft_update_tau = get_attr(kwargs, "soft_update_tau",
                                        0)  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = get_attr(kwargs, "state_value_tau", 5e-3)  # the tau of normalize for value and state

        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)

        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None).to(self.device)
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)

        # --- LR Scheduler 和 Warmup 初始化 ---
        self.use_lr_scheduler = get_attr(kwargs, "use_lr_scheduler", False)
        self.warmup_steps = int(get_attr(kwargs, "warmup_steps", 0))
        
        self.initial_lr_actor = self.act_optimizer.param_groups[0]['lr'] if self.act_optimizer else 0
        self.initial_lr_critic = self.cri_optimizer.param_groups[0]['lr'] if self.cri_optimizer else 0
        
        self.optimizer_steps = 0 # 用於追蹤優化器更新的總步數

        self.act_lr_scheduler = None
        self.cri_lr_scheduler = None

        if self.use_lr_scheduler and self.act_optimizer and self.cri_optimizer:
            # 示例：Warmup 之後使用 CosineAnnealingLR 進行衰減
            # lr_decay_scheduler_type = get_attr(kwargs, "lr_decay_scheduler_type", "CosineAnnealingLR")
            # total_training_steps = get_attr(kwargs, "total_training_steps_for_scheduler", 50000) # 估算的總優化步數
            
            # 簡單起見，我們先只關注 Warmup，衰減部分可以後續添加
            # 如果要添加衰減調度器，例如 CosineAnnealingLR：
            # decay_t_max = total_training_steps - self.warmup_steps
            # if decay_t_max > 0:
            #     self.act_lr_scheduler = lr_scheduler.CosineAnnealingLR(self.act_optimizer, T_max=decay_t_max, eta_min=self.initial_lr_actor * 0.01)
            #     self.cri_lr_scheduler = lr_scheduler.CosineAnnealingLR(self.cri_optimizer, T_max=decay_t_max, eta_min=self.initial_lr_critic * 0.01)
            pass # 暫時不初始化衰減調度器，只做 warmup

        self.criterion = get_attr(kwargs, "criterion", None)
        self.transition = namedtuple("Transition", ['state', 'action', 'reward', 'undone', 'next_state','s_market', 'next_s_market'])
        
    def _adjust_learning_rate(self, optimizer, initial_lr):
        """手動調整學習率以實現 Warmup"""
        if self.optimizer_steps < self.warmup_steps:
            # 線性 warmup
            lr_scale = float(self.optimizer_steps + 1) / float(self.warmup_steps) # 從 step 1 開始
            current_lr = initial_lr * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif self.optimizer_steps == self.warmup_steps: # Warmup 結束，恢復到初始(目標)LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        # else: Warmup 之後，如果配置了衰減調度器，則由衰減調度器負責
        #       如果沒有配置衰減調度器，學習率將保持在 initial_lr


    def get_save(self):
        models = {
            "act":self.act,
            "cri":self.cri
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

    def explore_env(self, env, horizon_len: int,global_step) -> Tuple[Tensor, ...]:
        D_s_market = self.act.s_market_dim # test

        s_markets = torch.zeros((horizon_len, self.num_envs, D_s_market), dtype=torch.float32).to(self.device)
        next_s_markets = torch.zeros((horizon_len, self.num_envs, D_s_market), dtype=torch.float32).to(self.device)

        states = torch.zeros((horizon_len,
                              self.num_envs,
                              self.action_dim,
                              self.time_steps,
                              self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim + 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        next_states = torch.zeros((horizon_len,
                                   self.num_envs,
                                   self.action_dim,
                                   self.time_steps,
                                   self.state_dim), dtype=torch.float32).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        get_action = self.act
        for t in range(horizon_len):
            # 假設 state 的形狀是 [B, N, T, F_in] (B=num_envs)
            # 如果 actor 的 forward 輸入是 [B, N, T, F_in]，則不需要 unsqueeze(0)
            # 如果 actor 的 forward 輸入是 [N, T, F_in] (單樣本)，則需要 state.squeeze(0)
            # 根據您 HCAR_Actor forward 的輸入處理，這裡的 state 應為 [num_envs, num_stocks, window_len, num_original_features]

            current_s_market = self.act.s_market_stock_pool(self.act.temporal_feature_extractor(state, global_step=None)) # [B, D_s_market]

            # action, _ = get_action(state) # 修改：接收 actor 返回的 S_market
            # 由於explore_env的state通常是 (num_envs, N, T, F_in)
            # 而 HCAR_Actor 的 forward 輸入是 (B, N, T, F_in) 或 (B, 1, N, T, F_in)
            # 我們假設 HCAR_Actor.forward 能夠處理 [num_envs, N, T, F_in] 的輸入
            action_probs, _ = self.act(state, global_step=global_step) # S_market 在 Actor 內部使用，這裡不需要它返回給 explore_env
                                                                        # 但我們確實需要 current_s_market for the buffer

            states[t] = state
            s_markets[t] = current_s_market.detach() # 儲存當前 state 對應的 S_market

            # ary_action = action[0].detach().cpu().numpy() # 如果 action 是 [1, N+1]
            ary_action = action_probs.detach().cpu().numpy() # 如果 action_probs 是 [B, N+1]
            if self.num_envs == 1:
                ary_action = ary_action[0] # 取第一個 (也是唯一的) env 的動作

            next_state_ary, reward, done, _ = env.step(ary_action)

            # 更新 state (環境返回的 next_state_ary 是 numpy)
            # done 是一個布爾值 (for single env) 或一個布爾數組 (for multiple envs)
            if self.num_envs == 1:
                if done:
                    state = torch.as_tensor(env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    state = torch.as_tensor(next_state_ary, dtype=torch.float32, device=self.device).unsqueeze(0)
            else: # 多環境情況 (目前您的代碼主要針對單環境)
                # state = ... (需要處理多環境的 reset 和 next_state_ary)
                raise NotImplementedError("Multi-environment S_market handling in explore_env not fully detailed here.")

            # 為 next_state 計算 next_s_market
            # next_s_market_val = get_s_market_from_state(self.act, state) # state 此時已經是 next_state
            next_s_market_val = self.act.s_market_stock_pool(self.act.temporal_feature_extractor(state, global_step=None))

            actions[t] = action_probs # 如果 actions 張量是存概率分佈
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state
            next_s_markets[t] = next_s_market_val.detach() # 儲存 next_state 對應的 S_market

        self.last_state = state.detach() # 保存最後的 next_state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)

        transition = self.transition(
            state=states,
            action=actions,
            reward=rewards,
            undone=undones,
            next_state=next_states,
            s_market=s_markets,          # 新增
            next_s_market=next_s_markets # 新增
        )
        return transition

    def update_net(self, buffer: GeneralReplayBuffer, global_step: int):
        obj_critics = 0.0
        obj_actors = 0.0
        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            # --- 在優化器 step 之前調整 LR (用於 Warmup) ---
            if self.use_lr_scheduler:
                if self.act_optimizer:
                    self._adjust_learning_rate(self.act_optimizer, self.initial_lr_actor)
                    wandb.log({"Learning_Rate/Actor_LR": self.act_optimizer.param_groups[0]['lr'], 
                               "agent_step": self.optimizer_steps}, step=self.optimizer_steps) # 使用 optimizer_steps 作為 x 軸
                if self.cri_optimizer:
                    self._adjust_learning_rate(self.cri_optimizer, self.initial_lr_critic)
                    wandb.log({"Learning_Rate/Critic_LR": self.cri_optimizer.param_groups[0]['lr'],
                               "agent_step": self.optimizer_steps}, step=self.optimizer_steps)
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size,self.optimizer_steps)
            # --- 在優化器 step 之後，如果配置了衰減調度器，則調用 scheduler.step() ---
            # （注意：衰減調度器的 step 通常只在 warmup 之後執行）
            # if self.use_lr_scheduler and self.optimizer_steps >= self.warmup_steps:
            #     if self.act_lr_scheduler:
            #         self.act_lr_scheduler.step()
            #     if self.cri_lr_scheduler:
            #         self.cri_lr_scheduler.step()
            
            self.optimizer_steps += 1 # 每次優化器更新後遞增總步數
            
            
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer: GeneralReplayBuffer, batch_size: int, global_step: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        transition = buffer.sample(self.batch_size)
        state = transition.state
        action = transition.action
        reward = transition.reward
        undone = transition.undone
        next_state = transition.next_state
        s_market = transition.s_market           
        next_s_market = transition.next_s_market
        
        a, _ = self.act(state, global_step=self.optimizer_steps)
        wandb.log({
            "Action/Portfolio_Weights": wandb.Histogram(a.detach().cpu().numpy()),
            "agent_step": self.optimizer_steps}, step=self.optimizer_steps)
        q = self.cri(state, a, s_market)
        a_loss = -torch.mean(q)

        self.act_optimizer.zero_grad()
        a_loss.backward()
        # 在梯度裁剪前後都記錄，以觀察裁剪效果

        ############## --- Log Actor (HCAR_Actor) Gradients ---
        if self.optimizer_steps % 100 == 0: # Log every 100 optimizer steps
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

            wandb.log({**actor_grad_abs_means, **actor_grad_stds, "agent_step": self.optimizer_steps}, step=self.optimizer_steps)

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), self.clip_grad_norm)
            # Optionally log Actor_Grad_Norm_After_Clip here        
            total_norm_act_after_clip = 0
            for p in self.act.parameters():
                if p.grad is not None: # 裁剪後梯度仍然存在
                    param_norm = p.grad.data.norm(2)
                    total_norm_act_after_clip += param_norm.item() ** 2
            total_norm_act_after_clip = total_norm_act_after_clip ** 0.5
            wandb.log({"Gradients/Actor_Grad_Norm_After_Clip": total_norm_act_after_clip, "agent_step": self.optimizer_steps}, step=self.optimizer_steps)
        ### 
        self.act_optimizer.step()

        a_, _  = self.act(next_state)
        q_ = self.cri(next_state, a_.detach(), next_s_market.detach())
        q_target = reward + self.gamma * q_
        q_eval = self.cri(state, action.detach(), s_market.detach())

        td_error = self.criterion(q_target.detach(), q_eval)

        self.cri_optimizer.zero_grad()
        td_error.backward()
        ### log
        if self.cri_optimizer:
            total_norm_cri_before_clip = 0
            for p in self.cri.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_cri_before_clip += param_norm.item() ** 2
            total_norm_cri_before_clip = total_norm_cri_before_clip ** 0.5
            wandb.log({"Gradients/Critic_Grad_Norm_Before_Clip": total_norm_cri_before_clip, "agent_step": self.optimizer_steps}, step=self.optimizer_steps)

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.cri.parameters(), self.clip_grad_norm)
                total_norm_cri_after_clip = 0
                for p in self.cri.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_cri_after_clip += param_norm.item() ** 2
                total_norm_cri_after_clip = total_norm_cri_after_clip ** 0.5
                wandb.log({"Gradients/Critic_Grad_Norm_After_Clip": total_norm_cri_after_clip, "agent_step": self.optimizer_steps}, step=self.optimizer_steps)
        ###
        self.cri_optimizer.step()

        # --- 記錄損失 ---
        # 假設 a_loss 是 actor loss, td_error 是 critic loss
        wandb.log({
            "Loss/Actor_Loss": -a_loss.item(), 
            "Loss/Critic_Loss": td_error.item(),
            "Values/Mean_Q_Value_from_buffer_action": q_eval.mean().item(), # 使用buffer中的action評估的Q值
            "Values/Mean_Q_Value_from_current_policy_action": q.mean().item(), # 使用當前策略動作評估的Q值
            "agent_step": self.optimizer_steps}, step=self.optimizer_steps)

        return td_error, q_target