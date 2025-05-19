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



@AGENTS.register_module()
class PortfolioManagementEIIE(AgentBase):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIE, self).__init__()
        # print("hello ass")
        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_step = get_attr(kwargs, "max_step",
                                 12345)  # the max step number of an episode. 'set as 12345 in default.
        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)
        self.time_steps = get_attr(kwargs, "time_steps", 50)

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

        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)

        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)


        self.transition = get_attr(kwargs, "transition", namedtuple("Transition", ['state', 'prev_action', 'action', 'reward', 'undone','next_state', 'price_ratio']))

    def get_save(self):
        models = {
            "act":self.act
        }
        optimizers = {
            "act_optimizer":self.act_optimizer
        }
        res = {
            "models":models,
            "optimizers":optimizers
        }
        return res

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
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
        prev_actions = torch.zeros((horizon_len, self.num_envs, self.action_dim + 1), dtype=torch.float32).to(self.device)
        price_ratios = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        get_action = self.act
        prev_action = torch.zeros((self.num_envs, self.action_dim + 1), device=self.device)
        # print("state shape before actor call:", state.shape)
        for t in range(horizon_len):
            if state.dim() == 3:
                state = state.unsqueeze(0)
            # print(f"print t : {t}")
            input_state = state.permute(0, 3, 1, 2)
            # print(f"[DEBUG] step {t}")
            # print(f"prev_action shape: {prev_action.shape}, numel: {prev_action.numel()}")
            # print(f"prev_action content: {prev_action}")
            action = get_action(input_state,prev_action.unsqueeze(0))
            action = action + torch.randn_like(action) * 0.01#added
            action = torch.clamp(action, 0, 1)  # added
            states[t] = state
            ary_action = action[0].detach().cpu().numpy()
            ary_state, reward, done, _, price_ratio = env.step(ary_action)  # next_state
            state = torch.as_tensor(env.reset() if done else ary_state, dtype=torch.float32, device=self.device)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state
            prev_actions[t] = prev_action
            price_ratios[t] = torch.tensor(price_ratio, dtype=torch.float32, device=self.device)
            prev_action = action.detach()

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)

        transition = self.transition(
            state = states,
            prev_action=prev_actions,
            action = actions,
            reward = rewards,
            undone = undones,
            next_state = next_states,
            price_ratios=price_ratios
        )
        return transition


    def update_net(self, buffer: GeneralReplayBuffer):
        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        total_loss = 0.0
        for _ in range(update_times):
            transition = buffer.sample(self.batch_size)
            state = transition.state.to(self.device)
            prev_action = transition.prev_action.to(self.device)
            price_rate = transition.price_ratios.to(self.device)
            # print(f"x.shape : {state.shape}")
            state = state.permute(0, 3, 1, 2)
            action = self.act(state , prev_action)  # actor output: w_t
            stock_weights = action[:, :-1]##unsure

            # y_{t+1} 是下一期價格變動比例 → 要預先儲存在 buffer 或重算
            # 假設 reward = ln(mu * y · w_t) 已存在 transition.reward 中
            # 則 loss 就是：-reward
            portfolio_return = torch.sum(0.9 *price_rate * stock_weights, dim=1)
            log_return = torch.log(portfolio_return + 1e-10)
            loss = -log_return.mean()

            self.act_optimizer.zero_grad()
            loss.backward()
            self.act_optimizer.step()

            total_loss += loss.item()
        return total_loss / update_times, 0.0  # 只有 actor loss