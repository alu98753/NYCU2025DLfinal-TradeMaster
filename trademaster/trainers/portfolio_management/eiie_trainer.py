from pathlib import Path
import torch
import numpy as np
import os
import pandas as pd
import random
from collections import OrderedDict
import wandb

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_model, \
    save_best_model, load_model, \
    load_best_model, GeneralReplayBuffer,plot_metric_against_baseline

@TRAINERS.register_module()
class PortfolioManagementEIIETrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIETrainer, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)

        self.epochs = get_attr(kwargs, "epochs", 20)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)

        self.num_threads = int(get_attr(kwargs, "num_threads", 8))

        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.if_discrete = get_attr(kwargs, "if_discrete", False)
        self.if_off_policy = get_attr(kwargs, "if_off_policy", True)
        self.if_keep_save = get_attr(kwargs, "if_keep_save", True)
        self.if_over_write = get_attr(kwargs, "if_over_write", False)
        self.if_save_buffer = get_attr(kwargs, "if_save_buffer", False)

        if self.if_off_policy:  # off-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 64))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 1000))
        else:  # on-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 128))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 128))
        self.epochs = int(get_attr(kwargs, "epochs", 20))

        self.state_dim = self.agent.state_dim
        #print('state_dim', self.state_dim)
        self.action_dim = self.agent.action_dim
        #print('action_dim', self.action_dim)
        self.time_steps = self.agent.time_steps
        self.transition = self.agent.transition
        D_s_market = self.agent.act.s_market_dim if hasattr(self.agent.act, 's_market_dim') else 0 # 處理 getattr 返回 None 的情況
        self.transition_shapes = OrderedDict({
            'state': (self.buffer_size, self.num_envs,
                      self.action_dim, self.time_steps,
                      self.state_dim),
            'action': (self.buffer_size, self.num_envs, self.action_dim + 1),
            'reward': (self.buffer_size, self.num_envs),
            'undone': (self.buffer_size, self.num_envs),
            'next_state': (self.buffer_size, self.num_envs,
                      self.action_dim, self.time_steps,
                      self.state_dim),
            's_market': (self.buffer_size, self.num_envs, D_s_market),  
            'next_s_market': (self.buffer_size, self.num_envs, D_s_market),
            'regime':          (self.buffer_size, self.num_envs),
            'next_regime':     (self.buffer_size, self.num_envs),
        })

        self.verbose = get_attr(kwargs, "verbose", False)
        
        # --- 新增 MarketNet 校準相關配置 ---
        self.calibrate_marketnet_every_epoch = get_attr(kwargs, "calibrate_marketnet_every_epoch", True)
        self.marketnet_calibrate_target_acc = get_attr(kwargs, "marketnet_calibrate_target_acc", 0.88)
        self.marketnet_calibrate_max_steps = get_attr(kwargs, "marketnet_calibrate_max_steps", 50000) # 每次校準的最大優化步數
        self.marketnet_calibrate_batch_size = get_attr(kwargs, "marketnet_calibrate_batch_size", 64)
        self.marketnet_calibrate_lr = get_attr(kwargs, "marketnet_calibrate_lr", 1e-4) # 校準專用學習率
        # ------------------------------------

        self.init_before_training()
        
        ### global data flow : trainer -> agent -> HCAR forward
        self.global_step = 0
        

    
    def init_before_training(self):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            if self.verbose:
                print(f"| Arguments Remove work_dir: {self.work_dir}")
        else:
            if self.verbose:
                print(f"| Arguments Keep work_dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path, exist_ok=True)


    def _calibrate_marketnet(self, buffer: GeneralReplayBuffer, current_epoch: int):
        if not self.agent.market:
            if self.verbose: print("MarketNet not available, skipping calibration.")
            return
        if not self.agent.market_criterion:
            if self.verbose: print("MarketNet criterion not set on agent, skipping calibration.")
            return

        # print(f"\n[Trainer] Calibrating MarketNet for epoch {current_epoch} at agent_step {self.agent.optimizer_steps}...")
        self.agent.market.train() 

        # 解凍 MarketNet 參數
        for param in self.agent.market.parameters():
            param.requires_grad = True # <--- 確保所有參數都設為 True

        # 準備 MarketNet 的優化器
        # 重新獲取可訓練參數列表，因為它们的 requires_grad 狀態可能剛改變
        current_trainable_market_params = list(filter(lambda p: p.requires_grad, self.agent.market.parameters()))

        if not current_trainable_market_params:
            print("No trainable parameters in MarketNet after setting requires_grad=True. Skipping calibration.")
            self.agent.market.eval()
            # 考慮是否需要重新凍結參數，如果這裡就返回了
            # for param in self.agent.market.parameters():
            #     param.requires_grad = False
            return

        # 無論如何都重新創建或確保優化器針對的是當前的可訓練參數和正確的學習率
        # 這是因為 Adam 等優化器可能會在其內部狀態中快照參數的 requires_grad 狀態
        print(f"Setting up MarketNet optimizer for calibration with LR: {self.marketnet_calibrate_lr}")
        # if self.agent.market_optimizer is not None:
        #     print(f"  Previous optimizer state: {self.agent.market_optimizer.state_dict()}") # 只是觀察，通常Adam狀態不需手動清
        
        # 總是為校準階段創建一個新的優化器實例，或確保現有實例更新其參數組
        # 最簡單且最安全的方式是重新創建，以避免舊狀態問題
        # self.agent.market_optimizer = torch.optim.Adam(
        #     current_trainable_market_params, # 使用剛剛獲取的可訓練參數列表
        #     lr=self.marketnet_calibrate_lr
        # )


        total_calibration_loss = 0
        total_calibration_correct = 0
        total_calibration_samples = 0

        for cal_step in range(self.marketnet_calibrate_max_steps):
            transition_sample = buffer.sample(self.marketnet_calibrate_batch_size)
            s_market_batch = transition_sample.s_market
            regime_batch = transition_sample.regime.view(-1).long()

            # BatchNorm1d 要求 batch_size > 1 在訓練模式下
            if s_market_batch.size(0) <= 1:
                print(f"Skipping MarketNet calibration step {cal_step} due to batch size {s_market_batch.size(0)} <= 1")
                continue
            
            s_market_batch = s_market_batch.view(s_market_batch.size(0), -1) # Flatten s_market

            # for name, param in self.agent.market.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter: {name}, requires_grad: {param.requires_grad}, grad: {param.grad is not None}")
        #
            _, logits = self.agent.market(s_market_batch)
            loss = self.agent.market_criterion(logits, regime_batch)
            # print(f"  Debug: s_market_batch.requires_grad = {s_market_batch.requires_grad}")
            # print(f"  Debug: regime_batch.requires_grad = {regime_batch.requires_grad}")
            # print(f"  Debug: logits.requires_grad = {logits.requires_grad}, logits.grad_fn = {logits.grad_fn}")
            # print(f"  Debug: loss.requires_grad = {loss.requires_grad}, loss.grad_fn = {loss.grad_fn}")

            self.agent.market_optimizer.zero_grad()
            loss.backward()
            self.agent.market_optimizer.step()
            # for name, param in self.agent.market.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(f"Parameter: {name}, grad norm: {param.grad.norm().item()}")

            total_calibration_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                total_calibration_correct += (preds == regime_batch).float().sum().item()
                total_calibration_samples += preds.size(0)

            if (cal_step + 1) % 50 == 0 and total_calibration_samples > 0: # 每50步打印一次進度
                current_acc = total_calibration_correct / total_calibration_samples
                if self.verbose:
                    print(f"  MarketNet Calibration Step: {cal_step+1}/{self.marketnet_calibrate_max_steps}, Avg Loss: {total_calibration_loss/(cal_step+1):.4f}, Current Avg Acc: {current_acc:.4f}")
                if current_acc >= self.marketnet_calibrate_target_acc:
                    if self.verbose: print(f"  MarketNet reached target accuracy of {self.marketnet_calibrate_target_acc:.4f}. Stopping calibration.")
                    break
        
        avg_epoch_calibration_loss = total_calibration_loss / (cal_step + 1) if cal_step >=0 else float('nan')
        avg_epoch_calibration_acc = total_calibration_correct / total_calibration_samples if total_calibration_samples > 0 else float('nan')

        wandb.log({
            "Market/Epoch_Calibration_AvgLoss": avg_epoch_calibration_loss,
            "Market/Epoch_Calibration_AvgAcc": avg_epoch_calibration_acc,
            "Market/Epoch_Calibration_Steps_Taken": cal_step + 1,
            "agent_step": self.agent.optimizer_steps # 使用 agent 主訓練的步數作為x軸
        })
        print(f"[Trainer] MarketNet calibration finished for epoch {current_epoch}. Steps: {cal_step+1}, AvgLoss: {avg_epoch_calibration_loss:.4f}, AvgAcc: {avg_epoch_calibration_acc:.4f}\n")

        self.agent.market.eval() # 校準完畢，設回評估模式
        for param in self.agent.market.parameters():
            param.requires_grad = False



    def train_and_valid(self):
        
        '''init agent.last_state'''
        state = self.train_environment.reset()

        #print(f"State shape: {state.shape}")
        #print(f"Expected shape: ({self.action_dim}, {self.time_steps}, {self.state_dim})")

        if state.shape[0] < self.action_dim:
            # 建立補零的 Tensor，填滿的部分用 0
            padding = torch.zeros((self.action_dim - state.shape[0], self.time_steps, self.state_dim), dtype=torch.float32)
    
            # 先將 state 轉成 Tensor 來做 concat，之後再轉回 numpy
            state = torch.cat((torch.tensor(state, dtype=torch.float32), padding), dim=0)
    
            # 最後轉回 numpy 陣列
            state = state.numpy()
            #print(f"State shape after padding: {state.shape}")
            #print(f"State type after padding: {type(state)}")

    
        if self.num_envs == 1:
            assert state.shape == (self.action_dim, self.time_steps, self.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            assert state.shape == (self.num_envs, self.state_dim)
            assert isinstance(state, torch.Tensor)
            state = state.to(self.device)
        assert state.shape == (self.num_envs, self.action_dim, self.time_steps, self.state_dim,)
        assert isinstance(state, torch.Tensor)
        self.agent.last_state = state.detach()

        '''init buffer'''
        if self.if_off_policy:
            buffer = GeneralReplayBuffer(
                transition=self.transition,
                shapes=self.transition_shapes,
                num_seqs=self.num_envs,
                max_size=self.buffer_size,
                device=self.device,
            )
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len,self.global_step )
            buffer.update(buffer_items)
        else:
            buffer = []

        valid_score_list = []
        save_dict_list = []
        current_epoch = 1
        print("Train Episode: [{}/{}]".format(current_epoch, self.epochs))
        while True:
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len,self.global_step)
            self.global_step += self.horizon_len # 更新 trainer 的 global_step

            # print("--- DataLoader Output / Agent Input ---\n\n")
            
            # print("Shape of batch['obs'] from DataLoader:",buffer_items.state.shape)
            # print("Shape of batch['action'] from DataLoader:",buffer_items.action.shape)
            # print("Shape of batch['reward'] from DataLoader:",buffer_items.reward.shape)
            # print("Shape of batch['undone'] from DataLoader:",buffer_items.undone.shape)
            # print("Shape of batch['next_state'] from DataLoader:",buffer_items.next_state.shape)
            
            if self.if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = self.agent.update_net(buffer,self.global_step)
            torch.set_grad_enabled(False)
            
            # 檢查是否一個 episode 結束 (undone < 1.0 表示至少有一個 env done)
            # 對於 portfolio management, episode 通常是整個回測期
            # 因此，這裡的 "epoch" 概念可能對應於完成一次完整的數據遍歷或固定數量的 agent 更新步驟
            # 我們假設當 buffer_items.undone 中有 True (即 0) 時，代表訓練環境的一個 episode 結束
            # 這個條件可能需要根據你的環境設計來調整
            if torch.mean(buffer_items.undone) < 1.0:
                print("Valid Episode: [{}/{}]".format(current_epoch, self.epochs))
                # --- 校準 MarketNet ---
                if self.calibrate_marketnet_every_epoch and self.agent.market:
                    print(f"--- Preparing to calibrate MarketNet for epoch {current_epoch} ---")
                    torch.set_grad_enabled(True) # 為 MarketNet 校準開啟梯度
                    self._calibrate_marketnet(buffer, current_epoch)
                    torch.set_grad_enabled(False) # 校準完畢後關閉梯度
                    print(f"--- Finished calibrating MarketNet for epoch {current_epoch} ---")
                if self.agent.market: # 驗證時 MarketNet 應為 eval
                    self.agent.market.eval()
                # --------------------
                
                state = self.valid_environment.reset()
                episode_reward_sum = 0.0  # sum of rewards in an episode
                while True:
                    # print("--- Agent.explore_env ---\n\n")
                    # print("Shape of state before unsqueeze:",state.shape)
                    
                    tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    # print("Shape of x  after unsqueeze (input to EIIEConv):",tensor_state.shape)
                    
                    # ———— 1) 先算出 S_market —— #
                    # 注意：不用带梯度，因为验证时不更新网络
                    with torch.no_grad():
                        temp_feat = self.agent.act.temporal_feature_extractor(tensor_state, global_step=None)  # [1, N, hidden_dim]
                        current_s_market = self.agent.act.s_market_stock_pool(temp_feat)                      # [1, D_s_market]

                        # ———— 2) 用 MarketNet 得到 regime_probs —— #
                        regime_probs_val, logits_val = self.agent.market(current_s_market)              # [1, 3]

                        # ———— 3) 把 (state, regime_probs_val) 传给 Actor —— #
                        val_action_probs, _ = self.agent.act(tensor_state, regime_probs_val)  # [1, N+1], _
                        

                    action = val_action_probs.detach().cpu().numpy()[0]
                    state, reward, done, save_dict = self.valid_environment.step(action)
                    episode_reward_sum += reward
                    if done:
                        print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                        # 從 save_dict 或 valid_environment 中獲取更詳細的指標
                        # save_dict 包含這些指標
                        '''
                        save_dict = OrderedDict(
                            {
                                "Profit Margin": tr * 100,
                                "Excess Profit": tr * 100 - 0,
                                "daily_return": daily_return_values,
                                "total_assets": assets_values
                            }
                        )
                        '''
                        # 為了獲取完整的8個指標，最好是調用環境的 analysis_result
                        current_metrics = self.valid_environment.analysis_result() # (tr, sharpe, vol, mdd, cr, sor)
                        # print("save dict keys:",list(save_dict.keys()))
                        wandb.log({
                            "Valid Reward Sum": episode_reward_sum,
                            "Validation/Profit_Margin": save_dict["Profit Margin"],
                            "Validation/Excess_Profit": save_dict["Excess Profit"],
                            "Validation/Daily_Return": save_dict["daily_return"],
                            "Validation/Total_Assets": save_dict["total_assets"],
                            "Validation/Total_Return": round(current_metrics[0]*100, 2),
                            "Validation/Sharpe_Ratio": round(current_metrics[1], 4),
                            "Validation/Volatility": round(current_metrics[2]*100, 2),
                            "Validation/Max_Drawdown": round(current_metrics[3]*100, 2),
                            "Validation/Calmar_Ratio": round(current_metrics[4], 4),
                            "Validation/Sortino_Ratio": round(current_metrics[5], 4),
                            "agent_step": self.agent.optimizer_steps
                            # ENT, ENB 需要額外計算
                        })
                        ### log
                        
                        break
                valid_score_list.append(episode_reward_sum)
                save_dict_list.append(save_dict)

                save_model(self.checkpoints_path,
                           epoch=current_epoch,
                           save=self.agent.get_save())
                current_epoch += 1
                if current_epoch <= self.epochs:
                    print("Train Episode: [{}/{}]".format(current_epoch, self.epochs))

            if current_epoch > self.epochs:
                break

        max_index = np.argmax(valid_score_list)
        plot_metric_against_baseline(total_asset=save_dict_list[max_index]['total_assets'],
                                     buy_and_hold=None, alg='Ensemble of Identical Independent Evaluators',
                                     task='valid', color='darkcyan', save_dir=self.work_dir)
        load_model(self.checkpoints_path,
                   epoch=max_index + 1,
                   save=self.agent.get_save())
        # 存max_index 也就是vaild_score 最高的model
        save_best_model(
            output_dir=self.checkpoints_path,
            epoch=max_index + 1,
            save=self.agent.get_save()
        )

    def test(self):
        load_best_model(self.checkpoints_path, save=self.agent.get_save(), is_train=False)

        print("Test Best Episode")
        state = self.test_environment.reset()
        if self.agent.market:
            self.agent.market.eval()
        self.agent.act.eval()
        episode_reward_sum = 0
        while True:
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # 1) 先算 S_market
            temp_feat = self.agent.act.temporal_feature_extractor(tensor_state, global_step=None)  # [1, N, hidden_dim]
            current_s_market = self.agent.act.s_market_stock_pool(temp_feat)                      # [1, D_s_market]

            # 2) 用 MarketNet 得 regime_probs
            regime_probs_test, logits_val = self.agent.market(current_s_market)              # [1, 3]

            # 3) 把 (tensor_state, regime_probs_test) 传给 Actor
            tensor_action, _ = self.agent.act(tensor_state, regime_probs_test)  # [1, N+1], _
            if self.if_discrete:
                tensor_action = tensor_action.argmax(dim=1)
            action = tensor_action.detach().cpu().numpy()[0]
            state, reward, done, return_dict = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                current_metrics = self.valid_environment.analysis_result() # (tr, sharpe, vol, mdd, cr, sor)
                    # print("save dict keys:",list(save_dict.keys()))
                wandb.log({
                    "Valid Reward Sum": episode_reward_sum,
                    "Validation/Profit_Margin": return_dict["Profit Margin"],
                    "Validation/Excess_Profit": return_dict["Excess Profit"],
                    "Validation/Daily_Return": return_dict["daily_return"],
                    "Validation/Total_Assets": return_dict["total_assets"],
                    "Validation/Total_Return": round(current_metrics[0]*100, 2),
                    "Validation/Sharpe_Ratio": round(current_metrics[1], 4),
                    "Validation/Volatility": round(current_metrics[2]*100, 2),
                    "Validation/Max_Drawdown": round(current_metrics[3]*100, 2),
                    "Validation/Calmar_Ratio": round(current_metrics[4], 4),
                    "Validation/Sortino_Ratio": round(current_metrics[5], 4),
                    "agent_step": self.agent.optimizer_steps
                })
                plot_metric_against_baseline(total_asset=return_dict['total_assets'],
                                             buy_and_hold=None, alg='Ensemble of Identical Independent Evaluators',
                                             task='test', color='darkcyan', save_dir=self.work_dir)
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break
        df_return = self.test_environment.save_portfolio_return_memory()
        df_assets = self.test_environment.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir + "test_result.csv"))
        daily_return = df.daily_return.values
        return daily_return

    def test_with_customize_policy(self, policy, customize_policy_id,extra_parameters=None):
        state = self.test_environment.reset()
        self.test_environment.test_id = customize_policy_id
        print(f"Test customize policy: {str(customize_policy_id)}")
        if self.agent.market:
            self.agent.market.eval()
        self.agent.act.eval()
        episode_reward_sum = 0
        weights_brandnew=None
        while True:
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # print('extra_parameters is ',extra_parameters)
            if customize_policy_id=="Average_holding":
                action = policy(tensor_state, self.test_environment,weights_brandnew)
            else:
                action = policy(tensor_state, self.test_environment)
            state, reward, done, return_dict = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                plot_metric_against_baseline(total_asset=return_dict['total_assets'],
                                             buy_and_hold=None, alg='Ensemble of Identical Independent Evaluators',
                                             task='test', color='darkcyan', save_dir=self.work_dir)
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break
            weights_brandnew = return_dict["weights_brandnew"]

        df_return = self.test_environment.save_portfolio_return_memory()
        df_assets = self.test_environment.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result_customize_actions_id_"+str(customize_policy_id)+".csv"), index=False)
        return daily_return