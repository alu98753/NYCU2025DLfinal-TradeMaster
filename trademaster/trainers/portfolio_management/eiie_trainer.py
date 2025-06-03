from pathlib import Path

import torch
import time

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_model, \
    save_best_model, load_model, \
    load_best_model, GeneralReplayBuffer,plot_metric_against_baseline
import numpy as np
import os
import pandas as pd
import random
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
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
        # <<< 新增 s_market 維度獲取 >>>
        # self.d_model_for_s_market 應該與 EIIEConv 的 d_model 一致
        if hasattr(self.agent, 'd_model_for_s_market'):
            self.d_model_for_s_market = self.agent.d_model_for_s_market
        elif hasattr(self.agent.act, 'd_model'):
             self.d_model_for_s_market = self.agent.act.d_model
        else:
            # 如果都找不到，需要一個默認值或從配置讀取
            self.d_model_for_s_market = 128 # 假設的默認值
            print(f"Warning: Trainer could not determine d_model_for_s_market, using default {self.d_model_for_s_market}")
        # <<< 新增結束 >>>

        # 更新 Transition namedtuple 的導入或定義 (如果 Agent 中已定義，則 Trainer 直接使用 Agent 的)
        self.transition = self.agent.transition 
        self.transition_shapes = OrderedDict({
            'state': (self.buffer_size, self.num_envs,
                      self.action_dim, self.time_steps,
                      self.state_dim), # state 維度
            'action': (self.buffer_size, self.num_envs, self.action_dim + 1), # 動作維度
            'reward': (self.buffer_size, self.num_envs),
            'undone': (self.buffer_size, self.num_envs),
            'next_state': (self.buffer_size, self.num_envs,
                           self.action_dim, self.time_steps,
                           self.state_dim), # next_state 維度
            's_market': (self.buffer_size, self.num_envs, self.d_model_for_s_market),
            'next_s_market': (self.buffer_size, self.num_envs, self.d_model_for_s_market),
        })

        self.verbose = get_attr(kwargs, "verbose", False)
        
        # <<< 新增 MarketNet 校準相關參數 >>>
        self.calibrate_marketnet_flag = get_attr(kwargs, "calibrate_marketnet_flag", False) # 是否啟用校準
        self.calibrate_marketnet_every_n_epochs = get_attr(kwargs, "calibrate_marketnet_every_n_epochs", 1)
        self.marketnet_calibrate_epochs = get_attr(kwargs, "marketnet_calibrate_epochs", 5) # 每次校準時訓練 MarketNet 的輪數
        self.marketnet_calibrate_lr = get_attr(kwargs, "marketnet_calibrate_lr", 1e-4)
        self.marketnet_calibrate_batch_size = get_attr(kwargs, "marketnet_calibrate_batch_size", 256) # MarketNet 訓練時的 batch_size
        self.marketnet_target_accuracy_bull = get_attr(kwargs, "marketnet_target_accuracy_bull", 0.70) # 熊市分類目標準確率
        self.marketnet_target_accuracy_bear = get_attr(kwargs, "marketnet_target_accuracy_bear", 0.70) # 牛市分類目標準確率
        self.marketnet_weight_decay = get_attr(kwargs, "marketnet_weight_decay", 0.0) # 默認為 0.0

        # 訓練 MarketNet 所需的帶標籤數據路徑 (通常是訓練環境的數據路徑)
        # # 假設 train_environment 已經被正確初始化並包含 df_path
        # if self.train_environment and hasattr(self.train_environment, 'df_path'):
        #     self.path_to_market_regime_data = self.train_environment.df_path
        # else:
        #     self.path_to_market_regime_data = get_attr(kwargs, "path_to_market_regime_data", None)
        #     print("Warning: path_to_market_regime_data not found via train_environment, using direct config or None.")
        # # <<< 新增結束 >>>
        
        # <<< 新增：在 Trainer 初始化時創建 MarketNet 的優化器 >>>
        if hasattr(self.agent, 'market_net') and self.agent.market_net is not None:
            # 確保 MarketNet 的參數已經移到正確的 device
            self.agent.market_net.to(self.device) 
            self.optimizer_marketnet = torch.optim.Adam(
                self.agent.market_net.parameters(), 
                lr=self.marketnet_calibrate_lr,
                weight_decay=self.marketnet_weight_decay # 在這裡應用 L2 正則化
            )
            print("MarketNet optimizer created in Trainer init.")
        else:
            self.optimizer_marketnet = None
            if self.calibrate_marketnet_flag:
                print("Warning: MarketNet calibration is enabled, but MarketNet or its optimizer could not be initialized in Trainer.")
        # <<< 新增結束 >>>
        self.init_before_training()

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

    def _get_marketnet_calibration_dataloader(self, env_for_calibration):
        """
        為 MarketNet 校準準備數據加載器。
        它會加載整個訓練數據集，為每個可能的時間窗口提取 s_market 特徵，
        並與對應的 regime 標籤配對。
        Args:
            env_for_calibration: 用於校準的環境對象 (可以是 train_environment 或 test_environment)。
        """
        if not (hasattr(env_for_calibration, 'data_cube') and \
                hasattr(env_for_calibration, 'regime_by_date') and \
                hasattr(env_for_calibration, 'unique_dates') and \
                env_for_calibration.data_cube is not None and \
                env_for_calibration.regime_by_date is not None):
            print(f"Error: {env_for_calibration.task}_environment is not fully initialized with data_cube or regime_by_date for MarketNet calibration.")
            return None
        
        actor_model = self.agent.act
        actor_model.eval() # 確保 Actor 處於評估模式以進行特徵提取

        lookback_window = self.time_steps # EIIEConv 的時間窗口
        
        data_cube_full = env_for_calibration.data_cube         # [TotalDays, N, F_original]
        regime_by_date_full = env_for_calibration.regime_by_date # [TotalDays], integer labels
        num_total_days = data_cube_full.shape[0]
        num_stocks = env_for_calibration.stock_dim # N

        all_s_market_features = []
        all_regime_labels = []

        print(f"Preparing MarketNet calibration data from {env_for_calibration.task} environment. Total days: {num_total_days}, Lookback window: {lookback_window}")

        # 遍歷所有可能的結束日期來形成窗口
        # 窗口的結束點是 day_idx，窗口從 day_idx - lookback_window + 1 開始
        for day_idx in range(lookback_window - 1, num_total_days):
            start_slice_idx = day_idx - lookback_window + 1
            end_slice_idx = day_idx + 1
            
            original_features_window = data_cube_full[start_slice_idx:end_slice_idx, :, :] # [T, N, F_original]
            
            # 構造 MarketNet 校準用的 Actor 輸入 (prev_weights 用0填充)
            # state_dim 是 F_original + 2
            actor_input_for_calib_np = np.zeros((num_stocks, lookback_window, self.state_dim), dtype=np.float32)
            # 填充原始特徵
            # 注意：這裡的 self.train_environment.original_feature_dim 應該改成 env_for_calibration.original_feature_dim
            # 因為原始特徵維度是環境的屬性
            actor_input_for_calib_np[:, :, :env_for_calibration.original_feature_dim] = np.transpose(original_features_window, (1,0,2))
            # prev_weights 部分保持為0 (或者用一個固定的平均值，例如1/N)

            actor_input_x_tensor = torch.tensor(actor_input_for_calib_np, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, N, T, F_augmented_with_dummy_prev_w]
            
            current_regime_label = regime_by_date_full[day_idx]

            with torch.no_grad():
                s_market_tensor = actor_model.extract_market_features(actor_input_x_tensor) # [1, d_model]
            
            all_s_market_features.append(s_market_tensor.squeeze(0).cpu())
            all_regime_labels.append(torch.tensor(current_regime_label, dtype=torch.long))


        if not all_s_market_features:
            print(f"No (s_market, regime) pairs were generated for MarketNet calibration from {env_for_calibration.task} environment. Skipping.")
            return None

        s_market_dataset_tensor = torch.stack(all_s_market_features)
        regime_labels_dataset_tensor = torch.stack(all_regime_labels)

        print(f"Collected {s_market_dataset_tensor.shape[0]} samples for MarketNet calibration.")
        regime_counts = torch.bincount(regime_labels_dataset_tensor, minlength=2) #確保至少有牛熊兩類計數
        print(f"Regime distribution: Bull (0) count = {regime_counts[0].item()}, Bear (1) count = {regime_counts[1].item()}")

        if regime_counts[0].item() == 0 or regime_counts[1].item() == 0 :
            print(f"Warning: One or both classes have zero samples in MarketNet calibration data. Training may be ineffective or fail.")
            sampler = None # 如果只有一類或沒有樣本，sampler可能出錯
            shuffle_flag = True
        else:
            class_weights = 1. / regime_counts.float() 
            class_weights[regime_counts == 0] = 0 # 避免除以0
            sample_weights_val = class_weights[regime_labels_dataset_tensor]
            sampler = WeightedRandomSampler(weights=sample_weights_val, num_samples=len(sample_weights_val), replacement=True)
            shuffle_flag = False

        calibration_dataset = TensorDataset(s_market_dataset_tensor, regime_labels_dataset_tensor)
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=self.marketnet_calibrate_batch_size,
            sampler=sampler,
            shuffle=shuffle_flag,
            drop_last=True 
        )
        return calibration_dataloader
    
    # 修改 calibrate_market_network 函數使其可以接受一個環境對象
    def calibrate_market_network(self, env_for_calibration):
        """
        使用從 Actor 提取的 s_market 特徵和真實的 regime 標籤來訓練/校準 MarketNet。
        Args:
            env_for_calibration: 用於校準的環境對象 (可以是 train_environment 或 test_environment)。
        """
        if not self.calibrate_marketnet_flag:
            print("MarketNet calibration is disabled by flag.")
            return
            
        if not hasattr(self.agent, 'market_net') or self.agent.market_net is None:
            print("MarketNet not found in agent. Skipping calibration.")
            return
        if not hasattr(self.agent.act, 'extract_market_features'):
            print("Actor does not have 'extract_market_features' method. Skipping MarketNet calibration.")
            return

        print(f"Starting MarketNet calibration phase for {env_for_calibration.task} environment...")
        
        original_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        market_net_model = self.agent.market_net.to(self.device) # 確保在正確的 device
        actor_model = self.agent.act.to(self.device) # 用於特徵提取
        original_actor_is_training = actor_model.training
        actor_model.eval()
        
        # 2. 設置 MarketNet 為訓練模式，準備優化器和損失函數
        for param in market_net_model.parameters():
            param.requires_grad = True 
        market_net_model.train()    
        optimizer_marketnet = self.optimizer_marketnet
        if optimizer_marketnet is None:
             print("MarketNet optimizer is None. Cannot calibrate MarketNet.")
             # Restore state before returning
             if original_actor_is_training: actor_model.train() 
             market_net_model.eval() 
             torch.set_grad_enabled(original_grad_state)
             return
        # scheduler_marketnet = LinearLR(optimizer_marketnet, start_factor=1.0, end_factor=0.0, total_iters=self.marketnet_calibrate_epochs)

        for g in optimizer_marketnet.param_groups:
            g['lr'] = self.marketnet_calibrate_lr
        print("MarketNet parameters AFTER setting requires_grad=True and train():")
        for name, param in market_net_model.named_parameters():
            print(f"   {name}: requires_grad={param.requires_grad}")
        
        # 1. 獲取數據加載器
        calibration_dataloader = self._get_marketnet_calibration_dataloader(env_for_calibration)
        if calibration_dataloader is None:
            print(f"Failed to get MarketNet calibration dataloader for {env_for_calibration.task} environment. Skipping calibration.")
            if original_actor_is_training: actor_model.train() 
            market_net_model.eval() 
            torch.set_grad_enabled(original_grad_state)
            return
        criterion_marketnet = nn.CrossEntropyLoss().to(self.device) # 如果 sampler 沒完全均衡，可以考慮加 class_weights

        print(f"Calibrating MarketNet for {self.marketnet_calibrate_epochs} epochs with LR {self.marketnet_calibrate_lr} and Batch Size {self.marketnet_calibrate_batch_size}")

        for epoch in range(self.marketnet_calibrate_epochs):
            epoch_loss = 0.0
            total_samples = 0
            correct_predictions = 0
            
            # 用於分類別統計準確率
            bull_correct = 0
            bull_total = 0
            bear_correct = 0
            bear_total = 0

            for s_market_batch, regime_labels_batch in calibration_dataloader:
                s_market_batch = s_market_batch.to(self.device)
                regime_labels_batch = regime_labels_batch.to(self.device)

                optimizer_marketnet.zero_grad()
                logits = market_net_model(s_market_batch) # [B, 2]
                loss = criterion_marketnet(logits, regime_labels_batch)
                
                # if not loss.requires_grad:
                #     print(f"Warning: Loss in MarketNet calibration epoch {epoch+1} does not require grad. Logits grad_fn: {logits.grad_fn}")
                #     for name, param in market_net_model.named_parameters():
                #         if not param.requires_grad:
                #             print(f"   Problematic param: {name} requires_grad={param.requires_grad}")
                #     print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")
                #     for name, param in market_net_model.named_parameters():
                #         if param.grad is None and param.requires_grad:
                #             print(f"   Param {name} requires grad but grad is None BEFORE backward.")

                loss.backward()

                # for name, param in market_net_model.named_parameters():
                #     if param.grad is not None:
                #         # print(f"   Param {name} grad norm after backward: {param.grad.norm().item()}")
                #         pass # 只在需要時打印，避免過多輸出
                #     elif param.requires_grad:
                #         print(f"   Param {name} requires grad but grad is STILL None AFTER backward.")
                
                optimizer_marketnet.step()
                # print(f"MarketNet param norm after step: {market_net_model.network[0].weight.norm().item()}")
                epoch_loss += loss.item() * s_market_batch.size(0)
                total_samples += s_market_batch.size(0)
                
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == regime_labels_batch).sum().item()

                # 分類別統計
                bull_mask = (regime_labels_batch == 0)
                bear_mask = (regime_labels_batch == 1)
                bull_correct += ((preds == regime_labels_batch) & bull_mask).sum().item()
                bull_total += bull_mask.sum().item()
                bear_correct += ((preds == regime_labels_batch) & bear_mask).sum().item()
                bear_total += bear_mask.sum().item()

            avg_epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
            overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            accuracy_bull = bull_correct / bull_total if bull_total > 0 else 0
            accuracy_bear = bear_correct / bear_total if bear_total > 0 else 0
            
            print(f"MarketNet Calib Epoch [{epoch+1}/{self.marketnet_calibrate_epochs}]: Loss={avg_epoch_loss:.4f}, "
                  f"OverallAcc={overall_accuracy:.4f}, AccBull={accuracy_bull:.4f}, AccBear={accuracy_bear:.4f}")

            # scheduler_marketnet.step()
            
            # 檢查早停條件 (基於配置文件中的目標準確率)
            if accuracy_bull >= self.marketnet_target_accuracy_bull and \
               accuracy_bear >= self.marketnet_target_accuracy_bear:
                print(f"MarketNet reached target accuracy for both classes. Early stopping calibration at epoch {epoch+1}.")
                break
        
        # 3. 校準結束後，將 MarketNet 設置回評估模式
        market_net_model.eval()
        if original_actor_is_training: 
            actor_model.train()
        del calibration_dataloader 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # <<< Restore original grad state >>>
        torch.set_grad_enabled(original_grad_state)
        print(f"MarketNet calibration for {env_for_calibration.task} environment finished.")

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
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len )
            buffer.update(buffer_items)
        else:
            buffer = []

        valid_score_list = []
        save_dict_list = []
        epoch_counter_for_rl  = 1
        print("Train Episode: [{}/{}]".format(epoch_counter_for_rl , self.epochs))
        
        print(f"Starting Training. Total RL Epochs: {self.epochs}")
        # 原有的 epoch 變量用於控制驗證和模型保存的頻率，現在改為 rl_update_batch_count
        rl_update_batch_count = 1 
        while True:
            
            if self.calibrate_marketnet_flag and \
               (rl_update_batch_count -1) % self.calibrate_marketnet_every_n_epochs == 0:
                print(f"\n===== Calibrating MarketNet before RL update batch {rl_update_batch_count} =====")
                # 在訓練階段，使用 train_environment 進行校準
                self.calibrate_market_network(self.train_environment) # <--- 修改這裡
                print(f"===== MarketNet Calibration Finished =====")
            # <<< MarketNet 校準結束 >>>

            if epoch_counter_for_rl > self.epochs : # 控制總的 RL 訓練輪次
                print(f"Reached target RL epochs ({self.epochs}). Stopping training.")
                break
            print(f"\n--- RL Update Batch: {rl_update_batch_count} (Overall RL Epoch: {epoch_counter_for_rl}) ---")

            # 確保 Actor 在 RL 探索前處於正確的模式 (通常是 train，除非 explore_env 內部處理)
            if hasattr(self.agent, 'act') and self.agent.act is not None:
                self.agent.act.train()

            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len)
            if self.if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            # 確保 Actor 和 Critic 梯度是開啟的，準備更新網絡
            torch.set_grad_enabled(True)
            self.agent.act.train()
            self.agent.cri.train()


            logging_tuple = self.agent.update_net(buffer)
            print(f"RL Update: Actor Loss: {logging_tuple[0]:.4f}, Critic Loss: {logging_tuple[1]:.4f}") # 假設順序

            torch.set_grad_enabled(False)
            time.sleep(3)
            if torch.mean(buffer_items.undone) < 1.0: # 標誌著至少一個訓練環境的 episode 結束
                print(f"\n--- Validation after RL Update Batch: {rl_update_batch_count} (Overall RL Epoch: {epoch_counter_for_rl}) ---")
                
                # 驗證時，Actor 和 MarketNet 都應處於評估模式
                if hasattr(self.agent, 'act') and self.agent.act is not None: self.agent.act.eval()
                if hasattr(self.agent, 'market_net') and self.agent.market_net is not None: self.agent.market_net.eval()

                state_valid_np = self.valid_environment.reset() # Numpy array from env
                
                # 將 numpy state 轉換為 tensor
                state_valid = torch.tensor(state_valid_np, dtype=torch.float32, device=self.device)
                if state_valid.dim() == 3: # [N, T, F] for single env
                    state_valid = state_valid.unsqueeze(0) # [1, N, T, F]
                
                episode_reward_sum = 0.0
                done_valid = False
                temp_save_dict_valid = {} # 用於存儲最後一步的 info

                # 驗證循環 (通常跑完一個完整的 episode)
                max_validation_steps = getattr(self.valid_environment, 'num_unique_dates', 2000) # 獲取驗證環境的最大可能步數
                current_validation_step = 0
                while not done_valid and current_validation_step < max_validation_steps:
                    # --- 在驗證時，Actor 也需要 market_regime_signal ---
                    market_regime_signal_valid_tensor = None # 初始化
                    if hasattr(self.agent, 'market_net') and self.agent.market_net and \
                       hasattr(self.agent.act, 'extract_market_features'):
                        with torch.no_grad():
                            s_market_valid = self.agent.act.extract_market_features(state_valid)
                            logits_market_valid = self.agent.market_net(s_market_valid)
                            market_regime_signal_valid_tensor = torch.argmax(logits_market_valid, dim=1) # Shape: [B]
                    else:
                        print("Warning: MarketNet or extract_market_features not available during validation. Actor may not behave as expected.")
                    
                    with torch.no_grad():
                        # Actor 的 forward 簽名需要是 forward(self, state, market_regime_signal)
                        tensor_action_valid = self.agent.act(state_valid, market_regime_signal_valid_tensor) # <--- 修正這裡
                    
                    # 處理離散/連續動作
                    if self.if_discrete:
                        action_idx_valid = tensor_action_valid.argmax(dim=1)
                        action_valid_np = action_idx_valid.detach().cpu().numpy()[0]
                    else:
                        action_valid_np = tensor_action_valid.detach().cpu().numpy()[0]
                    
                    next_state_valid_np, reward_valid, done_valid, info_valid_dict = self.valid_environment.step(action_valid_np)
                    temp_save_dict_valid = info_valid_dict # 持續更新，獲取最後的 info

                    episode_reward_sum += reward_valid
                    state_valid_np = self.valid_environment.reset() if done_valid else next_state_valid_np
                    
                    state_valid = torch.tensor(state_valid_np, dtype=torch.float32, device=self.device)
                    if state_valid.dim() == 3:
                        state_valid = state_valid.unsqueeze(0)
                    current_validation_step += 1
                
                print(f"Validation Episode Reward Sum: {episode_reward_sum:.4f}")
                
                valid_score_list.append(episode_reward_sum)
                save_dict_list.append(temp_save_dict_valid) # 存儲對應的 dict


                save_model(self.checkpoints_path,
                           epoch=rl_update_batch_count, # 使用 RL 更新批次計數作為 epoch 標識
                           save=self.agent.get_save())
                epoch_counter_for_rl += 1 
                if epoch_counter_for_rl <= self.epochs:
                    print(f"\nOverall RL Epoch progress: [{epoch_counter_for_rl}/{self.epochs}]")
                
            rl_update_batch_count += 1 # RL 更新批次計數器遞增

            if epoch_counter_for_rl > self.epochs:
                break

        max_index = np.argmax(valid_score_list)
        print(f"Best validation score: {valid_score_list[max_index]:.4f} at validation instance {max_index} (corresponds to model saved with epoch={max_index+1}).")

        plot_metric_against_baseline(total_asset=save_dict_list[max_index]['total_assets'],
                                     buy_and_hold=None, alg='Ensemble of Identical Independent Evaluators',
                                     task='valid', color='darkcyan', save_dir=self.work_dir)
        load_model(self.checkpoints_path,
                   epoch=max_index + 1,
                   save=self.agent.get_save())
        save_best_model(
            output_dir=self.checkpoints_path,
            epoch=max_index + 1,
            save=self.agent.get_save()
        )

    def test(self):
        # 確保加載的是最佳模型 (通常在訓練結束後執行測試)
        load_best_model(self.checkpoints_path, save=self.agent.get_save(), is_train=False)

        print("Test Best Episode")
        
        # <<< 新增：在測試階段進行 MarketNet 校準 (使用測試集數據) >>>
        if self.calibrate_marketnet_flag: # 只有當 flag 為 True 時才進行
            print("\n===== Calibrating MarketNet using TEST data before final test run (DATA LEAKAGE intended for debug/analysis) =====")
            # 使用 self.test_environment 進行校準
            self.calibrate_market_network(self.test_environment)
            print("===== MarketNet Calibration for TEST environment Finished =====")
        # <<< 新增結束 >>>

        state = self.test_environment.reset()

        episode_reward_sum = 0
        get_action = self.agent.act
        
        # 確保 Actor 和 MarketNet 處於評估模式
        get_action.eval()
        if hasattr(self.agent, 'market_net') and self.agent.market_net is not None:
            self.agent.market_net.eval()

        while True:
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # --- 新增：提取 s_market 並獲取 market_regime_signal ---   
            with torch.no_grad(): # 推斷不計算梯度
                s_market_test = self.agent.act.extract_market_features(tensor_state)
                logits_market_test = self.agent.market_net(s_market_test)
                market_regime_signal_test_tensor = torch.argmax(logits_market_test, dim=1)
                # print("market_regime_signal_test_tensor is ",market_regime_signal_test_tensor)
            # --- 新增結束 ---

            with torch.no_grad(): # 推斷不計算梯度
                # 傳遞 market_regime_signal 給 Actor
                tensor_action = get_action(tensor_state, market_regime_signal_test_tensor)
            
            if self.if_discrete:
                tensor_action = tensor_action.argmax(dim=1)
            action = tensor_action.detach().cpu().numpy()[0]
            
            state, reward, done, return_dict = self.test_environment.step(action)
            episode_reward_sum += reward
            
            if done:
                plot_metric_against_baseline(total_asset=return_dict['total_assets'],
                                             buy_and_hold=None, alg='Ensemble of Identical Independent Evaluators',
                                             task='test', color='darkcyan', save_dir=self.work_dir)
                break
        
        df_return = self.test_environment.save_portfolio_return_memory()
        df_assets = self.test_environment.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"))
        daily_return = df.daily_return.values
        return daily_return

    def test_with_customize_policy(self, policy, customize_policy_id,extra_parameters=None):
        state = self.test_environment.reset()
        self.test_environment.test_id = customize_policy_id
        print(f"Test customize policy: {str(customize_policy_id)}")

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