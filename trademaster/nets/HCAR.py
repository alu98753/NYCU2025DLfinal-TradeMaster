import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net
from trademaster.utils import build_conv2d
from torch import Tensor
import wandb
from trademaster.nets.HCAR_util import  (
    TemporalFeatureExtractor,
    RelationalContextIntegrator,
    AssetScoringHead,
    MultiHeadAttentionPooling,
)
from titans_pytorch.memory_models import MemoryMLP

@NETS.register_module()
class HCAR_Actor(nn.Module):
    def __init__(self,
                 # 子模塊 3.A (TemporalFeatureExtractor) 參數
                 num_original_features, # 例如 11
                 temporal_hidden_dim,   # 例如 64 (3.A的輸出維度，也是3.B的輸入維度)
                 tcn_kernel_size,
                 num_tcn_layers,
                 temporal_dropout,
                 
                 # 子模塊 3.B (RelationalContextIntegrator) 參數
                 num_stocks, # 即 num_nodes
                 relational_hidden_dim, # 例如 64 (3.B內部和輸出的維度，也是3.C的輸入維度)
                 num_gcn_layers,
                 relational_dropout,
                 supports, # 因果圖等
                 gcn_bool, spatialattn_bool, addaptiveadj,
                 
                 # 子模塊 3.C (AssetScoringHead) 參數
                 scoring_mlp_hidden_dims, # 例如 [32] 或 None
                 scoring_dropout,
                 
                s_market_extractor_config: dict,
                gate_controller_config: dict,
                 
                 # 用於整合到 EIIE Agent 的參數
                 output_final_weights=True, # 控制是否輸出最終權重
                 use_temporal_skip_to_scoring=True, # 是否啟用從3A到3C的跳躍
                 fusion_method_for_scoring='cat', # 'cat' for concatenation, 'add' for addition
                 

                 ):
        super(HCAR_Actor, self).__init__()

        self.temporal_feature_extractor = TemporalFeatureExtractor(
            in_features=num_original_features,
            hidden_dim=temporal_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            num_tcn_layers=num_tcn_layers,
            dropout=temporal_dropout
        )

        # s_market_extractor_config 從主設定檔傳入
        temporal_processed_dim = self.temporal_feature_extractor.temporal_attention_pool.attn.embed_dim # 或配置中的 self.temporal_hidden_dim
        self.s_market_stock_pool = MultiHeadAttentionPooling(
            input_dim=temporal_processed_dim,
            output_dim=s_market_extractor_config['s_market_dim'],
            num_heads=s_market_extractor_config['stock_pool_num_heads'],
            dropout=s_market_extractor_config['stock_pool_dropout']
        )
        # gate_controller_config 從主設定檔傳入
        self.gate_controller_mlp = MemoryMLP(
            dim=gate_controller_config['s_market_dim'],
            depth=gate_controller_config['controller_depth'],
            expansion_factor=gate_controller_config['controller_expansion_factor'],
        )
        self.cash_adjustment_scale = gate_controller_config['cash_adjustment_scale']
        self.ema_alpha_gate = gate_controller_config['ema_alpha_gate']

        # 初始化EMA的state，用 register_buffer 使其成為模型狀態但不參與梯度計算
        self.register_buffer('ema_stock_gate_prev', torch.tensor(0.5)) # 初始值設為中間值
        self.register_buffer('ema_cash_adjust_prev', torch.tensor(0.0))

        # 子模塊 3.B

        self.relational_context_integrator = RelationalContextIntegrator(
            num_nodes=num_stocks,
            in_feature_dim=temporal_hidden_dim, # 輸入來自 3.A 的輸出
            hidden_dim=relational_hidden_dim,
            num_gcn_layers=num_gcn_layers,
            dropout=relational_dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            spatialattn_bool=spatialattn_bool,
            addaptiveadj=addaptiveadj
        )
        self.use_temporal_skip_to_scoring = use_temporal_skip_to_scoring
        self.fusion_method_for_scoring = fusion_method_for_scoring
        
        # --- 根據是否使用跳躍連接和融合方式，動態計算 AssetScoringHead 的輸入維度 ---
        actual_scoring_head_input_dim = relational_hidden_dim # 默認情況 (只用 3.B 輸出)
        
        if self.use_temporal_skip_to_scoring:
            if self.fusion_method_for_scoring == 'cat':
                actual_scoring_head_input_dim = temporal_hidden_dim + relational_hidden_dim
            elif self.fusion_method_for_scoring == 'add':
                # 如果是相加，需要確保維度一致
                assert temporal_hidden_dim == relational_hidden_dim, \
                    "For 'add' fusion, temporal_hidden_dim and relational_hidden_dim must be equal."
                actual_scoring_head_input_dim = relational_hidden_dim # 或者 temporal_hidden_dim
            # 可以根據需要添加其他融合方式的處理
        # --------------------------------------------------------------------
                
        self.asset_scoring_head = AssetScoringHead(
            input_dim=actual_scoring_head_input_dim, # 輸入來自 3.B 的輸出
            hidden_layers_dims=scoring_mlp_hidden_dims,
            output_dim=1, # 每個股票一個評分
            dropout=scoring_dropout
        )
        
        self.output_final_weights = output_final_weights
        if self.output_final_weights:
            self.cash_bias_param = torch.nn.Parameter(torch.randn(1))

    def forward(self, stock_observations, asset_mask=None, global_step=None, current_dynamic_supports=None):
        # stock_observations: [batch_size, num_stocks, window_len, num_original_features]
        # asset_mask: [batch_size, num_stocks]
        # current_dynamic_supports: (可選) 如果模塊二提供了動態圖，可以在這裡傳入
        # stock_observations: [batch_size, num_envs, num_stocks, window_len, num_original_features]
        # 例如: [1, 1, 49, 10, 11] 在 explore_env 中
        # 或 [B, 1, 49, 10, 11] 在 update_net 中 (如果 buffer 存儲的是這種格式)
        # print(current_dynamic_supports)
        # print(f"HCAR_Actor - Input stock_observations shape (original): {stock_observations.shape}")

        if stock_observations.dim() == 5 and stock_observations.size(1) == 1: # 檢查是否是 [B, 1, N, T, F]
            stock_observations_squeezed = stock_observations.squeeze(1)
            # print(f"HCAR_Actor - stock_observations shape after squeeze(1): {stock_observations_squeezed.shape}")
        else:
            # 如果不是預期的5維且第二維為1，可能直接就是 [B, N, T, F]
            stock_observations_squeezed = stock_observations
            # print(f"HCAR_Actor - stock_observations shape (no squeeze needed or unexpected): {stock_observations_squeezed.shape}")
        if global_step is not None:
            wandb.log({
                "HCAR_Actor/0_Input_Obs_Squeezed_Mean": stock_observations_squeezed.mean().item(),
                "HCAR_Actor/0_Input_Obs_Squeezed_Std": stock_observations_squeezed.std().item(),
            }, step=global_step)
        # 子模塊 3.A
        h_temporal = self.temporal_feature_extractor(stock_observations_squeezed, global_step=global_step)
        # print(f"HCAR_Actor - Output from 3.A h_temporal mean: {h_temporal.mean().item()}")
        # print(f"HCAR_Actor - Output from 3.A h_temporal std: {h_temporal.std().item()}")

        # Smarket
        S_market = self.s_market_stock_pool(h_temporal) # [B, D_market]
        gate_outputs = self.gate_controller_mlp(S_market) # [B, 2]

        raw_stock_gate = gate_outputs[:, 0:1]    # [B, 1]
        raw_cash_adjust = gate_outputs[:, 1:2] # [B, 1]

        current_stock_gate = torch.sigmoid(raw_stock_gate) # (0, 1)
        current_cash_adjust = torch.tanh(raw_cash_adjust) * self.cash_adjustment_scale # (-scale, scale)

        # EMA 平滑 (僅在訓練時更新EMA，推理時使用最新的EMA值)
        if self.training:
            stock_gate_final = self.ema_alpha_gate * current_stock_gate + (1 - self.ema_alpha_gate) * self.ema_stock_gate_prev
            cash_adjust_final = self.ema_alpha_gate * current_cash_adjust + (1 - self.ema_alpha_gate) * self.ema_cash_adjust_prev
            # 更新 buffer 中的值 (in-place or reassign)
            self.ema_stock_gate_prev = stock_gate_final.detach().mean() # 保存批次均值作為下一次的prev，或者每個樣本獨立EMA
            self.ema_cash_adjust_prev = cash_adjust_final.detach().mean()
        else: # 推理時
            stock_gate_final = self.ema_stock_gate_prev.expand_as(current_stock_gate)
            cash_adjust_final = self.ema_cash_adjust_prev.expand_as(current_cash_adjust)

        # stock_gate_final 和 cash_adjust_final 將用於下一步

        # 子模塊 3.B
        h_relational = self.relational_context_integrator(h_temporal, global_step=global_step, current_supports=current_dynamic_supports)
        # print(f"HCAR_Actor - Output from 3.B h_relational mean: {h_relational.mean().item()}")
        # print(f"HCAR_Actor - Output from 3.B h_relational std: {h_relational.std().item()}")
        
        # --- 準備送入 AssetScoringHead 的特徵 (與 __init__ 中的邏輯對應) ---
        if self.use_temporal_skip_to_scoring:
            if self.fusion_method_for_scoring == 'cat':
                combined_features_for_scoring = torch.cat([h_temporal, h_relational], dim=-1)
            elif self.fusion_method_for_scoring == 'add':
                combined_features_for_scoring = h_temporal + h_relational
            else: 
                combined_features_for_scoring = h_relational 
        else:
            combined_features_for_scoring = h_temporal
        # -------------------------------------------------------------l
        
        if global_step is not None:
            wandb.log({
                "HCAR_Actor/3C_Scoring/0a_Input_CombinedFeatures_Mean": combined_features_for_scoring.mean().item(),
                "HCAR_Actor/3C_Scoring/0a_Input_CombinedFeatures_Std": combined_features_for_scoring.std().item(),
            }, step=global_step)
            
        # 子模塊 3.C
        stock_logits = self.asset_scoring_head(combined_features_for_scoring, global_step=global_step)
        adjusted_stock_logits = stock_logits * stock_gate_final # 股票 logits 被門控壓制

        # 輸出 stock_logits: [batch_size, num_stocks]
        # print(f"HCAR_Actor - Output from 3.C (stock_logits shape): {stock_logits.shape}")
        
        # 應用 mask (將無效股票的 logits 設為極小值)
        if asset_mask is not None:
            if adjusted_stock_logits.device != asset_mask.device:
                asset_mask = asset_mask.to(adjusted_stock_logits.device)
            adjusted_stock_logits[asset_mask] = -torch.finfo(adjusted_stock_logits.dtype).max

        if self.output_final_weights:
            cash_bias_repeated = self.cash_bias_param.repeat(stock_logits.shape[0], 1)# [B, 1]
            adjusted_cash_logit = cash_bias_repeated + cash_adjust_final

            if stock_logits.device != adjusted_cash_logit.device:
                adjusted_cash_logit = adjusted_cash_logit.to(stock_logits.device)
            
            final_logits = torch.cat([adjusted_stock_logits, adjusted_cash_logit], dim=1) # [B, N+1]
            action_probabilities = torch.softmax(final_logits, dim=1)
            # print(f"HCAR_Actor - Final action_probabilities shape: {action_probabilities.shape}")
            if global_step is not None:
                wandb.log({
                    "HCAR_Actor/4_Final_ActionProbs_Mean": action_probabilities.mean().item(),
                    "HCAR_Actor/4_Final_ActionProbs_Std": action_probabilities.std().item(),
                    "HCAR_Actor/4_CashBias_Value": self.cash_bias_param.item(),
                    
                    "Market/S_market_Mean": S_market.mean().item(), # 假設 S_market 在此 scope 可用
                    "Market/Stock_Gate_EMA_Mean": stock_gate_final.mean().item(),
                    "Market/Cash_Adjust_EMA_Mean": cash_adjust_final.mean().item(),
                    "Market/Raw_Stock_Gate_Mean": current_stock_gate.mean().item(),    # Log EMA之前的
                    "Market/Raw_Cash_Adjust_Mean": current_cash_adjust.mean().item(), # Log EMA之前的
      
                }, step=global_step)
            
            return action_probabilities
        else:
            # print(f"HCAR_Actor - Returning raw stock_logits shape: {stock_logits.shape}")
            return stock_logits

@NETS.register_module()
class HCAR_Critic(Net):
    def __init__(self,
                 input_dim,
                 action_dim,
                 output_dim=1,
                 time_steps=10,
                 num_layers= 1,
                 hidden_size = 32,
                 ):
        super(HCAR_Critic, self).__init__()

        self.time_steps = time_steps

        self.lstm = nn.LSTM(input_size=input_dim * time_steps,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear1 = nn.Linear(hidden_size, output_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(2 * (action_dim + 1), 1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x, a):
        if len(x.shape) >= 4:
            x = x.view(x.shape[0], x.shape[1], -1)
        lstm_out, _ = self.lstm(x)
        x = self.linear1(lstm_out)

        x = self.act(x)

        x = x.view(x.shape[0], -1)
        para = self.para.repeat(x.shape[0], 1)

        x = torch.cat((x, para, a), dim=1)
        x = self.linear2(x)
        # x = x.mean(dim = 1, keepdim=True)
        return x


# if __name__ == '__main__':
#     # --- 測試 HCAR_Actor ---
#     batch_size_test = 4
#     num_stocks_test = 49
#     window_len_test = 13 
#     num_original_features_test = 11

#     # 加載 supports
#     causal_graph_path_test = "/path/to/your/pc_causal_relation.npy" # 替換為您的路徑
#     try:
#         relation_matrix_np = np.load(causal_graph_path_test)
#         supports_list_test = [torch.from_numpy(relation_matrix_np).float()]
#     except:
#         supports_list_test = [torch.eye(num_stocks_test).float()]


#     # print("\n--- Initializing HCAR_Actor ---")
#     hcar_actor = HCAR_Actor(
#         num_original_features=num_original_features_test,
#         temporal_hidden_dim=64,
#         tcn_kernel_size=2,
#         num_tcn_layers=3,
#         temporal_dropout=0.3,
#         num_stocks=num_stocks_test,
#         relational_hidden_dim=64,
#         num_gcn_layers=2,
#         relational_dropout=0.3,
#         supports=supports_list_test,
#         gcn_bool=True, spatialattn_bool=True, addaptiveadj=True,
#         scoring_mlp_hidden_dims=[32],
#         scoring_dropout=0.3,
#         output_final_weights=True
#     )
#     hcar_actor.eval()

#     # print("\n--- Preparing Fake Input for HCAR_Actor ---")
#     fake_stock_obs = torch.randn(batch_size_test, num_stocks_test, window_len_test, num_original_features_test)
#     fake_asset_mask = torch.zeros(batch_size_test, num_stocks_test, dtype=torch.bool)
#     # print(f"Shape of fake_stock_obs: {fake_stock_obs.shape}")
#     # print(f"Shape of fake_asset_mask: {fake_asset_mask.shape}")


#     # print("\n--- Running Forward Pass through HCAR_Actor ---")
#     try:
#         with torch.no_grad():
#             portfolio_weights = hcar_actor(fake_stock_obs, fake_asset_mask)
#             # print("--- Forward Pass Successful ---")
#             # print(f"Output portfolio_weights shape: {portfolio_weights.shape}")
#             assert portfolio_weights.shape == (batch_size_test, num_stocks_test + 1), "Output shape mismatch!"
#             assert torch.allclose(torch.sum(portfolio_weights, dim=1), torch.tensor(1.0)), "Portfolio weights do not sum to 1!"

#             # print(f"Example portfolio_weights (first sample): \n{portfolio_weights[0]}")

#     except Exception as e:
#         # print(f"Error during HCAR_Actor forward pass: {e}")
#         import traceback
#         traceback.print_exc()
        