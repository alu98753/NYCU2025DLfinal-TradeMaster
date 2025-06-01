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
from titans_pytorch.memory_models import MemoryMLP,GatedResidualMemoryMLP
@NETS.register_module()
class MarketNet(nn.Module):
    def __init__(
        self,
        s_market_dim: int,
        hidden_depth: int,
        expansion_factor: float,
        market_lr: float = 1e-4
    ):
        super(MarketNet, self).__init__()
        self.market_lr = market_lr
        self.mem_mlp = MemoryMLP(
            dim=s_market_dim,
            depth=hidden_depth,
            expansion_factor=expansion_factor
        )
        self.bn = nn.BatchNorm1d(s_market_dim)
        self.dropout = nn.Dropout(p=0.1)

        # 把单层线性 -> 改成 “线性 → GELU → 线性”
        self.classifier = nn.Sequential(
            nn.Linear(s_market_dim, s_market_dim),
            nn.GELU(),
            nn.Linear(s_market_dim, 2)
        )

    def forward(self, s_market: torch.Tensor):
        h = self.mem_mlp(s_market)  # [B, 32]
        h = self.bn(h)
        h = self.dropout(h)
        logits = self.classifier(h) # [B,2]
        regime_probs = torch.softmax(logits, dim=1)
        return regime_probs, logits


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
        self.s_market_dim = s_market_extractor_config['s_market_dim']
        temporal_processed_dim = self.temporal_feature_extractor.temporal_attention_pool.attn.embed_dim # 或配置中的 self.temporal_hidden_dim
        self.s_market_stock_pool = MultiHeadAttentionPooling(
            input_dim=temporal_processed_dim,
            output_dim=s_market_extractor_config['s_market_dim'],
            num_heads=s_market_extractor_config['stock_pool_num_heads'],
            dropout=s_market_extractor_config['stock_pool_dropout']
        )
        # gate_controller_config 從主設定檔傳入
        # self.gate_controller_mlp = MemoryMLP(
        #     dim=gate_controller_config['s_market_dim'],
        #     depth=gate_controller_config['controller_depth'],
        #     expansion_factor=gate_controller_config['controller_expansion_factor'],
        # )
        # self.cash_adjustment_scale = gate_controller_config['cash_adjustment_scale']
        # self.ema_alpha_gate = gate_controller_config['ema_alpha_gate']

        # # 初始化EMA的state，用 register_buffer 使其成為模型狀態但不參與梯度計算
        # self.register_buffer('ema_stock_gate_prev', torch.tensor(0.5)) # 初始值設為中間值
        # self.register_buffer('ema_cash_adjust_prev', torch.tensor(0.0))

        # ===== 新增：條件化股票與現金的參數，用於根據 regime_probs 計算縮放與偏置 =====
        # 3 個 market regime (0=Sideways, 1=Bull, 2=Bear)
        self.num_regimes = 2
        # 每種市場狀態下整體股票打分的縮放係數 (初始化為 1.0)
        self.regime_stock_scales_actor = nn.Parameter(torch.ones(self.num_regimes, 1)) # 初始為 [[1],[1]]，表示二種市場預設下股票得分不縮放；訓練時，Actor 可學到在熊市把它調小、牛市調大。
        # 每種市場狀態下的現金 logit 偏置 (初始化為 0.0)
        self.regime_cash_biases_actor = nn.Parameter(torch.zeros(self.num_regimes, 1)) # 初始為 [[0],[0]]，表示二種市場預設下現金 logit 為 0；訓練時，Actor 可學到在熊市把它調大（誘導高現金比）或牛市給較小值。

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
        actual_scoring_head_input_dim = relational_hidden_dim
        if self.use_temporal_skip_to_scoring:
            if self.fusion_method_for_scoring == 'cat':
                actual_scoring_head_input_dim = temporal_hidden_dim + relational_hidden_dim
            elif self.fusion_method_for_scoring == 'add':
                assert temporal_hidden_dim == relational_hidden_dim, \
                    "For 'add' fusion, temporal_hidden_dim must equal relational_hidden_dim."
                actual_scoring_head_input_dim = relational_hidden_dim
        # --------------------------------------------------------------------
                
        self.asset_scoring_head = AssetScoringHead(
            input_dim=actual_scoring_head_input_dim,
            # input_dim=actual_scoring_head_input_dim+ self.num_regimes,
            hidden_layers_dims=scoring_mlp_hidden_dims,
            output_dim=1, # 每個股票一個評分
            dropout=scoring_dropout
        )
        
        self.output_final_weights = output_final_weights
        if self.output_final_weights:
            self.cash_bias_param = torch.nn.Parameter(torch.randn(1))
        
        self.print_xor = 0
        
    def forward(self, stock_observations,regime_probs , asset_mask=None, global_step=None, current_dynamic_supports=None):
        # 例如: [1, 1, 49, 10, 11] 在 explore_env 中
        # 或 [B, 1, 49, 10, 11] 在 update_net 中 (如果 buffer 存儲的是這種格式)
        # print(f"HCAR_Actor - Input stock_observations shape (original): {stock_observations.shape}")
        '''
        # stock_observations: [batch_size, num_stocks, window_len, num_original_features]
        # regime_probs: [B, 3] (one-hot 或概率分佈，由 MarketNet 產生)
        # asset_mask: [batch_size, num_stocks]
        '''
        if stock_observations.dim() == 5 and stock_observations.size(1) == 1: # 檢查是否是 [B, 1, N, T, F]
            stock_observations_squeezed = stock_observations.squeeze(1)
            # print(f"HCAR_Actor - stock_observations shape after squeeze(1): {stock_observations_squeezed.shape}")
        else:
            # 如果不是預期的5維且第二維為1，可能直接就是 [B, N, T, F]
            stock_observations_squeezed = stock_observations
            # print(f"HCAR_Actor - stock_observations shape (no squeeze needed or unexpected): {stock_observations_squeezed.shape}")
        if global_step is not None and global_step % 1000  == 0:
            wandb.log({
                "HCAR_Actor/0_Input_Obs_Squeezed_Mean": stock_observations_squeezed.mean().item(),
                "HCAR_Actor/0_Input_Obs_Squeezed_Std": stock_observations_squeezed.std().item(),
                "agent_step": global_step,
            })
        # 子模塊 3.A
        h_temporal = self.temporal_feature_extractor(stock_observations_squeezed, global_step=global_step)
        # print(f"HCAR_Actor - Output from 3.A h_temporal mean: {h_temporal.mean().item()}")
        # print(f"HCAR_Actor - Output from 3.A h_temporal std: {h_temporal.std().item()}")

        # # Smarket
        # S_market = self.s_market_stock_pool(h_temporal) # [B, D_market]
        # gate_outputs = self.gate_controller_mlp(S_market) # [B, 2]

        # raw_stock_gate = gate_outputs[:, 0:1]    # [B, 1]
        # raw_cash_adjust = gate_outputs[:, 1:2] # [B, 1]

        # current_stock_gate = torch.sigmoid(raw_stock_gate) # (0, 1)
        # current_cash_adjust = torch.tanh(raw_cash_adjust) * self.cash_adjustment_scale # (-scale, scale)

        # # EMA 平滑 (僅在訓練時更新EMA，推理時使用最新的EMA值)
        # if self.training:
        #     stock_gate_final = self.ema_alpha_gate * current_stock_gate + (1 - self.ema_alpha_gate) * self.ema_stock_gate_prev
        #     cash_adjust_final = self.ema_alpha_gate * current_cash_adjust + (1 - self.ema_alpha_gate) * self.ema_cash_adjust_prev
        #     # 更新 buffer 中的值 (in-place or reassign)
        #     self.ema_stock_gate_prev = stock_gate_final.detach().mean() # 保存批次均值作為下一次的prev，或者每個樣本獨立EMA
        #     self.ema_cash_adjust_prev = cash_adjust_final.detach().mean()
        # else: # 推理時
        #     stock_gate_final = self.ema_stock_gate_prev.expand_as(current_stock_gate)
        #     cash_adjust_final = self.ema_cash_adjust_prev.expand_as(current_cash_adjust)

        # stock_gate_final 和 cash_adjust_final 將用於下一步

        # 子模塊 3.B
        h_relational = self.relational_context_integrator(h_temporal, global_step=global_step, current_supports=current_dynamic_supports)
        # print(f"HCAR_Actor - Output from 3.B h_relational mean: {h_relational.mean().item()}")
        # print(f"HCAR_Actor - Output from 3.B h_relational std: {h_relational.std().item()}")
        
        # --- 準備送入 AssetScoringHead 的特徵 (與 __init__ 中的邏輯對應) ---
        if self.use_temporal_skip_to_scoring:
            if self.fusion_method_for_scoring == 'cat':
                feat = torch.cat([h_temporal, h_relational], dim=-1)  # [B,N,D_temporal+D_rel]
            else:  # add 拼法 (假設 D_temporal==D_rel)
                feat = h_temporal + h_relational  # [B, N, D_rel]
        else:
            feat = h_relational

            
        # --- 準備送入 AssetScoringHead 的特徵，並拼接 regime_probs 作為條件信號 ---
        B, N, D_rel = feat.shape  # h_relational: [B, N, D_rel (或 D_temporal+D_rel)]

        # regime_probs: [B, 3] -> expand to [B, N, 3]
        regime_context = regime_probs.unsqueeze(1).repeat(1, N, 1)  # [B,N,3]
        # 然後把 regime_context 和 h_relational 拼在一起：
        h_for_scoring = torch.cat([feat, regime_context], dim=-1)  # [B,N,(D_x)+3]
        
        if global_step is not None and global_step % 1000  == 0:
            wandb.log({
                "HCAR_Actor/3C_Scoring/0a_Input_CombinedFeatures_Mean": h_for_scoring.mean().item(),
                "HCAR_Actor/3C_Scoring/0a_Input_CombinedFeatures_Std": h_for_scoring.std().item(),
                "agent_step": global_step,
            })
            
        # 子模塊 3.C
        stock_logits = self.asset_scoring_head(feat, global_step=global_step) # [batch_size, num_stocks]
        # print(f"HCAR_Actor - Output from 3.C (stock_logits shape): {stock_logits.shape}")
        
        # ─── 計算條件化縮放 α 與現金偏置 c ────────────────────────
        # regime_probs: [B, 3]
        # regime_stock_scales_actor: [3,1], regime_cash_biases_actor: [3,1]
        alpha = regime_probs @ self.regime_stock_scales_actor  # [B,1]
        c = regime_probs @ self.regime_cash_biases_actor       # [B,1]


        if global_step is not None and global_step % 1000  == 0:
            wandb.log({
                "HCAR_Actor_Logits/stock_logits_std": stock_logits.std().item(),
                "HCAR_Actor_Logits/stock_logits_mean": stock_logits.mean().item(),
                "HCAR_Actor_Logits/cash_logit_c_std": c.std().item(), # c is likely [B,1]
                "HCAR_Actor_Logits/cash_logit_c_mean": c.mean().item(), # c is likely [B,1]
                "agent_step": global_step,
            })
        
        # 在 HCAR_Actor forward 中臨時修改：
        # alpha = torch.ones_like(alpha) # 強制 alpha 為 1
        # c = torch.zeros_like(c)     # 強制 c 為 0
        # scaled_stock_logits: [B, N]
        scaled_stock_logits = stock_logits * alpha

        # 如有 asset_mask，將被屏蔽的股票 logits 設為 -inf（或非常小）
        if asset_mask is not None:
            if scaled_stock_logits.device != asset_mask.device:
                asset_mask = asset_mask.to(scaled_stock_logits.device)
            scaled_stock_logits = scaled_stock_logits.masked_fill(asset_mask, float('-1e9'))

        # ─── 拼接股票 logits 與現金 logit，再做 softmax ─────────────────
        combined_logits = torch.cat([scaled_stock_logits, c], dim=1)  # [B, N+1]
        action_probs = torch.softmax(combined_logits, dim=1)         # [B, N+1], Σ=1
        if global_step is not None:
            if (global_step ^ self.print_xor) == 0:
                if self.print_xor % 3000 == 0:
                    print("global_step:",global_step)
                    print("combined_logits:",combined_logits)
                    print("action_probs:",action_probs)
                self.print_xor +=1
        if global_step is None:
            print("combined_logits:",combined_logits)
            print("action_probs:",action_probs)
        if global_step is not None and global_step % 1000 == 0:
            wandb.log({
                "HCAR_Actor_Logits/scaled_stock_after_logits_mean": scaled_stock_logits.mean().item(),
                "HCAR_Actor_Logits/scaled_stock_after_logits_std": scaled_stock_logits.std().item(),
                "HCAR_Actor_Logits/cash_logit_after_c_mean": c.mean().item(), # c is likely [B,1]
        
                
                "HCAR_Actor/4_ActionProbs_Mean": action_probs.mean().item(),
                "HCAR_Actor/4_ActionProbs_Std": action_probs.std().item(),
                "agent_step": global_step,
            })

        return action_probs, regime_probs # 返回：action_probs 與 regime_probs（後續 Critic 需要 regime_probs）

# 在 trademaster/nets/HCAR.py 的 HCAR_Critic 類的 __init__ 方法中

@NETS.register_module()
class HCAR_Critic(Net): # 確保繼承自 Net (如果 Net 是你的基礎網路類)
    def __init__(self,
                 input_dim,          # 每支股票的原始特征维度 F_in (由環境的 state_dim 提供)
                 action_dim,         # 股票数量 N (由環境的 action_dim 提供)
                 time_steps,         # 時間窗口長度 (由環境的 time_steps 提供)
                 num_regimes,        # <--- 修改點: MarketNet 輸出的 regime_probs 的維度 (例如 2 或 3)
                 output_dim=1,
                 num_layers=1,       # LSTM layers
                 hidden_size=32,     # LSTM hidden_size
                 ):
        super(HCAR_Critic, self).__init__()

        self.time_steps = time_steps
        self.num_stocks = action_dim  # N
        self.num_regimes = num_regimes # <--- 新增: 存儲 regime 維度

        # --- LSTM 部分 ---
        # input_dim 這裡指的是單個股票的原始特徵維度 (F_in)
        # LSTM 的 input_size 應該是 T * F_in
        self.lstm = nn.LSTM(
            input_size=input_dim * time_steps, # 確保 input_dim 是 F_in
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear1 = nn.Linear(hidden_size, output_dim) # output_dim 通常是 1
        self.relu_act = nn.ReLU() # 重命名以避免與 agent.act 衝突

        # --- 計算最終線性層 linear2 的輸入維度 ---
        # 1) x_flat: 每個股票LSTM處理後再經過linear1得到的維度 (N * output_dim, output_dim 通常是 1, 所以是 N)
        #    如果 self.linear1 的 output_dim 是 1, 那麼 x_flat 的維度是 N * 1 = N
        in_features_x_flat = self.num_stocks * output_dim

        # 2) para: 1 維
        in_features_para = 1

        # 3) action a: [B, N+1] (N+1 維)
        in_features_action = self.num_stocks + 1

        # 4) regime_probs: [B, num_regimes] (num_regimes 維)
        in_features_regime = self.num_regimes

        total_in_features_linear2 = in_features_x_flat + \
                                    in_features_para + \
                                    in_features_action + \
                                    in_features_regime
        
        self.linear2 = nn.Linear(total_in_features_linear2, 1) # 最終輸出 Q 值
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x: torch.Tensor, a: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        """
        x: 股票觀測值 [B, N, T, F_in] (N=num_stocks, T=time_steps, F_in=input_dim_per_stock)
        a: 投資組合動作 [B, N+1]
        regime_probs: 市場狀態概率 [B, num_regimes]
        """
        B = x.shape[0] # 獲取 batch_size

        # 1. LSTM 處理每個股票的時序特徵
        # x 的原始形狀是 [B, N, T, F_in]
        # LSTM期望輸入 [B*N, T, F_in] 或 [B, N, T*F_in] 取決於設計
        # 你的 LSTM input_size = input_dim * time_steps，且 batch_first=True
        # 所以 LSTM 輸入應為 [B, N, T*F_in]
        
        # 確認 x 的維度
        if x.dim() == 4: # [B, N, T, F_in]
            #  N = x.shape[1] # num_stocks
            #  T = x.shape[2] # time_steps
            #  F_in = x.shape[3] # features_per_stock
            #  x_lstm_input = x.reshape(B * N, T, F_in) # 如果LSTM逐個處理股票
            x_lstm_input = x.view(B, self.num_stocks, -1) # [B, N, T*F_in]
        elif x.dim() == 3: # 假設已經是 [B, N, T*F_in]
            x_lstm_input = x
        else:
            raise ValueError(f"Unsupported input shape for x: {x.shape}")

        lstm_out, _ = self.lstm(x_lstm_input)  # lstm_out: [B, N, hidden_size]
        
        # 2. 將 LSTM 輸出通過 linear1 和激活函數
        x1 = self.linear1(lstm_out)  # x1: [B, N, output_dim_linear1] (output_dim_linear1 通常是 1)
        x1 = self.relu_act(x1)

        # 3. Flatten x1 準備拼接
        x_flat = x1.view(B, -1)  # x_flat: [B, N * output_dim_linear1]

        # 4. 準備 para
        para_rep = self.para.repeat(B, 1)  # para_rep: [B, 1]

        # 5. 檢查 regime_probs 的形狀
        assert regime_probs.shape == (B, self.num_regimes), \
               f"Critic: Expected regime_probs shape [B, {self.num_regimes}], but got {regime_probs.shape}"

        # 6. 拼接所有特徵
        combined = torch.cat((x_flat, para_rep, a, regime_probs), dim=1)
        # 預期維度: (N*output_dim_linear1) + 1 + (N+1) + num_regimes

        # 7. 通過最終的線性層得到 Q 值
        q_value = self.linear2(combined)  # q_value: [B, 1]
        return q_value


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
        