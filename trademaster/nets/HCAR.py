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
                 
                 # 用於整合到 EIIE Agent 的參數
                 output_final_weights=True, # 控制是否輸出最終權重
                 use_temporal_skip_to_scoring=True, # 是否啟用從3A到3C的跳躍
                 fusion_method_for_scoring='cat' # 'cat' for concatenation, 'add' for addition
                 
                 ):
        super(HCAR_Actor, self).__init__()

        self.temporal_feature_extractor = TemporalFeatureExtractor(
            in_features=num_original_features,
            hidden_dim=temporal_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            num_tcn_layers=num_tcn_layers,
            dropout=temporal_dropout
        )
        
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
        
        # 輸出 stock_logits: [batch_size, num_stocks]
        # print(f"HCAR_Actor - Output from 3.C (stock_logits shape): {stock_logits.shape}")
        
        # 應用 mask (將無效股票的 logits 設為極小值)
        if asset_mask is not None:
            if stock_logits.device != asset_mask.device:
                asset_mask = asset_mask.to(stock_logits.device)
            stock_logits[asset_mask] = -torch.finfo(stock_logits.dtype).max

        if self.output_final_weights:
            cash_bias_repeated = self.cash_bias_param.repeat(stock_logits.shape[0], 1)
            if stock_logits.device != cash_bias_repeated.device:
                cash_bias_repeated = cash_bias_repeated.to(stock_logits.device)
            
            final_logits = torch.cat([stock_logits, cash_bias_repeated], dim=1)
            action_probabilities = torch.softmax(final_logits, dim=1)
            # print(f"HCAR_Actor - Final action_probabilities shape: {action_probabilities.shape}")
            if global_step is not None:
                wandb.log({
                    "HCAR_Actor/4_Final_ActionProbs_Mean": action_probabilities.mean().item(),
                    "HCAR_Actor/4_Final_ActionProbs_Std": action_probabilities.std().item(),
                    "HCAR_Actor/4_CashBias_Value": self.cash_bias_param.item(),
                }, step=global_step)
            
            return action_probabilities
        else:
            # print(f"HCAR_Actor - Returning raw stock_logits shape: {stock_logits.shape}")
            return stock_logits


# ---  ASU.py 中的 nconv, linear, GraphConvNet, SpatialAttentionLayer ---
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()
    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)


# TemporalAttentionPool 保持不變
class TemporalAttentionPool(nn.Module):
    def __init__(self, in_len, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_len, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_len)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, N, T_final_len]
        B, C, N, T_final_len = x.shape
        x_permuted = x.permute(0, 2, 3, 1)
        x_reshaped = x_permuted.reshape(B * N, T_final_len, C)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        out_res = x_reshaped + self.dropout_layer(attn_out)
        out_norm = self.norm(out_res)
        out_pooled = out_norm.mean(dim=1)
        out_final = out_pooled.view(B, N, C)
        return out_final

class PermuteAndApply(nn.Module):
    def __init__(self, module_to_apply): # 移除了 activation_fn
        super().__init__()
        self.module_to_apply = module_to_apply

    def forward(self, x_bcn_t): 
        x_bntc = x_bcn_t.permute(0, 2, 3, 1) 
        x_processed_bntc = self.module_to_apply(x_bntc) # module_to_apply 通常是 LayerNorm
        return x_processed_bntc.permute(0, 3, 1, 2)

class TCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.0, activation_fn_class=nn.GELU):
        super().__init__()
        self.norm1 = PermuteAndApply(nn.LayerNorm(channels)) 
        self.act1 = activation_fn_class()
        
        self.causal_padding_amount = dilation * (kernel_size - 1)
        
        self.conv1 = nn.Conv2d(channels, channels, 
                               kernel_size=(1, kernel_size), 
                               dilation=(1, dilation))

        self.dropout1 = nn.Dropout(dropout)
        
        self.skip_connection_op = nn.Identity() # 假設通道數不變

        # --- 新增：在殘差相加後使用的 LayerNorm ---
        self.norm2 = PermuteAndApply(nn.LayerNorm(channels))
        # 通常在殘差塊的最後不加激活，或者讓下一塊的 pre-activation 來處理
        # 如果要加，可以再加 self.act2 = activation_fn_class()

    def forward(self, x): # x: [B, C, N, T_in]
        x_identity = x 

        # 主路徑: Norm -> Activation -> Pad -> Conv -> Dropout
        out_norm1 = self.norm1(x)       
        out_act1  = self.act1(out_norm1)      
        
        out_padded = nn.functional.pad(out_act1, (self.causal_padding_amount, 0)) 
        
        out_conv1 = self.conv1(out_padded)     
        out_main = self.dropout1(out_conv1)  

        # 對齊跳過路徑的時間維度
        x_skip_aligned = self.skip_connection_op(x_identity) # Identity op
        if out_main.shape[3] != x_skip_aligned.shape[3]:
            diff = x_skip_aligned.shape[3] - out_main.shape[3]
            if diff > 0:
                x_skip_aligned = x_skip_aligned[:, :, :, diff:]
            else:
                # 由於因果 padding，out_main 的 T 應該等於 x_identity 的 T
                # 如果不相等，說明 padding 或卷積設置有問題
                 raise ValueError(f"TCN block main path T={out_main.shape[3]} "
                                 f"not equal to skip path T={x_skip_aligned.shape[3]} after causal padding.")
            
        # 殘差連接
        x_added = out_main + x_skip_aligned
        
        # --- 新增：在殘差相加後進行規範化 ---
        x_final_out = self.norm2(x_added)
        # 如果需要，可以在這裡再加一個激活: x_final_out = self.act2(x_final_out)
            
        return x_final_out


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, in_features, hidden_dim, tcn_kernel_size=2, num_tcn_layers=1, dropout=0.0): # 默認 dropout=0
        super().__init__()
        self.layers = num_tcn_layers
        self.tcn_blocks = nn.ModuleList()
        self.temporal_attention_pool = TemporalAttentionPool(in_len=hidden_dim, dropout=dropout)

        self.start_conv = nn.Conv2d(in_features, hidden_dim, kernel_size=(1,1))
        # --- 修改：start_conv 之後也遵循 Norm -> Activation 模式 ---
        self.start_norm = PermuteAndApply(nn.LayerNorm(hidden_dim))
        self.start_activation = nn.GELU() # 或者您選擇的激活函數

        nn.init.kaiming_normal_(self.start_conv.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # LeakyReLU/GELU 適合
        if self.start_conv.bias is not None:
            nn.init.constant_(self.start_conv.bias, 0.)

        rf_calc, dilation_calc_rf = 1,1
        for _ in range(num_tcn_layers):
            rf_calc += dilation_calc_rf * (tcn_kernel_size -1)
            dilation_calc_rf *=2
        self.receptive_field_causal = rf_calc
        print(f"TemporalFeatureExtractor Initialized. Causal Receptive field: {self.receptive_field_causal}")

        dilation = 1
        for i in range(num_tcn_layers):
            self.tcn_blocks.append(
                TCNResidualBlock(
                    channels=hidden_dim,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation_fn_class=nn.GELU 
                )
            )
            dilation *= 2

    def forward(self, X_input, global_step=None):
        # X_input: [B, N, T_window, F_in]
        # print(f"debug xinput:{X_input.shape}")
        if global_step is not None:
            wandb.log({"HCAR_Actor/3A_Temporal/0_Input_X_Mean": X_input.mean().item(),
                       "HCAR_Actor/3A_Temporal/0_Input_X_Std":  X_input.std().item()}, step=global_step)

        x = X_input.permute(0,3,1,2) # → [B, F_in, N, T_window]
        # print(f"DEBUG: TFE.forward - After permute x:{x.shape}")
        # print(f"DEBUG: TFE.forward - padding check:{x.shape[3] < self.receptive_field_causal}")
        if x.shape[3] < self.receptive_field_causal: 
            padding_needed = self.receptive_field_causal - x.shape[3]
            x = nn.functional.pad(x, (padding_needed, 0))
            # if global_step is not None and global_step % 200 == 0:
            #      print(f"Padding input for TFE: original T={x.shape[3]}, target T for receptive field={self.receptive_field_causal}, padded to T={x.shape[3]}")
        
        # print(f"DEBUG: TFE.forward - After padding x:{x.shape}")
        x = self.start_conv(x)   # [B, C_hidden, N, T_window_padded_or_original]
        # print(f"DEBUG: TFE.forward - After start_conv x:{x.shape}")

        # --- 修改：應用初始的 Norm 和 Activation ---
        x = self.start_norm(x)
        x = self.start_activation(x) # x 現在是第一個 TCN block 的輸入
        
        if global_step is not None:
            wandb.log({"HCAR_Actor/3A_Temporal/1_After_StartProcessing_Mean": x.mean().item(), # 日誌名可改為更準確的
                       "HCAR_Actor/3A_Temporal/1_After_StartProcessing_Std":  x.std().item()}, step=global_step)
            
        for i in range(self.layers):
            x = self.tcn_blocks[i](x) # 每個塊內部處理殘差和規範化
            if global_step is not None:
                wandb.log({f"HCAR_Actor/3A_Temporal/2_Layer{i}_BlockOutput_Mean": x.mean().item(),
                           f"HCAR_Actor/3A_Temporal/2_Layer{i}_BlockOutput_Std":  x.std().item()}, step=global_step)

        out = self.temporal_attention_pool(x) 
        
        if global_step is not None:
            wandb.log({"HCAR_Actor/3A_Temporal/3_Final_h_temporal_Mean": out.mean().item(),
                       "HCAR_Actor/3A_Temporal/3_Final_h_temporal_Std":  out.std().item()}, step=global_step)
        return out


class GraphConvNet(nn.Module):
    # ... (保持不變)
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(GraphConvNet, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = nn.functional.dropout(h, self.dropout, training=self.training)
        return h

class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_nodes, in_features, in_len=1): # in_features is F_actual from 3.A, in_len is 1
        super().__init__()
        assert in_len == 1, "SpatialAttentionLayer expects in_len=1 for this version of HCAR"
        self.in_len = in_len 
        
        # Linear layers
        self.W1 = nn.Linear(in_len, 1)      # Operates on the pseudo-time dim (size 1) -> output size 1
        self.W2 = nn.Linear(in_features, 1) # Operates on feature_dim F_actual, projects to 1
        self.W3 = nn.Linear(in_features, 1) # Operates on feature_dim F_actual, projects to 1
        self.V = nn.Linear(num_nodes, num_nodes) # For final attention score transformation

        # LayerNorm layers
        # Normalize over the feature dimension (in_features) for each stock
        self.ln1 = nn.LayerNorm(in_features) 
        # After W2 and W3, the feature dimension becomes 1.
        # Normalizing a single dimension might not be very impactful but can be kept for consistency or removed.
        # If kept, it normalizes this single value (effectively just recentering if learnable params are false).
        self.ln2 = nn.LayerNorm(1) 
        self.ln3 = nn.LayerNorm(1)

    def forward(self, inputs_relational):
        # inputs_relational shape: [B, F_actual, N, 1] (F_actual=in_features to this layer, N=num_nodes)
        B, F_actual, N, _ = inputs_relational.shape
        
        x_for_sa = inputs_relational # [B, F_actual, N, 1]

        # Path 1
        # permute to [B, N, F_actual, 1] to process each stock's features
        p1_path = x_for_sa.permute(0, 2, 1, 3)  
        p1_after_W1 = self.W1(p1_path).squeeze(-1) # [B, N, F_actual] (W1 operates on last dim, which is 1)
        p1_after_ln1 = self.ln1(p1_after_W1)      # Normalize over F_actual for each stock: [B, N, F_actual]
        
        p1_after_W2 = self.W2(p1_after_ln1)       # [B, N, 1] (W2 projects F_actual to 1)
        part1_final_for_bmm = self.ln2(p1_after_W2) # Normalize over the last dim (size 1): [B, N, 1]

        # Path 2
        # permute to [B, N, 1, F_actual] to process features for W3
        p2_path = x_for_sa.permute(0, 2, 3, 1) 
        p2_squeezed = p2_path.squeeze(-2) # [B, N, F_actual] (Squeeze the pseudo-time dim of size 1)
        # Note: W3 expects F_actual features. If p2_path directly used, W3 expects last dim to be F_actual.
        # So, permute(0,2,3,1) -> [B,N,1,F_actual]. W3 is Linear(F_actual,1).
        # self.W3(p2_path) -> [B,N,1,1].squeeze(-1) -> [B,N,1]
        p2_after_W3 = self.W3(p2_path.squeeze(-2)) if p2_path.shape[-2] == 1 else self.W3(p2_path) # Handle if squeeze already happened implicitly
        p2_after_W3 = p2_after_W3.view(B, N, 1) # Ensure [B,N,1]
        
        p2_after_ln3 = self.ln3(p2_after_W3)          # Normalize over the last dim (size 1): [B, N, 1]
        part2_final_for_bmm = p2_after_ln3.permute(0, 2, 1) # [B, 1, N] for bmm

        # Similarity and Attention Weights
        S_raw = torch.bmm(part1_final_for_bmm, part2_final_for_bmm) # [B, N, 1] @ [B, 1, N] -> [B, N, N]
        S_activated = torch.relu(S_raw) # Apply ReLU before V only if V is meant to operate on non-negative, or if V includes its own bias and non-linearity
        S = torch.softmax(self.V(S_activated), dim=-1) # Or self.V(torch.relu(S_raw)) as you had.
        
        return S

# --- 子模塊 3.B ---
class RelationalContextIntegrator(nn.Module):
    def __init__(self, num_nodes, in_feature_dim, hidden_dim, num_gcn_layers=2,
                 dropout=0.3, supports=None,
                 gcn_bool=True, spatialattn_bool=True,
                 addaptiveadj=True, aptinit=None):
        super(RelationalContextIntegrator, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_bool = gcn_bool
        self.spatialattn_bool = spatialattn_bool
        self.addaptiveadj = addaptiveadj

        # 初始線性映射和規範化
        self.start_linear = nn.Linear(in_feature_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.start_linear.weight)
        if self.start_linear.bias is not None:
            nn.init.zeros_(self.start_linear.bias)
        self.ln_start_relational = nn.LayerNorm(hidden_dim)

        # 只有當 GCN 或 SA 啟用時，才初始化相關的層列表
        if self.gcn_bool or self.spatialattn_bool:
            self.gcns = nn.ModuleList()
            self.sans = nn.ModuleList()
            self.gcn_residual_linears = nn.ModuleList()
            self.gcn_bns = nn.ModuleList() # 用於 GCN/SA 塊後的 LayerNorm

            self.supports = supports # 基礎 supports (e.g., causal graph)
            self.supports_len = 0
            if supports is not None:
                self.supports_len += len(supports)

            if self.gcn_bool and self.addaptiveadj:
                if aptinit is None:
                    if self.supports is None: self.supports = [] # 確保 self.supports 是列表
                    self.nodevec = nn.Parameter(torch.randn(num_nodes, 1), requires_grad=True)
                    self.supports_len += 1
                else:
                    raise NotImplementedError
            
            for l in range(self.num_gcn_layers):
                if self.gcn_bool:
                    self.gcns.append(
                        GraphConvNet(hidden_dim, hidden_dim, dropout, support_len=self.supports_len)
                    )
                else: # 即使 gcn_bool 為 False，如果 spatialattn_bool 為 True，我們可能仍需要佔位符或確保列表長度一致
                    self.gcns.append(None) # 或者一個 nn.Identity() 如果後續邏輯依賴於調用它

                if self.spatialattn_bool:
                    self.sans.append(
                        SpatialAttentionLayer(num_nodes, hidden_dim, in_len=1)
                    )
                else:
                    self.sans.append(None)
                
                self.gcn_residual_linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.gcn_bns.append(nn.LayerNorm(hidden_dim))
        else: # 如果 GCN 和 SA 都關閉
            self.gcns = None
            self.sans = None
            self.gcn_residual_linears = None
            self.gcn_bns = None
            self.supports = None # 不需要 supports
            self.nodevec = None  # 不需要 nodevec


    def forward(self, h_temporal, global_step=None, current_supports=None):
        # h_temporal 輸入形狀: [batch_size, num_stocks, in_feature_dim]
        if global_step is not None:
            wandb.log({
                "HCAR_Actor/3B_Relational/0_Input_h_temporal_Mean": h_temporal.mean().item(),
                "HCAR_Actor/3B_Relational/0_Input_h_temporal_Std": h_temporal.std().item(),
            }, step=global_step)
        
        x_after_start_linear = self.start_linear(h_temporal)
        
        if global_step is not None and global_step % 100 == 0: # 定期記錄權重和偏置
            wandb.log({
                "HCAR_Actor/3B_Relational/StartLinear_Weight_AbsMean": self.start_linear.weight.data.abs().mean().item(),
                "HCAR_Actor/3B_Relational/StartLinear_Weight_Std": self.start_linear.weight.data.std().item(),
                "HCAR_Actor/3B_Relational/StartLinear_Bias_AbsMean": self.start_linear.bias.data.abs().mean().item() if self.start_linear.bias is not None else 0,
            }, step=global_step)
            # 記錄梯度的部分應在 backward() 之後，優化器 step() 之前，通常在 Agent 中完成

        x = self.ln_start_relational(x_after_start_linear) # x 的形狀是 [B, N, C_hidden]
        
        if global_step is not None:
            wandb.log({ 
                "HCAR_Actor/3B_Relational/1b_After_StartLinearLN_Mean": x.mean().item(),
                "HCAR_Actor/3B_Relational/1b_After_StartLinearLN_Std": x.std().item(),
            }, step=global_step)
            
        # 只有當 GCN 或 SA 啟用時才執行 GCN/SA 堆疊
        if self.gcn_bool or self.spatialattn_bool:
            final_supports_for_gcn = self.supports
            if self.gcn_bool and self.addaptiveadj and hasattr(self, 'nodevec') and self.nodevec is not None:
                adp_matrix = torch.softmax(torch.relu(torch.mm(self.nodevec, self.nodevec.t())), dim=0)
                if final_supports_for_gcn is None:
                    final_supports_for_gcn = [adp_matrix]
                else:
                    # 確保 self.supports 是可修改的列表副本或進行拼接
                    current_static_supports = list(self.supports) if self.supports is not None else []
                    final_supports_for_gcn = current_static_supports + [adp_matrix]
            
            if current_supports is not None: # 允許外部傳入動態圖覆蓋
                final_supports_for_gcn = current_supports

            for i in range(self.num_gcn_layers):
                residual = self.gcn_residual_linears[i](x) 
                
                x_permuted_for_block = x.permute(0, 2, 1).unsqueeze(-1) # [B, C_hidden, N, 1]
                
                # GCN 處理
                if self.gcn_bool and final_supports_for_gcn is not None and self.gcns and self.gcns[i] is not None:
                    x_after_gcn = self.gcns[i](x_permuted_for_block, final_supports_for_gcn)
                else:
                    x_after_gcn = x_permuted_for_block # 透傳

                # Spatial Attention 處理
                if self.spatialattn_bool and self.sans and self.sans[i] is not None:
                    attn_weights = self.sans[i](x_after_gcn) # SA 作用於 GCN 輸出
                    if global_step is not None:
                        wandb.log({
                            f"HCAR_Actor/3B_Relational/2_Layer{i}_AttnWeights_Mean": attn_weights.mean().item(),
                            f"HCAR_Actor/3B_Relational/2_Layer{i}_AttnWeights_Std": attn_weights.std().item(),
                        }, step=global_step)
                    x_after_sa = torch.einsum('bnm,bfml->bfnl', (attn_weights, x_after_gcn))
                else:
                    x_after_sa = x_after_gcn # 如果不用 SA，則直接使用 GCN 的輸出

                x = x_after_sa.squeeze(-1).permute(0, 2, 1) # [B, N, C_hidden]
                x = x + residual 
                x = self.gcn_bns[i](x) 
                
                if global_step is not None:
                    wandb.log({
                        f"HCAR_Actor/3B_Relational/3_Layer{i}_BlockOutput_Mean": x.mean().item(),
                        f"HCAR_Actor/3B_Relational/3_Layer{i}_BlockOutput_Std": x.std().item(),
                    }, step=global_step)
        # 如果 GCN 和 SA 都為 False，x 就是 self.ln_start_relational(x_after_start_linear) 的結果
        
        if global_step is not None:
            wandb.log({
                "HCAR_Actor/3B_Relational/4_Final_h_final_Mean": x.mean().item(),
                "HCAR_Actor/3B_Relational/4_Final_h_final_Std": x.std().item(),
            }, step=global_step)
            
        return x


class AssetScoringHead(nn.Module):
    def __init__(self, input_dim, hidden_layers_dims=None, output_dim=1, dropout=0.3):
        """
        Args:
            input_dim (int): 輸入特徵維度 (即 RelationalContextIntegrator 輸出的 hidden_dim_3B)
            hidden_layers_dims (list of int, optional): MLP隱藏層的維度列表。
                                                       如果為 None 或空列表，則只有一個輸出層。
            output_dim (int): 每個股票輸出的評分維度，通常為 1。
            dropout (float): Dropout 比率。
        """
        super(AssetScoringHead, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        if hidden_layers_dims:
            for h_dim in hidden_layers_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.LayerNorm(h_dim)) # <--- 在激活之前或之後加入
                layers.append(nn.ReLU()) # 或者其他激活函數如 GELU, SiLU
                layers.append(nn.Dropout(dropout))
                current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.mlp_head = nn.Sequential(*layers)
        for layer in self.mlp_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu') # if using ReLU
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                    
    def forward(self, h_final_input, global_step=None):
        # h_final_input 期望形狀: [batch_size, num_stocks, input_dim]
        # print(f"AssetScoringHead - Input h_final_input shape: {h_final_input.shape}")
        if global_step is not None:
            wandb.log({
                "HCAR_Actor/3C_Scoring/0_Input_h_final_Mean": h_final_input.mean().item(),
                "HCAR_Actor/3C_Scoring/0_Input_h_final_Std": h_final_input.std().item(),
            }, step=global_step)
        batch_size, num_stocks, feature_dim = h_final_input.shape
        
        # 為了讓 MLP 共享參數處理每個股票的特徵，我們先將 batch 和 num_stocks 維度合併
        x_reshaped = h_final_input.reshape(batch_size * num_stocks, feature_dim)
        # print(f"AssetScoringHead - Reshaped input for MLP (x_reshaped shape): {x_reshaped.shape}")
        
        # MLP 處理
        scores_flat = self.mlp_head(x_reshaped) # 輸出 [batch_size * num_stocks, output_dim]
        # print(f"AssetScoringHead - Output from MLP (scores_flat shape): {scores_flat.shape}")
        
        # 將輸出 reshape 回 [batch_size, num_stocks, output_dim]
        # 因為 output_dim 通常為 1，我們可以 squeeze(-1) 得到 [batch_size, num_stocks]
        output_scores = scores_flat.view(batch_size, num_stocks, -1).squeeze(-1)
        # print(f"AssetScoringHead - Final output_scores shape: {output_scores.shape}")
        if global_step is not None:
            wandb.log({
                "HCAR_Actor/3C_Scoring/1_Output_StockLogits_Mean": output_scores.mean().item(),
                "HCAR_Actor/3C_Scoring/1_Output_StockLogits_Std": output_scores.std().item(),
            }, step=global_step)
        return output_scores # 形狀 [batch_size, num_stocks]

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