# util of HCAR
import torch
import torch.nn as nn
import torch.nn.functional as F
from titans_pytorch.memory_models import MemoryMLP
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import math
import numpy as np
from .builder import NETS
from .custom import Net
from trademaster.utils import build_conv2d
from torch import Tensor
import wandb

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


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.1):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.output_dim = output_dim

        self.query = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim)) # [1, H, 1, Dh_q] (可學習的市場查詢)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 最終的線性層將池化後的結果映射到 output_dim
        self.memory_mlp_intermediate = MemoryMLP(dim=input_dim, depth=1, expansion_factor=1) 

        # Add a linear layer to project to the desired output_dim if it's different
        if input_dim != output_dim:
            self.final_projection = nn.Linear(input_dim, output_dim)
        else:
            self.final_projection = nn.Identity()

    def forward(self, stock_features_input):
        # stock_features_input: [B, N, D_input] (例如 h_temporal)
        B, N, D_input = stock_features_input.shape

        keys = self.key_proj(stock_features_input).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)      # [B, H, N, Dh_kv]
        values = self.value_proj(stock_features_input).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, Dh_kv]
        
        # query: [1, H, 1, Dh_q] -> 廣播到 [B, H, 1, Dh_q]
        expanded_query = self.query.expand(B, -1, -1, -1)

        # Scaled Dot-Product Attention
        # (B, H, 1, Dh_q) @ (B, H, Dh_kv, N) -> (B, H, 1, N)  (假設 Dh_q == Dh_kv)
        attn_scores = torch.matmul(expanded_query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1) # 在 N 維度上 Softmax
        attn_probs = self.dropout(attn_probs)

        # (B, H, 1, N) @ (B, H, N, Dh_v) -> (B, H, 1, Dh_v)
        context_vector = torch.matmul(attn_probs, values) # [B, H, 1, Dh_v]
        
        # 拼接多頭或重塑
        context_vector = context_vector.permute(0, 2, 1, 3).reshape(B, 1, D_input) # [B, 1, D_input]
        
        s_market_pooled = context_vector.squeeze(1) # [B, input_dim]
        intermediate_output = self.memory_mlp_intermediate(s_market_pooled) # [B, input_dim]
        s_market_final = self.final_projection(intermediate_output) # [B, output_dim]
        return s_market_final