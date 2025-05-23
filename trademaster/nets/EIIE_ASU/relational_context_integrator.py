import torch
import torch.nn as nn
import math
import numpy as np

# --- 從您 ASU.py 複製必要的底層組件 ---
class nconv(nn.Module):
    # ... (保持不變)
    def __init__(self):
        super(nconv, self).__init__()
    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    # ... (保持不變)
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)

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
    def __init__(self, num_nodes, in_features, in_len): # in_len 在這裡固定為 1
        super().__init__()
        assert in_len == 1, "SpatialAttentionLayer in RelationalContextIntegrator expects in_len=1"
        self.in_len = in_len
        self.W1 = nn.Linear(in_len, 1, bias=False) 
        self.W2 = nn.Linear(in_features, in_len, bias=False) # in_len is 1, so W2 maps F to 1
        self.W3 = nn.Linear(in_features, 1, bias=False)
        self.V = nn.Linear(num_nodes, num_nodes)

        self.bn_w1 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w3 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w2 = nn.BatchNorm1d(num_features=num_nodes) # This will receive [B, N, 1]

    def forward(self, inputs_relational):
        # inputs_relational 期望形狀: [B, F, N, 1] (F=in_features, N=num_nodes)
        # 這是已經經過 permute 以適應 Conv2d 習慣的形狀
        B, F, N, T_pseudo = inputs_relational.shape 
        assert T_pseudo == self.in_len, f"Input pseudo time T={T_pseudo} is not in_len={self.in_len}"
        
        # SpatialAttentionLayer 原本的輸入 x 是 inputs[..., -self.in_len:]
        # 因為我們的 T_pseudo 就是 in_len (為1), 所以直接用 inputs_relational
        x_for_sa = inputs_relational 

        # Part 1 path
        part1_initial_permute = x_for_sa.permute(0, 2, 1, 3)  # [B, N, F_actual, 1]
        p1_squeezed_after_W1 = self.W1(part1_initial_permute).squeeze(-1) # [B, N, F_actual]
        p1_after_bn1 = self.bn_w1(p1_squeezed_after_W1) # bn_w1(num_features=N) 作用於 [B, N, F_actual] -> 正確

        p1_after_W2 = self.W2(p1_after_bn1) # W2 is Linear(F_actual, 1) -> [B, N, 1]
        # bn_w2(num_features=N) 作用於 [B, N] (squeeze後) -> 正確
        part1_final_for_bmm = self.bn_w2(p1_after_W2.squeeze(-1)).unsqueeze(-1) # [B, N, 1]

        # Part 2 path
        part2_initial_permute = x_for_sa.permute(0, 2, 3, 1)  # [B, N, 1, F_actual]
        # W3 is Linear(F_actual, 1)
        # part2_initial_permute.squeeze(-2) -> [B, N, F_actual]
        p2_squeezed_after_W3 = self.W3(part2_initial_permute.squeeze(-2)) # [B, N, 1]
        # bn_w3(num_features=N) 作用於 [B, N] (squeeze後) -> 正確
        p2_after_bn3_squeezed = self.bn_w3(p2_squeezed_after_W3.squeeze(-1)) # [B, N]
        part2_final_for_bmm = p2_after_bn3_squeezed.unsqueeze(-1).permute(0, 2, 1) # [B, N, 1] -> [B, 1, N]

        S_raw = torch.bmm(part1_final_for_bmm, part2_final_for_bmm) # [B, N, 1] @ [B, 1, N] -> [B, N, N]
        S = torch.softmax(self.V(torch.relu(S_raw)), dim=-1)
        return S

# --- 新的子模塊 3.B ---
class RelationalContextIntegrator(nn.Module):
    def __init__(self, num_nodes, in_feature_dim, hidden_dim, num_gcn_layers=2, # 和SAGCN的layers對應
                 dropout=0.3, supports=None, 
                 gcn_bool=True, spatialattn_bool=True, # 控制是否使用GCN和SA
                 addaptiveadj=True, aptinit=None):
        super(RelationalContextIntegrator, self).__init__()
        self.num_gcn_layers = num_gcn_layers # 改名以更清晰
        self.gcn_bool = gcn_bool
        self.spatialattn_bool = spatialattn_bool
        self.addaptiveadj = addaptiveadj
        
        self.gcns = nn.ModuleList()
        self.sans = nn.ModuleList()
        # SAGCN 中的 residual_convs 和 bns 是針對 TCN 之後的輸出的，
        # 這裡我們需要類似的機制，但輸入維度不同。
        # 殘差連接的 conv 可能需要變成 Linear 層或 1x1 Conv1d
        self.gcn_residual_linears = nn.ModuleList() # 用於 GCN/SA 塊的殘差
        self.gcn_bns = nn.ModuleList() # 用於 GCN/SA 塊的 BatchNorm

        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and addaptiveadj:
            if aptinit is None:
                if supports is None: # 應該不會發生，因為通常會有一個如因果圖的 support
                    self.supports = []
                self.nodevec = nn.Parameter(torch.randn(num_nodes, 1), requires_grad=True)
                self.supports_len += 1
            else:
                raise NotImplementedError
        
        # 輸入特徵維度是 in_feature_dim，GCN/SA 內部處理使用 hidden_dim
        # 所以可能需要一個初始的線性映射，或者直接讓 GCN/SA 的輸入是 in_feature_dim
        # 為了與 SAGCN 的 start_conv 對應，我們也加一個
        # 但 SAGCN 的 start_conv 是 Conv2d，這裡輸入是 [B, N, F]，所以用 Linear
        self.start_linear = nn.Linear(in_feature_dim, hidden_dim)
        self.bn_start_relational = nn.BatchNorm1d(num_features=num_nodes) # 對 [B, N, C] 的 N 維度 BN (或者 LayerNorm(C))
                                                                     # 或者 BatchNorm1d(hidden_dim) 然後 permute
        # 讓我們用 LayerNorm(hidden_dim) 會更簡單和常見
        self.ln_start_relational = nn.LayerNorm(hidden_dim)


        for l in range(self.num_gcn_layers):
            if self.gcn_bool:
                self.gcns.append(
                    GraphConvNet(hidden_dim, hidden_dim, dropout, support_len=self.supports_len)
                )
            if self.spatialattn_bool:
                # SpatialAttentionLayer 的 in_features 是 hidden_dim, in_len 固定為 1
                self.sans.append(
                    SpatialAttentionLayer(num_nodes, hidden_dim, in_len=1)
                )
            
            self.gcn_residual_linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.gcn_bns.append(nn.LayerNorm(hidden_dim)) # LayerNorm 更適合變長序列或圖節點特徵

    def forward(self, h_temporal, current_supports=None):
        # h_temporal 輸入形狀: [batch_size, num_stocks, in_feature_dim]
        print(f"RelationalContextIntegrator - Input h_temporal shape: {h_temporal.shape}")

        x = self.start_linear(h_temporal) # [B, N, in_feature_dim] -> [B, N, hidden_dim]
        x = self.ln_start_relational(x)   # LayerNorm 在 hidden_dim 維度上
        print(f"RelationalContextIntegrator - After start_linear & ln_start (x shape): {x.shape}")

        # 準備 GCN 和 SA 所需的 supports
        final_supports = self.supports # 使用初始化時傳入的基礎 supports
        if self.gcn_bool and self.addaptiveadj and hasattr(self, 'nodevec'):
            # 注意：pc_causal_relation.npy 是固定的，如果 self.supports 包含了它，
            # 這裡會將學習到的自適應矩陣疊加進去。
            adp_matrix = torch.softmax(torch.relu(torch.mm(self.nodevec, self.nodevec.t())), dim=0)
            if final_supports is None: # 以防萬一
                final_supports = [adp_matrix]
            else:
                final_supports = final_supports + [adp_matrix]
        
        # 如果外部傳入了當前時間步的動態圖，則使用它
        if current_supports is not None:
            final_supports = current_supports


        for i in range(self.num_gcn_layers):
            residual = self.gcn_residual_linears[i](x) # [B, N, hidden_dim]
            print(f"RelationalContextIntegrator - Layer {i} - Residual shape: {residual.shape}")
            
            # 為了送入 GraphConvNet 和 SpatialAttentionLayer，需要調整維度
            # 它們期望 [B, C, N, T_pseudo=1]
            x_permuted = x.permute(0, 2, 1).unsqueeze(-1) # [B, N, C] -> [B, C, N] -> [B, C, N, 1]
            print(f"RelationalContextIntegrator - Layer {i} - x_permuted for GCN/SA (shape): {x_permuted.shape}")

            if self.gcn_bool and final_supports is not None:
                if not self.gcns: # 檢查 gcns 列表是否為空
                    print("Warning: GCN is enabled but gcns list is empty. Skipping GCN.")
                else:
                    x_gcn_out_permuted = self.gcns[i](x_permuted, final_supports) # 輸出 [B, C, N, 1]
                    print(f"RelationalContextIntegrator - Layer {i} - After gcns[{i}] (shape): {x_gcn_out_permuted.shape}")
            else:
                x_gcn_out_permuted = x_permuted # 如果不用 GCN，直接透傳

            if self.spatialattn_bool:
                if not self.sans: # 檢查 sans 列表是否為空
                    print("Warning: Spatial Attention is enabled but sans list is empty. Skipping SA.")
                else:
                    # SpatialAttentionLayer 的輸入也是 [B, C, N, 1]
                    # 它的輸出是注意力權重 S，形狀 [B, N, N]
                    attn_weights = self.sans[i](x_gcn_out_permuted) # 傳入的是 GCN 處理後的數據
                    print(f"RelationalContextIntegrator - Layer {i} - attn_weights shape: {attn_weights.shape}")
                    
                    # 應用注意力權重
                    # x_gcn_out_permuted 是 [B, C, N, 1]，需要 squeeze 和 permute
                    # x_to_attend = x_gcn_out_permuted.squeeze(-1).permute(0, 2, 1) # [B, N, C]
                    # x_attended = torch.bmm(attn_weights, x_to_attend) # [B, N, N] @ [B, N, C] -> [B, N, C]
                    # x = x_attended # 更新 x
                    
                    # 參考 SAGCN 的做法: x = torch.einsum('bnm, bfml->bfnl', (attn_weights, x))
                    # attn_weights: [B, N_out, N_in] (這裡 N_out=N_in=N)
                    # x (GCN輸出): [B, F, N_in, L=1] (F=hidden_dim)
                    # 輸出應為: [B, F, N_out, L=1]
                    x_after_gcn_for_sa = x_gcn_out_permuted # 使用 GCN 的輸出
                    x_permuted = torch.einsum('bnm,bfml->bfnl', (attn_weights, x_after_gcn_for_sa))
                    print(f"RelationalContextIntegrator - Layer {i} - After SA (x_permuted shape): {x_permuted.shape}")

            else:
                # 如果不用 Spatial Attention，則 x_permuted 保持為 x_gcn_out_permuted
                pass 
                # x_permuted = x_gcn_out_permuted # 這句話是需要的，否則 x_permuted 可能是上一輪的


            # 將維度還原回 [B, N, C] 以便進行殘差連接和下一輪
            x = x_permuted.squeeze(-1).permute(0, 2, 1) # [B, C, N, 1] -> [B, C, N] -> [B, N, C]
            print(f"RelationalContextIntegrator - Layer {i} - x reshaped after GCN/SA: {x.shape}")

            x = x + residual # 殘差連接
            x = self.gcn_bns[i](x) # LayerNorm 在 C 維度上
            print(f"RelationalContextIntegrator - Layer {i} - After residual and bns[{i}] (x shape): {x.shape}")
        
        # 最終輸出 h_final
        print(f"RelationalContextIntegrator - Final output h_final shape: {x.shape}")
        return x # [batch_size, num_stocks, hidden_dim]

if __name__ == '__main__':
    # --- 測試參數 ---
    batch_size_test = 4
    num_stocks_test = 49
    # window_len_test = 13 # 子模塊 3.A 使用
    # num_features_test = 11 # 子模塊 3.A 使用
    
    # 子模塊 3.A 的輸出維度，將作為子模塊 3.B 的輸入特徵維度
    feature_dim_from_3A = 64 
    # 子模塊 3.B 內部的隱藏層維度
    hidden_dim_3B = 64 # 可以與 feature_dim_from_3A 相同或不同

    num_gcn_layers_test = 2
    dropout_test = 0.3

    # --- 準備 supports (因果圖或行業圖) ---
    # 假設您有一個 pc_causal_relation.npy
    causal_graph_path_test = "/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/ASU_data_corrected/pc_causal_relation.npy"
    try:
        relation_matrix_np = np.load(causal_graph_path_test)
        assert relation_matrix_np.shape == (num_stocks_test, num_stocks_test)
        supports_list_test = [torch.from_numpy(relation_matrix_np).float()]
        print(f"Successfully loaded supports from: {causal_graph_path_test}")
    except Exception as e:
        print(f"Error loading supports: {e}. Using identity matrix as fallback.")
        supports_list_test = [torch.eye(num_stocks_test).float()]


    print("\n--- Initializing RelationalContextIntegrator ---")
    relational_integrator = RelationalContextIntegrator(
        num_nodes=num_stocks_test,
        in_feature_dim=feature_dim_from_3A,
        hidden_dim=hidden_dim_3B,
        num_gcn_layers=num_gcn_layers_test,
        dropout=dropout_test,
        supports=supports_list_test, # 傳入基礎的 supports
        gcn_bool=True,
        spatialattn_bool=True,
        addaptiveadj=True # 假設您想使用自適應鄰接矩陣
    )
    relational_integrator.eval()

    print("\n--- Preparing Fake Input h_temporal (Output from Sub-module 3.A) ---")
    # h_temporal 的形狀是 [batch_size, num_stocks, feature_dim_from_3A]
    fake_h_temporal = torch.randn(batch_size_test, num_stocks_test, feature_dim_from_3A)
    print(f"Shape of fake_h_temporal: {fake_h_temporal.shape}")

    print("\n--- Running Forward Pass through RelationalContextIntegrator ---")
    try:
        with torch.no_grad():
            # h_final 的形狀將是 [batch_size, num_stocks, hidden_dim_3B]
            h_final = relational_integrator(fake_h_temporal)
            print("--- Forward Pass Successful ---")
            print(f"Output h_final shape: {h_final.shape}")
            assert h_final.shape == (batch_size_test, num_stocks_test, hidden_dim_3B), "Output shape mismatch!"

            print(f"Example h_final (first sample, first stock): \n{h_final[0, 0, :10]}")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()