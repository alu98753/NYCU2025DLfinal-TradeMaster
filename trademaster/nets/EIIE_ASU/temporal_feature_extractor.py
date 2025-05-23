import torch
import torch.nn as nn
import math # 確保導入

# --- 先複製您 ASU.py 中的 nconv, linear, GraphConvNet, SpatialAttentionLayer ---
# (這些在純 TCN 提取器中可能不會全部用到，但為了完整性，先放在這裡，下面會精簡)
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
# --- GraphConvNet 和 SpatialAttentionLayer 在純 TCN 模塊中不會用到，可以暫時不複製過來 ---

class TemporalAttentionPool(nn.Module):
    def __init__(self, in_len, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_len, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, N, T]
        B, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # → [B, N, T, C]
        x = x.reshape(B * N, T, C)  # → [B*N, T, C]

        attn_out, _ = self.attn(x, x, x)  # Self-Attention over time
        out = x + self.dropout(attn_out)  # Residual
        out = self.norm(out)             # LayerNorm
        out = out.mean(dim=1)            # → [B*N, C]
        out = out.view(B, N, C)          # → [B, N, C]
        return out

# --- 這是我們為子模塊 3.A 設計的新類 ---
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, in_features, hidden_dim, tcn_kernel_size=2, num_tcn_layers=4, dropout=0.3):
        super(TemporalFeatureExtractor, self).__init__()
        self.layers = num_tcn_layers
        self.tcns = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.temporal_attention_pool = TemporalAttentionPool(in_len=hidden_dim) # 這裡 in_len 應為 feature_dim (hidden_dim)

        # 初始的特徵維度映射
        self.start_conv = nn.Conv2d(in_features, hidden_dim, kernel_size=(1, 1))
        self.bn_start = nn.BatchNorm2d(hidden_dim)

        self.receptive_field = 1 # 初始化感受野
        additional_scope_total = 0 # 用於計算TCN輸出長度調整

        dilation = 1
        for l in range(self.layers):
            # 每個TCN層的定義，與SAGCN中類似，但kernel_size[0]固定為1
            # nn.Conv2d 期望 (in_channels, out_channels, kernel_size, dilation)
            # 這裡的 kernel_size 是 (height, width)
            # 我們希望在時間維度 (width) 上卷積，股票維度 (height) 上獨立
            self.tcns.append(nn.Sequential(
                nn.Conv2d(in_channels=hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=(1, tcn_kernel_size), # 高度為1，寬度為 tcn_kernel_size
                          dilation=(1, dilation)),        # 只在寬度（時間）維度膨脹
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm2d(hidden_dim)
            ))
            self.residual_convs.append(
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1))
            )
            self.bns.append(nn.BatchNorm2d(hidden_dim))

            # 計算感受野和輸出長度變化
            # 每次卷積輸出長度減少 dilation * (kernel_size - 1)
            # 這裡的 kernel_size 是 tcn_kernel_size
            current_additional_scope = dilation * (tcn_kernel_size - 1)
            self.receptive_field += current_additional_scope
            additional_scope_total += current_additional_scope
            dilation *= 2
        
        print(f"TemporalFeatureExtractor initialized. Receptive field: {self.receptive_field}")
        # 注意：這裡的 receptive_field 計算的是理論最大值。
        # 實際 forward 時，如果輸入長度不足，會有 padding。
        # 如果輸入長度固定，則輸出長度也固定，為 input_len - total_output_len_reduction

    def forward(self, X_input):
        # X_input 期望形狀: [batch_size, num_stocks, window_len, num_features]
        print(f"TemporalFeatureExtractor - Input X_input shape: {X_input.shape}")

        # 轉換維度以適應 Conv2d: [batch_size, num_features, num_stocks, window_len]
        x = X_input.permute(0, 3, 1, 2)
        print(f"TemporalFeatureExtractor - After permute for Conv2d (x shape): {x.shape}")

        # 初始 padding 以確保即使輸入窗口小於感受野也能處理
        # 或者讓用戶保證 window_len >= receptive_field
        # 為了簡化，這裡我們先假設 window_len >= receptive_field
        # DeepTrader ASU 中的 padding 邏輯:
        # if x.shape[3] < self.receptive_field:
        #     x = nn.functional.pad(x, (self.receptive_field - x.shape[3], 0, 0, 0))
        # print(f"TemporalFeatureExtractor - After padding (if any) (x shape): {x.shape}")


        x = self.start_conv(x)
        x = self.bn_start(x)
        print(f"TemporalFeatureExtractor - After start_conv & bn_start (x shape): {x.shape}")

        for i in range(self.layers):
            residual = self.residual_convs[i](x)
            print(f"TemporalFeatureExtractor - Layer {i} - Residual shape: {residual.shape}")
            
            x_before_tcn = x
            x = self.tcns[i](x)
            print(f"TemporalFeatureExtractor - Layer {i} - After tcns[{i}] (x shape): {x.shape}")

            # 殘差連接需要對齊維度，因為TCN會縮短序列長度
            # residual 的長度是 x_before_tcn 的長度
            # x 的長度比 x_before_tcn 短
            # 所以 residual 需要從尾部截取以匹配 x 的長度
            # residual_cropped = residual[..., -x.size(3):] # 截取時間維度
            # 或者更安全的做法，計算清楚 x 的輸出長度
            # x = x + residual_cropped

            # 根據SAGCN的實現，殘差連接是這樣做的：
            # x = x + residual[:, :, :, -x.shape[3]:]
            # 這裡的 x.shape[3] 是 TCN 輸出後的長度
            # residual 的時間長度與 TCN 輸入時的 x 相同
            # 所以 residual[:, :, :, -x.shape[3]:] 會從 residual 的尾部取 x.shape[3] 長度
            time_dim_x = x.shape[3]
            time_dim_residual = residual.shape[3]
            if time_dim_x < time_dim_residual:
                 # 這是常見情況，TCN 縮短了序列
                residual_to_add = residual[:, :, :, time_dim_residual - time_dim_x:]
            elif time_dim_x == time_dim_residual:
                residual_to_add = residual
            else: # 不太可能發生，除非TCN帶padding且kernel=1
                raise ValueError("TCN output is longer than its input, check TCN padding/kernel.")
            
            print(f"TemporalFeatureExtractor - Layer {i} - Residual to add shape: {residual_to_add.shape}")
            x = x + residual_to_add
            x = self.bns[i](x)
            print(f"TemporalFeatureExtractor - Layer {i} - After residual and bns[{i}] (x shape): {x.shape}")

        # 經過所有TCN層後，x 的形狀是 [batch_size, hidden_dim, num_stocks, final_window_len]
        # 我們需要將其轉換為 [batch_size, num_stocks, final_window_len, hidden_dim]
        # 或者，如果子模塊 3.A 的目標是為每個股票輸出一個總結性的特徵向量，
        # 那麼我們需要一個時間池化步驟。
        # 例如，取最後一個時間步的特徵，或者對時間維度進行平均池化/最大池化。
        
        # 方案A：輸出每個時間步的特徵，讓後續模塊（如GCN）處理序列
        # output = x.permute(0, 2, 3, 1) 
        # print(f"TemporalFeatureExtractor - Final output (permuted, shape): {output.shape}")
        # return output # [batch_size, num_stocks, final_window_len, hidden_dim]

        # 方案B：輸出每個股票的單一總結特徵向量 (例如，取最後時間步)
        # 這更接近原始IIE為每個股票輸出一個“評分”或“狀態”
        # x 的形狀是 [B, C_hidden, N_stocks, T_final_len]
        # 我們取 T_final_len 維度的最後一個時間步的特徵
        # summarized_features = x[:, :, :, -1] # 取最後一個時間步, [B, C_hidden, N_stocks]
        # 轉換為 [B, N_stocks, C_hidden]
        output = self.temporal_attention_pool(x) # TemporalAttentionPool 內部會處理維度轉換
                                                # 期望輸出 [B, N_stocks, C_hidden]
        print(f"TemporalFeatureExtractor - Final output (TemporalAttentionPool, shape): {output.shape}")
        return output


if __name__ == '__main__':
    # --- 測試參數 ---
    batch_size_test = 4
    num_stocks_test = 49
    window_len_test = 13  # 確保這個長度 >= TemporalFeatureExtractor 的 receptive_field
                          # 或者在 TemporalFeatureExtractor 中加入 padding 邏輯
    num_features_test = 11 # 與您的 tech_indicator_list 長度一致

    hidden_dim_test = 64
    tcn_kernel_size_test = 2
    num_tcn_layers_test = 3 # 可以調整層數
    dropout_test = 0.3

    print("--- Initializing TemporalFeatureExtractor ---")
    temporal_extractor = TemporalFeatureExtractor(
        in_features=num_features_test,
        hidden_dim=hidden_dim_test,
        tcn_kernel_size=tcn_kernel_size_test,
        num_tcn_layers=num_tcn_layers_test,
        dropout=dropout_test
    )
    temporal_extractor.eval() # 設置為評估模式

    print("\n--- Preparing Fake Input Data ---")
    # 創建符合 Agent 傳給 Actor 的數據形狀
    # Agent 的 state['obs'] 傳給 Actor 時是 [batch_size, num_stocks, window_len, num_features]
    fake_input_data = torch.randn(batch_size_test, num_stocks_test, window_len_test, num_features_test)
    print(f"Shape of fake_input_data: {fake_input_data.shape}")

    print("\n--- Running Forward Pass through TemporalFeatureExtractor ---")
    try:
        with torch.no_grad():
            # h_temporal 的形狀將是 [batch_size, num_stocks, hidden_dim_test]
            # 如果 TemporalFeatureExtractor 內部選擇了時間池化/取最後時間步
            h_temporal = temporal_extractor(fake_input_data) 
            print("--- Forward Pass Successful ---")
            print(f"Output h_temporal shape: {h_temporal.shape}")
            assert h_temporal.shape == (batch_size_test, num_stocks_test, hidden_dim_test), "Output shape mismatch!"

            # 打印一些輸出的例子
            print(f"Example h_temporal (first sample, first stock): \n{h_temporal[0, 0, :10]}")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()