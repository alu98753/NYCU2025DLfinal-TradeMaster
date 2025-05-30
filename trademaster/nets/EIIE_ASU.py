import math

import torch
import torch.nn as nn


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class GraphConvNet(nn.Module):
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
    def __init__(self, num_nodes, in_features, in_len):
        super().__init__()
        self.in_len = in_len
        """ self.W1
        1. in_len 為動態 代表 TCN能觀察到的時間感受野（receptive_field - a_s_records[i]）
        2. 對每支股票的時間序列做線性加權
        """
        self.W1 = nn.Linear(in_len, 1, bias=False)
        self.W2 = nn.Linear(in_features, in_len, bias=False)
        self.W3 = nn.Linear(in_features, 1, bias=False)
        self.V = nn.Linear(num_nodes, num_nodes)

        self.bn_w1 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w3 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w2 = nn.BatchNorm1d(num_features=num_nodes)

    def forward(self, inputs):
        # inputs: [B, F, N, T]
        B, F, N, T = inputs.shape
        assert T >= self.in_len, f"Input T={T} is less than in_len={self.in_len}"
        x = inputs[..., -self.in_len :]  # 取後面 in_len 長度

        # 保留 in_len 長度的序列 然後用不同方向做 W1, W2, W3 的學習式壓縮：

        part1 = x.permute(0, 2, 1, 3)  # [B, N, F, in_len]
        part2 = x.permute(0, 2, 3, 1)  # [B, N, in_len, F]

        part1 = self.bn_w1(self.W1(part1).squeeze(-1))
        part1 = self.bn_w2(self.W2(part1))
        part2 = self.bn_w3(self.W3(part2).squeeze(-1)).permute(0, 2, 1)

        """
        1. 用 batch matrix multiply + learnable transformation 保留了原始注意力架構
        2. 對每支股票的時間序列進行加權投影，形成空間注意力圖
        """
        S = torch.softmax(self.V(torch.relu(torch.bmm(part1, part2))), dim=-1)
        return S


class SAGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_features,
        hidden_dim,
        window_len,
        dropout=0.3,
        kernel_size=2,
        layers=4,
        supports=None,
        spatial_bool=True,
        addaptiveadj=True,
        aptinit=None,
    ):

        super(SAGCN, self).__init__()
        self.dropout = dropout
        self.layers = layers
        if spatial_bool:
            self.gcn_bool = True
            self.spatialattn_bool = True
        else:
            self.gcn_bool = False
            self.spatialattn_bool = False
        self.addaptiveadj = addaptiveadj

        self.tcns = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.sans = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.supports = supports

        self.start_conv = nn.Conv2d(in_features, hidden_dim, kernel_size=(1, 1))

        self.bn_start = nn.BatchNorm2d(hidden_dim)

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and addaptiveadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec = nn.Parameter(
                    torch.randn(num_nodes, 1), requires_grad=True
                )
                self.supports_len += 1

            else:
                raise NotImplementedError

        additional_scope = kernel_size - 1
        a_s_records = []
        dilation = 1
        for l in range(layers):
            tcn_sequence = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(1, kernel_size),
                    dilation=dilation,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm2d(hidden_dim),
            )

            self.tcns.append(tcn_sequence)

            self.residual_convs.append(
                nn.Conv2d(
                    in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1)
                )
            )

            self.bns.append(nn.BatchNorm2d(hidden_dim))

            if self.gcn_bool:
                self.gcns.append(
                    GraphConvNet(
                        hidden_dim, hidden_dim, dropout, support_len=self.supports_len
                    )
                )

            dilation *= 2
            a_s_records.append(additional_scope)
            receptive_field += additional_scope
            additional_scope *= 2

        self.receptive_field = receptive_field
        if self.spatialattn_bool:
            for i in range(layers):
                self.sans.append(
                    SpatialAttentionLayer(
                        num_nodes, hidden_dim, receptive_field - a_s_records[i]
                    )
                )
                receptive_field -= a_s_records[i]

        # ----added ---- #
        self.time_compressor = TimeCompressor(in_len=10)

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)  # [batch, feature, stocks, length]
        print("X.shape", X.shape)
        in_len = X.shape[3]
        if in_len < self.receptive_field:
            x = nn.functional.pad(X, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = X
        assert not torch.isnan(x).any()

        x = self.bn_start(self.start_conv(x))
        new_supports = None
        if self.gcn_bool and self.addaptiveadj and self.supports is not None:
            adp_matrix = torch.softmax(
                torch.relu(torch.mm(self.nodevec, self.nodevec.t())), dim=0
            )
            new_supports = self.supports + [adp_matrix]

        for i in range(self.layers):
            residual = self.residual_convs[i](x)
            x = self.tcns[i](x)
            if self.gcn_bool and self.supports is not None:
                if self.addaptiveadj:
                    x = self.gcns[i](x, new_supports)
                else:
                    x = self.gcns[i](x, self.supports)

            if self.spatialattn_bool:
                attn_weights = self.sans[i](x)
                x = torch.einsum("bnm, bfml->bfnl", (attn_weights, x))

            x = x + residual[:, :, :, -x.shape[3] :]

            x = self.bns[i](x)
        print("x.shape", x.shape)

        # ----added ---- #
        x = self.time_compressor(x)  # [batch, hidden_dim, num_nodes]
        print("x.shape", x.shape)
        return x.squeeze(-1).permute(0, 2, 1)  # (batch, num_nodes, hidden_dim)


class LiteTCN(nn.Module):
    def __init__(
        self, in_features, hidden_size, num_layers, kernel_size=2, dropout=0.4
    ):
        super(LiteTCN, self).__init__()
        self.num_layers = num_layers
        self.tcns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_features, hidden_size, kernel_size=1)
        self.end_conv = nn.Conv2d(hidden_size, 1, kernel_size=1)

        receptive_field = 1
        additional_scope = kernel_size - 1
        dilation = 1
        for l in range(num_layers):
            tcn_sequence = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            self.tcns.append(tcn_sequence)

            self.bns.append(nn.BatchNorm1d(hidden_size))

            dilation *= 2
            receptive_field += additional_scope
            additional_scope *= 2
        self.receptive_field = receptive_field

    def forward(self, X):
        X = X.permute(0, 2, 1)
        in_len = X.shape[2]
        if in_len < self.receptive_field:
            x = nn.functional.pad(X, (self.receptive_field - in_len, 0))
        else:
            x = X

        x = self.start_conv(x)

        for i in range(self.num_layers):
            residual = x
            assert not torch.isnan(x).any()
            x = self.tcns[i](x)
            assert not torch.isnan(x).any()
            x = x + residual[:, :, -x.shape[-1] :]

            x = self.bns[i](x)
        assert not torch.isnan(x).any()
        x = self.end_conv(x)

        return torch.sigmoid(x.squeeze())


class ASU(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_features,
        hidden_dim,
        window_len,
        dropout=0.3,
        kernel_size=2,
        layers=4,
        supports=None,
        spatial_bool=True,
        addaptiveadj=True,
        aptinit=None,
    ):
        super(ASU, self).__init__()
        self.sagcn = SAGCN(
            num_nodes,
            in_features,
            hidden_dim,
            window_len,
            dropout,
            kernel_size,
            layers,
            supports,
            spatial_bool,
            addaptiveadj,
            aptinit,
        )
        self.linear1 = nn.Linear(hidden_dim, 1)

        # 要不要用 LayerNorm ？ 可提升模型的穩定性與泛化能力
        self.bn1 = nn.BatchNorm1d(
            num_features=hidden_dim
        )  # author wrong, origin num_features=num_nodes
        self.in1 = nn.InstanceNorm1d(num_features=num_nodes)

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
        )
        self.hidden_dim = hidden_dim

    def forward(self, inputs, mask):
        """
        inputs: [batch, num_stock, window_len, num_features]
        mask: [batch, num_stock]
        outputs: [batch, scores]
        """
        x = self.sagcn(inputs)
        x = self.bn1(x).permute(0, 2, 1)
        x = self.linear1(x).squeeze(-1)  # [B, N, C] -> [B, N]

        score = 1 / ((-x).exp() + 1)
        score[mask] = -math.inf
        return score


###  SAGAN module (to make size well define)


class TimeCompressor(nn.Module):
    def __init__(self, in_len):
        super().__init__()
        self.linear = nn.Linear(in_len, 1)

    def forward(self, x):  # [B, C, N, T]
        B, C, N, T = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, N, C, T]
        x = self.linear(x).squeeze(-1)  # [B, N, C]
        return x


class TemporalAttentionPool(nn.Module):
    def __init__(self, in_len, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=in_len, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(in_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, N, T]
        B, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # → [B, N, T, C]
        x = x.reshape(B * N, T, C)  # → [B*N, T, C]

        attn_out, _ = self.attn(x, x, x)  # Self-Attention over time
        out = x + self.dropout(attn_out)  # Residual
        out = self.norm(out)  # LayerNorm
        out = out.mean(dim=1)  # → [B*N, C]
        out = out.view(B, N, C)  # → [B, N, C]
        return out


"""
support 作用是是為圖卷積操作（Graph Convolution）提供 鄰接矩陣（adjacency matrices）
"""

#### -----------------------------------------###

import math
import os

import numpy as np
import torch
import torch.nn as nn

# --- 1. 定義測試用的超參 和 數據文件路徑 ---
batch_size = 2
num_stocks = 49
window_len = 13  # deep trader 設定
num_features = (
    16  # 新的特徵數量 (例如 open, high, low, close, adjcp, zopen, ..., zd_30)
)

hidden_dim = 128  # SAGCN 隱藏層維度
gcn_layers = 2  # SAGCN 中的層數
kernel_size_tcn = 2
dropout_rate = 0.5
use_spatial_and_gcn = True  # 是否啟用 S-A 和 GCN
use_adaptive_adj = True  # 是否使用自適應鄰接矩陣

data_dir = "/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/ASU_data"  # 假設您的數據都放在這個目錄下
stocks_data_path = os.path.join(data_dir, "stocks_data.npy")
relation_file_path = os.path.join(data_dir, "industry_classification.npy")
# ROR_data_path = os.path.join(data_dir, 'ROR.npy') # ROR.npy 在ASU 前向傳播測試中不會直接用到


# --- 2. 加載數據並創建 ASU 模型實例 ---

# 加載 relation_file.npy (關係矩陣)
try:
    relation_matrix_np = np.load(relation_file_path)
    assert relation_matrix_np.shape == (
        num_stocks,
        num_stocks,
    ), f"Relation matrix shape mismatch! Expected ({num_stocks},{num_stocks}), Got {relation_matrix_np.shape}"
    support_matrix = torch.from_numpy(relation_matrix_np).float()
    supports_list = [support_matrix]  # SAGCN 期望一個列表
    print(f"成功加載關係矩陣: {relation_file_path}, shape: {support_matrix.shape}")
except FileNotFoundError:
    print(f"錯誤: 關係矩陣文件未找到於 {relation_file_path}")
    print("請確保文件路徑正確，或先使用隨機矩陣進行測試：")
    print("support_matrix = torch.randn(num_stocks, num_stocks)")
    print("supports_list = [support_matrix]")
    # Fallback to random if file not found, for script to run (optional)
    # support_matrix = torch.randn(num_stocks, num_stocks)
    # supports_list = [support_matrix]
    exit()  # 或者直接退出

asu_model = ASU(
    num_nodes=num_stocks,
    in_features=num_features,
    hidden_dim=hidden_dim,
    window_len=window_len,  # ASU/SAGCN 可能會用 window_len 進行內部計算或斷言
    dropout=dropout_rate,
    kernel_size=kernel_size_tcn,
    layers=gcn_layers,
    supports=supports_list,
    spatial_bool=use_spatial_and_gcn,
    addaptiveadj=use_adaptive_adj,
    aptinit=None,
)

# 將模型設置為評估模式
asu_model.eval()

# --- 3. 從 stocks_data.npy 創建輸入數據 ---
# inputs: [batch_size, num_stocks, window_len, num_features]
try:
    all_stocks_data_np = np.load(
        stocks_data_path
    )  # 預期 shape: [num_stocks, num_days, num_features]
    print(f"成功加載股票數據: {stocks_data_path}, shape: {all_stocks_data_np.shape}")

    loaded_num_stocks, loaded_num_days, loaded_num_features = all_stocks_data_np.shape

    # 驗證加載的數據維度是否與腳本參數匹配
    assert (
        loaded_num_stocks == num_stocks
    ), f"stocks_data.npy 中的股票數量 ({loaded_num_stocks}) 與參數 num_stocks ({num_stocks}) 不符"
    assert (
        loaded_num_features == num_features
    ), f"stocks_data.npy 中的特徵數量 ({loaded_num_features}) 與參數 num_features ({num_features}) 不符"
    assert (
        loaded_num_days >= window_len
    ), f"stocks_data.npy 中的總天數 ({loaded_num_days}) 不足以形成長度為 {window_len} 的時間窗口"

    # 從加載的數據中為 batch 創建輸入
    # 為了簡單測試，我們從頭開始取 `batch_size` 個連續（或重疊）的時間窗口
    inputs_list = []
    for i in range(batch_size):
        start_idx = i  # 簡單起見，每個 batch item 的起始天不同
        if start_idx + window_len > loaded_num_days:
            print(
                f"警告: 數據不足以為 batch item {i} 創建不重疊的窗口，將重複使用第一個窗口。"
            )
            start_idx = 0  # 如果數據不夠，就重複使用第一個窗口

        # 切片: [num_stocks, window_len, num_features]
        single_input_window_np = all_stocks_data_np[
            :, start_idx : start_idx + window_len, :
        ]
        inputs_list.append(torch.from_numpy(single_input_window_np).float())

    inputs_from_file = torch.stack(
        inputs_list, dim=0
    )  # 將列表堆疊成 batch -> [batch_size, num_stocks, window_len, num_features]
    print(f"從文件創建的輸入數據形狀: {inputs_from_file.shape}")

except FileNotFoundError:
    print(f"錯誤: 股票數據文件未找到於 {stocks_data_path}")
    print("請確保文件路徑正確。")
    exit()
except Exception as e:
    print(f"加載或處理 stocks_data.npy 時發生錯誤: {e}")
    exit()


# mask: [batch_size, num_stocks] - 假設沒有股票被遮罩
# 注意: mask 中 True 的位置表示該股票的分數會被設為 -inf
fake_mask = torch.zeros(batch_size, num_stocks, dtype=torch.bool)
# 如果想遮罩某些股票，可以這樣設定:
# fake_mask[0, 1] = True # 遮罩第一個樣本中的第二支股票

# --- 4. 進行前向傳播測試 ---
try:
    with torch.no_grad():  # 在測試時不需要計算梯度
        output_scores = asu_model(
            inputs_from_file, fake_mask
        )  # <--- 使用從文件加載的數據

    print("\nASU 模型成功輸出了結果！")
    print("輸出分數的形狀:", output_scores.shape)  # 預期: [batch_size, num_stocks]
    print("輸出分數範例:\n", output_scores)

    if fake_mask.any():
        masked_scores = output_scores[fake_mask]
        if torch.isneginf(masked_scores).all():
            print("被遮罩的股票分數已成功設置為負無窮大。")
        else:
            print("警告：被遮罩的股票分數未完全設置為負無窮大！")
            print("被遮罩位置的分數:", masked_scores)

except Exception as e:
    print(f"模型前向傳播過程中發生錯誤: {e}")
    import traceback

    traceback.print_exc()

# 備註：ROR.npy 在這個腳本中沒有直接使用，因為我們只測試 ASU 的前向傳播。
# ROR.npy 會在完整的 PortfolioEnv 環境中使用，用於計算獎勵和環境狀態轉移。
print("\n備註: ROR.npy 在此獨立 ASU 前向傳播測試中未使用。")
