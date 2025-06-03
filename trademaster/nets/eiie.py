import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net
from trademaster.utils import build_conv2d
from torch import Tensor
import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe is [max_len, 1, d_model], reshape for batch_first=True
        # We want [1, max_len, d_model] to add to [B, SeqLen, d_model]
        self.register_buffer('pe', pe.permute(1, 0, 2))

    def forward(self, x):
        # x shape: [Batch, SeqLen, Dim]
        # self.pe shape: [1, MaxLen, Dim]
        # Add positional encoding to x. self.pe is sliced to match x's SeqLen.
        return x + self.pe[:, :x.size(1), :]


@NETS.register_module()
class EIIEConv(Net): # This is your Transformer Actor
    def __init__(self, input_dim, time_steps,
                 d_model=128, n_heads=4, num_encoder_layers=2, dim_feedforward=256,
                 embed_dropout_p=0.1, transformer_dropout_p=0.1,
                 scoring_hidden_dim=64, scoring_dropout_p=0.1,
                 top_k_stocks_to_select: int = 10, # 新增：可配置的 Top-K 參數
                 default_bull_stock_alloc: float = 0.9 # 新增：當 MarketNet 不可用時的默認牛市股票配置
                ):
        super().__init__()
        self.d_model = d_model
        self.top_k_stocks_to_select = top_k_stocks_to_select
        self.default_bull_stock_alloc = default_bull_stock_alloc
        self.default_bull_cash_alloc = 1.0 - default_bull_stock_alloc

        # 1. Embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.embed_layer_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(embed_dropout_p)

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=time_steps)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout_p,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Scoring Head (after pooling)
        self.scoring_head = nn.Sequential(
            nn.Linear(d_model, scoring_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(scoring_hidden_dim),
            nn.Dropout(scoring_dropout_p),
            nn.Linear(scoring_hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "cash_bias_parameter" in name: # 如果選擇保留，可以單獨處理
                # nn.init.normal_(p, mean=0.0, std=0.1)
                continue # 因為我們移除了它
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name and p.dim() == 1:
                nn.init.zeros_(p)

    def extract_market_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        從原始輸入 x 中提取聚合的市場特徵向量 (s_market)。
        s_market 用於 MarketNet 的輸入。

        Args:
            x (torch.Tensor): 輸入張量，形狀為 [B, N, T, input_dim]
                              或在數據準備時單個樣本可能為 [N, T, input_dim]。

        Returns:
            torch.Tensor: s_market，形狀為 [B, d_model] 的張量。
        """
        # 處理可能的輸入維度變化 (例如，數據準備時可能沒有批次維度)
        if x.dim() == 3: # 假設是 [N, T, input_dim]
            x = x.unsqueeze(0) # 增加批次維度: [1, N, T, input_dim]
        
        # 處理RL緩衝區中可能存在的額外維度 [B, 1, N, T, input_dim]
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1) # 變為 [B, N, T, input_dim]
        
        B, N, T, _input_dim = x.shape

        # 為每個股票獨立處理：reshape 成 [B*N, T, input_dim]
        x_reshaped = x.reshape(B * N, T, _input_dim)

        # 1. Embedding
        x_embedded = self.input_projection(x_reshaped)
        x_embedded = self.embed_layer_norm(x_embedded)
        x_embedded = self.embed_dropout(x_embedded) # 形狀: [B*N, T, d_model]

        # 2. Positional Encoding
        x_with_pos_enc = self.positional_encoding(x_embedded) # 形狀: [B*N, T, d_model]
        
        # 3. Transformer Encoder
        encoded_sequence = self.transformer_encoder(x_with_pos_enc) # 形狀: [B*N, T, d_model]
        
        # 4. 時間維度池化 (對每個股票的時間序列取最大值)
        # pooled_output_per_stock 形狀: [B*N, d_model]
        pooled_output_per_stock = encoded_sequence.max(dim=1).values 
        
        # 5. Reshape 成 [B, N, d_model] 以便進行股票間聚合
        # self.d_model 在 __init__ 中已定義
        pooled_output_batch = pooled_output_per_stock.view(B, N, self.d_model)
        
        # 6. 跨股票聚合 (沿 N 維度進行平均池化)
        s_market = pooled_output_batch.mean(dim=1) # 形狀: [B, d_model]
        
        return s_market
    
    def forward(self, x: torch.Tensor, market_regime_signal):
        # x: [B, N, T, input_dim] 或 [B, 1, N, T, input_dim]
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        B, N, T, _input_dim = x.shape # Use _input_dim to avoid conflict with class member

        # --- 特徵提取部分 (與 extract_market_features 中的步驟類似) ---
        x_reshaped = x.reshape(B * N, T, _input_dim)
        x_embedded = self.input_projection(x_reshaped)
        x_embedded = self.embed_layer_norm(x_embedded)
        x_embedded = self.embed_dropout(x_embedded)
        x_with_pos_enc = self.positional_encoding(x_embedded)
        encoded_sequence = self.transformer_encoder(x_with_pos_enc)
        # print("encoded_sequence:",encoded_sequence.std(), "max:",encoded_sequence.max(), "min:",encoded_sequence.min())

        # pooled_output 是時間池化後的個股表示 [B*N, d_model]
        pooled_output = encoded_sequence.max(dim=1).values 
        # print("pooled_output, std:",pooled_output.std(), "max:",pooled_output.max(), "min:",pooled_output.min())

        logits_flat = self.scoring_head(pooled_output) # Shape: [B*N, 1]
        stock_logits = logits_flat.view(B, N) # Shape: [B, N]
        print("stock_logits.std():",stock_logits.std(), "max:",stock_logits.max(), "min:",stock_logits.min())

        # --- 根據 MarketNet 信號確定目標股票和現金配置比例 ---
        # 初始化為默認值 (例如，牛市配置)
        target_stock_alloc_pct = torch.full((B, 1), self.default_bull_stock_alloc, device=x.device, dtype=x.dtype)
        target_cash_alloc_pct = torch.full((B, 1), self.default_bull_cash_alloc, device=x.device, dtype=x.dtype)

        if market_regime_signal is not None:
            # print(f"Actor received market_regime_signal: {market_regime_signal.cpu().numpy()}")

            for i in range(B):
                if market_regime_signal[i].item() == 1:  # 熊市 (Bear)
                    target_stock_alloc_pct[i] = 0.1
                    target_cash_alloc_pct[i] = 0.9
                elif market_regime_signal[i].item() == 0:  # 牛市 (Bull)
                    target_stock_alloc_pct[i] = 0.9
                    target_cash_alloc_pct[i] = 0.1
                # else: 可以處理其他信號值或默認情況，目前默認是牛市配置
        # else:
            # print("EIIEConv Actor: MarketNet signal is None, using default allocation.")
            # 已在初始化時設置為默認值


        # --- Top-K 選股與權重分配 ---
        stock_weights_final = torch.zeros_like(stock_logits) # [B, N]
    
        topk_vals, topk_indices = torch.topk(stock_logits, self.top_k_stocks_to_select, dim=1) # [B, actual_top_k]
        topk_relative_weights = torch.softmax(topk_vals, dim=1) # [B, actual_top_k], 相對權重和為1
        # 將目標股票配置比例分配給這 top_k 支股票
        # target_stock_alloc_pct 是 [B, 1]
        distributed_stock_weights = topk_relative_weights * target_stock_alloc_pct # [B, actual_top_k]

        for i in range(B):
            stock_weights_final[i].scatter_(0, topk_indices[i], distributed_stock_weights[i])

        cash_weight_final = target_cash_alloc_pct # [B, 1]

        action_probs = torch.cat([stock_weights_final, cash_weight_final], dim=1) # [B, N+1]


        # Logging only batch 0
        order = torch.argsort(topk_indices[0], descending=True)
        print(f"\nBatch {0} Top-{self.top_k_stocks_to_select} Indices:", topk_indices[0][order].tolist())
        print(f"Batch {0} Top-{self.top_k_stocks_to_select} Weights:", [round(float(x), 4) for x in topk_relative_weights[0][order]])
        print(f"Batch {0} Sum of Top-{self.top_k_stocks_to_select} Weights: {topk_relative_weights[0].sum().item():.4f}")


        # print("\nFinal action_probs:", action_probs.detach().cpu().numpy().round(4))
        # print("Sum of action_probs (should be 1.0):", action_probs.sum(dim=1).detach().cpu().numpy().round(4))
        # ---- Custom Top-5 Logic End ----

        return action_probs        

@NETS.register_module()
class EIIECritic(Net):
    def __init__(self,
                 input_dim,            # F_in (features per stock per time step)
                 action_dim,           # N (number of stocks, action will be N+1 for cash)
                 time_steps,           # T (sequence length for Transformer)
                num_market_regimes: int = 2, # <<< 新增: 市場狀態的類別數量 (例如2: 牛/熊) >>>
                 d_model=128,          # Transformer's embedding dimension
                 n_heads=4,            # Number of attention heads
                 num_encoder_layers=2, # Number of Transformer encoder layers for state
                 dim_feedforward=256,  # Dimension of FFN in Transformer
                 embed_dropout_p=0.1,
                 transformer_dropout_p=0.1,
                 q_head_hidden_dim=128, # Hidden dimension for the Q-value MLP
                 q_head_dropout_p=0.1
                 ):
        super(EIIECritic, self).__init__()
        self.d_model = d_model
        self.num_stocks = action_dim # N
        self.num_market_regimes = num_market_regimes

        # --- 狀態處理路徑 (與 Actor 類似) ---
        self.input_projection_state = nn.Linear(input_dim, d_model)
        self.embed_layer_norm_state = nn.LayerNorm(d_model)
        self.embed_dropout_state = nn.Dropout(embed_dropout_p)
        self.positional_encoding_state = PositionalEncoding(d_model, max_len=time_steps)

        encoder_layer_state = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=transformer_dropout_p, activation='gelu', batch_first=True
        )
        self.transformer_encoder_state = nn.TransformerEncoder(
            encoder_layer_state, num_layers=num_encoder_layers
        )

        # --- Q 值頭 (Q-Value Head) ---
        # 輸入維度 = portfolio_state_representation (d_model) + action_a (N+1) + market_regime_feature (1)
        regime_feature_dim = 1 # 假設 market_regime_signal (0 或 1) 作為單個特徵
        q_head_input_dim = d_model + (self.num_stocks + 1) + regime_feature_dim
        
        self.q_value_head = nn.Sequential(
            nn.Linear(q_head_input_dim, q_head_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(q_head_hidden_dim),
            nn.Dropout(q_head_dropout_p),
            nn.Linear(q_head_hidden_dim, 1)  # 輸出單個 Q 值
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name and p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, state_x: torch.Tensor, action_a: torch.Tensor, market_regime_signal):
        # state_x: [B, N, T, input_dim]
        # action_a: [B, N+1] (投資組合權重)
        # market_regime_signal: [B] or None, 整數信號 (0 代表牛市, 1 代表熊市)

        if state_x.dim() == 5 and state_x.size(1) == 1:
            state_x = state_x.squeeze(1)
        B, N, T, _input_dim = state_x.shape

        # --- 狀態處理 (與之前一致) ---
        x_reshaped = state_x.reshape(B * N, T, _input_dim)
        x_embedded = self.input_projection_state(x_reshaped)
        x_embedded = self.embed_layer_norm_state(x_embedded)
        x_embedded = self.embed_dropout_state(x_embedded)
        x_with_pos_enc = self.positional_encoding_state(x_embedded)
        encoded_state_sequence = self.transformer_encoder_state(x_with_pos_enc)
        individual_stock_repr_flat = encoded_state_sequence.max(dim=1).values
        individual_stock_repr_batch = individual_stock_repr_flat.view(B, N, self.d_model)
        portfolio_state_representation = individual_stock_repr_batch.mean(dim=1) # Shape: [B, d_model]

        # --- 準備 market_regime_feature ---
        if market_regime_signal is not None:
            # market_regime_signal 是 [B]，轉換為 [B, 1] 的浮點數以便拼接
            market_regime_feature = market_regime_signal.float().unsqueeze(-1)
        else:
            # 如果信號為 None (例如 MarketNet 未啟用)，使用默認特徵 (例如全0)
            # print("EIIECritic: MarketNet signal is None, using zero feature for regime.")
            market_regime_feature = torch.zeros((B, 1), device=state_x.device, dtype=torch.float32)

        # --- 拼接特徵: 組合狀態表示、動作、市場狀態特徵 ---
        combined_features = torch.cat(
            (portfolio_state_representation, action_a, market_regime_feature),
            dim=1
        )
        # 預期形狀: [B, d_model + (N+1) + regime_feature_dim (即1)]
        
        # --- 計算 Q 值 ---
        q_value = self.q_value_head(combined_features) # Shape: [B, 1]
        return q_value
    
@NETS.register_module()
class MarketNet(nn.Module):
    def __init__(self, input_s_market_dim: int, hidden_dim: int = 64, num_classes: int = 2, dropout_p: float = 0.1):
        """
        一個用於市場狀態分類的簡單 MLP，使用 BatchNorm。
        Args:
            input_s_market_dim (int): 輸入 s_market 向量的維度 (例如 EIIEConv 的 d_model)。
            hidden_dim (int): 隱藏層的維度。
            num_classes (int): 輸出類別的數量 (例如 2，對應牛市/熊市)。
            dropout_p (float): Dropout 概率。
        """
        super(MarketNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_s_market_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm 在線性層之後，激活函數之前
            nn.GELU(),                   # 使用 GELU 作為激活函數
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_classes)  # 輸出 logits
        )
        self._init_weights() # 初始化權重

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # Xavier 初始化線性層權重
                if m.bias is not None:
                    nn.init.zeros_(m.bias) # 線性層偏置初始化為0
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)  # BatchNorm 的 gamma (weight) 初始化為1
                nn.init.zeros_(m.bias)   # BatchNorm 的 beta (bias) 初始化為0

    def forward(self, s_market: torch.Tensor) -> torch.Tensor:
        """
        MarketNet 的前向傳播。
        Args:
            s_market (torch.Tensor): 輸入張量，形狀為 [B, input_s_market_dim]。
        Returns:
            torch.Tensor: Logits，形狀為 [B, num_classes]。
        """
        logits = self.network(s_market)
        return logits
    
# @NETS.register_module()
# class EIIEConv(Net):
#     def __init__(self,
#                  input_dim,
#                  output_dim = 1,
#                  time_steps = 10,
#                  kernel_size = 3,
#                  dims = (32, )):
#         super(EIIEConv, self).__init__()

#         self.kernel_size = kernel_size
#         self.time_steps = time_steps

#         self.net = build_conv2d(
#             dims=[input_dim, *dims, output_dim],
#             kernel_size=[(1, self.kernel_size), (1, self.time_steps - self.kernel_size + 1)]
#         )
#         self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

#     def forward(self, x): # (batch_size, num_seqs, action_dim, time_steps, state_dim)
#         if len(x.shape) > 4:
#             x = x.squeeze(1)
#         x = x.permute(0, 3, 1, 2)
#         x = self.net(x)
#         x = x.view(x.shape[0], -1)

#         # print("combined_logits:",x)
#         para = self.para.repeat(x.shape[0], 1)
#         x = torch.cat((x, para), dim=1)
#         x = torch.softmax(x, dim=1)
#         # print("action_probs:",x)
#         return x

# @NETS.register_module()
# class EIIECritic(Net):
#     def __init__(self,
#                  input_dim,
#                  action_dim,
#                  output_dim=1,
#                  time_steps=10,
#                  num_layers= 1,
#                  hidden_size = 32,
#                  ):
#         super(EIIECritic, self).__init__()

#         self.time_steps = time_steps

#         self.lstm = nn.LSTM(input_size=input_dim * time_steps,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True)
#         self.linear1 = nn.Linear(hidden_size, output_dim)
#         self.act = nn.ReLU()
#         self.linear2 = nn.Linear(2 * (action_dim + 1), 1)
#         self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

#     def forward(self, x, a):
#         if len(x.shape) >= 4:
#             x = x.view(x.shape[0], x.shape[1], -1)
#         lstm_out, _ = self.lstm(x)
#         x = self.linear1(lstm_out)

#         x = self.act(x)

#         x = x.view(x.shape[0], -1)
#         para = self.para.repeat(x.shape[0], 1)

#         x = torch.cat((x, para, a), dim=1)
#         x = self.linear2(x)
#         # x = x.mean(dim = 1, keepdim=True)
#         return x