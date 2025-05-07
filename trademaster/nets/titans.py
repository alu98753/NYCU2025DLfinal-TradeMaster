# trademaster/nets/titans_portfolio.py
import torch
import torch.nn as nn
import math

# 從 trademaster 內部導入基礎建設
from .builder import NETS
from .custom import Net

# 正確地從已安裝的 titans-pytorch package 導入
# 來源: https://github.com/lucidrains/titans-pytorch
from titans_pytorch.mac_transformer import SegmentedAttention, FeedForward, GEGLU # GEGLU 被 FeedForward 使用
from rotary_embedding_torch import RotaryEmbedding # 被 SegmentedAttention 使用
from einops.layers.torch import Rearrange # 被 SegmentedAttention 使用
from functools import partial # 被 SegmentedAttention 使用
from x_transformers.attend import Attend # 被 SegmentedAttention 使用
LinearNoBias = partial(nn.Linear, bias = False) # 被 SegmentedAttention 使用
# 可能還需要導入其他輔助函數如 exists, default 等
def exists(v): return v is not None


class PositionalEncoding(nn.Module):
    """
    標準的 Sinusoidal 位置編碼。
    來源: 基於 "Attention Is All You Need" (Vaswani et al., 2017) 的實現細節。
    針對 batch_first=True 的 Transformer 進行了調整。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # Shape: [d_model / 2]
        pe = torch.zeros(max_len, d_model) # Shape: [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term) # Broadcast position
        pe[:, 1::2] = torch.cos(position * div_term) # Broadcast position
        pe = pe.unsqueeze(0) # Shape: [1, max_len, d_model] - 添加批次維度以匹配 batch_first=True

        # 將 pe 註冊為 buffer，這樣它不會被視為模型參數，但會隨模型移動 (e.g., to device)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x shape: (batch_size, seq_len, d_model)
        # self.pe[:, :x.size(1)] 會選擇與輸入序列長度匹配的位置編碼
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@NETS.register_module()
class TitansPortfolioActor(Net):
    """
    使用 SegmentedAttention 和 FeedForward 實現的 Actor 網絡。
    """
    def __init__(self,
                 input_dim: int, stock_dim: int, time_steps: int,
                 d_model: int, depth: int, heads: int, dim_head: int, # Transformer 參數
                 segment_len: int, # SegmentedAttention 參數
                 num_persist_mem_tokens: int = 4,
                 num_longterm_mem_tokens: int = 16,
                 ff_mult: int = 4, # FeedForward 參數
                 dropout: float = 0.1, **kwargs): # **kwargs 捕捉配置中可能多餘的參數
        super(TitansPortfolioActor, self).__init__()

        # 檢查是否成功導入/複製了所需類
        if SegmentedAttention is None or FeedForward is None:
             raise ImportError("未能定義或導入 SegmentedAttention/FeedForward。請檢查導入或複製程式碼。")

        self.input_dim = input_dim
        self.stock_dim = stock_dim
        self.time_steps = time_steps
        self.d_model = d_model
        self.seq_len = stock_dim * time_steps

        # 1. Embedding 層
        self.embedding = nn.Linear(input_dim, d_model)
        # 2. 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len + 10)

        # 3. Transformer 層堆疊
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 注意：這裡可能需要添加 LayerNorm 和 Residual 連接
                nn.LayerNorm(d_model), # Pre-Normalization
                SegmentedAttention(
                    dim=d_model, dim_head=dim_head, heads=heads,
                    segment_len=segment_len,
                    num_persist_mem_tokens=num_persist_mem_tokens,
                    num_longterm_mem_tokens=num_longterm_mem_tokens
                    # accept_value_residual=False # 如果需要跨層 value residual
                ),
                nn.LayerNorm(d_model), # Pre-Normalization
                FeedForward(dim=d_model, mult=ff_mult)
            ]))

        # 4. Final LayerNorm
        self.norm = nn.LayerNorm(d_model)
        # 5. 輸出層
        self.output_layer = nn.Linear(d_model, stock_dim + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 輸入維度處理 ---
        if x.ndim == 5:
            if x.shape[0] == 1: x = x.squeeze(0)
            else:
                print(f"警告: 收到 5D 輸入 {x.shape}，將移除第一維度。")
                x = x.squeeze(0)
        assert x.ndim == 4, f"Actor forward 預期 4D 輸入，但得到 {x.ndim}D shape {x.shape}"
        B, N, T, F = x.shape

        # --- Reshape, Embedding, Positional Encoding ---
        x = x.reshape(B, self.seq_len, F)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # --- Transformer Layers ---
        for norm1, attn, norm2, ff in self.layers:
            # Pre-Norm and Residual Connection for Attention
            residual = x
            x_norm = norm1(x)
            attn_output, _ = attn(x_norm) # SegmentedAttention 返回 (output, intermediates)
            x = residual + attn_output

            # Pre-Norm and Residual Connection for FeedForward
            residual = x
            x_norm = norm2(x)
            ff_output = ff(x_norm)
            x = residual + ff_output

        # --- Final Norm, Aggregation, Output ---
        x = self.norm(x) # Shape: (B, S, D_model)

        # Aggregation
        x = x.mean(dim=1) # Pool along sequence length S -> (B, D_model)

        # Output layer
        x = self.output_layer(x) # Shape: (B, N+1)
        x = self.softmax(x)
        return x

@NETS.register_module()
class TitansPortfolioCritic(Net):
    """
    使用 SegmentedAttention 和 FeedForward 實現的 Critic 網絡 (估計 V(s))。
    """
    def __init__(self,
                 input_dim: int, stock_dim: int, time_steps: int,
                 d_model: int, depth: int, heads: int, dim_head: int, # Transformer 參數
                 segment_len: int, # SegmentedAttention 參數
                 num_persist_mem_tokens: int = 4,
                 num_longterm_mem_tokens: int = 16,
                 ff_mult: int = 4, # FeedForward 參數
                 action_dim: int = None, # Ignored for V(s)
                 output_dim: int = 1,
                 dropout: float = 0.1, **kwargs):
        super(TitansPortfolioCritic, self).__init__()

        if SegmentedAttention is None or FeedForward is None:
             raise ImportError("未能定義或導入 SegmentedAttention/FeedForward。請檢查導入或複製程式碼。")

        self.input_dim = input_dim
        self.stock_dim = stock_dim
        self.time_steps = time_steps
        self.d_model = d_model
        self.seq_len = stock_dim * time_steps

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len + 10)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(d_model),
                SegmentedAttention(
                    dim=d_model, dim_head=dim_head, heads=heads,
                    segment_len=segment_len,
                    num_persist_mem_tokens=num_persist_mem_tokens,
                    num_longterm_mem_tokens=num_longterm_mem_tokens
                ),
                nn.LayerNorm(d_model),
                FeedForward(dim=d_model, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, output_dim) # output_dim should be 1

    def forward(self, x: torch.Tensor, a: torch.Tensor = None) -> torch.Tensor:
        # --- 輸入維度處理 ---
        if x.ndim == 5:
            if x.shape[0] == 1: x = x.squeeze(0)
            else:
                print(f"警告: 收到 5D 輸入 {x.shape}，將移除第一維度。")
                x = x.squeeze(0)
        assert x.ndim == 4, f"Critic forward 預期 4D 輸入，但得到 {x.ndim}D shape {x.shape}"
        B, N, T, F = x.shape

        # --- Reshape, Embedding, Positional Encoding ---
        x = x.reshape(B, self.seq_len, F)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # --- Transformer Layers ---
        for norm1, attn, norm2, ff in self.layers:
            residual = x
            x_norm = norm1(x)
            attn_output, _ = attn(x_norm)
            x = residual + attn_output

            residual = x
            x_norm = norm2(x)
            ff_output = ff(x_norm)
            x = residual + ff_output

        # --- Final Norm, Aggregation, Output ---
        x = self.norm(x)
        x = x.mean(dim=1) # Pool -> (B, D_model)
        x = self.output_layer(x) # Output V(s) -> (B, 1)
        return x