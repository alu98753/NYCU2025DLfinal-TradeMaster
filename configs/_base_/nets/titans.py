# --- 在配置文件的頂部或合適位置定義超參數 ---
titans_d_model = 256
titans_depth = 4
titans_heads = 8        # <--- 新增: Transformer heads 數量
titans_dim_head = 32    # <--- 新增: 每個 head 的維度 (d_model 應該是 heads * dim_head)
titans_segment_len = 128 # SegmentedAttention 參數
titans_persist_mem = 16
titans_longterm_mem = 16
titans_ff_mult = 4      # <--- 新增: FeedForward 擴展因子
titans_dropout = 0.1

# 確保 d_model = heads * dim_head
assert titans_d_model == titans_heads * titans_dim_head

# --- 修改 act (Actor) 網絡配置 ---
act = dict(
    type = "TitansPortfolioActor", # <--- 保持新類名 (或根據您的實現修改)
    input_dim = None,
    stock_dim = None,
    time_steps = None,
    d_model = titans_d_model,
    depth = titans_depth,
    heads = titans_heads,         # <--- 提供 heads
    dim_head = titans_dim_head,     # <--- 提供 dim_head
    segment_len = titans_segment_len,
    num_persist_mem_tokens = titans_persist_mem,
    num_longterm_mem_tokens = titans_longterm_mem,
    ff_mult = titans_ff_mult,       # <--- 提供 ff_mult
    output_dim = None,
    dropout = titans_dropout
)

# --- 修改 cri (Critic) 網絡配置 ---
cri = dict(
    type = "TitansPortfolioCritic", # <--- 保持新類名 (或根據您的實現修改)
    input_dim = None,
    stock_dim = None,
    time_steps = None,
    d_model = titans_d_model,
    depth = titans_depth,
    heads = titans_heads,           # <--- 提供 heads
    dim_head = titans_dim_head,       # <--- 提供 dim_head
    segment_len = titans_segment_len,
    num_persist_mem_tokens = titans_persist_mem,
    num_longterm_mem_tokens = titans_longterm_mem,
    ff_mult = titans_ff_mult,         # <--- 提供 ff_mult
    action_dim = None,
    output_dim = 1,
    dropout = titans_dropout
)
