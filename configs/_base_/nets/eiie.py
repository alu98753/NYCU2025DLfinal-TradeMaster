
eiie_d_model = 64 # 統一定義 d_model，方便引用

act = dict(
    type="EIIEConv",
    # input_dim 和 time_steps 會由 train_eiie.py 動態填充
    input_dim=None,
    time_steps=None, # data.time_steps 會被用作這個值 (例如30)
    
    d_model=eiie_d_model, # 例如 64
    n_heads=4,
    num_encoder_layers=2,
    dim_feedforward=128, # 通常是 d_model 的 2-4 倍

    embed_dropout_p=0.1,
    transformer_dropout_p=0.2,
    scoring_hidden_dim=256, # 你原來的設置
    scoring_dropout_p=0.2,

    # <<< 新增 Actor 特定參數 >>>
    top_k_stocks_to_select=10,       # Top-K 選股數量
    default_bull_stock_alloc=0.9     # MarketNet 不可用時，默認牛市股票配置比例
)

cri = dict(
    type='EIIECritic',
    # input_dim, action_dim, time_steps 會由 train_eiie.py 動態填充
    input_dim=None,
    action_dim=None,
    time_steps=None, # 應與 Actor 的 time_steps 一致

    d_model=eiie_d_model, # 與 Actor 的 d_model 一致
    n_heads=4,
    num_encoder_layers=2,
    dim_feedforward=128,
    embed_dropout_p=0.1,
    transformer_dropout_p=0.1,
    q_head_hidden_dim=64, # 你原來的設置
    q_head_dropout_p=0.1,

    # <<< 新增 Critic 特定參數 >>>
    num_market_regimes=2  # 市場狀態類別數量 (例如 0:牛, 1:熊)
)

market_net = dict(
    type="MarketNet", # 我們創建的 MarketNet 類名
    input_s_market_dim=eiie_d_model, # <<< 必須與 EIIEConv Actor 的 d_model 一致 >>>
    hidden_dim=64,                   # MarketNet 內部隱藏層維度
    num_classes=2,                   # 固定為2 (牛/熊)
    dropout_p=0.1                    # MarketNet 內部 dropout
)

# act = dict(
#     type = "EIIEConv",
#     input_dim = None,
#     output_dim=1,
#     time_steps=None,
#     kernel_size=3,
#     dims = [32]
# )

# cri = dict(
#     type = "EIIECritic",
#     input_dim = None,
#     action_dim = None,
#     output_dim=1,
#     time_steps=None,
#     num_layers = 1,
#     hidden_size=32
# )

