s_market_extractor_cfg = dict(
    # F_in, num_stocks, time_steps 將動態傳入
    temporal_processed_dim=64, # HCAR_Actor中TemporalFeatureExtractor的輸出維度 (temporal_hidden_dim)
    s_market_dim=32,           # S_market 向量的目標維度
    stock_pool_num_heads=4,    # 股票維度多頭注意力池化的頭數
    stock_pool_dropout=0.1,
    s_market_mlp_depth=2,
    s_market_mlp_expansion_factor=2
)

gate_controller_cfg = dict(
    s_market_dim=32, # 應與 s_market_extractor_cfg.s_market_dim 一致
    controller_depth=2,
    controller_expansion_factor=2,
    cash_adjustment_scale=1.0, # Tanh 縮放因子
    ema_alpha_gate=0.1         # 門控信號EMA平滑因子
)


act = dict(
    type="HCAR_Actor",
    # num_original_features, num_stocks, window_len, supports 會在 train_eiie.py 動態填充

    # 子模塊 3.A 參數
    temporal_hidden_dim = 64,
    tcn_kernel_size = 2,
    num_tcn_layers = 3,
    temporal_dropout = 0.3,

    # 子模塊 3.B 參數
    relational_hidden_dim = 64,
    num_gcn_layers = 2,
    relational_dropout = 0.3,
    gcn_bool = True, 
    spatialattn_bool = True, 
    addaptiveadj = True, # 如果 RelationalContextIntegrator 支持

    # 子模塊 3.C 參數
    scoring_mlp_hidden_dims = [32], # 或者 None
    scoring_dropout = 0.3,

    output_final_weights = True, # 指示 HCAR_Actor 輸出最終權重
    s_market_extractor_config= s_market_extractor_cfg,
    gate_controller_config = gate_controller_cfg
)

cri = dict(
    type = "HCAR_Critic",
    # s_market_dim = act['s_market_extractor_config']['s_market_dim'],
    num_regimes = 2,
    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 3,
    hidden_size=128,
    # film_hidden=32,      # FiLM 内部隐藏维度
    # aggregate_dim=128 
)
market = dict(
    type = "MarketNet",
    s_market_dim = act['s_market_extractor_config']['s_market_dim'],      # 必须和 Actor.s_market_dim 完全一致
    hidden_depth = 4,       # MLP 内部维度
    expansion_factor = 2,
    market_lr = 1e-4
)     