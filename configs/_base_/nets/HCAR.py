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

    output_final_weights = True # 指示 HCAR_Actor 輸出最終權重
)

cri = dict(
    type = "HCAR_Critic",
    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 1,
    hidden_size=32
)
