task_name = "portfolio_management"
dataset_name = "tw50"
net_name = "HCAR"
agent_name = "eiie"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/eiie_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py"
]

data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/tw50',
    train_path='data/portfolio_management/tw50/train.csv',
    valid_path='data/portfolio_management/tw50/valid.csv',
    test_path='data/portfolio_management/tw50/test.csv',
    test_dynamic_path='data/portfolio_management/tw50/test_with_label.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',# "limitflag", "check", #  ,  擇一
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    # time_steps=30,
    length_day=10, # 10 11 days is good for tcns , less is not good
    initial_amount=100000,
    transaction_cost_pct=0.001)

environment = dict(type='PortfolioManagementEIIEEnvironment')
transition = dict(
    type = "Transition"
)
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500,
    # --- 新增 LR Scheduler 和 Warmup 相關參數 ---
    use_lr_scheduler=True,          # 是否啟用學習率調度 (包含 warmup)
    warmup_steps=50000,              # 預熱的優化器步數 (不是 epoch！)
    initial_lr_actor = 0.0001,    # Actor 的目標學習率 (可以從 optimizer 配置中讀取)
    initial_lr_critic = 0.0001,   # Critic 的目標學習率 (可以從 optimizer 配置中讀取)
    lr_decay_scheduler_type = "CosineAnnealingLR", # 例如: "CosineAnnealingLR", "StepLR", "None"
    decay_t_max = 50000,          # CosineAnnealingLR 的 T_max (總優化步數 - warmup_steps)
    decay_step_size = 10000,      # StepLR 的 step_size
    decay_gamma = 0.5,            # StepLR 的 gamma
    )

trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=100,
    work_dir=work_dir,
    if_remove=False )

loss = dict(type='MSELoss')

# optimizer = dict(type='Adam', lr=1e-6, weight_decay=1e-5) # 0.001
optimizer = dict(type='Adam', lr=1e-6, weight_decay=1e-5) # 0.001 

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
    s_market_extractor_config = s_market_extractor_cfg,
    gate_controller_config = gate_controller_cfg,
    
    # 子模塊 3.A 參數
    temporal_hidden_dim = 64,#64,
    tcn_kernel_size = 2,
    num_tcn_layers = 4, #3
    temporal_dropout = 0.4, #0.4 -> 4 is good in only tcns

    # 子模塊 3.B 參數
    relational_hidden_dim = 64, #64,
    
    num_gcn_layers = 0, #1, 2
    relational_dropout = 0.0, #0.3
    gcn_bool = False, 
    
    spatialattn_bool = False, 
    addaptiveadj = False, # 先不用
    
    scoring_mlp_hidden_dims =  [32], # [32],或者 None
    scoring_dropout = 0.0, #0.3
    use_temporal_skip_to_scoring = True,
    fusion_method_for_scoring = 'cat',

    output_final_weights = True # 指示 HCAR_Actor 輸出最終權重
)

cri = dict(
    type = "HCAR_Critic",
    # s_market_extractor_config_critic = s_market_extractor_cfg, # Critic 使用相同的S_market提取器配置

    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 3,
    hidden_size=128
)

