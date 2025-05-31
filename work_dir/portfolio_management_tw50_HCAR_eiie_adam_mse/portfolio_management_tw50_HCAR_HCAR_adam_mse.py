data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/tw50',
    train_path='data/portfolio_management/tw50/train.csv',
    valid_path='data/portfolio_management/tw50/valid.csv',
    test_path='data/portfolio_management/tw50/test.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15',
        'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    test_dynamic_path='data/portfolio_management/tw50/test_with_label.csv',
    test_dynamic='-1')
environment = dict(type='PortfolioManagementEIIEEnvironment')
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500,
    use_lr_scheduler=True,
    warmup_steps=50000,
    initial_lr_actor=0.0001,
    initial_lr_critic=0.0001,
    lr_decay_scheduler_type='CosineAnnealingLR',
    decay_t_max=50000,
    decay_step_size=10000,
    decay_gamma=0.5)
trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=32,
    work_dir='work_dir/portfolio_management_tw50_HCAR_eiie_adam_mse',
    if_remove=False)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=1e-06, weight_decay=1e-05)
s_market_extractor_cfg = dict(
    temporal_processed_dim=64,
    s_market_dim=32,
    stock_pool_num_heads=4,
    stock_pool_dropout=0.1,
    s_market_mlp_depth=1,
    s_market_mlp_expansion_factor=1)
gate_controller_cfg = dict(
    s_market_dim=32,
    controller_depth=2,
    controller_expansion_factor=2,
    cash_adjustment_scale=1.0,
    ema_alpha_gate=0.1)
act = dict(
    type='HCAR_Actor',
    temporal_hidden_dim=64,
    tcn_kernel_size=2,
    num_tcn_layers=4,
    temporal_dropout=0.4,
    relational_hidden_dim=64,
    num_gcn_layers=0,
    relational_dropout=0.0,
    gcn_bool=False,
    spatialattn_bool=False,
    addaptiveadj=False,
    scoring_mlp_hidden_dims=[32],
    scoring_dropout=0.0,
    output_final_weights=True,
    s_market_extractor_config=dict(
        temporal_processed_dim=64,
        s_market_dim=32,
        stock_pool_num_heads=4,
        stock_pool_dropout=0.1,
        s_market_mlp_depth=1,
        s_market_mlp_expansion_factor=1),
    gate_controller_config=dict(
        s_market_dim=32,
        controller_depth=2,
        controller_expansion_factor=2,
        cash_adjustment_scale=1.0,
        ema_alpha_gate=0.1),
    use_temporal_skip_to_scoring=True,
    fusion_method_for_scoring='cat',
    num_original_features=11,
    num_stocks=49,
    supports=[
        tensor(
            [[0., 1., 1., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 1., 0.],
             [0., 1., 0., ..., 0., 1., 0.], ..., [0., 0., 0., ..., 0., 0., 0.],
             [0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.]],
            device='cuda:0')
    ])
cri = dict(
    type='HCAR_Critic',
    s_market_dim=32,
    input_dim=11,
    action_dim=49,
    output_dim=1,
    time_steps=10,
    num_layers=3,
    hidden_size=128)
transition = dict(type='Transition')
task_name = 'portfolio_management'
dataset_name = 'tw50'
net_name = 'HCAR'
agent_name = 'eiie'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_tw50_HCAR_eiie_adam_mse'
