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
    transaction_cost_pct=0.0,
    test_dynamic_path='data/portfolio_management/tw50/test_with_label.csv',
    time_steps=30,
    start_date_filter='2018-01-01',
    test_dynamic='-1')
environment = dict(
    type='PortfolioManagementEIIEEnvironment',
    rebalance_interval=15,
    start_date_filter='2018-01-01')
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)
trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=10,
    work_dir='work_dir/portfolio_management_tw50_eiie_eiie_adam_mse',
    if_remove=False)
loss = dict(type='MSELoss')
optimizer = dict(type='AdamW', lr=4e-05, weight_decay=1e-05)
act = dict(
    type='EIIEConv',
    input_dim=13,
    time_steps=30,
    d_model=64,
    n_heads=4,
    num_encoder_layers=2,
    dim_feedforward=128,
    embed_dropout_p=0.1,
    transformer_dropout_p=0.1,
    scoring_hidden_dim=256,
    scoring_dropout_p=0.1)
cri = dict(
    type='EIIECritic',
    input_dim=13,
    action_dim=49,
    time_steps=30,
    d_model=64,
    n_heads=4,
    num_encoder_layers=2,
    dim_feedforward=128,
    embed_dropout_p=0.1,
    transformer_dropout_p=0.1,
    q_head_hidden_dim=64,
    q_head_dropout_p=0.1)
transition = dict(type='Transition')
task_name = 'portfolio_management'
dataset_name = 'tw50'
net_name = 'eiie'
agent_name = 'eiie'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_tw50_eiie_eiie_adam_mse'
