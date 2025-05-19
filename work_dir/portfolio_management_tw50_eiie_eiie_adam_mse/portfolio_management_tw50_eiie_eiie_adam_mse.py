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
    length_day=50,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    time_steps=50,
    test_dynamic_path='data/portfolio_management/tw50/test_with_label.csv',
    test_dynamic='-1')
environment = dict(type='PortfolioManagementEIIEEnvironment')
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=10000,
    gamma=0.99,
    policy_update_frequency=10)
trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=100,
    work_dir='work_dir/portfolio_management_tw50_eiie_eiie_adam_mse',
    if_remove=False,
    repeat_times=5)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.0005)
act = dict(
    type='EIIEConv',
    input_dim=11,
    output_dim=1,
    time_steps=50,
    kernel_size=[(1, 3), (1, 48)],
    dims=[32, 20])
cri = dict(
    type='EIIECritic',
    input_dim=None,
    action_dim=None,
    output_dim=1,
    time_steps=None,
    num_layers=1,
    hidden_size=32)
transition = dict(type='Transition')
task_name = 'portfolio_management'
dataset_name = 'tw50'
net_name = 'eiie'
agent_name = 'eiie'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_tw50_eiie_eiie_adam_mse'
