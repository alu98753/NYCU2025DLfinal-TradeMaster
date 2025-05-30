task_name = "portfolio_management"
dataset_name = "tw50"
net_name = "eiie"
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
    f"../_base_/nets/tw_eiie.py",
    f"../_base_/transition/transition.py"
]

data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/tw50',
    train_path='data/portfolio_management/tw50/train_.csv',
    valid_path='data/portfolio_management/tw50/valid_.csv',
    test_path='data/portfolio_management/tw50/test_.csv',
    test_dynamic_path='data/portfolio_management/tw50/test_with_label.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001)

environment = dict(type='PortfolioManagementEIIEEnvironment')
transition = dict(
    type = "Transition"
)
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=2000, #added 1000->2000
    gamma=0.99,
    policy_update_frequency=500)#500->10

trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=20,##2
    repeat_times=5,##added
    work_dir=work_dir,
    if_remove=False )

loss = dict(type='MSELoss')

optimizer = dict(type='Adam', lr=0.001)##0.001

act = dict(
    type = "EIIEConv",
    input_dim = None,
    output_dim=1,
    time_steps=10,
    kernel_size=[(1,3),(1,8)],
    dims = [32,20]  ## added [32]->[64]
)

cri = dict(
    type = "EIIECritic",
    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 1,
    hidden_size=32
)
