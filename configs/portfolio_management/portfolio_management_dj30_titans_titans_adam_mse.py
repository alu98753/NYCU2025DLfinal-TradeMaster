task_name = "portfolio_management"
dataset_name = "dj30"
net_name = "titans"
agent_name = "eiie"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

titans_d_model = 128
titans_depth = 2
titans_heads = 4        # <--- 新增: Transformer heads 數量
titans_dim_head = 32    # <--- 新增: 每個 head 的維度 (d_model 應該是 heads * dim_head)
titans_segment_len = 32 # SegmentedAttention 參數
titans_persist_mem = 4
titans_longterm_mem = 16
titans_ff_mult = 4      # <--- 新增: FeedForward 擴展因子
titans_dropout = 0.1
# 確保 d_model = heads * dim_head
assert titans_d_model == titans_heads * titans_dim_head

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
    data_path='data/portfolio_management/dj30',
    train_path='data/portfolio_management/dj30/train.csv',
    valid_path='data/portfolio_management/dj30/valid.csv',
    test_path='data/portfolio_management/dj30/test.csv',
    test_dynamic_path='data/portfolio_management/dj30/test_with_label.csv',
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
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)

trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=2,
    work_dir=work_dir,
    if_remove=False )

loss = dict(type='MSELoss')

optimizer = dict(type='Adam', lr=3e-4)

aact = dict(
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