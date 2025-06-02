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
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    time_steps=15,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    start_date_filter='2018-01-01',)

environment = dict(
    type='PortfolioManagementEIIEEnvironment',
    rebalance_interval=7,
    start_date_filter='2018-01-01',
)
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
    epochs=5,
    work_dir=work_dir,
    if_remove=False )

loss = dict(type='MSELoss')

# optimizer = dict(type='Adam', lr=0.0001) #EIIE為 0.001

optimizer = dict(type='AdamW', lr=4e-5, weight_decay=1e-5) # 或者先用 Adam 試試
'''
較小的基礎學習率：對於 Transformer，基礎學習率通常設置得更小，例如 1e-4, 5e-5, 3e-5 是常見的範圍。你可以從 optimizer = dict(type='Adam', lr=0.0001) 開始嘗試。
學習率預熱 (Warmup)：在訓練初期，從一個非常小的學習率開始，經過一定數量的 steps (warmup steps) 線性增加到你設定的基礎學習率。這有助於穩定訓練初期。
學習率衰減 (Decay)：在 warmup 之後，學習率可以按照一定的策略衰減，例如線性衰減、餘弦衰減等。
AdamW 優化器：對於 Transformer，AdamW 通常比 Adam 表現更好，因為它對權重衰減 (weight decay) 的處理方式不同。
'''
# act = dict(
#     type = "EIIEConv",
#     input_dim = None,
#     output_dim=1,
#     time_steps=15,
#     kernel_size=3,
#     dims = [32]
# )

act = dict(
    type = "EIIEConv",

    # --- 核心輸入維度 (通常由你的訓練腳本動態填充) ---
    input_dim = None,                       
    time_steps=15,                    
    # --- Transformer Encoder 主要參數 ---
    d_model = 64,                      # Transformer 內部的主要維度 (embedding dim)
    n_heads = 4,                        # 多頭注意力機制的頭數 (需確保 d_model % n_heads == 0)
    num_encoder_layers = 2,             # Transformer Encoder 層的數量 (建議從1-3層開始)
    dim_feedforward = 128,              # Encoder內部前饋網絡的隱藏層維度 (通常是 d_model 的 2倍或4倍)

    # --- Dropout 機率 ---
    embed_dropout_p = 0.1,              # 輸入嵌入層後的 Dropout
    transformer_dropout_p = 0.1,        # Transformer Encoder 內部各子層的 Dropout

    # --- 評分頭 (Scoring Head) 參數 ---
    scoring_hidden_dim = 256,            # 評分MLP的隱藏層維度
    scoring_dropout_p = 0.1             # 評分MLP的 Dropout
)


# cri = dict(
#     type = "EIIECritic",
#     input_dim = None,
#     action_dim = None,
#     output_dim=1,
#     time_steps=None,
#     num_layers = 1,
#     hidden_size=32
# )

cri = dict(
    type='EIIECritic',
    input_dim = None,        # Features per stock (F)
    action_dim = None,     # Number of stocks (N)
    time_steps=None,       # Look-back window (T)
    d_model=64,                 # Or match Actor's d_model
    n_heads=4,                   # Or match Actor's n_heads
    num_encoder_layers=2,        # Or match Actor's num_encoder_layers
    dim_feedforward=128,         # Or match Actor's dim_feedforward
    embed_dropout_p=0.1,
    transformer_dropout_p=0.1,
    q_head_hidden_dim=64,       # Hidden layer size for the MLP predicting Q value
    q_head_dropout_p=0.1
)
