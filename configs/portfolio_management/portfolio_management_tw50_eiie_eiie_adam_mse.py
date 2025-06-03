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
    train_path='data/portfolio_management/tw50/2024/train.csv',
    valid_path='data/portfolio_management/tw50/2024/valid.csv',
    test_path='data/portfolio_management/tw50/2024/test.csv',
    test_dynamic_path='data/portfolio_management/tw50/test_with_label.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    time_steps=30,#30
    initial_amount=100000,
    transaction_cost_pct=0.000,
    start_date_filter='2018-01-01',)

environment = dict(
    type='PortfolioManagementEIIEEnvironment',
    rebalance_interval=7, #train用 7  test用 15 is good   , 22 都用7
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
    epochs=10,
    work_dir=work_dir,
    if_remove=False,
    # --- MarketNet Calibration Parameters ---
    calibrate_marketnet_flag=True,                # 是否啟用 MarketNet 校準
    calibrate_marketnet_every_n_epochs=1,         # 每隔多少個 RL epoch 校準一次 MarketNet
    marketnet_calibrate_epochs=1000,                # 每次校準時 MarketNet 自身的訓練 epoch 數
    marketnet_calibrate_lr=1e-4,                  # MarketNet 校準時的學習率
    marketnet_calibrate_batch_size=32,           # MarketNet 校準時的 batch size
    marketnet_target_accuracy_bull=0.97,#0.97,          # MarketNet 對牛市的目標準確率 (0.0 到 1.0)
    marketnet_target_accuracy_bear=0.97, #0.97,          # MarketNet 對熊市的目標準確率 (0.0 到 1.0)
    marketnet_weight_decay=1e-4,#1e-4 or 1e-5 在test差不多
    # path_to_market_regime_data 會從 train_environment.df_path 自動獲取
    # 如果需要指定特定文件，可以在這裡配置並在 Trainer 的 __init__ 中優先使用
)

loss = dict(type='MSELoss')

# optimizer = dict(type='Adam', lr=0.0001) #EIIE為 0.001

optimizer = dict(type='AdamW', lr=4e-5, weight_decay=1e-3) # 或者先用 Adam 試試

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
    hidden_dim=256,                   # MarketNet 內部隱藏層維度
    num_classes=2,                   # 固定為2 (牛/熊)
    dropout_p=0.1                    # MarketNet 內部 dropout
)


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
#     time_steps=10,
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