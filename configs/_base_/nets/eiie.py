
act = dict(
    type = "EIIEConv",

    # --- 核心輸入維度 (通常由你的訓練腳本動態填充) ---
    input_dim = None,                       
    time_steps = 10,                    
    # --- Transformer Encoder 主要參數 ---
    d_model = 128,                      # Transformer 內部的主要維度 (embedding dim)
    n_heads = 4,                        # 多頭注意力機制的頭數 (需確保 d_model % n_heads == 0)
    num_encoder_layers = 2,             # Transformer Encoder 層的數量 (建議從1-3層開始)
    dim_feedforward = 256,              # Encoder內部前饋網絡的隱藏層維度 (通常是 d_model 的 2倍或4倍)

    # --- Dropout 機率 ---
    embed_dropout_p = 0.1,              # 輸入嵌入層後的 Dropout
    transformer_dropout_p = 0.1,        # Transformer Encoder 內部各子層的 Dropout

    # --- 評分頭 (Scoring Head) 參數 ---
    scoring_hidden_dim = 64,            # 評分MLP的隱藏層維度
    scoring_dropout_p = 0.1             # 評分MLP的 Dropout
)

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


# act = dict(
#     type = "EIIEConv",
#     input_dim = None,
#     output_dim=1,
#     time_steps=None,
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

