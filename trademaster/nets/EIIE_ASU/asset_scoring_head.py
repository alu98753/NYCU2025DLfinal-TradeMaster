import torch
import torch.nn as nn

class AssetScoringHead(nn.Module):
    def __init__(self, input_dim, hidden_layers_dims=None, output_dim=1, dropout=0.3):
        """
        Args:
            input_dim (int): 輸入特徵維度 (即 RelationalContextIntegrator 輸出的 hidden_dim_3B)
            hidden_layers_dims (list of int, optional): MLP隱藏層的維度列表。
                                                       如果為 None 或空列表，則只有一個輸出層。
            output_dim (int): 每個股票輸出的評分維度，通常為 1。
            dropout (float): Dropout 比率。
        """
        super(AssetScoringHead, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        if hidden_layers_dims:
            for h_dim in hidden_layers_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU()) # 或者其他激活函數如 GELU, SiLU
                layers.append(nn.Dropout(dropout))
                current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, h_final_input):
        # h_final_input 期望形狀: [batch_size, num_stocks, input_dim]
        print(f"AssetScoringHead - Input h_final_input shape: {h_final_input.shape}")
        
        batch_size, num_stocks, feature_dim = h_final_input.shape
        
        # 為了讓 MLP 共享參數處理每個股票的特徵，我們先將 batch 和 num_stocks 維度合併
        x_reshaped = h_final_input.reshape(batch_size * num_stocks, feature_dim)
        print(f"AssetScoringHead - Reshaped input for MLP (x_reshaped shape): {x_reshaped.shape}")
        
        # MLP 處理
        scores_flat = self.mlp_head(x_reshaped) # 輸出 [batch_size * num_stocks, output_dim]
        print(f"AssetScoringHead - Output from MLP (scores_flat shape): {scores_flat.shape}")
        
        # 將輸出 reshape 回 [batch_size, num_stocks, output_dim]
        # 因為 output_dim 通常為 1，我們可以 squeeze(-1) 得到 [batch_size, num_stocks]
        output_scores = scores_flat.view(batch_size, num_stocks, -1).squeeze(-1)
        print(f"AssetScoringHead - Final output_scores shape: {output_scores.shape}")
        
        return output_scores # 形狀 [batch_size, num_stocks]


if __name__ == '__main__':
    # --- 假設之前的模塊已定義或可以導入 ---
    # from your_temporal_module import TemporalFeatureExtractor
    # from your_relational_module import RelationalContextIntegrator 
    # (實際上我們只需要 AssetScoringHead 進行獨立測試)

    # --- 測試參數 ---
    batch_size_test = 4
    num_stocks_test = 49
    # 假設 RelationalContextIntegrator 輸出的 hidden_dim 是 64
    input_dim_for_scoring_head = 64 
    mlp_hidden_dims_test = [32] # MLP頭可以有一層或多層隱藏層，或沒有 (None)
    dropout_test = 0.3

    print("\n--- Initializing AssetScoringHead ---")
    scoring_head = AssetScoringHead(
        input_dim=input_dim_for_scoring_head,
        hidden_layers_dims=mlp_hidden_dims_test,
        output_dim=1, # 每個股票一個評分
        dropout=dropout_test
    )
    scoring_head.eval()

    print("\n--- Preparing Fake Input h_final (Output from Sub-module 3.B) ---")
    # h_final 的形狀是 [batch_size, num_stocks, input_dim_for_scoring_head]
    fake_h_final = torch.randn(batch_size_test, num_stocks_test, input_dim_for_scoring_head)
    print(f"Shape of fake_h_final: {fake_h_final.shape}")

    print("\n--- Running Forward Pass through AssetScoringHead ---")
    try:
        with torch.no_grad():
            # S_assets 的形狀將是 [batch_size, num_stocks]
            S_assets = scoring_head(fake_h_final)
            print("--- Forward Pass Successful ---")
            print(f"Output S_assets shape: {S_assets.shape}")
            assert S_assets.shape == (batch_size_test, num_stocks_test), "Output shape mismatch!"

            print(f"Example S_assets (first sample, first 5 stocks): \n{S_assets[0, :5]}")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()