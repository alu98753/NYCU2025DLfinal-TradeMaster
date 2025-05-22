import numpy as np
import pandas as pd
from cdt.causality.graph import PC # 導入 PC 演算法
import networkx as nx # 用於處理圖結構 (cdt 的 PC 算法會返回 networkx 圖)
import os
# --- 1. 加載並準備 ROR 數據 ---

data_path = "/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/"
save_data = os.path.join(data_path, "ASU_data")


ror_data_np = np.load(save_data + '/ROR.npy') # Shape: [num_stocks, num_days]

# 假設 num_stocks 和 stock_list (股票代碼列表，順序與 ror_data_np 一致) 已定義
num_stocks = ror_data_np.shape[0]



stock_list = [
    "1101", "1216", "1301", "1303", "2002", "2207", "2301", "2303", "2308", "2317",
    "2327", "2330", "2337", "2345", "2357", "2379", "2382", "2395", "2412", "2454",
    "2603", "2609", "2615", "2880", "2881", "2882", "2883", "2884", "2885", "2886",
    "2887", "2890", "2891", "2892", "2912", "3008", "3017", "3034", "3037", "3045",
    "3231", "3661", "3711", "4904", "4938", "5871", "5876", "5880", "6505", "6669"
]

# 轉換為 [num_days, num_stocks] 並創建 Pandas DataFrame (cdt 的 PC 通常接受 DataFrame)
ror_data_for_pc = pd.DataFrame(ror_data_np.T, columns=stock_list) 
                                    # 或者如果您有 date_index:
                                    # pd.DataFrame(ror_data_np.T, columns=stock_list, index=date_index_train)

# 假設 ror_data_for_pc 現在只包含訓練期的數據
# 如果不是，您需要先進行數據劃分，例如：
# train_ror_df = ror_data_for_pc.loc[train_start_date:train_end_date]

print(f"用於 PC 演算法的 ROR 數據形狀: {ror_data_for_pc.shape}") # 應為 [num_training_days, num_stocks]

# --- 2. 初始化並運行 PC 演算法 ---
# 創建 PC 演算法物件
# method_type: 'statistical' (基於統計檢驗) 或 'functional' (基於函數關係)
# ci_estimator: 對於連續數據，常用 'pearsonr' (基於偏相關)
# alpha: 條件獨立性檢驗的顯著性水平 (例如 0.01, 0.05)
obj = PC(method_type='statistical', ci_estimator='pearsonr', alpha=0.05) 
# 您可能需要根據 cdt 的文檔調整參數，例如 'PearsonCorrelation' 等

print("開始運行 PC 演算法...")
# 學習因果圖 (輸出是一個 networkx.DiGraph 物件，表示有向圖)
# 注意：標準PC算法發現的是馬可夫等價類，可能包含無向邊，通常以CPDAG表示。
# CDT中的PC實現可能會直接給出一個有向圖（可能通過某種規則為無向邊定向）或允許你獲取CPDAG。
# 我們先假設它給出一個可以直接轉為鄰接矩陣的圖。
causal_graph = obj.predict(ror_data_for_pc)
print("PC 演算法運行完畢。")

# --- 3. 將圖轉換為鄰接矩陣 ---
# networkx 圖的節點可能直接是股票代碼 (如果 DataFrame 的 columns 是股票代碼)
# 我們需要確保鄰接矩陣的行和列順序與我們的 stock_list 一致

# 獲取排序後的節點列表 (以防 causal_graph.nodes 的順序不一致)
# 但 CDT 的 PC 通常會保持輸入 DataFrame 的欄位順序作為節點順序
adj_matrix = np.zeros((num_stocks, num_stocks), dtype=np.float32)

# 填充鄰接矩陣
# causal_graph.edges() 返回的是 (u, v) 對，表示從 u 到 v 的一條有向邊
# 鄰接矩陣 A 中 A[i, j] = 1 表示從節點 j 到節點 i 的邊 (或者相反，取決於慣例)
# 我們需要確認 ASU 中的 GCN 如何解釋這個矩陣。
# 通常，如果 A[source, target] = 1，表示 source -> target
# 如果 GCN 的訊息傳播是 X_new = A @ X_old，那麼 A[i, j]=1 通常意味著訊息從 j 流向 i。
# 讓我們假設 A[target_idx, source_idx] = 1 表示 source -> target

# stock_to_idx = {tic: i for i, tic in enumerate(stock_list)} # 確保有這個映射

# for u, v in causal_graph.edges():
#     source_idx = stock_to_idx[u]
#     target_idx = stock_to_idx[v]
#     adj_matrix[target_idx, source_idx] = 1.0 
        # 或者 adj_matrix[source_idx, target_idx] = 1.0，取決於您的GCN實現如何使用它
        # DeepTrader 論文中的GCN公式可能需要查看。
        # 為了簡單起見，我們也可以先創建一個無向圖的對稱鄰接矩陣：
        # adj_matrix[source_idx, target_idx] = 1.0
        # adj_matrix[target_idx, source_idx] = 1.0


# 一個更直接從 networkx 獲取鄰接矩陣的方式 (需要確保節點順序)
# adj_matrix_nx = nx.to_numpy_array(causal_graph, nodelist=stock_list, dtype=np.float32)
# 這會生成 A[i,j]=1 表示從 nodelist[i] 到 nodelist[j] 的邊
# 您可能需要 `import networkx as nx`
# 如果 causal_graph.nodes() 的順序已經是 stock_list 的順序，可以簡化
try:
    import networkx as nx
    # 確保 causal_graph 中的節點名稱與 stock_list 中的一致
    # nodelist 參數確保了輸出的鄰接矩陣的行/列順序與 stock_list 一致
    adj_matrix = nx.to_numpy_array(causal_graph, nodelist=stock_list, dtype=np.float32)
    print("成功從 causal_graph 轉換為鄰接矩陣。")
except ImportError:
    print("請安裝 networkx 函式庫 (pip install networkx) 以便將圖轉換為鄰接矩陣。")
    print("或者，您需要手動遍歷 causal_graph.edges() 來構建 adj_matrix。")
    # 手動構建 (如果沒有 networkx 或想更精確控制方向):
    # stock_to_idx = {tic: i for i, tic in enumerate(stock_list)}
    # adj_matrix = np.zeros((num_stocks, num_stocks), dtype=np.float32)
    # for u_tic, v_tic in causal_graph.edges():
    #     u_idx = stock_to_idx.get(u_tic)
    #     v_idx = stock_to_idx.get(v_tic)
    #     if u_idx is not None and v_idx is not None:
    #         adj_matrix[u_idx, v_idx] = 1.0 # A[source, target] = 1 for source -> target
    # print("手動構建鄰接矩陣完畢 (方向: source -> target)。")


# 可選：將矩陣對稱化 (如果GCN期望無向圖或者您希望如此)
# adj_matrix_symmetric = np.maximum(adj_matrix, adj_matrix.T)
# relation_matrix_final = adj_matrix_symmetric

# 或者直接使用PC算法得到的有向圖的鄰接矩陣
relation_matrix_final = adj_matrix

# 可選：加上自環 (每個節點與自身相連)
# np.fill_diagonal(relation_matrix_final, 1)


# --- 4. 驗證並保存 ---
assert relation_matrix_final.shape == (num_stocks, num_stocks), "關係矩陣形狀不符！"
# np.save('./data/TW50/pc_causal_relation.npy', relation_matrix_final)
print(f"基於 PC 演算法的因果關係矩陣已生成 (輸出形狀: {relation_matrix_final.shape})，保存已註釋。")
print("請注意檢查鄰接矩陣的方向性是否符合您的GCN層的期望。")