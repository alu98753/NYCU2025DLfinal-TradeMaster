import pandas as pd
import numpy as np
import os
# --- Add these imports for PC Algorithm ---
from cdt.causality.graph import PC 
import networkx as nx # CDT's PC often returns a NetworkX graph object
import time
# ------------------------------------------

# --- Load data into `df`  ---

save_path = "/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/"
df_path = os.path.join(save_path, "train.csv") # Assuming train.csv for causality learning

df = pd.read_csv(df_path)
# Ensure 'date' and 'tic' columns are present.

# --- Define stock_list and date_list (as shown in Prerequisites) ---
all_unique_dates_from_df = sorted(df['date'].unique())
all_unique_tics_from_df = sorted(df['tic'].unique())

stock_list = sorted(df['tic'].unique().tolist())
num_stocks = 49 # Your target number of stocks
if len(stock_list) != num_stocks:
    print(f"Warning: Number of unique tics found ({len(stock_list)}) is {len(stock_list)}, but num_stocks is set to {num_stocks}.")
    print("Ensure your 'train.csv' contains exactly the desired stocks, or adjust 'stock_list' and 'num_stocks' accordingly.")
    assert len(stock_list) == num_stocks, f"Final stock_list length ({len(stock_list)}) must match num_stocks ({num_stocks})."

print(f"Using stock_list for processing (first 5): {stock_list[:5]}, total: {len(stock_list)}")

date_list = all_unique_dates_from_df
num_days = len(date_list)

# --- MODIFICATION 1: Update feature_columns for ASU ---
feature_columns = [
    'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
    'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
]
num_asu_features = len(feature_columns) # 現在應該是 11

print(f"\n--- Preparing stocks_data.npy ({num_stocks} stocks, {num_days} days, {num_asu_features} features) ---")
stocks_data_np = np.zeros((num_stocks, num_days, num_asu_features), dtype=np.float32)
multi_index = pd.MultiIndex.from_product([date_list, stock_list], names=['date', 'tic'])
your_df_indexed = df.set_index(['date', 'tic'])

for feature_idx, feature_name in enumerate(feature_columns):
    feature_series = your_df_indexed[feature_name].reindex(multi_index)
    feature_series_filled = feature_series.ffill().fillna(0) # 保持前向填充和0填充
    feature_df_pivoted = feature_series_filled.unstack(level='tic')
    feature_df_pivoted = feature_df_pivoted[stock_list] # 確保股票順序一致
    stocks_data_np[:, :, feature_idx] = feature_df_pivoted.to_numpy(dtype=np.float32).T

print(f"Shape of stocks_data_np: {stocks_data_np.shape}")
assert stocks_data_np.shape == (num_stocks, num_days, num_asu_features), "Shape mismatch for stocks_data_np!"
assert not np.isnan(stocks_data_np).any(), "NaNs found in stocks_data_np!"
assert not np.isinf(stocks_data_np).any(), "Infs found in stocks_data_np!"

save_data_dir = os.path.join(save_path, "ASU_data_corrected") # 可以用新的目錄名以示區別
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)

np.save(os.path.join(save_data_dir, "stocks_data.npy"), stocks_data_np)
print("stocks_data.npy prepared and saved.")

# --- Preparing ROR.npy ---
print(f"\n--- Preparing ROR.npy ---")
# MODIFICATION 2: ROR calculation needs a raw price column from the original train.csv,
# as `stocks_data_np` no longer contains raw 'adjcp' or 'close'.
# We need to re-extract the chosen price directly from `your_df_indexed`.

price_column_for_ror_calculation = 'adjcp' # 或者 'close'，取決於您在 train.csv 中哪個更可靠
if price_column_for_ror_calculation not in your_df_indexed.columns:
    print(f"Warning: Column '{price_column_for_ror_calculation}' not in input DataFrame. Falling back to 'close'.")
    price_column_for_ror_calculation = 'close'
    if price_column_for_ror_calculation not in your_df_indexed.columns:
        raise ValueError(f"Neither 'adjcp' nor 'close' found in DataFrame columns for ROR calculation.")

# 提取原始價格數據並塑形
price_series_for_ror = your_df_indexed[price_column_for_ror_calculation].reindex(multi_index)
price_series_for_ror_filled = price_series_for_ror.ffill().fillna(0) # 保持填充邏輯
price_df_pivoted_for_ror = price_series_for_ror_filled.unstack(level='tic')
price_df_pivoted_for_ror = price_df_pivoted_for_ror[stock_list] # 確保股票順序
raw_close_prices_for_ror = price_df_pivoted_for_ror.to_numpy(dtype=np.float32).T # Shape: [num_stocks, num_days]

ror_data_np = np.zeros((num_stocks, num_days), dtype=np.float32)
prices_today = raw_close_prices_for_ror[:, 1:]
prices_yesterday = raw_close_prices_for_ror[:, :-1]

daily_returns_slice = np.divide(
    prices_today - prices_yesterday,
    prices_yesterday,
    out=np.zeros_like(prices_today, dtype=np.float32),
    where=prices_yesterday != 0
)
ror_data_np[:, 1:] = daily_returns_slice
ror_data_np[:, 0] = 0.0 # 第一天收益率設為0
ror_data_np = np.nan_to_num(ror_data_np, nan=0.0, posinf=0.0, neginf=0.0) # 處理可能的 NaN/inf

print(f"Shape of ror_data_np: {ror_data_np.shape}")
assert ror_data_np.shape == (num_stocks, num_days), "Shape mismatch for ror_data_np!"
assert not np.isnan(ror_data_np).any(), "NaNs found in ror_data_np!"
assert not np.isinf(ror_data_np).any(), "Infs found in ror_data_np!"
np.save(os.path.join(save_data_dir, "ROR.npy"), ror_data_np)
print("ROR.npy prepared and saved.")


# --- Preparing industry_classification.npy (保持不變) ---
# ... (您的行業分類矩陣生成代碼保持不變) ...
print(f"\n--- Preparing industry_classification.npy ---")
stock_to_industry_map = {
    "1101": "水泥工業", "1216": "食品工業", "1301": "塑膠工業", "1303": "塑膠工業",
    "2002": "鋼鐵工業", "2207": "汽車", "2301": "其他電子業", "2303": "半導體業",
    "2308": "電子零組件", "2317": "其他電子業", "2327": "電子零組件", "2330": "半導體業",
    "2345": "通信網路業", "2357": "電腦及週邊設備業", "2379": "半導體業", "2382": "電腦及週邊設備業",
    "2395": "電腦及週邊設備業", "2412": "通信網路業", "2454": "半導體業", "2603": "航運業",
    "2609": "航運業", "2615": "航運業", "2880": "金控業", "2881": "金控業",
    "2882": "金控業", "2883": "金融", "2884": "金控業", "2885": "金控業",
    "2886": "金控業", "2887": "金融", "2890": "金融保險業", "2891": "金融",
    "2892": "金控業", "2912": "貿易百貨", "3008": "電子元件", "3017": "電腦及週邊設備業",
    "3034": "半導體", "3037": "電子零組件業", "3045": "通信網路業", "3231": "電腦及週邊設備業",
    "3661": "半導體業", "3711": "半導體業", "4904": "通信網路業", "4938": "電腦及週邊設備業",
    "5871": "其他業", "5876": "銀行業", "5880": "金控業", "6505": "油電燃氣業",
    "6669": "電腦及週邊設備業"
}
relation_matrix_industry = np.zeros((num_stocks, num_stocks), dtype=np.float32)
for i in range(num_stocks):
    relation_matrix_industry[i, i] = 1.0
    for j in range(i + 1, num_stocks):
        tic_i = stock_list[i]
        tic_j = stock_list[j]
        industry_i = stock_to_industry_map.get(str(tic_i))
        industry_j = stock_to_industry_map.get(str(tic_j))
        if industry_i is not None and industry_i == industry_j:
            relation_matrix_industry[i, j] = 1.0
            relation_matrix_industry[j, i] = 1.0
np.save(os.path.join(save_data_dir, "industry_classification.npy"), relation_matrix_industry)
print("industry_classification.npy prepared and saved.")


# --- Preparing Causality Matrix (pc_causal_relation.npy) using PC Algorithm ---
# 這部分依賴於 ror_data_np，由於 ror_data_np 的生成邏輯已更新，這裡不需要額外修改，
# 只要確保傳入 PC 算法的 ror_data_for_pc_algo 是正確的即可。
print(f"\n--- Preparing Causality Matrix (pc_causal_relation.npy) using PC Algorithm ---")
ror_data_for_pc_algo = pd.DataFrame(ror_data_np.T, columns=stock_list)
print(f"Data shape for PC algorithm: {ror_data_for_pc_algo.shape}")
start_time = time.time()
try:
    pc_obj = PC(CItest='gaussian', alpha=0.05, verbose=False) # verbose 設為 True 或 False 均可
    print("cdt.PC object initialized successfully.")
    print("Running PC algorithm...")
    causal_graph = pc_obj.predict(ror_data_for_pc_algo)
    print("PC algorithm finished.")
    causality_matrix_np = nx.to_numpy_array(causal_graph, nodelist=stock_list, dtype=np.float32)
    print(f"Shape of PC causality_matrix_np: {causality_matrix_np.shape}")
    assert causality_matrix_np.shape == (num_stocks, num_stocks), "Shape mismatch for PC causality matrix!"
    np.save(os.path.join(save_data_dir, "pc_causal_relation.npy"), causality_matrix_np)
    print("pc_causal_relation.npy prepared and saved.")
except Exception as e:
    print(f"AN UNEXPECTED ERROR occurred during PC algorithm processing: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Finished attempt for PC algorithm causality matrix generation.")

print("\nAll data preparation steps finished. Cost time: ",time.time()-start_time)