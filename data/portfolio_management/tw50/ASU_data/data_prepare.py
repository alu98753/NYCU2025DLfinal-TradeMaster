import pandas as pd
import numpy as np
import os
# --- Add these imports for PC Algorithm ---
from cdt.causality.graph import PC
import networkx as nx # CDT's PC often returns a NetworkX graph object
# ------------------------------------------

# --- Load your data into `your_df` here ---

save_path = "/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/"
df_path = os.path.join(save_path, "train.csv") # Assuming train.csv for causality learning

your_df = pd.read_csv(df_path)
# Ensure 'date' and 'tic' columns are present.

# --- Define stock_list and date_list (as shown in Prerequisites) ---
all_unique_dates_from_df = sorted(your_df['date'].unique())
all_unique_tics_from_df = sorted(your_df['tic'].unique())

# !!! IMPORTANT: Ensure stock_list is the definitive, sorted list of 49 stocks !!!
stock_list = sorted(your_df['tic'].unique().tolist()) # Using all unique tics from train.csv
num_stocks = 49 # Your target number of stocks
if len(stock_list) != num_stocks:
    print(f"Warning: Number of unique tics found ({len(stock_list)}) is {len(stock_list)}, but num_stocks is set to {num_stocks}.")
    print("Ensure your 'train.csv' contains exactly the desired stocks, or adjust 'stock_list' and 'num_stocks' accordingly.")
    # Example: if you need to filter to a predefined list of 49
    # predefined_49_tics = [...] # your list of 49 tics
    # stock_list = sorted([tic for tic in predefined_49_tics if tic in all_unique_tics_from_df])
    # num_stocks = len(stock_list)
    # your_df = your_df[your_df['tic'].isin(stock_list)] # Filter DataFrame
    # all_unique_dates_from_df = sorted(your_df['date'].unique()) # Re-evaluate dates if df is filtered
    assert len(stock_list) == num_stocks, f"Final stock_list length ({len(stock_list)}) must match num_stocks ({num_stocks})."


print(f"Using stock_list for processing (first 5): {stock_list[:5]}, total: {len(stock_list)}")

date_list = all_unique_dates_from_df
num_days = len(date_list)

feature_columns = [
    'open', 'high', 'low', 'close', 'adjcp',
    'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
    'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
]
num_asu_features = len(feature_columns)

print(f"\n--- Preparing stocks_data.npy ({num_stocks} stocks, {num_days} days, {num_asu_features} features) ---")
stocks_data_np = np.zeros((num_stocks, num_days, num_asu_features), dtype=np.float32)
multi_index = pd.MultiIndex.from_product([date_list, stock_list], names=['date', 'tic'])
your_df_indexed = your_df.set_index(['date', 'tic'])

for feature_idx, feature_name in enumerate(feature_columns):
    # print(f"Processing feature: {feature_name}") # Can be verbose, commenting out
    feature_series = your_df_indexed[feature_name].reindex(multi_index)
    feature_series_filled = feature_series.ffill().fillna(0)
    feature_df_pivoted = feature_series_filled.unstack(level='tic')
    feature_df_pivoted = feature_df_pivoted[stock_list]
    stocks_data_np[:, :, feature_idx] = feature_df_pivoted.to_numpy(dtype=np.float32).T

print(f"Shape of stocks_data_np: {stocks_data_np.shape}")
assert stocks_data_np.shape == (num_stocks, num_days, num_asu_features), "Shape mismatch for stocks_data_np!"
assert not np.isnan(stocks_data_np).any(), "NaNs found in stocks_data_np!"
assert not np.isinf(stocks_data_np).any(), "Infs found in stocks_data_np!"

save_data_dir = os.path.join(save_path, "ASU_data") # Renamed variable for clarity
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)

np.save(os.path.join(save_data_dir, "stocks_data.npy"), stocks_data_np)
print("stocks_data.npy prepared and saved.")

# --- Preparing ROR.npy ---
print(f"\n--- Preparing ROR.npy ---")
try:
    price_feature_name_for_ror = 'adjcp'
    price_feature_index = feature_columns.index(price_feature_name_for_ror)
except ValueError:
    print(f"Warning: '{price_feature_name_for_ror}' not in feature_columns. Falling back to 'close'.")
    price_feature_name_for_ror = 'close'
    price_feature_index = feature_columns.index(price_feature_name_for_ror)

close_prices_for_ror = stocks_data_np[:, :, price_feature_index]
ror_data_np = np.zeros((num_stocks, num_days), dtype=np.float32)
prices_today = close_prices_for_ror[:, 1:]
prices_yesterday = close_prices_for_ror[:, :-1]

daily_returns_slice = np.divide(
    prices_today - prices_yesterday,
    prices_yesterday,
    out=np.zeros_like(prices_today, dtype=np.float32),
    where=prices_yesterday != 0
)
ror_data_np[:, 1:] = daily_returns_slice
ror_data_np[:, 0] = 0.0
ror_data_np = np.nan_to_num(ror_data_np, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Shape of ror_data_np: {ror_data_np.shape}")
assert ror_data_np.shape == (num_stocks, num_days), "Shape mismatch for ror_data_np!"
assert not np.isnan(ror_data_np).any(), "NaNs found in ror_data_np!"
assert not np.isinf(ror_data_np).any(), "Infs found in ror_data_np!"
np.save(os.path.join(save_data_dir, "ROR.npy"), ror_data_np)
print("ROR.npy prepared and saved.")


# --- Preparing industry_classification.npy (as you had it) ---
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
# Ensure your stock_list only contains keys present in stock_to_industry_map if using this
# Or handle missing keys in the map

relation_matrix_industry = np.zeros((num_stocks, num_stocks), dtype=np.float32)
for i in range(num_stocks):
    relation_matrix_industry[i, i] = 1.0
    for j in range(i + 1, num_stocks):
        tic_i = stock_list[i]
        tic_j = stock_list[j]
        # Check if tics are in the map before accessing
        industry_i = stock_to_industry_map.get(str(tic_i)) # Convert tic to string if map keys are strings
        industry_j = stock_to_industry_map.get(str(tic_j))
        if industry_i is not None and industry_i == industry_j:
            relation_matrix_industry[i, j] = 1.0
            relation_matrix_industry[j, i] = 1.0

print(f"Shape of relation_matrix_industry: {relation_matrix_industry.shape}")
assert relation_matrix_industry.shape == (num_stocks, num_stocks), "Shape mismatch for industry relation matrix!"
np.save(os.path.join(save_data_dir, "industry_classification.npy"), relation_matrix_industry)
print("industry_classification.npy prepared and saved.")


# --- Preparing Causality Matrix using PC Algorithm (replaces previous relation matrix logic or is an alternative) ---
print(f"\n--- Preparing Causality Matrix (relation_file.npy) using PC Algorithm ---")

# 1. Prepare ROR data for PC algorithm input
# PC algorithm in CDT usually expects data in shape [num_samples, num_variables]
# which means [num_days, num_stocks] for us.
# ror_data_np is currently [num_stocks, num_days]. We need to transpose it.
ror_data_for_pc_algo = pd.DataFrame(ror_data_np.T, columns=stock_list)
# Note: PC algorithm is typically run on training data.
# If your ror_data_np covers more than training, you might want to slice it:
# train_period_ror_for_pc_algo = ror_data_for_pc_algo.loc[train_start_date_idx:train_end_date_idx] 
# For this script, we'll use all available ROR data from train.csv.

print(f"Data shape for PC algorithm: {ror_data_for_pc_algo.shape}") # Expected [num_days, num_stocks]

# 2. Initialize and run PC algorithm
# You might need to install cdt: pip install cdt
# And potentially setup R and pcalg if cdt's PC uses it as a backend.
# Check CDT documentation for installation details.
try:
    from cdt.causality.graph import PC
    import networkx as nx
    print("Python check: Successfully imported cdt.PC and networkx.")

    # 初始化 PC  (這一步會觸發 R 依賴檢查)

    # from  /home/asiadragon/miniconda3/envs/TradeMaster/lib/python3.9/site-packages/cdt/causality/graph/PC.py
    pc_obj = PC(CItest='gaussian', alpha=0.05, verbose=True)

    print("cdt.PC object initialized successfully.")
    
    print("Running PC algorithm...")
    causal_graph = pc_obj.predict(ror_data_for_pc_algo)
    print("PC algorithm finished.")

    causality_matrix_np = nx.to_numpy_array(causal_graph, nodelist=stock_list, dtype=np.float32)
    print(f"Shape of PC causality_matrix_np: {causality_matrix_np.shape}")
    assert causality_matrix_np.shape == (num_stocks, num_stocks), "Shape mismatch!"

    np.save(os.path.join(save_data_dir, "pc_causal_relation.npy"), causality_matrix_np)
    print("pc_causal_relation.npy prepared and saved.")

except ModuleNotFoundError as mnfe: # 捕捉 Python 模塊找不到的錯誤
    print(f"PYTHON MODULE NOT FOUND ERROR: {mnfe}")
    print("Please ensure 'cdt' and 'networkx' are installed in your Python environment (e.g., pip install cdt networkx or via conda).")
except ImportError as ie: # 更可能捕捉到 cdt 內部因 R 依賴問題拋出的 ImportError
    print(f"CDT IMPORT ERROR (likely R related): {ie}")
    print("This error usually means that CDT's PC algorithm could not initialize due to missing R dependencies.")
    print("Please ensure R is installed AND the following R packages are correctly installed in your R environment:")
    print("1. pcalg (from CRAN: install.packages('pcalg'))")
    print("2. kpcalg (you mentioned having kpcalg_1.0.1.tar.gz, ensure it was installed correctly in R)")
    print("3. RCIT (from GitHub: devtools::install_github('Diviyan-Kalainathan/RCIT'))")
except Exception as e: # 捕捉其他所有預期之外的錯誤
    print(f"AN UNEXPECTED ERROR occurred during PC algorithm processing: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Finished attempt for PC algorithm causality matrix generation.")

print("\nAll data preparation steps finished.")

print("\nAll data preparation steps finished.")