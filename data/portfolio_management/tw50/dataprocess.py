import os
import pandas as pd
import numpy as np

# 資料夾路徑
original_folder = 'original_data'
new_folder = 'data_processed'
os.makedirs(new_folder, exist_ok=True)

def convert_roc_to_ad(roc_date):
    parts = roc_date.split('/')
    if len(parts) != 3:
        return None
    year = int(parts[0]) + 1911
    month = int(parts[1])
    day = int(parts[2])
    return f"{year:04d}-{month:02d}-{day:02d}"

def clean_numeric_column(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace('+', '').str.strip(), errors='coerce')

def load_and_clean_csv(filepath):
    df = pd.read_csv(filepath)
    numeric_columns = ['成交股數', '成交金額', '開盤價', '最高價', '最低價', '收盤價', '漲跌價差', '成交筆數']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    return df

def calculate_z_scores(series):
    series = pd.to_numeric(series, errors='coerce')
    return (series - series.mean()) / series.std()

def calculate_moving_average_diff(series, window):
    series = pd.to_numeric(series, errors='coerce')
    ma = series.rolling(window=window).mean().fillna(series)
    return (series - ma) / ma

def process_file(input_path, output_path):
    df = load_and_clean_csv(input_path)
    if len(df) < 30:
        print(f"檔案 {input_path} 數據不足 (少於30行)，跳過處理")
        return

    new_df = pd.DataFrame()
    new_df['date'] = df['日期'].apply(convert_roc_to_ad)
    new_df['open'] = df['開盤價']
    new_df['high'] = df['最高價']
    new_df['low'] = df['最低價']
    new_df['close'] = df['收盤價']
    new_df['adjcp'] = df['收盤價']
    new_df['tic'] = df['股票代號']
    new_df['zopen'] = calculate_z_scores(df['開盤價'])
    new_df['zhigh'] = calculate_z_scores(df['最高價'])
    new_df['zlow'] = calculate_z_scores(df['最低價'])
    new_df['zclose'] = calculate_z_scores(df['收盤價'])
    new_df['zadjcp'] = calculate_z_scores(df['收盤價'])
    for w in [5, 10, 15, 20, 25, 30]:
        new_df[f'zd_{w}'] = calculate_moving_average_diff(df['收盤價'], w)

    new_df.to_csv(output_path, index=False)

# 處理所有檔案
for filename in os.listdir(original_folder):
    if filename.endswith('.csv'):
        input_path = os.path.join(original_folder, filename)
        output_path = os.path.join(new_folder, filename)
        process_file(input_path, output_path)
        print(f"已處理檔案: {filename}")

print("所有檔案處理完成!")

# 合併資料
merged_df = pd.DataFrame()
for filename in os.listdir(new_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(new_folder, filename)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        
all_tics = sorted(merged_df['tic'].dropna().unique())
all_dates = sorted(merged_df['date'].dropna().unique())
full_index = pd.MultiIndex.from_product([all_dates, all_tics], names=['date', 'tic'])

# Step 2: 設定 index 並補齊缺失行
merged_df = merged_df.set_index(['date', 'tic']).reindex(full_index)

# Step 3: 前向與後向填補技術指標欄位

fill_cols = ['open', 'high', 'low', 'close', 'adjcp',
             'zopen', 'zhigh', 'zlow', 'zclose', 'zadjcp'] + \
            [col for col in merged_df.columns if col.startswith('zd_')]

merged_df[fill_cols] = merged_df[fill_cols].ffill().bfill()
# merged_df = merged_df.reset_index()
# merged_df = merged_df.groupby('tic', group_keys=False).apply(lambda df: df.iloc[30:]).reset_index(drop=True)

# Step 4: 重置 index 回 dataframe
merged_df = merged_df.reset_index()
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
merged_df.sort_values(by='date', inplace=True)
merged_df = merged_df.iloc[1764:].reset_index(drop=True)
merged_df.to_csv('merged_data.csv', index=False)

print("已整合所有檔案為: merged_data.csv")

# 產生 tw50.csv（依照 tic 排序）
tw50_df = merged_df.copy()
tw50_df['date'] = pd.to_datetime(tw50_df['date'], errors='coerce')

tw50_df.sort_values(by=['tic', 'date'], inplace=True)

tw50_df.reset_index(drop=True, inplace=True)
tw50_df.insert(0, 'time_id', range(len(tw50_df)))

tw50_df.to_csv('tw50.csv', index=False)

# 分割資料集
train_df = merged_df[(merged_df['date'].dt.year >= 2010) & (merged_df['date'].dt.year <= 2022)].copy()
valid_df = merged_df[merged_df['date'].dt.year == 2023].copy()
test_df = merged_df[merged_df['date'].dt.year == 2024].copy()

# 為每筆資料新增 time_id（根據 date 群組）
def add_time_id(df, filename):
    df = df.copy()
    df.loc[:, 'date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(by='date')
    df.loc[:, 'time_id'] = df.groupby('date').ngroup()
    cols = ['time_id'] + [col for col in df.columns if col != 'time_id']
    df = df[cols]
    df.to_csv(filename, index=False)
    return df

train_df = add_time_id(train_df, 'train.csv')
valid_df = add_time_id(valid_df, 'valid.csv')
test_df = add_time_id(test_df, 'test.csv')

# 產生 test_with_label.csv
test_with_label = test_df.copy()
test_with_label.loc[:, 'label'] = 0
start_idx = int(len(test_with_label) * 0.8)
end_idx = int(len(test_with_label) * 0.9)
test_with_label.iloc[start_idx:end_idx, test_with_label.columns.get_loc('label')] = 2  # validation
test_with_label.iloc[end_idx:, test_with_label.columns.get_loc('label')] = 1          # test
test_with_label.to_csv('test_with_label.csv', index=False)

print("所有資料集已完成並輸出！")
