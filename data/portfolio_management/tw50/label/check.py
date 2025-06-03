import pandas as pd
import matplotlib.pyplot as plt

# 載入你的CSV檔案，或直接用DataFrame（此處假設你已讀入為 df）
df = pd.read_csv("/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/2024/test.csv")  # 若是從CSV檔載入
# 或直接用你提供的 dataframe 開始操作

# 確保日期為 datetime 格式
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= pd.to_datetime("2022-01-01")]

# 建立圖表
plt.figure(figsize=(16, 10))

# 依照 tic 分組畫出每支股票的收盤價（adjcp）走勢
for tic in df['tic'].unique():
    stock_df = df[df['tic'] == tic].sort_values(by='date')
    plt.plot(stock_df['date'], stock_df['adjcp'], label=f"{tic}")

# 圖表設定
plt.title("TW50 股票收盤價走勢圖")
plt.xlabel("日期")
plt.ylabel("收盤價（adjcp）")
plt.legend(loc='upper left', fontsize='small', ncol=3)
plt.grid(True)
plt.tight_layout()
# 儲存圖片在當前資料夾
plt.savefig("tw50_adjcp_trend2024.png")

# 不顯示圖
plt.close()