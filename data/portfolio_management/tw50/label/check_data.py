import pandas as pd
import matplotlib.pyplot as plt

# 載入CSV
df = pd.read_csv('/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/train.csv')
# print(df)
# 確保日期為datetime格式

# 對每支股票的adjcp進行繪圖
plt.figure(figsize=(16, 10))

# 排除 tic == 3008
for tic in df['tic'].unique():
    if tic == 3008:
        continue
    stock_df = df[df['tic'] == tic]
    plt.plot(stock_df['date'], stock_df['adjcp'], label=str(tic), linewidth=1)

plt.title('Stock Adjusted Close Price Trends')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend(loc='upper left', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/stock_plot.png')
