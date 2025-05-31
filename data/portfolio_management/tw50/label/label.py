import pandas as pd
import numpy as np

# 1. 讀 CSV，group by date、pivot 出 adjcp
df = pd.read_csv('/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/train.csv', parse_dates=['date'])
df = df.sort_values(['date','tic']).reset_index(drop=True)
adjcp_pivot = df.pivot(index='date', columns='tic', values='adjcp')

# 2. 算每日等權報酬
returns  = adjcp_pivot.pct_change().fillna(0)  # [T,50], idx=日期
R_market = returns.mean(axis=1)                # [T]

# 3. 算 20 日滾動複合報酬
W = 20
roll_return = (
    (1 + R_market)
    .rolling(window=W, min_periods=W)
    .apply(lambda x: np.prod(1 + x) - 1.0, raw=True)
).fillna(0)  # 前 W-1 天可能是 NaN

# 4. 把 R_market 轉成等權「累積指數」
#    這裡假設第一個有效日的 index = 100.0 (純為了看畫面比較直觀)
cum_index   = (1 + R_market).cumprod() * 100.0

# 5. 算 20 日內的「高點」→「跌幅」
rolling_max = cum_index.rolling(window=W, min_periods=W).max()
drawdown    = 1.0 - (cum_index / rolling_max)
drawdown    = drawdown.fillna(0)  # 前幾天補 0

# 6. 設置門檻，打上 regime label
#    0 = Sideways, 1 = Bull, 2 = Bear
theta_up   = 0.08   # 20 日漲超 5% 以上看為牛市
theta_down = -0.03  # 20 日跌超 -5% 以上看為熊市
dd_thresh  = 0.03   # 20 日內任意回撤 > 10% 視為熊市
dd_safe    = 0.03   # 如果 20 日內沒跌破 7%，利於判斷牛市

regime = pd.Series(data=np.zeros_like(roll_return.values, dtype=int), index=roll_return.index)

# 一開始先把「Bear (2)」劃好
regime[ (roll_return < theta_down) | (drawdown > dd_thresh) ] = 2

# 再把「Bull (1)」：必須 > +0.05 且 drawdown < 0.07
regime[ (roll_return > theta_up) & (drawdown < dd_safe) ] = 1

# 剩下皆為 0 (Sideways)
# regime 已經預設為 0，故不需再特別標

# 7. 把 regime 放回 df，再存新檔
regime_df = pd.DataFrame({
    'date':      roll_return.index,
    'R_market':  R_market.values,
    'roll_ret':  roll_return.values,
    'drawdown':  drawdown.values,
    'regime':    regime.values.astype(int)
})
df_labeled = pd.merge(df, regime_df[['date','regime']], on='date', how='left')
df_labeled.to_csv('/home/asiadragon/Desktop/zi/NYCU2025DLfinal-TradeMaster/data/portfolio_management/tw50/label/tw50_all_with_regime_A.csv', index=False)
# 讀回標完號的檔案，檢查 2010-2022 之間各 regime 分佈
df2 = df_labeled
# 由於每個 'date' 底下有 50 筆股票，我們只取『某一天的第一支股票』來代表該日 regime
# 也可以先 drop_duplicates，只保留 date+regime 的第一筆
df_unique = df2.drop_duplicates(subset=['date'])  # 每日只留 1 筆，代表當日 regime
counts = df_unique['regime'].value_counts().sort_index()
print("Regime counts (0,1,2) = ", counts.values)
# 應該會看到 0、1、2 各有合理樣本(例如 30% / 40% / 30%)，視參數調整
