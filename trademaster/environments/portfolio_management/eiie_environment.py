from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import numpy as np
from trademaster.utils import get_attr, print_metrics
import pandas as pd
from ..custom import Environments
from ..builder import ENVIRONMENTS
from gym import spaces
from collections import OrderedDict
import pickle
import os.path as osp


@ENVIRONMENTS.register_module()
class PortfolioManagementEIIEEnvironment(Environments):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIEEnvironment, self).__init__()

        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")
        self.test_dynamic=int(get_attr(kwargs, "test_dynamic", "-1"))
        self.task_index = int(get_attr(kwargs, "task_index", "-1"))
        self.work_dir = get_attr(kwargs, "work_dir", "")
        self.time_steps = get_attr(self.dataset, "time_steps", 10)
        

        self.df_path = None
        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)



        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        # Load DataFrame
        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            df = pd.read_csv(dynamics_test_path)
            self.start_date_str = df['date'].iloc[0]
            self.end_date_str = df['date'].iloc[-1]
        else:
            df = pd.read_csv(self.df_path)
        
        # Store the original df only if an exact copy is needed elsewhere
        # self.df_raw = df.copy()
        start_date_filter = str(get_attr(kwargs, "start_date_filter", "2020-01-01")) # 從 kwargs 讀取，預設為不篩選或早期日期
        if start_date_filter:
            df['date'] = pd.to_datetime(df['date']) # 確保 date 列是 datetime 對象
            df = df[df['date'] >= pd.to_datetime(start_date_filter)]
            df['date'] = df['date'].dt.strftime('%Y-%m-%d') # 轉換回字串格式，如果後續代碼需要
            if df.empty:
                raise ValueError(f"No data remaining after filtering with start_date_filter: {start_date_filter} for {self.df_path}")
        # --- NumPy Data Pre-processing ---
        self.unique_dates = sorted(df['date'].unique())
        self.unique_tics = sorted(df['tic'].unique())

        self.stock_dim = len(self.unique_tics)
        self.num_tech_indicators = len(self.tech_indicator_list)
        self.num_unique_dates = len(self.unique_dates)

        self.date_to_idx = {date: i for i, date in enumerate(self.unique_dates)}
        self.tic_to_idx = {tic: i for i, tic in enumerate(self.unique_tics)}

        self.data_cube = np.zeros((self.num_unique_dates, self.stock_dim, self.num_tech_indicators), dtype=np.float32) # (num_dates, num_stocks, num_tech_indicators)
        self.close_prices_cube = np.zeros((self.num_unique_dates, self.stock_dim), dtype=np.float32) # (num_dates, num_stocks)

        # Pivot and fill data_cube
        multi_index = pd.MultiIndex.from_product([self.unique_dates, self.unique_tics], names=['date', 'tic'])
        
        # Process technical indicators
        df_for_pivot = df.set_index(['date', 'tic'])
        for i, tech in enumerate(self.tech_indicator_list):
            tech_series = df_for_pivot[tech].reindex(multi_index)
            tech_series_filled = tech_series.ffill().bfill() # Fill within each group first if possible, then globally
            tech_series_filled = tech_series_filled.fillna(0) # Final fill for any remaining NaNs (e.g., stock starts late)
            self.data_cube[:, :, i] = tech_series_filled.unstack(level='tic')[self.unique_tics].to_numpy(dtype=np.float32)
            
        # Process close prices
        close_series = df_for_pivot['close'].reindex(multi_index)
        close_series_filled = close_series.ffill().bfill()
        close_series_filled = close_series_filled.fillna(0) # Should be cautious with filling close price with 0
        self.close_prices_cube = close_series_filled.unstack(level='tic')[self.unique_tics].to_numpy(dtype=np.float32)

        self.state_space_shape = self.stock_dim 
        self.action_space_shape = self.stock_dim 
        
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.action_space_shape,))
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,
            shape=(self.stock_dim, self.time_steps,self.num_tech_indicators) )

        self.action_dim = self.action_space.shape[0] # stock_dim
        self.state_dim = self.num_tech_indicators # self.observation_space.shape[0] was num_features

        self.day_idx = self.time_steps - 1 # Current day index in unique_dates

        # Initial state calculation
        start_slice_idx = self.day_idx - self.time_steps + 1
        end_slice_idx = self.day_idx + 1

        # Ensure slices are within bounds, pad if necessary for initial steps
        actual_start_idx = max(0, start_slice_idx)
        state_window_data = self.data_cube[actual_start_idx:end_slice_idx, :, :] # (slice_len, stock_dim, num_features)

        if state_window_data.shape[0] < self.time_steps: # If not enough history at the beginning
            padding_needed = self.time_steps - state_window_data.shape[0]
            padding_array = np.zeros((padding_needed, self.stock_dim, self.num_tech_indicators), dtype=np.float32)
            state_window_data = np.concatenate((padding_array, state_window_data), axis=0)
        
        # Reshape to (stock_dim, num_features, time_steps) then transpose to (stock_dim, time_steps, num_features)
        raw_state_format = np.transpose(state_window_data, (1, 2, 0)) # (stock_dim, num_features, time_steps)
        self.state = np.transpose(raw_state_format, (0, 2, 1))       # (stock_dim, time_steps, num_features)

        self.terminal = False
        self.portfolio_value = float(self.initial_amount)
        self.asset_memory = [float(self.initial_amount)]
        self.portfolio_return_memory = [0.0]
        self.weights_memory = [[1.0] + [0.0] * self.stock_dim] 
        self.date_memory = [self.unique_dates[self.day_idx]]
        self.transaction_cost_memory = [] 
        self.test_id = 'agent'

    def reset(self):
        self.day_idx = self.time_steps - 1

        start_slice_idx = self.day_idx - self.time_steps + 1
        end_slice_idx = self.day_idx + 1
        
        actual_start_idx = max(0, start_slice_idx)
        state_window_data = self.data_cube[actual_start_idx:end_slice_idx, :, :]

        if state_window_data.shape[0] < self.time_steps:
            padding_needed = self.time_steps - state_window_data.shape[0]
            padding_array = np.zeros((padding_needed, self.stock_dim, self.num_tech_indicators), dtype=np.float32)
            state_window_data = np.concatenate((padding_array, state_window_data), axis=0)

        raw_state_format = np.transpose(state_window_data, (1, 2, 0)) # (stock_dim, num_features, time_steps)
        self.state = np.transpose(raw_state_format, (0, 2, 1))       # (stock_dim, time_steps, num_features)

        self.terminal = False
        self.portfolio_value = float(self.initial_amount)
        self.asset_memory = [float(self.initial_amount)]
        self.portfolio_return_memory = [0.0]
        # Original reset weights: [[1 / (self.stock_dim + 1)] * (self.stock_dim + 1)]
        initial_weight_val = 1.0 / (self.stock_dim + 1)
        self.weights_memory = [[initial_weight_val] * (self.stock_dim + 1)]
        self.date_memory = [self.unique_dates[self.day_idx]]
        self.transaction_cost_memory = []
        
        return self.state.astype(np.float32)

    def step(self, weights: np.ndarray):
        weights = np.asarray(weights, dtype=np.float32)

        self.terminal = self.day_idx >= self.num_unique_dates - 1

        if self.terminal:
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date_str} to {self.end_date_str}')
            
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            stats = OrderedDict(
                {
                    "Total Return": ["{:04f}%".format(tr * 100)],
                    "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                    "Volatility": ["{:04f}%".format(vol * 100)],
                    "Max Drawdown": ["{:04f}%".format(mdd * 100)],
                }
            )
            table = print_metrics(stats)
            print(table)

            df_return = self.save_portfolio_return_memory()
            daily_return_values = df_return.daily_return.values
            df_value = self.save_asset_memory()
            assets_values = df_value["total assets"].values

            save_dict = OrderedDict(
                {
                    "Profit Margin": tr * 100,
                    "Excess Profit": tr * 100 - 0,
                    "daily_return": daily_return_values,
                    "total_assets": assets_values
                }
            )
            metric_save_path = osp.join(self.work_dir, f'metric_{self.task}_{self.test_dynamic}_{self.test_id}_{self.task_index}.pickle')
            if self.task == 'test_dynamic': 
                with open(metric_save_path, 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # For terminal state, return the last valid state that led to termination.
            # The current self.state is for day_idx, which is the terminal day.
            return self.state.astype(np.float32), 0.0, self.terminal, {"sharpe_ratio": sharpe_ratio, "total_assets": assets_values}

        else:
            self.weights_memory.append(weights.tolist()) # Store agent's target weights for this step

            last_day_close_prices_slice = self.close_prices_cube[self.day_idx, :] # Prices at current self.day_idx (t)

            self.day_idx += 1 # Move to next day (t+1)

            # New state for day t+1
            start_slice_idx = self.day_idx - self.time_steps + 1
            end_slice_idx = self.day_idx + 1
            actual_start_idx = max(0, start_slice_idx)
            state_window_data = self.data_cube[actual_start_idx:end_slice_idx, :, :]

            if state_window_data.shape[0] < self.time_steps:
                padding_needed = self.time_steps - state_window_data.shape[0]
                padding_array = np.zeros((padding_needed, self.stock_dim, self.num_tech_indicators), dtype=np.float32)
                state_window_data = np.concatenate((padding_array, state_window_data), axis=0)
            
            raw_state_format = np.transpose(state_window_data, (1, 2, 0))
            self.state = np.transpose(raw_state_format, (0, 2, 1))

            current_day_close_prices_slice = self.close_prices_cube[self.day_idx, :] # Prices at new self.day_idx (t+1)

            portfolio_weights_stocks = weights[1:] # Stock weights (excluding cash weight at index 0)
            
            # Price ratios from t to t+1.
            price_ratios = np.divide(current_day_close_prices_slice, last_day_close_prices_slice,
                                     out=np.ones_like(current_day_close_prices_slice, dtype=np.float32),
                                     where=last_day_close_prices_slice != 0)
            
            gross_portfolio_return = np.sum((price_ratios - 1) * portfolio_weights_stocks)

            new_stock_weights_after_price_change = portfolio_weights_stocks * price_ratios
            weights_after_price_change = np.concatenate(([weights[0]], new_stock_weights_after_price_change))
            weights_brandnew = self.normalization(weights_after_price_change)
            self.weights_memory.append(weights_brandnew.tolist()) # weights_memory has: ..., W_brandnew_prev, W_agent_curr (weights), W_brandnew_curr

            # Transaction cost calculation 
            weights_old_tx = np.array(self.weights_memory[-3], dtype=np.float32) # W_brandnew_prev
            weights_new_tx = np.array(self.weights_memory[-2], dtype=np.float32) # W_agent_curr
            diff_weights = np.sum(np.abs(weights_old_tx - weights_new_tx))

            transcationfee = diff_weights * self.transaction_cost_pct * self.portfolio_value
            
            value_after_cost_and_return = (self.portfolio_value - transcationfee) * (1 + gross_portfolio_return)
            
            if self.portfolio_value == 0: # Avoid division by zero for net_portfolio_return
                 net_portfolio_return = 0.0 if value_after_cost_and_return == 0 else np.inf # Or some large number
            else:
                net_portfolio_return = (value_after_cost_and_return - self.portfolio_value) / self.portfolio_value
            
            # Reward: log return, with safe guards for log(0) or log(<0)
            if value_after_cost_and_return > 1e-9 and self.portfolio_value > 1e-9: # Use small epsilon
                self.reward = np.log(value_after_cost_and_return / self.portfolio_value)
            elif value_after_cost_and_return > 1e-9 and self.portfolio_value <= 1e-9:
                self.reward = np.log(value_after_cost_and_return / 1e-9) # Large positive reward
            elif value_after_cost_and_return <= 1e-9 and self.portfolio_value > 1e-9:
                self.reward = np.log(1e-9 / self.portfolio_value) # Large negative reward
            else: # both are very small or zero
                self.reward = 0.0
            
            self.portfolio_value = value_after_cost_and_return

            self.portfolio_return_memory.append(net_portfolio_return)
            self.date_memory.append(self.unique_dates[self.day_idx])
            self.asset_memory.append(self.portfolio_value)

            return self.state.astype(np.float32), float(self.reward), self.terminal, {"weights_brandnew": weights_brandnew.tolist()}

    def normalization(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        s = np.sum(actions)
        if np.abs(s) < 1e-9: # Check if sum is close to zero
            if not np.any(actions): # All actions are zero
                return actions 
            # If actions are not all zero but sum to zero (e.g., [1, -1]), return NaNs
            return np.full_like(actions, np.nan, dtype=np.float32)
        return actions / s

    def save_portfolio_return_memory(self) -> pd.DataFrame:
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        if not df_date.empty: # Check if df_date is not empty before setting index
             df_return.index = df_date.date
        return df_return

    def save_asset_memory(self) -> pd.DataFrame:
        # a record of asset values for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        if not df_date.empty: # Check if df_date is not empty
            df_value.index = df_date.date
        return df_value

    def analysis_result(self) -> tuple[float, float, float, float, float, float]:
        df_return = self.save_portfolio_return_memory()
        df_value = self.save_asset_memory()
        
        df_eval = pd.DataFrame()
        # Ensure columns exist before assigning, especially if memory lists are empty
        df_eval["daily_return"] = df_return["daily_return"] if "daily_return" in df_return else pd.Series(dtype=np.float64)
        df_eval["total assets"] = df_value["total assets"] if "total assets" in df_value else pd.Series(dtype=np.float64)
        
        if df_eval.empty or df_eval["total assets"].empty: # Handle empty data case
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        return self.evaualte(df_eval)

    def get_daily_return_rate(self, price_list: np.ndarray | list[float]) -> list[float]:
        price_array = np.asarray(price_list, dtype=np.float32)
        if len(price_array) < 2:
            return []
        safe_denominator = np.where(price_array[:-1] == 0, 1e-9, price_array[:-1]) # Replace 0 with small number
        return_rates = (price_array[1:] / safe_denominator) - 1
        return return_rates.tolist()
        
    def evaualte(self, df: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
        if df.empty or df["daily_return"].empty or df["total assets"].empty:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # Default values for empty df

        daily_return_series = df["daily_return"]
        total_assets_series = df["total assets"]

        neg_ret_series = daily_return_series[daily_return_series < 0]
        
        # Total return
        initial_assets = total_assets_series.iloc[0]
        final_assets = total_assets_series.iloc[-1]
        tr = final_assets / (initial_assets + 1e-10) - 1 
        return_rate_list_for_sharpe_vol = self.get_daily_return_rate(total_assets_series.values)

        if not return_rate_list_for_sharpe_vol:
            sharpe_ratio = 0.0
            vol = 0.0
        else:
            mean_return = np.mean(return_rate_list_for_sharpe_vol)
            std_return = np.std(return_rate_list_for_sharpe_vol)
            sharpe_ratio = mean_return * (252 ** 0.5) / (std_return + 1e-10)
            vol = std_return
        
        # Max Drawdown
        mdd = 0.0
        if not total_assets_series.empty:
            peak = total_assets_series.iloc[0]
            for value in total_assets_series:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 1e-9 else 0.0 
                if dd > mdd:
                    mdd = dd
        
        sum_daily_returns = np.sum(daily_return_series.values)
        
        # Calmar Ratio
        cr = sum_daily_returns / (mdd + 1e-10) if (mdd > 1e-9 or sum_daily_returns !=0) else 0.0 # Avoid 0/0
        
        # Sortino Ratio
        std_neg_ret = np.std(neg_ret_series.values) if not neg_ret_series.empty else 0.0
        len_daily_return_sqrt = np.sqrt(len(daily_return_series)) if len(daily_return_series) > 0 else 0.0
        
        denominator_sor = (np.nan_to_num(std_neg_ret, nan=0.0) + 1e-10) * (len_daily_return_sqrt + 1e-10)
        sor = sum_daily_returns / denominator_sor if denominator_sor > 1e-9 else 0.0
        
        return tr, sharpe_ratio, vol, mdd, cr, sor