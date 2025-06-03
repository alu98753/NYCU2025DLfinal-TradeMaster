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
import os
import time

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
        
        # <<< MODIFICATION START: Add rebalance_interval and related attributes >>>
        self.rebalance_interval = int(get_attr(kwargs, "rebalance_interval", 7))
        if self.rebalance_interval <= 0:
            raise ValueError("rebalance_interval must be a positive integer.")
        self.days_since_last_rebalance = 0 # Counter for days since last rebalance
        # <<< MODIFICATION END >>>

        self.df_path = None
        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)

        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.000)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            df = pd.read_csv(dynamics_test_path)
            self.start_date_str = df['date'].iloc[0]
            self.end_date_str = df['date'].iloc[-1]
        else:
            df = pd.read_csv(self.df_path)
        
        start_date_filter = str(get_attr(kwargs, "start_date_filter", "2020-01-01"))
        if start_date_filter:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= pd.to_datetime(start_date_filter)]
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            if df.empty:
                raise ValueError(f"No data remaining after filtering with start_date_filter: {start_date_filter} for {self.df_path}")

        self.unique_dates = sorted(df['date'].unique())
        self.unique_tics = sorted(df['tic'].unique())

        self.stock_dim = len(self.unique_tics)
        self.num_tech_indicators = len(self.tech_indicator_list)
        self.num_unique_dates = len(self.unique_dates)

        self.date_to_idx = {date: i for i, date in enumerate(self.unique_dates)}
        self.tic_to_idx = {tic: i for i, tic in enumerate(self.unique_tics)}

        self.data_cube = np.zeros((self.num_unique_dates, self.stock_dim, self.num_tech_indicators), dtype=np.float32)
        self.close_prices_cube = np.zeros((self.num_unique_dates, self.stock_dim), dtype=np.float32)

        multi_index = pd.MultiIndex.from_product([self.unique_dates, self.unique_tics], names=['date', 'tic'])
        
        df_for_pivot = df.set_index(['date', 'tic'])
        for i, tech in enumerate(self.tech_indicator_list):
            tech_series = df_for_pivot[tech].reindex(multi_index)
            tech_series_filled = tech_series.ffill().bfill()
            tech_series_filled = tech_series_filled.fillna(0)
            self.data_cube[:, :, i] = tech_series_filled.unstack(level='tic')[self.unique_tics].to_numpy(dtype=np.float32)
            
        close_series = df_for_pivot['close'].reindex(multi_index)
        close_series_filled = close_series.ffill().bfill()
        close_series_filled = close_series_filled.fillna(0)
        self.close_prices_cube = close_series_filled.unstack(level='tic')[self.unique_tics].to_numpy(dtype=np.float32)
        
        if 'regime' not in df_for_pivot.columns:
            raise ValueError(f"Column 'regime' not found in DataFrame columns: {df_for_pivot.columns.tolist()} from path {self.df_path}. This column is required for MarketNet calibration.")
        try:
            regime_series_pivoted = df_for_pivot['regime'].unstack(level='tic') # Shape: [num_unique_dates, num_unique_tics]
            # 假設當天所有股票的regime相同，取第一列即可
            if not regime_series_pivoted.empty:
                 self.regime_by_date = regime_series_pivoted[self.unique_tics[0]].reindex(self.unique_dates).ffill().bfill().to_numpy(dtype=np.int64)
            else: # 如果數據集很小或特殊情況
                 self.regime_by_date = np.zeros(self.num_unique_dates, dtype=np.int64) # 默認為0 (牛市)
                 print("Warning: regime_series_pivoted was empty. Defaulting self.regime_by_date to all zeros.")

            if len(self.regime_by_date) != self.num_unique_dates:
                 raise ValueError(f"Length of regime_by_date ({len(self.regime_by_date)}) does not match num_unique_dates ({self.num_unique_dates}).")
            print(f"Successfully loaded 'regime_by_date'. Example values: {self.regime_by_date[:5]}")
        except Exception as e:
            print(f"Error processing 'regime' column: {e}")
            print("Ensure 'regime' column exists and is consistent per date. Defaulting self.regime_by_date to zeros.")
            self.regime_by_date = np.zeros(self.num_unique_dates, dtype=np.int64) # 備用方案：默認為0 (牛市)
        # <<< 新增結束 >>>        
        self.state_space_shape = self.stock_dim 
        self.action_space_shape = self.stock_dim 
        
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.action_space_shape,))

        self.action_dim = self.action_space.shape[0]
        # <<< MODIFICATION START: Update state_dim for new features >>>
        self.original_feature_dim = self.num_tech_indicators
        self.state_dim = self.original_feature_dim + 2 # Add 2 for prev_stock_weight and prev_cash_weight
        
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,
            shape=(self.stock_dim, self.time_steps,self.state_dim) )

        # <<< MODIFICATION END >>>
        self.day_idx = self.time_steps - 1 

        start_slice_idx = self.day_idx - self.time_steps + 1
        end_slice_idx = self.day_idx + 1
        actual_start_idx = max(0, start_slice_idx)
        state_window_data = self.data_cube[actual_start_idx:end_slice_idx, :, :]
        if state_window_data.shape[0] < self.time_steps:
            padding_needed = self.time_steps - state_window_data.shape[0]
            padding_array = np.zeros((padding_needed, self.stock_dim, self.num_tech_indicators), dtype=np.float32)
            state_window_data = np.concatenate((padding_array, state_window_data), axis=0)
        
        
        # <<< MODIFICATION START: Initialize active_target_weights >>>
        # These are the weights that are "live" in the market.
        # They are set at rebalancing and then drift with market prices.
        initial_active_weight_val = 1.0 / (self.stock_dim + 1) # Default to equal weight if not rebalancing on day 0
        self.active_target_weights = np.array([initial_active_weight_val] * (self.stock_dim + 1), dtype=np.float32)
        self.weights_memory = [self.active_target_weights.tolist()] # Store the initial active weights
        # <<< MODIFICATION END >>>
        
        raw_state_format = np.transpose(state_window_data, (1, 2, 0))
        # self.state = np.transpose(raw_state_format, (0, 2, 1))
        self.state = self._get_augmented_state() # Use a helper function

        self.terminal = False
        self.portfolio_value = float(self.initial_amount)
        self.asset_memory = [float(self.initial_amount)]
        self.portfolio_return_memory = [0.0]

        self.date_memory = [self.unique_dates[self.day_idx]]
        self.transaction_cost_memory = [] 
        self.test_id = 'agent'
        
        # log file
        self.txt = "work_dir/portfolio_management_tw50_eiie_eiie_adam_mse/metrics.txt"
        os.makedirs(os.path.dirname(self.txt), exist_ok=True)
        self.empty = True        
                

    def _get_augmented_state(self) -> np.ndarray:
        # Get original market features for the window
        start_slice_idx = self.day_idx - self.time_steps + 1
        end_slice_idx = self.day_idx + 1 # slice up to current day_idx
        actual_start_idx = max(0, start_slice_idx)
        
        # state_window_data_original: [current_slice_len, N, F_original]
        state_window_data_original = self.data_cube[actual_start_idx:end_slice_idx, :, :]

        # Pad if history is shorter than time_steps
        if state_window_data_original.shape[0] < self.time_steps:
            padding_needed = self.time_steps - state_window_data_original.shape[0]
            padding_array = np.zeros(
                (padding_needed, self.stock_dim, self.original_feature_dim), # Use original_feature_dim
                dtype=np.float32
            )
            state_window_data_original = np.concatenate((padding_array, state_window_data_original), axis=0)
        # state_window_data_original is now [T, N, F_original]

        # Prepare w_{t-1} features
        # self.active_target_weights are the weights that were active leading to the current state
        prev_cash_w = self.active_target_weights[0]
        prev_stock_ws = self.active_target_weights[1:] # Shape [N]

        # Tile prev_cash_w to shape [T, N, 1]
        w_cash_t_minus_1_feature = np.full(
            (self.time_steps, self.stock_dim, 1),
            prev_cash_w,
            dtype=np.float32
        )

        # Tile each stock's prev_stock_w to shape [T, N, 1]
        w_stocks_t_minus_1_feature = np.zeros(
            (self.time_steps, self.stock_dim, 1),
            dtype=np.float32
        )
        for i in range(self.stock_dim):
            w_stocks_t_minus_1_feature[:, i, 0] = prev_stock_ws[i]

        # Concatenate: [T, N, F_original + 2]
        state_window_data_augmented = np.concatenate(
            (state_window_data_original, w_cash_t_minus_1_feature, w_stocks_t_minus_1_feature),
            axis=2
        )

        # Transpose to agent's expected input format: [N, T, F_new]
        final_state = np.transpose(state_window_data_augmented, (1, 0, 2))
        return final_state.astype(np.float32)

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

        # <<< MODIFICATION START: Reset active_target_weights and rebalance counter >>>
        initial_active_weight_val = 1.0 / (self.stock_dim + 1)
        self.active_target_weights = np.array([initial_active_weight_val] * (self.stock_dim + 1), dtype=np.float32)
        self.weights_memory = [self.active_target_weights.tolist()] # Store initial active weights
        self.days_since_last_rebalance = 0
        # <<< MODIFICATION END >>>


        raw_state_format = np.transpose(state_window_data, (1, 2, 0))
        self.state = self._get_augmented_state() # Use helper method

        self.terminal = False
        self.portfolio_value = float(self.initial_amount)
        self.asset_memory = [float(self.initial_amount)]
        self.portfolio_return_memory = [0.0]
        

        self.date_memory = [self.unique_dates[self.day_idx]]
        self.transaction_cost_memory = []
        
        return self.state

    def step(self, weights_from_agent: np.ndarray): # weights_from_agent is the action proposed by actor
        weights_from_agent = np.asarray(weights_from_agent, dtype=np.float32)
        # Ensure weights_from_agent is normalized if it isn't already
        # (though Actor's softmax output should already be normalized)
        weights_from_agent = self.normalization(weights_from_agent)


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
                    "Calmar Ratio": ["{:04f}".format(cr)],
                    "Sortino Ratio": ["{:04f}".format(sor)],
                }
            )
            table = print_metrics(stats)
            print(table)
            if self.task.startswith("train") or self.task.startswith("valid"):
                if self.empty:
                    open(self.txt, "w").close()  # 清空檔案
                    self.empty = False
            
                with open(self.txt, "a", encoding="utf-8") as f:
                    if (self.task.startswith("valid")):
                        f.write("Valid Episode: " + "\n")
                    if (self.task.startswith("train")):
                        f.write("Train Episode: " + "\n")
                    f.write(str(table) + "\n")
            df_return = self.save_portfolio_return_memory()
            daily_return_values = df_return.daily_return.values
            df_value = self.save_asset_memory()
            assets_values = df_value["total assets"].values # This is a numpy array from pd.Series.values

            save_dict = OrderedDict(
                {
                    "Profit Margin": tr * 100,
                    "Excess Profit": tr * 100 - 0,
                    "daily_return": daily_return_values,
                    "total_assets": assets_values # assets_values is already a numpy array
                }
            )
            metric_save_path = osp.join(self.work_dir, f'metric_{self.task}_{self.test_dynamic}_{self.test_id}_{self.task_index}.pickle')
            if self.task == 'test_dynamic': 
                with open(metric_save_path, 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            return self.state.astype(np.float32), 0.0, self.terminal, {"sharpe_ratio": sharpe_ratio, "total_assets": assets_values}

        # --- Non-Terminal Step ---
        
        # Prices at the END of the current self.day_idx (let's call this day t)
        # These are needed to calculate the return from t to t+1
        prices_at_t = self.close_prices_cube[self.day_idx, :]

        # Advance time and rebalance counter
        self.day_idx += 1
        self.days_since_last_rebalance += 1
        
        # Prices at the END of new self.day_idx (day t+1)
        prices_at_t_plus_1 = self.close_prices_cube[self.day_idx, :]

        # Weights held throughout day t (from market open of t to market open of t+1)
        # These are self.active_target_weights set at the end of day t-1 (or start of day t)
        held_stock_weights = self.active_target_weights[1:]
        held_cash_weight = self.active_target_weights[0]
        
        # Calculate price ratios from end of day t to end of day t+1
        price_ratios = np.divide(prices_at_t_plus_1, prices_at_t,
                                 out=np.ones_like(prices_at_t_plus_1, dtype=np.float32),
                                 where=prices_at_t != 0)
        
        # Calculate gross return based on weights held *during* the day
        # The portfolio_value at the beginning of this step is value at end of day t (start of t+1 trading)
        # The portion invested in stocks experiences market return
        stock_values_at_t = self.portfolio_value * held_stock_weights
        stock_values_at_t_plus_1 = stock_values_at_t * price_ratios
        cash_value_at_t_plus_1 = self.portfolio_value * held_cash_weight
        
        portfolio_value_after_drift = np.sum(stock_values_at_t_plus_1) + cash_value_at_t_plus_1

        # Calculate drifted weights at the end of day t+1 (before any rebalancing for day t+2)
        drifted_weights_numerator = np.concatenate(([cash_value_at_t_plus_1], stock_values_at_t_plus_1))
        drifted_weights = self.normalization(drifted_weights_numerator)
        
        transaction_fee_for_today_rebalance = 0.0
        effective_weights_for_next_period = drifted_weights # Default if no rebalance

        # Check if it's a rebalancing day (decision made at end of t+1, for t+2)
        # The first day (day_idx = time_steps-1) is not a rebalance day unless rebalance_interval is 1
        # and days_since_last_rebalance would be 1.
        # A simple check: if it's the first step of an episode, force rebalance based on initial weights.
        # However, self.days_since_last_rebalance starts at 0 in reset, becomes 1 at first step.
        if self.days_since_last_rebalance % self.rebalance_interval == 0:
            # Rebalance using weights_from_agent (which are target weights for t+1 to t+2 period)
            # Cost is based on rebalancing from drifted_weights to weights_from_agent
            # The value used for cost calculation is portfolio_value_after_drift
            diff_for_tx = np.sum(np.abs(drifted_weights - weights_from_agent))
            transaction_fee_for_today_rebalance = diff_for_tx * self.transaction_cost_pct * portfolio_value_after_drift
            
            effective_weights_for_next_period = weights_from_agent
            self.days_since_last_rebalance = 0 # Reset counter
        
        # Portfolio value after transaction cost (if any rebalancing occurred)
        portfolio_value_after_cost_and_return = portfolio_value_after_drift - transaction_fee_for_today_rebalance
        
        # Calculate reward (using portfolio_value at start of day t vs end of day t+1 after costs)
        # self.portfolio_value at this point is the value at the END of day t (or start of day t+1 before trading)
        # Reward: log return, with safe guards for log(0) or log(<0)
        if portfolio_value_after_cost_and_return > 1e-9 and self.portfolio_value > 1e-9: # Use small epsilon
            self.reward = (portfolio_value_after_cost_and_return - self.portfolio_value) / self.portfolio_value #np.log(value_after_cost_and_return / self.portfolio_value)
            # print("yes!!!")
        elif portfolio_value_after_cost_and_return > 1e-9 and self.portfolio_value <= 1e-9:
            self.reward = (portfolio_value_after_cost_and_return - self.portfolio_value) / self.portfolio_value #np.log(value_after_cost_and_return / 1e-9) # Large positive reward
            # print("no!!!")
        elif portfolio_value_after_cost_and_return <= 1e-9 and self.portfolio_value > 1e-9:
            self.reward = (portfolio_value_after_cost_and_return - self.portfolio_value) / self.portfolio_value  # np.log(1e-9 / self.portfolio_value) # Large negative reward
            # print("no2!!!")
        else: # both are very small or zero
            self.reward = 0.0
        
        # Update master portfolio value and active weights for the next period
        self.portfolio_value = portfolio_value_after_cost_and_return
        self.active_target_weights = effective_weights_for_next_period

        # Update memories
        # net_portfolio_return for the period t to t+1
        net_daily_return = (self.portfolio_value / (self.asset_memory[-1] if self.asset_memory else self.initial_amount) - 1) \
            if (self.asset_memory and self.asset_memory[-1] > 1e-9) else 0.0
        self.portfolio_return_memory.append(net_daily_return)
        self.asset_memory.append(self.portfolio_value)
        self.weights_memory.append(self.active_target_weights.tolist()) # Record weights that will be active
        self.date_memory.append(self.unique_dates[self.day_idx])
        self.transaction_cost_memory.append(transaction_fee_for_today_rebalance)

        # Prepare next state (for day t+1 observation, to decide action for t+2)
        start_slice_idx = self.day_idx - self.time_steps + 1
        end_slice_idx = self.day_idx + 1
        actual_start_idx = max(0, start_slice_idx)
        state_window_data = self.data_cube[actual_start_idx:end_slice_idx, :, :]
        if state_window_data.shape[0] < self.time_steps:
            padding_needed = self.time_steps - state_window_data.shape[0]
            padding_array = np.zeros((padding_needed, self.stock_dim, self.num_tech_indicators), dtype=np.float32)
            state_window_data = np.concatenate((padding_array, state_window_data), axis=0)
        raw_state_format = np.transpose(state_window_data, (1, 2, 0))
        self.state = self._get_augmented_state() # Use helper method
        
        info_dict = {
            "drifted_weights_eod": drifted_weights.tolist(),
            "agent_target_if_rebalanced": weights_from_agent.tolist() if (self.days_since_last_rebalance == 0) else None,
            "active_weights_next_period": self.active_target_weights.tolist()
        }
        return self.state, float(self.reward), self.terminal, info_dict

    def normalization(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        # Handle potential NaN from previous normalization if sum was zero
        if np.isnan(actions).any(): # If actions contains NaN, it means it was problematic before.
             # Default to 100% cash if weights are unrecoverable
            actions = np.zeros_like(actions)
            actions[0] = 1.0
            return actions

        s = np.sum(actions)
        if np.abs(s) < 1e-9: 
            if not np.any(actions): 
                return actions 
            # If sum is zero but not all actions are zero (e.g. [0.1, -0.1, 0]), default to 100% cash
            actions = np.zeros_like(actions)
            actions[0] = 1.0
            return actions
        return actions / s

    def save_portfolio_return_memory(self) -> pd.DataFrame:
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        if not df_date.empty: 
             df_return.index = df_date.date
        return df_return

    def save_asset_memory(self) -> pd.DataFrame:
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        if not df_date.empty: 
            df_value.index = df_date.date
        return df_value

    def analysis_result(self):
        # A simpler API for the environment to analysis itself when coming to terminal
        df_return = self.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df_value = self.save_asset_memory()
        assets = df_value["total assets"].values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        return self.evaualte(df)

    def get_daily_return_rate(self,price_list:list):
        return_rate_list=[]
        for i in range(len(price_list)-1):
            return_rate=(price_list[i+1]/price_list[i])-1
            return_rate_list.append(return_rate)
        return return_rate_list
        

    def evaualte(self, df):
        daily_return = df["daily_return"]
        # print(df, df.shape, len(df),len(daily_return))
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1] / (df["total assets"].values[0] + 1e-10) - 1
        return_rate_list=self.get_daily_return_rate(df["total assets"].values)

        sharpe_ratio = np.mean(return_rate_list)*(252)** 0.5 / (np.std(return_rate_list) + 1e-10)
        vol = np.std(return_rate_list)
        mdd = 0
        peak=df["total assets"][0]
        for value in df["total assets"]:
            if value>peak:
                peak=value
            dd=(peak-value)/peak
            if dd>mdd:
                mdd=dd
        cr = np.sum(daily_return) / (mdd + 1e-10)
        sor = np.sum(daily_return) / (np.nan_to_num(np.std(neg_ret_lst),0) + 1e-10) / (np.sqrt(len(daily_return))+1e-10)
        return tr, sharpe_ratio, vol, mdd, cr, sor