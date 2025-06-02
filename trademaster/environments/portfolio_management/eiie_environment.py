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
        
        # <<< MODIFICATION START: Add rebalance_interval and related attributes >>>
        self.rebalance_interval = int(get_attr(kwargs, "rebalance_interval", 1))
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

        self.state_space_shape = self.stock_dim 
        self.action_space_shape = self.stock_dim 
        
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.action_space_shape,))
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,
            shape=(self.stock_dim, self.time_steps,self.num_tech_indicators) )

        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.num_tech_indicators

        self.day_idx = self.time_steps - 1 

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

        self.terminal = False
        self.portfolio_value = float(self.initial_amount)
        self.asset_memory = [float(self.initial_amount)]
        self.portfolio_return_memory = [0.0]
        
        # <<< MODIFICATION START: Initialize active_target_weights >>>
        # These are the weights that are "live" in the market.
        # They are set at rebalancing and then drift with market prices.
        initial_active_weight_val = 1.0 / (self.stock_dim + 1) # Default to equal weight if not rebalancing on day 0
        self.active_target_weights = np.array([initial_active_weight_val] * (self.stock_dim + 1), dtype=np.float32)
        self.weights_memory = [self.active_target_weights.tolist()] # Store the initial active weights
        # <<< MODIFICATION END >>>

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

        raw_state_format = np.transpose(state_window_data, (1, 2, 0))
        self.state = np.transpose(raw_state_format, (0, 2, 1))

        self.terminal = False
        self.portfolio_value = float(self.initial_amount)
        self.asset_memory = [float(self.initial_amount)]
        self.portfolio_return_memory = [0.0]
        
        # <<< MODIFICATION START: Reset active_target_weights and rebalance counter >>>
        initial_active_weight_val = 1.0 / (self.stock_dim + 1)
        self.active_target_weights = np.array([initial_active_weight_val] * (self.stock_dim + 1), dtype=np.float32)
        self.weights_memory = [self.active_target_weights.tolist()] # Store initial active weights
        self.days_since_last_rebalance = 0
        # <<< MODIFICATION END >>>

        self.date_memory = [self.unique_dates[self.day_idx]]
        self.transaction_cost_memory = []
        
        return self.state.astype(np.float32)

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
                }
            )
            table = print_metrics(stats)
            print(table)

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
        self.state = np.transpose(raw_state_format, (0, 2, 1))
        
        info_dict = {
            "drifted_weights_eod": drifted_weights.tolist(),
            "agent_target_if_rebalanced": weights_from_agent.tolist() if (self.days_since_last_rebalance == 0) else None,
            "active_weights_next_period": self.active_target_weights.tolist()
        }
        return self.state.astype(np.float32), float(self.reward), self.terminal, info_dict

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

    def analysis_result(self) -> tuple[float, float, float, float, float, float]:
        df_return = self.save_portfolio_return_memory()
        df_value = self.save_asset_memory()
        
        df_eval = pd.DataFrame()
        df_eval["daily_return"] = df_return["daily_return"] if "daily_return" in df_return else pd.Series(dtype=np.float64)
        df_eval["total assets"] = df_value["total assets"] if "total assets" in df_value else pd.Series(dtype=np.float64)
        
        if df_eval.empty or df_eval["total assets"].empty or len(df_eval["total assets"]) < 1: # Check if series has at least one element
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        return self.evaualte(df_eval)

    def get_daily_return_rate(self, price_list: np.ndarray | list[float]) -> list[float]:
        price_array = np.asarray(price_list, dtype=np.float32)
        if len(price_array) < 2:
            return []
        # Ensure previous prices are not zero to avoid division by zero
        safe_denominator = np.where(np.abs(price_array[:-1]) < 1e-9, 1e-9, price_array[:-1])
        return_rates = (price_array[1:] / safe_denominator) - 1
        return return_rates.tolist()
        
    def evaualte(self, df: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
        if df.empty or df["daily_return"].empty or df["total assets"].empty or len(df["total assets"]) < 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 

        daily_return_series = df["daily_return"]
        total_assets_series = df["total assets"]

        # Ensure series are not empty before attempting to access iloc[0] or iloc[-1]
        if total_assets_series.empty:
             return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        neg_ret_series = daily_return_series[daily_return_series < 0]
        
        initial_assets = total_assets_series.iloc[0]
        final_assets = total_assets_series.iloc[-1]

        if np.abs(initial_assets) < 1e-10: # Avoid division by zero if initial assets is ~0
            tr = 0.0 if np.abs(final_assets) < 1e-10 else np.sign(final_assets) * np.inf
        else:
            tr = final_assets / initial_assets - 1
            
        return_rate_list_for_sharpe_vol = self.get_daily_return_rate(total_assets_series.values)

        if not return_rate_list_for_sharpe_vol or len(return_rate_list_for_sharpe_vol) == 0: # ensure list is not empty
            sharpe_ratio = 0.0
            vol = 0.0
        else:
            mean_return = np.mean(return_rate_list_for_sharpe_vol)
            std_return = np.std(return_rate_list_for_sharpe_vol)
            sharpe_ratio = mean_return * (252 ** 0.5) / (std_return + 1e-10) # Annualized Sharpe
            vol = std_return * (252 ** 0.5) # Annualized Volatility

        mdd = 0.0
        if not total_assets_series.empty:
            peak = total_assets_series.iloc[0]
            for value in total_assets_series:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if np.abs(peak) > 1e-9 else 0.0 
                if dd > mdd:
                    mdd = dd
        
        # Use annualized return for Calmar Ratio if Sharpe is annualized
        annualized_return = 0.0
        if len(total_assets_series) > 1: # Need at least two points for a return period
            num_days = len(total_assets_series) -1
            if num_days > 0 :
                annualized_return = tr * (252.0 / num_days) if num_days < 252 else tr # Simple scaling for periods < 1 year


        cr = annualized_return / (mdd + 1e-10) if (mdd > 1e-9 or annualized_return !=0) else 0.0
        
        # Sortino Ratio also typically uses annualized mean return
        std_neg_ret = np.std(neg_ret_series.values) if not neg_ret_series.empty else 0.0
        # Denominator for Sortino: downside deviation (annualized)
        downside_deviation_annualized = std_neg_ret * (252**0.5)

        sor = annualized_return / (downside_deviation_annualized + 1e-10) if downside_deviation_annualized > 1e-9 else 0.0
        
        return tr, sharpe_ratio, vol, mdd, cr, sor