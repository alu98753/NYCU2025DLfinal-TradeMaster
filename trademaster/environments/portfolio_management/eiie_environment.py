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
        time_steps = get_attr(self.dataset, "time_steps", 10)
        self.day = time_steps - 1
        self.train_flag = 0

        self.df_path = None
        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
            self.train_flag = 1
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)



        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
            self.start_date = self.df.loc[:, 'date'].iloc[0]
            self.end_date = self.df.loc[:, 'date'].iloc[-1]
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)
            # print(f"df is : {self.df}")

        self.stock_dim = len(self.df.tic.unique())
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim
        self.time_steps = time_steps
        # print(f"env timesteps:{time_steps}")
        self.action_space = spaces.Box(low=-5,
                                       high=5,
                                       shape=(self.action_space_shape,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list),
                   self.state_space_shape,
                   self.time_steps))

        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        self.data = self.df.loc[self.day - self.time_steps + 1:self.day, :]
        self.state = np.zeros((len(self.tech_indicator_list), self.stock_dim, self.time_steps))

        for j, tic in enumerate(self.data.tic.unique()):
            for i, tech in enumerate(self.tech_indicator_list):
                series = self.data[self.data.tic == tic][tech].values
                series = pd.Series(series).ffill().bfill().values
                if len(series) < self.time_steps:
                    series = np.pad(series, (self.time_steps - len(series), 0), mode='edge')
                else:
                    series = series[-self.time_steps:]

                self.state[i, j] = series
        # self.state = np.array([[
        #     self.data[self.data.tic == tic][tech].values.tolist()
        #     for tech in self.tech_indicator_list
        # ] for tic in self.data.tic.unique()])
        # print("[DEBUG] state shape before transpose:", self.state.shape)
        self.state = np.transpose(self.state, (1, 2, 0))

        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        # if len(self.asset_memory) <= 1:
        #     print("[Warning] Not enough asset memory to evaluate.")
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1] + [0] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []
        self.test_id = 'agent'
    def _build_state_tensor(self, data_slice):
        state = np.zeros((len(self.tech_indicator_list), self.stock_dim, self.time_steps))
        for j, tic in enumerate(data_slice.tic.unique()):
            for i, tech in enumerate(self.tech_indicator_list):
                series = data_slice[data_slice.tic == tic][tech].values
                if np.any(np.isnan(series)):
                    print(f"[Warning] NaN in series for {tic}-{tech}")
                series = pd.Series(series).ffill().bfill().values
                if len(series) < self.time_steps:
                    series = np.pad(series, (self.time_steps - len(series), 0), mode='edge')
                else:
                    series = series[-self.time_steps:]
                state[i, j] = series
        return np.transpose(state, (1, 2, 0))
    def reset(self):
        self.day = self.time_steps - 1
        self.data = self.df.loc[self.day - self.time_steps + 1:self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        self.state = self._build_state_tensor(self.data)
        # self.state = np.transpose(self.state, (2, 0, 1))
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / (self.stock_dim + 1)] *
                               (self.stock_dim + 1)]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []

        return self.state

    def step(self, weights):
        # make judgement about whether our data is running out
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        weights = np.array(weights)

        if self.terminal:
            print("something has done")
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            stats = OrderedDict(
                {
                    "Total Return": ["{:04f}%".format(tr * 100)],
                    "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                    "Volatility": ["{:04f}%".format(vol* 100)],
                    "Max Drawdown": ["{:04f}%".format(mdd* 100)],
                    # "Calmar Ratio": ["{:04f}".format(cr)],
                    # "Sortino Ratio": ["{:04f}".format(sor)],
                }
            )
            table = print_metrics(stats)
            print(table)

            df_return = self.save_portfolio_return_memory()
            daily_return = df_return.daily_return.values
            df_value = self.save_asset_memory()
            assets = df_value["total assets"].values
            #TODO calculate the buy and hold
            save_dict = OrderedDict(
                {
                    "Profit Margin": tr * 100,
                    "Excess Profit": tr * 100-0,
                    "daily_return": daily_return,
                    "total_assets": assets
                }
            )
            metric_save_path=osp.join(self.work_dir,'metric_'+str(self.task)+'_'+str(self.test_dynamic)+'_'+str(self.test_id)+'_'+str(self.task_index)+'.pickle')
            if self.task == 'test_dynamic':
                with open(metric_save_path, 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return self.state, 0, self.terminal, {"sharpe_ratio": sharpe_ratio,"total_assets": assets}, 1

        else:  # directly use the process of
            self.weights_memory.append(weights)
            last_day_memory = self.df.loc[self.day, :]
            # print(f"last_day_memory :{last_day_memory}")
            self.day += 1
            self.data = self.df.loc[self.day - self.time_steps + 1:self.day, :] ##拿前50天的所有data

            self.state = self._build_state_tensor(self.data)

            new_price_memory = self.df.loc[self.day, :] ##拿最新當天的price
            portfolio_weights = weights[:-1] ##已經透過actor輸出的weight 不包括cash
            
            portfolio_return = sum(((new_price_memory.close.values / last_day_memory.close.values) - 1) * portfolio_weights)##看前一天與今天的漲幅
            if self.day < len(self.df.index.unique()) - 1 :
                temp = self.day+1
            else :
                temp = self.day
            next_day_price_memory = self.df.loc[temp, :] ## 明天的價格資料
            price_rate = 1
            if self.train_flag == 1:
                price_rate = (next_day_price_memory.close.values / new_price_memory.close.values) ##得到明天跟今天價格資料的變化

            weights_brandnew = self.normalization([weights[-1]] + list(np.array(weights[:-1]) *
                            np.array((new_price_memory.close.values /last_day_memory.close.values))))##調整作天到今天的現金比例

            self.weights_memory.append(weights_brandnew)
            weights_old = (self.weights_memory[-3])
            weights_new = (self.weights_memory[-2])
            diff_weights = np.sum(
                np.abs(np.array(weights_old) - np.array(weights_new)))
            transcationfee = diff_weights * self.transaction_cost_pct * self.portfolio_value
            new_portfolio_value = (self.portfolio_value -transcationfee) * (1 + portfolio_return)
            # new_portfolio_value = max(1.0, (self.portfolio_value - transcationfee) * (1 + portfolio_return))
            portfolio_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
            if self.portfolio_value > 0 and new_portfolio_value > 0:
                self.reward = np.log(new_portfolio_value) - np.log(self.portfolio_value)
                self.reward = np.clip(self.reward, -1.0, 1.0)
            else:
                self.reward = 0.0
            # self.reward = np.log(new_portfolio_value) - np.log(self.portfolio_value)
            self.portfolio_value = new_portfolio_value
            # print(self.reward)
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[-1])
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.reward

        return self.state, self.reward, self.terminal, {"weights_brandnew":weights_brandnew}, price_rate

    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        actions = np.array(actions)
        sum = np.sum(actions)
        actions = actions / sum
        return actions

    def save_portfolio_return_memory(self):
        # a record of return for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = df_date.date

        return df_return

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = df_date.date

        return df_value

    def analysis_result(self):
        # A simpler API for the environment to analysis itself when coming to terminal
        df_return = self.save_portfolio_return_memory()
        # print(f"df_return {df_return}")
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
            # return_rate=(price_list[i+1]/price_list[i])-1
            # return_rate_list.append(return_rate)
            if price_list[i] == 0 or np.isnan(price_list[i]) or np.isnan(price_list[i+1]):
                return_rate_list.append(0.0)  # or continue
            else:
                return_rate = (price_list[i+1] / price_list[i]) - 1
                return_rate_list.append(return_rate)
        return return_rate_list
        

    def evaualte(self, df):
        daily_return = df["daily_return"]
        # print(df, df.shape, len(df),len(daily_return))
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        # print(f"ned_ret_lst :{neg_ret_lst}")
        # if df["total assets"].values[-1]
        tr = df["total assets"].values[-1] / (df["total assets"].values[0] + 1e-10) - 1
        # df["total assets"].to_csv(f'assets.csv', index=False, header=False)
        if np.isnan(tr) or np.isinf(tr):
            print("error tr")
        # print("=== asset_memory ===")
        # print(self.asset_memory[-5:])
        # print("=== return_rate_list ===")
        # print(self.get_daily_return_rate(self.asset_memory[-10:]))
        return_rate_list=self.get_daily_return_rate(df["total assets"].values)
        # print(f"tr :{tr}")
        # print(f"return_rate_list :{return_rate_list}")
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
