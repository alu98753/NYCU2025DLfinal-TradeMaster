"""
{'data':
    {'type': 'PortfolioManagementDataset',
        'data_path': 'data/portfolio_management/dj30',
        'train_path': 'data/portfolio_management/dj30/train.csv',
        'valid_path': 'data/portfolio_management/dj30/valid.csv',
        'test_path': 'data/portfolio_management/dj30/test.csv',
        'tech_indicator_list': ['zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'],
        'length_day': 10,
        'initial_amount': 100000,
        'transaction_cost_pct': 0.001,
        'test_dynamic_path': 'data/portfolio_management/dj30/test_with_label.csv',
        'test_dynamic': '-1'
        },
    'environment':
    {'type': 'PortfolioManagementEIIEEnvironment'}, 'agent': {'type': 'PortfolioManagementEIIE',
    'memory_capacity': 1000, 'gamma': 0.99, 'policy_update_frequency': 500},
    'trainer': {'type': 'PortfolioManagementEIIETrainer', 'epochs': 1,
    'work_dir': 'work_dir/portfolio_management_dj30_eiie_eiie_adam_mse',
    'if_remove': False}, 'loss': {'type': 'MSELoss'}, 'optimizer': {'type': 'Adam', 'lr': 0.001},
    'act': {'type': 'EIIEConv', 'input_dim': None, 'output_dim': 1, 'time_steps': 10, 'kernel_size': 3, 'dims': [32]},
    'cri': {'type': 'EIIECritic', 'input_dim': None, 'action_dim': None, 'output_dim': 1, 'time_steps': None, 'num_layers': 1, 'hidden_size': 32},
    'transition': {'type': 'Transition'}, 'task_name': 'portfolio_management',
    'dataset_name': 'dj30', 'net_name': 'eiie', 'agent_name': 'eiie', 'optimizer_name': 'adam', 'loss_name': 'mse',
    'work_dir': 'work_dir/portfolio_management_dj30_eiie_eiie_adam_mse'}
"""

import warnings

warnings.filterwarnings("ignore")
import argparse
import os
import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config

import wandb

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from trademaster.agents.builder import build_agent
from trademaster.datasets.builder import build_dataset
from trademaster.environments.builder import build_environment
from trademaster.losses.builder import build_loss
from trademaster.nets.builder import build_net
from trademaster.optimizers.builder import build_optimizer
from trademaster.trainers.builder import build_trainer
from trademaster.transition.builder import build_transition
from trademaster.utils import (
    calculate_radar_score,
    create_radar_score_baseline,
    plot_radar_chart,
    replace_cfg_vals,
    set_seed,
)

set_seed(2023)


def parse_args():
    parser = argparse.ArgumentParser(description="Download Alpaca Datasets")
    parser.add_argument(
        "--config",
        default=osp.join(
            ROOT,
            "configs",
            "portfolio_management",
            "portfolio_management_tw50_HCAR_HCAR_adam_mse.py",
        ),
        help="download datasets config file path",
    )
    parser.add_argument("--task_name", type=str, default="train")
    parser.add_argument("--test_dynamic", type=str, default="-1")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    task_name = args.task_name

    cfg = replace_cfg_vals(cfg)
    # update test style
    cfg.data.update({"test_dynamic": args.test_dynamic})
    if args.verbose == 1:
        print(cfg)

    if task_name.startswith("test"):
        mode='disabled'
    else:
        mode='online'



    ### init wandb
    wandb.init(
        project="HCAR_test_moduleB",
        name=f"{cfg.net_name}_{cfg.agent_name}_{cfg.optimizer_name}_{cfg.loss_name}_run_{time.time()}",
        config=cfg.to_dict(),
        mode=mode,
        # mode='disabled'

    )
    wandb.define_metric("agent_step")

    # Tell wandb which metrics should use which x-axis
    # For metrics logged per epoch (e.g., validation metrics)
    wandb.define_metric(
        "Validation/*", step_metric="agent_step"
    )  # Log all Validation metrics against epoch

    # For metrics logged per training step (e.g., losses, gradients)
    wandb.define_metric("Loss/*", step_metric="agent_step")
    wandb.define_metric("Gradients/*", step_metric="agent_step")
    wandb.define_metric("Values/*", step_metric="agent_step")  # For Q-values
    wandb.define_metric(
        "HCAR_Actor/*", step_metric="agent_step"
    )  # For internal actor stats
    # Add other per-step metrics here if any, e.g., learning rates if logged per step
    wandb.define_metric(
        "Learning_Rate/*", step_metric="agent_step"
    )  # Or "epoch" if logged per epoch
    wandb.define_metric(
        "Action/*", step_metric="agent_step"
    )  # Or "epoch" if logged per epoch

    dataset = build_dataset(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print('one')
    train_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="train")
    )
    # print('two')
    valid_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="valid")
    )
    # print('three')
    test_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="test")
    )
    # print('four')

    if task_name.startswith("dynamics_test"):
        test_dynamic_environments = []
        for i, path in enumerate(dataset.test_dynamic_paths):
            test_dynamic_environments.append(
                build_environment(
                    cfg,
                    default_args=dict(
                        dataset=dataset,
                        task="test_dynamic",
                        dynamics_test_path=path,
                        task_index=i,
                        work_dir=cfg.work_dir,
                    ),
                )
            )

    # 從環境中獲取必要的維度信息
    action_dim = train_environment.action_dim  # 股票數量, 例如 49
    state_dim = train_environment.state_dim  # 11
    # 我們 HCAR_Actor 的 num_original_features 就是這個
    num_original_features_for_actor = train_environment.state_dim
    input_dim = len(train_environment.tech_indicator_list)
    time_steps = (
        train_environment.time_steps
    )  # time_steps 在原 EIIE 環境中代表 length_day, 例如 10 (我們為ASU/HCAR設為13)
    print(f"times step:{time_steps}")
    # --- MODIFICATION START: 準備 HCAR_Actor 所需的額外參數 ---
    num_stocks_for_actor = (
        action_dim  # 對應 HCAR_Actor 的 num_stocks 和 ASU 的 num_nodes
    )

    # 指定您的因果圖路徑 (請確保路徑正確)
    # 假設您的因果圖是為 tw50 準備的
    # 您可以將這個路徑也放到配置文件中，然後從 cfg 中讀取
    causal_graph_path = osp.join(
        ROOT, "data/portfolio_management/tw50/ASU_data/pc_causal_relation.npy"
    )

    try:
        relation_matrix_np = np.load(causal_graph_path)
        assert relation_matrix_np.shape == (
            num_stocks_for_actor,
            num_stocks_for_actor,
        ), f"Causal graph shape {relation_matrix_np.shape} does not match num_stocks {num_stocks_for_actor}"
        supports_list = [torch.from_numpy(relation_matrix_np).float().to(device)]
        print(f"Successfully loaded causal graph from: {causal_graph_path}")
    except Exception as e:
        print(f"Error loading causal graph: {e}")
        print(
            f"Falling back to identity matrix for supports for {num_stocks_for_actor} stocks."
        )
        supports_list = [torch.eye(num_stocks_for_actor).float().to(device)]

    # 更新 cfg.act 以包含 HCAR_Actor 初始化所需的全部參數
    # 原先的 input_dim, time_steps 會被這裡的 num_original_features, window_len 覆蓋或對應
    cfg.act.update(
        dict(
            num_original_features=num_original_features_for_actor,
            # window_len_for_temporal=window_len_for_actor, # <--- 新增一個清晰的參數名給 HCAR_Actor 的 temporal_feature_extractor
            num_stocks=num_stocks_for_actor,
            supports=supports_list,
            # 其他 HCAR_Actor 在配置文件中定義的參數 (如 hidden_dims, dropouts 等) 會被保留
        )
    )
    # --- MODIFICATION END ---

    # 更新 Critic 的配置 (如果 Critic 的輸入依賴於 Actor 的輸入維度，也需要對應調整)
    # 原 EIIECritic 的 input_dim 是 num_original_features, time_steps 是 window_len
    # action_dim 是 num_stocks
    cfg.cri.update(
        dict(
            input_dim=num_original_features_for_actor,
            action_dim=action_dim,  # 股票數量
            time_steps=time_steps,  # 與 Actor 的時間窗口一致
        )
    )

    act = build_net(cfg.act)
    cri = build_net(cfg.cri)
    wandb.watch(act, log="gradients", log_freq=500, log_graph=True)
    wandb.watch(cri, log="gradients", log_freq=500, log_graph=True)

    work_dir = os.path.join(ROOT, cfg.trainer.work_dir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))
    cri_optimizer = build_optimizer(cfg, default_args=dict(params=cri.parameters()))
    criterion = build_loss(cfg)
    transition = build_transition(cfg)

    agent = build_agent(
        cfg,
        default_args=dict(
            action_dim=action_dim,
            state_dim=state_dim,
            time_steps=time_steps,
            act=act,
            cri=cri,
            act_optimizer=act_optimizer,
            cri_optimizer=cri_optimizer,
            criterion=criterion,
            transition=transition,
            device=device,
            # --- 從 cfg.agent 中獲取 LR 調度參數 ---
            use_lr_scheduler=cfg.agent.get("use_lr_scheduler", False),  # 提供默認值
            warmup_steps=cfg.agent.get("warmup_steps", 0),
        ),
    )

    if task_name.startswith("dynamics_test"):
        trainers = []
        for env in test_dynamic_environments:
            trainers.append(
                build_trainer(
                    cfg,
                    default_args=dict(
                        train_environment=train_environment,
                        valid_environment=valid_environment,
                        test_environment=env,
                        agent=agent,
                        device=device,
                    ),
                )
            )
    else:
        trainer = build_trainer(
            cfg,
            default_args=dict(
                train_environment=train_environment,
                valid_environment=valid_environment,
                test_environment=test_environment,
                agent=agent,
                device=device,
            ),
        )
    if task_name.startswith("train"):
        trainer.train_and_valid()
        print("train end")
    elif task_name.startswith("test"):
        trainer.test()
        print("test end")
    elif task_name.startswith("dynamics_test"):

        def Average_holding(states, env, weights_brandnew):
            if weights_brandnew is None:
                action = [0] + [1 / env.stock_dim for _ in range(env.stock_dim)]
                return action
            else:
                return weights_brandnew

        def Do_Nothing(states, env):
            return [1] + [0 for _ in range(env.stock_dim)]

        daily_return_list = []
        daily_return_list_Average_holding = []
        daily_return_list_Do_Nothing = []
        for trainer in trainers:
            daily_return_list.extend(trainer.test())
            daily_return_list_Average_holding.extend(
                trainer.test_with_customize_policy(Average_holding, "Average_holding")
            )
            daily_return_list_Do_Nothing.extend(
                trainer.test_with_customize_policy(Do_Nothing, "Do_Nothing")
            )
            metric_path = (
                "metric_"
                + str(trainer.test_environment.task)
                + "_"
                + str(trainer.test_environment.test_dynamic)
            )
        metrics_sigma_dict, zero_metrics = create_radar_score_baseline(
            cfg.work_dir,
            metric_path,
            zero_score_id="Do_Nothing",
            fifty_score_id="Average_holding",
        )
        test_metrics_scores_dict = calculate_radar_score(
            cfg.work_dir, metric_path, "agent", metrics_sigma_dict, zero_metrics
        )
        radar_plot_path = cfg.work_dir
        # 'metric_' + str(self.task) + '_' + str(self.test_dynamic) + '_' + str(id) + '_radar.png')
        # print('test_metrics_scores are: ',test_metrics_scores_dict)
        # print('test_metrics_scores are:')
        # print_metrics(test_metrics_scores_dict)
        test_dynamic = args.test_dynamic
        plot_radar_chart(
            test_metrics_scores_dict,
            "radar_plot_agent_" + str(test_dynamic) + ".png",
            radar_plot_path,
        )
        # print('win rate is: ', sum(float(r) > 0 for r in daily_return_list) / len(daily_return_list))
        # print('Random_buy win rate is: ', sum(float(r) > 0 for r in daily_return_list_Average_holding) / len(daily_return_list_Average_holding))
        print("dynamics test end")


if __name__ == "__main__":
    main()
    """
    algorithmic_trading
    portfolio_management
    """
