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

################################################################################
# ———— 新增部分：pretrain_marketnet 函数 ————
################################################################################
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR

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

def pretrain_marketnet(
    actor: torch.nn.Module,
    market: torch.nn.Module,
    env,
    device: torch.device,
    target_acc: float = 0.90,        # 牛/熊两类累积准确率都要 >= 0.90
    max_pretrain_steps: int = 5000,  # 最多训练多少个 minibatch
    batch_size: int = 32
):
    """
    预训练 MarketNet（二分类）。要求：
      1. 先采集所有 (s_market, regime_t) 样本，regime_t ∈ {0=牛市,1=熊市}。
      2. 构造二分类数据集 + WeightedRandomSampler 平衡采样。
      3. 训练时：
         - 保留 minibatch 级别的 loss/accuracy 打印（或 log）；
         - 每轮 epoch 结束时，计算“整轮 epoch”上对牛(0)/熊(1)两类的累计准确率，
           如果牛/熊都 ≥ target_acc，就进行 Early Stop。
      4. 学习率策略：初始 lr=1e-4，使用 StepLR 每 100 次 optimizer.step() 将 lr *= 0.5。
      5. 最后冻结 market 网络，返回 market 和 actor（actor 恢复成 train 模式）。
    """

    # 1) 切换模式
    actor.to(device).eval()
    market.to(device).train()

    # 2) 从环境里采集 (s_market, regime_t) 样本
    all_s_market = []
    all_labels   = []

    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            # actor 提取 S_market
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, T, F_in]
            h_temporal = actor.temporal_feature_extractor(st)        # [1, N, D_temp]
            s_market_t = actor.s_market_stock_pool(h_temporal).squeeze(0)  # [D_s_market]

        # 用“均匀持仓”让环境前进一步，获取 info["regime_t"]
        dummy_action = np.ones(env.stock_dim + 1, dtype=np.float32)
        dummy_action /= (env.stock_dim + 1)
        next_state, reward, done, info = env.step(dummy_action)

        lbl = int(info["regime_t"])  # 0 = 牛市, 1 = 熊市
        all_s_market.append(s_market_t.cpu().numpy())
        all_labels.append(lbl)

        state = next_state

    all_s_market = np.stack(all_s_market, axis=0)      # (样本数, D_s_market)
    all_labels   = np.array(all_labels, dtype=np.int64) # (样本数,)
    cnts = np.bincount(all_labels, minlength=2)
    print(f"[Pretrain] 收集到牛熊样本数 = {len(all_labels)}, 分布 (Bull=0,Bear=1) = {cnts}")
    if len(all_labels) == 0:
        raise RuntimeError("预训练时没有任何牛熊样本！")

    # 3) 构造 Dataset + WeightedRandomSampler，做类别均衡
    tensor_s = torch.tensor(all_s_market, dtype=torch.float32)
    tensor_y = torch.tensor(all_labels, dtype=torch.long)
    dataset  = TensorDataset(tensor_s, tensor_y)

    class_counts = cnts.astype(np.float32)
    freq_bull = class_counts[0] if class_counts[0] > 0 else 1.0
    freq_bear = class_counts[1] if class_counts[1] > 0 else 1.0

    sample_weights = np.zeros(len(all_labels), dtype=np.float32)
    sample_weights[all_labels == 0] = 1.0 / freq_bull
    sample_weights[all_labels == 1] = (1.0 / freq_bear)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True
    )

    # 4) 定义 Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss()
    initial_lr = 1e-4  # 建议先用 1e-4，让网络快速“记住”牛/熊模式
    optimizer = torch.optim.Adam(market.parameters(), lr=initial_lr)
    # 每 100 次 optimizer.step() 就让 lr *= 0.5
    scheduler = StepLR(optimizer, step_size=3000, gamma=0.98)

    # 5) 在 wandb 中定义度量
    wandb.define_metric("Pretrain/CE_Loss",     step_metric="pretrain_step")
    wandb.define_metric("Pretrain/Acc_Bull",    step_metric="pretrain_step")
    wandb.define_metric("Pretrain/Acc_Bear",    step_metric="pretrain_step")
    wandb.define_metric("Pretrain/Acc_Overall", step_metric="pretrain_step")

    step = 0
    last_acc_bull = 0.0
    last_acc_bear = 0.0

    # 6) 训练循环：每轮 epoch 结束后检查是否 Early Stop
    for epoch in range(1_000_000):
        # 统计“整轮 epoch”上牛/熊两类的累计正确数和总样本数
        bull_correct_sum = 0
        bull_total_sum   = 0
        bear_correct_sum = 0
        bear_total_sum   = 0

        for batch_s, batch_y in train_loader:
            batch_s = batch_s.to(device)  # [B, D_s_market]
            batch_y = batch_y.to(device)  # [B], ∈ {0,1}

            # Forward
            regime_probs, logits = market(batch_s)  # logits: [B,2]
            loss = criterion(logits, batch_y)

            # 反向传播 & 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率衰减

            # minibatch 级别的统计（可直接打 PRINT 或者 wandb.log）
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)  # [B]

                # bull 类准确率 (label=0)
                mask_bull = (batch_y == 0)
                if mask_bull.sum().item() > 0:
                    correct_bull = ((preds == batch_y) & mask_bull).float().sum().item()
                else:
                    correct_bull = 0.0
                # bear 类准确率 (label=1)
                mask_bear = (batch_y == 1)
                if mask_bear.sum().item() > 0:
                    correct_bear = ((preds == batch_y) & mask_bear).float().sum().item()
                else:
                    correct_bear = 0.0

                # 更新“整轮 epoch”统计
                bull_correct_sum += correct_bull
                bull_total_sum   += mask_bull.sum().item()
                bear_correct_sum += correct_bear
                bear_total_sum   += mask_bear.sum().item()

                # 计算 minibatch 上整体准确率
                overall_acc = (preds == batch_y).float().mean().item()

                # minibatch 级别的 log（你也可以在这里 print 或者 wandb.log）
                wandb.log({
                    "Pretrain/CE_Loss":     loss.item(),
                    "Pretrain/Acc_Bull":    (correct_bull / (mask_bull.sum().item() + 1e-9)),
                    "Pretrain/Acc_Bear":    (correct_bear / (mask_bear.sum().item() + 1e-9)),
                    "Pretrain/Acc_Overall": overall_acc,
                    "pretrain_step": step
                },step = step)

            step += 1
            if step >= max_pretrain_steps:
                print(f"[Pretrain] 达到 max_pretrain_steps={max_pretrain_steps}，强制停止")
                break
            
        epoch_acc_bull = bull_correct_sum / (bull_total_sum + 1e-9)
        epoch_acc_bear = bear_correct_sum / (bear_total_sum + 1e-9)

        # 每个 epoch 结束后打印一次“整轮统计” + 当前 lr
        if epoch % 30 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[Pretrain] epoch={epoch}, "
                f"step={step}, "
                f"EpochAcc_Bull={epoch_acc_bull:.3f}, "
                f"EpochAcc_Bear={epoch_acc_bear:.3f}, "
                f"lr={current_lr:.2e}"
            )

        # 检查 Early Stop 条件：整轮 epoch 上牛/熊都达标
        if (not np.isnan(epoch_acc_bull) and epoch_acc_bull >= target_acc) and \
           (not np.isnan(epoch_acc_bear) and epoch_acc_bear >= target_acc):
            print(
                f"[Pretrain] 【Early Stop】Epoch={epoch}, "
                f"EpochAcc_Bull={epoch_acc_bull:.3f}, "
                f"EpochAcc_Bear={epoch_acc_bear:.3f}"
            )
            break

        if step >= max_pretrain_steps:
            break

    print(
        f"[Pretrain] 结束：共 pretrain_steps={step}, "
        f"FinalEpochAcc_Bull={epoch_acc_bull:.3f}, "
        f"FinalEpochAcc_Bear={epoch_acc_bear:.3f}"
    )

    # 7) Freeze MarketNet 参数，恢复 Actor 为 train
    market.eval()
    for p in market.parameters():
        p.requires_grad = False

    actor.train()

    return market, actor

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
        project="HCAR_test_market",
        name=f"{cfg.net_name}_{cfg.agent_name}_{cfg.optimizer_name}_{cfg.loss_name}_run_{time.time()}",
        config=cfg.to_dict(),
        mode=mode,
        # mode='disabled'

    )
    
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
    wandb.define_metric("Market/*", step_metric="agent_step") # 確保這個也有

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
    market = build_net(cfg.market)
    wandb.watch(act, log="gradients", log_freq=500, log_graph=True)
    wandb.watch(cri, log="gradients", log_freq=500, log_graph=True)

    work_dir = os.path.join(ROOT, cfg.trainer.work_dir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))
    cri_optimizer = build_optimizer(cfg, default_args=dict(params=cri.parameters()))
    market_optimizer = build_optimizer(cfg, default_args=dict(params=market.parameters()))
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
            market=market,
            act_optimizer=act_optimizer,
            cri_optimizer=cri_optimizer,
            market_optimizer=market_optimizer,
            criterion=criterion,
            transition=transition,
            device=device,
            # --- 從 cfg.agent 中獲取 LR 調度參數 ---
            use_lr_scheduler=cfg.agent.get("use_lr_scheduler", False),  # 提供默認值
            warmup_steps=cfg.agent.get("warmup_steps", 0),
        ),
    )

    # ——— 在 Trainer.train_and_valid 之前：先对 MarketNet 做 pretrain ———
    print("—— 开始预训练 MarketNet（二分类，均衡采样）——")

    market_binary ,actor= pretrain_marketnet(
        actor=act,
        market=market.to(device),
        env=train_environment,
        device=device,
        target_acc=0.90,
        max_pretrain_steps=5000000,
        batch_size=32
    )
    print("—— MarketNetBinary 预训练结束 (已 Freeze) ——")

    # 然后再把预训练好的 market_binary 赋回给 agent.market：
    agent.market = market_binary
    agent.act = actor.to(device).train()

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
