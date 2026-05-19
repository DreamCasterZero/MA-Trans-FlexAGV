# MA-Trans-FlexAGV

> Industrial Multi-AGV Fleet Scheduling System based on MA-Trans (Multi-Agent Transformer + PPO) with Spatio-temporal Path Planning and ROS2 Integration

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-green.svg)](https://docs.ros.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements an industrial multi-AGV scheduling and dispatching system. The core scheduling algorithm is based on **MA-Trans**, a Multi-Agent Transformer architecture combined with PPO reinforcement learning, originally proposed in our SCI paper published in *Applied Soft Computing* (JCR Q1).

The system integrates:
- **Task Allocation**: MA-Trans multi-agent reinforcement learning scheduler
- **Path Planning**: Spatio-temporal conflict-free path planning (A\*, Time Window, CBS)
- **Visualization**: Real-time Gantt chart and grid map display
- **Simulation**: ROS2 + Gazebo multi-AGV simulation environment

## System Architecture

```
MA-Trans-FlexAGV/
├── scheduling/                  # 调度核心模块（本项目主体）
│   ├── envs/
│   │   ├── base/                # BaseConfig / BaseTask 基类
│   │   └── custom/              # SchedulingEnv + SchedulingConfig
│   ├── algo/
│   │   └── ppo/                 # ActorNetwork / CriticNetwork / PPOAgent / Runner
│   ├── utils/                   # task_registry / helpers
│   ├── scripts/                 # 训练 / 评估 / 数据集生成入口
│   ├── data/
│   │   ├── val/                 # 验证集 (.pt)
│   │   └── test/                # 测试集 (.pt)
│   └── logs/                    # TensorBoard 日志 & 模型权重
├── path_planning/               # 路径规划模块（解耦独立）
└── ros2_ws/                     # ROS2 仿真工作空间（进行中）
```


## Related Paper

> **Transformer-based Multi-Agent Reinforcement Learning for Flexible Job Shop Scheduling with AGVs**  
> Shen Yuxiang, et al.  
> *Applied Soft Computing*, JCR Q1, 2026. [[Paper](https://doi.org/10.1016/j.asoc.2026.114899)]

## Demo

*Coming soon*

## Installation

```bash
git clone https://github.com/DreamCasterZero/MA-Trans-FlexAGV.git
cd MA-Trans-FlexAGV
pip install -r requirements.txt
```

## Quick Start

> 所有命令均在项目根目录 `MA-Trans-FlexAGV/` 下执行。

### 1. 生成数据集

训练前先生成固定验证集（只需做一次），测试集可按需生成多个规模。

```bash
# 生成验证集（100 个案例，保存至 scheduling/data/val/）
python -m scheduling.scripts.generate_dataset --split val --num_cases 100

# 生成测试集（200 个案例）
python -m scheduling.scripts.generate_dataset --split test --num_cases 200

# 自定义文件名
python -m scheduling.scripts.generate_dataset --split test --num_cases 200 --name fjsp_agv_200_large
```

生成结果保存在：
```
scheduling/data/
  val/  fjsp_agv_100.pt
  test/ fjsp_agv_200.pt
```

---

### 2. 训练

```bash
# 基础训练（使用 config 默认轮数）
python -m scheduling.scripts.train

# 指定训练轮数
python -m scheduling.scripts.train --total_episodes 50000

# 带验证集训练（每 100 ep 评估一次，自动保存最优模型）
python -m scheduling.scripts.train \
    --total_episodes 50000 \
    --val_set scheduling/data/val/fjsp_agv_100.pt

# 调整日志打印频率（默认每 10 ep 打印一次）
python -m scheduling.scripts.train --total_episodes 50000 --log_interval 20

# 断点续训
python -m scheduling.scripts.train \
    --resume \
    --checkpoint scheduling/logs/fjsp_agv/20240101_120000/model_5000.pt \
    --val_set scheduling/data/val/fjsp_agv_100.pt

# 调试模式（不写日志，快速跑几百轮验证流程）
python -m scheduling.scripts.train --total_episodes 500 --log_root none
```

训练日志与模型自动保存在 `scheduling/logs/fjsp_agv/<时间戳>/`：
```
scheduling/logs/fjsp_agv/20240101_120000/
  model_best.pt          # 验证集最优模型
  model_1000.pt          # 每 save_interval 轮的检查点
  events.out.tfevents.*  # TensorBoard 日志
```

训练过程输出示例：
```
[Ep      70/50000 | Update     0/390]
  ETA      : 08:19:40  (~29980 s)
  Makespan : 360.20    Time/ep : 0.1458 s
  ────────────────────────────────────────────────────
                     transport  agv_wait  makespan  complet  balance      total
  Job  (rs2+ms+rf1)         —   -0.0234   -0.5420   0.0000       —   -699.9120
  AGV  (all)           -0.0120  -0.0234   -0.5420   0.0000  -0.0050  -703.3000
```

---

### 3. 评估 / 测试

```bash
# 用固定测试集评估
python -m scheduling.scripts.play \
    --checkpoint scheduling/logs/fjsp_agv/20240101_120000/model_best.pt \
    --test_set scheduling/data/test/fjsp_agv_200.pt

# 随机生成 100 个案例快速测试
python -m scheduling.scripts.play \
    --checkpoint scheduling/logs/fjsp_agv/20240101_120000/model_best.pt \
    --num_cases 100

# 保存每个案例的 makespan 到 CSV
python -m scheduling.scripts.play \
    --checkpoint scheduling/logs/fjsp_agv/20240101_120000/model_best.pt \
    --test_set scheduling/data/test/fjsp_agv_200.pt \
    --save_csv results.csv
```

输出示例：
```
Results (200 cases):
  Mean :  142.35
  Std  :   18.21
  Min  :  105.00
  Max  :  198.50
```

---

### 4. 查看 TensorBoard

```bash
tensorboard --logdir scheduling/logs/fjsp_agv
```

---

### 5. 关键超参数位置

| 参数 | 文件 | 字段 |
|------|------|------|
| 训练轮数 | `scheduling/envs/custom/scheduling_config.py` | `runner.total_episodes` |
| 批大小 | 同上 | `runner.batch_episodes` |
| 日志频率 | 同上 | `runner.log_interval` |
| 学习率 | 同上 | `ppo.learning_rate` |
| 网络结构 | 同上 | `network.embed_size / num_heads / num_layers` |
| 奖励系数 | 同上 | `reward.*_coef` |
| 机器数 / AGV 数 | 同上 | `env.num_machines` / `problem.num_agvs` |

## Features

- [x] MA-Trans multi-agent scheduling algorithm
- [x] Spatio-temporal conflict-free path planning
- [x] Gantt chart visualization
- [ ] ROS2 integration (in progress)
- [ ] Gazebo simulation (in progress)

## License

MIT License

## Author

**Shen Yuxiang** | Harbin Institute of Technology  
📧 syx743@foxmail.com

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{shen2026transformer,
  title={Transformer-based multi-agent reinforcement learning for flexible job shop scheduling with AGVs},
  author={Shen, Yuxiang and Zhang, Xutang and Jin, Tianguo},
  journal={Applied Soft Computing},
  pages={114899},
  year={2026},
  publisher={Elsevier}
}
```