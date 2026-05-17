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
├── scheduling/              # Core scheduling module
│   ├── envs/                # Gym-compatible environments
│   ├── algorithms/          # MA-Trans, PPO implementation
│   └── configs/             # Training configurations
├── path_planning/           # Path planning module
│   ├── grid_map/            # Grid map visualization
│   ├── a_star.py            # A* algorithm
│   ├── time_window.py       # Time window based planning
│   └── cbs.py               # Conflict-Based Search
└── ros2_ws/                 # ROS2 workspace (in progress)
    ├── agv_scheduler/       # Scheduling node (C++)
    ├── agv_path_planner/    # Path planning node (C++)
    └── agv_sim/             # Gazebo simulation
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

```bash
# Train the MA-Trans scheduler
python scheduling/train.py

# Run path planning demo
python path_planning/demo.py
```

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