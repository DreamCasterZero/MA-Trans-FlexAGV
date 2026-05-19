import torch
from ..base.base_config import BaseConfig

class SchedulingConfig(BaseConfig):

    class env:
        num_machines = 5
        max_seq_len = 50        # Token 序列上限（工件数 + AGV 数）
        feat_dim = 24           # 每个 Token 的特征维度
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class problem:
        num_agvs = 2
        num_jobs_min = 8
        num_jobs_max = 12
        num_ops_min = 4         # 每个工件的最少工序数
        num_ops_max = 6
        proc_time_min = 1       # 工序加工时间下限
        proc_time_max = 20
        num_capable_min = 1     # 每道工序可选机器数下限
        num_capable_max = 5

    class obs:
        norm_time = 200.0
        norm_dist = 100.0
        norm_count = 20.0
        norm_agv_count = 5.0    # AGV 数量归一化上限
        job_type = [1, 0]       # Token 类型头，工件
        agv_type = [0, 1]       # Token 类型头，AGV

    class reward:
        transport_coef = 0.01   # AGV 空载行驶惩罚系数（对应 rs1）
        wait_coef = 0.02        # AGV 等待工件惩罚系数（对应 rs2）
        makespan_coef = 1.0     # Makespan 增量惩罚系数
        completion_bonus = 50.0 # 全部完工奖励（对应 rf1）
        balance_coef = 0.5      # AGV 负载均衡惩罚系数（对应 rf2）
        job_makespan_scale = 2.0  # job 智能体的 makespan 额外权重
        agv_makespan_scale = 2.0  # AGV 智能体的 makespan 额外权重

    class transport:
        # 行列索引对应位置编号：0 = 原料区/仓库，1~N = 机器
        agv_trans = [
            [0,  12, 14,  2, 12, 11, 16, 15, 12,  8,  3],
            [12,  0, 11, 10, 13, 16, 18,  9,  6, 18,  9],
            [14, 11,  0, 14,  4,  9,  8,  2,  4, 14, 12],
            [2,  10, 14,  0, 12, 12, 17, 14, 10, 10,  2],
            [12, 13,  4, 12,  0,  4,  5,  6,  7, 10, 10],
            [11, 16,  9, 12,  4,  0,  5, 11, 11,  6, 11],
            [16, 18,  8, 17,  5,  5,  0, 10, 11, 11, 15],
            [15,  9,  2, 14,  6, 11, 10,  0,  4, 16, 12],
            [12,  6,  4, 10,  7, 11, 11,  4,  0, 14,  8],
            [8,  18, 14, 10, 10,  6, 11, 16, 14,  0, 10],
            [3,   9, 12,  2, 10, 11, 15, 12,  8, 10,  0],
        ]


class SchedulingConfigAlgo(BaseConfig):
    seed = 42

    class runner:
        total_episodes = 50000
        batch_episodes = 128
        instance_change_freq = 128
        experiment_name = 'fjsp_agv'
        save_interval = 1000        # 每多少 episode 保存一次 checkpoint
        log_interval  = 10          # 每多少 episode 打印一次训练日志
        resume = False
        checkpoint_path = None      # 断点文件路径，resume=True 时使用

    class ppo:
        n_epochs = 5
        policy_clip = 0.2
        gamma = 0.99
        gae_lambda = 0.95
        learning_rate = 2e-5
        value_loss_coef = 0.5       # critic loss 权重
        max_grad_norm = 1.0         # 梯度裁剪上限
        ent_coef_start = 0.05
        ent_coef_end = 0.01

    class network:
        embed_size = 128
        num_heads = 4
        num_layers = 3
        dropout_rate = 0.1


