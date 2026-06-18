import argparse
import time
import numpy as np
import pandas as pd
import torch
import random
import os

# 假设你的环境和智能体在这些文件里
from env import EnvWorkShop
from ppo import Job_Agent, AGV_Agent

# ================= ⚙️ 命令行参数配置 =================
parser = argparse.ArgumentParser(description="Inference Time Test")
parser.add_argument('--method', type=str, default="MA-Trans", help="算法名称")
parser.add_argument('--jobs', type=int, default=20, help="工件数量")
parser.add_argument('--machines', type=int, default=10, help="机器数量")
parser.add_argument('--agvs', type=int, default=5, help="AGV数量")
parser.add_argument('--episodes', type=int, default=100, help="测试轮数")
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

print(f"🚀 开始测试 [{args.method}] | 规模: {args.jobs}x{args.machines} (AGV={args.agvs})")
print(f"⚙️ 设备: {args.device}")

# ================= 数据生成函数 =================
def generate_train_instance(n_jobs, n_machines, n_agvs):
    # 根据你的逻辑生成数据
    PT = []
    # 这里稍微简化一下范围，让它围绕指定规模波动，或者你可以固定死
    OPS_RANGE = (4, 6) 
    TIME_RANGE = (1, 20)
    
    for _ in range(n_jobs):
        num_ops = random.randint(OPS_RANGE[0], OPS_RANGE[1])
        job_ops = []
        for _ in range(num_ops):
            op_machines = [0] * n_machines
            # 随机生成每道工序可用的机器
            num_capable = random.randint(1, n_machines)
            capable_indices = random.sample(range(n_machines), num_capable)
            for m_idx in capable_indices:
                op_machines[m_idx] = random.randint(TIME_RANGE[0], TIME_RANGE[1])
            job_ops.append(op_machines)
        PT.append(job_ops)
    total_ops_count = sum([len(job) for job in PT])
    return PT, n_agvs, total_ops_count

# ================= 初始化 =================
# 注意：如果你的 input_size 会随机器数量变化，请在这里动态调整
# 这里假设 input_size=24 是固定的，如果不是请修改
jobagent = Job_Agent(input_size=24, batch_size=1, total_updates=1, alpha=3e-4)
agvagent = AGV_Agent(input_size=24, batch_size=1, total_updates=1, alpha=3e-4)

model_dir = f'./result/{args.jobs}-{args.machines}-{args.agvs}/param'  # 根据你的文件夹命名规则修改
job_model_path = os.path.join(model_dir, 'job_agent_best_20x10.pt') # 或者是 .pt
agv_model_path = os.path.join(model_dir, 'agv_agent_best_20x10.pt')

print(f"🔍 尝试加载权重: {job_model_path}")

if os.path.exists(job_model_path) and os.path.exists(agv_model_path):
    # 注意：一定要 map_location 到 args.device，否则可能报错
    # job_state = torch.load(job_model_path, map_location=args.device)
    # agv_state = torch.load(agv_model_path, map_location=args.device)
    # # 针对 PPO 常见的保存方式（有时保存的是整个 state_dict，有时只保存了 actor）
    # # 如果你的保存代码是 torch.save(agent.job_actor.state_dict(), ...)
    # jobagent.job_actor.load_state_dict(job_state)
    # agvagent.agv_actor.load_state_dict(agv_state)

    jobagent.load_snapshot(job_model_path)
    agvagent.load_snapshot(agv_model_path)

    print("✅ 成功加载训练权重！(推理速度应当与随机权重一致)")
else:
    print(f"⚠️ 未找到权重文件 (路径: {model_dir})")
    print("⚠️ 将使用随机初始化的权重进行测试 (这对静态图模型的速度测试是有效的，不用担心)")

# 移动到 GPU (一定要在加载权重之后，或者加载时指定 map_location)
jobagent.job_actor.to(args.device)
agvagent.agv_actor.to(args.device)

jobagent.job_actor.eval()
agvagent.agv_actor.eval()

# ================= 🔴 新增：计算参数量和 FLOPs (修正版) =================
try:
    from thop import profile, clever_format
    
    print("\n" + "="*20 + " 📊 模型复杂度统计 " + "="*20)
    
    # 1. 构造 Dummy Input
    # 形状必须严格对应 env.py: (Batch=1, Seq_Len=50, Dim=24)
    dummy_seq_len = 50
    dummy_input_dim = 24
    
    # (1) State
    dummy_state = torch.randn(1, dummy_seq_len, dummy_input_dim).to(args.device)
    
    # (2) Padding Mask (全 False 代表所有节点有效，计算量最大)
    dummy_mask = torch.zeros(1, dummy_seq_len, dtype=torch.bool).to(args.device)
    
    # (3) Action Mask (全 True 代表所有动作可选，防止内部 assert 报错) [新增修正点!]
    dummy_action_mask = torch.ones(1, dummy_seq_len, dtype=torch.bool).to(args.device)
    
    # 2. 统计 Job Agent
    # inputs=(state, padding_mask, action_mask) <--- 顺序必须和 forward 一致
    job_flops, _ = profile(jobagent.job_actor, inputs=(dummy_state, dummy_mask, dummy_action_mask), verbose=False)
    job_params = sum(p.numel() for p in jobagent.job_actor.parameters())
    
    # 3. 统计 AGV Agent
    agv_flops, _ = profile(agvagent.agv_actor, inputs=(dummy_state, dummy_mask, dummy_action_mask), verbose=False)
    agv_params = sum(p.numel() for p in agvagent.agv_actor.parameters())
    
    # 4. 汇总
    total_flops = job_flops + agv_flops
    total_params = job_params + agv_params
    
    # 5. 格式化输出
    flops_str, params_str = clever_format([total_flops, total_params], "%.3f")
    
    print(f"Model: {args.method} (Dual-Agent)")
    print(f"-----------------------------------")
    print(f"Job Agent Params: {job_params/1e6:.3f} M")
    print(f"AGV Agent Params: {agv_params/1e6:.3f} M")
    print(f"-----------------------------------")
    print(f"👉 Total Parameters: {params_str} (请填入 Table 7)")
    print(f"👉 Total FLOPs:      {flops_str} (请填入 Table 7)")
    print("="*60 + "\n")

except Exception as e:
    print(f"⚠️ 无法计算 FLOPs，跳过此步骤。原因: {e}")
    # 打印详细错误栈以便调试
    import traceback
    traceback.print_exc()

# ================= 🔴 结束新增部分 =================
