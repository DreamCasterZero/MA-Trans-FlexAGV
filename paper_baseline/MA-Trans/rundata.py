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

myenv = EnvWorkShop(machine_num=args.machines)

results_ms_per_op = []

# ================= 主循环 =================
for i in range(1, args.episodes + 1):
    current_train_pt, current_train_agv, current_total_ops = generate_train_instance(args.jobs, args.machines, args.agvs)
    
    state, padding_mask, job_mask, agv_mask = myenv.reset(
        new_job_data=current_train_pt,
        new_agv_num=current_train_agv
    )
    
    done = False
    ep_inference_time = 0.0  # ✅ 修正：必须在这里初始化！

    with torch.no_grad():
        while not done:
            # --- 1. Job Agent 计时 ---
            # 如果是 GPU，必须先同步，否则测的是 kernel 启动时间
            if args.device == 'cuda': torch.cuda.synchronize()
            
            t_start = time.perf_counter() # ✅ 优化：用 perf_counter 更精准
            
            # 执行动作选择
            jobaction, _, _ = jobagent.choose_action(state, padding_mask, job_mask, greedy=True)
            
            if args.device == 'cuda': torch.cuda.synchronize()
            ep_inference_time += (time.perf_counter() - t_start)

            # 环境交互 (不计时)
            mid_state, padding_mask, agv_mask, machaction = myenv.job_step(jobaction)

            # --- 2. AGV Agent 计时 ---
            if args.device == 'cuda': torch.cuda.synchronize()
            t_start = time.perf_counter()
            
            agvaction, _, _ = agvagent.choose_action(mid_state, padding_mask, agv_mask, greedy=True)
            
            if args.device == 'cuda': torch.cuda.synchronize()
            ep_inference_time += (time.perf_counter() - t_start)

            # 环境交互 (不计时)
            real_agv_action = agvaction - myenv.num_jobs
            next_state, next_padding_mask, next_job_mask, next_agv_mask, _, done, _ = \
                myenv.step(jobaction, machaction, real_agv_action)

            state = next_state
            padding_mask = next_padding_mask
            job_mask = next_job_mask
            agv_mask = next_agv_mask

    # 计算 ms/op
    if current_total_ops > 0:
        ms_per_op = (ep_inference_time / current_total_ops) * 1000
        results_ms_per_op.append(ms_per_op)

    if i % 20 == 0:
        print(f"进度: {i}/{args.episodes} | 当前速度: {ms_per_op:.4f} ms/op")

# ================= 保存结果 =================
# 文件名包含规模信息，防止覆盖
filename = f"time_{args.method}_{args.jobs}x{args.machines}x{args.agvs}.csv"
df = pd.DataFrame({'Value': results_ms_per_op})
df.to_csv(filename, index=False)

print(f"✅ 保存完毕: {filename}")
print(f"   平均耗时: {np.mean(results_ms_per_op):.4f} ms/op")