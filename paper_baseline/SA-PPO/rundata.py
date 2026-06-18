from env import EnvWorkShop
from ppo import Job_Agent  # 🔴 改动: SA-PPO 只需要 Job_Agent
import torch
import random
import time
import numpy as np
import pandas as pd
import os

# ================= ⚙️ 配置区域 =================
# 🔴 算法名称
METHOD_NAME = "SA-PPO"

# 🔴 模型路径 (请修改为你存放 SA-PPO 权重的实际文件夹)
MODEL_PATH = './result/10-5-2-50000/param'

# 测试参数 (保持 10x5 规模不变)
TARGET_MACHINES = 5
TARGET_AGVS = 2
JOB_RANGE = (10, 10)
OPS_RANGE = (4, 6)
TIME_RANGE = (1, 20)
TOTAL_EPISODES = 100
BATCH_EPISODES = 128

print(f"🚀 开始测试 [{METHOD_NAME}] 的推理速度...")
print(f"📄 结果将保存为: inference_time_{METHOD_NAME}.csv")


# === 数据生成 (保持模板一致) ===
def generate_train_instance():
    n_jobs = random.randint(JOB_RANGE[0], JOB_RANGE[1])
    n_machines = TARGET_MACHINES
    n_agvs = TARGET_AGVS
    PT = []
    for _ in range(n_jobs):
        num_ops = random.randint(OPS_RANGE[0], OPS_RANGE[1])
        job_ops = []
        for _ in range(num_ops):
            op_machines = [0] * n_machines
            num_capable = random.randint(1, n_machines)
            capable_indices = random.sample(range(n_machines), num_capable)
            for m_idx in capable_indices:
                op_machines[m_idx] = random.randint(TIME_RANGE[0], TIME_RANGE[1])
            job_ops.append(op_machines)
        PT.append(job_ops)
    total_ops_count = sum([len(job) for job in PT])
    return PT, n_agvs, total_ops_count


# === 初始化 Agent ===
# 🔴 改动: SA-PPO 只初始化 Job Agent
jobagent = Job_Agent(input_size=24, batch_size=BATCH_EPISODES, total_updates=1, alpha=3e-4)

# 尝试加载权重 (针对 SA-PPO)
try:
    # 你的权重文件名可能是 job_actor_sa_ppo_best.pt 或 job_actor_best.pt，请根据实际情况调整
    weight_file = f'{MODEL_PATH}/job_actor_sa_ppo_best.pt'
    if not os.path.exists(weight_file):
        weight_file = f'{MODEL_PATH}/job_actor_best_10x5.pt'  # 备用尝试

    jobagent.job_actor.load_state_dict(torch.load(weight_file))
    print(f"✅ 已加载 Job Agent 权重: {weight_file}")
except:
    print("⚠️ 未加载权重，使用随机初始化进行速度测试 (不影响速度结果)")

jobagent.job_actor.eval()
# AGV Agent 不需要初始化

myenv = EnvWorkShop(machine_num=TARGET_MACHINES)

# === 📊 数据收集 ===
results_ms_per_op = []

# === 主循环 ===
current_train_pt, current_train_agv, current_total_ops = generate_train_instance()

for i in range(1, TOTAL_EPISODES + 1):
    state, padding_mask, job_mask, agv_mask = myenv.reset(
        new_job_data=current_train_pt,
        new_agv_num=current_train_agv
    )

    done = False
    ep_inference_time = 0.0  # 单轮纯推理时间

    with torch.no_grad():
        while not done:
            # ⏱️ 1. Job Agent 计时 (神经网络)
            t_start = time.time()
            jobaction, _, _ = jobagent.choose_action(state, padding_mask, job_mask, greedy=True)
            # if torch.cuda.is_available(): torch.cuda.synchronize()
            ep_inference_time += (time.time() - t_start)

            # 环境交互 (不计时)
            # 注意: job_step 返回 machaction
            mid_state, padding_mask, agv_mask, machaction = myenv.job_step(jobaction)

            # ⏱️ 2. AGV 决策计时 (启发式规则)
            # 🔴 改动: 这里测的是规则的计算时间，这也属于"推理"的一部分
            t_start = time.time()
            agv_action_1based = myenv.get_heuristic_agv_action(jobaction)
            ep_inference_time += (time.time() - t_start)

            # 环境交互 (不计时)
            # 🔴 改动: 规则返回的已经是 1-based ID，直接传入 step 即可
            next_state, next_padding_mask, next_job_mask, next_agv_mask, _, done, _ = \
                myenv.step(jobaction, machaction, agv_action_1based)

            state = next_state
            padding_mask = next_padding_mask
            job_mask = next_job_mask
            agv_mask = next_agv_mask

    # === 计算本轮指标 ===
    if current_total_ops > 0:
        ms_per_op = (ep_inference_time / current_total_ops) * 1000
        results_ms_per_op.append(ms_per_op)

    if i % 20 == 0:
        print(f"进度: {i}/{TOTAL_EPISODES} | 当前速度: {ms_per_op:.4f} ms/op")

    # 换个题继续
    current_train_pt, current_train_agv, current_total_ops = generate_train_instance()

# === 💾 保存文件 ===
avg_speed = np.mean(results_ms_per_op)
std_speed = np.std(results_ms_per_op)

df = pd.DataFrame({'Value': results_ms_per_op})
filename = f"inference_time_{METHOD_NAME}.csv"
df.to_csv(filename, index=False)

print("\n" + "=" * 40)
print(f"✅ 测试完成！文件已保存: {filename}")
print(f"   样本数: {len(results_ms_per_op)}")
print(f"   平均值: {avg_speed:.4f} ms/op")
print(f"   标准差: {std_speed:.4f} ms/op")
print("=" * 40)