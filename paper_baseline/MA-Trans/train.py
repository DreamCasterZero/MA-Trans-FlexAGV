from env import EnvWorkShop
from ppo import Job_Agent, AGV_Agent
import os
import config
from torch.utils.tensorboard import SummaryWriter
import torch
import random
# === 1. 初始化路径 ===
experiment_dir = config.experi_dir()
param_dir = os.path.join(experiment_dir, 'param')
if not os.path.exists(param_dir):
    os.makedirs(param_dir)

writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "runs"))

# === 2. 训练超参 ===
best_avg_makespan = 999999
TOTAL_EPISODES = 200000 # 先跑 1万轮看看收敛
BATCH_EPISODES = 128    # 每 32 局更新一次网络 (On-Policy)
INSTANCE_CHANGE_FREQ = 128
ACTUAL_UPDATES = TOTAL_EPISODES // BATCH_EPISODES

# === 3. 初始化 Agent ===
# input_size=16 (特征维度)
jobagent = Job_Agent(input_size=16, batch_size=BATCH_EPISODES, total_updates=ACTUAL_UPDATES, alpha=2e-5)
agvagent = AGV_Agent(input_size=16, batch_size=BATCH_EPISODES, total_updates=ACTUAL_UPDATES, alpha=2e-5)
jobagent.writer = writer
agvagent.writer = writer

ep_buffer_count = 0 # 计数器，记录攒了多少局数据
lrcount = 0

# === 4. 加载固定验证集 (用于评估) ===
# 假设你的文件在当前目录下的 validation_data 文件夹里
val_path = './validation_data/val_dataset.pt'
if os.path.exists(val_path):
    validation_set = torch.load(val_path)
    print(f"✅ 成功加载验证集，共 {len(validation_set)} 个案例")
else:
    raise FileNotFoundError(f"❌ 找不到验证集: {val_path}，请先运行生成脚本！")


print(f"开始训练! 目标: Fixed 5x8x3 Case. Device: {config.device}")


# === 5. 定义随机训练生成器 ===
def generate_train_instance():
    """生成符合你描述的 5-10 工件规模的随机案例"""
    # 随机范围设置
    n_jobs = random.randint(8, 12)
    n_agvs = 2
    n_machines = 5
    PT = []
    for _ in range(n_jobs):
        num_ops = random.randint(4, 6)
        job_ops = []
        for _ in range(num_ops):
            op_machines = [0] * n_machines
            # 每个工序 2-3 个可选机器
            num_capable = random.randint(1, 5)
            capable_indices = random.sample(range(n_machines), num_capable)
            for m_idx in capable_indices:
                op_machines[m_idx] = random.randint(1, 20)
            job_ops.append(op_machines)
        PT.append(job_ops)

    return PT, n_agvs

print(f"🚀 开始泛化训练! Job:8-12, AGV:2. Device: {config.device}")

# 注意：这里使用 config.num_machines 或直接给 5
myenv = EnvWorkShop(machine_num=config.num_machines)

# 初始化第一个训练案例
current_train_pt, current_train_agv = generate_train_instance()
eval_env = EnvWorkShop(machine_num=5)  # 创建独立的评估环境
# === 4. 主循环 ===
for i in range(TOTAL_EPISODES + 1):
    # --- 【策略核心】每隔 N 轮换一个新题 ---
    # 这样可以让 Agent 在同一个环境下多试几次，梯度更稳定 (模仿 GNN 的做法)
    if i > 0 and i % INSTANCE_CHANGE_FREQ == 0:
        current_train_pt, current_train_agv = generate_train_instance()

    # reset 使用当前生成的案例
    state, padding_mask, job_mask, agv_mask = myenv.reset(
        new_job_data=current_train_pt,
        new_agv_num=current_train_agv
    )
    done = False
    job_score = 0
    agv_score = 0

    while not done:
        # --- A. 工件智能体决策 ---
        # 必须传入 Env 返回的 mask
        jobaction, jobprob, jobval = jobagent.choose_action(state, padding_mask, job_mask, greedy=False)

        # --- B. 环境半步 (计算机器贪心) ---
        # 注意：job_step 返回 mid_state (含Flag=1) 和 AGV 专用 Mask
        mid_state, padding_mask, agv_mask, machaction = myenv.job_step(jobaction)

        # --- C. AGV 智能体决策 ---
        agvaction, agvprob, agvval = agvagent.choose_action(mid_state, padding_mask, agv_mask, greedy=False)
        # --- D. 关键修正：AGV 动作索引映射 ---
        # 智能体输出的是序列索引 (例如工件有5个，AGV是第6个，输出6)
        # 环境需要的是 AGV 物理 ID (例如 1)
        # 所以: real_agv_id = agvaction - num_jobs
        # (注意：jobaction 不需要减，因为工件在序列最前面，索引即 ID)
        real_agv_action = agvaction - myenv.num_jobs

        # --- E. 环境完整一步 ---
        # 返回 7 个值
        next_state, next_padding_mask, next_job_mask, next_agv_mask, reward, done, info = \
            myenv.step(jobaction, machaction, real_agv_action)

        # --- F. 奖励分配 ---
        job_reward = info['rs2'] + info['r_makespan'] * 2.0 + info['rf1']
        agv_reward = info['rs1'] + info['rs2'] + info['r_makespan'] * 2.0 + info['rf2'] + info['rf1']
        # --- G. 存入 Memory ---
        # 注意存入的是原始的 jobaction 和 agvaction (序列索引)，方便 learn 时还原
        jobagent.remember(state, padding_mask, job_mask, jobaction, jobprob, jobval, job_reward, done)
        agvagent.remember(mid_state, padding_mask, agv_mask, agvaction, agvprob, agvval, agv_reward, done)

        # 状态流转
        state = next_state
        padding_mask = next_padding_mask
        job_mask = next_job_mask
        agv_mask = next_agv_mask

        job_score += job_reward
        agv_score += agv_reward

    ep_buffer_count += 1

    # === 6. 网络更新 ===
    if ep_buffer_count >= BATCH_EPISODES:
        jobagent.learn()
        agvagent.learn()
        ep_buffer_count = 0
        lrcount += 1
        # 打印一下当前的训练难度和结果
        if lrcount % 1000 == 0:
            print(f"Ep {i}: Update {lrcount} | Train Makespan: {myenv.FJSP.max:.1f} (Jobs: {len(current_train_pt)})")
    # === 7. 记录与评估 ===
    writer.add_scalar("Train/Makespan", myenv.FJSP.max, i)
    writer.add_scalar("Train/Job_Reward", job_score, i)
    writer.add_scalar("Train/AGV_Reward", agv_score, i)
    # 每 50 轮评估一次
    if i > 0 and i % 100 == 0:
        print(f"\n--- Evaluation @ Ep {i} ---")

        total_eval_makespan = 0
        # eval_env = EnvWorkShop(machine_num=8)  # 创建独立的评估环境

        # 遍历 100 个验证案例
        for case_idx, case_data in enumerate(validation_set):
            # 加载特定案例
            eval_state, p_mask, j_mask, a_mask = eval_env.reset(
                new_job_data=case_data['PT'],
                new_agv_num=case_data['agv_num']
            )
            eval_done = False

            # 单局推理 (Greedy)
            while not eval_done:
                j_act, _, _ = jobagent.choose_action(eval_state, p_mask, j_mask, greedy=True)
                mid_s, p_mask, a_mask, m_act = eval_env.job_step(j_act)
                a_act, _, _ = agvagent.choose_action(mid_s, p_mask, a_mask, greedy=True)
                real_a = a_act - eval_env.num_jobs
                eval_state, p_mask, j_mask, a_mask, _, eval_done, _ = \
                    eval_env.step(j_act, m_act, real_a)

            total_eval_makespan += eval_env.FJSP.max
        # 计算平均分
        avg_eval_score = total_eval_makespan / len(validation_set)

        print(f"Validation Avg Score: {avg_eval_score:.2f} (Best: {best_avg_makespan:.2f})")
        writer.add_scalar("Eval/Avg_Score", avg_eval_score, i)
        # === 保存最佳模型 (基于验证集平均分) ===
        if avg_eval_score < best_avg_makespan:
            best_avg_makespan = avg_eval_score
            print(f"!!! New Best General Model Found !!! Saving...")
            jobagent.save('best')
            agvagent.save('best')

    # 定期保存 Checkpoint
    if i > 0 and i % 1000 == 0:
        jobagent.save(f'ckpt_{i}')
        agvagent.save(f'ckpt_{i}')

writer.close()