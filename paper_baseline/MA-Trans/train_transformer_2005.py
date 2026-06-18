from env import EnvWorkShop
from ppo import Job_Agent, AGV_Agent
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import time  # <--- 顶部引入
import numpy as np



torch.set_float32_matmul_precision('high')

experiment_dir = f'TrainingResult'


TARGET_MACHINES = 5      # 必须与 GNN 一致
TARGET_AGVS = 3          # 模拟硬件限制，固定为 2
JOB_RANGE = (16, 24)      # 【你的王牌】变长训练，体现泛化性
OPS_RANGE = (4, 6)       # 对齐 GNN 数据分布 (num_mas * 0.8 ~ 1.2)
TIME_RANGE = (1, 20)     # 对齐 GNN 数据分布

VAL_FOLDER_PATH = './validation_data/2005'

START_EPISODE = 0
TOTAL_EPISODES = 100000   # 小规模问题，5万轮通常足够收敛
BATCH_EPISODES = 128     # Update frequency
current_learn_step = (START_EPISODE // BATCH_EPISODES) * 5

param_dir = os.path.join(experiment_dir, 'param')
if not os.path.exists(param_dir):
    os.makedirs(param_dir)

writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "runs_lstm_20x5"))

best_avg_makespan = 999999.0
# 计算总更新步数 (向上取整防止除零)
total_update_steps = (TOTAL_EPISODES // BATCH_EPISODES) + 10
# === 2. FJS 文件解析工具 (内置) ===
def load_fjs_file(file_path, total_machines):
    """读取标准 .fjs 文件并转换为环境需要的 PT 矩阵"""
    PT = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过第一行 Header
            line = line.strip()
            if not line: continue

            data = list(map(int, line.split()))
            num_ops = data[0]
            idx = 1
            job_ops = []

            for _ in range(num_ops):
                op_machines = [0] * total_machines
                num_capable = data[idx]
                idx += 1
                for _ in range(num_capable):
                    m_id = data[idx] - 1  # 1-based -> 0-based
                    time = data[idx + 1]
                    idx += 2
                    if m_id < total_machines:
                        op_machines[m_id] = time
                job_ops.append(op_machines)
            PT.append(job_ops)
    return PT

def load_validation_set(folder_path, n_machines, n_agvs):
    """加载文件夹下所有 .fjs 文件"""
    dataset = []
    # 使用 glob 查找所有 .fjs 文件
    files = glob.glob(os.path.join(folder_path, "*.fjs"))

    if len(files) == 0:
        print(f"❌ 警告: 在 {folder_path} 下没有找到 .fjs 文件！")
        return []

    print(f"📂 正在加载 GNN 标准验证集 (from {folder_path})...")
    for file in files:
        try:
            pt = load_fjs_file(file, n_machines)
            # ⚠️ 关键：手动补全 AGV 数量，必须固定为 2
            dataset.append({'PT': pt, 'agv_num': n_agvs})
        except Exception as e:
            print(f"⚠️ 跳过损坏文件 {file}: {e}")

    print(f"✅ 成功加载 {len(dataset)} 个验证案例 (AGV={n_agvs})")
    return dataset

# === 3. 初始化 Agent ===
jobagent = Job_Agent(input_size=24, batch_size=BATCH_EPISODES, total_updates=total_update_steps, alpha=2e-4, initial_step=current_learn_step)
agvagent = AGV_Agent(input_size=24, batch_size=BATCH_EPISODES, total_updates=total_update_steps, alpha=2e-4, initial_step=current_learn_step)
jobagent.writer = writer
agvagent.writer = writer

# ================= 🔄 新增：断点续训设置 =================
RESUME = False        # 🔴 开关：如果要继续训练，改为 True
if RESUME:
    # 定义断点路径 (根据你的命名规则)
    job_ckpt_path = f'./TrainingResult/param/job_agent_{START_EPISODE}.pt'
    agv_ckpt_path = f'./TrainingResult/param/agv_agent_{START_EPISODE}.pt'

    # 检查文件是否存在
    if os.path.exists(job_ckpt_path) and os.path.exists(agv_ckpt_path):
        print(f"🚀 发现断点，正在恢复训练进度: Episode {START_EPISODE}")

        # 调用你在 ppo.py 里写好的加载函数
        jobagent.load_snapshot(job_ckpt_path)
        agvagent.load_snapshot(agv_ckpt_path)

        updates_already_done = START_EPISODE // BATCH_EPISODES
        if hasattr(jobagent, 'current_update'):
            jobagent.current_update = updates_already_done
            print(f"🔄 Job Agent 学习率步数已恢复至: {updates_already_done}")
        if hasattr(agvagent, 'current_update'):
            agvagent.current_update = updates_already_done
            print(f"🔄 AGV Agent 学习率步数已恢复至: {updates_already_done}")

        print("✅ 模型权重与优化器状态已恢复！")
    else:
        print(f"⚠️ 未找到断点文件: {job_ckpt_path}，将从头开始训练。")
        START_EPISODE = 0

# === 4. 加载验证集 ===
# 如果文件夹不存在，请务必先去复制 GNN 的数据！
if os.path.exists(VAL_FOLDER_PATH):
    validation_set = load_validation_set(VAL_FOLDER_PATH, TARGET_MACHINES, TARGET_AGVS)
else:
    print(f"错误: 找不到文件夹 {VAL_FOLDER_PATH}")
    print("请把 GNN 项目里的 data_dev/1005 文件夹复制到这里！")
    exit()
# === 5. 定义训练数据生成器 (模仿 GNN 分布) ===
def generate_train_instance():
    """
    生成 10x5 规模的训练数据
    关键策略：工件数在 8-12 波动，但工序特征和 AGV 保持与 GNN 一致
    """
    # 1. 工件数量：泛化训练的核心 (8-12)
    n_jobs = random.randint(JOB_RANGE[0], JOB_RANGE[1])

    # 2. 硬件锁定
    n_machines = TARGET_MACHINES
    n_agvs = TARGET_AGVS

    PT = []
    for _ in range(n_jobs):
        # 对齐 GNN 数据特征 (4-6道工序)
        num_ops = random.randint(OPS_RANGE[0], OPS_RANGE[1])

        job_ops = []
        for _ in range(num_ops):
            op_machines = [0] * n_machines

            # 部分柔性模拟：随机选 1 到 n_machines 个机器可加工
            num_capable = random.randint(1, n_machines)
            capable_indices = random.sample(range(n_machines), num_capable)

            for m_idx in capable_indices:
                # 对齐 GNN 加工时间 (1-20)
                op_machines[m_idx] = random.randint(TIME_RANGE[0], TIME_RANGE[1])

            job_ops.append(op_machines)
        PT.append(job_ops)

    total_ops_count = sum([len(job) for job in PT])
    return PT, n_agvs, total_ops_count  # <--- 多返回一个 total_ops_count

print(f"开始 GNN 阻击战训练! Jobs:{JOB_RANGE}, Ops:{OPS_RANGE}, AGV:{TARGET_AGVS}")

# === 6. 训练主循环 ===
myenv = EnvWorkShop(machine_num=TARGET_MACHINES)
eval_env = EnvWorkShop(machine_num=TARGET_MACHINES) # 评估环境

# 主循环外
current_train_pt, current_train_agv, current_total_ops = generate_train_instance() # <--- 接收 3 个值
ep_buffer_count = 0
lrcount = 0
# ... 在 myenv = EnvWorkShop(...) 之后 ...
start_time = time.time()
frame_count = 0  # 用于计算 FPS
inference_time_buffer = 0  # <--- 🟢【新增】初始化推理时间缓存
for i in range(START_EPISODE, TOTAL_EPISODES + 1):

    # 每 128 轮换一道新题，保持梯度稳定
    if i > 0 and i % 128 == 0:
        current_train_pt, current_train_agv, current_total_ops = generate_train_instance()

    state, padding_mask, job_mask, agv_mask = myenv.reset(
        new_job_data=current_train_pt,
        new_agv_num=current_train_agv
    )
    # 🟢【新增】开始计时 (纯推理)
    t0_inference = time.time()
    done = False
    job_score = 0
    agv_score = 0
    with torch.no_grad():
        while not done:
            # A. Job Agent
            jobaction, jobprob, jobval = jobagent.choose_action(state, padding_mask, job_mask, greedy=False)
            # B. Env Half Step
            mid_state, padding_mask, agv_mask, machaction = myenv.job_step(jobaction)
            # C. AGV Agent
            agvaction, agvprob, agvval = agvagent.choose_action(mid_state, padding_mask, agv_mask, greedy=False)
            # D. Map Action (索引转ID)
            real_agv_action = agvaction - myenv.num_jobs
            # E. Env Full Step
            next_state, next_padding_mask, next_job_mask, next_agv_mask, reward, done, info = \
                myenv.step(jobaction, machaction, real_agv_action)
            # F. Rewards (为了收敛快，可以用最简单的 MakeSpan 奖励)
            # rs1: 运输成本 (负数，绝对值越小越好)
            # rs2: AGV等待成本 (负数)
            # r_makespan: 完工时间增量惩罚 (负数)
            rs1 = info['rs1']
            rs2 = info['rs2']
            rf1 = info['rf1']
            rf2 = info['rf2']
            r_makespan = info['r_makespan']
            makespan_penalty = r_makespan * 5.0
            # 如果 rs1 < 0，说明 AGV 产生了位移（干活了）
            # 给它一点正向反馈，防止它为了避免 rs1 而选择死等最近的工件
            # agv_active_bonus = 0.01 if rs1 < 0 else 0.0
            # Job 奖励重点关注完工时间
            job_reward = rs2 + rf1 + makespan_penalty
            # AGV 奖励重点关注：动起来 + 完工快
            agv_reward = rs1 + rs2 + rf2 + rf1 + makespan_penalty # + agv_active_bonus
            # job_reward = info['rs2'] + info['r_makespan'] * 2.0 + info['rf1']
            # agv_reward = info['rs1'] + info['rs2'] + info['r_makespan'] * 2.0 + info['rf2'] + info['rf1']
            # G. Remember
            jobagent.remember(state, padding_mask, job_mask, jobaction, jobprob, jobval, job_reward, done)
            agvagent.remember(mid_state, padding_mask, agv_mask, agvaction, agvprob, agvval, agv_reward, done)
            state = next_state
            padding_mask = next_padding_mask
            job_mask = next_job_mask
            agv_mask = next_agv_mask
            job_score += job_reward
            agv_score += agv_reward
    # 🟢【新增】结束计时并累加
    inference_time_buffer += (time.time() - t0_inference)
    ep_buffer_count += 1
    # === 更新网络 ===
    if ep_buffer_count >= BATCH_EPISODES:
        jobagent.learn()
        agvagent.learn()
        ep_buffer_count = 0
        lrcount += 1

        if lrcount % 100 == 0:
            print(f"Ep {i} | Train Makespan: {myenv.FJSP.max:.1f} (JobNum: {len(current_train_pt)})")

    # === 记录日志 (每10轮记一次，减少IO) ===
    if i % 10 == 0:
        writer.add_scalar("Train/Raw_Makespan", myenv.FJSP.max, i)
        # 🟢【新增】记录纯推理时间 (排除 learn 的时间)
        # 这能证明你的模型实际跑起来很快，慢是因为在训练
        avg_inference_time = inference_time_buffer / 10.0
        writer.add_scalar("Efficiency/Pure_Inference_Time_Sec", avg_inference_time, i)
        inference_time_buffer = 0  # 归零，为下个 10 轮做准备
        # 2. 🟢【针对意见 #11】奖励成分拆解 (解释性分析)
        # 能够展示模型是先学会了“不等待”，还是先学会了“跑得快”
        writer.add_scalar("Reward_Details/Transport_Cost_rs1", info['rs1'], i)
        writer.add_scalar("Reward_Details/Waiting_Penalty_rs2", info['rs2'], i)
        writer.add_scalar("Reward_Details/Makespan_Reward", info['r_makespan'], i)
        writer.add_scalar("Reward_Details/Completion_Bonus", info['rf1'], i)
        # 3. 🟢【针对意见 #16】计算效率监控
        # 计算当前的 FPS (Frames Per Second)
        current_time = time.time()
        elapsed_time = current_time - start_time
        # 这里 frame_count 是你这 10 个 episode 跑的总 step 数，
        # 如果你没统计 step 数，粗略用 "10个episode * 平均工序数" 估算也可以
        # 但更精准的是在 while not done 里面维护一个全局 global_step_count
        # 这里简单起见，我们记录 "每轮耗时 (Seconds per Episode)"
        writer.add_scalar("Efficiency/Seconds_Per_Episode", elapsed_time / 10.0, i)
        start_time = current_time  # 重置计时器
        # 🟢【新增】GPU 显存监控 (针对 Reviewer #3 Q16)
        if torch.cuda.is_available():
            # 记录当前显存占用 (MB)
            memory_used = torch.cuda.memory_allocated() / 1024 / 1024
            writer.add_scalar("Efficiency/GPU_Memory_MB", memory_used, i)
        # 4. 🟢【针对意见 #17】变规模下的归一化性能
        # 你的变规模训练 Makespan 会抖动，这个指标能证明你在“进步”
        avg_time_per_op = myenv.FJSP.max / max(1, current_total_ops)
        writer.add_scalar("Train/Norm_Makespan_Per_Op", avg_time_per_op, i)
        # 2. 🟢【论文用】归一化 Makespan (平均每道工序耗时)
        # 这个曲线会很平滑，证明模型在变规模下依然在稳定学习
        avg_time_per_op = myenv.FJSP.max / max(1, current_total_ops)
        writer.add_scalar("Train/Norm_Makespan_Per_Op", avg_time_per_op, i)
        # 3. 记录当前题目难度 (工件数量)，证明你在做变规模训练
        writer.add_scalar("Train/Current_Job_Num", len(current_train_pt), i)

        writer.add_scalar("Train/Total_Reward", job_score + agv_score, i)

    # === 评估 (Evaluation) ===
    # 策略：平时只抽查 5 个，每 5000 轮才全量跑一次
    # 1. 决定是否评估
    is_quick_eval = (i > 0 and i % 500 == 0)  # 每 500 轮快测
    is_full_eval = (i > 0 and i % 5000 == 0)  # 每 5000 轮普查
    # 每 500 轮评估一次，使用真实的 GNN 验证集
    if is_quick_eval or is_full_eval:
        print(f"\n--- Evaluation against GNN Dataset @ Ep {i} ---")
        # 【修改点 2】必须切换到 Eval 模式，关闭 Dropout
        jobagent.job_actor.eval()
        jobagent.job_critic.eval()
        agvagent.agv_actor.eval()
        agvagent.agv_critic.eval()
        # 2. 决定测试集规模
        if is_full_eval:
            eval_candidates = validation_set
            print(f"🔍 [Full Eval] Testing all {len(validation_set)} cases...")
        else:
            # 随机抽 10 个来看看，省时间
            eval_candidates = random.sample(validation_set, min(20, len(validation_set)))
            print(f"⚡ [Quick Eval] Sampling {len(eval_candidates)} cases...")

        total_eval_makespan = 0
        # 遍历所有验证集文件
        for case_data in eval_candidates:
            eval_state, p_mask, j_mask, a_mask = eval_env.reset(
                new_job_data=case_data['PT'],
                new_agv_num=case_data['agv_num']
            )
            eval_done = False

            # 推理模式 (Greedy)
            while not eval_done:
                j_act, _, _ = jobagent.choose_action(eval_state, p_mask, j_mask, greedy=True)
                mid_s, p_mask, a_mask, m_act = eval_env.job_step(j_act)
                a_act, _, _ = agvagent.choose_action(mid_s, p_mask, a_mask, greedy=True)
                real_a = a_act - eval_env.num_jobs
                eval_state, p_mask, j_mask, a_mask, _, eval_done, _ = \
                    eval_env.step(j_act, m_act, real_a)
            total_eval_makespan += eval_env.FJSP.max

        avg_eval_score = total_eval_makespan / len(eval_candidates)
        print(f"📊 Validation Avg Score: {avg_eval_score:.2f} (Best: {best_avg_makespan:.2f})")
        # 只在 全量评估 或者 分数极其优异 时记录 BEST
        # 防止因为抽样运气好而虚报成绩
        if is_full_eval or avg_eval_score < best_avg_makespan:
            writer.add_scalar("Eval/transformer_Dataset_Score", avg_eval_score, i)
            # 保存历史最佳 (专门针对 10x5 优化)
            if is_full_eval:  # 只有全量评估才配谈“历史最佳”
                if avg_eval_score < best_avg_makespan:
                    best_avg_makespan = avg_eval_score
                    print(f"🏆 NEW BEST for 15x10! Saving model...")
                    jobagent.save_snapshot(os.path.join(param_dir, 'job_agent_best_20x5.pt'))
                    agvagent.save_snapshot(os.path.join(param_dir, 'agv_agent_best_20x5.pt'))
        # 定期保存 Checkpoint (防崩)
        if i % 5000 == 0:
            # 同时也保存常规 Checkpoint
            jobagent.save_snapshot(os.path.join(param_dir, f'job_agent_{i}.pt'))
            agvagent.save_snapshot(os.path.join(param_dir, f'agv_agent_{i}.pt'))
        jobagent.job_actor.train()
        jobagent.job_critic.train()
        agvagent.agv_actor.train()
        agvagent.agv_critic.train()

writer.close()
