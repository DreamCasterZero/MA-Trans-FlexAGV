import os
import glob
import torch
import numpy as np
import time

# 引入你的环境和智能体
from gnn_env import EnvWorkShop
from gnn_ppo import Job_Agent, AGV_Agent

# === 1. 配置区域 ===
# 测试集文件夹路径 (确保里面放的是那100个 .fjs 文件)
TEST_DATA_DIR = './test/1510'

# 模型路径配置
MODEL_DIR = './result/15-10-4/param'  # 模型保存文件夹
CHECKPOINT_NAME = 'gnn_best_15x10'  # 检查点名称

TARGET_MACHINES = 10  # 机器数
TARGET_AGVS = 4  # AGV数


# === 2. 工具函数 ===
def load_fjs_file(file_path, total_machines):
    """读取标准 .fjs 文件并转换为环境需要的 PT 矩阵"""
    PT = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过第一行
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


def run_batch_test():
    print(f"🚀 开始批量测试...")
    print(f"📂 测试集目录: {TEST_DATA_DIR}")
    print(f"🧠 加载模型: {CHECKPOINT_NAME}")

    # 1. 获取所有测试文件
    # 假设文件后缀是 .fjs，如果是 .txt 请自行修改
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, '*.fjs'))

    # 排序，保证每次运行顺序一致（可选）
    test_files.sort()

    if len(test_files) == 0:
        print(f"❌ 错误: 在 {TEST_DATA_DIR} 下没找到 .fjs 文件！")
        return

    print(f"📊 发现 {len(test_files)} 个测试算例，准备开始...")

    # 2. 初始化环境和 Agent (只做一次)
    env = EnvWorkShop(machine_num=TARGET_MACHINES)

    # 注意: input_size 必须和训练时一致 (通常是24)
    jobagent = Job_Agent(input_size=24, batch_size=1, total_updates=10000)
    agvagent = AGV_Agent(input_size=24, batch_size=1, total_updates=10000)

    # 3. 加载权重 (只做一次)
    try:
        job_actor_path = os.path.join(MODEL_DIR, f'job_actor_{CHECKPOINT_NAME}.pt')
        # 处理 AGV 文件名可能的命名不一致问题 (agv_actor_ vs agv_actor)
        agv_actor_path_1 = os.path.join(MODEL_DIR, f'agv_actor{CHECKPOINT_NAME}.pt')
        agv_actor_path_2 = os.path.join(MODEL_DIR, f'agv_actor_{CHECKPOINT_NAME}.pt')

        if os.path.exists(agv_actor_path_1):
            agv_actor_path = agv_actor_path_1
        elif os.path.exists(agv_actor_path_2):
            agv_actor_path = agv_actor_path_2
        else:
            raise FileNotFoundError(f"找不到 AGV 模型文件: {agv_actor_path_1} 或 {agv_actor_path_2}")

        jobagent.job_actor.load_state_dict(torch.load(job_actor_path))
        agvagent.agv_actor.load_state_dict(torch.load(agv_actor_path))

        jobagent.job_actor.eval()
        agvagent.agv_actor.eval()
        print("✅ 模型权重加载成功，推理引擎就绪！\n")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 批量推理循环
    results = []
    start_time = time.time()

    with torch.no_grad():
        for i, file_path in enumerate(test_files):
            # 加载当前算例数据
            pt_matrix = load_fjs_file(file_path, TARGET_MACHINES)

            # 重置环境
            state, p_mask, j_mask, a_mask = env.reset(new_job_data=pt_matrix, new_agv_num=TARGET_AGVS)
            done = False

            # 单个算例推理
            while not done:
                # Job Action (Greedy)
                j_act, _, _ = jobagent.choose_action(state, p_mask, j_mask, greedy=True)

                # Step 1
                mid_s, p_mask, a_mask, m_act = env.job_step(j_act)

                # AGV Action (Greedy)
                a_act, _, _ = agvagent.choose_action(mid_s, p_mask, a_mask, greedy=True)
                real_a = a_act - env.num_jobs

                # Step 2
                state, p_mask, j_mask, a_mask, _, done, _ = env.step(j_act, m_act, real_a)

            # 记录结果
            makespan = env.FJSP.max
            results.append(makespan)

            # 打印进度条 (每10个打印一次)
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(test_files)} | Last Cmax: {makespan:.2f}")

    total_time = time.time() - start_time

    # 5. 统计与输出
    results = np.array(results)
    mean_cmax = np.mean(results)
    std_cmax = np.std(results)
    min_cmax = np.min(results)
    max_cmax = np.max(results)

    print("\n" + "=" * 40)
    print(f"🎉 批量测试完成！(耗时 {total_time:.2f}s)")
    print("=" * 40)
    print(f"📌 模型: PPO-Transformer ({CHECKPOINT_NAME})")
    print(f"📂 数据集: {TEST_DATA_DIR} ({len(results)} instances)")
    print("-" * 40)
    print(f"🏆 Average Makespan (Mean): {mean_cmax:.2f}")
    print(f"📊 Standard Deviation (Std): {std_cmax:.2f}")
    print(f"🌟 Best Makespan (Min)   : {min_cmax:.2f}")
    print(f"💀 Worst Makespan (Max)  : {max_cmax:.2f}")
    print("=" * 40)

    # 可选：保存详细结果到 CSV，方便做箱线图
    np.savetxt("test.csv", results, delimiter=",", header="Makespan", comments="")


if __name__ == '__main__':
    run_batch_test()