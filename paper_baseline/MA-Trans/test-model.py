import os
import glob
import torch
import numpy as np
import time
from utils import draw_gantte
# 引入你的环境和智能体
from env import EnvWorkShop
from ppo import Job_Agent, AGV_Agent
import json
temperatures = [0.1]
# temperatures = [0.1, 0.5, 0.8]
# === 1. 配置区域 ===
# 测试集文件夹路径 (确保里面放的是那100个 .fjs 文件)
TEST_DATA_DIR = './test/1005'

# 模型路径配置
MODEL_DIR = './result/10-5-2/param'  # 模型保存文件夹
CHECKPOINT_NAME = 'best_10x5'  # 检查点名称

TARGET_MACHINES = 5  # 机器数
TARGET_AGVS = 2# AGV数

def convert_schedule_to_json(machines, agvs):
    gantt_data = []

    # Machine schedule
    for m in machines:
        for i, job_id in enumerate(m.on):
            time_range = m.using_time[i]
            gantt_data.append({
                "resource": f"M{m.idx + 1}",
                "job": f"J{job_id}",
                "start": time_range[0],
                "end": time_range[1]
            })

    # AGV schedule
    for agv in agvs:
        for i, job_id in enumerate(agv.on):
            time_range = agv.using_time[i]
            job_label = "Empty" if job_id is None else f"J{job_id}"
            gantt_data.append({
                "resource": f"AGV{agv.idx + 1}",
                "job": job_label,
                "start": time_range[0],
                "end": time_range[1]
            })

    return gantt_data

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
    for temp in temperatures:
        print(f"\n 🔥 Testing with Temperature = {temp} ...")
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
                # pt_matrix = load_fjs_file("/home/syx/fjsp-agv/MA-Trans/validation_data/1005/10j_5m_001.fjs", 5)
                pt_matrix = [[[12, 0, 12, 14, 13], [13, 11, 0, 12, 13], [0, 0, 15, 18, 16], [10, 0, 10, 0, 0]], [[19, 0, 0, 0, 19], [1, 1, 1, 0, 1], [0, 17, 16, 19, 16], [0, 0, 0, 2, 0], [4, 3, 0, 0, 4]], [[13, 10, 0, 0, 0], [6, 5, 5, 5, 5], [14, 11, 15, 14, 16], [10, 7, 10, 6, 9], [0, 15, 0, 0, 20]], [[11, 10, 12, 12, 0], [0, 0, 0, 0, 11], [0, 12, 0, 0, 10], [5, 6, 7, 7, 7]], [[0, 0, 14, 11, 10], [3, 0, 5, 4, 4], [7, 0, 8, 6, 8], [0, 0, 0, 17, 0]], [[0, 13, 11, 0, 0], [0, 11, 10, 11, 11], [9, 0, 11, 0, 0], [0, 0, 1, 0, 1]], [[4, 5, 4, 5, 3], [11, 14, 12, 14, 14], [4, 3, 4, 3, 4], [0, 2, 0, 0, 0]], [[5, 6, 5, 6, 6], [14, 13, 11, 15, 11], [0, 6, 0, 5, 5], [17, 18, 18, 16, 0], [17, 0, 0, 18, 18]], [[15, 14, 14, 17, 20], [2, 0, 0, 0, 2], [11, 11, 9, 11, 0], [9, 0, 9, 0, 8]], [[7, 7, 0, 7, 7], [18, 19, 19, 0, 0], [0, 4, 0, 3, 0], [4, 0, 0, 0, 0], [5, 5, 6, 5, 7]]]

                # 重置环境
                state, p_mask, j_mask, a_mask = env.reset(new_job_data=pt_matrix, new_agv_num=TARGET_AGVS)
                done = False
                # extra_jobs_added = False
                # dynamic_event_triggered = False
                # agv_fail_triggered = False
                agv_add_triggered = False
                # 单个算例推理
                while not done:
                    if env.FJSP.max >= 100 and not agv_add_triggered:
                        # 增加 1 台 AGV
                        state, p_mask, j_mask, a_mask = env.add_new_agv(num=1)
                        agv_add_triggered = True
                    # if env.FJSP.max >= 100 and not agv_fail_triggered:
                    #     # AGV2 对应的索引是 1
                    #     state, p_mask, j_mask, a_mask = env.disable_agv(1)
                    #     agv_fail_triggered = True
                    # if env.FJSP.max >= 100 and not dynamic_event_triggered:
                    #     # A. 去掉第一个工件 (Index 0)
                    #     state, p_mask, j_mask, a_mask = env.remove_job_by_idx(0)
                        
                    #     dynamic_event_triggered = True

                    # if env.FJSP.max >= 100 and not extra_jobs_added:
                    #     print(f"--- Event: New jobs arriving at time {env.FJSP.max} ---")
                    #     new_pt_data = [
                    #         [[0, 0, 0, 0, 2], [0, 19, 18, 19, 17], [4, 0, 0, 6, 0], [13, 13, 0, 15, 18], [0, 0, 0, 0, 17]], # 新工件 1
                    #         [[17, 0, 18, 0, 14], [8, 7, 8, 8, 0], [12, 0, 0, 0, 16], [0, 0, 0, 6, 6]]  # 新工件 2
                    #     ]
                    #     # 调用新写的函数，动态更新环境内部的 Jobs 列表
                    #     state, p_mask, j_mask, a_mask = env.add_new_jobs(new_pt_data)
                    #     extra_jobs_added = True
                    # Job Action (Greedy)
                    j_act, _, _ = jobagent.choose_action(state, p_mask, j_mask, greedy=True, temperature=temp)

                    # Step 1
                    mid_s, p_mask, a_mask, m_act = env.job_step(j_act)

                    # AGV Action (Greedy)
                    a_act, _, _ = agvagent.choose_action(mid_s, p_mask, a_mask, greedy=True, temperature=temp)
                    real_a = a_act - env.num_jobs

                    # Step 2
                    state, p_mask, j_mask, a_mask, _, done, _ = env.step(j_act, m_act, real_a)

                # 记录结果
                makespan = env.FJSP.max
                results.append(makespan)

                # 打印进度条 (每10个打印一次)
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{len(test_files)} | Last Cmax: {makespan:.2f}")

                result = convert_schedule_to_json(env.FJSP.Machines,env.FJSP.AGVs)
                file_path = "gantt_data.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(result)
                print(env.FJSP.max)
                draw_gantte.draw_gantt(env.FJSP.Machines,env.FJSP.AGVs)
                break

        total_time = time.time() - start_time


if __name__ == '__main__':
    run_batch_test()

    # pt = load_fjs_file("/home/syx/fjsp-agv/MA-Trans/validation_data/1005/10j_5m_002.fjs", 5)
    # print(pt)