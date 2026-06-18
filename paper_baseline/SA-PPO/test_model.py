import os
import glob
import torch
import numpy as np
import time

# 引入你的环境和智能体
from env import EnvWorkShop
# 注意：这里只需要 Job_Agent，因为 AGV 用规则
from ppo import Job_Agent

TEST_DATA_DIR = './test/1510'

MODEL_DIR = './result/15-10-4/param'  # 模型保存文件夹
CHECKPOINT_NAME = 'best_15x10'  # 检查点名称

TARGET_MACHINES = 10  # 机器数
TARGET_AGVS = 4 # AGV数


# === 2. 工具函数 (与之前一致) ===
def load_fjs_file(file_path, total_machines):
    """读取标准 .fjs 文件"""
    PT = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
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
                    m_id = data[idx] - 1
                    time = data[idx + 1]
                    idx += 2
                    if m_id < total_machines:
                        op_machines[m_id] = time
                job_ops.append(op_machines)
            PT.append(job_ops)
    return PT


def run_sa_ppo_test():
    print(f"🚀 开始 SA-PPO (Single-Agent) 批量测试...")
    print(f"📂 测试集目录: {TEST_DATA_DIR}")
    print(f"🧠 加载模型: {CHECKPOINT_NAME}")

    test_files = glob.glob(os.path.join(TEST_DATA_DIR, '*.fjs'))
    test_files.sort()

    if len(test_files) == 0:
        print(f"❌ 错误: 在 {TEST_DATA_DIR} 下没找到 .fjs 文件！")
        return

    # 2. 初始化环境和 Agent
    env = EnvWorkShop(machine_num=TARGET_MACHINES)

    # 🔴 只初始化 Job Agent
    jobagent = Job_Agent(input_size=24, batch_size=1, total_updates=1)

    # 3. 加载 Job Agent 权重
    try:
        # 注意文件名的拼接逻辑，确保和你 save() 时的逻辑一致
        # 通常是 job_actor_ckpt_best.pt 或 job_actor_sa_ppo_best.pt
        job_actor_path = os.path.join(MODEL_DIR, f'job_actor_{CHECKPOINT_NAME}.pt')

        if not os.path.exists(job_actor_path):
            # 尝试另一种命名可能
            job_actor_path = os.path.join(MODEL_DIR, f'job_actor{CHECKPOINT_NAME}.pt')

        jobagent.job_actor.load_state_dict(torch.load(job_actor_path))
        jobagent.job_actor.eval()
        print(f"✅ Job Agent 权重加载成功: {job_actor_path}")
        print("🤖 AGV 策略: Heuristic (Nearest Idle)")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 提示: 请检查 MODEL_DIR 和 CHECKPOINT_NAME 是否正确。")
        return

    # 4. 批量推理循环
    results = []
    start_time = time.time()

    with torch.no_grad():
        for i, file_path in enumerate(test_files):
            pt_matrix = load_fjs_file(file_path, TARGET_MACHINES)

            # 重置环境
            state, p_mask, j_mask, a_mask = env.reset(new_job_data=pt_matrix, new_agv_num=TARGET_AGVS)
            done = False

            while not done:
                # A. Job Agent (PPO) 选工件
                j_act, _, _ = jobagent.choose_action(state, p_mask, j_mask, greedy=True)

                # B. 环境半步更新 (计算机器选择)
                # 注意：env.job_step 会返回 machine_action
                # 但这里我们需要 machine_action 来传给 step，env.job_step 内部并没有执行 machine.update
                mid_s, p_mask, a_mask, m_act = env.job_step(j_act)

                # 🔴 C. AGV 规则选择 (Heuristic)
                # 直接调用环境内置的规则函数
                # 注意：这个函数返回的是 1-based 的物理 ID
                agv_act_1based = env.get_heuristic_agv_action(j_act)

                # D. 环境全步更新
                # 注意 env.step 需要的是 (1-based job, 1-based mach, 1-based agv)
                # 或者 (1-based job, 1-based mach, 0-based agv_idx) ???
                # 让我们看一眼 env.py 的 step 函数:
                # agv = self.FJSP.AGVs[agv_action-1]
                # 所以 env.step 接收的是 1-based 的 ID。

                # 你的 get_heuristic_agv_action 返回的是 1-based ID
                # 所以直接传进去即可
                state, p_mask, j_mask, a_mask, _, done, _ = env.step(j_act, m_act, agv_act_1based)

            makespan = env.FJSP.max
            results.append(makespan)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(test_files)} | Last Cmax: {makespan:.1f}")

    total_time = time.time() - start_time

    # 5. 统计输出
    results = np.array(results)
    print("\n" + "=" * 40)
    print(f"🎉 SA-PPO (RL+Rule) 测试完成！")
    print(f"📂 模型: {CHECKPOINT_NAME}")
    print("-" * 40)
    print(f"🏆 Mean Makespan : {np.mean(results):.2f}")
    print(f"📊 Std Deviation : {np.std(results):.2f}")
    print(f"🌟 Best (Min)    : {np.min(results):.2f}")
    print(f"💀 Worst (Max)   : {np.max(results):.2f}")
    print("=" * 40)
    np.savetxt("test.csv", results, delimiter=",", header="Makespan", comments="")


if __name__ == '__main__':
    run_sa_ppo_test()