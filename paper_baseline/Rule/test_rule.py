import glob
import os
import time
import numpy as np  # 引入 numpy 用于保存 CSV
from rule_env import EnvWorkShop
from rule_solver import RuleSolver

# === 配置 ===
VAL_FOLDER_PATH = './newtest/2010'  # 确保这里有 .fjs 文件
TARGET_MACHINES = 10
TARGET_AGVS = 3

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

def load_validation_set(folder_path):
    dataset = []
    # 确保按文件名排序，保证和 baseline 的顺序一致
    files = sorted(glob.glob(os.path.join(folder_path, "*.fjs")))
    print(f"📂 加载验证集: {folder_path} ... 共 {len(files)} 个文件")
    for file in files:
        pt = load_fjs_file(file, TARGET_MACHINES)
        dataset.append({'PT': pt, 'agv_num': TARGET_AGVS, 'name': os.path.basename(file)})
    return dataset

def run_test(rule_name):
    """
    运行指定规则的测试并保存 CSV
    rule_name: 'MWKR' 或 'G-EST'
    """
    dataset = load_validation_set(VAL_FOLDER_PATH)
    env = EnvWorkShop(machine_num=TARGET_MACHINES)
    solver = RuleSolver(env)

    total_makespan = 0
    results = []

    print(f"\n🚀 开始测试规则: [{rule_name}]")
    t_start = time.time()

    for i, case in enumerate(dataset):
        # 重置环境
        env.reset(new_job_data=case['PT'], new_agv_num=case['agv_num'])
        done = False

        while not done:
            # === 核心：调用 RuleSolver 获取动作 ===
            if rule_name == 'MWKR':
                j_act, m_act, a_act = solver.solve_mwkr()
            elif rule_name == 'G-EST':
                j_act, m_act, a_act = solver.solve_global_greedy()
            else:
                raise ValueError(f"Unknown Rule: {rule_name}")
            
            # 执行环境步
            _, _, _, _, _, done, _ = env.step(j_act, m_act, a_act)

        makespan = env.FJSP.max
        total_makespan += makespan
        results.append(makespan)

        # 打印进度 (每10个打印一次)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(dataset)} | Last Cmax: {makespan:.1f}")

    avg_makespan = total_makespan / len(dataset)
    t_end = time.time()
    
    # === 💾 保存结果到 CSV ===
    csv_filename = f"{rule_name}_results.csv"
    # 保存为一列，带表头 'Makespan'
    np.savetxt(csv_filename, np.array(results), delimiter=",", header="Makespan", comments="", fmt='%.4f')
    
    print(f"✅ 测试结束 - {rule_name}")
    print(f"💾 结果已保存至: {csv_filename}")
    print(f"📊 平均 Makespan: {avg_makespan:.2f}")
    print(f"⏱️ 总耗时: {t_end - t_start:.2f}s")
    print("-" * 30)
    return avg_makespan

if __name__ == "__main__":
    # 运行两种规则并保存 CSV
    # score_G_EST = run_test('G-EST')
    score_MWKR = run_test('MWKR')

    print("\n🏆 === 最终结果对比 ===")
    # print(f"G-EST (Mean) : {score_G_EST:.2f}")
    print(f"MWKR  (Mean) : {score_MWKR:.2f}")