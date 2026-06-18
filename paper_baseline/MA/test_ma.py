from ma import MA
import time
import numpy as np
import pandas as pd
import glob
import os

CURRENT_SCALE = "15-10-4"  # 可选: "10-5-2", "15-10-4", "20-5-3", "20-10-5"
# 参数查找表
CONFIG_MAP = {
    "10-5-2":  {"pop": 100, "gen": 100, "agv_num": 2, "mach_num": 5,  "folder": "./test/1005"},
    "15-10-4": {"pop": 100, "gen": 200, "agv_num": 4, "mach_num": 10, "folder": "./test/1510"},
    "20-5-3":  {"pop": 100, "gen": 200, "agv_num": 3, "mach_num": 5,  "folder": "./test/2005"},
    "20-10-5": {"pop": 200, "gen": 300, "agv_num": 5, "mach_num": 10, "folder": "./test/2010"},
}
# 自动加载当前配置
cfg = CONFIG_MAP[CURRENT_SCALE]
VAL_FOLDER = cfg["folder"]
N_MACHINES = cfg["mach_num"]
N_AGVS = cfg["agv_num"]
POP_SIZE = cfg["pop"]
GEN_SIZE = cfg["gen"]
N_RUNS = 5
print(f"⚙️ 当前测试规模: {CURRENT_SCALE}")
print(f"   - 机器数: {N_MACHINES}, AGV数: {N_AGVS}")
print(f"   - MA参数: Pop={POP_SIZE}, Gen={GEN_SIZE} (Run {N_RUNS} times)")

MAT_5_MACHINES = [
    [0, 11, 9, 15, 9, 8],
    [11, 0, 7, 11, 12, 12],
    [9, 7, 0, 7, 11, 5],
    [15, 11, 7, 0, 9, 11],
    [9, 12, 11, 9, 0, 9],
    [8, 12, 5, 11, 9, 0],
]

MAT_10_MACHINES = [
    [0, 12, 14, 2, 12, 11, 16, 15, 12, 8, 3],
    [12, 0, 11, 10, 13, 16, 18, 9, 6, 18, 9],
    [14, 11, 0, 14, 4, 9, 8, 2, 4, 14, 12],
    [2, 10, 14, 0, 12, 12, 17, 14, 10, 10, 2],
    [12, 13, 4, 12, 0, 4, 5, 6, 7, 10, 10],
    [11, 16, 9, 12, 4, 0, 5, 11, 11, 6, 11],
    [16, 18, 8, 17, 5, 5, 0, 10, 11, 11, 15],
    [15, 9, 2, 14, 6, 11, 10, 0, 4, 16, 12],
    [12, 6, 4, 10, 7, 11, 11, 4, 0, 14, 8],
    [8, 18, 14, 10, 10, 6, 11, 16, 14, 0, 10],
    [3, 9, 12, 2, 10, 11, 15, 12, 8, 10, 0],
]

AGV_TRANS_DICT = {
    "10-5-2": MAT_5_MACHINES,   # 5台机器用这个
    "20-5-3": MAT_5_MACHINES,   # 5台机器用这个
    "15-10-4": MAT_10_MACHINES, # 10台机器用这个
    "20-10-5": MAT_10_MACHINES  # 10台机器用这个
}
# 这样获取才是安全的，如果 Key 写错了会直接报错提示，而不是默默用错误的矩阵
if CURRENT_SCALE not in AGV_TRANS_DICT:
    raise ValueError(f"❌ 错误: 这里的 {CURRENT_SCALE} 没有对应的 AGV 矩阵配置！")
AGV_TRANS = AGV_TRANS_DICT[CURRENT_SCALE]
# ================= 2. 工具函数 =================
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


def load_validation_set(folder_path, n_machines, n_agvs):
    dataset = []
    files = glob.glob(os.path.join(folder_path, "*.fjs"))
    files.sort()  # 排序保证顺序一致
    print(f"📂 发现 {len(files)} 个测试文件...")
    for file in files:
        pt = load_fjs_file(file, n_machines)
        dataset.append({'PT': pt, 'agv_num': n_agvs, 'name': os.path.basename(file)})
    return dataset


# ================= 3. 主测试逻辑 =================
if __name__ == "__main__":
    print(f"🚀 开始 MA (Memetic Algorithm) 批量基准测试")
    print(f"⚙️ 参数设置: Pop={POP_SIZE}, Gen={GEN_SIZE}, Runs per Case={N_RUNS}")
    print("-" * 60)

    validation_set = load_validation_set(VAL_FOLDER, N_MACHINES, N_AGVS)

    # 存储最终所有算例的平均分，用于计算整个数据集的 Mean
    all_case_avg_makespans = []
    total_start_time = time.time()

    for i, data in enumerate(validation_set):
        PT = data['PT']
        # 动态生成 MT (Machine Table)
        MT = []
        for job in PT:
            job_mt = []
            for op in job:
                machines = {j + 1 for j, t in enumerate(op) if t > 0}
                job_mt.append(machines)
            MT.append(job_mt)

        n = len(PT)

        # 单个算例跑 N 次
        case_scores = []

        # 简单的行内进度打印
        print(f"Case {i + 1:02d}/{len(validation_set)} ({data['name']}): ", end="", flush=True)

        for run in range(N_RUNS):
            # 实例化 MA
            # 注意: pc (交叉率) 和 pm (变异率) 使用经典值 0.8, 0.1
            ma_solver = MA(n, N_MACHINES, N_AGVS, PT, MT, AGV_TRANS,
                           pop_size=POP_SIZE,
                           gene_size=GEN_SIZE,
                           pc=0.8,
                           pm=0.1,
                           N_elite=int(POP_SIZE * 0.1))  # 精英数量 10%

            # 运行求解
            best_cmax = ma_solver.main()
            case_scores.append(best_cmax)
            print(".", end="", flush=True)

        # 统计该算例的平均分 (这是对比 RL 最公平的指标)
        case_avg = np.mean(case_scores)
        all_case_avg_makespans.append(case_avg)

        print(f" -> Avg: {case_avg:.1f} | Best: {np.min(case_scores)}")

    total_time = time.time() - total_start_time

    # ================= 4. 最终统计 (对齐 RL 输出) =================
    results = np.array(all_case_avg_makespans)

    mean_cmax = np.mean(results)  # 所有算例平均分的平均分
    std_cmax = np.std(results)  # 标准差
    min_cmax = np.min(results)  # 最好算例的平均分
    max_cmax = np.max(results)  # 最差算例的平均分
    avg_time_per_case = total_time / len(validation_set)
    np.savetxt("test.csv", results, delimiter=",", header="Makespan", comments="")
    print("\n" + "=" * 40)
    print(f"🎉 MA 批量测试完成！")
    print(f"⏱️ 总耗时: {total_time:.2f}s (单例平均: {avg_time_per_case:.2f}s)")
    print("=" * 40)
    print(f"📌 算法: MA (Pop={POP_SIZE}, Gen={GEN_SIZE})")
    print(f"📂 数据集: {VAL_FOLDER} ({len(results)} instances)")
    print("-" * 40)
    print(f"🏆 Average Makespan (Mean): {mean_cmax:.2f}  <-- 对比 RL 的核心指标")
    print(f"📊 Standard Deviation (Std): {std_cmax:.2f}")
    print(f"🌟 Best Case Avg    (Min): {min_cmax:.2f}")
    print(f"💀 Worst Case Avg   (Max): {max_cmax:.2f}")
    print("=" * 40)