from ma import MA
import time
import numpy as np
import pandas as pd # 建议用来存数据
import glob
import os

def load_fjs_file(file_path, total_machines):
    """
    读取标准 .fjs 文件并转换为你环境需要的 PT 矩阵格式
    :param file_path: fjs 文件路径
    :param total_machines: 你的环境设定的总机器数 (比如 5 或 10)
    :return: PT 矩阵 (jobs x ops x machines)
    """
    PT = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        # 第一行通常是：工件数 机器数 平均机器数
        # header = lines[0].strip().split()
        # num_jobs_in_file = int(header[0])
        # num_machines_in_file = int(header[1])

        # 从第二行开始，每一行代表一个 Job
        for line in lines[1:]:
            line = line.strip()
            if not line: continue

            # 数据格式通常是: 总工序数  (机器数 机器ID 时间) (机器数 机器ID 时间) ...
            data = list(map(int, line.split()))

            num_ops = data[0]  # 这一行的第一个数是该工件的工序总数
            idx = 1  # 指针，跳过第一个数

            job_ops = []

            for _ in range(num_ops):
                # 初始化当前工序在所有机器上的时间为 0 (表示不可加工)
                op_machines = [0] * total_machines

                # 当前工序能被多少台机器加工
                num_capable = data[idx]
                idx += 1

                for _ in range(num_capable):
                    m_id = data[idx] - 1  # .fjs 通常是 1-based，转为 0-based
                    time = data[idx + 1]
                    idx += 2

                    if m_id < total_machines:
                        op_machines[m_id] = time

                job_ops.append(op_machines)

            PT.append(job_ops)

    return PT
# === 加载 GNN 的标准验证集 ===
def load_validation_set(folder_path, n_machines, n_agvs):
    dataset = []
    # 找到文件夹下所有的 .fjs 文件
    files = glob.glob(os.path.join(folder_path, "*.fjs"))

    print(f"📂 发现 {len(files)} 个验证文件，正在加载...")

    for file in files:
        # 使用上面的解析函数读取工件数据
        pt = load_fjs_file(file, n_machines)

        # ⚠️ 关键：手动补全 AGV 数量 (因为 fjs 文件里没有)
        # 这里必须和你训练时的设定一致 (比如 10x5 对应 2 AGV)
        dataset.append({'PT': pt, 'agv_num': n_agvs})
    print("✅ 验证集加载完成！")
    return dataset

val_folder = './1005'
n_machines = 5
n_agvs = 2
N_RUNS = 10
# 加载所有数据
validation_set = load_validation_set(val_folder, n_machines, n_agvs)
# 结果存储
results = []
total_start_time = time.time()
print(f"🚀 开始批量测试 MA，共 {len(validation_set)} 个算例...")
print("-" * 50)

agv_trans = [
    [0, 11, 9, 15, 9, 8],
    [11, 0, 7, 11, 12, 12],
    [9, 7, 0, 7, 11, 5],
    [15, 11, 7, 0, 9, 11],
    [9, 12, 11, 9, 0, 9],
    [8, 12, 5, 11, 9, 0],
]

# === 2. 运行遗传算法 ===
if __name__ == "__main__":
    print(f"🚀 开始批量测试 MA (每个算例跑 {N_RUNS} 次取平均)...")
    print("-" * 60)

    for i, data in enumerate(validation_set):
        PT = data['PT']
        # 动态生成 MT
        MT = []
        for job in PT:
            job_mt = []
            for op in job:
                machines = {j + 1 for j, t in enumerate(op) if t > 0}
                job_mt.append(machines)
            MT.append(job_mt)

        n = len(PT)

        # === 存储该算例 N 次运行的结果 ===
        run_scores = []
        run_times = []

        print(f"Case {i + 1}/{len(validation_set)} Running...", end="", flush=True)

        for run in range(N_RUNS):
            t0 = time.time()

            # 实例化 MA (保持参数不变)
            ma_solver = MA(n, n_machines, n_agvs, PT, MT, agv_trans,
                           pop_size=100,
                           gene_size=100,
                           pc=0.8,
                           pm=0.1,
                           N_elite=10)

            # ⚠️ 注意：去 ma.py 把 main 里的 print(Gen...) 注释掉，不然屏幕会被刷爆
            best_cmax = ma_solver.main()

            t1 = time.time()
            run_scores.append(best_cmax)
            run_times.append(t1 - t0)
            print(f".", end="", flush=True)  # 进度条点点点

        # === 统计该算例的指标 ===
        best_of_runs = np.min(run_scores)  # 最好成绩
        avg_of_runs = np.mean(run_scores)  # 平均成绩
        std_of_runs = np.std(run_scores)  # 标准差 (波动幅度)
        avg_time = np.mean(run_times)  # 平均耗时

        print(f"\n -> Best: {best_of_runs} | Avg: {avg_of_runs:.1f} | Std: {std_of_runs:.1f}")

        results.append({
            "Case_ID": i,
            "MA_Best": best_of_runs,
            "MA_Avg": avg_of_runs,
            "MA_Std": std_of_runs,
            "MA_Time": avg_time
        })
        break

    print("-" * 60)
    # === 最终大汇总 ===
    df = pd.DataFrame(results)
    final_avg_best = df["MA_Best"].mean()  # 所有算例“最好成绩”的平均值
    final_avg_mean = df["MA_Avg"].mean()  # 所有算例“平均成绩”的平均值

    print(f"🏆 最终总结 (共 {len(validation_set)} 个算例):")
    print(f"MA Average Best (上限): {final_avg_best:.2f}")
    print(f"MA Average Mean (期望): {final_avg_mean:.2f}")

