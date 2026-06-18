from ma import MA
from gantte import draw_gantt
import matplotlib.pyplot as plt
m = 5 #机器数量
agv_num = 2
import os
import glob

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
        print(pt)
        print(dataset)
        break

    print("✅ 验证集加载完成！")
    return dataset
# 运行侦察
# 假设你把 GNN 的 data_dev/1005 拷过来了
# analyze_fjs_stats('./data_dev/1005', 5)

val_folder = './1005'
fixed_validation_set = load_validation_set(val_folder, n_machines=5, n_agvs=2)
data = fixed_validation_set[0]
PT = data['PT']


MT = []  # 存储每个工件的机器可加工集合

for job in PT:
    job_mt = []
    for op in job:
        machines = {j+1 for j, time in enumerate(op) if time > 0}
        job_mt.append(machines)
    MT.append(job_mt)

n = len(PT) #工件数量
print(f"工件数量: {n}, 机器数量: {m}, AGV数量: {agv_num}")
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
    print(" 正在启动模因算法 (MA) ...")
    # pop_size=100 (标准种群大小)
    # gene_size=200 (迭代次数，作为基准可以设大一点，比如 200 或 500，确保它收敛)
    # pc=0.8, pm=0.1 (标准交叉变异概率)
    ma_solver = MA(n, m, agv_num, PT, MT, agv_trans,
                   pop_size=200,  # 种群稍微改小一点点，提高单代速度
                   gene_size=1000,  # 1000代 MA 通常比 3000代 GA 还要强
                   pc=0.8,
                   pm=0.1,
                   N_elite=20)  # 精英数量，参与局部搜索的个体数
    # 运行主循环
    best_cmax = ma_solver.main()
    print(f"🏆 MA 搜索结束，最佳 Makespan: {best_cmax}")
    # 重新解码最佳染色体
    final_makespan = ma_solver.decode(ma_solver.best_os, ma_solver.best_ms)
    print(f"✅ 验证解码 Makespan: {final_makespan}")

    # 画甘特图
    # draw_gantt(ma_solver.rjsp.Machines, ma_solver.rjsp.AGVs)
    # plt.show()







# job_counts = []
# makespan_transformer = []
# makespan_algorithm1 = []
#
# ga = GA(n, m, agv_num, PT, MT, agv_trans)
# count = ga.main()
# ga.rjsp.reset()
# ga.decode(ga.best_job_chrom, ga.best_machine_chrom)
# print(ga.rjsp.C_max)
# draw_gantt(ga.rjsp.Machines, ga.rjsp.AGVs)

