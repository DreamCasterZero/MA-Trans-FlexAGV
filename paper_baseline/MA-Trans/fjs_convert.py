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


# === 1. 加载 GNN 的标准验证集 ===
def load_validation_set(folder_path, n_machines, n_agvs):
    dataset = []
    # 找到文件夹下所有的 .fjs 文件
    files = glob.glob(os.path.join(folder_path, "*.fjs"))

    print(f"📂 发现 {len(files)} 个验证文件，正在加载...")
    i = 0
    for file in files:
        # 使用上面的解析函数读取工件数据
        pt = load_fjs_file(file, n_machines)

        # ⚠️ 关键：手动补全 AGV 数量 (因为 fjs 文件里没有)
        # 这里必须和你训练时的设定一致 (比如 10x5 对应 2 AGV)
        dataset.append({'PT': pt, 'agv_num': n_agvs})
        print(pt)
        print(dataset)
        if i == 2:
            break
        i+=1


    print("✅ 验证集加载完成！")
    return dataset



# 运行侦察
# 假设你把 GNN 的 data_dev/1005 拷过来了
# analyze_fjs_stats('./data_dev/1005', 5)

val_folder = './validation_data/1005'
fixed_validation_set = load_validation_set(val_folder, n_machines=5, n_agvs=2)

