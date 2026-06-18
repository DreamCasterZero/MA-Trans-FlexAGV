import os
import random

# ================= 配置区域 =================
# 规模设置
N_JOBS = 25          # 工件数
N_MACHINES = 10      # 机器数
N_INSTANCES = 100    # 生成文件数量

# 随机范围设置 (根据你的要求)
OPS_RANGE = (8, 12)  # 每个工件的工序数范围 [min, max]
TIME_RANGE = (1, 20) # 加工时间范围 [min, max]

# 输出目录
OUTPUT_DIR = "./25_10_dataset"
# ===========================================

def generate_instance(file_path):
    """
    生成单个 FJSP 实例并保存为标准文本格式
    格式说明:
    第一行: <工件数> <机器数> <平均工序数(可选)>
    后续行 (每个工件一行):
    <工序总数>  [工序1可用机器数] <机器ID> <时间> ...  [工序2可用机器数] <机器ID> <时间> ...
    """
    lines = []
    
    # 1. Header (第一行)
    # 计算平均工序数用于header (虽然有些解析器不看这个，但保持标准格式更好)
    # 这里简单写 0 或者估算值，标准格式通常只需要 Jobs Machines
    header = f"{N_JOBS} {N_MACHINES} {sum(OPS_RANGE)/2}"
    lines.append(header)

    # 2. 生成每个工件的数据
    for job_id in range(N_JOBS):
        # 随机确定该工件的工序数量 (8-12)
        num_ops = random.randint(OPS_RANGE[0], OPS_RANGE[1])
        
        job_line_parts = [str(num_ops)] # 该行第一个数是工序总数

        for op_idx in range(num_ops):
            # 确定该工序有多少个机器能做 (部分柔性: 1 到 N_MACHINES)
            num_capable = random.randint(1, N_MACHINES)
            
            # 随机选择 num_capable 个机器 (机器ID通常从1开始，适配标准Benchmark)
            # 如果你的解析器是从0开始读的，把 range(1, N_MACHINES + 1) 改为 range(0, N_MACHINES)
            capable_machines = random.sample(range(1, N_MACHINES + 1), num_capable)
            
            # 构建该工序的字符串: [可用机器数] (机器ID 时间) (机器ID 时间)...
            op_segment = [str(num_capable)]
            
            for m_id in capable_machines:
                proc_time = random.randint(TIME_RANGE[0], TIME_RANGE[1])
                op_segment.append(str(m_id))      # 机器ID
                op_segment.append(str(proc_time)) # 加工时间
            
            # 将该工序的信息加入行
            job_line_parts.append(" ".join(op_segment))

        # 将整个工件的一行组合起来
        lines.append("  ".join(job_line_parts))

    # 3. 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def main():
    # 创建文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已创建目录: {OUTPUT_DIR}")

    print(f"🚀 开始生成 {N_INSTANCES} 个算例...")
    print(f"⚙️  规模: {N_JOBS} Jobs, {N_MACHINES} Machines")
    print(f"⚙️  工序范围: {OPS_RANGE}, 时间范围: {TIME_RANGE}")

    for i in range(N_INSTANCES):
        # 文件名格式: 35_10_001.fjs
        filename = f"{N_JOBS}_{N_MACHINES}_{i+1:03d}.fjs"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        generate_instance(file_path)

    print(f"✅ 生成完毕! 所有文件已保存在 '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()