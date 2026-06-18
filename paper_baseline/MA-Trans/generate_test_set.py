import torch
import random
import os
import json


def generate_constrained_instance(n_machines=8, job_range=(5, 10), agv_range=(2, 4)):
    """
    参数化生成案例，支持不同规模
    """
    num_jobs = random.randint(job_range[0], job_range[1])
    num_agvs = random.randint(agv_range[0], agv_range[1])
    PT = []

    for _ in range(num_jobs):
        num_ops = random.randint(2, 4)
        job_ops = []
        for _ in range(num_ops):
            op_machines = [0] * n_machines
            num_capable = random.randint(2, 3)
            capable_indices = random.sample(range(n_machines), num_capable)
            for m_idx in capable_indices:
                op_machines[m_idx] = random.randint(10, 30)
            job_ops.append(op_machines)
        PT.append(job_ops)

    return {'PT': PT, 'agv_num': num_agvs}


def generate_and_save(folder_name, job_range, agv_range, num_instances=100):
    """
    封装好的生成函数
    """
    BASE_DIR = f'./test_data/{folder_name}'  # 放在 test_data 下的子文件夹
    PT_FILE_NAME = f'test_{folder_name}.pt'
    JSON_DIR_NAME = 'json_cases'

    PT_PATH = os.path.join(BASE_DIR, PT_FILE_NAME)
    JSON_DIR_PATH = os.path.join(BASE_DIR, JSON_DIR_NAME)

    if not os.path.exists(JSON_DIR_PATH):
        os.makedirs(JSON_DIR_PATH)

    print(f"正在生成测试集 [{folder_name}]: Jobs={job_range}, Count={num_instances}...")

    dataset = []
    for i in range(num_instances):
        instance = generate_constrained_instance(n_machines=8, job_range=job_range, agv_range=agv_range)
        dataset.append(instance)

        # 存 JSON (给 GNN 用)
        json_path = os.path.join(JSON_DIR_PATH, f'case_{i}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=4)

    # 存 PT (给 PPO 用)
    torch.save(dataset, PT_PATH)
    print(f"  -> 已保存至: {BASE_DIR}")


def main():
    # === 1. 生成同分布测试集 (跟训练一样) ===
    generate_and_save(
        folder_name="test_small_5_10",
        job_range=(5, 10),
        agv_range=(2, 4),
        num_instances=100
    )

    # === 2. 生成中等规模测试集 (泛化测试 1) ===
    # 比如训练只用了 5-10，这里测 15-20
    generate_and_save(
        folder_name="test_medium_15_20",
        job_range=(15, 20),
        agv_range=(3, 5),  # 工件多了，AGV稍微多给一个防止死锁太频繁
        num_instances=100
    )

    # === 3. 生成大规模测试集 (泛化测试 2 - 压力测试) ===
    # 比如测 30-40 个工件
    generate_and_save(
        folder_name="test_large_30_40",
        job_range=(30, 40),
        agv_range=(4, 6),
        num_instances=100
    )


if __name__ == "__main__":
    main()