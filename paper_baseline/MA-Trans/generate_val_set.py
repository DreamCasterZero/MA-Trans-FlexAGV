import torch
import random
import os
import json  # <--- 引入 json 库


def generate_constrained_instance(n_machines=8):
    """
    根据你的特定约束生成单个案例
    """
    num_jobs = random.randint(5, 10)
    num_agvs = random.randint(2, 4)
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

    return {
        'PT': PT,
        'agv_num': num_agvs
    }


def main():
    # === 路径配置 ===
    BASE_DIR = './validation_data'
    PT_FILE_NAME = 'val_dataset.pt'
    JSON_DIR_NAME = 'json_cases'  # 专门存放 json 的子文件夹

    # 完整路径
    PT_PATH = os.path.join(BASE_DIR, PT_FILE_NAME)
    JSON_DIR_PATH = os.path.join(BASE_DIR, JSON_DIR_NAME)

    # 确保文件夹存在
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(JSON_DIR_PATH):
        os.makedirs(JSON_DIR_PATH)

    NUM_INSTANCES = 100
    print(f"正在生成 {NUM_INSTANCES} 个验证案例...")

    dataset = []

    # === 生成并保存 ===
    for i in range(NUM_INSTANCES):
        instance = generate_constrained_instance(n_machines=8)
        dataset.append(instance)

        # --- 顺手存成 JSON 文件 ---
        # 文件名: case_0.json, case_1.json ...
        json_path = os.path.join(JSON_DIR_PATH, f'case_{i}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            # indent=4 让文件里面的格式缩进，方便人眼看
            json.dump(instance, f, indent=4)

            # --- 统一存成 PT 文件 (给 Python 训练读) ---
    torch.save(dataset, PT_PATH)

    print("-" * 30)
    print(f"1. PPO专用文件已保存: {os.path.abspath(PT_PATH)}")
    print(f"2. GNN/人工查看文件已保存: {os.path.abspath(JSON_DIR_PATH)} (共{NUM_INSTANCES}个json)")


if __name__ == "__main__":
    main()