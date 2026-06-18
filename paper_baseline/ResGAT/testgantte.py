import os
import torch
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt  # <--- 1. 必须导入这个库来创建画布
import gantte           # 假设你的绘图代码在 utils/gantte.py 中

# 引入你的环境和智能体
from gnn_env import EnvWorkShop
from gnn_ppo import Job_Agent, AGV_Agent

# ================= ⚙️ 配置区域 =================
TEST_FILE = './best_test.fjs'  # 或者 .pkl 文件路径

# 模型路径配置
MODEL_DIR = './result/15-10-4/param'  
CHECKPOINT_NAME = 'gnn_best_15x10'  

# 规模配置
TARGET_MACHINES = 10  
TARGET_AGVS = 4       
INPUT_DIM = 24        

# ================= 🛠️ 数据加载工具 (保持不变) =================
def load_fjs_file(file_path, total_machines):
    PT = []
    try:
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
    except Exception as e:
        print(f"❌ 解析 .fjs 文件失败: {e}")
        return None

def load_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            return data[0]
        return data
    except Exception as e:
        print(f"❌ 解析 .pkl 文件失败: {e}")
        return None

# ================= 🚀 主程序 =================
def run_single_visualization():
    print(f"🚀 开始单案例可视化...")
    print(f"📄 目标文件: {TEST_FILE}")
    print(f"🧠 加载模型: {CHECKPOINT_NAME}")

    # 1. 加载数据
    if TEST_FILE.endswith('.fjs') or TEST_FILE.endswith('.txt'):
        pt_matrix = load_fjs_file(TEST_FILE, TARGET_MACHINES)
    elif TEST_FILE.endswith('.pkl'):
        pt_matrix = load_pkl_file(TEST_FILE)
    else:
        print("❌ 未知文件格式，请确保是 .fjs 或 .pkl")
        return

    if pt_matrix is None:
        return

    print(f"📊 数据加载成功: {len(pt_matrix)} 个工件")

    # 2. 初始化环境
    env = EnvWorkShop(machine_num=TARGET_MACHINES)
    
    # 3. 初始化智能体
    jobagent = Job_Agent(input_size=INPUT_DIM, batch_size=1, total_updates=1)
    agvagent = AGV_Agent(input_size=INPUT_DIM, batch_size=1, total_updates=1)

    # 4. 加载权重
    try:
        job_actor_path = os.path.join(MODEL_DIR, f'job_actor_{CHECKPOINT_NAME}.pt')
        agv_path_candidates = [
            os.path.join(MODEL_DIR, f'agv_actor_{CHECKPOINT_NAME}.pt'),
            os.path.join(MODEL_DIR, f'agv_actor{CHECKPOINT_NAME}.pt')
        ]
        
        agv_actor_path = None
        for p in agv_path_candidates:
            if os.path.exists(p):
                agv_actor_path = p
                break
        
        if not agv_actor_path:
            raise FileNotFoundError(f"找不到 AGV 模型: {agv_path_candidates}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        jobagent.job_actor.load_state_dict(torch.load(job_actor_path, map_location=device))
        agvagent.agv_actor.load_state_dict(torch.load(agv_actor_path, map_location=device))
        
        jobagent.job_actor.to(device)
        agvagent.agv_actor.to(device)
        jobagent.job_actor.eval()
        agvagent.agv_actor.eval()
        print("✅ 模型权重加载完毕")

    except Exception as e:
        print(f"❌ 模型加载错误: {e}")
        return

    # 5. 执行推理
    print("\n⏳ 正在进行调度推理...")
    start_time = time.time()
    
    with torch.no_grad():
        state, p_mask, j_mask, a_mask = env.reset(new_job_data=pt_matrix, new_agv_num=TARGET_AGVS)
        done = False
        
        step_count = 0
        while not done:
            j_act, _, _ = jobagent.choose_action(state, p_mask, j_mask, greedy=True)
            mid_s, p_mask, a_mask, m_act = env.job_step(j_act)
            a_act, _, _ = agvagent.choose_action(mid_s, p_mask, a_mask, greedy=True)
            real_a = a_act - env.num_jobs
            state, p_mask, j_mask, a_mask, _, done, _ = env.step(j_act, m_act, real_a)
            step_count += 1

    inference_time = time.time() - start_time
    final_makespan = env.FJSP.max

    print("-" * 40)
    print(f"🎉 调度完成！Makespan: {final_makespan:.2f}")
    print("-" * 40)

    # ================= 🔴 6. 画甘特图 (核心修改部分) =================
    print("🎨 正在生成甘特图...")
    try:
        # 1. 创建 Figure 和 Axis (画布)
        # figsize=(12, 6) 控制图片长宽比例
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        # 2. 调用 utils/gantte.py 里的函数
        # 必须把 ax 传进去！
        gantte.plot_gantt_on_axis(
            ax, 
            env.FJSP.Machines, 
            env.FJSP.AGVs, 
            title=f"Schedule Gantt Chart (Makespan: {final_makespan:.1f})",
            show_job_id=False # 论文配图建议设为 False，看着整洁
        )

        # 3. 保存和显示
        save_name = "single_gantt_chart.pdf"
        plt.tight_layout() # 自动调整边距，防止字被切掉
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
        plt.savefig("single_gantt_chart.png", format='png', dpi=300, bbox_inches='tight')
        
        print(f"✅ 图片已保存至: {save_name}")
        plt.show() # 弹窗显示

    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_single_visualization()