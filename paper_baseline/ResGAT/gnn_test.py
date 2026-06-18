import os
import torch
# 引入你的环境和智能体
from gnn_env import EnvWorkShop
from gnn_ppo import Job_Agent, AGV_Agent

# === 1. 配置区域 ===
TEST_FILE_PATH = './10j_5m_001.fjs'  # 你想测的 FJS 文件路径
MODEL_DIR = 'TrainingResult/param'  # 模型保存文件夹
CHECKPOINT_NAME = 'gnn_best_10x5'  # 你想测的具体检查点名称 (不带后缀)
# 或者用 'best_10x5'

TARGET_MACHINES = 5  # 必须与 FJS 文件里的机器数一致 (10j_5m -> 5)
TARGET_AGVS = 2  # 想给这个测试配几台 AGV


# === 2. 工具函数 ===
def load_fjs_file(file_path, total_machines):
    """读取标准 .fjs 文件并转换为环境需要的 PT 矩阵"""
    PT = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过第一行
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
                    m_id = data[idx] - 1  # 1-based -> 0-based
                    time = data[idx + 1]
                    idx += 2
                    if m_id < total_machines:
                        op_machines[m_id] = time
                job_ops.append(op_machines)
            PT.append(job_ops)
    return PT


def run_test():
    print(f"🚀 开始测试: {TEST_FILE_PATH}")
    print(f"📂 加载模型: {CHECKPOINT_NAME} from {MODEL_DIR}")

    # 1. 准备数据
    if not os.path.exists(TEST_FILE_PATH):
        print("❌ 错误: 找不到 FJS 文件")
        return
    pt_matrix = load_fjs_file(TEST_FILE_PATH, TARGET_MACHINES)
    print(f"📊 数据加载成功: {len(pt_matrix)} 个工件")

    # 2. 初始化环境
    env = EnvWorkShop(machine_num=TARGET_MACHINES)

    # 3. 初始化 Agent
    # 注意: input_size 必须和训练时一致 (24)
    # total_updates 随便填一个非零值即可，测试时不训练
    jobagent = Job_Agent(input_size=24, batch_size=1, total_updates=10000)
    agvagent = AGV_Agent(input_size=24, batch_size=1, total_updates=10000)

    # 4. 加载权重
    try:
        # 构造路径 (注意文件名格式要和你 save 时的一致)
        # 你的代码里 save 是: jobagent.save(f'ckpt_{i}') -> job_actor_ckpt_{i}.pt
        # 所以这里要注意拼接逻辑

        # 假设文件名是标准的 job_actor_ + NAME + .pt
        # 如果你的 save 代码是 f'ckpt_{i}'，那生成的文件可能是 job_actor_ckpt_15000.pt

        # 为了保险，我们尝试几种常见的拼法，或者你手动去文件夹确认一下文件名
        # 下面按你提供的 save 函数逻辑: self.chkpt_dir + f'job_actor_{i}.pt'
        # 如果你传入的是 'ckpt_15000'，那文件应该是 job_actor_ckpt_15000.pt

        job_actor_path = os.path.join(MODEL_DIR, f'job_actor_{CHECKPOINT_NAME}.pt')
        agv_actor_path = os.path.join(MODEL_DIR, f'agv_actor{CHECKPOINT_NAME}.pt')  # 你的save代码里好像少个下划线？去看看 agvagent.save

        # 修复 AGV 文件名可能的拼写 (根据之前的记录，save里可能是 agv_actorckpt_...)
        if not os.path.exists(agv_actor_path):
            agv_actor_path = os.path.join(MODEL_DIR, f'agv_actor_{CHECKPOINT_NAME}.pt')

        jobagent.job_actor.load_state_dict(torch.load(job_actor_path))
        agvagent.agv_actor.load_state_dict(torch.load(agv_actor_path))

        jobagent.job_actor.eval()  # 切换到评估模式
        agvagent.agv_actor.eval()
        print("✅ 模型权重加载完毕!")

    except FileNotFoundError as e:
        print(f"❌ 加载失败，找不到模型文件: {e}")
        print("💡 提示: 请去 TrainingResult/param 文件夹确认一下确切的文件名。")
        return

    # 5. 执行推理 (Greedy)
    state, p_mask, j_mask, a_mask = env.reset(new_job_data=pt_matrix, new_agv_num=TARGET_AGVS)
    done = False

    print("\n🏁 开始调度仿真...")
    step_count = 0

    with torch.no_grad():
        while not done:
            # Job Action
            j_act, _, _ = jobagent.choose_action(state, p_mask, j_mask, greedy=True)

            # Env Half Step
            mid_s, p_mask, a_mask, m_act = env.job_step(j_act)

            # AGV Action
            a_act, _, _ = agvagent.choose_action(mid_s, p_mask, a_mask, greedy=True)
            real_a = a_act - env.num_jobs

            # Env Full Step
            state, p_mask, j_mask, a_mask, _, done, _ = env.step(j_act, m_act, real_a)
            step_count += 1

            # 可选: 打印每一步的动作
            # print(f"Step {step_count}: Job {j_act} -> Machine {m_act} (AGV {real_a})")

    # 6. 打印结果
    print(f"\n🎉 仿真结束!")
    print(f"⏱️ 最终 Makespan: {env.FJSP.max:.2f}")

if __name__ == '__main__':
    run_test()