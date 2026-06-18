import torch

embed_size = 128
num_heads = 4
num_layers = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# 编码类工具
job_type = torch.tensor([1, 0], dtype=torch.float)  # 工件
agv_type = torch.tensor([0, 1], dtype=torch.float)  # AGV

agv_trans = [
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

# agv_trans = [
#     [0, 11, 9, 15, 9, 8],
#     [11, 0, 7, 11, 12, 12],
#     [9, 7, 0, 7, 11, 5],
#     [15, 11, 7, 0, 9, 11],
#     [9, 12, 11, 9, 0, 9],
#     [8, 12, 5, 11, 9, 0],
# ]




# def evaluate_model(env, job_agent, agv_agent):
#     """
#     在当前 Env 上跑一局 Greedy 评估
#     """
#     state, padding_mask, job_mask, agv_mask = env.reset(
#         new_job_data=PT,  # 这里的 PT 是 config.py 里定义的变量
#         new_agv_num=num_agvs
#     )
#     done = False
#
#     while not done:
#         # 1. 工件智能体 (Greedy)
#         # 注意: 必须传入 mask
#         jobaction, _, _ = job_agent.choose_action(state, padding_mask, job_mask, greedy=True)
#
#         # 2. 环境执行半步
#         mid_state, padding_mask, agv_mask, machaction = env.job_step(jobaction)
#
#         # 3. AGV 智能体 (Greedy)
#         agvaction, _, _ = agv_agent.choose_action(mid_state, padding_mask, agv_mask, greedy=True)
#         # 智能体选的是序列中的位置(比如第6个Token)，我们要转成 AGV 的 ID(第1个AGV)
#         real_agv_action = agvaction - env.num_jobs
#         # 4. 环境执行完一步
#         # 注意: env.step 返回了很多东西，我们需要用 _ 忽略不需要的
#         state, padding_mask, job_mask, agv_mask, reward, done, info = env.step(jobaction, machaction, real_agv_action)
#
#     return env.FJSP.max  # 返回完工时间 (Makespan)

def experi_dir():
    directory_name = f'TrainingResult'
    return directory_name

def pad_to_length(token, target_length=16): # 正式的时候需要设置为32
    current_length = token.shape[0]
    if current_length > target_length:
        raise ValueError(f"Token length {current_length} exceeds target length {target_length}.")
    elif current_length == target_length:
        return token
    else:
        pad_len = target_length - current_length
        padding = torch.zeros(pad_len, dtype=token.dtype)
        return torch.cat([token, padding], dim=0)