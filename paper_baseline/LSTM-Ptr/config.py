import torch

embed_size = 128
num_heads = 4
num_layers = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# 编码类工具
job_type = torch.tensor([1, 0], dtype=torch.float)  # 工件
agv_type = torch.tensor([0, 1], dtype=torch.float)  # AGV


# agv_trans = [
#     [0, 11, 9, 15, 9, 8],
#     [11, 0, 7, 11, 12, 12],
#     [9, 7, 0, 7, 11, 5],
#     [15, 11, 7, 0, 9, 11],
#     [9, 12, 11, 9, 0, 9],
#     [8, 12, 5, 11, 9, 0],
# ]

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