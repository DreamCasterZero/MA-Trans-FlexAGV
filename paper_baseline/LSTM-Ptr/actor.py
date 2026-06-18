import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import config
import os


class LSTMPtrBase(nn.Module):
    """
    基类：封装通用的 LSTM-Pointer 逻辑
    修改点：支持自定义 hidden_dim 和 num_layers
    """

    def __init__(self, input_size, hidden_dim=128, num_layers=3):  # <--- 默认改为 3
        super(LSTMPtrBase, self).__init__()

        # 1. Encoder: 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,  # <--- 使用传入的层数
            batch_first=True,
            bidirectional=True
        )

        # LSTM 输出维度 (双向所以乘2)
        self.enc_out_dim = hidden_dim * 2

        # 2. Attention Mechanism (Pointer)
        self.W_q = nn.Linear(self.enc_out_dim, self.enc_out_dim, bias=False)
        self.W_k = nn.Linear(self.enc_out_dim, self.enc_out_dim, bias=False)
        self.v = nn.Linear(self.enc_out_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward_ptr(self, state, padding_mask, action_mask, temperature=1.0):
        # ... (保持原样，逻辑不需要变) ...
        lstm_out, _ = self.lstm(state)
        valid_mask = (~padding_mask).float().unsqueeze(-1)
        sum_feat = torch.sum(lstm_out * valid_mask, dim=1)
        valid_count = torch.clamp(valid_mask.sum(dim=1), min=1.0)
        global_query = sum_feat / valid_count

        proj_query = self.W_q(global_query).unsqueeze(1)
        proj_keys = self.W_k(lstm_out)
        energy = torch.tanh(proj_query + proj_keys)
        scores = self.v(energy).squeeze(-1)

        final_mask = action_mask & (~padding_mask)
        scores = scores.masked_fill(final_mask == False, float("-1e9"))
        scores = scores / temperature
        dist = Categorical(logits=scores)
        return dist


class JobActorLSTMPtr(LSTMPtrBase):
    def __init__(self, input_size, dim_k=None, dim_v=None, num_heads=None, num_layers=3,
                 chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        # 🟢 关键修改：
        # 1. hidden_dim 设为 config.embed_size (128)
        # 2. num_layers 设为 config.num_layers (3)
        # 如果调用时不传，就用默认值 3
        super(JobActorLSTMPtr, self).__init__(
            input_size=input_size,
            hidden_dim=config.embed_size if hasattr(config, 'embed_size') else 128,
            num_layers=num_layers
        )
        self.chkpt_dir = chkpt_dir
        self.device = config.device
        self.to(self.device)

    def forward(self, state, padding_mask, action_mask, temperature=1.0):
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)
        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        return self.forward_ptr(state, padding_mask, action_mask, temperature)

    # ... save/load checkpoint 保持不变 ...
    # def save_checkpoint(self, i=0):
    #     if not os.path.exists(self.chkpt_dir): os.makedirs(self.chkpt_dir)
    #     torch.save(self.state_dict(), os.path.join(self.chkpt_dir, f'job_actor_{i}.pt'))
    #
    # def load_checkpoint(self, checkpoint_file):
    #     path = os.path.join(self.chkpt_dir, checkpoint_file)
    #     print(f"...Loading Job Actor from: {path}")
    #     self.load_state_dict(torch.load(path))




class AGVActorLSTMPtr(LSTMPtrBase):
    def __init__(self, input_size, dim_k=None, dim_v=None, num_heads=None, num_layers=3,
                 chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        # 🟢 同样对齐参数
        super(AGVActorLSTMPtr, self).__init__(
            input_size=input_size,
            hidden_dim=config.embed_size if hasattr(config, 'embed_size') else 128,
            num_layers=num_layers
        )
        self.chkpt_dir = chkpt_dir
        self.device = config.device
        self.to(self.device)

    def forward(self, state, padding_mask, action_mask, temperature=1.0):
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)
        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        return self.forward_ptr(state, padding_mask, action_mask, temperature)

    # ... save/load checkpoint 保持不变 ...
    # def save_checkpoint(self, i=0):
    #     if not os.path.exists(self.chkpt_dir): os.makedirs(self.chkpt_dir)
    #     torch.save(self.state_dict(), os.path.join(self.chkpt_dir, f'agv_actor_{i}.pt'))
    #
    # def load_checkpoint(self, checkpoint_file):
    #     path = os.path.join(self.chkpt_dir, checkpoint_file)
    #     print(f"...Loading AGV Actor from: {path}")
    #     self.load_state_dict(torch.load(path))