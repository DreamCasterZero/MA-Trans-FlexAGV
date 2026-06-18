from torch import nn
import config
import os
from attention import SelfAttention
import torch
from torch.distributions.categorical import Categorical
import numpy as np

class JobActorNetwork(nn.Module):
    def __init__(self, input_size, dim_k=config.embed_size, dim_v=config.embed_size, num_heads=config.num_heads,
                 num_layers=config.num_layers, chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        super(JobActorNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir

        # 1. 第一层 Attention (负责将 input_size 24 映射到 dim_k 128)
        # [新增] 传入 dropout_rate (建议在 config 里设一个 dropout, 或者默认 0.1)
        self.attention = SelfAttention(input_size, embed_size=dim_k, num_heads=num_heads, dropout_rate=0.1)

        # 2. 后续堆叠层 (输入输出都是 dim_k 128)
        self.more_layers = nn.ModuleList([
            SelfAttention(dim_k, dim_k, num_heads, dropout_rate=0.1) for _ in range(num_layers - 1)
        ])

        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.Linear(dim_v, 1)
        )
        self.device = config.device
        self.to(self.device)
        # === 🟢 [修正版] 权重初始化 ===
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 关键修正：LayerNorm 的 weight 是 1维的，不能正交初始化
                if param.data.ndim >= 2:
                    if 'fc_out' in name and '0' not in name:  # 最后一层 Linear
                        nn.init.orthogonal_(param, gain=0.01)
                    else:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    # 对于 1维的 weight (如 LayerNorm)，初始化为 1
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, state, padding_mask, action_mask, temperature=1.0):
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        # 此时调用的是 attention.py 里新改的 Pre-Norm 逻辑
        x = self.attention(state, mask=padding_mask)

        for layer in self.more_layers:
            x = layer(x, mask=padding_mask)

        dist = self.fc_out(x).squeeze(-1)

        # Mask 处理
        dist = dist.masked_fill(action_mask == False, float("-1e9"))

        # 温度缩放
        dist = dist / temperature

        dist = Categorical(logits=dist)
        return dist

    def save_checkpoint(self, i=0):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'job_actor_{i}.pt')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.chkpt_dir, checkpoint_file)
        print(f"...Loading checkpoint from: {path}")
        self.load_state_dict(torch.load(path))

