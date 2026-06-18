from torch import nn
import config
import os
from attention import SelfAttention
import torch
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, input_size, dim_k=config.embed_size, dim_v=config.embed_size, num_heads=config.num_heads,
                 num_layers=config.num_layers, name='critic', chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        super(CriticNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name  # 用于区分保存文件名 (job_critic 或 agv_critic)

        # === 修改点 1: 传入 dropout_rate ===
        # 确保 Critic 也使用 Dropout，防止对特定状态过拟合
        self.attention = SelfAttention(input_size, embed_size=dim_k, num_heads=num_heads, dropout_rate=0.1)

        self.more_layers = nn.ModuleList([
            SelfAttention(dim_k, dim_k, num_heads, dropout_rate=0.1) for _ in range(num_layers - 1)
        ])

        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.Linear(dim_v, 1),
        )
        self.device = config.device
        self.to(self.device)
        # === 🟢 [修正版] 权重初始化 ===
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 关键修正：检查维度
                if param.data.ndim >= 2:
                    if 'fc_out' in name and '0' not in name:  # 最后一层
                        # Critic 最后一层不需要 0.01，用 1.0 即可
                        nn.init.orthogonal_(param, gain=1.0)
                    else:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    # LayerNorm weight
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, state, padding_mask):
        """
        state: (Batch, Seq_Len, Feat_Dim)
        padding_mask: (Batch, Seq_Len) -> True 代表 Padding
        """
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)

        # 1. 维度调整：处理 batch_size=1
        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        # 2. Transformer 编码 (带 Mask)
        # 此时调用的是 attention.py 里的 Pre-Norm 逻辑
        x = self.attention(state, mask=padding_mask)

        for layer in self.more_layers:
            x = layer(x, mask=padding_mask)

        # === 3. Masked Global Pooling (保持不变，这部分逻辑写得很好) ===
        # 目标：把 (Batch, 50, 128) 压缩成 (Batch, 128)

        # A. 创建数值掩码 (True/False -> 0.0/1.0)
        # valid_mask: (Batch, Seq_Len, 1)
        valid_mask = (padding_mask == False).float().unsqueeze(-1)

        # B. 屏蔽无效特征
        x = x * valid_mask

        # C. 求平均 (Sum / Count)
        sum_x = torch.sum(x, dim=1)  # (Batch, 128)
        count = torch.sum(valid_mask, dim=1)  # (Batch, 1)

        # 避免除以 0
        mean_x = sum_x / (count + 1e-9)

        # 4. 输出价值
        value = self.fc_out(mean_x)  # (Batch, 1)

        return value

    def save_checkpoint(self, i=0):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.name}_{i}.pt')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.chkpt_dir, checkpoint_file)
        print(f"...Loading Critic checkpoint from: {path}")
        self.load_state_dict(torch.load(path))