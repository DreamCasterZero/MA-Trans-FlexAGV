# gnn_critic.py

from torch import nn
import config
import os
import torch
from gnn_model import GNN_Encoder


class CriticNetwork(nn.Module):
    def __init__(self, input_size, dim_k=config.embed_size, dim_v=config.embed_size,
                 num_layers=config.num_layers, name='critic', num_heads=config.num_heads,chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        super(CriticNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name

        # Critic 初始化参数里也有 num_heads=config.num_heads，直接传给 Encoder
        self.encoder = GNN_Encoder(input_dim=input_size, hidden_dim=dim_k, num_layers=num_layers, num_heads=num_heads)

        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.Linear(dim_v, 1),
        )
        self.device = config.device
        self.to(self.device)

    def forward(self, state, padding_mask):
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)

        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        # 🟢【核心替换】GNN 编码
        x = self.encoder(state, mask=padding_mask)

        # === 下面的 Pooling 逻辑完全复用 ===
        # 把 GNN 提取的节点特征聚合成一个图特征 (Graph Embedding)
        valid_mask = (padding_mask == False).float().unsqueeze(-1)
        x = x * valid_mask
        sum_x = torch.sum(x, dim=1)
        count = torch.sum(valid_mask, dim=1)
        mean_x = sum_x / (count + 1e-9)

        value = self.fc_out(mean_x)

        return value

    def save_checkpoint(self, i=0):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.name}_gnn_{i}.pt')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.chkpt_dir, checkpoint_file)
        print(f"...Loading GNN Critic checkpoint from: {path}")
        self.load_state_dict(torch.load(path))