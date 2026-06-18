# gnn_actor.py

from torch import nn
import config
import os
import torch
from torch.distributions.categorical import Categorical
# 引入你刚才定义的 GNN Encoder
from gnn_model import GNN_Encoder


class JobActorNetwork(nn.Module):
    def __init__(self, input_size, dim_k=config.embed_size, dim_v=config.embed_size,
                 num_layers=config.num_layers,num_heads=config.num_heads, chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        super(JobActorNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir

        # 🟢【核心替换】使用 GNN Encoder 替代 Transformer Attention
        # GNN 的 hidden_dim 直接沿用 dim_k (128)
        self.encoder = GNN_Encoder(input_dim=input_size, hidden_dim=dim_k, num_layers=num_layers, num_heads=num_heads)

        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.Linear(dim_v, 1)
        )
        self.device = config.device
        self.to(self.device)

    def forward(self, state, padding_mask, action_mask):
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        # 维度调整：处理 batch_size=1
        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        # 🟢【核心替换】调用 GNN Encoder
        # 注意：padding_mask 在 GAT 里也用来屏蔽无效节点
        x = self.encoder(state, mask=padding_mask)

        # 后面的逻辑完全不变
        dist = self.fc_out(x).squeeze(-1)  # [batch_size, seq_len]
        dist = dist.masked_fill(action_mask == False, float("-1e9"))
        dist = Categorical(logits=dist)
        return dist

    def save_checkpoint(self, i=0):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'job_actor_gnn_{i}.pt')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.chkpt_dir, checkpoint_file)
        print(f"...Loading GNN Actor checkpoint from: {path}")
        self.load_state_dict(torch.load(path))


class AGVActorNetwork(nn.Module):
    def __init__(self, input_size, dim_k=config.embed_size, dim_v=config.embed_size,
                 num_layers=config.num_layers, num_heads=config.num_heads,chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        super(AGVActorNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir

        # 同样加上 num_heads
        self.encoder = GNN_Encoder(input_dim=input_size, hidden_dim=dim_k, num_layers=num_layers, num_heads=num_heads)

        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.Linear(dim_v, 1)
        )
        self.device = config.device
        self.to(self.device)

    def forward(self, state, padding_mask, action_mask):
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        # 🟢【核心替换】
        x = self.encoder(state, mask=padding_mask)

        dist = self.fc_out(x).squeeze(-1)
        dist = dist.masked_fill(action_mask == False, float("-1e9"))
        dist = Categorical(logits=dist)
        return dist

    def save_checkpoint(self, i=0):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'agv_actor_gnn_{i}.pt')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.chkpt_dir, checkpoint_file)
        print(f"...Loading GNN AGV checkpoint from: {path}")
        self.load_state_dict(torch.load(path))