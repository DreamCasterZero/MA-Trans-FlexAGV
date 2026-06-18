import torch
import torch.nn as nn
import config
import os
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, input_size, dim_k=None, dim_v=None, num_heads=None, num_layers=3,
                 name='critic', chkpt_dir=os.path.join(config.experi_dir(), 'param')):
        super(CriticNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name

        # 🟢 对齐 hidden_dim
        hidden_dim = config.embed_size if hasattr(config, 'embed_size') else 128

        # 1. Encoder: 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,  # 🟢 对齐层数 (3)
            batch_first=True,
            bidirectional=True
        )

        # 2. Value Head
        # 输入维度: hidden_dim * 2 (双向)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.device = config.device
        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    if 'value_head' in name and '2' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    else:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, state, padding_mask):
        # ... (保持原样) ...
        state = state.float().to(self.device)
        padding_mask = padding_mask.to(self.device)
        if state.dim() == 2:
            state = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        lstm_out, _ = self.lstm(state)

        valid_mask = (~padding_mask).float().unsqueeze(-1)
        x = lstm_out * valid_mask
        sum_x = torch.sum(x, dim=1)
        count = torch.sum(valid_mask, dim=1)
        mean_x = sum_x / torch.clamp(count, min=1.0)

        value = self.value_head(mean_x)
        return value

    # ... save/load checkpoint 保持不变 ...
    def save_checkpoint(self, i=0):
        if not os.path.exists(self.chkpt_dir): os.makedirs(self.chkpt_dir)
        self.checkpoint_file = os.path.join(self.chkpt_dir, f'{self.name}_{i}.pt')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.chkpt_dir, checkpoint_file)
        print(f"...Loading Critic checkpoint from: {path}")
        self.load_state_dict(torch.load(path))