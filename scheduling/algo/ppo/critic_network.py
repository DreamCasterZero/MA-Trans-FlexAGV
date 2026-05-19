import numpy as np
import torch
from torch import nn
from .transformer import SelfAttentionBlock


class CriticNetwork(nn.Module):
    def __init__(self, feat_dim, embed_size, num_heads, num_layers, dropout_rate=0.1):
        super().__init__()
        self.first_layer = SelfAttentionBlock(feat_dim, embed_size, num_heads, dropout_rate)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(embed_size, embed_size, num_heads, dropout_rate)
            for _ in range(num_layers - 1)
        ])
        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.data.ndim >= 2:
                    gain = 1.0 if 'fc_out.2' in name else np.sqrt(2)
                    nn.init.orthogonal_(param, gain=gain)
                else:
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, state, padding_mask):
        if state.dim() == 2:
            state        = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        x = self.first_layer(state, mask=padding_mask)
        for layer in self.layers:
            x = layer(x, mask=padding_mask)

        # Masked mean pooling: 忽略 padding 位置
        valid_mask = (~padding_mask).float().unsqueeze(-1)   # (B, L, 1)
        mean_x = torch.sum(x * valid_mask, dim=1) / (valid_mask.sum(dim=1) + 1e-9)
        return self.fc_out(mean_x)   # (B, 1)
