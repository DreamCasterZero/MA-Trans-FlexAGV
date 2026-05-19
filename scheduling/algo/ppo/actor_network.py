import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .transformer import SelfAttentionBlock


class ActorNetwork(nn.Module):
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
                    gain = 0.01 if 'fc_out.2' in name else np.sqrt(2)
                    nn.init.orthogonal_(param, gain=gain)
                else:
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, state, padding_mask, action_mask, temperature=1.0):
        if state.dim() == 2:
            state        = state.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            action_mask  = action_mask.unsqueeze(0)

        x = self.first_layer(state, mask=padding_mask)
        for layer in self.layers:
            x = layer(x, mask=padding_mask)

        logits = self.fc_out(x).squeeze(-1)
        logits = logits.masked_fill(~action_mask, float('-1e9'))
        logits = logits / temperature
        return Categorical(logits=logits)
