import torch
from torch import nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, dropout_rate=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, embed_size)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size),
        )
        self.norm1   = nn.LayerNorm(embed_size)
        self.norm2   = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        # Pre-Norm Attention
        x_normed = self.norm1(x)
        attn_out, _ = self.mha(x_normed, x_normed, x_normed,
                               key_padding_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        # Pre-Norm FFN
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x
