import torch
from torch import nn
import config


class SelfAttention(nn.Module):
    def __init__(self, input_size, embed_size=config.embed_size, num_heads=config.num_heads, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        # 1. 输入映射 (保持不变)
        self.input_projection = nn.Linear(input_size, embed_size)

        # 2. 🔥核心修改：使用官方优化过的 MultiheadAttention🔥
        # batch_first=True 让输入输出都是 (Batch, Seq, Dim)，符合你的习惯
        self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,
                                         dropout=dropout_rate, batch_first=True)

        # 3. Feed Forward (保持不变)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size)
        )

        # 4. Norm (Pre-Norm)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 5. Dropout
        self.dropout = nn.Dropout(dropout_rate)

        self.device = config.device
        self.to(self.device)

    def forward(self, state, mask=None):
        """
        state: (Batch, Seq_Len, Input_Size)
        mask: (Batch, Seq_Len) -> True 代表 Padding (要屏蔽)
        """
        # 1. 映射
        x = self.input_projection(state)  # (Batch, Seq, Embed)

        # === Block 1: Attention (Pre-Norm) ===
        x_norm = self.norm1(x)

        # 🔥 使用官方 MHA 接口 🔥
        # key_padding_mask: PyTorch 要求 True 代表 Padding (跟你的定义一致)
        # need_weights=False: 不返回 attention map，速度更快
        attn_output, _ = self.mha(query=x_norm, key=x_norm, value=x_norm,
                                  key_padding_mask=mask,
                                  need_weights=False)

        # 残差连接 + Dropout (MHA 内部已经有 Dropout了，这里通常是对残差结果再做一次，或者直接加)
        x = x + self.dropout(attn_output)

        # === Block 2: Feed Forward (Pre-Norm) ===
        x_norm2 = self.norm2(x)
        ff_out = self.feed_forward(x_norm2)
        x = x + self.dropout(ff_out)

        return x