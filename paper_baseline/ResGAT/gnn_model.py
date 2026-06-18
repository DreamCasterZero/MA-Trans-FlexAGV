import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    修改版 GAT：增加了 Residual 和 LayerNorm，防止梯度消失
    """

    def __init__(self, in_features, out_features, dropout, alpha, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_heads = num_heads

        assert out_features % num_heads == 0
        self.head_dim = out_features // num_heads

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.orthogonal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * self.head_dim, 1)))
        nn.init.orthogonal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # 🟢【新增】层归一化，训练稳定的关键
        self.ln = nn.LayerNorm(out_features)
        # 🟢【新增】如果输入输出维度不匹配，需要线性投影来做残差
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = None

    def forward(self, h, mask=None):
        batch_size, N, _ = h.size()

        # 保存原始输入用于残差
        residual = h

        # 1. 线性变换 & 分头
        Wh = torch.matmul(h, self.W)
        Wh = Wh.view(batch_size, N, self.num_heads, self.head_dim)

        # 2. Attention 计算 (保持不变)
        Wh_trans = Wh.permute(0, 2, 1, 3)
        a_l = self.a[:, :self.head_dim, :]
        a_r = self.a[:, self.head_dim:, :]
        e_l = torch.einsum('bhnd,hdl->bhnl', Wh_trans, a_l)
        e_r = torch.einsum('bhnd,hdl->bhnl', Wh_trans, a_r)
        e = e_l + e_r.permute(0, 1, 3, 2)
        e = self.leakyrelu(e)

        if mask is not None:
            mask_ex = mask.view(batch_size, 1, 1, N)
            e = e.masked_fill(mask_ex, -1e9)

        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 3. 聚合
        h_prime = torch.matmul(attention, Wh_trans)
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous()
        h_prime = h_prime.view(batch_size, N, self.out_features)

        # 🟢【核心修改】加入残差连接 (Residual Connection)
        # 如果维度不对，先投影再相加
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        # Output = LayerNorm(Activation(Aggregated) + Residual)
        # 这是最标准的 Pre-Norm 或 Post-Norm 写法之一
        return self.ln(F.elu(h_prime) + residual)


class GNN_Encoder(nn.Module):
    """
    完全对齐 Transformer 参数的 GNN Encoder
    """

    def __init__(self, input_dim, hidden_dim, num_layers=3, num_heads=8):  # 默认8头
        super(GNN_Encoder, self).__init__()
        self.layers = nn.ModuleList()

        # 第一层 (Input -> Hidden)
        self.layers.append(GraphAttentionLayer(input_dim, hidden_dim, dropout=0.1, alpha=0.2, num_heads=num_heads))

        # 后续层 (Hidden -> Hidden)
        for _ in range(num_layers - 1):
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, dropout=0.1, alpha=0.2, num_heads=num_heads))

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x