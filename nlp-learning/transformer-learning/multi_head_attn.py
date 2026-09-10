import torch
from torch import nn
from utils import attention, clones


class MultiHeadedAttention(nn.Module):
    """多头注意力机制 (Multi-Head Attention)"""

    def __init__(self, head: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % head == 0, "d_model 必须能被 head 整除"

        self.d_k = d_model // head
        self.head = head

        # 定义 4 个线性层：分别对应 Q、K、V 的投影，以及最后的 Output 拼接投影
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn: torch.Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        if mask is not None:
            # 扩展为 (batch_size, 1, 1, seq_len)，以便与 (batch_size, head, seq_len, d_k) 正确广播
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1. 线性变换投影并分头 (batch_size, seq_len, d_model) -> (batch_size, head, seq_len, d_k)
        query, key, value = [
            linear(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 2. 计算点积注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. 多头拼接还原 (batch_size, head, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 4. 经过输出线性层融合信息
        return self.linears[-1](x)