from collections.abc import Callable

import torch
from layer_norm import LayerNorm
from torch import nn


class SublayerConnection(nn.Module):
    """
    Transformer 残差子层连接 (Residual Connection + LayerNorm + Dropout)
    默认采用 Pre-LN 结构: Output = x + Dropout(SubLayer(LayerNorm(x)))
    """

    def __init__(self, size: int, dropout: float = 0.1):
        """
        :param size: 特征维度 (即 d_model，例如 512)
        :param dropout: Dropout 丢弃概率
        """
        super().__init__()  # Python 3 简化继承调用
        self.norm = LayerNorm(size)  # 层规范化
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播函数

        :param x: 来自上一层的输入张量，形状为 (batch_size, seq_len, d_model)
        :param sublayer: 子层函数/模块 (例如 MultiHeadedAttention 或 PositionwiseFeedForward)
        :return: 经过残差连接与规范化后的张量，形状与 x 完全一致
        """
        # Pre-LN 计算路径:
        # 1. 对输入 x 进行规范化: self.norm(x)
        # 2. 传入子层计算特征: sublayer(...)
        # 3. 对子层输出做 Dropout: self.dropout(...)
        # 4. 残差跳跃连接: x + ...
        return x + self.dropout(sublayer(self.norm(x)))