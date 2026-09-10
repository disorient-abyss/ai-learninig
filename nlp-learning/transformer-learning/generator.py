import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    """Transformer 输出生成器：将隐层特征投影到词表维度并输出对数概率"""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 解码器输出特征，形状为 (batch_size, seq_len, d_model)
        :return: 词表对数概率，形状为 (batch_size, seq_len, vocab_size)
        """
        return F.log_softmax(self.proj(x), dim=-1)