# 导入必备工具包
import math

import torch
from torch import nn


# 预定义
class Embeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        padding_idx: int
    ):
        super().__init__()
        
        self.d_model = d_model
        self.lut = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  self.lut(x) * math.sqrt(self.d_model)
