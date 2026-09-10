import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    自定义 Layer Normalization 层
    数学公式: y = [(x - μ) / √(σ² + ε)] * γ + β
    """

    def __init__(self, features: int, eps: float = 1e-6):
        """
        :param features: 特征维度 (例如 d_model = 512)
        :param eps: 防止分母为 0 的数值微小量
        """
        super().__init__()

        # 1. 规范化命名：使用深度学习通用的 gamma (缩放) 和 beta (平移)
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 来自上一层的输出，形状为 (batch_size, seq_len, features)
        :return: 规范化后的张量，形状与 x 完全一致
        """
        # 1. 计算最后一个维度的均值
        mean = x.mean(dim=-1, keepdim=True)

        # 2. 计算最后一个维度的方差 (必须指定 unbiased=False，使用总体方差)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 3. 标准 LayerNorm 公式：eps 必须加在方差 var 内部再开平方根！
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 4. 仿射变换 (Scale & Shift)
        return self.gamma * x_norm + self.beta