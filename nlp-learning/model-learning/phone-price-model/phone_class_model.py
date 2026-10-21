import torch
from torch import nn


class PhoneClassModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        # 定义结构化神经元模型
        self.net = nn.Sequential(
            # 第一隐藏层
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 第二隐藏层
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 输出层
            nn.Linear(8, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 数据会依次自动穿过上面的 1 -> 2 -> 3 -> 4 -> 5 -> 6 层
        return self.net(x)