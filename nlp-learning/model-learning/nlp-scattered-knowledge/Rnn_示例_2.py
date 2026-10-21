# 导入工具包
import torch
from torch import nn

lstm = nn.LSTM(5, 6, 1)
input = torch.randn(1, 3, 5)
h0 = torch.randn(1, 3, 6)
c0 = torch.randn(1, 3, 6)
output, (hn, cn) = lstm(input, (h0, c0))

print(output)
print('-' * 50)
print(hn)
print('-' * 50)
print(cn)