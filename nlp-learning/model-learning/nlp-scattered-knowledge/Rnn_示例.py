# 导入工具包
import torch
from torch import nn

rnn = nn.RNN(5, 6, 2)
input = torch.randn(1, 3, 5)
h0 = torch.randn(2, 3, 6)
output, hn = rnn(input, h0)

print(output)
print(hn)