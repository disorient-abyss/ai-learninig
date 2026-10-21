import copy

from torch import nn

# 首先需要定义克隆函数，因为在多头注意力机制的实现中，用到多个结构相同的线性层。
# 我们将使用 clone 函数将他们一同初始化在一个网络层列表对象中。之后的结构中也会用到该函数。

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """生成 N 个相同网络层的深拷贝列表"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
