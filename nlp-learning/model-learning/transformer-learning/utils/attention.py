import math

import torch
import torch.nn.functional as F
from torch import nn


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 缩放点积注意力机制 (Scaled Dot-Product Attention)
    d_k = query.size(-1)
    # 按照注意力公式，将query与key的转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数根号下d_k，这种计算方法也称为缩放点积注意力计算。
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 使用 tensor 的 masked_fill 方法，将掩码张量和 scores 张量每个位置一一比较，如果掩码张量处为0
        # 则对应的 scores 张量用浮点数据表达的最小值来替换，如下演示
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

    # 对 scores 的最后一维进行 softmax 操作，使用 F.softmax 方法，第一个参数是 softmax 对象，第二个是目标维度。
    # 即将张量中的每一行都进行归一化 dim = -1
    # 这样获得最终的注意力张量
    attn_weights = F.softmax(scores, dim=-1)

    # 之后判断是否使用 dropout 进行随机置0
    if dropout is not None:
        # 将 p_attn 传入 dropout 对象中进行'丢弃'处理
        attn_weights = dropout(attn_weights)

    # 最后，根据公式将 p_attn 与 value 张量相乘获得最终的 query 注意力表示，同时返回注意力张量
    return torch.matmul(attn_weights, value), attn_weights