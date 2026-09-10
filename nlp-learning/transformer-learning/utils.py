import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """生成 N 个相同网络层的深拷贝列表"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放点积注意力机制 (Scaled Dot-Product Attention)"""
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    return torch.matmul(attn_weights, value), attn_weights


def make_pad_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """生成 Padding 掩码: (batch_size, 1, seq_len)"""
    return (tokens != pad_idx).unsqueeze(1)


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """生成因果下三角掩码 (防止未来词泄露): (1, size, size)"""
    return torch.tril(torch.ones((1, size, size), device=device)).bool()


def make_target_mask(tgt_tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """生成目标端复合掩码 (Padding 掩码 & 因果未来词掩码): (batch_size, tgt_len, tgt_len)"""
    tgt_len = tgt_tokens.size(1)
    pad_mask = make_pad_mask(tgt_tokens, pad_idx)
    causal_mask = make_causal_mask(tgt_len, device=tgt_tokens.device)
    return pad_mask & causal_mask