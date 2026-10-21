import torch


def make_padding_mask(vocab: torch.Tensor, padding_idx: int):
    """生成 Padding 掩码: (batch_size, 1, seq_len)"""
    return (vocab != padding_idx).unsqueeze(1)

def make_causal_mask(size: int):
    """生成因果下三角掩码 (防止未来词泄露): (1, size, size)"""
    causal_mask = (1, size, size)
    return torch.tril(torch.ones(causal_mask)).bool()