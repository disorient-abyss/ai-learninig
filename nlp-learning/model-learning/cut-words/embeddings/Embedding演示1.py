import math

import torch
from torch import nn


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, padding_idx: int | None = None):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

if __name__ == "__main__":
# 1. 自动选择计算设备 (CUDA GPU > Apple Silicon MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"当前使用的设备: {device}")

    # 2. 超参数设置
    d_model = 512
    vocab_size = 1000
    pad_idx = 0

    # 3. 直接在目标设备上创建张量
    x = torch.tensor(
        [[100, 2, 421, pad_idx], [491, 999, 1, 221]],
        dtype=torch.long,
        device=device
    )

    # 4. 实例化模型并推送到目标设备
    emb = Embeddings(d_model, vocab_size, padding_idx=pad_idx).to(device)
    # 5. 前向传播
    embr = emb(x)

    print("embr:", embr)
    print("embr.shape:", embr.shape)

    