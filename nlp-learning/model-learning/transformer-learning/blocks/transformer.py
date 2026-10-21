import torch
from blocks.decoder import Decoder
from blocks.encoder import Encoder
from blocks.generator import Generator
from torch import nn


class Transformer(nn.Module):
    """Transformer 顶层容器，串联 Embedding、Encoder、Decoder 和 Generator"""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param src: 源端 Token 序列，形状为 (batch_size, src_len)
        :param tgt: 目标端 Token 序列，形状为 (batch_size, tgt_len)
        :param src_mask: 源端 Padding Mask
        :param tgt_mask: 目标端因果掩码及 Padding Mask
        :return: 词表对数概率分布，形状为 (batch_size, tgt_len, vocab_size)
        """
        # 1. 编码源序列得到 memory
        memory = \
            self.encoder(
                self.src_embed(src), 
                mask=src_mask
            )

        # 2. 解码目标序列并融合 memory
        out = \
            self.decoder(
                x=self.tgt_embed(tgt),
                memory=memory,
                source_mask=src_mask,
                target_mask=tgt_mask,
            )

        # 3. 映射到词表对数概率
        return self.generator(out)


