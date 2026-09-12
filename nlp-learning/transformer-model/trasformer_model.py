import copy

import torch
from decoder import Decoder, DecoderLayer
from embedding import Embeddings, PositionalEncoding  # 1. 导入这两个基础组件
from encoder import Encoder, EncoderLayer
from forward_net import PositionwiseFeedForward
from generator import Generator
from multi_head_attn import MultiHeadedAttention
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
        memory = self.encoder(self.src_embed(src), mask=src_mask)

        # 2. 解码目标序列并融合 memory
        out = self.decoder(
            x=self.tgt_embed(tgt),
            memory=memory,
            source_mask=src_mask,
            target_mask=tgt_mask,
        )

        # 3. 映射到词表对数概率
        return self.generator(out)


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    num_heads: int = 8,
    dropout: float = 0.1,
    pad_idx: int = 0,
) -> Transformer:
    """构建并初始化一个完整的 Transformer 实例"""
    
    # 基础注意力与 FFN 模板
    attn = MultiHeadedAttention(head=num_heads, d_model=d_model, dropout=dropout)
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    # 实例化 Encoder 与 Decoder
    encoder = Encoder(
        layer=EncoderLayer(size=d_model, self_attn=copy.deepcopy(attn), feed_forward=copy.deepcopy(ffn), dropout=dropout),
        N=N,
    )
    decoder = Decoder(
        layer=DecoderLayer(
            size=d_model,
            self_attn=copy.deepcopy(attn),
            src_attn=copy.deepcopy(attn),
            feed_forward=copy.deepcopy(ffn),
            dropout=dropout,
        ),
        N=N,
    )

    # 2. 直接使用 nn.Sequential 组装嵌入层（与教学完全一致）
    src_embed = nn.Sequential(
        Embeddings(d_model=d_model, vocab_size=src_vocab, padding_idx=pad_idx),
        PositionalEncoding(d_model=d_model, dropout=dropout),
    )
    tgt_embed = nn.Sequential(
        Embeddings(d_model=d_model, vocab_size=tgt_vocab, padding_idx=pad_idx),
        PositionalEncoding(d_model=d_model, dropout=dropout),
    )
    generator = Generator(d_model=d_model, vocab_size=tgt_vocab)

    # 组装完整模型
    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=generator,
    )

    # Xavier 参数初始化 (维度大于 1 的权重矩阵统一初始化)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model