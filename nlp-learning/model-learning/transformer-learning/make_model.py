import copy

from blocks.decoder import Decoder, DecoderLayer
from blocks.encoder import Encoder, EncoderLayer
from blocks.generator import Generator
from blocks.transformer import Transformer
from modules.embeddings import Embeddings
from modules.feed_forward import PositionwiseFeedForward
from modules.mult_head_attn import MultiHeadedAttention
from modules.positional import PositionalEncoding
from torch import nn


def make_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    num_heads: int = 8,
    dropout_rate: float = 0.1,
    padding_idx: int = 0,
) -> Transformer:
    """构建并初始化一个完整的 Transformer 实例"""
    
    # 基础注意力与 FFN 模板
    mh_attn = MultiHeadedAttention(head=num_heads, d_model=d_model, dropout_rate=dropout_rate)
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    # 实例化 Encoder 与 Decoder
    encoder = Encoder(
        layer=EncoderLayer(
            size=d_model, 
            self_attn=copy.deepcopy(mh_attn), 
            feed_forward=copy.deepcopy(ffn), 
            dropout_rate=dropout_rate
        ),
        N=N,
    )

    decoder = Decoder(
        layer=DecoderLayer(
            size=d_model,
            self_attn=copy.deepcopy(mh_attn),
            src_attn=copy.deepcopy(mh_attn),
            feed_forward=copy.deepcopy(ffn),
            dropout_rate=dropout_rate,
        ),
        N=N,
    )

    # 2. 直接使用 nn.Sequential 组装嵌入层（与教学完全一致）
    src_embed = nn.Sequential(
        Embeddings(d_model=d_model, vocab_size=src_vocab_size, padding_idx=padding_idx),
        PositionalEncoding(d_model=d_model, dropout_rate=dropout_rate),
    )
    tgt_embed = nn.Sequential(
        Embeddings(d_model=d_model, vocab_size=tgt_vocab_size, padding_idx=padding_idx),
        PositionalEncoding(d_model=d_model, dropout_rate=dropout_rate),
    )
    generator = Generator(d_model=d_model, vocab_size=tgt_vocab_size)

    # 组装完整模型
    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=generator,
    )

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵，
    # 这里一但判断参数的维度大于 1，则会将其初始化成一个服从均匀分布的矩阵。
    # Xavier 参数初始化 (维度大于 1 的权重矩阵统一初始化)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model