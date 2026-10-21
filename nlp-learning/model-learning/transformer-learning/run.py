import copy

import torch
from blocks.decoder import Decoder, DecoderLayer
from blocks.encoder import Encoder, EncoderLayer
from blocks.generator import Generator
from modules.embeddings import Embeddings
from modules.feed_forward import PositionwiseFeedForward
from modules.layer_norm import LayerNorm
from modules.mult_head_attn import MultiHeadedAttention
from modules.positional import PositionalEncoding
from modules.sublayer_connection import SublayerConnection
from torch import nn
from utils.attention import attention
from utils.subsequent_mask import make_causal_mask, make_padding_mask

# 1. 自动判断设备并设为全局默认
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def embeddings_test(input_vocab: torch.Tensor,d_model: int, vocab_size: int, padding_idx: int) -> torch.Tensor:

    # 1  通过 embedings 
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)

    print('-' * 50)
    print(f"result_emb:{result_emb}")
    print(f"result_emb.shape:{result_emb.shape}")
    print('-' * 50)

def positional_encoding_test(input_vocab: torch.Tensor, d_model: int, vocab_size: int, padding_idx: int) -> torch.Tensor:
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb = pe(result_emb)

    print(f"pe:{result_pe_emb}")
    print(f"pe.shape:{result_pe_emb.shape}")
    print('-' * 50)

def mask_test(padding_idx: int):
    # 模拟 2 个句子，长度都为 4
    # 句子 1 有一个 pad，句子 2 有两个 pad
    # 输入序列张量 tokens 的默认形状通常是 (batch_size, seq_len)
    tokens = torch.tensor([
        [101, 205, 308,   0],  # batch 1
        [101, 506,   0,   0]   # batch 2
    ])
    pad_mask = make_padding_mask(tokens, padding_idx)
    causal_mask = make_causal_mask(tokens.size(1))

    print(f"pad_mask: \n{pad_mask}")
    print()
    print(f"causal_mask: \n{causal_mask}")
    print('-' * 50)

def attention_test(
    vocab_size: int, 
    input_vocab: torch.Tensor, 
    padding_idx: int,
    d_model: int,
    dropout_rate: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1  通过 embedings 
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb = pe(result_emb)

    # 3. 生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    mask = make_padding_mask(input_vocab, padding_idx)

    # 4. query, key, value 使用带有特征维度的连续浮点张量 pe_emb
    query = key = value = result_pe_emb

    # 5. 把参数传给 attention
    attn, p_attn = attention(query, key, value, mask=mask, dropout=nn.Dropout(p=dropout_rate).eval())

    print(f"attn:\n{attn}")
    print("attn_shape:", attn.shape)
    print()
    print(f"p_attn:\n{p_attn}")
    print("p_attn_shape:", p_attn.shape)
    print('-' * 50)

def mutl_head_attn_test(
    vocab_size: int, 
    input_vocab: torch.Tensor, 
    padding_idx: int,
    head: int,
    d_model: int,
    dropout_rate: float
) -> torch.Tensor:
    # 1  通过 embedings 
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb = pe(result_emb)

    # 3  生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    mask = make_padding_mask(input_vocab, padding_idx)

    # 4  query, key, value 使用带有特征维度的连续浮点张量 pe_emb
    query = key = value = result_pe_emb   

    # 5  初始化 MultiHeadedAttention 对象 mult_head_attn
    mh_attn = MultiHeadedAttention(head=head, d_model=d_model, dropout_rate=dropout_rate)

    # 6  输出 dim=-1 维度的张量结果
    result_mha = mh_attn(query=query, key=key, value=value, mask=mask)

    print("result_mha:\n", result_mha)
    print("result_mha_shape:\n", result_mha.shape)
    print('-' * 50)

def feed_forward_test(    
    vocab_size: int, 
    input_vocab: torch.Tensor, 
    padding_idx: int,
    head: int,
    d_model: int,
    dropout_rate: float,
    d_ff: int
    ) -> torch.Tensor:

    # 1  通过 embedings 
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb = pe(result_emb)

    # 3  生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    mask = make_padding_mask(input_vocab, padding_idx)

    # 4  query, key, value 使用带有特征维度的连续浮点张量 pe_emb
    query = key = value = result_pe_emb   

    # 5  初始化 MultiHeadedAttention 对象 mult_head_attn
    mh_attn = MultiHeadedAttention(head=head, d_model=d_model, dropout_rate=dropout_rate)

    # 6  输出 dim=-1 维度的张量结果
    result_mha = mh_attn(query=query, key=key, value=value, mask=mask)

    # 7  new 一个 PositionwiseFeedForward 对象
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    # 8  将输出的多头注意力张量结果传给 ff_mha 对象
    result_ff_mha = ff(result_mha)

    print("result_ff_mha:\n", result_ff_mha)
    print("result_ff_mha_shape:\n", result_ff_mha.shape)
    print('-' * 50)

def layer_norm_test(  
    vocab_size: int,     
    input_vocab: torch.Tensor, 
    padding_idx: int,
    head: int,
    d_model: int,
    dropout_rate: float,
    ) -> torch.Tensor:

    # 1  通过 embedings 
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb = pe(result_emb)

    # 3  生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    mask = make_padding_mask(input_vocab, padding_idx)

    # 4  query, key, value 使用带有特征维度的连续浮点张量 pe_emb
    query = key = value = result_pe_emb   

    # 5  初始化 MultiHeadedAttention 对象 mult_head_attn
    mh_attn = MultiHeadedAttention(head=head, d_model=d_model, dropout_rate=dropout_rate)

    # 6  输出 dim=-1 维度的张量结果
    result_mha = mh_attn(query=query, key=key, value=value, mask=mask)

    # 7  new 一个 LN 层对象
    ln_mha = LayerNorm(features=d_model)

    # 8  经过 LN 层进行归一化
    ln_mha_result = ln_mha(result_mha)

    print("ln_mha_result:\n", ln_mha_result)
    print("ln_mha_result_shape:\n", ln_mha_result.shape)
    print('-' * 50)


def sublayer_connection_test(  
    vocab_size: int, 
    input_vocab: torch.Tensor, 
    padding_idx: int,
    head: int,
    d_model: int,
    dropout_rate: float,
    ) -> torch.Tensor:

    # 1  通过 embedings 
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb = pe(result_emb)

    # 3  生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    mask = make_padding_mask(input_vocab, padding_idx)

    # 5  初始化 MultiHeadedAttention 对象 mult_head_attn
    self_mha = MultiHeadedAttention(head=head, d_model=d_model, dropout_rate=dropout_rate)

    # 6  new 一个 sc 对象
    sc = SublayerConnection(size=d_model, dropout_rate=dropout_rate)

    # 7  调用 sc 得到残差连接后的结果
    sc_result = sc(result_pe_emb, lambda x: self_mha(x, x, x, mask=mask))

    print("sc_result:\n", sc_result)
    print("sc_result_shape:\n", sc_result.shape)
    print('-' * 50)

def encoder_block_test(
    vocab_size: int, 
    input_vocab: torch.Tensor, 
    padding_idx: int,
    head: int,
    d_model: int,
    dropout_rate: float,
    d_ff: int
) -> torch.Tensor:
    # 1  通过 embedings 
    emb = Embeddings(d_model, vocab_size, padding_idx)
    result_emb = emb(input_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb = pe(result_emb)

    # 3  生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    mask = make_padding_mask(input_vocab, padding_idx)

    # 4  初始化 MultiHeadedAttention 对象 mult_head_attn
    mh_attn = MultiHeadedAttention(head=head, d_model=d_model, dropout_rate=dropout_rate)

    # 5  new 一个 PositionwiseFeedForward 对象
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    # 9  new 一个 Encoder 对象并传入 mh_attn 和 ff_mha 对象参数
    encoder = Encoder(
        layer=EncoderLayer(
            size=d_model,
            self_attn=copy.deepcopy(mh_attn), 
            feed_forward=copy.deepcopy(ff), 
            dropout_rate=dropout_rate
        ), 
        N=N
    )

    result_encoder = encoder(x=result_pe_emb, mask=mask)

    print("result_encoder_block:\n", result_encoder)
    print("result_encoder_block_shape:\n", result_encoder.shape)
    print('-' * 50)

def decoder_block_test(
    src_vocab_size: int, 
    tgt_vocab_size: int,
    src_vocab: torch.Tensor, 
    tgt_vocab: torch.Tensor,
    padding_idx: int,
    head: int,
    d_model: int,
    dropout_rate: float,
    d_ff: int
) -> torch.Tensor:
    # 1  通过 embedings 
    emb_src = Embeddings(d_model, src_vocab_size, padding_idx=padding_idx)
    emb_tgt = Embeddings(d_model, tgt_vocab_size, padding_idx=padding_idx)
    result_emb_src = emb_src(src_vocab)
    result_emb_tgt = emb_tgt(tgt_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb_src = pe(result_emb_src)
    result_pe_emb_tgt = pe(result_emb_tgt)

    # 3  生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    src_padding_mask = make_padding_mask(src_vocab, padding_idx)
    tgt_padding_mask = make_padding_mask(tgt_vocab, padding_idx)
    tgt_causal_mask = make_causal_mask(tgt_vocab.size(1)) 
    tgt_mask = tgt_padding_mask & tgt_causal_mask

    # 4  初始化 MultiHeadedAttention 对象 mult_head_attn
    mh_attn = MultiHeadedAttention(head=head, d_model=d_model, dropout_rate=dropout_rate)

    # 5  new 一个 PositionwiseFeedForward 对象
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    # 6  new 一个 Encoder 对象并传入 mh_attn 和 ff_mha 对象参数
    encoder = Encoder(
        layer=EncoderLayer(
            size=d_model,
            self_attn=copy.deepcopy(mh_attn), 
            feed_forward=copy.deepcopy(ff), 
            dropout_rate=dropout_rate
        ), 
        N=N
    )

    result_encoder_block = encoder(x=result_pe_emb_src, mask=src_padding_mask)

    # 7  new 一个 Decoder 对象并传入
    decoder = Decoder(
        layer=DecoderLayer(
            size=d_model,
            self_attn=copy.deepcopy(mh_attn),
            src_attn=copy.deepcopy(mh_attn),
            feed_forward=copy.deepcopy(ff),
            dropout_rate=dropout_rate,
        ),
        N=N,
    )

    result_decoder_block = \
        decoder(
            x=result_pe_emb_tgt, 
            memory=result_encoder_block, 
            source_mask=src_padding_mask, 
            target_mask=tgt_mask
        )

    print("result_decoder_block:\n", result_decoder_block)
    print("result_decoder_block_shape:\n", result_decoder_block.shape)
    print('-' * 50)

def generator_test(
    src_vocab_size: int, 
    tgt_vocab_size: int,
    src_vocab: torch.Tensor, 
    tgt_vocab: torch.Tensor,
    padding_idx: int,
    head: int,
    d_model: int,
    dropout_rate: float,
    d_ff: int
) -> torch.Tensor:
    # 1  通过 embedings 
    emb_src = Embeddings(d_model, src_vocab_size, padding_idx=padding_idx)
    emb_tgt = Embeddings(d_model, tgt_vocab_size, padding_idx=padding_idx)
    result_emb_src = emb_src(src_vocab)
    result_emb_tgt = emb_tgt(tgt_vocab)

    # 2  加入旋转位置编码
    pe = PositionalEncoding(d_model=d_model)
    result_pe_emb_src = pe(result_emb_src)
    result_pe_emb_tgt = pe(result_emb_tgt)

    # 3  生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    src_padding_mask = make_padding_mask(src_vocab, padding_idx)
    tgt_padding_mask = make_padding_mask(tgt_vocab, padding_idx)
    tgt_causal_mask = make_causal_mask(tgt_vocab.size(1)) 
    tgt_mask = tgt_padding_mask & tgt_causal_mask

    # 4  初始化 MultiHeadedAttention 对象 mult_head_attn
    mh_attn = MultiHeadedAttention(head=head, d_model=d_model, dropout_rate=dropout_rate)

    # 5  new 一个 PositionwiseFeedForward 对象
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    # 6  new 一个 Encoder 对象并传入 mh_attn 和 ff_mha 对象参数
    encoder = Encoder(
        layer=EncoderLayer(
            size=d_model,
            self_attn=copy.deepcopy(mh_attn), 
            feed_forward=copy.deepcopy(ff), 
            dropout_rate=dropout_rate
        ), 
        N=N
    )

    result_encoder_block = encoder(x=result_pe_emb_src, mask=src_padding_mask)

    # 7  new 一个 Decoder 对象并传入
    decoder = Decoder(
        layer=DecoderLayer(
            size=d_model,
            self_attn=copy.deepcopy(mh_attn),
            src_attn=copy.deepcopy(mh_attn),
            feed_forward=copy.deepcopy(ff),
            dropout_rate=dropout_rate,
        ),
        N=N,
    )

    result_decoder_block = \
        decoder(
            x=result_pe_emb_tgt, 
            memory=result_encoder_block, 
            source_mask=src_padding_mask, 
            target_mask=tgt_mask
        )

    # 8  通过最后的 generator 层
    gen = Generator(vocab_size=tgt_vocab_size, d_model=d_model)
    gen_result = gen(result_decoder_block)
    print("gen_result:\n", gen_result)
    print("gen_result_shape:\n", gen_result.shape)
    print('-' * 50)


if __name__ == "__main__":
    # 超参数配置
    d_model = 512
    d_ff = 2048
    num_heads = 8
    vocab_size = 1000
    padding_idx = 0
    dropout_rate = 0.1
    N = 6

    # 输入长整形张量 x，形状为 2 x 4，模拟英文 
    x = torch.tensor([[999, 2, 423, 0], [491, 999, 1, 221]], dtype=torch.long, device=device)

    # 输入长整形张量 y，形状为 2 x 4，模拟法文 
    y = torch.tensor([[73, 215, 489, 562, 701, 938], [104, 327, 456, 618, 775, 892]], dtype=torch.long, device=device)

    # 1  测试 embeddings
    embeddings_test(input_vocab=x, d_model=d_model, vocab_size=vocab_size, padding_idx=padding_idx)    

    # 2  测试 positional_encoding
    positional_encoding_test(input_vocab=x, d_model=d_model, vocab_size=vocab_size, padding_idx=padding_idx)

    # 3  测试 mask

    mask_test(padding_idx=padding_idx)

    # 4  测试 attention
    attention_test(
        input_vocab=x, 
        d_model=d_model, 
        padding_idx=padding_idx, 
        dropout_rate=dropout_rate,
        vocab_size=vocab_size
    )

    # 5  测试 mult_head_attn
    mutl_head_attn_test(
        input_vocab=x, 
        head=num_heads, 
        d_model=d_model, 
        padding_idx=padding_idx, 
        dropout_rate=dropout_rate,
        vocab_size=vocab_size
    )

    # 6  测试 mult_head_attn
    feed_forward_test(
        input_vocab=x, 
        head=num_heads, 
        d_model=d_model, 
        padding_idx=padding_idx, 
        dropout_rate=dropout_rate,
        d_ff=d_ff,
        vocab_size=vocab_size
    )

    # 7  测试 layer_norm
    layer_norm_test(
        input_vocab=x, 
        head=num_heads, 
        d_model=d_model, 
        padding_idx=padding_idx, 
        dropout_rate=dropout_rate,
        vocab_size=vocab_size
    )

    # 8  测试 sublayer_connection
    sublayer_connection_test(
        input_vocab=x, 
        head=num_heads, 
        d_model=d_model, 
        padding_idx=padding_idx, 
        dropout_rate=dropout_rate,
        vocab_size=vocab_size
    )

    # 9  测试 encoder_test
    encoder_block_test(
        vocab_size=vocab_size, 
        input_vocab=x, 
        padding_idx=padding_idx,
        head=num_heads,
        d_model=d_model,
        dropout_rate=dropout_rate,
        d_ff=d_ff
    )

    # 10  测试 decoder_test
    decoder_block_test(
        src_vocab_size=vocab_size, 
        tgt_vocab_size=vocab_size,
        src_vocab=x, 
        tgt_vocab=y,
        padding_idx=padding_idx,
        head=num_heads,
        d_model=d_model,
        dropout_rate=dropout_rate,
        d_ff=d_ff
    )

    # 11  测试 generator_test
    generator_test(
        src_vocab_size=vocab_size, 
        tgt_vocab_size=vocab_size,
        src_vocab=x, 
        tgt_vocab=y,
        padding_idx=padding_idx,
        head=num_heads,
        d_model=d_model,
        dropout_rate=dropout_rate,
        d_ff=d_ff
    )


    