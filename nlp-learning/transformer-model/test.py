import torch
from decoder import Decoder, DecoderLayer
from embedding import TransformerEmbedding
from encoder import Encoder, EncoderLayer
from forward_net import PositionwiseFeedForward
from generator import Generator
from multi_head_attn import MultiHeadedAttention
from utils import make_pad_mask, make_target_mask

if __name__ == "__main__":
    # =========================================================================
    # 0. 全局环境与超参数配置
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    d_model = 512       # 词向量及所有子层特征维度
    d_ff = 2048         # 前馈网络中间隐层升维维度 (4 * d_model)
    num_heads = 8       # 多头注意力的头数
    vocab_size = 1000   # 词表总大小
    pad_idx = 0         # 填充 Token ID
    dropout_rate = 0.1  # Dropout 概率
    N = 6               # 编码器与解码器堆叠层数

    # =========================================================================
    # 1. ENCODER 区域 (源语言端：提取全文双向特征)
    # =========================================================================
    print("\n" + "=" * 30 + " [1. ENCODER 区域] " + "=" * 30)

    # 1.1 构造源端输入序列与 Padding 掩码 (batch_size=2, seq_len=4)
    src_tokens = torch.tensor(
        [[100, 2, 421, pad_idx], [491, 999, 1, 221]],
        dtype=torch.long,
        device=device,
    )
    # 显式传入 pad_idx；src_tokens 已经在 device 上，无需额外 .to(device)
    src_mask = make_pad_mask(src_tokens, pad_idx=pad_idx)
    print(f"源端原始 Token 形状: {src_tokens.shape}")

    # 1.2 源端词嵌入与位置编码
    src_embedding = TransformerEmbedding(
        d_model=d_model,
        vocab_size=vocab_size,
        dropout=dropout_rate,
        padding_idx=pad_idx,
    ).to(device)
    src_emb_output = src_embedding(src_tokens)
    print(f"源端 Embedding 输出形状: {src_emb_output.shape}")

    # 1.3 实例化 EncoderLayer 模板并堆叠 N 层
    encoder_layer_template = EncoderLayer(
        size=d_model,
        self_attn=MultiHeadedAttention(head=num_heads, d_model=d_model, dropout=dropout_rate),
        feed_forward=PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout_rate),
        dropout=dropout_rate,
    ).to(device)
    encoder = Encoder(layer=encoder_layer_template, N=N).to(device)

    # 1.4 运行 Encoder，生成供 Decoder 交叉检索的 memory
    encoder_memory = encoder(src_emb_output, mask=src_mask)
    print(f"Encoder 最终输出 (Memory) 形状: {encoder_memory.shape}")
    assert encoder_memory.shape == (2, 4, d_model), "Encoder 输出维度异常！"

    # =========================================================================
    # 2. DECODER 区域 (目标语言端：自回归生成并结合源端记忆)
    # =========================================================================
    print("\n" + "=" * 30 + " [2. DECODER 区域] " + "=" * 30)

    # 2.1 构造目标端输入序列 (batch_size=2, tgt_len=4)
    tgt_tokens = torch.tensor(
        [[100, 45, 88, pad_idx], [300, 20, 11, 99]],
        dtype=torch.long,
        device=device,
    )
    print(f"目标端原始 Token 形状: {tgt_tokens.shape}")

    # 2.2 构造复合 Target Mask (Padding 掩码 + 因果下三角掩码)
    tgt_mask = make_target_mask(tgt_tokens, pad_idx=pad_idx)

    # 2.3 目标端词嵌入与位置编码
    tgt_embedding = TransformerEmbedding(
        d_model=d_model,
        vocab_size=vocab_size,
        dropout=dropout_rate,
        padding_idx=pad_idx,
    ).to(device)
    tgt_emb_output = tgt_embedding(tgt_tokens)
    print(f"目标端 Embedding 输出形状: {tgt_emb_output.shape}")

    # 2.4 实例化 DecoderLayer 模板并堆叠 N 层
    decoder_layer_template = DecoderLayer(
        size=d_model,
        self_attn=MultiHeadedAttention(head=num_heads, d_model=d_model, dropout=dropout_rate),
        src_attn=MultiHeadedAttention(head=num_heads, d_model=d_model, dropout=dropout_rate),
        feed_forward=PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout_rate),
        dropout=dropout_rate,
    ).to(device)
    decoder = Decoder(layer=decoder_layer_template, N=N).to(device)

    # 2.5 运行 Decoder
    decoder_output = decoder(
        x=tgt_emb_output,
        memory=encoder_memory,
        source_mask=src_mask,
        target_mask=tgt_mask,
    )
    print(f"Decoder 最终输出形状: {decoder_output.shape}")
    assert decoder_output.shape == (2, 4, d_model), "Decoder 输出维度异常！"

    # =========================================================================
    # 3. GENERATOR 区域 (分类头：隐层向量映射至词表，输出对数概率或预测词)
    # =========================================================================
    print("\n" + "=" * 30 + " [3. GENERATOR 区域] " + "=" * 30)

    # 3.1 实例化 Generator (输入维度 d_model，输出维度 vocab_size) 并移至设备
    generator = Generator(d_model=d_model, vocab_size=vocab_size).to(device)

    # 3.2 前向传播：将 Decoder 输出 (2, 4, 512) 投影为词表对数概率 (2, 4, 1000)
    log_probs = generator(decoder_output)
    print(f"Generator 输出对数概率分布形状: {log_probs.shape}")
    assert log_probs.shape == (2, 4, vocab_size), "Generator 输出维度异常！"

    # 3.3 贪婪解码测试：通过 argmax 沿词表维度取出概率最大的 Token ID
    pred_tokens = torch.argmax(log_probs, dim=-1)
    print(f"最终预测出的 Token ID 矩阵形状: {pred_tokens.shape}")
    print(f"预测出的 Token ID 结果:\n{pred_tokens}")

    # =========================================================================
    # 4. 最终全链路校验
    # =========================================================================
    assert not torch.isnan(log_probs).any(), "输出包含 NaN 异常值，请检查网络！"
    print("\n" + "=" * 30 + " [全链路测试成功] " + "=" * 30)
    print("Transformer: Embedding -> Encoder -> Decoder -> Generator 完整流水线全部打通！")