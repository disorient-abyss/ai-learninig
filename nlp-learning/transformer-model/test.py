import torch
from trasformer_model import make_model
from utils import make_pad_mask, make_target_mask

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # 1. 超参数配置
    d_model = 512
    d_ff = 2048
    num_heads = 8
    vocab_size = 1000
    pad_idx = 0
    dropout_rate = 0.1
    N = 6

    # 2. 构造测试数据与掩码 (batch_size=2, seq_len=4)
    src_tokens = torch.tensor(
        [[100, 2, 421, pad_idx], [491, 999, 1, 221]],
        dtype=torch.long,
        device=device,
    )
    tgt_tokens = torch.tensor(
        [[100, 45, 88, pad_idx], [300, 20, 11, 99]],
        dtype=torch.long,
        device=device,
    )

    src_mask = make_pad_mask(src_tokens, pad_idx=pad_idx)
    tgt_mask = make_target_mask(tgt_tokens, pad_idx=pad_idx)

    # 3. 一键初始化完整模型并迁移到设备
    model = make_model(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        N=N,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        dropout=dropout_rate,
        pad_idx=pad_idx,
    ).to(device)

    # 设置为评估模式（关闭 Dropout 随机失活）
    model.eval()

    # 4. 全链路端到端前向推理
    with torch.no_grad():
        log_probs = model(
            src=src_tokens,
            tgt=tgt_tokens,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

    print(f"Generator 输出对数概率分布形状: {log_probs.shape}")
    assert log_probs.shape == (2, 4, vocab_size), "模型输出维度异常！"

    # 5. 贪婪解码与结果校验
    pred_tokens = torch.argmax(log_probs, dim=-1)
    print(f"最终预测出的 Token ID 矩阵形状: {pred_tokens.shape}")
    print(f"预测出的 Token ID 结果:\n{pred_tokens}")

    assert not torch.isnan(log_probs).any(), "输出包含 NaN 异常值，请检查网络！"
    print("\n" + "=" * 30 + " [全链路封装测试成功] " + "=" * 30)