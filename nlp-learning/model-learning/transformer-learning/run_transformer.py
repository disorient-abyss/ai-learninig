import torch
from make_model import make_model
from utils.subsequent_mask import make_causal_mask, make_padding_mask

# 1. 自动判断设备并设为全局默认
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


if __name__ == "__main__":
    # 超参数配置
    d_model = 512
    d_ff = 2048
    num_heads = 8
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    padding_idx = 0
    dropout_rate = 0.1
    N = 6

    transformer_model=make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        N = N,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        padding_idx=padding_idx,
        dropout_rate=dropout_rate
    ).eval()

    # 输入长整形张量 x，形状为 2 x 4，模拟英文 
    x = torch.tensor([[999, 2, 423, 0], [491, 999, 1, 221]], dtype=torch.long, device=device)

    # 输入长整形张量 y，形状为 2 x 4，模拟法文 
    y = torch.tensor([[73, 215, 489, 562, 701, 938], [104, 327, 456, 618, 775, 892]], dtype=torch.long, device=device)
    # 令
    src_vocab=x
    tgt_vocab=y
    # 生成 mask（注意：传入 input_vocab 而不是 pe_emb）
    src_padding_mask = make_padding_mask(vocab=src_vocab, padding_idx=padding_idx)
    tgt_padding_mask = make_padding_mask(vocab=tgt_vocab, padding_idx=padding_idx)
    tgt_causal_mask = make_causal_mask(tgt_vocab.size(1)) 
    tgt_mask = tgt_padding_mask & tgt_causal_mask

    transformer_result=transformer_model(
        src=x,
        tgt=y,
        src_mask=src_padding_mask,
        tgt_mask=tgt_mask,
    )

    print("transformer_result:\n", transformer_result)
    print("transformer_result_shape:\n", transformer_result.shape)
    print('=' * 50)