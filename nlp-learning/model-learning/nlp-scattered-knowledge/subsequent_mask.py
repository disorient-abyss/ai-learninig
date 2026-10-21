import matplotlib.pyplot as plt
import torch


def subsequent_mask(size, device=None, dtype=torch.long):
    """
    生成 Transformer 解码器使用的后续遮掩掩码 (输出 0 和 1)
    
    :param size: 掩码矩阵的边长 (序列长度)
    :param device: 指定设备 (如 'cuda' 或 'cpu')
    :param dtype: 数据类型，默认 torch.long (也可指定 torch.float32)
    :return: 形状为 (1, size, size) 的下三角掩码张量 (元素为 0 和 1)
    """
    # torch.tril 保留下三角部分的 1，其余右上角位置自动补 0
    return torch.tril(torch.ones((1, size, size), dtype=dtype, device=device))

if __name__ == "__main__":
    size = 5
    sm = subsequent_mask(size)
    print(f"sm (shape: {sm.shape}):\n{sm}")

    # 可视化 (绘制 20x20 的掩码)
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0].cpu().numpy())
    plt.title("Subsequent Mask (20x20)")
    plt.colorbar()
    plt.show()