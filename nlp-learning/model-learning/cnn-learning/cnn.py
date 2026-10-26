import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def show(img, title=None):
    """通用图像展示函数，支持 Tensor / NumPy 格式的灰度图与彩色图"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze().numpy()
    elif isinstance(img, np.ndarray):
        img = img.squeeze()

    plt.figure()
    if img.ndim == 2:
        # 单通道灰度图
        plt.imshow(img, cmap="gray")
    else:
        # 多通道彩色图 (H, W, C)
        plt.imshow(np.clip(img, 0.0, 1.0))

    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def load_and_preprocess_rgba(image_path, alpha_value=0.5):
    """
    读取 JPG 图片并补齐 Alpha 通道，输出形状为 [1, 4, H, W] 的浮点张量
    """
    # 1. 读取并归一化到 [0.0, 1.0]
    img = plt.imread(image_path).astype(np.float32) / 255.0

    # 2. 拼接 Alpha 通道: (H, W, 3) + (H, W, 1) -> (H, W, 4)
    h, w = img.shape[:2]
    alpha = np.full((h, w, 1), fill_value=alpha_value, dtype=np.float32)
    img_rgba = np.concatenate([img, alpha], axis=-1)

    # 3. 维度重排与扩充 Batch: (H, W, 4) -> (4, H, W) -> (1, 4, H, W)
    img_tensor = torch.tensor(img_rgba).permute(2, 0, 1).unsqueeze(0)
    return img_rgba, img_tensor


# 1. 单个多通道卷积核 (融合为单通道)
def test01():
    raw_img, img_tensor = load_and_preprocess_rgba("images/1.jpg")
    show(raw_img, title="Original RGBA")

    # 定义卷积层: 4通道 -> 1通道
    conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)

    with torch.no_grad():
        out = conv(img_tensor)  # 输出形状: [1, 1, H, W]

    show(out, title="Single Kernel Feature Map")


# 2. 多个多通道卷积核 (提取3通道特征)
def test02():
    _, img_tensor = load_and_preprocess_rgba("images/1.jpg")

    # 定义卷积层: 4通道 -> 3通道
    conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)

    with torch.no_grad():
        out = conv(img_tensor)  # 输出形状: [1, 3, H, W]

    # 分别展示 3 个卷积核提取出的特征图
    for c in range(out.shape[1]):
        show(out[0, c], title=f"Kernel {c} Feature Map")


# 3. 池化基础操作 (最大池化 vs 平均池化)
def test03():
    # 生成 1x1x3x3 矩阵
    inputs = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)

    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

    print("=== Test 03: 基础池化 ===")
    print("最大池化:\n", max_pool(inputs))
    print("平均池化:\n", avg_pool(inputs))


# 4. 步长 (Stride) 对池化输出尺寸的影响
def test04():
    # 生成 1x1x4x4 矩阵
    inputs = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    print("\n=== Test 04: Stride=2 池化 ===")
    print("最大池化:\n", max_pool(inputs))
    print("平均池化:\n", avg_pool(inputs))


# 5. 填充 (Padding) 对池化输出尺寸的影响
def test05():
    inputs = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)

    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)

    print("\n=== Test 05: Padding=1 池化 ===")
    print("最大池化:\n", max_pool(inputs))
    print("平均池化:\n", avg_pool(inputs))


# 6. 多通道独立池化计算
def test06():
    # 生成 1x3x3x3 张量 (3个通道分别独立池化，不跨通道融合)
    c1 = torch.arange(0, 9).reshape(3, 3)
    c2 = torch.arange(10, 100, 10).reshape(3, 3)
    c3 = torch.arange(11, 110, 11).reshape(3, 3)
    inputs = torch.stack([c1, c2, c3]).unsqueeze(0).float()

    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

    print("\n=== Test 06: 多通道池化 ===")
    print("多通道最大池化输出:\n", max_pool(inputs))


if __name__ == "__main__":
    # 执行测试
    test01()
    test02()
    test03()
    test04()
    test05()
    test06()