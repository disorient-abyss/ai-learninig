from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent



# 1 图像基本理解
def test01():
    img = np.zeros([200, 200])
    print(img)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()

    img = np.full([255, 255], 255)
    print(img)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()

def test02():
    img_path = current_dir / "images" / "1.jpg"

    # 1. 读取图片并解除只读限制 (.copy())，同时归一化为 0~1 的 float 类型
    img = plt.imread(img_path).copy().astype(np.float32) / 255.0

    # 2. 修改维度: [H, W, C] -> [C, H, W]
    img = np.transpose(img, [2, 0, 1])

    # 3. 打印并展示各个通道
    for channel in img:
        print("通道形状:", channel.shape)
        plt.imshow(channel, cmap="gray")
        plt.show()

    # 4. 创建一个形状为 (1, H, W) 的 Alpha 透明度通道（例如 0.5 表示半透明）
    h, w = img.shape[1], img.shape[2]
    alpha = np.full((1, h, w), fill_value=0.5, dtype=np.float32)

    # 5. 拼接到通道维度 (axis=0)，组合为 RGBA 4 通道 [4, H, W]
    img = np.concatenate([img, alpha], axis=0)

    # 修改透明度
    img[3] = 0.25
    img = np.transpose(img, [1, 2, 0])
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    # test01()
    test02()