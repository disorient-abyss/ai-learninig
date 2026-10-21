import matplotlib.pyplot as plt  # noqa: EXE002
import torch
import torch.nn.functional as F


def test():
    _, axes = plt.subplots(1, 2)

    # 函数图像
    x = torch.linspace(-20, 20, 1000)
    y = F.tanh(x)
    axes[0].plot(x, y)
    axes[0].grid()
    axes[0].set_title("Sigmoid Function Picture")

    # 导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    # pytorch中使用反向传播（求导）一定要在标量上用，所以这里添加了sum()
    torch.sigmoid(x).sum().backward()

    axes[1].plot(x.detach(), x.grad)
    axes[1].grid()
    axes[1].set_title("Sigmoid Function derivative")

    plt.show()


if __name__ == "__main__":
    test()
