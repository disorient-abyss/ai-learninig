import matplotlib.pyplot as plt
import torch


def test():
    _, axes = plt.subplots(1, 2)

    # 函数图像
    x = torch.linspace(-20, 20, 500)
    y = torch.tanh(x)
    axes[0].plot(x, y)
    axes[0].grid()
    axes[0].set_title("Sigmoid Function Picture")
  
    x = torch.linspace(-20, 20, 500, requires_grad=True)
    # 反响传播需要把 x 标量化
    y = torch.tanh(x).sum().backward()

    axes[1].plot(x.detach(), x.grad)
    axes[1].grid()
    axes[1].set_title("Sigmoid Function derivative")

    plt.show()

if __name__ == "__main__":
    test()