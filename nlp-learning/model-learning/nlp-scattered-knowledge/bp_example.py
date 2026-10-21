import torch
from torch import nn, optim

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前正在使用的设备: {device}")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

        # 【优化】网络参数初始化：采用 copy_ 方式
        # 这样不管外面 net 怎么 .to(device)，自定义的初始值都能完美同步过去
        with torch.no_grad():
            self.linear1.weight.copy_(torch.tensor([[0.15, 0.20], [0.25, 0.30]]))
            self.linear2.weight.copy_(torch.tensor([[0.40, 0.45], [0.50, 0.55]]))
            self.linear1.bias.copy_(torch.tensor([0.35, 0.35]))
            self.linear2.bias.copy_(torch.tensor([0.70, 0.80]))

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

# 实例化网络并移至目标设备
net = Net().to(device)

if __name__ == '__main__':
    # 【修复】将数据同样挪到对应的设备（GPU 或 CPU）
    inputs = torch.tensor([[0.05, 0.10]]).to(device)
    target = torch.tensor([[0.01, 0.99]]).to(device)

    # 获得网络输出值
    output = net(inputs)
    print("网络输出值:", output)

    # 计算误差
    loss = torch.sum((output - target) ** 2) / 2
    print("当前 Loss:", loss)

    # 优化方法
    optimizer = optim.SGD(net.parameters(), lr=0.5)

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播
    loss.backward() 
    # 【修复】修正严格的 None 检查逻辑，并去掉了不推荐使用的 .data
    if net.linear1.weight.grad is not None and net.linear2.weight.grad is not None:
        # 打印 w5、w7、w1 的梯度值
        print("Linear1 梯度:\n", net.linear1.weight.grad)
        print("Linear2 梯度:\n", net.linear2.weight.grad)

        # 更新参数
        optimizer.step()
        print("更新后的网络参数:\n", net.state_dict())