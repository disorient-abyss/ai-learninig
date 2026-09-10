import torch


# 1. 使用 cuda 方法
def test01():
    data = torch.tensor([10, 20, 30])
    print("存储设备:", data.device)

    # 如果安装的不是 gpu 版本的 PyTorch
    # 或电脑本身没有 NVIDIA 卡的计算环境
    # 下面代码可能会报错

    data = data.cuda()
    print("存储设备:", data.device)

    # 使用 cpu 函数将张量移动到 cpu 上
    data = data.cpu()
    print("存储设备:", data.device)

    # 输出结果:
    # 存储设备: cpu
    # 存储设备: cuda:0
    # 存储设备: cpu


# 2. 直接将张量创建在 GPU 上
def test02():
    data = torch.tensor([10, 20, 30], device="cuda")
    # data = torch.tensor([10, 20, 30], device='cpu')
    print("存储设备:", data.device)
    # 使用 cpu 函数将张量移动到 cpu 上
    data = data.cpu()
    print("存储设备:", data.device)

# 3. 使用 to 的方法
def test03():
    data = torch.tensor([10, 20, 30])
    data = data.to("cuda:0")    # 这里可以填具体放入的 gpu 设备位置，比如 cuda:0, cuda:1 ...
    print("存储设备:", data.device)
    data = data.to("cpu")
    print("存储设备:", data.device)


def test04():
    data1 = torch.tensor([10, 20, 30], device="cuda:0")
    # data1 = torch.tensor([10, 20, 30], device='cpu')
    data2 = torch.tensor([10, 20, 30])
    print(data1.device, data2.device)
    # RuntimeError: Expected all tensors to be on the same device,
    # but found at least two devices, cuda:0 and cpu!
    data = data1 + data2
    # data2 = torch.tensor([10, 20, 30], device='cuda:0')
    # data = data1 + data2
    print(data)


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    test04()
