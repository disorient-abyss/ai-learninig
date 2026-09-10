import torch

# ==========================================
# 核心：配置 GPU 设备
# ==========================================
# 如果有英伟达显卡，device 会变成 "cuda"，否则是 "cpu"
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
# 核心：设置全局默认设备
torch.set_default_device(device)
print(f"当前全局默认设备是: {device}")

def test():
    data = torch.full([2, 3], 10.)
    print(data.dtype)

    # 将data元素类型转换为float64类型

    # 1. 第一种方法
    print("第一种方法------------")
    data = data.type("torch.DoubleTensor")
    print(data.dtype)

    # 转换为其他类型
    # data = data.type(torch.ShortTensor)
    # data = data.type(torch.IntTensor)
    # data = data.type(torch.LongTensor)
    # data = data.type(torch.FloatTensor)

    # 2.第二种方法
    print("第二种方法-------------")
    data = data.to(torch.float64)
    print(data.dtype)
    data = data.to(torch.float32)
    print(data.dtype)
    data = data.to(torch.float16)
    print(data.dtype)
    data = data.to(torch.int64)
    print(data.dtype)
    data = data.to(torch.int32)
    print(data.dtype)
    data = data.to(torch.int16)
    print(data.dtype)
    data = data.to(torch.uint8)
    print(data.dtype)
    data = data.to(torch.bool)
    print(data.dtype)

    # 3. 第三种方法(为第二种方法的简写)
    print("第三种方法------------")
    data = data.double()
    print(data.dtype)

    # 转换为其他类型
    # data = data.short()
    # data = data.int()
    # data = data.long()
    # data = data.float()


if __name__ == "__main__":
    test()
