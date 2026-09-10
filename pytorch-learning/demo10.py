import torch


def test01():

    data1 = torch.randint(0, 10, [3, 5, 4])
    data2 = torch.randint(0, 10, [3, 5, 4])

    print(data1)
    print(data2)
    print("-" * 50)

    # 1. 按0维度拼接
    new_data = torch.cat([data1, data2], dim=0)
    print(new_data)
    print(new_data.shape)
    print("-" * 50)

    # 2. 按1维度拼接
    new_data = torch.cat([data1, data2], dim=1)
    print(new_data)
    print(new_data.shape)
    print("-" * 50)

    # 3. 按2维度拼接
    new_data = torch.cat([data1, data2], dim=2)
    print(new_data)
    print(new_data.shape)

def test02():
    data1 = torch.randint(0, 10, [2, 3, 4])
    data2 = torch.randint(0, 10, [2, 3, 4])
    # data2= torch.randint(0, 10, [2, 3, 4])
    print(data1)
    print(data2)
    print('-' * 50)
    new_data1 = torch.stack([data1, data2], dim=0)
    print(new_data1)
    print(new_data1.shape)

    new_data2 = torch.cat([data1, data2], dim=0)
    print(new_data2)
    print(new_data2.shape)
    '''
    print('-' * 50)

    new_data2 = torch.stack([data1, data2], dim=1)
    print(new_data2)
    print(new_data2.shape)

    new_data3 = torch.stack([data1, data2], dim=2)
    print(new_data3)
    print(new_data3.shape)

    print(new_data1.shape)
    print(new_data2.shape)
    print(new_data3.shape)
    print(new_data1)
    print(new_data2)
    print(new_data3)
    print('-' * 50)
    '''
    '''
    x = torch.Tensor([1])
    y = torch.Tensor([2])
    z = torch.stack([x, y], dim=0)
    print(z)


    new_data = torch.stack([data1, data2], dim=1)
    print(new_data.shape)
    print(new_data)
    print('---------------------------------')

    new_data = torch.stack([data1, data2], dim=2)
    print(new_data.shape)
    print(new_data)
    '''

if __name__ == "__main__":
    # test01()
    test02()

