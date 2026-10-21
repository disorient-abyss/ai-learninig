import torch
from PhonePriceModel import PhonePriceModel
from price_classify import create_dataset
from torch.utils.data import DataLoader


def test():
    # 1. 检查并分配设备（保持与训练一致）
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 2. 调用函数获取验证集与维度信息
    _, valid_dataset, input_dim, class_num = create_dataset()

    # 3. 初始化模型并送到指定设备
    model = PhonePriceModel(input_dim, class_num).to(device)

    # 4. 加载模型参数（将权重映射到当前设备）
    state_dict = torch.load('model/phone-price-model.bin', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 5. 切换模型为评估模式
    model.eval()

    # 6. 构建验证集的数据加载器（测试集不需要打乱 shuffle=False）
    dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    correct = 0

    # 7. 禁用梯度计算，加速预测并节省显存
    with torch.no_grad():
        for x, y in dataloader:
            # 数据送入设备
            x = x.to(device)
            y = y.to(device)

            # 前向传播预测
            output = model(x)

            # 获取预测结果概率最大的类别索引 (沿着维度 1 找最大值)
            y_pred = torch.argmax(output, dim=1)

            # 统计预测正确的样本数
            correct += (y_pred == y).sum().item()

    # 8. 计算并打印准确率
    accuracy = correct / len(valid_dataset)
    print(f'Validation Accuracy: {accuracy:.4%}')


if __name__ == '__main__':
    test()