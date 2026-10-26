from torchvision import datasets, transforms


def build_dataset(config, only_test=False):
    # 1. 测试集预处理规范（保持原生 32x32 分辨率）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    # 2. 如果仅测试，直接构建并返回测试集
    if only_test:
        test_dataset = datasets.CIFAR10(
            root=str(config.data_path),
            train=False,
            download=False,
            transform=test_transform,
        )
        return None, test_dataset

    # 3. 训练集数据增强与构建
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=str(config.data_path),
        train=True,
        download=False,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(config.data_path),
        train=False,
        download=False,
        transform=test_transform,
    )

    return train_dataset, test_dataset