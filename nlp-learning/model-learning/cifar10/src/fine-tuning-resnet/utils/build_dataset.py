from torchvision import datasets, transforms


def build_dataset(config, only_test=False):
    # 1. 图像预处理规范
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
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

    # 3. 训练集增强与构建
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
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