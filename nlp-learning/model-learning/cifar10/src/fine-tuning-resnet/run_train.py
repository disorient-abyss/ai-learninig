# coding: UTF-8
import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from utils.build_dataset import build_dataset
from utils.train_eval import train


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classification")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet_34",
        choices=["resnet_34", "resnet_18", "resnet-34", "resnet-18"],
        help="选择训练的模型架构 (默认: resnet_34)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model.replace("-", "_")

    try:
        model_module = importlib.import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"未在 models 目录下找到对应的模型文件: models/{model_name}.py"
        )

    Config = model_module.Config
    Model = model_module.Model

    config = Config()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {config.device}")
    print(f"Model selected: {config.model_name}")
    print(f"Batch size: {config.batch_size} | Epochs: {config.num_epochs}")

    # 1. 拆分并获取训练集、验证集、测试集
    train_dataset, dev_dataset, test_dataset = build_dataset(config)

    # 2. 构造数据迭代器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}, Test samples: {len(test_dataset)}")

    print("Initializing model...")
    model = Model(config).to(config.device)

    print("Start training...")
    # 传入 dev_loader 进行每 200 batch 评估，同时传入 test_loader 实现训练后自动测试
    train(config, model, train_loader, dev_loader, test_loader)


if __name__ == "__main__":
    main()