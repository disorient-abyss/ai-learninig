import argparse
import os
import random
from importlib import import_module

import numpy as np
import torch
from utils.build_dataset import build_dataset
from utils.build_iterator import build_iterator
from utils.train_eval import test, train


def set_seed(seed: int = 1) -> None:
    """锁定随机种子以保证训练可复现"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chinese Text Classification - Train")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Choose a model name (e.g., bert, ernie, bert_cnn)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="toutiao",
        help="Choose a dataset name (default: toutiao)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device (e.g., cuda, cuda:0, cpu)",
    )
    args = parser.parse_args()

    # 1. 固定随机种子
    set_seed(args.seed)

    # 2. 动态加载模型与配置模块
    model_name = args.model.strip().lower()
    try:
        model_module = import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        raise ValueError(f"未找到对应的模型文件: models/{model_name}.py")

    config = model_module.Config(args.dataset)

    # 规范化设备参数：兼容用户输入 'gpu' 的情况
    device_str = "cuda" if args.device.lower() == "gpu" else args.device
    config.device = torch.device(device_str)

    # 3. 构建训练集、验证集和测试集
    print(f"Loading data for [{model_name.upper()}] on [{args.dataset}] dataset...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    # 4. 初始化模型并迁移至目标设备
    model = model_module.Model(config).to(config.device)

    # 5. 启动训练与验证
    print(f"Starting training on device: {config.device}...")
    train(config, model, train_iter, dev_iter)

    # 6. 加载训练产出的最佳权重进行最终基准测试（对应 bert.pt）
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
    test(config, model, test_iter)


if __name__ == "__main__":
    main()