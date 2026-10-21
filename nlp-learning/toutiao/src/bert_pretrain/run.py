import argparse
import os
import random
from importlib import import_module

import numpy as np
import torch
from models.train_eval import test, train
from utils.build_dataset import build_dataset
from utils.build_iterator import build_iterator


def set_seed(seed: int = 1) -> None:
    """锁定所有随机种子，确保实验完全可复现"""
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
    parser = argparse.ArgumentParser(description="Chinese Text Classification")
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
    args = parser.parse_args()

    # 1. 固定全局随机种子
    set_seed(args.seed)

    # 2. 动态加载模型与配置模块
    model_name = args.model.strip().lower()
    try:
        model_module = import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        raise ValueError(f"未找到对应的模型文件: models/{model_name}.py")

    config = model_module.Config(args.dataset)

    # 3. 构建数据集与批次迭代器
    print(f"Loading data for [{model_name.upper()}] on [{args.dataset}] dataset...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    # 4. 初始化模型并迁移至目标设备 (GPU / CPU)
    model = model_module.Model(config).to(config.device)

    # 5. 启动训练与最终测试
    train(config, model, train_iter, dev_iter)
    test(config, model, test_iter)


if __name__ == "__main__":
    main()