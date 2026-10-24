import argparse
import os
import random
from importlib import import_module

import numpy as np
import torch
from utils.build_dataset import build_dataset
from utils.build_iterator import build_iterator
from utils.train_eval import test


def set_seed(seed: int = 1) -> None:
    """锁定随机种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_quant_plan(requested_device: torch.device):
    """
    根据请求设备与底层硬件架构，自动决策最终运行设备与量化方案。
    返回值: (actual_device, quant_config, quantize_fn, save_suffix)
    """
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA 设备，但当前环境无可用的 CUDA GPU！")

        major, minor = torch.cuda.get_device_capability(requested_device)
        gpu_name = torch.cuda.get_device_name(requested_device)

        # 具备原生 FP8 Tensor Core 硬件支持 (Compute Capability >= 8.9)
        if (major, minor) >= (8, 9):
            print(f"[Device] 使用 GPU: {gpu_name} (CC {major}.{minor}) -> 启用 GPU + FP8 动态量化")
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                quantize_,
            )

            return (
                requested_device,
                Float8DynamicActivationFloat8WeightConfig(),
                quantize_,
                "_fp8.pt",
            )
        else:
            print(
                f"[Device] GPU: {gpu_name} (CC {major}.{minor}) 不支持原生 FP8 (需 CC>=8.9)。"
                "自动回退至 CPU + INT8 动态量化。"
            )
            from torchao.quantization import (
                Int8DynamicActivationInt8WeightConfig,
                quantize_,
            )

            return (
                torch.device("cpu"),
                Int8DynamicActivationInt8WeightConfig(),
                quantize_,
                "_int8.pt",
            )
    else:
        print("[Device] 使用 CPU -> 启用 CPU + INT8 动态量化")
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )

        return (
            torch.device("cpu"),
            Int8DynamicActivationInt8WeightConfig(),
            quantize_,
            "_int8.pt",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Chinese Text Classification - Test"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Choose model name (e.g., bert)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="toutiao",
        help="Choose dataset name (default: toutiao)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for evaluation (e.g., cuda, cpu, cuda:0)",
    )
    parser.add_argument(
        "--use_quantized",
        action="store_true",
        help="Whether to test the quantized model",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model_name = args.model.strip().lower()
    try:
        model_module = import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        raise ValueError(f"未找到对应的模型文件: models/{model_name}.py")

    config = model_module.Config(args.dataset)

    # 规范化用户传入的设备参数
    device_str = "cuda" if args.device.lower() == "gpu" else args.device
    requested_device = torch.device(device_str)

    # 1. 确定最终执行设备与量化方案
    if args.use_quantized:
        config.device, quant_config, quantize_fn, save_suffix = resolve_quant_plan(requested_device)
    else:
        config.device = requested_device

    # 2. 仅加载测试集数据并构建测试迭代器（确保在 config.device 最终敲定后再构建）
    print(
        f"Loading test dataset for [{model_name.upper()}] on [{args.dataset}]..."
    )
    _, _, test_data = build_dataset(config, only_test=True)
    test_iter = build_iterator(test_data, config)

    # 3. 初始化模型结构与载入权重
    model = model_module.Model(config)

    if args.use_quantized:
        weight_path = os.path.join(config.save_dir, f"{model_name}{save_suffix}")
        
        # 兜底回退：若不存在专用后缀，尝试读取旧的 bert_quant.pt
        if not os.path.exists(weight_path):
            legacy_path = os.path.join(config.save_dir, f"{model_name}_quant.pt")
            if os.path.exists(legacy_path):
                weight_path = legacy_path

        print(f"Mode: Testing QUANTIZED model from [{weight_path}]")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"未找到量化权重文件: {weight_path}，请先执行 run_quantization_dynamic.py"
            )

        # 原地量化替换并加载权重
        quantize_fn(model, quant_config)
        model.load_state_dict(torch.load(weight_path, map_location=config.device))
    else:
        weight_path = config.save_path
        print(f"Mode: Testing ORIGINAL float model from [{weight_path}]")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"未找到浮点权重文件: {weight_path}，请先执行 run_train.py"
            )

        model.load_state_dict(torch.load(weight_path, map_location=config.device))

    model.to(config.device)
    model.eval()

    # 4. 执行测试评估
    print(f"Running evaluation on device: {config.device}...")
    test(config, model, test_iter)


if __name__ == "__main__":
    main()