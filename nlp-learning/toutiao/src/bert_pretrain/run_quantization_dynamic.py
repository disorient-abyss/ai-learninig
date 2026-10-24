import argparse
import os
from importlib import import_module

import torch


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
        description="Chinese Text Classification - Dynamic Quantization"
    )
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to perform quantization (e.g., cuda, cpu, cuda:0)",
    )
    args = parser.parse_args()

    model_name = args.model.strip().lower()
    try:
        model_module = import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        raise ValueError(f"未找到对应的模型文件: models/{model_name}.py")

    config = model_module.Config(args.dataset)

    # 1. 规范化设备参数并获取量化方案
    device_str = "cuda" if args.device.lower() == "gpu" else args.device
    requested_device = torch.device(device_str)

    actual_device, quant_config, quantize_fn, save_suffix = resolve_quant_plan(requested_device)
    config.device = actual_device

    # 2. 检查原始训练权重是否存在
    if not os.path.exists(config.save_path):
        raise FileNotFoundError(
            f"未找到原始模型权重: {config.save_path}，请先执行 run_train.py"
        )

    # 3. 确定最终量化产物保存路径
    save_path_quant = os.path.join(config.save_dir, f"{model_name}{save_suffix}")

    # 4. 加载原始浮点模型至最终校正后的设备
    print(f"Loading float model weights from [{config.save_path}]...")
    model = model_module.Model(config)
    model.load_state_dict(torch.load(config.save_path, map_location=actual_device))
    model.to(actual_device)
    model.eval()

    # 5. 执行原地量化
    print(f"Applying quantization using {quant_config.__class__.__name__} on {actual_device}...")
    quantize_fn(model, quant_config)

    # 6. 保存量化权重
    torch.save(model.state_dict(), save_path_quant)
    print(f"Quantized model successfully saved to: {save_path_quant}")


if __name__ == "__main__":
    main()