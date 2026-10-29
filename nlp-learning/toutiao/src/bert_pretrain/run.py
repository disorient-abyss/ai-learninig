# coding: UTF-8
"""统一执行入口脚本：整合训练 (Train)、知识蒸馏 (Distill)、测试评估 (Test) 与动态量化 (Quantize)"""

import argparse
import os
import pickle
import random
from importlib import import_module

import numpy as np
import torch
from utils.build_dataset import build_dataset, build_dataset_CNN
from utils.build_iterator import build_iterator
from utils.train_eval import test, train, train_kd


def set_seed(seed: int = 1) -> None:
    """锁定随机种子以保证实验可复现"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model_module(model_name: str):
    """动态加载模型模块，兼容驼峰 (textCNN) 与蛇形 (text_cnn) 命名"""
    name_clean = model_name.strip()
    try:
        return import_module(f"models.{name_clean}")
    except ModuleNotFoundError:
        alt_name = (
            "text_cnn"
            if name_clean.lower() in ["textcnn", "text_cnn"]
            else name_clean.lower()
        )
        try:
            return import_module(f"models.{alt_name}")
        except ModuleNotFoundError:
            raise ValueError(
                f"未找到对应的模型文件: models/{name_clean}.py 或 models/{alt_name}.py"
            )


def resolve_quant_plan(requested_device: torch.device):
    """根据请求设备与底层硬件架构，自动决策最终运行设备与量化方案。

    返回值: (actual_device, quant_config, quantize_fn, save_suffix)
    """
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA 设备，但当前环境无可用的 CUDA GPU！")

        major, minor = torch.cuda.get_device_capability(requested_device)
        gpu_name = torch.cuda.get_device_name(requested_device)

        # 具备原生 FP8 支持 (Compute Capability >= 8.9，如 RTX 40系 / Hopper)
        if (major, minor) >= (8, 9):
            print(
                f"[Device] GPU: {gpu_name} (CC {major}.{minor}) -> 启用 GPU + FP8 动态量化"
            )
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


# =====================================================================
# 1. 单模型常规训练 (Train)
# =====================================================================
def run_train(args, target_device):
    model_module = load_model_module(args.model)
    config = model_module.Config(args.dataset)
    config.device = target_device

    print(f"\n{'='*20} Mode: Standard Training [{config.model_name.upper()}] {'='*20}")

    if "cnn" in config.model_name.lower():
        vocab, train_data, dev_data, test_data = build_dataset_CNN(config)
        config.n_vocab = len(vocab)
        config.pad_idx = vocab.get("<PAD>", 0)
    else:
        train_data, dev_data, test_data = build_dataset(config)

    train_iter = build_iterator(train_data, config, shuffle=True)
    dev_iter = build_iterator(dev_data, config, shuffle=False)
    test_iter = build_iterator(test_data, config, shuffle=False)

    model = model_module.Model(config).to(config.device)
    print(f"Starting training on device: {config.device}...")
    train(config, model, train_iter, dev_iter, test_iter)


# =====================================================================
# 2. 知识蒸馏训练 (Distill)
# =====================================================================
def run_distill(args, target_device):
    teacher_name = args.teacher.strip()
    student_name = (args.student or args.model).strip()

    print(
        f"\n{'='*20} Mode: Knowledge Distillation [{teacher_name.upper()} -> {student_name.upper()}] {'='*20}"
    )

    teacher_module = load_model_module(teacher_name)
    student_module = load_model_module(student_name)

    teacher_config = teacher_module.Config(args.dataset)
    teacher_config.device = target_device

    student_config = student_module.Config(args.dataset)
    student_config.device = target_device

    # 1. 检查并载入预训练好的教师模型
    if not os.path.exists(teacher_config.save_path):
        raise FileNotFoundError(
            f"未找到教师模型权重: {teacher_config.save_path}，请先执行 --mode train --model {teacher_name} 微调教师模型！"
        )

    print(f"Loading teacher weights from [{teacher_config.save_path}]...")
    teacher_model = teacher_module.Model(teacher_config).to(target_device)
    teacher_model.load_state_dict(
        torch.load(
            teacher_config.save_path,
            map_location=target_device,
            weights_only=True,
        )
    )
    teacher_model.eval()

    # 彻底冻结教师参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 2. 处理双输入数据集并同步词表
    print("Processing dual-input dataset for distillation...")
    teacher_config.student_tokenizer = getattr(
        student_config, "student_tokenizer", lambda x: list(x)
    )
    teacher_config.vocab_path = student_config.vocab_path

    vocab, train_data, dev_data, test_data = build_dataset_CNN(teacher_config)

    # 同步词表至学生配置
    student_config.n_vocab = len(vocab)
    student_config.pad_idx = vocab.get("<PAD>", 0)

    # 3. 构建迭代器 (训练集双输入并打乱，验证/测试集仅用于学生评估)
    train_iter = build_iterator(train_data, student_config, shuffle=True)
    train_iter.model_name = "kd"

    dev_iter = build_iterator(dev_data, student_config, shuffle=False)
    dev_iter.model_name = student_config.model_name

    test_iter = build_iterator(test_data, student_config, shuffle=False)
    test_iter.model_name = student_config.model_name

    # 4. 初始化学生模型并启动蒸馏
    student_model = student_module.Model(student_config).to(target_device)
    print(
        f"Distillation Parameters: Alpha={args.alpha}, Temperature={args.temperature}, LR={student_config.learning_rate}"
    )
    train_kd(
        bert_config=teacher_config,
        cnn_config=student_config,
        bert_model=teacher_model,
        cnn_model=student_model,
        train_iter=train_iter,
        dev_iter=dev_iter,
        test_iter=test_iter,
        alpha=args.alpha,
        T=args.temperature,
    )


# =====================================================================
# 3. 模型测试与指标评估 (Test)
# =====================================================================
def run_test(args, requested_device):
    model_name = (
        args.student if args.student and not args.model else args.model
    ).strip()
    model_module = load_model_module(model_name)
    config = model_module.Config(args.dataset)

    # 决策设备与量化方案
    if args.use_quantized:
        config.device, quant_config, quantize_fn, save_suffix = (
            resolve_quant_plan(requested_device)
        )
    else:
        config.device = requested_device

    print(f"\n{'='*20} Mode: Evaluation [{config.model_name.upper()}] {'='*20}")

    # 数据集加载与词表还原
    is_cnn = "cnn" in config.model_name.lower()
    if is_cnn:
        if not os.path.exists(config.vocab_path):
            raise FileNotFoundError(
                f"未找到词表文件: {config.vocab_path}，请先执行训练或蒸馏以生成词表！"
            )
        with open(config.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        config.n_vocab = len(vocab)
        config.pad_idx = vocab.get("<PAD>", 0)
        # 修复：开启 only_test=True，避免加载 train/dev
        _, _, _, test_data = build_dataset_CNN(config, only_test=True)
    else:
        _, _, test_data = build_dataset(config, only_test=True)

    test_iter = build_iterator(test_data, config, shuffle=False)
    test_iter.model_name = config.model_name

    model = model_module.Model(config)

    # 权重加载
    if args.use_quantized:
        weight_path = os.path.join(
            config.save_dir, f"{config.model_name}{save_suffix}"
        )
        if not os.path.exists(weight_path):
            legacy_path = os.path.join(
                config.save_dir, f"{config.model_name}_quant.pt"
            )
            if os.path.exists(legacy_path):
                weight_path = legacy_path

        print(f"Loading QUANTIZED model weights from: [{weight_path}]")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"未找到量化权重文件: {weight_path}，请先执行 --mode quantize"
            )

        quantize_fn(model, quant_config)
        # torchao 量化张量包含自定义子类，需指定 weights_only=False
        model.load_state_dict(
            torch.load(
                weight_path, map_location=config.device, weights_only=False
            )
        )
    else:
        weight_path = config.save_path
        print(f"Loading ORIGINAL float weights from: [{weight_path}]")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"未找到权重文件: {weight_path}，请先执行训练或蒸馏！"
            )
        model.load_state_dict(
            torch.load(
                weight_path, map_location=config.device, weights_only=True
            )
        )

    model.to(config.device)
    model.eval()

    print(f"Running evaluation on device: {config.device}...")
    # 修复：已在外部手动加载好对应权重（浮点或量化），禁止 test() 内部覆盖加载
    test(config, model, test_iter, load_from_save_path=False)


# =====================================================================
# 4. 动态量化导出 (Quantize)
# =====================================================================
def run_quantize(args, requested_device):
    model_name = args.model.strip()
    model_module = load_model_module(model_name)
    config = model_module.Config(args.dataset)

    actual_device, quant_config, quantize_fn, save_suffix = resolve_quant_plan(
        requested_device
    )
    config.device = actual_device

    print(
        f"\n{'='*20} Mode: Dynamic Quantization [{config.model_name.upper()}] {'='*20}"
    )

    if not os.path.exists(config.save_path):
        raise FileNotFoundError(
            f"未找到浮点权重: {config.save_path}，请先执行训练或蒸馏！"
        )

    # 针对 CNN 模型动态载入词表以保证模型尺寸匹配
    if "cnn" in config.model_name.lower() and os.path.exists(config.vocab_path):
        with open(config.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        config.n_vocab = len(vocab)
        config.pad_idx = vocab.get("<PAD>", 0)

    save_path_quant = os.path.join(
        config.save_dir, f"{config.model_name}{save_suffix}"
    )

    print(f"Loading float model weights from [{config.save_path}]...")
    model = model_module.Model(config)
    model.load_state_dict(
        torch.load(
            config.save_path, map_location=actual_device, weights_only=True
        )
    )
    model.to(actual_device)
    model.eval()

    print(
        f"Applying quantization using {quant_config.__class__.__name__} on {actual_device}..."
    )
    quantize_fn(model, quant_config)

    torch.save(model.state_dict(), save_path_quant)
    print(f"Quantized model successfully saved to: {save_path_quant}")


# =====================================================================
# 主函数与参数解析
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline: Train, Distill, Test, and Quantize"
    )
    parser.add_argument(
        "--mode",
        "--task",
        dest="mode",
        type=str,
        default="train",
        choices=["train", "distill", "test", "quantize", "trainbert", "train_kd"],
        help="Execution mode: train (standard), distill (BERT->CNN), test (evaluate), quantize (dynamic quantization)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        help="Target / Student model name (e.g., bert, textCNN, text_cnn)",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Teacher model name for knowledge distillation (e.g., bert)",
    )
    parser.add_argument(
        "--student",
        type=str,
        default=None,
        help="Student model name for knowledge distillation (default: same as --model)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="toutiao",
        help="Dataset name (default: toutiao)",
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
        help="Target device (e.g., cuda, cpu, cuda:0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Weight for KD soft loss (default: 0.8)",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=2.0,
        help="Distillation temperature T (default: 2.0)",
    )
    parser.add_argument(
        "--use_quantized",
        action="store_true",
        help="Whether to evaluate the quantized model checkpoint",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device_str = "cuda" if args.device.lower() == "gpu" else args.device
    target_device = torch.device(device_str)

    mode = args.mode.lower()
    if mode == "trainbert":
        mode = "train"
        args.model = "bert"
    elif mode == "train_kd":
        mode = "distill"
        if not args.teacher:
            args.teacher = "bert"

    if args.teacher and mode == "train":
        mode = "distill"

    if mode == "train":
        run_train(args, target_device)
    elif mode == "distill":
        if not args.teacher:
            args.teacher = "bert"
        run_distill(args, target_device)
    elif mode == "test":
        run_test(args, target_device)
    elif mode == "quantize":
        run_quantize(args, target_device)
    else:
        raise ValueError(f"未知运行模式: {args.mode}")


if __name__ == "__main__":
    main()