# coding: UTF-8
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class Config:
    def __init__(self):
        self.model_name = "resnet-18"

        # 动态定位项目根目录（cifar10/）：
        # 本文件绝对路径为 .../cifar10/src/models/resnet_18.py
        # 回溯 2 级：models -> src -> cifar10
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

        self.class_list = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # 1. 数据集相关路径
        self.data_path = os.path.join(project_root, "data")

        # 2. 模型保存路径
        self.save_dir = os.path.join(project_root, "out_models_bin")
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.model_name + ".pt")
        self.save_path_quant = os.path.join(self.save_dir, self.model_name + "_quant.pt")

        # 3. 硬件环境与训练超参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 8
        self.batch_size = 128
        self.learning_rate = 1e-4

        # 4. 模型结构参数（ResNet-18 最终输出维度为 512，CIFAR-10 为 10 类）
        self.hidden_size = 512
        self.num_classes = 10

        # 5. 预训练模型相关（对应 data/resnet-18-pretrain）
        self.resnet_18_path = os.path.join(self.data_path, "resnet-18-pretrain")
        self.resnet_18_config = AutoConfig.from_pretrained(
            self.resnet_18_path,
            local_files_only=True
        )


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载本地预训练特征提取器
        self.resnet_18 = AutoModel.from_pretrained(
            config.resnet_18_path,
            config=config.resnet_18_config,
            local_files_only=True
        )

        # 自定义分类线性层：512 -> 10
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # 传入 pixel_values 计算前向特征
        outputs = self.resnet_18(pixel_values=x)
        
        # Hugging Face ResNet 的 pooler_output 维度是 (batch_size, 512, 1, 1)
        # 展平成 (batch_size, 512) 后送入分类器
        pooled = outputs.pooler_output.flatten(1)
        logits = self.fc(pooled)
        
        return logits