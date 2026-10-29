# coding: UTF-8
import os

import torch
import torch.nn.functional as F
from torch import nn


def _find_project_root(current_file: str) -> str:
    """自动向上回溯查找包含 data/class.txt 的项目根目录，兼容不同目录层级结构"""
    curr_dir = os.path.dirname(os.path.abspath(current_file))
    for _ in range(5):
        if os.path.exists(os.path.join(curr_dir, "data", "class.txt")):
            return curr_dir
        parent_dir = os.path.dirname(curr_dir)
        if parent_dir == curr_dir:
            break
        curr_dir = parent_dir
    # 若未探测到，默认回退至当前文件上两级或当前工作目录
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(current_file)), "..")
    )


class Config:
    def __init__(self, dataset="toutiao"):
        self.model_name = "text_cnn"

        # 动态定位项目根目录
        project_root = _find_project_root(__file__)

        # 1. 数据集相关路径
        self.data_path = os.path.join(project_root, "data")
        self.train_path = os.path.join(self.data_path, "train.txt")
        self.dev_path = os.path.join(self.data_path, "dev.txt")
        self.test_path = os.path.join(self.data_path, "test.txt")
        self.class_path = os.path.join(self.data_path, "class.txt")

        # 读取类别清单
        if not os.path.exists(self.class_path):
            raise FileNotFoundError(f"未找到类别标签文件: {self.class_path}")

        with open(self.class_path, "r", encoding="utf-8") as f:
            self.class_list = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.class_list)

        # 2. 模型保存路径
        self.save_dir = os.path.join(project_root, "save_dict")
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.model_name + ".pt")
        self.save_path_quant = os.path.join(
            self.save_dir, self.model_name + "_quant.pt"
        )

        # 3. 硬件环境与训练超参数
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.require_improvement = 4000
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3

        # 4. TextCNN 专属模型结构参数
        self.text_cnn_path = os.path.join(self.data_path, "text_cnn")
        os.makedirs(self.text_cnn_path, exist_ok=True)
        self.vocab_path = os.path.join(self.text_cnn_path, "vocab.pkl")

        self.n_vocab = 10000
        # 与 build_vocab.py 中 <PAD> 索引 0 保持一致
        self.pad_idx = 0
        self.embedding_dim = 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        pad_idx = config.pad_idx if config.pad_idx is not None else 0

        self.embedding = nn.Embedding(
            config.n_vocab, config.embedding_dim, padding_idx=pad_idx
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, config.num_filters, (k, config.embedding_dim))
                for k in config.filter_sizes
            ]
        )

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(
            config.num_filters * len(config.filter_sizes), config.num_classes
        )

    def conv_and_pool(self, x, conv):
        # 1. 2D 卷积变换: 
        #    输入: [B, 1, L, D]
        #    核大小: (k, D) -> 纵向滑动步长为 1，横向正好覆盖整个词向量维度 D
        #    输出: [B, num_filters, L - k + 1, 1]
        x = conv(x)

        # 2. 激活函数 + 去除宽度维度:
        #    F.relu(...) 激活不改变形状 -> [B, num_filters, L - k + 1, 1]
        #    squeeze(3) 挤压掉大小为 1 的最后一维 -> [B, num_filters, L - k + 1]
        x = F.relu(x).squeeze(3)

        # 3. 1D 最大池化 (Global 1D Max Pooling):
        #    此时时间步序列长度为 x.size(2) = (L - k + 1)
        #    池化核大小直接设为序列长度 x.size(2)，横跨整句话提取最强特征
        #    池化后形状: [B, num_filters, 1]
        #    squeeze(2) 挤压掉池化后的时间步维度 -> [B, num_filters]
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)

        # 返回单核提取的代表性特征: [B, num_filters]
        return x  
    def forward(self, x):
        """前向传播逻辑
        
        Args:
            x: 输入张量或元组 (context, seq_len)，形状为 [batch_size, seq_len]
            
        Returns:
            logits: 各类别的未归一化得分 [batch_size, num_classes]
        """
        # 兼容输入为元组/列表 (如 (student_ids, seq_len)) 与单张量形式
        if isinstance(x, (tuple, list)):
            context = x[0]
        else:
            context = x

        # 1. 查表获取词向量并增加通道维度 (单通道图像模式):
        #    self.embedding(context) -> [B, L, D]
        #    unsqueeze(1)            -> [B, 1, L, D] (相当于单通道灰度图: 1 通道, 高度 L, 宽度 D)
        out = self.embedding(context).unsqueeze(1)

        # 2. 多尺度卷积抽取与全局特征拼接:
        #    对每个 k in filter_sizes 执行 conv_and_pool:
        #      k=2 -> [B, num_filters]
        #      k=3 -> [B, num_filters]
        #      k=4 -> [B, num_filters]
        #    torch.cat(..., dim=1) 在特征通道维度进行拼接:
        #    拼接后维度 -> [B, num_filters * len(filter_sizes)]
        out = torch.cat(
            [self.conv_and_pool(out, conv) for conv in self.convs], dim=1
        )

        # 3. 随机失活 (防止全连接层过拟合):
        #    形状保持不变 -> [B, num_filters * len(filter_sizes)]
        out = self.dropout(out)

        # 4. 全连接分类层:
        #    线性映射: [B, num_filters * len(filter_sizes)] -> [B, num_classes]
        logits = self.fc(out)

        # [B, num_classes]
        return logits