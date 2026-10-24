# coding: UTF-8
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Config:
    def __init__(self, dataset="toutiao"):
        self.model_name = "roberta"

        # 动态定位项目根目录（toutiao/）：
        # 本文件绝对路径为 .../toutiao/src/bert_pretrain/models/bert.py
        # 向上回溯 3 级目录即可精确锁定 toutiao 项目根路径，避免不同运行目录下找不到文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

        # 1. 数据集相关路径
        # 根据结构，数据直接存放在根目录的 data/ 下
        self.data_path = os.path.join(project_root, "data")
        self.train_path = os.path.join(self.data_path, "train.txt")
        self.dev_path = os.path.join(self.data_path, "dev.txt")
        self.test_path = os.path.join(self.data_path, "test.txt")
        self.class_path = os.path.join(self.data_path, "class.txt")

        # 读取类别清单
        if not os.path.exists(self.class_path):
            raise FileNotFoundError(f"未找到类别标签文件: {self.class_path}")

        # 安全读取类别名单并及时释放文件句柄，过滤空行
        with open(self.class_path, "r", encoding="utf-8") as f:
            self.class_list = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.class_list)

        # 2. 模型保存路径
        self.save_dir = os.path.join(project_root, "model_bin")
        # os.makedirs 支持递归创建多级目录，exist_ok=True 避免并发或已存在时崩溃
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.model_name + ".pt")
        # INT8 动态量化模型权重路径
        self.save_path_quant = os.path.join(self.save_dir, self.model_name + "_quant.pt")

        # 3. 硬件环境与训练超参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.require_improvement = 1000  # 若超过1000个batch效果还没提升，则提前结束训练
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32               # 每句话截断/填充的目标长度
        self.learning_rate = 5e-5

        # 4. 预训练模型相关
        self.roberta_path = os.path.join(self.data_path, "roberta_pretrain")
        self.tokenizer = AutoTokenizer.from_pretrained(self.roberta_path)
        # from_pretrained 传入文件夹会自动寻找 config.json，无需手动拼文件名
        self.bert_config = AutoConfig.from_pretrained(self.roberta_path)
        # 动态读取隐藏层维度，避免后续换用其他模型时写死 768 导致尺寸不匹配
        self.hidden_size = getattr(self.bert_config, "hidden_size", 768)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载预训练模型
        self.bert = AutoModel.from_pretrained(config.roberta_path, config=config.bert_config)

        # 此处用于调节是否将 BERT 纳入微调训练（置为 True 进行端到端全量微调；置为 False 则冻结骨干只训练分类头）
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = True
        #     print(name)

        # 加入 Dropout 防止全连接层过拟合（读取配置中的 dropout，默认 0.1）
        dropout_prob = getattr(config.bert_config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        # 分类线性层
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
    def forward(self, x):
        # 优化 2：优雅解包，同时兼容 (context, seq_len, mask) 或 (context, mask)
        if isinstance(x, (tuple, list)):
            context = x[0]
            # 如果元组长度 >= 3，mask 在第 2 位（即 x[2]）；如果长度是 2，mask 在第 1 位
            mask = x[2] if len(x) >= 3 else x[1]
        else:
            context, mask = x, None

        # 得到 BERT 输出（只对 cls 做处理）
        # out.pooler_output 对应 [CLS] 经过内置 Dense + Tanh 后的张量，形状为 [batch_size, hidden_size]
        outputs = self.bert(context, attention_mask=mask)
        # pooler_output（官方内置处理好的写法），
        # 它完整执行了 $\text{pooled\_out} = \tanh(\text{CLS\_token} \times W_{\text{pooler}} + b_{\text{pooler}})$
        # 简单来说就是用 cls 进行分类
        pooled_out = outputs.pooler_output
        
        # 经过 dropout 后送入分类器
        pooled_out = self.dropout(pooled_out)
        logits = self.fc(pooled_out)
        
        return logits