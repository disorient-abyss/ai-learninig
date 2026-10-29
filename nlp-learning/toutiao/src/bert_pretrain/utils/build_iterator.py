# coding: UTF-8
import random

import torch


class DatasetIterater:

    def __init__(
        self,
        dataset,
        batch_size,
        model_name="kd",
        device="cpu",
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_name = str(model_name).lower()
        self.device = device
        self.shuffle = shuffle

        self.n_batches = len(dataset) // batch_size
        self.residue = len(dataset) % batch_size != 0
        self.index = 0

        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)

    def _to_tensor(self, datas):
        first_item = datas[0]

        # 分支 1：知识蒸馏双输入模式 (KD)
        if len(first_item) == 3 and self.model_name == "kd":
            labels = [item[2] for item in datas]
            y = torch.as_tensor(labels, dtype=torch.long)

            t_ids = torch.as_tensor([item[0][0] for item in datas], dtype=torch.long)
            t_masks = torch.as_tensor([item[0][1] for item in datas], dtype=torch.long)
            s_ids = torch.as_tensor([item[1][0] for item in datas], dtype=torch.long)
            s_lens = torch.as_tensor([item[1][1] for item in datas], dtype=torch.long)

            return ((t_ids, t_masks), (s_ids, s_lens)), y

        # 分支 2：TextCNN 学生模型单输入模式
        if len(first_item) == 2 and isinstance(first_item[0], (tuple, list)):
            labels = [item[1] for item in datas]
            y = torch.as_tensor(labels, dtype=torch.long)

            s_ids = torch.as_tensor([item[0][0] for item in datas], dtype=torch.long)
            s_lens = torch.as_tensor([item[0][1] for item in datas], dtype=torch.long)
            return (s_ids, s_lens), y

        # 分支 3：标准 BERT 教师模型单输入模式
        if len(first_item) == 4:
            labels = [item[1] for item in datas]
            y = torch.as_tensor(labels, dtype=torch.long)

            t_ids = torch.as_tensor([item[0] for item in datas], dtype=torch.long)
            seq_lens = torch.as_tensor([item[2] for item in datas], dtype=torch.long)
            t_masks = torch.as_tensor([item[3] for item in datas], dtype=torch.long)
            return (t_ids, seq_lens, t_masks), y

        # 通用兜底格式: (x, y)
        labels = [item[-1] for item in datas]
        y = torch.as_tensor(labels, dtype=torch.long)
        features = torch.as_tensor([item[0] for item in datas], dtype=torch.long)
        return features, y

    def __next__(self):
        total_batches = self.n_batches + 1 if self.residue else self.n_batches
        if self.index >= total_batches:
            self.index = 0
            raise StopIteration

        start_idx = self.index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))

        batch_indices = self.indices[start_idx:end_idx]
        batch_slice = [self.dataset[i] for i in batch_indices]

        self.index += 1
        return self._to_tensor(batch_slice)

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        return self.n_batches


def build_iterator(dataset, config, shuffle: bool = False):
    """构建数据迭代器入口函数"""
    return DatasetIterater(
        dataset=dataset,
        batch_size=config.batch_size,
        model_name=getattr(config, "model_name", "kd"),
        device=config.device,
        shuffle=shuffle,
    )