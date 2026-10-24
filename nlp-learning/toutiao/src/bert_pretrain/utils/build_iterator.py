import torch


class DatasetIterater:

  def __init__(self, dataset, batch_size, model_name="bert", device="cpu"):
    self.dataset = dataset
    self.batch_size = batch_size
    self.model_name = model_name
    self.device = device

    self.n_batches = len(dataset) // batch_size
    self.residue = len(dataset) % batch_size != 0
    self.index = 0

    # 根据 dataset 提供的 contents (token_ids, int(label), seq_len, mask) 结果进行转换
  def _to_tensor(self, datas):
    x = torch.as_tensor([item[0] for item in datas], dtype=torch.long)
    y = torch.as_tensor([item[1] for item in datas], dtype=torch.long)
    seq_len = torch.as_tensor([item[2] for item in datas], dtype=torch.long)

    if self.model_name in ("bert", "multi_task_bert"):
      mask = torch.as_tensor([item[3] for item in datas], dtype=torch.long)
      return (x, seq_len, mask), y

    return (x, seq_len), y

  def __next__(self):
    # 计算总共有多少个 Batch（如果有余数，总批次数要在整除基础上加 1）。
    total_batches = self.n_batches + 1 if self.residue else self.n_batches
    # 抛出标准迭代终止异常，通知外部的 for 循环停止当前轮次的遍历。
    if self.index >= total_batches:
      self.index = 0
      raise StopIteration


    start_idx = self.index * self.batch_size
    end_idx = min(start_idx + self.batch_size, len(self.dataset))
    batch_slice = self.dataset[start_idx:end_idx]

    self.index += 1
    return self._to_tensor(batch_slice)

  def __iter__(self):
    self.index = 0
    return self

  def __len__(self):
    if self.residue:
      return self.n_batches + 1
    return self.n_batches


def build_iterator(dataset, config):
  return DatasetIterater(dataset, config.batch_size, config.model_name, device=config.device)