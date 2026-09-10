import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """ Transformer 前馈全连接网络 (FFN) """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """
        :param d_model: 输入与输出特征维度 (例如 512 或 768)
        :param d_ff: 隐层升维维度，通常为 4 * d_model (例如 2048)
        :param dropout: Dropout 丢弃概率
        :param activation: 激活函数选择，可选 "relu" 或 "gelu"
        """
        super().__init__()  # Python 3 推荐使用简化的 super()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # 1. 将激活函数实例化存入类成员变量，便于打印模型结构与导出 ONNX/TorchScript
        if activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}，请选择 'relu' 或 'gelu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入张量，形状为 (batch_size, seq_len, d_model)
        :return: 输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 计算路径: x -> Linear(d_model, d_ff) -> Act -> Dropout -> Linear(d_ff, d_model)
        return self.w2(self.dropout(self.act(self.w1(x))))