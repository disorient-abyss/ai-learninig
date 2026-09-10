import torch
from layer_norm import LayerNorm
from sublayer_connection import SublayerConnection
from torch import nn
from utils import clones


# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(
        self, 
        size: int, 
        self_attn: nn.Module, 
        feed_forward: nn.Module, 
        dropout: float = 0.1
    ):
        
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 复制两个残差连接层：一个用于自注意力，一个用于前馈网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask。"""
        # 1. 经过自注意力子层 + 残差与归一化
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 2. 经过前馈网络子层 + 残差与归一化
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        # 使用 clones 深拷贝 N 个独立的编码器层
        self.layers = clones(layer, N)
        # Pre-LN 结构在经过 N 层残差累加后，必须在最终输出前进行全局归一化
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # 数据顺次通过 N 个独立的编码器层
        for layer in self.layers:
            x = layer(x, mask=mask)
        # 最终归一化输出
        return self.norm(x)