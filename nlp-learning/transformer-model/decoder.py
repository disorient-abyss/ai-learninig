import torch
from layer_norm import LayerNorm
from sublayer_connection import SublayerConnection
from torch import nn
from utils import clones


# 使用EncoderLayer类实现编码器层
class DecoderLayer(nn.Module):
    def __init__(
        self, 
        size: int, 
        self_attn: nn.Module, 
        src_attn: nn.Module,
        feed_forward: nn.Module, 
        dropout: float = 0.1
    ):
        
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 3 个残差连接块：分别对应 Self-Attn、Cross-Attn、Feed-Forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        """ 
        :param x: 目标序列张量 (batch_size, tgt_len, d_model)
        :param memory: 编码器最终输出特征 (batch_size, src_len, d_model)
        :param source_mask: 针对源端编码器的掩码 (防止注意到源端 Padding)
        :param target_mask: 针对目标端自注意力的掩码 (Padding Mask + 因果未来词 Mask)
        """
        # 将 memory 表示成 m 方便之后使用
        m = memory

        # 1. 经过自注意力子层 + 残差与归一化
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=target_mask))

        # 2. 经过注意力子层 + 残差与归一化
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, mask=source_mask))

        # 2. 经过前馈网络子层 + 残差与归一化
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        # 使用 clones 深拷贝 N 个独立的编码器层
        self.layers = clones(layer, N)
        # Pre-LN 结构在经过 N 层残差累加后，必须在最终输出前进行全局归一化
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 数据顺次通过 N 个独立的编码器层
        for layer in self.layers:
            x = layer(
            x,
            memory=memory,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        # 最终归一化输出
        return self.norm(x)