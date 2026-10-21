import torch
from torch import nn
from utils.attention import attention
from utils.net_clones import clones


# 这里使用一个类处理多头注意力机制
class MultiHeadedAttention(nn.Module):
    """多头注意力机制 (Multi-Head Attention)"""

    def __init__(self, head: int, d_model: int, dropout_rate: float = 0.1):
        # 在类的初始化时，会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，dropout 代表进行 dropout 操作时置 0 比率，默认是 0.1。
        super().__init__()

        # 在函数中，首先使用了一个测试中常用的 assert 语句，判断h是否能被 d_model 整除，
        # 这是因为我们之后要给每个头分配等量的词特征。也就是 embedding_dim/head 个。
        assert d_model % head == 0, "d_model 必须能被 head 整除"

        # 得到每个头获得的分割词向量维度 d_k
        self.d_k = d_model // head
        # 传入头数 h
        self.head = head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是 embedding_dim x embedding_dim，然后使用 clones 函数克隆四个，
        # 为什么呢，这是因为在多头注意力中，Q，K，V 各需要一个，最后拼接的矩阵还需要一个，因此一共是 4 个。
        # 这里拿到 d_model x d_model 的方形矩阵进行拷贝，累计拷贝 4 个
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None。
        self.attn: torch.Tensor | None = None

        # 最后就是一个self.dropout对象，它通过nn中的Dropout实例化而来，置0比率为传进来的参数dropout。
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 前向逻辑函数，它的输入参数有四个，前三个就是注意力机制需要的Q, K, V,
        # 最后一个是注意力机制中可能需要的mask掩码张量，默认是None。

        # 如果存在掩码张量 mask
        if mask is not None:
            # 扩展为 (batch_size, 1, 1, seq_len)，以便与 (batch_size, head, seq_len, d_k) 正确广播
            mask = mask.unsqueeze(1)

        # 接着，我们获得一个 batch_size 的变量，他是 query 尺寸的第 1 个数字，代表有多少条样本。
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用 zip 将输入 QKV 与三个线性层组到一起，然后使用 for 循环，将输入 QKV 分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度 h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的 -1 代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值。然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从 attention 函数中可以看到，利用的是原始输入的倒数第一和第二维。这样我们就得到了每个头的输入。

        # 1. 线性变换投影并分头 (batch_size, seq_len, d_model) -> (batch_size, head, seq_len, d_k)
        query, key, value = [
            linear(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 2. 得到每个头的输入后，接下来就是将他们传入到 attention 中，
        # 这里直接调用我们之前实现的attention函数。同时也将 mask 和 dropout 传入其中。
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. 多头拼接还原 (batch_size, head, seq_len, d_k) -> (batch_size, seq_len, d_model)
        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用 contiguous 方法，
        # 这个方法的作用就是能够让转置后的张量应用 view 方法，否则将无法直接使用，
        # 所以，下一步就是使用 view 重塑形状，变成和输入形状相同。
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 4. 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的每一行输出
        return self.linears[-1](x)