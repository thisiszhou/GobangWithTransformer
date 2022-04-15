from torch.nn import Module, Linear, Conv2d, Sequential, BatchNorm2d, ReLU
import torch
from torch import Tensor
from typing import Optional


class Attention(Module):
    def __init__(self,
                 tgt_size,
                 src_size,
                 emb_dim_2d,
                 dropout_prob=0.
                 ):
        super(Attention, self).__init__()
        self.emb_dim_2d = emb_dim_2d
        self.dropout_prob = dropout_prob

        self.q_in_proj = Sequential(
            Conv2d(tgt_size, tgt_size, kernel_size=(3, 3), padding=1),
            BatchNorm2d(tgt_size),
            ReLU()
        )
        self.k_in_proj = Sequential(
            Conv2d(src_size, src_size, kernel_size=(3, 3), padding=1),
            BatchNorm2d(src_size),
            ReLU()
        )
        self.v_in_proj = Sequential(
            Conv2d(src_size, src_size, kernel_size=(3, 3), padding=1),
            BatchNorm2d(src_size),
            ReLU()
        )
        self.out_proj = Sequential(
            Linear(emb_dim_2d * emb_dim_2d, 2 * emb_dim_2d * emb_dim_2d),
            ReLU(),
            Linear(2 * emb_dim_2d * emb_dim_2d, emb_dim_2d * emb_dim_2d),
            ReLU()
        )

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor):
        """
        :param query: Tensor, shape: [batch_size, tgt_size, emb_dim_2d, emb_dim_2d]
        :param key:   Tensor, shape: [batch_size, src_size, emb_dim_2d, emb_dim_2d]
        :param value: Tensor, shape: [batch_size, src_sequence_size, emb_dim_2d, emb_dim_2d]
        :return: Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        """

        #
        batch_size, tgt_size, emb, emb = query.size()
        _, src_size, _, _ = key.size()

        # 三个Q、K、V的卷积层
        q = self.q_in_proj(query)
        k = self.k_in_proj(key)
        v = self.v_in_proj(value)
        scaling = float(self.emb_dim_2d) ** -0.5

        # 这里对Q进行一个统一常数放缩
        q = q * scaling

        # attention
        q = q.contiguous().view(batch_size, tgt_size, emb * emb)
        k = k.contiguous().view(batch_size, src_size, emb * emb)
        v = v.contiguous().view(batch_size, src_size, emb * emb)

        # Q、K进行bmm批次矩阵乘法，得到权重矩阵
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # 权重矩阵进行softmax，使得单行的权重和为1
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.dropout(attn_output_weights, p=self.dropout_prob, train=self.training)

        # 权重矩阵与V矩阵进行bmm操作，得到输出
        attn_output = torch.bmm(attn_output_weights, v)

        # 最后一层全连接层，得到最终输出
        attn_output = self.out_proj(attn_output)

        # 转换维度，将num_heads * head_dim reshape回word_emb_dim，并且将batch_size调回至第1维
        attn_output = attn_output.contiguous().view(batch_size, tgt_size, emb, emb)

        return attn_output
