from torch import nn
from torch import Tensor
import torch
import copy
from user.transformer.attention import Attention


class Transformer(nn.Module):
    def __init__(
            self,
            tgt_size: int,
            src_size: int,
            emb_dim_2d: int = 15,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1,
            dropout_prob: float = 0.1,
    ) -> None:
        super(Transformer, self).__init__()

        # para
        self.emb_dim_2d = emb_dim_2d
        self.src_size = src_size
        self.tgt_size = tgt_size

        # layers
        encoder_layer = TransformerEncoderLayer(src_size, src_size, emb_dim_2d, dropout_prob)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer1 = TransformerDecoderLayer(tgt_size, src_size, emb_dim_2d, dropout_prob)
        self.decoder1 = TransformerDecoder(decoder_layer1, num_decoder_layers)
        decoder_layer2 = TransformerDecoderLayer(tgt_size, src_size, emb_dim_2d, dropout_prob)
        self.decoder2 = TransformerDecoder(decoder_layer2, num_decoder_layers)

    def forward(
            self,
            src: Tensor,
            tgt1: Tensor,
            tgt2: Tensor
    ) -> Tensor:
        # check batch size
        assert src.size(0) == tgt1.size(0), "batch_size of src and tgt is not same!"
        assert src.size(0) == tgt2.size(0)
        batch_size, tgt_size, emb, _ = tgt1.size()
        memory = self.encoder(src)
        d1 = self.decoder1(tgt1, memory).contiguous().view(batch_size, emb, emb * tgt_size)
        d2 = self.decoder2(tgt2, memory).contiguous().view(batch_size, emb, emb * tgt_size)
        output = torch.bmm(d1, d2.transpose(1, 2))
        output = torch.softmax(
            output.contiguous().view(batch_size, emb * emb),
            dim=-1
        )
        return output.contiguous().view(batch_size, emb, emb)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, tgt_size, src_size, emb_dim_2d, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = Attention(tgt_size, src_size, emb_dim_2d, dropout_prob=dropout_prob)
        self.out_proj = nn.Sequential(
            nn.Conv2d(tgt_size, tgt_size, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(tgt_size),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,
                src: Tensor) -> Tensor:
        """
        :param src: Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param src_mask: Tensor, shape: [src_sequence_size, src_sequence_size]
        :param src_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        :return: Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        """
        # self attention
        src = src + self.dropout(self.self_attn(src, src, src))
        src = self.out_proj(src)
        return src


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        # 将同一个encoder_layer进行deepcopy n次
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self,
                src: Tensor,
                ) -> Tensor:
        output = src
        # 串行n个encoder_layer
        for mod in self.layers:
            output = mod(output)
        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 tgt_size,
                 src_size,
                 emb_dim_2d,
                 dropout_prob=0.1):
        super(TransformerDecoderLayer, self).__init__()
        # 初始化基本层
        self.self_attn = Attention(tgt_size, tgt_size, emb_dim_2d, dropout_prob=dropout_prob)
        self.multihead_attn = Attention(tgt_size, src_size, emb_dim_2d, dropout_prob=dropout_prob)
        self.out_proj = nn.Sequential(
            nn.Conv2d(tgt_size, tgt_size, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(tgt_size),
            nn.LeakyReLU(0.3)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,
                tgt: Tensor,
                memory: Tensor) -> Tensor:
        """
        :param tgt:                     Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        :param memory:                  Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param tgt_mask:                Tensor, shape: [tgt_sequence_size, tgt_sequence_size]
        :param memory_mask:             Tensor, shape: [src_sequence_size, src_sequence_size]
        :param tgt_key_padding_mask:    Tensor, shape: [batch_size, tgt_sequence_size]
        :param memory_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        :return:                        Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        """
        # tgt的self attention
        tgt = tgt + self.dropout(self.self_attn(tgt, tgt, tgt))
        # tgt与memory的attention
        tgt = tgt + self.dropout(self.multihead_attn(tgt, memory, memory))
        tgt = self.out_proj(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """
        :param tgt:                     Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        :param memory:                  Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param tgt_mask:                Tensor, shape: [tgt_sequence_size, tgt_sequence_size]
        :param memory_mask:             Tensor, shape: [src_sequence_size, src_sequence_size]
        :param tgt_key_padding_mask:    Tensor, shape: [batch_size, tgt_sequence_size]
        :param memory_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        :return:                        Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
