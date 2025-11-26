#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   decoder_layer.py
@Time    :   2025/11/24 10:59:54
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
import torch.nn as nn

from typing import Optional

from ..attention.multi_head_attention import MultiHeadAttention
from ..embeddings.positional_encoding import PositionalEncoding
from ..feedforward.FFN import FeedForwardNetwork
from .residual_connection import ResidualConnection


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()

        # three attention modules:
        # 1. masked self-attention
        # 2. cross-attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # residual connections
        self.residual1 = ResidualConnection(embed_dim, dropout)
        self.residual2 = ResidualConnection(embed_dim, dropout)
        self.residual3 = ResidualConnection(embed_dim, dropout)

        # feed forward
        self.ffn = FeedForwardNetwork(embed_dim, ff_hidden_dim, dropout)

    def forward(
        self,
        x,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):

        # 1. Masked Self-Attention
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask=tgt_mask)[0])

        # 2. Cross-Attention: Q = decoder, K/V = encoder output
        x = self.residual2(
            x,
            lambda x: self.cross_attn(x, memory, memory, mask=memory_mask)[0],
        )

        # 3. FFN
        x = self.residual3(x, self.ffn)

        return x
