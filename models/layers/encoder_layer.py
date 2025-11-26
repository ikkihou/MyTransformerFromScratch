#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   encoder_layer.py
@Time    :   2025/11/24 10:51:03
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


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.residual1 = ResidualConnection(embed_dim, dropout)
        self.residual2 = ResidualConnection(embed_dim, dropout)

        self.ffn = FeedForwardNetwork(embed_dim, ff_hidden_dim, dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        # 1. Self-Attention + Residual
        x = self.residual1(x, lambda x: self.self_attn(x, mask=src_mask)[0])

        # 2. FFN + Residual
        x = self.residual2(x, self.ffn)

        return x
