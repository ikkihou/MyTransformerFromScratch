#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   multi_head_attention.py
@Time    :   2025/11/23 12:40:54
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        H = self.num_heads

        return x.view(B, L, H, D // H).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Hd = x.size()
        D = H * Hd

        return x.transpose(1, 2).contiguous().view(B, L, D)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if k is None:
            k = q
        if v is None:
            v = q

        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)

        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output, attn_weights
