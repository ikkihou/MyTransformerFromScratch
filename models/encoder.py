#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   encoder.py
@Time    :   2025/11/24 13:24:23
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
import torch.nn as nn
from typing import Optional, List

from .layers.encoder_layer import TransformerEncoderLayer
from .embeddings.positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        max_len: int = 512,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.use_pos_embed = use_pos_embed
        self.pos_embed = (
            PositionalEncoding(embed_dim, max_len) if use_pos_embed else None
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, S, D)
        src_mask: mask broadcastable to (B, H, S, S) or (B, 1, 1, S) etc.
        """
        if self.use_pos_embed and self.pos_embed is not None:
            x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        x = self.norm(x)
        return x
