#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   decoder.py
@Time    :   2025/11/24 13:26:32
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
import torch.nn as nn
from typing import Optional

from .layers.decoder_layer import TransformerDecoderLayer
from .embeddings.positional_encoding import PositionalEncoding


class TransformerDecoder(nn.Module):
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
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.use_pos_embed = use_pos_embed
        self.pos_embed = (
            PositionalEncoding(embed_dim, max_len) if use_pos_embed else None
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_pos_embed and self.pos_embed is not None:
            x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.norm(x)
        return x
