#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   residual.py
@Time    :   2025/11/24 10:47:34
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch.nn as nn


class ResidualConnection(nn.Module):
    """
    Pre-LN Transformer: Norm → Sublayer → Dropout → Residual
    """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        sublayer: a function that takes normalized x
        """
        out = sublayer(self.norm(x))
        return x + self.dropout(out)
