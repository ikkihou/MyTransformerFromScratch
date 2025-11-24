#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   FFN.py
@Time    :   2025/11/23 23:21:36
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch.nn as nn
from torch.nn import functional as F


class FeedForwardNetwork(nn.Module):
    """
    A simple Feed Forward Neural Network (FFN) module.
    """

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
