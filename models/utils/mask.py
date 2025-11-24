#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   mask.py
@Time    :   2025/11/24 13:29:45
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    seq: (B, L) token ids
    returns mask: (B, 1, 1, L) with 1 for valid tokens and 0 for pad positions,
    ready to be used with attention scores (where masked positions are masked).
    """
    # valid positions -> 1, pad -> 0
    mask = (
        (seq != pad_token_id).unsqueeze(1).unsqueeze(1).to(dtype=torch.uint8)
    )  # (B,1,1,L)
    return mask


def create_causal_mask(size: int, device=None) -> torch.Tensor:
    """
    Create an upper-triangular (causal) mask for target sequence of length size.
    returns (1, 1, size, size) with 1 in allowed positions (lower triangle), 0 elsewhere.
    """
    mask = torch.tril(torch.ones((size, size), dtype=torch.uint8, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,S,S)


def make_decoder_mask(tgt_seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Combine padding mask and causal mask to produce final tgt_mask:
      - tgt_seq: (B, T)
      returns mask: (B, 1, T, T) with 1 for allowed positions
    """
    B, T = tgt_seq.shape
    padding_mask = create_padding_mask(tgt_seq, pad_token_id=pad_token_id)  # (B,1,1,T)
    causal = create_causal_mask(T, device=tgt_seq.device)  # (1,1,T,T)
    # broadcast padding to (B,1,T,T)
    padding_mask = padding_mask.expand(-1, -1, T, -1)  # (B,1,T,T)
    # causal broadcast to (B,1,T,T)
    causal = causal.expand(B, -1, -1, -1)
    combined = padding_mask & causal  # bitwise and; dtype uint8
    return combined
