#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   transformer.py
@Time    :   2025/11/24 13:27:39
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
import torch.nn as nn
from typing import Optional

from encoder import TransformerEncoder
from decoder import TransformerDecoder
from embeddings.positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, scale: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.scale = scale
        self.embed_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, S)
        out = self.embed(token_ids)  # (B, S, D)
        if self.scale:
            out = out * (self.embed_dim**0.5)
        return out


class Transformer(nn.Module):
    """
    Unified Transformer:
      - encoder_decoder: use both encoder & decoder (translation)
      - encoder_only: just encoder + classifier/projection head
      - decoder_only: just decoder (e.g., causal LM)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        max_len: int = 512,
        mode: str = "encoder_decoder",  # "encoder_decoder" | "encoder_only" | "decoder_only"
    ):
        super().__init__()
        self.mode = mode
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim, max_len)

        if mode in ("encoder_decoder", "encoder_only"):
            self.encoder = TransformerEncoder(
                num_encoder_layers,
                embed_dim,
                num_heads,
                ff_hidden_dim,
                dropout,
                max_len,
            )
        else:
            self.encoder = None

        if mode in ("encoder_decoder", "decoder_only"):
            self.decoder = TransformerDecoder(
                num_decoder_layers,
                embed_dim,
                num_heads,
                ff_hidden_dim,
                dropout,
                max_len,
            )
        else:
            self.decoder = None

        # LM head / projection for logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        src_tokens: Optional[torch.Tensor] = None,
        tgt_tokens: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """
        For encoder-decoder:
           - src_tokens: (B, S_src)
           - tgt_tokens: (B, S_tgt)
           - return logits for tgt positions (B, S_tgt, V)
        For encoder-only:
           - src_tokens provided; returns embeddings or pooled outputs (we return lm_head logits)
        For decoder-only:
           - tgt_tokens provided; returns logits
        """
        if self.mode == "encoder_decoder":
            assert src_tokens is not None and tgt_tokens is not None
            enc_in = self.pos_embed(self.token_embed(src_tokens))
            memory = self.encoder(enc_in, src_mask)
            dec_in = self.pos_embed(self.token_embed(tgt_tokens))
            dec_out = self.decoder(
                dec_in, memory, tgt_mask=tgt_mask, memory_mask=src_mask
            )
            logits = self.lm_head(dec_out)
            return logits

        elif self.mode == "encoder_only":
            assert src_tokens is not None
            enc_in = self.pos_embed(self.token_embed(src_tokens))
            enc_out = self.encoder(enc_in, src_mask)
            logits = self.lm_head(enc_out)
            return logits

        elif self.mode == "decoder_only":
            assert tgt_tokens is not None
            dec_in = self.pos_embed(self.token_embed(tgt_tokens))
            # in decoder-only, memory is None and cross-attn not used
            dec_out = self.decoder(
                dec_in, memory=None, tgt_mask=tgt_mask, memory_mask=None
            )
            logits = self.lm_head(dec_out)
            return logits

        else:
            raise ValueError(f"Unknown mode {self.mode}")
