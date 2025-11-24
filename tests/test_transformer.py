#!/usr/bin/env python3
"""
##
##       filename: test_dataset.py
##        created: 2025/11/24
##         author: Paul_Bao
##            IDE: Neovim
##       Version : 1.0
##       Contact : paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
from models.transformer import Transformer
from models.utils.mask import create_padding_mask, make_decoder_mask

B, S_src, S_tgt = 2, 6, 5
vocab_size = 1000
embed_dim = 32
num_heads = 4
enc_layers = 2
dec_layers = 2
ff_hidden = 128

model = Transformer(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_encoder_layers=enc_layers,
    num_decoder_layers=dec_layers,
    ff_hidden_dim=ff_hidden,
    mode="encoder_decoder",
)

src = torch.randint(1, vocab_size, (B, S_src))
tgt = torch.randint(1, vocab_size, (B, S_tgt))

src_mask = create_padding_mask(src)  # (B,1,1,S_src)
tgt_mask = make_decoder_mask(tgt)  # (B,1,S_tgt,S_tgt)

logits = model(src_tokens=src, tgt_tokens=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
print("logits:", logits.shape)  # expect (B, S_tgt, V)
