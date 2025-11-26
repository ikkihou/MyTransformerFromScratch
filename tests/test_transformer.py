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
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.transformer import Transformer
from models.utils.mask import create_padding_mask, make_decoder_mask

raw = load_dataset("wmt14", "de-en")

train_small = raw["train"].shuffle(seed=42).select(range(2000))
valid_small = raw["validation"]

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

max_length = 32


def preprocess_fn(examples):
    src_texts = [ex["en"] for ex in examples["translation"]]
    tgt_texts = [ex["de"] for ex in examples["translation"]]

    src = tokenizer(
        src_texts, padding="max_length", truncation=True, max_length=max_length
    )
    tgt = tokenizer(
        tgt_texts, padding="max_length", truncation=True, max_length=max_length
    )

    return {
        "src_input_ids": src["input_ids"],
        "src_attention_mask": src["attention_mask"],
        "tgt_input_ids": tgt["input_ids"],
        "tgt_attention_mask": tgt["attention_mask"],
    }


processed_train = train_small.map(preprocess_fn, batched=True)
processed_valid = valid_small.map(preprocess_fn, batched=True)
processed_train.set_format(
    type="torch",
    columns=[
        "src_input_ids",
        "src_attention_mask",
        "tgt_input_ids",
        "tgt_attention_mask",
    ],
)

processed_valid.set_format(
    type="torch",
    columns=[
        "src_input_ids",
        "src_attention_mask",
        "tgt_input_ids",
        "tgt_attention_mask",
    ],
)

train_dataloader = DataLoader(processed_train, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(processed_valid, batch_size=32)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = Transformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    ff_hidden_dim=128,
    mode="encoder_decoder",
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for epoch in range(10):
    for batch in train_dataloader:
        src_ids = batch["src_input_ids"].to(DEVICE)
        tgt_ids = batch["tgt_input_ids"].to(DEVICE)

        src_pad_mask = create_padding_mask(
            src_ids, pad_token_id=tokenizer.pad_token_id
        )  # (B,1,1,T_src), 1=allow
        tgt_mask = make_decoder_mask(
            tgt_ids, pad_token_id=tokenizer.pad_token_id
        )  # (B,1,T_tgt,T_tgt), 1=allow

        src_pad_mask_bool = src_pad_mask.bool()  # (B,1,1,T_src)
        tgt_mask_bool = tgt_mask.bool()  # (B,1,T_tgt,T_tgt)

        logits = model(
            src_tokens=src_ids,
            tgt_tokens=tgt_ids,
            src_mask=src_pad_mask_bool,
            tgt_mask=tgt_mask_bool,
        )

        # 计算 loss —— 需要 shift 一位
        logits = logits[:, :-1, :].reshape(-1, tokenizer.vocab_size)
        labels = tgt_ids[:, 1:].reshape(-1)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch={epoch} loss={loss.item()}")
