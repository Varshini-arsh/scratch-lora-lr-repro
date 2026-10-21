"""Commonsense-15k data (LLM-Adapters format): prompt building, tokenisation, batching."""
import json
import random

import torch

PROMPT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


def load_json(path, n=None, seed=0):
    rows = json.load(open(path, encoding="utf-8"))
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def encode_train(row, tok, max_len):
    prompt = tok(PROMPT.format(instruction=row["instruction"]), add_special_tokens=False).input_ids
    answer = tok(row["output"], add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = (prompt + answer)[:max_len]
    labels = ([-100] * len(prompt) + answer)[:max_len]  # loss on the response only
    return ids, labels


def collate(batch, pad_id):
    """Right-pad. Causal attention means pad tokens never affect earlier positions."""
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, : len(i)] = torch.tensor(i)
        lab[k, : len(l)] = torch.tensor(l)
    return ids, lab


def batches(rows, tok, max_len, batch_size, seed, shuffle=True):
    enc = [encode_train(r, tok, max_len) for r in rows]
    order = list(range(len(enc)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for s in range(0, len(order), batch_size):
        yield collate([enc[i] for i in order[s : s + batch_size]], tok.pad_token_id or tok.eos_token_id)
