"""Greedy-generation accuracy on the commonsense test sets, using the from-scratch model."""
import re
from pathlib import Path

import torch

from .data import PROMPT, load_json

# label pattern per task, as in the LLM-Adapters evaluation scripts
PATTERNS = {
    "boolq": r"true|false",
    "piqa": r"solution1|solution2",
    "social_i_qa": r"answer1|answer2|answer3",
    "ARC-Challenge": r"answer1|answer2|answer3|answer4",
    "ARC-Easy": r"answer1|answer2|answer3|answer4",
    "openbookqa": r"answer1|answer2|answer3|answer4",
    "hellaswag": r"ending1|ending2|ending3|ending4",
    "winogrande": r"option1|option2",
}


SAMPLES = {}  # first few raw generations per task, saved with each run to sanity-check the scorer


def extract(task, text):
    m = re.search(PATTERNS[task], text.lower())
    return m.group(0) if m else None


@torch.no_grad()
def generate(model, tok, prompts, device, max_new=10, batch_size=16):
    """Greedy decoding without a KV cache. Right-padded: read logits at each row's last real token."""
    model.eval()
    outs = []
    for s in range(0, len(prompts), batch_size):
        chunk = [tok(p, add_special_tokens=False).input_ids for p in prompts[s : s + batch_size]]
        lens = [len(c) for c in chunk]
        L = max(lens) + max_new
        ids = torch.full((len(chunk), L), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
        for k, c in enumerate(chunk):
            ids[k, : len(c)] = torch.tensor(c)
        ids = ids.to(device)
        cur = torch.tensor(lens, device=device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        for _ in range(max_new):
            logits = model(ids[:, : int(cur.max())])
            nxt = logits[torch.arange(len(chunk)), cur - 1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, tok.eos_token_id), nxt)
            ids[torch.arange(len(chunk)), cur] = nxt
            done |= nxt == tok.eos_token_id
            cur = cur + 1
            if done.all():
                break
        for k, c in enumerate(chunk):
            outs.append(tok.decode(ids[k, lens[k] : int(cur[k])].tolist(), skip_special_tokens=True))
    return outs


def evaluate(model, tok, data_dir, device, n_per_task=200, tasks=None, seed=0):
    results = {}
    for task in tasks or PATTERNS:
        rows = load_json(Path(data_dir) / f"{task}.json", n=n_per_task, seed=seed)
        prompts = [PROMPT.format(instruction=r["instruction"]) for r in rows]
        preds = generate(model, tok, prompts, device)
        SAMPLES[task] = [(p, extract(task, p), r["answer"]) for p, r in list(zip(preds, rows))[:4]]
        ok = sum(extract(task, p) == r["answer"].strip().lower() for p, r in zip(preds, rows))
        results[task] = ok / len(rows)
    results["avg"] = sum(results.values()) / len(results)
    return results
