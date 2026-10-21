"""Fine-tune the from-scratch Qwen3 with LoRA / PiSSA / DoRA, then evaluate. One run = one JSON.

Example (CPU smoke test):
  python -m src.train --model Qwen/Qwen3-0.6B --variant lora --lr 2e-4 --n_train 64 --epochs 1 \
      --batch_size 4 --max_len 128 --n_eval 8 --eval_tasks boolq --out results/smoke.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .data import batches, load_json
from .evaluate import SAMPLES, evaluate
from .lora import apply_lora, trainable_parameters
from .model import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def load_qwen3(repo, device):
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    cfg = Qwen3Config()
    # Build on the meta device and assign the checkpoint tensors: avoids allocating the
    # model twice, which matters on a 16 GB machine.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)
    state = {}
    for f in Path(path).glob("*.safetensors"):
        state.update(load_file(str(f)))
    model.load_state_dict(state, strict=False, assign=True)
    del state
    model.float()
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.to(device), AutoTokenizer.from_pretrained(path)


def lr_at(step, total, base_lr, warmup=0.03):
    w = max(1, int(total * warmup))
    if step < w:
        return base_lr * (step + 1) / w
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--variant", choices=["lora", "pissa", "dora"], default="lora")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=None, help="default: alpha = rank")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16, help="effective batch size")
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--eval_tasks", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (GPU only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_qwen3(a.model, device)
    apply_lora(model, r=a.rank, alpha=a.alpha, init="pissa" if a.variant == "pissa" else "lora",
               dora=a.variant == "dora")
    params = trainable_parameters(model)
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=a.amp and device == "cuda")

    rows = load_json(ROOT / "data" / "commonsense_15k.json", n=a.n_train, seed=a.seed)
    accum = max(1, a.batch_size // a.micro_batch)
    total = max(1, (len(rows) // a.batch_size) * a.epochs)
    print(f"{a.variant} lr={a.lr} rank={a.rank} trainable={n_params/1e6:.1f}M steps={total} device={device}")

    model.train()
    step, t0, losses = 0, time.time(), []
    for ep in range(a.epochs):
        buf = []
        for ids, lab in batches(rows, tok, a.max_len, a.micro_batch, a.seed + ep):
            buf.append((ids.to(device), lab.to(device)))
            if len(buf) < accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.lr)
            tokens = sum((l[:, 1:] != -100).sum().item() for _, l in buf)
            step_loss = 0.0
            for ids_b, lab_b in buf:
                with torch.autocast(device, dtype=torch.float16, enabled=a.amp and device == "cuda"):
                    logits = model(ids_b)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)), lab_b[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="sum") / max(tokens, 1)
                scaler.scale(loss).backward()
                step_loss += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            buf = []
            step += 1
            losses.append(step_loss)
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total} loss {step_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if step >= total:
                break

    acc = evaluate(model, tok, ROOT / "data" / "eval", device, n_per_task=a.n_eval,
                   tasks=a.eval_tasks, seed=a.seed)
    print("accuracy:", {k: round(v, 4) for k, v in acc.items()})
    out = dict(vars(a), trainable_params=n_params, final_loss=losses[-1] if losses else None,
               first_loss=losses[0] if losses else None, losses=losses, accuracy=acc,
               seconds=time.time() - t0, samples=SAMPLES)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
