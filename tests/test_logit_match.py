"""Check the from-scratch Qwen3 against Hugging Face on the same weights.

Run: python tests/test_logit_match.py
HF is used only as the reference here; src/model.py does not import it.
"""
import gc
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import Qwen3Config, Qwen3ForCausalLM

REPO = "Qwen/Qwen3-0.6B"


def main():
    path = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok("def fibonacci(n):\n    if n < 2:\n        return n", return_tensors="pt").input_ids

    # Load the reference first and free it, so both models are never in RAM together.
    ref = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        b = ref(ids).logits
    del ref
    gc.collect()

    mine = Qwen3ForCausalLM(Qwen3Config()).eval()
    state = {}
    for f in Path(path).glob("*.safetensors"):
        state.update(load_file(str(f)))
    missing, unexpected = mine.load_state_dict(state, strict=False)
    print("missing:", missing, "unexpected:", unexpected)

    with torch.no_grad():
        a = mine(ids)
    diff = (a - b).abs().max().item()
    print(f"max abs logit diff: {diff:.2e}")
    print("argmax match:", bool((a.argmax(-1) == b.argmax(-1)).all()))
    assert diff < 1e-3, "logits differ from the reference"
    # Negative control: a tiny weight change must be visible, otherwise the check proves nothing.
    with torch.no_grad():
        mine.model.layers[10].mlp.up_proj.weight.add_(1e-3)
        c = mine(ids)
    ctrl = (c - b).abs().max().item()
    print(f"control (perturbed weight) diff: {ctrl:.2e}")
    assert ctrl > 1e-4, "control did not change the logits; the test is not sensitive"
    print("PASS")


if __name__ == "__main__":
    main()
