"""Tests for src/lora.py on a tiny Qwen3. Run: python -m pytest tests/test_lora.py -q"""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lora import LoRALinear, apply_lora, trainable_parameters
from src.model import Qwen3Config, Qwen3ForCausalLM

CFG = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)


def tiny():
    torch.manual_seed(0)
    return Qwen3ForCausalLM(CFG).eval()


def ids():
    return torch.randint(0, CFG.vocab_size, (2, 10))


VARIANTS = [("lora", False), ("pissa", False), ("lora", True), ("pissa", True)]


@pytest.mark.parametrize("init,dora", VARIANTS)
def test_noop_at_init(init, dora):
    base, x = tiny(), ids()
    with torch.no_grad():
        ref = base(x)
    apply_lora(base, r=4, init=init, dora=dora)
    with torch.no_grad():
        out = base(x)
    assert torch.allclose(ref, out, atol=1e-4), (ref - out).abs().max()


@pytest.mark.parametrize("init,dora", VARIANTS)
def test_only_adapter_params_get_grad(init, dora):
    model = tiny()
    apply_lora(model, r=4, init=init, dora=dora)
    names = {n.split(".")[-1] for n, p in model.named_parameters() if p.requires_grad}
    assert names <= {"A", "B", "m"}
    loss = model(ids()).float().pow(2).mean()
    loss.backward()
    for p in trainable_parameters(model):
        assert p.grad is not None
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n


@pytest.mark.parametrize("init,dora", VARIANTS)
def test_repeated_training_steps(init, dora):
    """Regression: PiSSA init once left a grad_fn on the frozen weight, so step 2 crashed."""
    model = tiny().train()
    apply_lora(model, r=4, init=init, dora=dora)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert m.weight.grad_fn is None and not m.weight.requires_grad
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1e-3)
    losses = []
    for _ in range(3):
        loss = model(ids()).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert all(l == l for l in losses)


def test_b_zero_init_lora_grad_flows_to_B_first():
    model = tiny()
    apply_lora(model, r=4)
    model(ids()).float().pow(2).mean().backward()
    layer = next(m for m in model.modules() if isinstance(m, LoRALinear))
    assert layer.B.grad.abs().sum() > 0
    assert layer.A.grad.abs().sum() == 0  # B = 0 so A gets no gradient on step 1


@pytest.mark.parametrize("init,dora", VARIANTS)
def test_merge_equals_unmerged(init, dora):
    model = tiny()
    apply_lora(model, r=4, init=init, dora=dora)
    for p in trainable_parameters(model):  # make adapters non-trivial
        p.data += 0.05 * torch.randn_like(p)
    x = ids()
    with torch.no_grad():
        ref = model(x)
        for m in model.modules():
            if isinstance(m, LoRALinear):
                merged = m.merged_weight()
                m.dora = False
                m.weight.copy_(merged)
                m.A.zero_()
                m.B.zero_()
        out = model(x)
    assert torch.allclose(ref, out, atol=1e-4), (ref - out).abs().max()


def test_pissa_residual_plus_adapter_recovers_weight():
    torch.manual_seed(1)
    lin = nn.Linear(16, 24, bias=False)
    layer = LoRALinear(lin, r=4, alpha=4, init="pissa")
    assert torch.allclose(layer.weight + layer.delta(), lin.weight, atol=1e-5)
    # the adapter holds the principal part: the residual has smaller norm than the original
    assert layer.weight.norm() < lin.weight.norm()


def test_matches_peft_lora():
    peft = pytest.importorskip("peft")
    torch.manual_seed(2)
    lin = nn.Linear(16, 24, bias=False)
    mine = LoRALinear(copy.deepcopy(lin), r=4, alpha=8)
    holder = nn.Sequential(copy.deepcopy(lin))
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["0"], init_lora_weights=True)
    ref = peft.get_peft_model(holder, cfg)
    # copy my adapter values into peft's, then compare outputs
    mine.B.data.normal_(0, 0.1)
    with torch.no_grad():
        ref.base_model.model[0].lora_A["default"].weight.copy_(mine.A)
        ref.base_model.model[0].lora_B["default"].weight.copy_(mine.B)
    x = torch.randn(3, 16)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_dora_matches_peft():
    peft = pytest.importorskip("peft")
    torch.manual_seed(3)
    lin = nn.Linear(16, 24, bias=False)
    mine = LoRALinear(copy.deepcopy(lin), r=4, alpha=8, dora=True)
    holder = nn.Sequential(copy.deepcopy(lin))
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["0"], use_dora=True)
    ref = peft.get_peft_model(holder, cfg)
    layer = ref.base_model.model[0]
    mine.B.data.normal_(0, 0.1)
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(mine.A)
        layer.lora_B["default"].weight.copy_(mine.B)
        mag = layer.lora_magnitude_vector["default"]
        (mag.weight if hasattr(mag, "weight") else mag).copy_(mine.m)
    x = torch.randn(3, 16)
    assert torch.allclose(mine(x), ref(x), atol=1e-4), (mine(x) - ref(x)).abs().max()
