"""LoRA, PiSSA and DoRA written from scratch in plain PyTorch (no peft).

Weight convention follows nn.Linear: W has shape (out, in).
    LoRA:  y = x W^T + s * x A^T B^T,  A: (r, in), B: (out, r), s = alpha / r
    PiSSA: same layer, but A and B start from the top-r singular directions of W and the
           frozen weight holds the residual, so the layer output is unchanged at init.
    DoRA:  W' = m * (W + s B A) / ||W + s B A||  with the norm taken per output row and
           detached from the graph, and m a learned magnitude vector.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, init: str = "lora", dora: bool = False):
        super().__init__()
        assert base.bias is None, "bias is not supported"
        assert init in ("lora", "pissa")
        out_f, in_f = base.weight.shape
        self.r, self.scale, self.dora = r, alpha / r, dora
        self.register_buffer("weight", base.weight.detach().clone())  # frozen
        opts = dict(dtype=base.weight.dtype, device=base.weight.device)
        self.A = nn.Parameter(torch.empty(r, in_f, **opts))
        self.B = nn.Parameter(torch.zeros(out_f, r, **opts))

        # Initialise under no_grad: an in-place update of the frozen weight with A and B in the
        # graph would give it a grad_fn, and the second backward pass would then fail.
        with torch.no_grad():
            if init == "lora":
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            else:
                U, S, Vh = torch.linalg.svd(self.weight.float(), full_matrices=False)
                root = torch.sqrt(S[:r] / self.scale)
                self.A.copy_(root[:, None] * Vh[:r])
                self.B.copy_(U[:, :r] * root[None, :])
                self.weight -= self.scale * (self.B @ self.A)  # residual

            if dora:
                self.m = nn.Parameter((self.weight + self.scale * (self.B @ self.A)).norm(dim=1))

    def delta(self):
        return self.scale * (self.B @ self.A)

    def effective_weight(self):
        w = self.weight + self.delta()
        if self.dora:
            w = self.m[:, None] * w / w.norm(dim=1, keepdim=True).detach()
        return w

    def forward(self, x):
        if self.dora:
            return F.linear(x, self.effective_weight())
        return F.linear(x, self.weight) + self.scale * F.linear(F.linear(x, self.A), self.B)

    @torch.no_grad()
    def merged_weight(self):
        return self.effective_weight()


def apply_lora(model, r=8, alpha=None, init="lora", dora=False, targets=TARGETS):
    """Freeze the model and replace target Linear layers with adapters. Returns the adapter list."""
    alpha = r if alpha is None else alpha
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = []
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                new = LoRALinear(child, r, alpha, init=init, dora=dora)
                setattr(module, name, new)
                adapters.append(new)
    return adapters


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
