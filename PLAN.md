# Plan

Goal: reproduce a slice of "Learning Rate Matters" with everything written from scratch.

## Stages
1. Model: Qwen3-0.6B decoder, logits match HF. (done)
2. LoRA layer (`src/lora.py`): W + (alpha/r) * B A. Tests on a toy model:
   - adapter is a no-op at init (B = 0)
   - gradients reach only A and B
   - merge gives the same output as the unmerged layer
   - matches `peft` on the same seed (one test only)
3. Variants: DoRA, PiSSA (initialization from SVD of W).
4. Data + training loop: one task (commonsense 15k, or CodeFeedback subset), fixed rank, alpha = r.
5. Sweep on Kaggle: learning rate 1e-6 to 1e-3 (log scale), vanilla vs variants, 1-2 seeds.
6. Write-up: table of best accuracy per method, learning-rate curves, comparison with the paper.

## Constraints
- No local GPU. CPU for development, Kaggle for runs.
- T4 has no good bf16: use fp16 with care or fp32, and note it in the write-up.
- Use a data subset and short runs so the sweep fits the weekly Kaggle quota.

## Open items
- Read the paper's hardware appendix and the full hyperparameter table.
- Check the official repo's license and read its LoRA code only after writing mine.
