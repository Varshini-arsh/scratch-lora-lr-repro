# scratch-lora-lr-repro

From-scratch PyTorch reproduction of **"Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning"**
(arXiv 2602.04998). The paper claims that once the learning rate is tuned, vanilla LoRA is within about
1-2% of nine LoRA variants.

Everything is written in plain PyTorch: the Qwen3-0.6B architecture, the LoRA layers, the training loop.
Hugging Face is used only as a reference in tests, and for downloading weights and tokenizer.

## Status

| Step | State |
|---|---|
| Qwen3-0.6B decoder from scratch (`src/model.py`) | done |
| Logit match against Hugging Face (`tests/test_logit_match.py`) | done, max abs diff 0.0 in fp32 on CPU |
| LoRA, PiSSA, DoRA layers + tests (`src/lora.py`, `tests/test_lora.py`) | done, 16 tests pass, outputs match `peft` for LoRA and DoRA |
| Data, training loop, evaluation (`src/data.py`, `src/train.py`, `src/evaluate.py`) | written; ran end to end on CPU (4 steps, loss 5.41 to 2.89); longer CPU runs hit the laptop's RAM limit |
| GPU smoke run on Colab T4 (8 steps, rank 32, fp16 autocast) | works end to end; caught and fixed a CPU/GPU device bug in `src/lora.py` that the CPU tests could not see |
| Sweep + summary scripts (`scripts/sweep.py`, `scripts/summarize.py`) | done; all 9 runs finished on a Colab T4, results in `results/` |
| Learning-rate sweep (3 variants x 3 learning rates, 1 seed) | done, see table below |
| More seeds, finer learning-rate grid, larger eval sets | next |

## First results

Setup: Qwen3-0.6B-Base, rank 32, alpha = rank, 2,000 training examples, 1 epoch, batch 16, fp16 autocast,
100 test samples per task. Average accuracy (%) over 8 commonsense tasks, seed 0.

| Variant | lr 1e-4 | lr 3e-4 | lr 1e-3 | best |
|---|---|---|---|---|
| LoRA  | 63.4 | 63.5 | 41.4 | 63.5 |
| PiSSA | 64.8 | 37.9 | 38.4 | 64.8 |
| DoRA  | 62.2 | 59.8 | 38.4 | 62.2 |

What this does and does not show:
- At a good learning rate the three variants land within about 2.6 points of each other, which is in the
  direction of the paper's claim. With 100 samples per task the noise is roughly +-5 points, so this is
  consistent with the claim, not a confirmation of it.
- The usable learning-rate range differs by variant: PiSSA collapses at 3e-4, where LoRA is still fine.
- LoRA at 1e-3 is unstable. An earlier run of the same setting on a different VM scored 57.1 (this run: 41.4).
- Limits: one seed, a coarse 3-point learning-rate grid, small evaluation sets, and hyperparameters
  (rank, batch size, epochs) that I have not checked against the paper's appendix.
| Write-up: my numbers vs the paper's | planned |

## Get the data

The training and test sets come from the LLM-Adapters repository and are not stored here.

```
mkdir -p data/eval
B=https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main
curl -L $B/ft-training_set/commonsense_15k.json -o data/commonsense_15k.json
for d in boolq piqa social_i_qa ARC-Easy ARC-Challenge openbookqa hellaswag winogrande; do
  curl -L $B/dataset/$d/test.json -o data/eval/$d.json
done
```

## Run the sweep

```
python scripts/sweep.py --amp --n_train 2000 --n_eval 100
python scripts/summarize.py results
```

These are the settings used for the results above, run on a Colab T4. Finished runs are skipped, so an
interrupted sweep can be restarted.

## Run the logit-match test

```
PYTHONPATH=. python tests/test_logit_match.py
```

It loads the reference first and frees it, then loads the from-scratch model, compares logits, and runs a
negative control (a perturbed weight must change the logits) so a pass is meaningful.

## Honest notes
- Reference paper and its official code are by other authors; this repo re-implements and re-runs a slice of it.
- Results will be reported as measured, including where they differ from the paper.
