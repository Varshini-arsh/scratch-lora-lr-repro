"""Learning-rate sweep over LoRA variants. Skips runs whose result file already exists,
so a Kaggle session that is cut off can simply be rerun.

  python scripts/sweep.py --variants lora pissa dora --lrs 1e-4 3e-4 1e-3 --n_train 2000 --n_eval 100 --amp
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["lora", "pissa", "dora"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out_dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    for seed in a.seeds:
        for variant in a.variants:
            for lr in a.lrs:
                out = Path(a.out_dir) / f"{variant}_r{a.rank}_lr{lr:g}_s{seed}.json"
                if out.exists():
                    print("skip", out.name)
                    continue
                cmd = [sys.executable, "-m", "src.train", "--variant", variant, "--lr", str(lr),
                       "--rank", str(a.rank), "--epochs", str(a.epochs), "--n_eval", str(a.n_eval),
                       "--seed", str(seed), "--out", str(out)]
                if a.n_train:
                    cmd += ["--n_train", str(a.n_train)]
                if a.amp:
                    cmd.append("--amp")
                print(" ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
