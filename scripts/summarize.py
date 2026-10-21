"""Print best average accuracy per variant and the accuracy at each learning rate."""
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results")
table = defaultdict(dict)
for f in sorted(root.glob("*_s*.json")):
    r = json.load(open(f))
    table[r["variant"]].setdefault(r["lr"], []).append(r["accuracy"]["avg"])

lrs = sorted({lr for v in table.values() for lr in v})
print("variant  " + "  ".join(f"{lr:>8g}" for lr in lrs) + "   best")
for variant, d in table.items():
    cells = [f"{sum(d[lr]) / len(d[lr]) * 100:8.1f}" if lr in d else "       -" for lr in lrs]
    best = max(sum(v) / len(v) for v in d.values()) * 100
    print(f"{variant:8} " + "  ".join(cells) + f"   {best:5.1f}")
