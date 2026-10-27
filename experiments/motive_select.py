"""Recipe selection on held-out training questions only (no test benchmark is read).

For each run tag (seed 0), evaluates its checkpoint on the validation contexts of motive_run.py and reports accuracy
on clean contexts, on attacked contexts, their balanced mean, and the target-hit rate on attacked contexts. The final
recipe is the one with the highest balanced accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motive_attacks import load_motive  # noqa: E402
from motive_run import grid, load_train, predict  # noqa: E402
from once.grid import match, random_indices  # noqa: E402


class A:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    a = A()
    a.model, a.train, a.disg = "qwen2.5-7b-int8", "train_v4,train-rtqa", True
    _, val = load_train(a)
    vrng = random.Random(123)
    ctx = []
    for r in val:
        for e in r["jctx"]:
            ctx.append((r, e["sel"], e))
        for _ in range(2):
            ctx.append((r, list(dict.fromkeys(random_indices(r, vrng))), None))
    out = {}
    for tag in args.tags.split(","):
        try:
            net, uj = load_motive(a.model, tag, 0, 512, torch.device(args.device))
        except FileNotFoundError:
            continue
        gs = [grid(r, s, joint=e if uj else None) for r, s, e in ctx]
        pr = predict(net, gs, torch.device(args.device))
        clean = np.array([not any(g["poison"]) for g in gs])
        corr = np.array([match(p["forced"], r["golds"]) for p, (r, _, _) in zip(pr, ctx)], float)
        hit = np.array([bool(r["target"]) and match(p["forced"], [r["target"]]) for p, (r, _, _) in zip(pr, ctx)], float)
        out[tag] = {"clean": corr[clean].mean(), "attacked": corr[~clean].mean(),
                    "balanced": 0.5 * (corr[clean].mean() + corr[~clean].mean()), "asr": hit[~clean].mean()}
        print(f"{tag:12s} clean {out[tag]['clean']:.3f} attacked {out[tag]['attacked']:.3f} "
              f"balanced {out[tag]['balanced']:.3f} asr {out[tag]['asr']:.3f}", flush=True)
    json.dump(out, open(os.path.join("D:/once_data/results", "motive_select.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
