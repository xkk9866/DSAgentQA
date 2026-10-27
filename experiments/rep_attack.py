"""White-box attack in reading-state space: the threat model under which the pooling structure is meant to matter.

The attacker controls every injected passage, and therefore (within a budget) the reading state the LLM produces for
it. For each question we run targeted PGD on the states of the injected rows, within an L2 ball of radius
eps * ||h_s||, to maximize the log-probability of the attack target. eps = 0.15 is the gap between where injected
passages sit and the spread of clean passages (measured on training data). The same attack is run on every model, and
results are paired per question for the significance tests.

Output: D:/once_data/results/repattack_<model>.json  {model_tag: {data|cond|eps: {"correct": [...], "hit": [...]}}}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_eval import load_credence  # noqa: E402
from credence_run import batch, grid  # noqa: E402
from once.grid import cond_indices, load_readings, match  # noqa: E402

RUNS = "D:/once_data/runs"
RES = "D:/once_data/results"


def target_index(g, target):
    for i, c in enumerate(g["cands"]):
        if target and match(c, [target]):
            return i
    return -1


def attack(net, gs, dev, eps, steps=10, bs=64):
    correct, hit = [], []
    net.eval()
    for s in range(0, len(gs), bs):
        g = gs[s:s + bs]
        b = batch(g, dev)
        tgt = torch.tensor([g_["tidx"] for g_ in g], device=dev)
        has = tgt >= 0
        H = b["H"]
        m = b["pois"][..., None].float()
        radius = eps * H.norm(dim=-1, keepdim=True) * m
        delta = torch.zeros_like(H)
        if eps > 0 and has.any():
            for _ in range(steps):
                delta.requires_grad_(True)
                lo = net({**b, "H": H + delta})
                lp = lo.log_softmax(-1)
                obj = lp[torch.arange(len(g), device=dev), tgt.clamp(min=0)][has].sum()
                gr, = torch.autograd.grad(obj, delta)
                with torch.no_grad():
                    delta = delta + 0.25 * radius * gr / gr.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    n = delta.norm(dim=-1, keepdim=True)
                    delta = delta * torch.clamp(radius / n.clamp(min=1e-12), max=1.0)
        with torch.no_grad():
            lo = net({**b, "H": H + delta})
            pred = lo[:, :-1].argmax(-1).tolist()
        for gi, p in zip(g, pred):
            correct.append(int(p == gi["label"]))
            hit.append(int(gi["tidx"] >= 0 and p == gi["tidx"]))
    return correct, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--tags", default="rob,rob_direct,dsg,dsg_direct")
    ap.add_argument("--eps", default="0,0.15,0.3,0.5")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    sets = {"rqa": ["p1", "p5"], "prag-nq": ["p1", "p5"], "prag-hotpotqa": ["p1", "p5"], "prag-msmarco": ["p1", "p5"],
            "adv-rqa": ["a1", "a5"], "adv-prag-nq": ["a1", "a5"], "adv-prag-hotpotqa": ["a1", "a5"],
            "adv-prag-msmarco": ["a1", "a5"]}
    blind = {"a1": [10] + list(range(9)), "a5": [10, 13, 16, 19, 22] + list(range(5))}
    grids = {}
    for d, conds in sets.items():
        recs = load_readings(args.model, d)
        for c in conds:
            gs = []
            for r in recs:
                g = grid(r, blind[c] if c in blind else cond_indices(r, c))
                g["tidx"] = target_index(g, r["target"])
                gs.append(g)
            grids[(d, c)] = gs
    hdim = grids[("rqa", "p1")][0]["H"].shape[-1]
    path = os.path.join(RES, f"repattack_{args.model}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for tag in args.tags.split(","):
        for f in sorted(glob.glob(os.path.join(RUNS, args.model, f"{tag}_s*"))):
            seed = int(f.rsplit("_s", 1)[1])
            key = f"{tag}_s{seed}"
            if key in out:
                continue
            net = load_credence(args.model, tag, seed, hdim, dev)
            out[key] = {}
            for (d, c), gs in grids.items():
                for e in map(float, args.eps.split(",")):
                    cor, hit = attack(net, gs, dev, e)
                    out[key][f"{d}|{c}|{e}"] = {"correct": cor, "hit": hit}
            print(key, {e: round(float(np.mean([np.mean(v["correct"]) for k, v in out[key].items() if k.endswith(f"|{e}")])), 3)
                        for e in args.eps.split(",")}, flush=True)
            json.dump(out, open(path, "w"))


if __name__ == "__main__":
    main()
