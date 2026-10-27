"""Train ONCE-Net on NQ-train readings and evaluate on held-out benchmarks.

Model selection uses a held-out slice of the training questions only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from once.grid import collate, condition_indices, load_joint, load_readings, make_grid, random_indices  # noqa: E402
from once.model import OnceNet  # noqa: E402

RUNS = os.environ.get("ONCE_RUNS", "D:/once_data/runs")


def evaluate(net, recs, device, conds, k=10, bs=64):
    net.eval()
    out = {}
    for p in conds:
        grids = [make_grid(r, condition_indices(r, k, p)) for r in recs]
        correct = forced = abstain = 0
        with torch.no_grad():
            for s in range(0, len(grids), bs):
                g = grids[s:s + bs]
                logit = net(collate(g, device))
                pred = logit.argmax(-1).tolist()
                fpred = logit[:, :-1].argmax(-1).tolist()  # answer anyway: best candidate, NONE excluded
                for gi, pi, fi in zip(g, pred, fpred):
                    forced += fi == gi["label"]
                    if pi >= len(gi["cands"]):
                        abstain += 1
                        continue
                    correct += pi == gi["label"]
        out[f"p{p}"] = {"acc": correct / len(grids), "forced": forced / len(grids), "abstain": abstain / len(grids)}
    return out


def evaluate_ctx(net, recs, device, bs=64):
    """Validation on the stored training contexts (clean / light / heavy poisoning)."""
    net.eval()
    out = {}
    for ci, name in enumerate(["clean", "light", "heavy"]):
        grids = [make_grid(r, r["jctx"][ci]["sel"], joint=r["jctx"][ci]) for r in recs if len(r["jctx"]) > ci]
        forced = 0
        with torch.no_grad():
            for s in range(0, len(grids), bs):
                g = grids[s:s + bs]
                fpred = net(collate(g, device))[:, :-1].argmax(-1).tolist()
                forced += sum(fi == gi["label"] for fi, gi in zip(fpred, g))
        out[name] = {"forced": forced / max(1, len(grids))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--train", default="train")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--neff", action="store_true", help="effective-source count readout")
    ap.add_argument("--joint", action="store_true", help="audit the LLM's joint reading of the whole context")
    ap.add_argument("--no_iso", action="store_true", help="ablation: drop the isolated readings")
    ap.add_argument("--residual", action="store_true", help="score = joint verdict + learned correction")
    ap.add_argument("--prior_source", action="store_true", help="closed-book reading counted as one source")
    ap.add_argument("--neutral_id", action="store_true", help="candidate identity from passage readings only")
    ap.add_argument("--no_rna", action="store_true")
    ap.add_argument("--no_diff", action="store_true")
    ap.add_argument("--no_dup_aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="once")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)

    tr, val = [], []
    for name in args.train.split(","):  # e.g. "train,train-rtqa": static-knowledge + knowledge-update questions
        recs = load_readings(args.model, name)
        if args.joint:
            jt = load_joint(args.model, name).get("ctx", {})
            recs = [r for r in recs if r["id"] in jt]
            for r in recs:
                r["jctx"] = jt[r["id"]]
        random.Random(0).shuffle(recs)
        n_val = max(len(recs) // 10, min(200, len(recs) // 5))
        val += recs[:n_val]
        tr += recs[n_val:]
    print(f"train {len(tr)}  val {len(val)}", flush=True)

    net = OnceNet(hdim=tr[0]["H"].shape[-1], d=args.d, layers=args.layers,
                  redundancy=not args.no_rna, differential=not args.no_diff, dropout=args.dropout,
                  neff=args.neff, joint=args.joint, isolated=not args.no_iso,
                  residual=args.residual, prior_source=args.prior_source, neutral_id=args.neutral_id).to(dev)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    out_dir = os.path.join(RUNS, args.model, args.tag + f"_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    best, t0 = -1.0, time.time()
    for ep in range(args.epochs):
        net.train()
        order = list(range(len(tr)))
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order) - args.bs + 1, args.bs):
            grids = []
            for i in order[s:s + args.bs]:
                if args.joint:
                    e = rng.choice(tr[i]["jctx"])
                    grids.append(make_grid(tr[i], e["sel"], joint=e))
                    continue
                sel = random_indices(tr[i], rng)
                if args.no_dup_aug:
                    sel = list(dict.fromkeys(sel))
                grids.append(make_grid(tr[i], sel))
            b = collate(grids, dev)
            logit = net(b)
            C = b["cmask"].shape[1]
            tgt = torch.zeros_like(net.last_binary)
            tgt[torch.arange(len(grids)), b["y"].clamp(max=C - 1)] = (b["y"] < C).float()
            for gi, g in enumerate(grids):
                for ci in g.get("multi", []):
                    tgt[gi, ci] = 1.0
            bce = F.binary_cross_entropy_with_logits(net.last_binary, tgt, reduction="none")
            bce = (bce * b["cmask"]).sum() / b["cmask"].sum()
            loss = F.cross_entropy(logit, b["y"]) + 0.5 * bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        res = evaluate_ctx(net, val, dev) if args.joint else evaluate(net, val, dev, conds=(0, 1, 3, 5))
        score = float(np.mean([v["forced"] for v in res.values()]))
        print(f"ep {ep} loss {tot / max(1, len(order) // args.bs):.4f} val {json.dumps(res)} "
              f"mean {score:.4f} {time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            torch.save({"state": net.state_dict(), "args": vars(args)}, os.path.join(out_dir, "best.pt"))
    print("best val", best)


if __name__ == "__main__":
    main()
