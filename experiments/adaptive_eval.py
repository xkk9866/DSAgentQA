"""Adaptive attacks against CREDENCE (sets built by experiments/build_adaptive.py and read by once_read.py).

Passages 0-9 are clean; passage 10 + 3g + v is genre g, variant v (5 genres x 3 variants).
  a<t>: blind adaptive attacker, t genres, variant 0 each (disguised, mutually dissimilar, no question restatement).
  o<t>: query-access adaptive attacker: for each of t genres it picks, greedily, the variant to which the defense
        (CREDENCE, seed 0) assigns the highest credibility given the context built so far. The chosen contexts are then
        evaluated on all seeds (seed 0 is the white-box case) and on every baseline.
Writes D:/once_data/results/adaptive_<model>.json and the selections D:/once_data/results/adaptive_sel_<model>.json
(used by run_baselines.py through ONCE_SEL).
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
from credence_run import batch, grid, predict  # noqa: E402
from eval_once import reading_baselines  # noqa: E402
from once.credence import Credence  # noqa: E402
from once.grid import collate, load_readings, make_grid, match  # noqa: E402
from once.model import OnceNet  # noqa: E402

RUNS = "D:/once_data/runs"
RES = "D:/once_data/results"
T = [1, 3, 5]


def load_credence(model, tag, seed, hdim, dev):
    ck = torch.load(os.path.join(RUNS, model, f"{tag}_s{seed}", "best.pt"), map_location=dev)
    a = ck["args"]
    net = Credence(hdim=hdim, d=a["d"], layers=a["layers"], iters=a["iters"], dropout=a["dropout"],
                   robust=not a["no_robust"], learn_cred=not a["no_cred"], use_joint=not a["no_joint"],
                   use_prior=not a["no_prior"], residual=a["residual"], dep=not a.get("no_dep", False),
                   use_z=not a.get("no_z", False), use_h=not a.get("no_h", False), direct=a.get("direct", False)).to(dev)
    net.load_state_dict(ck["state"])
    return net.eval()


def blind(t):
    return [10 + 3 * g for g in range(t)] + list(range(10 - t))


@torch.no_grad()
def oracle(net, r, t, dev):
    """Greedy per genre: pick the variant whose credibility the defense rates highest in the current context."""
    chosen = []
    for g in range(t):
        best, bv = -1.0, 0
        for v in range(3):
            j = 10 + 3 * g + v
            sel = chosen + [j] + list(range(10 - t))
            p = predict(net, [grid(r, sel)], dev)[0]
            c = p["cred_pass"][len(chosen)]
            if c > best:
                best, bv = c, j
        chosen.append(bv)
    return chosen + list(range(10 - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--tag", default="min_jz")
    ap.add_argument("--extra", default="direct", help="other CREDENCE-style runs to evaluate on the same contexts")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    sets = ["adv-rqa", "adv-prag-nq", "adv-prag-hotpotqa", "adv-prag-msmarco"]
    data = {d: load_readings(args.model, d) for d in sets}
    hdim = next(iter(data.values()))[0]["H"].shape[-1]
    seeds = sorted(int(f.split("_s")[-1]) for f in glob.glob(os.path.join(RUNS, args.model, f"{args.tag}_s*")))
    attacker = load_credence(args.model, args.tag, seeds[0], hdim, dev)
    sel = {}
    for d, recs in data.items():
        for r in recs:
            s = {f"p0": list(range(10))}
            for t in T:
                s[f"a{t}"] = blind(t)
                s[f"o{t}"] = oracle(attacker, r, t, dev)
            sel[r["id"]] = s
        print("selected", d, flush=True)
    json.dump(sel, open(os.path.join(RES, f"adaptive_sel_{args.model}_{args.tag}.json"), "w"))
    conds = ["p0"] + [f"a{t}" for t in T] + [f"o{t}" for t in T]
    res = {}
    models = [(f"credence_s{s}", load_credence(args.model, args.tag, s, hdim, dev)) for s in seeds]
    for tag in [x for x in args.extra.split(",") if x]:
        for s in sorted(int(f.split("_s")[-1]) for f in glob.glob(os.path.join(RUNS, args.model, f"{tag}_s*"))):
            models.append((f"{tag}_s{s}", load_credence(args.model, tag, s, hdim, dev)))
    once = None
    ck = os.path.join(RUNS, args.model, "v9_nojoint_s0", "best.pt")
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev)
        a = c["args"]
        once = OnceNet(hdim=hdim, d=a["d"], layers=a["layers"], redundancy=not a["no_rna"], differential=not a["no_diff"],
                       dropout=a.get("dropout", 0.1), neff=a.get("neff", False), joint=False, isolated=True,
                       residual=a.get("residual", False), prior_source=a.get("prior_source", False),
                       neutral_id=a.get("neutral_id", False)).to(dev)
        once.load_state_dict(c["state"])
        once.eval()
    for d, recs in data.items():
        res[d] = {}
        for c in conds:
            gs = [grid(r, sel[r["id"]][c]) for r in recs]
            preds = {}
            for name, net in models:
                preds[name] = [p["forced"] for p in predict(net, gs, dev)]
            for g in gs:
                for k, v in reading_baselines(g).items():
                    if k in ("majority", "replug", "closed_book"):
                        preds.setdefault(k, []).append(v)
            if once is not None:
                out = []
                with torch.no_grad():
                    for s in range(0, len(gs), 64):
                        gg = [make_grid(r, sel[r["id"]][c]) for r in recs[s:s + 64]]
                        lo = once(collate(gg, dev))
                        out += [g["cands"][i] if i < len(g["cands"]) else "I don't know"
                                for g, i in zip(gg, lo[:, :-1].argmax(-1).tolist())]
                preds["once_net"] = out
            for m, pr in preds.items():
                acc = float(np.mean([match(x, r["golds"]) for x, r in zip(pr, recs)]))
                asr = float(np.mean([match(x, [r["target"]]) for x, r in zip(pr, recs)]))
                res[d][f"{m}|{c}"] = {"acc": acc, "asr": asr, "n": len(pr), "preds": pr}
            print(d, c, {m: round(res[d][f"{m}|{c}"]["acc"], 3) for m in preds}, flush=True)
    json.dump(res, open(os.path.join(RES, f"adaptive_{args.model}_{args.tag}.json"), "w"))


if __name__ == "__main__":
    main()
