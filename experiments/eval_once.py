"""Evaluate ONCE-Net and reading-level baselines on held-out benchmarks.

Every method here reads the same cached isolated readings, so differences come from how the
readings are combined.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from once.core import Reading, once  # noqa: E402
from once.grid import collate, cond_indices, load_joint, load_readings, make_grid, match  # noqa: E402
from once.model import OnceNet  # noqa: E402

RUNS = os.environ.get("ONCE_RUNS", "D:/once_data/runs")
RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")


def ls(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def reading_baselines(g):
    """Answers from the grid alone (no learning)."""
    L, L0, cands = g["L"], g["L0"], g["cands"]
    C = len(cands)
    out = {}
    if C == 0:
        return {k: "I don't know" for k in ("closed_book", "majority", "replug", "poe", "once_rule")}
    out["closed_book"] = cands[int(np.argmax(L0[:C]))]
    full = np.stack([ls(r) for r in L]) if len(L) else np.zeros((0, C + 1))
    claims = full.argmax(1) if len(L) else np.zeros(0, int)
    votes = Counter(int(c) for c in claims if c < C)
    out["majority"] = cands[votes.most_common(1)[0][0]] if votes else out["closed_book"]
    if len(L):
        q = np.stack([ls(r[:C]) for r in L])
        out["replug"] = cands[int(np.argmax(np.exp(full[:, :C]).mean(0)))]
        out["poe"] = cands[int(np.argmax(q.sum(0)))]
        emb = g["H"] / np.linalg.norm(g["H"], axis=1, keepdims=True)
        r = Reading(L=np.concatenate([L0[None], L]), emb=emb)
        out["once_rule"] = cands[once(r, beta=1.0, M=3.0, tau=0.9)["pred"]]
    else:
        out["replug"] = out["poe"] = out["once_rule"] = out["closed_book"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--data", default="nq")
    ap.add_argument("--ckpt", default="once_s0")
    ap.add_argument("--conds", default="p0,p1,p3,p5")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    recs = load_readings(args.model, args.data)
    ck = torch.load(os.path.join(RUNS, args.model, args.ckpt, "best.pt"), map_location="cpu")
    a = ck["args"]
    net = OnceNet(hdim=recs[0]["H"].shape[-1], d=a["d"], layers=a["layers"],
                  redundancy=not a["no_rna"], differential=not a["no_diff"], dropout=a.get("dropout", 0.1),
                  neff=a.get("neff", False), joint=a.get("joint", False), isolated=not a.get("no_iso", False),
                  residual=a.get("residual", False), prior_source=a.get("prior_source", False),
                  neutral_id=a.get("neutral_id", False))
    net.load_state_dict(ck["state"])
    net.to(args.device).eval()

    jt = load_joint(args.model, args.data) if a.get("joint") else {}
    res = {}
    for p in args.conds.split(","):
        if a.get("joint"):
            grids = [make_grid(r, cond_indices(r, p, args.k), joint=jt[p][r["id"]][0]) for r in recs]
        else:
            grids = [make_grid(r, cond_indices(r, p, args.k)) for r in recs]
        preds = {}
        with torch.no_grad():
            for s in range(0, len(grids), 64):
                g = grids[s:s + 64]
                lo = net(collate(g, args.device))
                prob = torch.sigmoid(net.last_binary).cpu().numpy()
                for bi, (gi, pi, fi) in enumerate(zip(g, lo.argmax(-1).tolist(), lo[:, :-1].argmax(-1).tolist())):
                    preds.setdefault("once_net_abst", []).append(gi["cands"][pi] if pi < len(gi["cands"]) else "I don't know")
                    preds.setdefault("once_net", []).append(gi["cands"][fi] if fi < len(gi["cands"]) else "I don't know")
                    chosen = [c for ci, c in enumerate(gi["cands"]) if prob[bi, ci] > 0.5]
                    if not chosen and pi < len(gi["cands"]):
                        chosen = [gi["cands"][pi]]
                    preds.setdefault("once_net_set", []).append(", ".join(chosen) or "I don't know")
        for g in grids:
            for k, v in reading_baselines(g).items():
                preds.setdefault(k, []).append(v)
            if "LJ" in g:  # the LLM's own verdict on the joint context, read off the same candidates
                c = len(g["cands"])
                preds.setdefault("joint_verdict", []).append(g["cands"][int(np.argmax(g["LJ"][:c]))] if c else "I don't know")
        for m, pr in preds.items():
            acc = float(np.mean([match(x, r["golds"]) for x, r in zip(pr, recs)]))
            asr = float(np.mean([bool(r["target"]) and match(x, [r["target"]]) for x, r in zip(pr, recs)]))
            res[f"{m}|{p}"] = {"acc": acc, "asr": asr, "n": len(pr), "preds": pr}
            extra = ""
            if recs[0].get("wrong") is not None:  # RAMDocs strict exact match
                em = float(np.mean([all(match(x, [g]) for g in r["golds"]) and
                                    not any(match(x, [w]) for w in (r["wrong"] or [])) for x, r in zip(pr, recs)]))
                res[f"{m}|{p}"]["em"] = em
                res[f"{m}|{p}"]["wrong"] = float(np.mean([any(match(x, [w]) for w in (r["wrong"] or []))
                                                          for x, r in zip(pr, recs)]))
                extra = f" em {em:.3f}"
            print(f"{args.data} {p} {m:13s} acc {acc:.3f} asr {asr:.3f}{extra}", flush=True)
    os.makedirs(RES, exist_ok=True)
    tag = args.conds.replace(",", "-")
    json.dump(res, open(os.path.join(RES, f"once_{args.model}_{args.data}_{args.ckpt}_{tag}.json"), "w"))


if __name__ == "__main__":
    main()
