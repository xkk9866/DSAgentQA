"""Summaries: Clean / Pois. per benchmark, Overall accuracy and ASR, flooding, RAMDocs, for CREDENCE runs and ONCE-Net.

usage: python experiments/credence_summary.py tag1 [tag2 ...]   (each tag averages over its seeds)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")
M = os.environ.get("ONCE_LLM", "qwen2.5-7b-int8")
DS = ["rqa", "nq", "prag-nq", "prag-hotpotqa", "prag-msmarco"]
P = ["p0", "p1", "p3", "p5"]
F = ["f1", "f5", "f10", "f20"]


def row(get):
    """get(d, cond) -> (acc, asr) or None."""
    cells, accs, asrs, fl = [], [], [], []
    for d in DS:
        v = [get(d, c) for c in P]
        if any(x is None for x in v):
            cells += [None, None]
            continue
        cells += [v[0][0], np.mean([x[0] for x in v[1:]])]
        accs += [x[0] for x in v]
        asrs += [x[1] for x in v[1:]]
        f = [get(d, c) for c in F]
        if all(x is not None for x in f):
            fl.append(f[-1][0])
    ok = len(accs) == 4 * len(DS)
    return cells, (np.mean(accs) if ok else None), (np.mean(asrs) if ok else None), (np.mean(fl) if len(fl) == 3 else None)


def fmt(x):
    return "  -- " if x is None else f"{100 * x:5.1f}"


def credence_getter(tag, key="credence"):
    files = sorted(glob.glob(os.path.join(RES, f"credence_{M}_{tag}_s*.json")))
    runs = [json.load(open(f)) for f in files]

    def get(d, c):
        v = [r[d][f"{key}|{c}"] for r in runs if d in r and f"{key}|{c}" in r[d]]
        return (np.mean([x["acc"] for x in v]), np.mean([x["asr"] for x in v])) if v else None
    return get, runs


def once_getter(group="v9", key="once_net"):
    def get(d, c):
        conds = "f1-f5-f10-f20" if c.startswith("f") else "p0-p1-p3-p5"
        fs = glob.glob(os.path.join(RES, f"once_{M}_{d}_{group}_s?_{conds}.json"))
        v = [json.load(open(f))[f"{key}|{c}"] for f in fs]
        v = [x for x in v if x]
        return (np.mean([x["acc"] for x in v]), np.mean([x["asr"] for x in v])) if v else None
    return get


def baseline_getter(method):
    def get(d, c):
        f = os.path.join(RES, f"baselines_{M}_{d}.json")
        if not os.path.exists(f):
            return None
        x = json.load(open(f)).get(f"{method}|{c}")
        return (x["acc"], x["asr"]) if x and x.get("n") else None
    return get


def main():
    hdr = "".join(f"{d[:10]:>13s}" for d in DS)
    print(f"{'method':24s}{hdr}   Overall  ASR   Flood20")
    lines = []
    for m in ["vanilla", "instructrag", "astute", "trustrag", "keyword", "decoding", "madam"]:
        lines.append((m, row(baseline_getter(m))))
    lines.append(("once-net(v9,3 seeds)", row(once_getter())))
    lines.append(("once-net nojoint", row(once_getter("v9_nojoint"))))
    for tag in sys.argv[1:]:
        g, runs = credence_getter(tag)
        lines.append((f"{tag}({len(runs)})", row(g)))
    for name, (cells, acc, asr, fl) in lines:
        cs = "".join(f"{fmt(cells[i])}/{fmt(cells[i + 1]).strip():>5s}  " for i in range(0, len(cells), 2))
        print(f"{name:24s}{cs} {fmt(acc)}  {fmt(asr)}  {fmt(fl)}")
    for tag in sys.argv[1:]:
        _, runs = credence_getter(tag)
        rd = [r["ramdocs"] for r in runs if "ramdocs" in r]
        if rd:
            em = np.mean([x["credence_set|all"]["em"] for x in rd])
            acc = np.mean([x["credence|all"]["acc"] for x in rd])
            print(f"RAMDocs {tag}: acc {100 * acc:.1f}  EM(set) {100 * em:.1f}")
        au = [r[d][f"diag|{c}"]["auroc_poison"] for r in runs for d in DS for c in ["p1", "p3", "p5"] if d in r]
        ap = [r[d][f"diag|{c}"]["auroc_prior"] for r in runs for d in DS for c in ["p0"] if d in r]
        print(f"credibility {tag}: AUROC poison {np.nanmean(au):.3f}  AUROC prior-correct {np.nanmean(ap):.3f}")


if __name__ == "__main__":
    main()
