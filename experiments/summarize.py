"""Collect ONCE-Net and baseline results into the paper tables (markdown)."""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")
DATA = ["rqa", "nq", "prag-nq", "prag-hotpotqa", "prag-msmarco"]
NAMES = {"rqa": "RQA", "nq": "Open-NQ", "prag-nq": "PR-NQ", "prag-hotpotqa": "PR-HotpotQA",
         "prag-msmarco": "PR-MSMARCO"}
CONDS = ["p0", "p1", "p3", "p5"]


def load(model=os.environ.get("ONCE_LLM", "qwen2.5-7b-int8")):
    """table[method][data][cond] = list of {acc, asr, em}; several entries = seeds."""
    t = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(os.path.join(RES, "**", f"baselines_{model}_*.json"), recursive=True):
        d = os.path.basename(f)[len(f"baselines_{model}_"):-5].split("@")[0]
        for k, v in json.load(open(f)).items():
            m, c = k.split("|")
            t[m][d][c].append(v)
    sft = os.path.join(RES, f"sft_{model}.json")
    if os.path.exists(sft):  # keys look like "rqa|sft|p0"
        for k, v in json.load(open(sft)).items():
            d, m, c = k.split("|")
            t[m][d][c].append(v)
    for f in glob.glob(os.path.join(RES, f"once_{model}_*.json")):
        stem = os.path.basename(f)[len(f"once_{model}_"):-5]
        d = next(x for x in sorted(DATA + ["ramdocs"], key=len, reverse=True) if stem.startswith(x + "_"))
        ck = stem[len(d) + 1:].rsplit("_", 1)[0]
        group = ck.rsplit("_s", 1)[0]  # v3_s0, v3_s1 -> v3
        for k, v in json.load(open(f)).items():
            m, c = k.split("|")
            name = m if m not in ("once_net", "once_net_set", "once_net_abst") else f"{m}[{group}]"
            if m not in ("once_net", "once_net_set", "once_net_abst") and not ck.endswith("_s0"):
                continue  # reading baselines do not depend on the checkpoint; count them once
            if m not in ("once_net", "once_net_set", "once_net_abst") and group != os.environ.get("ONCE_OURS", "final"):
                continue
            t[name][d][c].append(v)
    return t


def cell(vals, key):
    xs = [v[key] for v in vals if key in v]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{100 * xs[0]:.1f}"
    return f"{100 * np.mean(xs):.1f}±{100 * np.std(xs):.1f}"


def main_table(t, methods, conds=CONDS):
    head = "| Method | " + " | ".join(f"{NAMES[d]} {c}" for d in DATA for c in conds) + " |"
    lines = [head, "|" + "---|" * (1 + len(DATA) * len(conds))]
    for m in methods:
        if m not in t:
            continue
        row = [f"{cell(t[m][d][c], 'acc')} / {cell(t[m][d][c], 'asr')}" for d in DATA for c in conds]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    t = load(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ONCE_LLM", "qwen2.5-7b-int8"))
    print(sorted(t.keys()))
    methods = ["closed_book", "vanilla", "instructrag", "astute", "trustrag", "keyword", "decoding", "madam",
               "majority", "replug", "poe", "once_rule", "once_net[v3]", "once_net[v3_norna]",
               "once_net[v3_nodiff]", "once_net[v3_nodup]"]
    print(main_table(t, methods))
    print("\nRAMDocs (acc / em)")
    for m in methods + ["once_net_set[v3]", "once_net_set[v3_norna]"]:
        if m in t and t[m]["ramdocs"]["all"]:
            print(m, cell(t[m]["ramdocs"]["all"], "acc"), cell(t[m]["ramdocs"]["all"], "em"))
