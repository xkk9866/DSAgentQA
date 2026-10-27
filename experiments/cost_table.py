"""Measured cost per question on the same GPU (RTX 4090 D, INT8 Qwen2.5-7B), from the timings logged during the runs.

Generation baselines log seconds per (benchmark, condition) cell; we report the median over cells of seconds / question.
CREDENCE's LLM cost is the isolated reading (one short generation + prefix-cached scoring per passage and for memory),
logged per 32-question chunk by once_read.py; its network cost is timed here. Timings come from a shared machine, so
they are indicative (medians over many cells), not a controlled benchmark.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES = "D:/once_data/results"
LOGS = "D:/once_data/logs"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "credence", "tables", "cost.tex")
M = "qwen2.5-7b-int8"
CALLS = {"vanilla": ("1", "24"), "instructrag": ("1", "320"), "astute": ("3", "960"), "madam": ("$\\le$33", "4{,}080"),
         "trustrag": ("1", "24"), "keyword": ("11", "264"), "decoding": ("11", "176")}
NAMES = {"vanilla": "Vanilla RAG", "instructrag": "InstructRAG-ICL", "astute": "Astute RAG", "madam": "MADAM-RAG",
         "trustrag": "TrustRAG", "keyword": "RobustRAG (keyword)", "decoding": "RobustRAG (decoding)"}


def main():
    per = {}
    for f in glob.glob(os.path.join(RES, f"baselines_{M}_*.json")):
        if "adv-" in f:
            continue
        for k, v in json.load(open(f)).items():
            m = k.split("|")[0]
            if v.get("n") and v.get("sec"):
                per.setdefault(m, []).append(v["sec"] / v["n"])
    chunk = []
    for f in glob.glob(os.path.join(LOGS, "**", "*.log"), recursive=True):
        if "mis" in os.path.basename(f):
            continue
        txt = open(f, encoding="utf8", errors="ignore").read()
        if "qwen2.5-7b-int8" not in txt and "fresh" not in txt and "adv" not in os.path.basename(f):
            continue
        prev = 0.0
        for m in re.finditer(r"chunk (\d+)/(\d+)\s+(\d+)s", txt):
            t = float(m.group(3))
            if t > prev:
                chunk.append((t - prev) / 32)
            prev = t if int(m.group(1)) < int(m.group(2)) else 0.0
    from credence_run import batch, grid
    from adaptive_eval import load_credence
    from once.grid import cond_indices, load_readings
    recs = load_readings(M, "prag-nq")
    gs = [grid(r, cond_indices(r, "p5")) for r in recs]
    dev = torch.device("cuda:0")
    net = load_credence(M, "rob", 0, recs[0]["H"].shape[-1], dev)
    with torch.no_grad():
        net(batch(gs[:8], dev))
        torch.cuda.synchronize()
        t0 = time.time()
        for g in gs:
            net(batch([g], dev))
        torch.cuda.synchronize()
    net_ms = 1000 * (time.time() - t0) / len(gs)
    rows = []
    for m in ["vanilla", "trustrag", "instructrag", "astute", "madam", "keyword", "decoding"]:
        if m in per:
            c, t = CALLS[m]
            rows.append(f"{NAMES[m]} & {c} & {t} & {np.median(per[m]):.2f} \\\\")
    rows.append("\\midrule")
    rows.append(f"\\textsc{{Credence}} (LLM readings) & 11 & 264 & {np.median(chunk):.2f} \\\\")
    rows.append(f"\\quad + inference network & -- & -- & {net_ms / 1000:.3f} \\\\")
    tab = ("\\begin{tabular}{lccc}\n\\toprule\nMethod & LLM calls & Max.\\ gen.\\ tokens & Sec./question \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(OUT, "w").write(tab)
    print(tab, len(chunk), "reading chunks;", net_ms, "ms network")


if __name__ == "__main__":
    main()
