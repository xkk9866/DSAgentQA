"""Query-adaptive text attack on MOTIVE with the reader in the loop.

For each question and budget t, the attacker adds disguised passages one genre at a time; within each genre it tries
the three variants and keeps the one that maximizes MOTIVE's probability of the attack target in the resulting
context. Every candidate context is read again by the reader as a whole (new joint reading), so the attack sees
exactly what MOTIVE sees. The objective is the final answer probability of the target (not a proxy such as passage
credibility). The attacked network is seed 0; other seeds receive the same contexts (transfer).

Writes the selections to results/motive_sel_<model>_<set>.json ({id: {m1, m3, m5}}) and the joint readings of the selected
contexts to cache/<model>/joint/<set>/m<t>.pkl, so that every defense is evaluated on the same contexts.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motive_attacks import SETS, load_motive, target_logp, tidx_of  # noqa: E402
from motive_run import batch, grid  # noqa: E402
from once.grid import CACHE, load_readings  # noqa: E402
from once_read import IDK, load_items, passages  # noqa: E402
from once_read_joint import joint_prompt  # noqa: E402

RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")


def entries(llm, recs, raw, sels, rbs):
    prompts = [joint_prompt(llm, r["question"], [passages(raw[r["id"]])[j] for j in s]) for r, s in zip(recs, sels)]
    out = llm.read(prompts, [r["cands"] + [IDK] for r in recs], batch_size=rbs)
    return [{"sel": list(s), "L": sc.astype(np.float32), "H": h.astype(np.float16), "Z": z[:-1].astype(np.float16)}
            for s, (sc, h, z) in zip(sels, out)]


@torch.no_grad()
def objective(net, recs, sels, ents, dev):
    gs = [grid(r, s, joint=e) for r, s, e in zip(recs, sels, ents)]
    obj = []
    for i in range(0, len(gs), 64):
        g, rr = gs[i:i + 64], recs[i:i + 64]
        b = batch(g, dev)
        ti = tidx_of(g, rr, dev)
        lp = target_logp(net, b, ti)
        obj += torch.where(ti >= 0, lp, torch.full_like(lp, -1e9)).tolist()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--budgets", default="1,3,5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rbs", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device(args.device)
    from once.llm import LLM
    llm = LLM(args.model, device=args.device)
    for d in args.sets.split(","):
        sel_path = os.path.join(RES, f"motive_sel_{args.model}_{d}.json")
        recs = load_readings(args.model, d)
        raw = {x["id"]: x for x in load_items(d)}
        net, _ = load_motive(args.model, args.tag, args.seed, recs[0]["H"].shape[-1], dev)
        for t in map(int, args.budgets.split(",")):
            out_path = os.path.join(CACHE, args.model, "joint", d, f"m{t}.pkl")
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            clean = list(range(10 - t))
            chosen = [[] for _ in recs]
            best_ent = [None] * len(recs)
            for g in range(t):
                scores, ents_all = [], []
                for v in range(3):
                    sels = [c + [10 + 3 * g + v] + clean for c in chosen]
                    ents = entries(llm, recs, raw, sels, args.rbs)
                    scores.append(objective(net, recs, sels, ents, dev))
                    ents_all.append(ents)
                S = np.array(scores)  # (3, n)
                pick = S.argmax(0)
                for i, v in enumerate(pick):
                    chosen[i] = chosen[i] + [10 + 3 * g + int(v)]
                    best_ent[i] = ents_all[int(v)][i]
                print(f"{d} t={t} genre {g}: mean best target logp {S.max(0)[S.max(0) > -1e8].mean():.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
            store = {r["id"]: [e] for r, e in zip(recs, best_ent)}
            pickle.dump(store, open(out_path + ".tmp", "wb"))
            os.replace(out_path + ".tmp", out_path)
            sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
            for r, c in zip(recs, chosen):
                sel.setdefault(r["id"], {})[f"m{t}"] = c + clean
            json.dump(sel, open(sel_path, "w"))


if __name__ == "__main__":
    main()
