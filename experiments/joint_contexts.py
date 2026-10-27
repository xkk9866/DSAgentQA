"""Joint readings for contexts that once_read_joint.py does not cover.

The frozen reader reads a whole context and scores the candidates of the isolated readings (same prompt and output
format as once_read_joint.py). Covered here:

  test sets      conditions p<t> (top-10 with t injected), f<m> (flooding), a<t> (blind disguised attack, adaptive
                 sets) and any condition stored in a selection file ({id: {cond: [passage indices]}})
  disguised      training questions with disguised poison (cache <data>-disg): two fixed contexts per question,
                 written to joint/<data>-disg/ctx_*.pkl like the standard training contexts

Output format is identical to once_read_joint.py, so once.grid.load_joint reads it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from once.grid import CACHE, cond_indices, load_readings  # noqa: E402
from once_read import IDK, load_items, passages  # noqa: E402
from once_read_joint import joint_prompt  # noqa: E402


def blind(t):
    """Adaptive sets: passages 0-9 clean, 10 + 3g + v is genre g, variant v; the blind attacker takes variant 0."""
    return [10 + 3 * g for g in range(t)] + list(range(10 - t))


def test_sel(r, cond, sel_file):
    if sel_file and r["id"] in sel_file and cond in sel_file[r["id"]]:
        return sel_file[r["id"]][cond]
    if cond.startswith("a"):
        return blind(int(cond[1:]))
    return cond_indices(r, cond)


def disg_contexts(r, seed=0):
    """Two fixed contexts: a light one with 1-2 disguised passages, a heavy one with 3-5 mixed disguised/standard."""
    rng = random.Random(f"{r['id']}-disg-{seed}")
    clean = [i for i, p in enumerate(r["poison"]) if not p]
    n_std = r["n_pass"] - len(clean) - len(r["disguised"])
    std = [i for i, p in enumerate(r["poison"]) if p][:n_std]
    dsg = list(range(r["n_pass"] - len(r["disguised"]), r["n_pass"]))
    out = []
    # light: 1-2 disguised passages; heavy: 3-5 injected passages, at least two of them disguised
    for lo, hi, n_dsg_min in [(1, 2, None), (3, 5, 2)]:
        k = rng.randint(5, 10)
        p = min(rng.randint(lo, hi), k - 1)
        if n_dsg_min is None:
            inj = rng.sample(dsg, min(p, len(dsg)))
        else:
            nd = rng.randint(min(n_dsg_min, p), min(p, len(dsg)))
            inj = rng.sample(dsg, nd) + rng.sample(std, min(p - nd, len(std)))
        sel = inj + rng.sample(clean, min(k - len(inj), len(clean)))
        rng.shuffle(sel)
        out.append(sel)
    return out


def run(llm_box, model, device, jobs, rbs):
    from once.llm import LLM
    for path, prompts, cands, keys in jobs:
        if os.path.exists(path) or not prompts:
            continue
        if llm_box[0] is None:
            llm_box[0] = LLM(model, device=device)
        llm = llm_box[0]
        t0 = time.time()
        res = llm.read(prompts, cands, batch_size=rbs)
        store = {}
        for (rid, sel), (sc, h, z) in zip(keys, res):
            store.setdefault(rid, []).append({"sel": list(sel), "L": sc.astype(np.float32),
                                              "H": h.astype(np.float16), "Z": z[:-1].astype(np.float16)})
        tmp = path + ".tmp"
        pickle.dump(store, open(tmp, "wb"))
        os.replace(tmp, path)
        print(f"{path}: {len(keys)} joint readings, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--data", required=True)
    ap.add_argument("--conds", default="p0,p1,p3,p5")
    ap.add_argument("--sel", default="", help="selection file for conditions chosen by an adaptive attacker")
    ap.add_argument("--tag", default="", help="suffix of the output file name (cond_<tag>.pkl)")
    ap.add_argument("--disg", action="store_true", help="disguised training contexts of <data>-disg")
    ap.add_argument("--shard", default="0/1", help="i/n: this worker handles every n-th job")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rbs", type=int, default=2)
    args = ap.parse_args()
    sel_file = json.load(open(args.sel)) if args.sel else {}
    si, sn = map(int, args.shard.split("/"))
    llm_box = [None]
    for name in args.data.split(","):
        raw = {x["id"]: x for x in load_items(name)}
        if args.disg:
            recs = load_readings(args.model, name + "-disg")
            recs = [r for r in recs if r.get("disguised")]
            out_dir = os.path.join(CACHE, args.model, "joint", name + "-disg")
            os.makedirs(out_dir, exist_ok=True)
            jobs = []
            for i in range(0, len(recs), 100):
                if (i // 100) % sn != si:
                    continue
                chunk = recs[i:i + 100]
                prompts, cands, keys = [], [], []
                for r in chunk:
                    pas = passages(raw[r["id"]]) + list(r["disguised"])
                    for sel in disg_contexts(r):
                        prompts.append((r["question"], [pas[j] for j in sel]))
                        cands.append(r["cands"] + [IDK])
                        keys.append((r["id"], tuple(sel)))
                jobs.append((os.path.join(out_dir, f"ctx_{i // 100:03d}.pkl"), prompts, cands, keys))
        else:
            recs = load_readings(args.model, name)
            out_dir = os.path.join(CACHE, args.model, "joint", name)
            os.makedirs(out_dir, exist_ok=True)
            jobs = []
            for ci, cond in enumerate(args.conds.split(",")):
                if ci % sn != si:
                    continue
                prompts, cands, keys = [], [], []
                for r in recs:
                    sel = test_sel(r, cond, sel_file)
                    pas = passages(raw[r["id"]])
                    prompts.append((r["question"], [pas[j] for j in sel]))
                    cands.append(r["cands"] + [IDK])
                    keys.append((r["id"], tuple(sel)))
                fn = f"{cond}{'_' + args.tag if args.tag else ''}.pkl"
                jobs.append((os.path.join(out_dir, fn), prompts, cands, keys))
        # prompts are built lazily once the reader (and its chat template) is loaded
        lazy = []
        for path, pq, cands, keys in jobs:
            if os.path.exists(path):
                continue
            if llm_box[0] is None:
                from once.llm import LLM
                llm_box[0] = LLM(args.model, device=args.device)
            ps = [joint_prompt(llm_box[0], q, docs) for q, docs in pq]
            lazy.append((path, ps, cands, keys))
        run(llm_box, args.model, args.device, lazy, args.rbs)


if __name__ == "__main__":
    main()
