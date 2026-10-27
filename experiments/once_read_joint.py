"""Joint readings: the frozen LLM reads all passages of a context together and scores the same
candidates as the isolated readings. This is the verdict that ONCE-Net audits.

Test sets: one joint reading per (question, condition). Training sets: each question gets a few
fixed contexts (different numbers of injected passages); the trainer samples among them.
"""

from __future__ import annotations

import argparse
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

JOINT_PROMPT = ("Passages:\n{ctx}\n\nAnswer the question using the passages. Reply with only the answer as a short "
                "phrase, without explanation. If the passages do not contain the answer, reply \"I don't know\"."
                "\n\nQuestion: {q}")
JOINT_PROMPT_MED = ("Passages:\n{ctx}\n\nBased on the passages, answer the research question with yes, no, or maybe. "
                    "Reply with one word.\n\nQuestion: {q}")


def joint_prompt(llm, q, docs, med=False):
    ctx = "\n\n".join(f"[{i + 1}] {d[:1200]}" for i, d in enumerate(docs))
    return llm.chat((JOINT_PROMPT_MED if med else JOINT_PROMPT).format(ctx=ctx, q=q))


def train_contexts(r, n_ctx=3, seed=0):
    """Fixed training contexts: one clean, one lightly and one heavily poisoned."""
    rng = random.Random(f"{r['id']}-{seed}")
    clean = [i for i, p in enumerate(r["poison"]) if not p]
    poison = [i for i, p in enumerate(r["poison"]) if p]
    out = []
    for lo, hi in [(0, 0), (1, 2), (3, 5)][:n_ctx]:
        k = rng.randint(4, 10)
        p = min(rng.randint(lo, hi), len(poison), k - 1)
        sel = rng.sample(poison, p) + rng.sample(clean, min(k - p, len(clean)))
        rng.shuffle(sel)
        out.append(sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--data", default="rqa")
    ap.add_argument("--conds", default="p0,p1,p3,p5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rbs", type=int, default=2)
    args = ap.parse_args()
    from once.llm import LLM
    llm = None
    for name in args.data.split(","):
        recs = load_readings(args.model, name)
        raw = {x["id"]: x for x in load_items(name)}
        out_dir = os.path.join(CACHE, args.model, "joint", name)
        os.makedirs(out_dir, exist_ok=True)
        training = name.startswith("train")
        conds = ["ctx"] if training else (["all"] if name == "ramdocs" else args.conds.split(","))
        jobs = []
        for cond in conds:
            if training:  # shards of 200 questions so an interrupted run resumes
                jobs += [(cond, recs[i:i + 200], os.path.join(out_dir, f"ctx_{i // 200:03d}.pkl"))
                         for i in range(0, len(recs), 200)]
            else:
                jobs.append((cond, recs, os.path.join(out_dir, f"{cond}.pkl")))
        all_recs = recs
        for cond, recs, path in jobs:
            if os.path.exists(path):
                continue
            if llm is None:
                llm = LLM(args.model, device=args.device)
            t0 = time.time()
            prompts, cands, keys = [], [], []
            for r in recs:
                x = raw[r["id"]]
                pas = passages(x)
                sels = train_contexts(r) if training else [cond_indices(r, cond)]
                for ci, sel in enumerate(sels):
                    prompts.append(joint_prompt(llm, r["question"], [pas[j] for j in sel], name.startswith("pubmedqa")))
                    cands.append(r["cands"] + [IDK])
                    keys.append((r["id"], ci, tuple(sel)))
            res = llm.read(prompts, cands, batch_size=args.rbs)
            store = {}
            for (rid, ci, sel), (sc, h, z) in zip(keys, res):
                store.setdefault(rid, []).append({"sel": list(sel), "L": sc.astype(np.float32),
                                                  "H": h.astype(np.float16), "Z": z[:-1].astype(np.float16)})
            pickle.dump(store, open(path, "wb"))
            print(f"{name} {cond}: {len(keys)} joint readings, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
