"""Baselines computed from cached readings (no new generation).

revprag   RevPRAG-style activation detector (Tan et al., 2025): a triplet-trained embedding of the reader's state at
          the last input token of the whole-context prompt, classified by the nearest labelled support example as
          a poisoned or a correct response. On a flag the system either abstains (revprag_abst) or re-answers from
          the reader's memory (revprag_mem); otherwise it returns the Vanilla RAG answer. Training contexts are the
          cached joint readings of the training questions (label: the joint verdict is the attack target). Our
          cache keeps one layer (3/4 depth, 512-d projection) instead of all layers.
rarag     RA-RAG aggregation (Hwang et al., 2025): per-passage answers, an alignment filter (NLI entailment of the
          question-answer pair by the passage, replacing AlignScore), semantic clustering, weighted majority vote.
          Passages have no persistent source identity across questions here, so every source gets the same
          reliability weight and RA-RAG's offline reliability estimation has nothing to estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from once.core import clean_answer, is_idk, normalize  # noqa: E402
from once.grid import cond_indices, load_joint, load_readings, match  # noqa: E402

RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")
TESTS = {"rqa": "p0,p1,p3,p5,f1,f5,f10,f20", "nq": "p0,p1,p3,p5", "prag-nq": "p0,p1,p3,p5,f1,f5,f10,f20",
         "prag-hotpotqa": "p0,p1,p3,p5", "prag-msmarco": "p0,p1,p3,p5,f1,f5,f10,f20", "ramdocs": "all",
         "fresh-rqa": "p0,p1,p3,p5"}


def blind(t):
    return [10 + 3 * g for g in range(t)] + list(range(10 - t))


def sel_of(r, c, sel_file):
    if sel_file and r["id"] in sel_file and c in sel_file[r["id"]]:
        return sel_file[r["id"]][c]
    if c.startswith("a"):
        return blind(int(c[1:]))
    return cond_indices(r, c)


class Embed(nn.Module):
    def __init__(self, h=512, d=128):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 256), nn.GELU(), nn.Linear(256, d))

    def forward(self, x):
        return nn.functional.normalize(self.f(x), dim=-1)


def train_revprag(model, dev, seed=0):
    X, Y = [], []
    for name in ("train_v4", "train-rtqa"):
        recs = {r["id"]: r for r in load_readings(model, name)}
        jt = load_joint(model, name).get("ctx", {})
        for rid, es in jt.items():
            r = recs.get(rid)
            if r is None or not r.get("target"):
                continue
            for e in es:
                C = len(r["cands"])
                if C == 0:
                    continue
                ans = r["cands"][int(np.argmax(e["L"][:C]))]
                pois = match(ans, [r["target"]])
                ok = match(ans, r["golds"])
                if pois or ok:  # RevPRAG labels poisoned vs correct responses
                    X.append(e["H"].astype(np.float32))
                    Y.append(int(pois))
    X, Y = torch.tensor(np.stack(X)), torch.tensor(Y)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_sup = len(X) // 10
    sup, tr = perm[:n_sup], perm[n_sup:]
    net = Embed(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xd, Yd = X.to(dev), Y.to(dev)
    rng = random.Random(seed)
    pos_idx = {c: [int(i) for i in tr if int(Y[i]) == c] for c in (0, 1)}
    for step in range(3000):
        a = [rng.choice(tr.tolist()) for _ in range(256)]
        p = [rng.choice(pos_idx[int(Y[i])]) for i in a]
        n = [rng.choice(pos_idx[1 - int(Y[i])]) for i in a]
        za, zp, zn = net(Xd[a]), net(Xd[p]), net(Xd[n])
        loss = nn.functional.triplet_margin_loss(za, zp, zn, margin=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        S = net(Xd[sup.to(dev)])
    return net, S, Yd[sup.to(dev)], {"n_train": len(tr), "n_support": n_sup, "pos_rate": float(Y.float().mean())}


@torch.no_grad()
def revprag_flag(net, S, SY, H, dev):
    z = net(torch.tensor(np.stack(H), dtype=torch.float32, device=dev))
    nn_idx = (z @ S.T).argmax(1)
    return SY[nn_idx].cpu().numpy().astype(bool)


class NLI:
    def __init__(self, dev):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder("cross-encoder/nli-deberta-v3-base", device=str(dev))
        self.cache = {}

    def entail(self, pairs):
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        if todo:
            lo = self.m.predict(todo, batch_size=64, apply_softmax=True)  # contradiction, entailment, neutral
            for p, l in zip(todo, lo):
                self.cache[p] = float(l[1])
        return [self.cache[p] for p in pairs]


def rarag_answer(r, sel, texts, nli, tau=0.5):
    """Filtered answers -> clusters (normalized string equality, merged by mutual NLI entailment) -> vote."""
    ans = [clean_answer(r["iso"][j]) for j in sel]
    pairs, idx = [], []
    for k, (j, a) in enumerate(zip(sel, ans)):
        if not is_idk(a):
            pairs.append((texts[j][:1500], f"{r['question']} {a}"))
            idx.append(k)
    keep = [ans[k] for k, e in zip(idx, nli.entail(pairs) if pairs else []) if e >= tau]
    if not keep:
        return "I don't know"
    clusters = []
    for a in keep:
        for c in clusters:
            if normalize(a) == normalize(c[0]):
                c.append(a)
                break
        else:
            clusters.append([a])
    if len(clusters) > 1:  # merge clusters whose representatives entail each other
        reps = [c[0] for c in clusters]
        q = r["question"]
        merged = list(range(len(reps)))
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                e1, e2 = nli.entail([(f"{q} {reps[i]}", f"{q} {reps[j]}"), (f"{q} {reps[j]}", f"{q} {reps[i]}")])
                if e1 > 0.5 and e2 > 0.5:
                    merged[j] = merged[i]
        tot = {}
        for i, c in enumerate(clusters):
            tot.setdefault(merged[i], []).extend(c)
        clusters = list(tot.values())
    return max(clusters, key=len)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--data", default=",".join(TESTS))
    ap.add_argument("--conds", default="")
    ap.add_argument("--sel", default="")
    ap.add_argument("--methods", default="revprag,rarag")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    from once_read import load_items, passages
    sel_file = json.load(open(args.sel)) if args.sel else {}
    meths = args.methods.split(",")
    rev = train_revprag(args.model, dev) if "revprag" in meths else None
    nli = NLI(dev) if "rarag" in meths else None
    for d in args.data.split(","):
        recs = load_readings(args.model, d)
        raw = {x["id"]: x for x in load_items(d)}
        jt = load_joint(args.model, d)
        van_path = os.path.join(RES, f"baselines_{args.model}_{d}.json")
        van = json.load(open(van_path)) if os.path.exists(van_path) else {}
        path = os.path.join(RES, f"baselines_cached_{args.model}_{d}.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        conds = args.conds.split(",") if args.conds else TESTS.get(d, "p0,a1,a3,a5").split(",")
        for c in conds:
            preds = {}
            if rev is not None and f"vanilla|{c}" in van:
                H, ok = [], []
                for r in recs:
                    e = jt.get(c, {}).get(r["id"], [None])[0]
                    ok.append(e is not None)
                    H.append(e["H"].astype(np.float32) if e is not None else np.zeros(512, np.float32))
                if all(ok):
                    flag = revprag_flag(rev[0], rev[1], rev[2], H, dev)
                    vp = van[f"vanilla|{c}"]["preds"]
                    preds["revprag_abst"] = ["I don't know" if f else v for f, v in zip(flag, vp)]
                    preds["revprag_mem"] = [(r["closed_book"] or "I don't know") if f else v
                                            for f, v, r in zip(flag, vp, recs)]
                    res[f"revprag_flag|{c}"] = {"rate": float(flag.mean())}
            if nli is not None:
                preds["rarag"] = [rarag_answer(r, sel_of(r, c, sel_file), passages(raw[r["id"]]), nli) for r in recs]
            for m, p in preds.items():
                cell = {"acc": float(np.mean([match(a, r["golds"]) for a, r in zip(p, recs)])),
                        "asr": float(np.mean([bool(r["target"]) and match(a, [r["target"]]) for a, r in zip(p, recs)])),
                        "n": len(recs), "preds": p}
                if recs[0].get("wrong") is not None:
                    cell["em"] = float(np.mean([all(match(a, [g]) for g in r["golds"]) and not any(
                        match(a, [w]) for w in (r["wrong"] or [])) for a, r in zip(p, recs)]))
                    cell["wrong"] = float(np.mean([any(match(a, [w]) for w in (r["wrong"] or []))
                                                   for a, r in zip(p, recs)]))
                res[f"{m}|{c}"] = cell
                print(f"{d:14s} {c:4s} {m:13s} acc {cell['acc']:.3f} asr {cell['asr']:.3f}", flush=True)
        if rev is not None:
            res["_revprag_meta"] = rev[3]
        json.dump(res, open(path, "w"))


if __name__ == "__main__":
    main()
