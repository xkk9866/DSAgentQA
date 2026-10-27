"""Adaptive attacks on MOTIVE and on the learned controls, from cached readings.

Contexts on the four adaptive sets (RQA + three PoisonedRAG sets, 400 questions; passages 0-9 clean, 10 + 3g + v is
disguised passage of genre g, variant v):
  p0        clean
  a<t>      blind disguised attack (variant 0 of t genres)
  o<t>      CREDENCE-targeted selection (transfer)
  m<t>      MOTIVE-targeted selection with the joint reader in the loop (motive_query_attack.py)

White-box feature attacks (the attacker controls every injected row):
  mimic     each injected passage's reading state is replaced by a genuine passage's state; the attacker searches
            n_cand random genuine states per row and keeps the combination that maximizes P(target) (realizable:
            the states of real passages)
  pgd       targeted PGD on the injected reading states within relative L2 radius eps, n_restarts random starts;
            the reported result at each radius is the worst over that radius and every smaller one (monotone
            envelope), so a larger budget can never look safer because of optimization failure
  lie       targeted PGD on the injected rows' testimony logits (unbounded): the attacker chooses how confidently
            and how often its passages assert each candidate, and the joint reading's testimony as well
Results: D:/once_data/results/attacks_<model>_<tag>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motive_run import batch, build_model, grid, predict  # noqa: E402
from once.grid import load_joint, load_readings, match  # noqa: E402
from once.motive import Motive  # noqa: E402

RUNS = os.environ.get("ONCE_RUNS", "D:/once_data/runs")
RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")
SETS = ["adv-rqa", "adv-prag-nq", "adv-prag-hotpotqa", "adv-prag-msmarco"]
STD = {"adv-rqa": "rqa", "adv-prag-nq": "prag-nq", "adv-prag-hotpotqa": "prag-hotpotqa",
       "adv-prag-msmarco": "prag-msmarco"}


def blind(t):
    return [10 + 3 * g for g in range(t)] + list(range(10 - t))


class Args:
    pass


def load_motive(model, tag, seed, hdim, dev):
    ck = torch.load(os.path.join(RUNS, model, f"motive_{tag}_s{seed}", "best.pt"), map_location=dev)
    a = Args()
    a.__dict__.update(ck["args"])
    net = build_model(a, hdim).to(dev)
    net.load_state_dict(ck["state"])
    return net.eval(), not a.no_joint


def target_logp(net, b, tidx):
    """log P(answer = target) under MOTIVE."""
    C = b["cmask"].shape[1]
    ans, _ = Motive.split(net(b), C)
    lp = ans.log_softmax(-1)
    return lp[torch.arange(len(tidx), device=lp.device), tidx.clamp(min=0)]


def tidx_of(gs, recs, dev):
    t = []
    for g, r in zip(gs, recs):
        k = [i for i, c in enumerate(g["cands"]) if r.get("target") and match(c, [r["target"]]) and i not in g["multi"]]
        t.append(k[0] if k else -1)
    return torch.tensor(t, device=dev)


def decide(net, b, gs):
    C = b["cmask"].shape[1]
    with torch.no_grad():
        ans, _ = Motive.split(net(b), C)
    out = []
    for i, g in enumerate(gs):
        c = len(g["cands"])
        out.append(g["cands"][int(ans[i, :c].argmax())] if c else "I don't know")
    return out


def pgd_states(net, b, tidx, eps, steps=20, restarts=3, seed=0):
    """Worst (highest target log-prob) state perturbation over restarts; returns per-question best delta."""
    H = b["H"]
    m = b["pois"][..., None].float()
    radius = eps * H.norm(dim=-1, keepdim=True) * m
    has = tidx >= 0
    best_obj = torch.full((H.shape[0],), -1e9, device=H.device)
    best = torch.zeros_like(H)
    g = torch.Generator(device=H.device).manual_seed(seed)
    for r in range(restarts):
        delta = torch.randn(H.shape, generator=g, device=H.device) if r else torch.zeros_like(H)
        delta = delta * radius / delta.norm(dim=-1, keepdim=True).clamp(min=1e-12) * (0.5 if r else 0)
        for _ in range(steps):
            delta.requires_grad_(True)
            obj = target_logp(net, {**b, "H": H + delta}, tidx)
            gr, = torch.autograd.grad(obj[has].sum(), delta)
            with torch.no_grad():
                delta = delta + 0.2 * radius * gr / gr.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                n = delta.norm(dim=-1, keepdim=True)
                delta = delta * torch.clamp(radius / n.clamp(min=1e-12), max=1.0)
        with torch.no_grad():
            obj = target_logp(net, {**b, "H": H + delta}, tidx)
            upd = obj > best_obj
            best_obj = torch.where(upd, obj, best_obj)
            best[upd] = delta[upd]
    return best


def pgd_testimony(net, b, tidx, steps=30):
    """The attacker rewrites the testimony (log-likelihood rows) of its passages and of the joint reading."""
    L = b["L"].clone()
    m = b["pois"][..., None].float()
    delta = torch.zeros_like(L)
    jm = "LJ" in b
    dJ = torch.zeros_like(b["LJ"]) if jm else None
    has = tidx >= 0
    for _ in range(steps):
        delta.requires_grad_(True)
        bb = {**b, "L": L + delta * m}
        if jm:
            dJ.requires_grad_(True)
            bb["LJ"] = b["LJ"] + dJ
        obj = target_logp(net, bb, tidx)
        grads = torch.autograd.grad(obj[has].sum(), [delta] + ([dJ] if jm else []))
        with torch.no_grad():
            delta = delta + 2.0 * torch.sign(grads[0])
            if jm:
                dJ = dJ + 2.0 * torch.sign(grads[1])
    out = {**b, "L": (L + delta * m).detach()}
    if jm:
        out["LJ"] = (b["LJ"] + dJ).detach()
    return out


def mimic_states(net, b, tidx, genuine, n_cand=16, seed=0):
    """Replace every injected state by one of n_cand genuine passage states; greedy row by row, keep the best."""
    rng = np.random.default_rng(seed)
    H = b["H"].clone()
    B, N, _ = H.shape
    G = torch.tensor(genuine, device=H.device)
    for j in range(N):
        rows = b["pois"][:, j]
        if not rows.any():
            continue
        best_obj = torch.full((B,), -1e9, device=H.device)
        best_h = H[:, j].clone()
        for _ in range(n_cand):
            cand = G[torch.tensor(rng.integers(0, len(G), B), device=H.device)]
            Ht = H.clone()
            Ht[rows, j] = cand[rows]
            with torch.no_grad():
                obj = target_logp(net, {**b, "H": Ht}, tidx)
            upd = (obj > best_obj) & rows
            best_obj = torch.where(upd, obj, best_obj)
            best_h[upd] = cand[upd]
        H[rows, j] = best_h[rows]
    return {**b, "H": H}


def score(preds, recs):
    return {"acc": float(np.mean([match(p, r["golds"]) for p, r in zip(preds, recs)])),
            "asr": float(np.mean([bool(r["target"]) and match(p, [r["target"]]) for p, r in zip(preds, recs)])),
            "n": len(recs), "preds": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--tags", default="final", help="MOTIVE run tags to evaluate (all seeds)")
    ap.add_argument("--conds", default="p0,a1,a3,a5,o1,o3,o5,m1,m3,m5")
    ap.add_argument("--wb", default="mimic,pgd,lie")
    ap.add_argument("--wb_conds", default="p3,p5,a5", help="contexts for the white-box attacks")
    ap.add_argument("--eps", default="0.15,0.3,0.5,1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    ap.add_argument("--others", action="store_true", help="also CREDENCE (rob30 seeds) and majority / REPLUG")
    args = ap.parse_args()
    dev = torch.device(args.device)
    data = {d: load_readings(args.model, d) for d in SETS}
    joint = {d: load_joint(args.model, d) for d in SETS}
    sel_c = json.load(open(os.path.join(RES, f"adaptive_sel_{args.model}_rob30.json")))
    sel_m = {}
    for f in glob.glob(os.path.join(RES, f"motive_sel_{args.model}_*.json")):
        sel_m.update(json.load(open(f)))
    hdim = data[SETS[0]][0]["H"].shape[-1]
    genuine = np.concatenate([r["H"][1:11].astype(np.float32) for d in ("train_v4",)
                              for r in load_readings(args.model, d)[:400]])
    models = []
    for tag in args.tags.split(","):
        for f in sorted(glob.glob(os.path.join(RUNS, args.model, f"motive_{tag}_s*"))):
            s = int(f.split("_s")[-1])
            net, uj = load_motive(args.model, tag, s, hdim, dev)
            models.append((f"{tag}_s{s}", net, uj))
    out_path = args.out or os.path.join(RES, f"attacks_{args.model}_{args.tags.replace(',', '+')}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}

    def sel_of(r, c):
        if c == "p0":
            return list(range(10))
        if c.startswith("a"):
            return blind(int(c[1:]))
        if c.startswith("o"):
            return sel_c[r["id"]][c]
        if c.startswith("m"):
            return sel_m.get(r["id"], {}).get(c)
        if c.startswith("p"):  # standard poison on the adaptive sets' own passages: poison rows 10-14 are disguised,
            t = int(c[1:])     # so p<t> here means t disguised genres, variant 0 (= a<t>); kept for the WB grid
            return blind(t)
        raise ValueError(c)

    if args.others:  # learned credibility pooling and reading baselines on exactly the same contexts
        from adaptive_eval import load_credence
        from credence_run import grid as cgrid, predict as cpredict
        from eval_once import reading_baselines
        cmods = [(f"credence_s{s}", load_credence(args.model, "rob30", s, hdim, dev)) for s in range(3)]
        for d, recs in data.items():
            for c in args.conds.split(","):
                sels = [sel_of(r, c) for r in recs]
                if any(x is None for x in sels):
                    continue
                gs = [cgrid(r, x) for r, x in zip(recs, sels)]
                for cname, cnet in cmods:
                    res.setdefault(cname, {}).setdefault(d, {})[c] = score([p["forced"] for p in cpredict(cnet, gs, dev)], recs)
                rb = {}
                for g in gs:
                    for k, v in reading_baselines(g).items():
                        if k in ("majority", "replug", "closed_book"):
                            rb.setdefault(k, []).append(v)
                for k, v in rb.items():
                    res.setdefault(k, {}).setdefault(d, {})[c] = score(v, recs)
                print("others", d, c, flush=True)
        json.dump(res, open(out_path, "w"))
    for name, net, uj in models:
        for d, recs in data.items():
            R = res.setdefault(name, {}).setdefault(d, {})
            for c in args.conds.split(","):
                sels = [sel_of(r, c) for r in recs]
                if any(s is None for s in sels):
                    continue
                jt = joint[d].get(c, {})
                gs = [grid(r, s, joint=(jt.get(r["id"], [None])[0] if uj else None)) for r, s in zip(recs, sels)]
                pr = predict(net, gs, dev)
                R[c] = score([p["forced"] for p in pr], recs)
                R[c]["has_joint"] = float(np.mean(["LJ" in g for g in gs]))
                print(name, d, c, f"acc {R[c]['acc']:.3f} asr {R[c]['asr']:.3f} J {R[c]['has_joint']:.2f}", flush=True)
            for c in args.wb_conds.split(","):
                cc = c if not c.startswith("p") else "a" + c[1:]
                sels = [sel_of(r, cc) for r in recs]
                jt = joint[d].get(cc, {})
                gs = [grid(r, s, joint=(jt.get(r["id"], [None])[0] if uj else None)) for r, s in zip(recs, sels)]
                for wb in args.wb.split(","):
                    keys = [f"{wb}@{c}"] if wb != "pgd" else [f"pgd{e}@{c}" for e in args.eps.split(",")]
                    if all(k in R for k in keys):
                        continue
                    preds = {k: [] for k in keys}
                    for s in range(0, len(gs), 50):
                        g, rr = gs[s:s + 50], recs[s:s + 50]
                        b = batch(g, dev)
                        ti = tidx_of(g, rr, dev)
                        if wb == "mimic":
                            preds[keys[0]] += decide(net, mimic_states(net, b, ti, genuine), g)
                        elif wb == "lie":
                            preds[keys[0]] += decide(net, pgd_testimony(net, b, ti), g)
                        else:
                            worst = None  # monotone envelope over radii
                            for e, k in zip(args.eps.split(","), keys):
                                bd = {**b, "H": b["H"] + pgd_states(net, b, ti, float(e))}
                                pk = decide(net, bd, g)
                                hit = np.array([bool(r["target"]) and match(p, [r["target"]]) for p, r in zip(pk, rr)])
                                if worst is None:
                                    worst = (pk, hit)
                                else:
                                    pk = [p if h else w for p, h, w in zip(pk, hit, worst[0])]
                                    hit = hit | worst[1]
                                    # a smaller radius that already reached the target is also feasible here
                                    worst = (pk, hit)
                                preds[k] += pk
                    for k in keys:
                        R[k] = score(preds[k], recs)
                        print(name, d, k, f"acc {R[k]['acc']:.3f} asr {R[k]['asr']:.3f}", flush=True)
            json.dump(res, open(out_path, "w"))


if __name__ == "__main__":
    main()
