"""Train MOTIVE on training readings, select on held-out training questions, evaluate on every benchmark.

No test benchmark is seen before the final evaluation. Results: D:/once_data/results/motive_<model>_<tag>_s<seed>.json
as {data: {method|cond: {...}}}, in the same layout as credence_run.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from once.grid import collate, cond_indices, load_joint, load_readings, make_grid, match, random_indices  # noqa: E402
from once.motive import Motive  # noqa: E402

RUNS = os.environ.get("ONCE_RUNS", "D:/once_data/runs")
RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")
TESTS = {"rqa": "p0,p1,p3,p5,f1,f5,f10,f20", "nq": "p0,p1,p3,p5", "prag-nq": "p0,p1,p3,p5,f1,f5,f10,f20",
         "prag-hotpotqa": "p0,p1,p3,p5", "prag-msmarco": "p0,p1,p3,p5,f1,f5,f10,f20", "ramdocs": "all",
         "fresh-rqa": "p0,p1,p3,p5"}


def grid(r, sel, joint=None):
    g = make_grid(r, sel, joint=joint)
    C = len(g["cands"])
    g["poison"] = [bool(r["poison"][j]) for j in sel]
    # motive label: the attacker's target among the candidates if the context holds injected passages, C if clean,
    # -1 if injected passages are present but their target never became a candidate
    if any(g["poison"]):
        t = [ci for ci, c in enumerate(g["cands"]) if r.get("target") and match(c, [r["target"]])
             and ci not in g["multi"]]
        g["mot"] = t[0] if t else -1
    else:
        g["mot"] = C
    return g


def batch(gs, dev):
    has = [("LJ" in g) for g in gs]
    if any(has) and not all(has):
        hd = gs[0]["H0"].shape[-1]
        for g in gs:
            if "LJ" not in g:
                c = len(g["cands"])
                g.update(LJ=np.zeros(c + 1, np.float32), HJ=np.zeros(hd, np.float32), ZJ=np.zeros((c, hd), np.float32))
    b = collate(gs, dev)
    if "LJ" in gs[0]:
        b["jmask"] = torch.tensor(has, device=dev)
    B, N = b["pmask"].shape
    C = b["cmask"].shape[1]
    pois = torch.zeros(B, N, dtype=torch.bool, device=dev)
    mot = torch.full((B,), -1, dtype=torch.long, device=dev)
    for i, g in enumerate(gs):
        pois[i, :len(g["poison"])] = torch.tensor(g["poison"], dtype=torch.bool, device=dev)
        c = len(g["cands"])
        mot[i] = C if g["mot"] == c else g["mot"]
    b["pois"], b["mot"] = pois, mot
    return b


def hyp_loss(flat, b, use_mot=True):
    """-log P(a*, t*) under the hypothesis posterior; -log P(a*) when the motive is unknown, the gold is absent, or the
    model has no attack hypotheses (matched controls)."""
    B = flat.shape[0]
    C = b["cmask"].shape[1]
    y, mot = b["y"], b["mot"]
    logZ = torch.logsumexp(flat, 1)
    S = flat[:, :-1].view(B, C, C + 1)
    gold_in = y < C
    yc = y.clamp(max=C - 1)
    marg = torch.logsumexp(S[torch.arange(B), yc], 1)  # log sum_t exp S(a*, t)
    joint = S[torch.arange(B), yc, mot.clamp(min=0)]
    num = torch.where((mot >= 0) & use_mot, joint, marg)
    num = torch.where(gold_in, num, flat[:, -1])
    return (logZ - num).mean()


def uses_motive(net):
    return net.pool == "motive" and not net.direct


def worst_case_states(net, b, eps, steps=3):
    """PGD on the reading states of injected passages within a relative L2 ball (attacker-controlled rows only)."""
    H = b["H"]
    m = b["pois"][..., None].float()
    radius = eps * H.norm(dim=-1, keepdim=True) * m
    delta = torch.zeros_like(H)
    for _ in range(steps):
        delta.requires_grad_(True)
        lo = hyp_loss(net({**b, "H": H + delta}), b, uses_motive(net))
        g, = torch.autograd.grad(lo, delta)
        with torch.no_grad():
            delta = delta + 0.5 * radius * g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            n = delta.norm(dim=-1, keepdim=True)
            delta = delta * torch.clamp(radius / n.clamp(min=1e-12), max=1.0)
    return {**b, "H": (H + delta).detach()}


@torch.no_grad()
def predict(net, gs, dev, bs=64):
    net.eval()
    out = []
    for s in range(0, len(gs), bs):
        g = gs[s:s + bs]
        b = batch(g, dev)
        flat = net(b)
        C = b["cmask"].shape[1]
        ans, mot = Motive.split(flat, C)
        ansp = ans.softmax(-1).cpu().numpy()
        motp = mot.softmax(-1).cpu().numpy()
        for i, gi in enumerate(g):
            c = len(gi["cands"])
            f = int(ansp[i, :c].argmax()) if c else 0
            a = int(np.argmax(np.r_[ansp[i, :c], ansp[i, -1]])) if c else 0
            out.append({"forced": gi["cands"][f] if c else "I don't know",
                        "abst": gi["cands"][a] if a < c else "I don't know",
                        "p_attack": float(1 - motp[i, C]),
                        "motive": gi["cands"][int(motp[i, :c].argmax())] if c else None,
                        "motive_p": float(motp[i, :c].max()) if c else 0.0})
    return out


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def score_cells(pr, recs, key):
    acc = float(np.mean([match(p[key], r["golds"]) for p, r in zip(pr, recs)]))
    asr = float(np.mean([bool(r["target"]) and match(p[key], [r["target"]]) for p, r in zip(pr, recs)]))
    cell = {"acc": acc, "asr": asr, "n": len(recs), "preds": [p[key] for p in pr]}
    if recs[0].get("wrong") is not None:
        cell["em"] = float(np.mean([all(match(p[key], [g]) for g in r["golds"]) and
                                    not any(match(p[key], [w]) for w in (r["wrong"] or [])) for p, r in zip(pr, recs)]))
        cell["wrong"] = float(np.mean([any(match(p[key], [w]) for w in (r["wrong"] or [])) for p, r in zip(pr, recs)]))
    return cell


def evaluate_all(net, model, dev, use_joint, datasets=None, name="motive"):
    res = {}
    for d, conds in TESTS.items():
        if datasets and d not in datasets:
            continue
        recs = load_readings(model, d)
        if not recs:
            continue
        jt = load_joint(model, d) if use_joint else {}
        res[d] = {}
        for c in conds.split(","):
            gs = []
            for r in recs:
                e = jt.get(c, {}).get(r["id"], [None])[0] if use_joint else None
                gs.append(grid(r, cond_indices(r, c), joint=e))
            pr = predict(net, gs, dev)
            for key in ("forced", "abst"):
                res[d][f"{name}{'' if key == 'forced' else '_' + key}|{c}"] = score_cells(pr, recs, key)
            res[d][f"diag|{c}"] = {"p_attack": [p["p_attack"] for p in pr], "motive": [p["motive"] for p in pr],
                                   "has_joint": float(np.mean(["LJ" in g for g in gs]))}
            print(f"{d:14s} {c:4s} acc {res[d][f'{name}|{c}']['acc']:.3f} asr {res[d][f'{name}|{c}']['asr']:.3f} "
                  f"p_att {np.mean([p['p_attack'] for p in pr]):.2f} J {res[d][f'diag|{c}']['has_joint']:.2f}", flush=True)
    return res


def build_model(args, hdim):
    return Motive(hdim=hdim, d=args.d, layers=args.layers, dropout=args.dropout, use_joint=not args.no_joint,
                  use_memory=not args.no_memory, use_h=not args.no_h, symmetric=not args.no_sym,
                  motive=not args.no_motive, direct=args.direct, learn_rho=not args.no_rho,
                  pool=getattr(args, "pool", "motive"), j_suff=getattr(args, "j_suff", False),
                  blocks=getattr(args, "blocks", False), motive_prior=getattr(args, "motive_prior", False))


def load_train(args):
    tr, val = [], []
    for name in args.train.split(","):
        recs = load_readings(args.model, name)
        aug = {r["id"]: r for r in load_readings(args.model, name + "-disg")} if args.disg else {}
        recs = [aug.get(r["id"], r) for r in recs]
        jt = load_joint(args.model, name).get("ctx", {})
        jd = load_joint(args.model, name + "-disg").get("ctx", {}) if args.disg else {}
        for r in recs:
            r["jctx"] = jt.get(r["id"], []) + jd.get(r["id"], [])
        random.Random(0).shuffle(recs)  # the same held-out questions for every seed and variant
        n_val = max(len(recs) // 10, min(200, len(recs) // 5))
        val += recs[:n_val]
        tr += recs[n_val:]
    return tr, val


def mimic(b, gs, rng, q):
    """Mimicry augmentation: an injected passage's reading state is replaced by the state of a genuine passage of
    another question in the batch (a realizable perfect disguise)."""
    if q <= 0:
        return b
    H = b["H"].clone()
    pool = [(i, j) for i, g in enumerate(gs) for j, p in enumerate(g["poison"]) if not p]
    for i, g in enumerate(gs):
        for j, p in enumerate(g["poison"]):
            if p and rng.random() < q:
                k = rng.choice(pool)
                while k[0] == i and len(pool) > 1:
                    k = rng.choice(pool)
                H[i, j] = b["H"][k[0], k[1]]
    return {**b, "H": H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--train", default="train_v4,train-rtqa")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pj", type=float, default=0.7, help="share of training contexts with a cached joint reading")
    ap.add_argument("--no_joint", action="store_true")
    ap.add_argument("--no_memory", action="store_true")
    ap.add_argument("--no_h", action="store_true")
    ap.add_argument("--no_sym", action="store_true")
    ap.add_argument("--no_motive", action="store_true")
    ap.add_argument("--no_rho", action="store_true")
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--motive_prior", action="store_true", help="uniform prior over the K-1 motives")
    ap.add_argument("--p_clean", type=float, default=-1, help="share of clean training contexts (-1: natural mix)")
    ap.add_argument("--blocks", action="store_true", help="separately normalized attack / no-attack blocks")
    ap.add_argument("--j_suff", action="store_true", help="joint reading summarizes the context under no attack")
    ap.add_argument("--pool", default="motive", help="motive | poe | trunc | mixture | vote (matched controls)")
    ap.add_argument("--disg", action="store_true")
    ap.add_argument("--adv_eps", type=float, default=0.0)
    ap.add_argument("--mimic", type=float, default=0.0, help="probability of mimicry augmentation per injected row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="motive")
    ap.add_argument("--eval", default="", help="comma list of test sets; 'none' = selection run only")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    use_joint = not args.no_joint
    out_dir = os.path.join(RUNS, args.model, f"motive_{args.tag}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(RES, f"motive_{args.model}_{args.tag}_s{args.seed}.json")

    tr, val = load_train(args)
    net = build_model(args, tr[0]["H"].shape[-1]).to(dev)
    if args.eval_only:
        net.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)["state"])
        res = json.load(open(path)) if os.path.exists(path) else {}
        res.update(evaluate_all(net, args.model, dev, use_joint, args.eval.split(",") if args.eval else None))
        json.dump(res, open(path, "w"))
        return
    vrng = random.Random(123)
    vgrids = []
    for r in val:
        for e in r["jctx"]:
            vgrids.append(grid(r, e["sel"], joint=e if use_joint else None))
        for _ in range(2):
            vgrids.append(grid(r, list(dict.fromkeys(random_indices(r, vrng)))))
    print(f"train {len(tr)}  val {len(val)} ({len(vgrids)} contexts)  params {sum(p.numel() for p in net.parameters())}",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    best, t0 = -1.0, time.time()
    for ep in range(args.epochs):
        net.train()
        order = list(range(len(tr)))
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order) - args.bs + 1, args.bs):
            gs = []
            for i in order[s:s + args.bs]:
                r = tr[i]
                if args.p_clean >= 0:  # balanced prior: clean and attacked contexts drawn with fixed shares
                    clean = rng.random() < args.p_clean
                    pool = [e for e in r["jctx"] if any(r["poison"][j] for j in e["sel"]) != clean] if use_joint else []
                    if pool and rng.random() < args.pj:
                        e = rng.choice(pool)
                        gs.append(grid(r, e["sel"], joint=e))
                    else:
                        for _ in range(50):
                            sel = random_indices(r, rng)
                            if any(r["poison"][j] for j in sel) != clean:
                                break
                        gs.append(grid(r, sel))
                elif use_joint and r["jctx"] and rng.random() < args.pj:
                    e = rng.choice(r["jctx"])
                    gs.append(grid(r, e["sel"], joint=e))
                else:
                    gs.append(grid(r, random_indices(r, rng)))
            b = mimic(batch(gs, dev), gs, rng, args.mimic)
            loss = hyp_loss(net(b), b, uses_motive(net))
            if args.pool == "vote":  # the filter needs credibility labels: injected passages are not credible
                tm = torch.zeros_like(net.last_cred, dtype=torch.bool)
                tt = torch.zeros_like(net.last_cred)
                N = b["pmask"].shape[1]
                tm[:, 1:1 + N] = b["pmask"]
                tt[:, 1:1 + N] = (~b["pois"]).float()
                bce = F.binary_cross_entropy_with_logits(net.last_cred, tt, reduction="none")
                loss = loss + 0.5 * (bce * tm.float()).sum() / tm.float().sum().clamp(min=1)
            if args.adv_eps > 0:
                ba = worst_case_states(net, b, args.adv_eps)
                loss = 0.5 * loss + 0.5 * hyp_loss(net(ba), ba, uses_motive(net))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        pr = predict(net, vgrids, dev)
        score = float(np.mean([g["cands"].index(p["forced"]) in g["multi"] if p["forced"] in g["cands"] else 0
                               for p, g in zip(pr, vgrids)]))
        print(f"ep {ep} loss {tot / max(1, len(order) // args.bs):.4f} val {score:.4f} {time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            torch.save({"state": net.state_dict(), "args": vars(args)}, os.path.join(out_dir, "best.pt"))
    print("best val", best, flush=True)
    net.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)["state"])
    if args.eval == "none":
        json.dump({"_meta": {"args": vars(args), "best_val": best}}, open(os.path.join(out_dir, "val.json"), "w"))
        return
    res = evaluate_all(net, args.model, dev, use_joint, args.eval.split(",") if args.eval else None)
    res["_meta"] = {"args": vars(args), "best_val": best}
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(path, "w"))


if __name__ == "__main__":
    main()
