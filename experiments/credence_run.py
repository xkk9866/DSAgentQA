"""Train CREDENCE on the training readings, select on held-out training questions, evaluate on every benchmark.

No test benchmark is seen before the final evaluation. Results go to
D:/once_data/results/credence_<model>_<tag>_s<seed>.json as {data: {method|cond: {...}}}.
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
from once.credence import Credence  # noqa: E402
from once.grid import collate, cond_indices, load_joint, load_readings, make_grid, match, random_indices  # noqa: E402

RUNS = os.environ.get("ONCE_RUNS", "D:/once_data/runs")
RES = os.environ.get("ONCE_RESULTS", "D:/once_data/results")
TESTS = {"rqa": "p0,p1,p3,p5,f1,f5,f10,f20", "nq": "p0,p1,p3,p5", "prag-nq": "p0,p1,p3,p5,f1,f5,f10,f20",
         "prag-hotpotqa": "p0,p1,p3,p5", "prag-msmarco": "p0,p1,p3,p5,f1,f5,f10,f20", "ramdocs": "all",
         "fresh-rqa": "p0,p1,p3,p5"}


def grid(r, sel, joint=None):
    g = make_grid(r, sel, joint=joint)
    C = len(g["cands"])
    g["poison"] = [bool(r["poison"][j]) for j in sel]
    g["cb_ok"] = float(match(r["closed_book"] or "", r["golds"]))
    if joint is not None:
        g["j_ok"] = float(C > 0 and g["label"] < C and int(np.argmax(g["LJ"][:C])) == g["label"])
    return g


def batch(gs, dev):
    """collate() plus a joint row that may be missing for some questions, and credibility targets."""
    has = [("LJ" in g) for g in gs]
    if any(has) and not all(has):
        hd = gs[0]["H0"].shape[-1]
        for g in gs:
            if "LJ" not in g:
                c = len(g["cands"])
                g.update(LJ=np.zeros(c + 1, np.float32), HJ=np.zeros(hd, np.float32), ZJ=np.zeros((c, hd), np.float32))
    b = collate(gs, dev)
    if "LJ" in gs[0]:
        b["jmask"] = torch.tensor(has if any(has) else [True] * len(gs), device=dev)
    B, N = b["pmask"].shape
    R = 1 + N + (1 if "LJ" in b else 0)
    tgt = torch.zeros(B, R, device=dev)
    tm = torch.zeros(B, R, dtype=torch.bool, device=dev)
    for i, g in enumerate(gs):
        tgt[i, 0], tm[i, 0] = g["cb_ok"], True
        n = len(g["poison"])
        tgt[i, 1:1 + n] = torch.tensor([0.0 if p else 1.0 for p in g["poison"]], device=dev)
        tm[i, 1:1 + n] = True
        if "LJ" in b and has[i]:
            tgt[i, -1], tm[i, -1] = g.get("j_ok", 0.0), True
    b["cred_t"], b["cred_m"] = tgt, tm
    pois = torch.zeros(B, N, dtype=torch.bool, device=dev)
    for i, g in enumerate(gs):
        pois[i, :len(g["poison"])] = torch.tensor(g["poison"], dtype=torch.bool, device=dev)
    b["pois"] = pois  # rows the attacker controls
    return b


def worst_case_states(net, b, loss_of, eps, steps=3):
    """PGD in reading-state space: the attacker moves the state of every injected passage within an L2 ball of radius
    eps * ||h|| to maximize loss_of(b). Returns a copy of b with the perturbed states."""
    H = b["H"]
    m = b["pois"][..., None].float()
    radius = eps * H.norm(dim=-1, keepdim=True) * m
    delta = torch.zeros_like(H)
    for _ in range(steps):
        delta.requires_grad_(True)
        lo = loss_of({**b, "H": H + delta})
        g, = torch.autograd.grad(lo, delta)
        with torch.no_grad():
            delta = delta + 0.5 * radius * g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            n = delta.norm(dim=-1, keepdim=True)
            delta = delta * torch.clamp(radius / n.clamp(min=1e-12), max=1.0)
    return {**b, "H": (H + delta).detach()}


def multi_target(b, gs, net):
    C = b["cmask"].shape[1]
    t = torch.zeros_like(net.last_binary)
    for i, g in enumerate(gs):
        for ci in g.get("multi", []) or ([g["label"]] if g["label"] < len(g["cands"]) else []):
            t[i, ci] = 1.0
    return t[:, :C]


def loss_fn(net, b, gs, args):
    logit, outs = net(b, return_all=True)
    ce = sum(F.cross_entropy(lo, b["y"]) for lo, *_ in outs) / len(outs)
    loss = ce
    if args.lam_cred > 0 and net.learn_cred:
        zc = net.last_cred
        bce = F.binary_cross_entropy_with_logits(zc, b["cred_t"], reduction="none")
        m = (b["cred_m"] & net.last_rmask).float()
        loss = loss + args.lam_cred * (bce * m).sum() / m.sum().clamp(min=1)
    bt = multi_target(b, gs, net)
    bce = F.binary_cross_entropy_with_logits(net.last_binary, bt, reduction="none")
    loss = loss + 0.3 * (bce * b["cmask"]).sum() / b["cmask"].sum()
    return loss


@torch.no_grad()
def predict(net, gs, dev, bs=64):
    net.eval()
    out = []
    for s in range(0, len(gs), bs):
        g = gs[s:s + bs]
        b = batch(g, dev)
        lo = net(b)
        prob = torch.sigmoid(net.last_binary).cpu().numpy()
        cred = torch.sigmoid(net.last_cred).cpu().numpy()
        for i, gi in enumerate(g):
            C = len(gi["cands"])
            f = int(lo[i, :-1].argmax()) if C else 0
            a = int(lo[i].argmax())
            chosen = [c for ci, c in enumerate(gi["cands"]) if prob[i, ci] > 0.5]
            if not chosen and a < C:
                chosen = [gi["cands"][a]]
            n = len(gi["poison"])
            out.append({"forced": gi["cands"][f] if C else "I don't know",
                        "abst": gi["cands"][a] if a < C else "I don't know",
                        "set": ", ".join(chosen) or "I don't know",
                        "cred_prior": float(cred[i, 0]), "cred_pass": cred[i, 1:1 + n].tolist(),
                        "cred_joint": float(cred[i, -1]) if "LJ" in gi else None})
    return out


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def evaluate_all(net, model, dev, use_joint, datasets=None):
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
            for key in ("forced", "abst", "set"):
                acc = float(np.mean([match(p[key], r["golds"]) for p, r in zip(pr, recs)]))
                asr = float(np.mean([bool(r["target"]) and match(p[key], [r["target"]]) for p, r in zip(pr, recs)]))
                cell = {"acc": acc, "asr": asr, "n": len(recs), "preds": [p[key] for p in pr]}
                if recs[0].get("wrong") is not None:
                    cell["em"] = float(np.mean([all(match(p[key], [g]) for g in r["golds"]) and
                                                not any(match(p[key], [w]) for w in (r["wrong"] or []))
                                                for p, r in zip(pr, recs)]))
                    cell["wrong"] = float(np.mean([any(match(p[key], [w]) for w in (r["wrong"] or []))
                                                   for p, r in zip(pr, recs)]))
                res[d][f"credence{'' if key == 'forced' else '_' + key}|{c}"] = cell
            # credibility diagnostics: does r separate injected from clean passages, and right from wrong priors?
            sc = [x for p in pr for x in p["cred_pass"]]
            lb = [0 if pz else 1 for g in gs for pz in g["poison"]]
            res[d][f"diag|{c}"] = {"auroc_poison": auroc(sc, lb),
                                   "auroc_prior": auroc([p["cred_prior"] for p in pr], [g["cb_ok"] for g in gs]),
                                   "cred_prior": [p["cred_prior"] for p in pr],
                                   "cred_joint": [p["cred_joint"] for p in pr]}
            print(f"{d:14s} {c:4s} acc {res[d][f'credence|{c}']['acc']:.3f} asr {res[d][f'credence|{c}']['asr']:.3f}"
                  f"  auroc_poison {res[d][f'diag|{c}']['auroc_poison']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--train", default="train_v4,train-rtqa")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lam_cred", type=float, default=0.5)
    ap.add_argument("--pj", type=float, default=0.5, help="share of training contexts with a cached joint reading")
    ap.add_argument("--no_robust", action="store_true")
    ap.add_argument("--no_cred", action="store_true")
    ap.add_argument("--no_joint", action="store_true")
    ap.add_argument("--no_prior", action="store_true")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--no_dep", action="store_true")
    ap.add_argument("--no_z", action="store_true", help="drop per-candidate hidden states")
    ap.add_argument("--direct", action="store_true", help="baseline: same input/encoder, direct answer logits")
    ap.add_argument("--no_h", action="store_true", help="drop per-source reading states")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="credence")
    ap.add_argument("--eval", default="", help="comma list of test sets (default: all)")
    ap.add_argument("--eval_only", action="store_true", help="load the trained checkpoint and add the --eval sets")
    ap.add_argument("--disg", action="store_true", help="use training records augmented with disguised poison")
    ap.add_argument("--adv_eps", type=float, default=0.0, help="worst-case reading states of injected rows (relative L2)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)

    tr, val = [], []
    for name in args.train.split(","):
        recs = load_readings(args.model, name)
        if args.disg:  # same questions, same order; augmented records replace the originals where available
            aug = {r["id"]: r for r in load_readings(args.model, name + "-disg")}
            recs = [aug.get(r["id"], r) for r in recs]
        jt = load_joint(args.model, name).get("ctx", {})
        for r in recs:
            r["jctx"] = jt.get(r["id"], [])
        random.Random(0).shuffle(recs)  # the same held-out questions for every seed and variant
        n_val = max(len(recs) // 10, min(200, len(recs) // 5))
        val += recs[:n_val]
        tr += recs[n_val:]
    use_joint = not args.no_joint
    vrng = random.Random(123)
    vgrids = []
    for r in val:
        for e in r["jctx"][:3]:
            vgrids.append(grid(r, e["sel"], joint=e if use_joint else None))
        for _ in range(2):
            vgrids.append(grid(r, list(dict.fromkeys(random_indices(r, vrng)))))
    print(f"train {len(tr)}  val {len(val)} ({len(vgrids)} contexts)", flush=True)

    net = Credence(hdim=tr[0]["H"].shape[-1], d=args.d, layers=args.layers, iters=args.iters, dropout=args.dropout,
                   robust=not args.no_robust, learn_cred=not args.no_cred, use_joint=use_joint,
                   use_prior=not args.no_prior, residual=args.residual, dep=not args.no_dep, use_z=not args.no_z, use_h=not args.no_h, direct=args.direct).to(dev)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    out_dir = os.path.join(RUNS, args.model, f"{args.tag}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    best, t0 = -1.0, time.time()
    if args.eval_only:  # add test sets to an existing run without retraining
        net.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)["state"])
        path = os.path.join(RES, f"credence_{args.model}_{args.tag}_s{args.seed}.json")
        res = json.load(open(path))
        res.update(evaluate_all(net, args.model, dev, use_joint, args.eval.split(",")))
        json.dump(res, open(path, "w"))
        return
    for ep in range(args.epochs):
        net.train()
        order = list(range(len(tr)))
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order) - args.bs + 1, args.bs):
            gs = []
            for i in order[s:s + args.bs]:
                r = tr[i]
                if use_joint and r["jctx"] and rng.random() < args.pj:
                    e = rng.choice(r["jctx"])
                    gs.append(grid(r, e["sel"], joint=e))
                else:
                    gs.append(grid(r, random_indices(r, rng)))
            b = batch(gs, dev)
            loss = loss_fn(net, b, gs, args)
            if args.adv_eps > 0:  # credibility must not hinge on what the attacker controls
                ba = worst_case_states(net, b, lambda x: loss_fn(net, x, gs, args), args.adv_eps)
                loss = 0.5 * loss + 0.5 * loss_fn(net, ba, gs, args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        pr = predict(net, vgrids, dev)
        score = float(np.mean([g["cands"].index(p["forced"]) == g["label"] if p["forced"] in g["cands"] else 0
                               for p, g in zip(pr, vgrids)]))
        print(f"ep {ep} loss {tot / max(1, len(order) // args.bs):.4f} val {score:.4f} {time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            torch.save({"state": net.state_dict(), "args": vars(args)}, os.path.join(out_dir, "best.pt"))
    print("best val", best, flush=True)
    net.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)["state"])
    if args.eval == "none":  # model/config selection runs: never touch the test benchmarks
        json.dump({"_meta": {"args": vars(args), "best_val": best}},
                  open(os.path.join(out_dir, "val.json"), "w"))
        return
    res = evaluate_all(net, args.model, dev, use_joint, args.eval.split(",") if args.eval else None)
    res["_meta"] = {"args": vars(args), "best_val": best}
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, f"credence_{args.model}_{args.tag}_s{args.seed}.json"), "w"))


if __name__ == "__main__":
    main()
