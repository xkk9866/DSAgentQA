"""Build passage x candidate grids from cached readings."""

from __future__ import annotations

import glob
import os
import pickle
import random

import numpy as np
import torch

from .core import clean_answer, is_idk, normalize

CACHE = os.environ.get("ONCE_CACHE", "D:/once_data/cache")


def load_readings(model: str, data: str) -> list[dict]:
    recs = []
    drop_z = os.environ.get("ONCE_NO_Z") == "1"  # candidate states are unused by MOTIVE; dropping them saves RAM
    for f in sorted(glob.glob(os.path.join(CACHE, model, data, "*.pkl"))):
        part = pickle.load(open(f, "rb"))
        if drop_z:
            for r in part:
                r.pop("Z", None)
        recs += part
    for r in recs:
        keys = [normalize(c) for c in r["cands"]]
        src = [set() for _ in keys]
        for j, a in enumerate(r["iso"]):
            a = clean_answer(a)
            if is_idk(a):
                continue
            k = normalize(a)
            if k in keys:
                src[keys.index(k)].add(j)
        cb = normalize(clean_answer(r["closed_book"])) if r["closed_book"] else None
        r["src"] = src
        r["cb_idx"] = keys.index(cb) if cb in keys else -1
    return recs


def load_joint(model: str, data: str) -> dict:
    """Joint readings: {cond: {id: [entry]}} for test sets, {"ctx": {id: [entry, ...]}} for training."""
    out = {}
    drop_z = os.environ.get("ONCE_NO_Z") == "1"
    for f in sorted(glob.glob(os.path.join(CACHE, model, "joint", data, "*.pkl"))):
        cond = os.path.basename(f)[:-4].split("_")[0]
        for rid, v in pickle.load(open(f, "rb")).items():
            if drop_z:
                for e in v:
                    e.pop("Z", None)
            out.setdefault(cond, {}).setdefault(rid, []).extend(v)
    return out


def match(pred: str, golds: list[str]) -> bool:
    """Gold answer appears in the prediction as a whole-word span (normalized)."""
    p = f" {normalize(pred)} "
    return any(normalize(g) and f" {normalize(g)} " in p for g in golds)


def condition_indices(r: dict, k: int, n_poison: int) -> list[int]:
    """Top-k context: injected passages first, then clean passages in retrieval order."""
    clean = [i for i, p in enumerate(r["poison"]) if not p]
    poison = [i for i, p in enumerate(r["poison"]) if p]
    sel = poison[:n_poison]
    return sel + clean[:k - len(sel)]


def flood_indices(r: dict, m: int, n_clean: int = 5) -> list[int]:
    """Flooding attack: n_clean clean passages plus m injected passages cycling over the poison set."""
    clean = [i for i, p in enumerate(r["poison"]) if not p]
    poison = [i for i, p in enumerate(r["poison"]) if p]
    return [poison[j % len(poison)] for j in range(m)] + clean[:n_clean]


def cond_indices(r: dict, cond: str, k: int = 10) -> list[int]:
    """'p3' = top-k with 3 injected passages; 'f12' = flooding with 12; 'all' = every passage."""
    if cond == "all":
        return list(range(len(r["poison"])))
    if cond.startswith("f"):
        return flood_indices(r, int(cond[1:]))
    return condition_indices(r, k, int(cond[1:]))


def random_indices(r: dict, rng: random.Random, k_max: int = 10) -> list[int]:
    clean = [i for i, p in enumerate(r["poison"]) if not p]
    poison = [i for i, p in enumerate(r["poison"]) if p]
    k = rng.randint(3, k_max)
    p = 0 if rng.random() < 0.4 else rng.randint(1, 5)  # most retrievals are not poisoned
    p = min(p, len(poison), k - 1)
    sel = rng.sample(poison, p) + rng.sample(clean, min(k - p, len(clean)))
    if rng.random() < 0.3 and sel:  # duplicate passages so the model sees copy-paste flooding
        src = rng.choice(sel)
        sel += [src] * rng.randint(1, 4)
    rng.shuffle(sel)
    return sel


def make_grid(r: dict, sel: list[int], all_cands: bool = False, joint: dict | None = None) -> dict:
    """Rows are selected passages; columns are answers asserted by them or by the closed book."""
    s = set(sel)
    if all_cands:
        act = list(range(len(r["cands"])))
    else:
        act = [c for c, src in enumerate(r["src"]) if (src & s) or c == r["cb_idx"]]
    rows = [j + 1 for j in sel]
    m = len(r["cands"])
    L = r["L"][np.ix_([0] + rows, act + [m])]
    multi = [ci for ci, c in enumerate(act) if match(r["cands"][c], r["golds"])]
    jt = {}
    hd = r["H"].shape[-1]
    if joint is not None:  # the LLM's verdict on the same context, restricted to the same candidates
        jt = {"LJ": joint["L"][act + [m]], "HJ": joint["H"].astype(np.float32),
              "ZJ": joint["Z"][act].astype(np.float32) if "Z" in joint else np.zeros((len(act), hd), np.float32)}
    label = multi[0] if multi else len(act)
    if "Z" in r:
        Z, Z0 = r["Z"][np.ix_(rows, act)].astype(np.float32), r["Z"][0, act].astype(np.float32)
    else:  # placeholders of the right shape (models that use candidate states never run with ONCE_NO_Z)
        Z, Z0 = np.zeros((len(rows), len(act), 1), np.float32), np.zeros((len(act), 1), np.float32)
    return {"L": L[1:], "L0": L[0], "H": r["H"][rows].astype(np.float32), "H0": r["H"][0].astype(np.float32),
            "Z": Z, "Z0": Z0, "cands": [r["cands"][c] for c in act], "label": label, "multi": multi, **jt}


def collate(grids: list[dict], device) -> dict:
    B = len(grids)
    N = max(1, max(g["L"].shape[0] for g in grids))
    C = max(1, max(len(g["cands"]) for g in grids))
    hd = grids[0]["H0"].shape[-1]
    zd = grids[0]["Z0"].shape[-1]
    L = np.zeros((B, N, C + 1), np.float32)
    L0 = np.zeros((B, C + 1), np.float32)
    H = np.zeros((B, N, hd), np.float32)
    Z = np.zeros((B, N, C, zd), np.float32)
    Z0 = np.zeros((B, C, zd), np.float32)
    H0 = np.stack([g["H0"] for g in grids])
    pm = np.zeros((B, N), bool)
    cm = np.zeros((B, C), bool)
    y = np.zeros(B, np.int64)
    for b, g in enumerate(grids):
        n, c = g["L"].shape[0], len(g["cands"])
        L[b, :n, :c] = g["L"][:, :c]
        L[b, :n, C] = g["L"][:, c]
        L0[b, :c] = g["L0"][:c]
        L0[b, C] = g["L0"][c]
        H[b, :n] = g["H"]
        Z[b, :n, :c] = g["Z"]
        Z0[b, :c] = g["Z0"]
        pm[b, :n] = True
        cm[b, :c] = True
        y[b] = g["label"] if g["label"] < c else C
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    extra = {}
    if "LJ" in grids[0]:
        LJ = np.zeros((B, C + 1), np.float32)
        ZJ = np.zeros((B, C, grids[0]["ZJ"].shape[-1]), np.float32)
        for b, g in enumerate(grids):
            c = len(g["cands"])
            LJ[b, :c], LJ[b, C] = g["LJ"][:c], g["LJ"][c]
            ZJ[b, :c] = g["ZJ"]
        extra = {"LJ": t(LJ), "HJ": t(np.stack([g["HJ"] for g in grids])), "ZJ": t(ZJ)}
    return {**extra, "L": t(L), "L0": t(L0), "H": t(H), "H0": t(H0), "Z": t(Z), "Z0": t(Z0),
            "pmask": t(pm), "cmask": t(cm), "y": t(y)}
