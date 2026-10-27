"""Benchmarks as lists of instances: question, gold answers, attack target, passages."""

from __future__ import annotations

import ast
import json
import os
import random

RAW = os.environ.get("ONCE_RAW", "D:/once_data/raw")
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def robustrag(name: str) -> list[dict]:
    """RobustRAG's RealtimeQA / Open-NQ with Google-search passages and 5 poisoned passages each."""
    fn = {"rqa": "realtimeqa.json", "nq": "open_nq.json"}[name]
    data = json.load(open(os.path.join(RAW, "RobustRAG/data", fn), encoding="utf-8"))
    out = []
    for i, x in enumerate(data):
        ctx = _lit(x["context"])
        golds = list(_lit(x["correct answer"]))
        if "expanded answer" in x:
            golds += list(_lit(x["expanded answer"]))
        out.append({
            "id": f"{name}-{x.get('id', i)}",
            "question": x["question"],
            "golds": [g for g in golds if g],
            "target": x["incorrect answer"],
            "clean": [(c.get("title", "") + "\n" + c.get("text", "")).strip() for c in ctx],
            "poison": list(_lit(x["incorrect_context"])),
        })
    return out


def poisonedrag(name: str) -> list[dict]:
    """PoisonedRAG targets on BEIR (NQ, HotpotQA, MS MARCO); clean passages from Contriever top-k."""
    adv = json.load(open(os.path.join(RAW, f"prag/adv_targeted_results_{name}.json"), encoding="utf-8"))
    corpus_path = os.path.join(RAW, f"prag/{name}_passages.json")
    passages = json.load(open(corpus_path, encoding="utf-8"))
    out = []
    for qid, x in adv.items():
        p = passages[x["id"]]
        out.append({
            "id": f"prag-{name}-{x['id']}",
            "question": x["question"],
            "golds": [x["correct answer"]],
            "target": x["incorrect answer"],
            "clean": p["clean"],
            "poison": x["adv_texts"],
        })
    return out


def with_attack(items: list[dict], k: int, n_poison: int, seed: int = 0) -> list[dict]:
    """Top-k context: injected passages rank first and push out the lowest clean ones."""
    out = []
    for x in items:
        docs = [{"text": t, "poison": True} for t in x["poison"][:n_poison]]
        docs += [{"text": t, "poison": False} for t in x["clean"][:k - len(docs)]]
        y = dict(x)
        y["docs"] = docs
        y["cond"] = f"p{n_poison}"
        out.append(y)
    return out


def ramdocs() -> list[dict]:
    rows = [json.loads(l) for l in open(os.path.join(RAW, "ramdocs/RAMDocs_test.jsonl"), encoding="utf-8")]
    out = []
    for i, x in enumerate(rows):
        docs = [{"text": d["text"], "poison": d.get("type") == "misinfo", "type": d.get("type")}
                for d in x["documents"]]
        out.append({
            "id": f"ramdocs-{i}", "question": x["question"], "golds": x["gold_answers"],
            "wrong": x.get("wrong_answers", []), "target": None, "docs": docs, "cond": "natural",
        })
    return out


def split(items: list[dict], n_dev: int, seed: int = 13) -> tuple[list[dict], list[dict]]:
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    dev = [items[i] for i in sorted(idx[:n_dev])]
    test = [items[i] for i in sorted(idx[n_dev:])]
    return dev, test
