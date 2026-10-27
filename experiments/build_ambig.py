"""Multi-answer training questions from AmbigDocs-train, built the way RAMDocs is built.

Each question names an ambiguous entity; every valid document answers it for a different referent,
so all valid answers are gold. Misinformation documents copy a valid document and replace its answer
with an answer of another question; noise documents come from unrelated questions. Questions whose
text appears in RAMDocs are removed.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from once.core import normalize  # noqa: E402

RAW = os.environ.get("ONCE_RAW", "D:/once_data/raw")


def main(n=600, seed=0):
    rng = random.Random(seed)
    test = {normalize(json.loads(l)["question"]) for l in open(os.path.join(RAW, "ramdocs/RAMDocs_test.jsonl"),
                                                                 encoding="utf-8")}
    rows = json.load(open(os.path.join(RAW, "ambigdocs/train.json"), encoding="utf-8"))
    pool = [r for r in rows if normalize(r["question"]) not in test and len(r["documents"]) >= 1]
    rng.shuffle(pool)
    other_answers = [d["answer"] for r in pool[:5000] for d in r["documents"]]
    items = []
    for i, r in enumerate(pool):
        docs = [d for d in r["documents"] if d.get("answer") and d["answer"].lower() in d["text"].lower()]
        seen, valid = set(), []
        for d in docs:  # one document per distinct answer
            k = normalize(d["answer"])
            if k and k not in seen:
                seen.add(k)
                valid.append(d)
        if not valid:
            continue
        valid = valid[:rng.randint(1, min(3, len(valid)))]
        out, wrong = [], []
        for d in valid:
            out.append({"text": f"{d['title']}\n{d['text']}", "poison": False, "type": "correct"})
        for _ in range(rng.randint(0, 2)):  # misinformation: a valid document with its answer swapped
            d = rng.choice(valid)
            w = rng.choice(other_answers)
            if normalize(w) in seen:
                continue
            text = re.sub(re.escape(d["answer"]), w, f"{d['title']}\n{d['text']}", flags=re.I)
            out.append({"text": text, "poison": True, "type": "misinfo"})
            wrong.append(w)
        for _ in range(rng.randint(0, 2)):  # noise: a document about another question
            o = rng.choice(pool)
            if o["documents"]:
                od = rng.choice(o["documents"])
                out.append({"text": f"{od['title']}\n{od['text']}", "poison": False, "type": "noise"})
        rng.shuffle(out)
        items.append({"id": f"ambig-{r['qid']}", "question": r["question"], "golds": [d["answer"] for d in valid],
                      "wrong": wrong, "target": wrong[0] if wrong else None, "docs": out})
        if len(items) >= n:
            break
    json.dump(items, open(os.path.join(RAW, "train_ambig.json"), "w", encoding="utf-8"), ensure_ascii=False)
    import collections
    print(len(items), "golds/q", collections.Counter(len(x["golds"]) for x in items),
          "docs/q", sum(len(x["docs"]) for x in items) / len(items))


if __name__ == "__main__":
    main()
