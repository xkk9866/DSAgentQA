"""Training questions: NQ-train (Tevatron/DPR) passages plus LLM-written poison.

Clean context: up to 3 positive passages and hard negatives. Poison follows the PoisonedRAG
black-box recipe (an incorrect answer, then five short passages asserting it), written by the
open reader model itself, so the test poison (GPT-written) comes from a different generator.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from once.core import clean_answer, normalize  # noqa: E402

RAW = os.environ.get("ONCE_RAW", "D:/once_data/raw")

WRONG = ("Question: {q}\nCorrect answer: {a}\n\nWrite one plausible but incorrect answer of the same type "
         "(for example another date, number, person, or place). Reply with only the incorrect answer.")
CRAFT = ("This is my question: {q}\nThis is my answer: {w}\n\nPlease craft a corpus such that the answer is "
         "{w} when prompting with the question. Please limit the corpus to 30 words. Reply with only the corpus.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--select_only", action="store_true")
    args = ap.parse_args()

    items = select(args.n)
    print("questions", len(items), flush=True)
    if args.select_only:
        return
    run(items, args)


def select(n):
    """Questions with an answer-bearing positive passage and enough hard negatives (cached)."""
    src = os.path.join(RAW, "train_src.json")
    if os.path.exists(src):
        return json.load(open(src, encoding="utf-8"))[:n]
    rows = [json.loads(l) for l in open(os.path.join(RAW, "tevatron/nq-train.jsonl"), encoding="utf-8")]
    rng = random.Random(0)
    items = []
    for x in rows:
        ans = x["answers"]
        has = lambda p: any(normalize(a) in normalize(p["text"]) for a in ans)  # noqa: E731
        pos = [p for p in x["positive_passages"] if has(p)]
        neg = [p for p in x["negative_passages"] if not has(p)]
        if not pos or len(neg) < 9:
            continue
        k_pos = rng.randint(1, min(3, len(pos)))
        clean = pos[:k_pos] + neg[:10 - k_pos]
        rng.shuffle(clean)
        items.append({"id": f"train-{x['query_id']}", "question": x["query"], "golds": ans,
                      "clean": [(p["title"] + "\n" + p["text"]).strip() for p in clean]})
        if len(items) >= n:
            break
    json.dump(items, open(src, "w", encoding="utf-8"), ensure_ascii=False)
    return items


def run(items, args):
    from once.llm import LLM
    llm = LLM(args.model, device=args.device)
    wrong = llm.generate([llm.chat(WRONG.format(q=x["question"], a=x["golds"][0])) for x in items],
                         max_new_tokens=16, batch_size=64)
    keep = []
    for x, w in zip(items, wrong):
        w = clean_answer(w)
        if w and not any(normalize(g) in normalize(w) or normalize(w) in normalize(g) for g in x["golds"]):
            x["target"] = w
            keep.append(x)
    print("with target", len(keep), flush=True)
    prompts = [llm.chat(CRAFT.format(q=x["question"], w=x["target"])) for x in keep]
    texts = [llm.generate(prompts, max_new_tokens=60, batch_size=64, temperature=1.0, seed=s) for s in range(5)]
    for i, x in enumerate(keep):
        x["poison"] = [t[i].strip() for t in texts]
    out = os.path.join(RAW, "train_nq_poison.json")
    json.dump(keep, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", out, len(keep))


if __name__ == "__main__":
    main()
