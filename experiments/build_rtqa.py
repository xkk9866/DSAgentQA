"""Knowledge-update training questions from RealtimeQA weekly releases (2022-06 .. 2023-12).

The reader's closed-book knowledge is mostly stale for these news questions, so they teach the
aggregator when the prior should lose to the passages. Questions that appear in the RobustRAG
RealtimeQA test set are removed. Clean passages: for each Google search result, the paragraph with
the largest word overlap with the question, prefixed by the page title. The attack target is a
distractor choice of the multiple-choice question; poison passages follow the PoisonedRAG recipe.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from once.core import normalize  # noqa: E402

RAW = os.environ.get("ONCE_RAW", "D:/once_data/raw")
STOP = set("the a an of to in on for and or is was are were which what who whom when where how why this that "
           "with by from at as it its be has have had did does do new according recently".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def best_paragraph(text, q, max_chars=700):
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if len(p.strip()) > 60]
    if not paras:
        return ""
    qw = words(q)
    p = max(paras, key=lambda p: len(qw & words(p)))
    return p[:max_chars]


def select(n, seed=0):
    test = {normalize(x["question"]) for x in json.load(open(os.path.join(RAW, "RobustRAG/data/realtimeqa.json"),
                                                                 encoding="utf-8"))}
    items = []
    for f in sorted(glob.glob(os.path.join(RAW, "rtqa", "*_qa.jsonl"))):
        g = f.replace("_qa.jsonl", "_gcs.jsonl")
        if not os.path.exists(g):
            continue
        search = {}
        for l in open(g, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            search[r["question_id"]] = r.get("search_result", [])
        for l in open(f, encoding="utf-8"):
            try:
                x = json.loads(l)
            except json.JSONDecodeError:
                continue
            q = x["question_sentence"]
            if normalize(q) in test or not x.get("answer"):
                continue
            ans = int(x["answer"][0])
            choices = [c.strip().strip("\u201c\u201d\"") for c in x["choices"]]
            gold = choices[ans]
            if normalize(gold) in ("none of above", "none of the above"):
                continue
            clean = []
            for r in search.get(x["question_id"], [])[:10]:
                p = best_paragraph(r.get("text", ""), q)
                if p:
                    clean.append((r.get("title", "") + "\n" + p).strip())
            if len(clean) < 6:
                continue
            distract = [c for i, c in enumerate(choices) if i != ans and "none of" not in c.lower()]
            if not distract:
                continue
            items.append({"id": f"rtqa-{x['question_id']}", "question": q, "golds": [gold], "clean": clean,
                          "target": random.Random(x["question_id"]).choice(distract)})
    random.Random(seed).shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--select_only", action="store_true")
    ap.add_argument("--offset", type=int, default=0, help="skip questions already used by an earlier build")
    ap.add_argument("--out", default="train_rtqa_poison.json")
    args = ap.parse_args()
    items = select(args.offset + args.n)[args.offset:]
    print("questions", len(items), flush=True)
    if args.select_only:
        for x in items[:3]:
            print(x["question"], "|", x["golds"], "| target", x["target"], "\n   ", x["clean"][0][:200])
        return
    from build_train import CRAFT
    from once.llm import LLM
    llm = LLM(args.model, device=args.device)
    prompts = [llm.chat(CRAFT.format(q=x["question"], w=x["target"])) for x in items]
    texts = [llm.generate(prompts, max_new_tokens=60, batch_size=64, temperature=1.0, seed=s) for s in range(5)]
    for i, x in enumerate(items):
        x["poison"] = [t[i].strip() for t in texts]
    out = os.path.join(RAW, args.out)
    json.dump(items, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", out, len(items))


if __name__ == "__main__":
    main()
