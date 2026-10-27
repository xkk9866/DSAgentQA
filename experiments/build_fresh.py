"""Fresh test set, built after the method was frozen: RealtimeQA news questions never used for training or testing.

Poison passages follow the PoisonedRAG black-box recipe but are written by Qwen3-8B, a model different from the reader
(which wrote the training poison) and from the GPT models that wrote the other test sets.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_train import CRAFT  # noqa: E402
from once.grid import load_readings  # noqa: E402
from once.llm import LLM  # noqa: E402

RAW = "D:/once_data/raw"


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    items = json.load(open(os.path.join(RAW, "train_rtqa2_items.json"), encoding="utf-8"))
    seen = {r["question"] for d in ("train-rtqa", "rqa", "train_v4") for r in load_readings("qwen2.5-7b-int8", d)}
    items = [x for x in items if x["question"] not in seen and len(x["clean"]) >= 6 and x.get("target")]
    for x in items:
        x["id"] = "fresh-" + x["id"]
    llm = LLM("qwen3-8b", device=dev)
    prompts = [llm.chat(CRAFT.format(q=x["question"], w=x["target"])) for x in items]
    texts = [llm.generate(prompts, max_new_tokens=80, batch_size=32, temperature=1.0, seed=s) for s in range(5)]
    for i, x in enumerate(items):
        x["poison"] = [t[i].strip() for t in texts]
    json.dump(items, open(os.path.join(RAW, "fresh_rqa_poison.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("fresh items", len(items))


if __name__ == "__main__":
    main()
