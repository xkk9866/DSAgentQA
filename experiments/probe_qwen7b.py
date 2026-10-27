"""Same four-way prompt as the 3B reader, on 100 official test abstracts.

Prints 7B accuracy next to the cached 3B accuracy on those same questions.
Does not touch the 3B cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import importlib.util

spec = importlib.util.spec_from_file_location("prop", ROOT / "experiments" / "run_proposition_experiment.py")
prop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prop)

QWEN7 = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
)
N = 100


def main() -> None:
    records = prop.mod.load_records()
    by_pmid = {row["pmid"]: index for index, row in enumerate(records)}
    official = json.loads((ROOT / "data" / "test_ground_truth.json").read_text(encoding="utf-8"))
    test_index = [by_pmid[pmid] for pmid in official if pmid in by_pmid][:N]
    cached = prop.load_cache()
    bias = np.array([1.5, 1.0, -1.5, 0.0])
    three_b = []
    for index in test_index:
        record = records[index]
        probabilities = prop.softmax((cached[prop.key_of(record["abstract"], record["question"])] + bias)[None, :])[0]
        three_b.append(prop.WORDS[int(np.argmax(probabilities))] == record["gold"])
    print(f"3B same {N}: {float(np.mean(three_b)):.3f}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN7, local_files_only=True)
    groups = [
        tokenizer.encode(word, add_special_tokens=False)
        + tokenizer.encode(word[:1].upper() + word[1:], add_special_tokens=False)
        for word in prop.WORDS
    ]
    model = AutoModelForCausalLM.from_pretrained(
        QWEN7, dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True
    ).cuda().eval()
    hits = []
    for index, row_index in enumerate(test_index, start=1):
        record = records[row_index]
        messages = [
            {"role": "system", "content": prop.SYSTEM},
            {"role": "user", "content": prop.user_text(record["abstract"], record["question"])},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            logp = torch.log_softmax(model(**encoded).logits[0, -1].float(), dim=-1)
            scores = torch.stack([torch.logsumexp(logp[ids], dim=0) for ids in groups])
        adjusted = scores.detach().cpu().numpy() + bias
        hits.append(prop.WORDS[int(np.argmax(adjusted))] == record["gold"])
        if index % 20 == 0:
            print(f"  7B {index}/{N} running={float(np.mean(hits)):.3f}", flush=True)
    print(f"7B same {N}: {float(np.mean(hits)):.3f}", flush=True)


if __name__ == "__main__":
    main()
