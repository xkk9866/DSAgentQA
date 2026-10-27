"""Val-only probe: full abstract, four-way same-proposition reading."""

from __future__ import annotations

import importlib.util
import os
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "experiments" / "run_experiment.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

records = mod.load_records()
val = [records[i] for i in mod.split_records(records, 42)["val"][:40]]
print("n", len(val), "chars", int(sum(len(r["abstract"]) for r in val) / len(val)), flush=True)

tok = AutoTokenizer.from_pretrained(mod.QWEN_PATH, local_files_only=True)
tok.padding_side = "left"
tok.truncation_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
words = ("yes", "no", "maybe", "other")
groups = [tok.encode(w, add_special_tokens=False) + tok.encode(w[:1].upper() + w[1:], add_special_tokens=False) for w in words]
model = AutoModelForCausalLM.from_pretrained(
    mod.QWEN_PATH, dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True, device_map={"": 0}
).eval()

labels = ("yes", "no", "maybe")
correct = 0
votes = Counter()
for row in val:
    messages = [
        {"role": "system", "content": "Reply with one word: yes, no, maybe, or other."},
        {
            "role": "user",
            "content": (
                f"Evidence:\n{row['abstract']}\n\nQuestion: {row['question']}\n\n"
                "If the evidence answers the question, reply yes, no, or maybe. "
                "If it is about a different question, reply other."
            ),
        },
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        logp = torch.log_softmax(model(**encoded).logits[0, -1].float(), dim=-1)
        scores = torch.stack([torch.logsumexp(logp[ids], dim=0) for ids in groups])
        probs = torch.softmax(scores, dim=0)
    pred = words[int(torch.argmax(probs))]
    votes[pred] += 1
    same = probs[:3]
    answered = labels[int(torch.argmax(same))]
    correct += int(answered == row["gold"] and pred != "other")
    if sum(votes.values()) <= 8:
        print(row["gold"], pred, [round(float(x), 3) for x in probs], row["question"][:80], flush=True)
print("strict", round(correct / len(val), 3), dict(votes), flush=True)
