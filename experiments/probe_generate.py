"""Val-only generation probe. The instruction maps a nonsignificant result to no."""

from __future__ import annotations

import importlib.util
import os
import re
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

SYSTEM = (
    "You answer biomedical yes/no/maybe questions from the evidence. "
    "A nonsignificant, absent, or negative result is no. "
    "An inconclusive or insufficient result is maybe. "
    "Reply with one word: yes, no, or maybe."
)

records = mod.load_records()
val = [records[i] for i in mod.split_records(records, 42)["val"][:40]]
tok = AutoTokenizer.from_pretrained(mod.QWEN_PATH, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    mod.QWEN_PATH, dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True, device_map={"": 0}
).eval()


def parse(text: str) -> str:
    match = re.search(r"\b(yes|no|maybe)\b", text.lower())
    return match.group(1) if match else "none"


correct = 0
votes = Counter()
for row in val:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Evidence:\n{row['abstract']}\n\nQuestion: {row['question']}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    pred = parse(text)
    votes[pred] += 1
    correct += int(pred == row["gold"])
    if sum(votes.values()) <= 8:
        print(row["gold"], pred, repr(text[:40]), flush=True)
print("gen", round(correct / len(val), 3), dict(votes), flush=True)
