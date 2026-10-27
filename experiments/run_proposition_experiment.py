"""Official PubMedQA split: same-proposition reading, then agreement pooling.

The 500 official test questions are scored once. Thresholds are chosen on a
stratified half of the remaining 500 questions. The other half is used only
to fit a four-class log-probability bias.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "experiments" / "run_experiment.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from caer.methods import PreparedItem, caer_output  # noqa: E402

CACHE = Path(r"D:\caer_cache\qwen3b_prop_log.npz")
QWEN = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1"
)
LABELS = ("yes", "no", "maybe")
WORDS = ("yes", "no", "maybe", "other")
SYSTEM = "Reply with one word: yes, no, maybe, or other."


def user_text(evidence: str, question: str) -> str:
    return (
        f"Evidence:\n{evidence}\n\nQuestion: {question}\n\n"
        "If the evidence answers the question, reply yes, no, or maybe. "
        "A nonsignificant or negative result is no. An inconclusive result is maybe. "
        "If the evidence is about a different question, reply other."
    )


def key_of(evidence: str, question: str) -> str:
    return hashlib.sha1(f"{evidence}\n{question}".encode("utf-8")).hexdigest()


def load_cache() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        return {}
    stored = np.load(CACHE, allow_pickle=False)
    return {key: value for key, value in zip(stored["keys"].tolist(), stored["values"])}


def save_cache(cached: dict[str, np.ndarray]) -> None:
    keys = list(cached.keys())
    values = np.stack([cached[key] for key in keys], axis=0)
    np.savez(CACHE, keys=np.array(keys), values=values)


def score_missing(pairs: list[tuple[str, str]], cached: dict[str, np.ndarray]) -> None:
    missing = [(evidence, question) for evidence, question in pairs if key_of(evidence, question) not in cached]
    print(f"pairs={len(pairs)} cached={len(pairs) - len(missing)} missing={len(missing)}", flush=True)
    if not missing:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    groups = [
        tokenizer.encode(word, add_special_tokens=False) + tokenizer.encode(word[:1].upper() + word[1:], add_special_tokens=False)
        for word in WORDS
    ]
    model = AutoModelForCausalLM.from_pretrained(
        QWEN,
        dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    for index, (evidence, question) in enumerate(missing, start=1):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text(evidence, question)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            logp = torch.log_softmax(model(**encoded).logits[0, -1].float(), dim=-1)
            scores = torch.stack([torch.logsumexp(logp[ids], dim=0) for ids in groups])
        cached[key_of(evidence, question)] = scores.detach().cpu().numpy().astype(np.float32)
        if index % 20 == 0 or index == len(missing):
            print(f"  scored {index}/{len(missing)}", flush=True)
            save_cache(cached)
    del model
    torch.cuda.empty_cache()


def softmax(rows: np.ndarray) -> np.ndarray:
    shifted = rows - rows.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_bias(log_rows: np.ndarray, golds: list[str]) -> np.ndarray:
    gold_index = np.asarray([LABELS.index(gold) for gold in golds])
    best_bias = np.zeros(4, dtype=np.float64)
    best_nll = 1e9
    grid = np.linspace(-1.5, 1.5, 7)
    for yes_bias in grid:
        for no_bias in grid:
            for maybe_bias in grid:
                bias = np.array([yes_bias, no_bias, maybe_bias, 0.0])
                probabilities = softmax(log_rows + bias)
                nll = -np.log(probabilities[np.arange(len(golds)), gold_index] + 1e-12).mean()
                if nll < best_nll:
                    best_nll = float(nll)
                    best_bias = bias
    print("calib_nll", round(best_nll, 4), "bias", [round(float(x), 3) for x in best_bias], flush=True)
    return best_bias


def opposite_finding(record: dict) -> str:
    flipped = mod.polarity_swap(record["conclusion"])
    return (
        "A replication in the same population reported the opposite result. "
        f"{flipped}"
    )


def bundles_for(records: list[dict], question_emb: np.ndarray, abstract_emb: np.ndarray) -> dict[str, dict[str, list[str]]]:
    similarity = question_emb @ abstract_emb.T
    similarity = similarity.copy()
    np.fill_diagonal(similarity, -2.0)
    golds = [row["gold"] for row in records]
    bundles = {}
    for index, record in enumerate(records):
        order = np.argsort(similarity[index])[::-1]
        nearest = [int(i) for i in order[:3]]
        opposite_scores = similarity[index].copy()
        for other, gold in enumerate(golds):
            if gold == record["gold"] or other == index:
                opposite_scores[other] = -2.0
        opposite = [int(i) for i in np.argsort(opposite_scores)[::-1][:2]]
        far = [int(i) for i in np.argsort(similarity[index])[:2]]
        gold = record["abstract"]
        bundles[record["pmid"]] = {
            "source": [gold],
            "clean": [gold, records[nearest[0]]["abstract"], records[nearest[1]]["abstract"]],
            "flip": [gold, opposite_finding(record)],
            "real_conflict": [gold, records[opposite[0]]["abstract"], records[opposite[1]]["abstract"]],
            "delete": [records[nearest[0]]["abstract"], records[nearest[1]]["abstract"], records[nearest[2]]["abstract"]],
            "irrelevant": [gold, records[far[0]]["abstract"], records[far[1]]["abstract"]],
        }
    return bundles


def pack_rows(evidences: list[str], question: str, cached: dict[str, np.ndarray], bias: np.ndarray) -> np.ndarray:
    rows = []
    for evidence in evidences:
        probabilities = softmax((cached[key_of(evidence, question)] + bias)[None, :])[0]
        packed = np.zeros((3, 3), dtype=np.float32)
        packed[:, 0] = probabilities[:3]
        rows.append(packed)
    return np.stack(rows, axis=0)


def prepare(records, indices, bundles, cached, bias):
    prepared = {condition: [] for condition in mod.CONDITIONS}
    for index in indices:
        record = records[index]
        for condition in mod.CONDITIONS:
            evidences = bundles[record["pmid"]][condition]
            item = PreparedItem.build(
                record["question"],
                evidences,
                pack_rows(evidences, record["question"], cached, bias),
            )
            prepared[condition].append(
                {
                    "pmid": record["pmid"],
                    "gold": record["gold"],
                    "mental_health": record["mental_health"],
                    "evidences": evidences,
                    "item": item,
                    "outputs": mod.method_outputs(item, LABELS),
                }
            )
    return prepared


def main() -> None:
    records = mod.load_records()
    by_pmid = {row["pmid"]: index for index, row in enumerate(records)}
    official = json.loads((ROOT / "data" / "test_ground_truth.json").read_text(encoding="utf-8"))
    test_index = [by_pmid[pmid] for pmid in official if pmid in by_pmid]
    rest = [index for index in range(len(records)) if index not in set(test_index)]
    rest_gold = [records[index]["gold"] for index in rest]
    calib_pos, val_pos = train_test_split(np.arange(len(rest)), test_size=0.5, random_state=42, stratify=rest_gold)
    calib_index = [rest[int(i)] for i in calib_pos]
    val_index = [rest[int(i)] for i in val_pos]
    print({"calib": len(calib_index), "val": len(val_index), "test": len(test_index)}, flush=True)

    question_emb, abstract_emb = mod.embed_corpus(records)
    bundles = bundles_for(records, question_emb, abstract_emb)
    needed = calib_index + val_index + test_index
    pairs = []
    seen = set()
    for index in needed:
        record = records[index]
        for evidences in bundles[record["pmid"]].values():
            for evidence in evidences:
                pair = (evidence, record["question"])
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    cached = load_cache()
    score_missing(pairs, cached)

    calib_logs = np.stack(
        [cached[key_of(records[index]["abstract"], records[index]["question"])] for index in calib_index]
    )
    bias = fit_bias(calib_logs, [records[index]["gold"] for index in calib_index])
    prepared = {
        "val": prepare(records, val_index, bundles, cached, bias),
        "test": prepare(records, test_index, bundles, cached, bias),
    }
    params = mod.tune(prepared["val"])
    mod.apply_caer_decisive(prepared["test"], params["caer"]["decisive"])
    report = {}
    for method in mod.METHODS:
        report[method] = {}
        for condition, rows in prepared["test"].items():
            report[method][condition] = {
                "forced": mod.forced_summary(rows, method),
                "selective": mod.selective_summary(rows, method, params[method]),
            }
    out = ROOT / "results" / "metrics_official.json"
    def convert(value):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        raise TypeError(type(value))

    out.write_text(
        json.dumps({"params": params, "test": report, "bias": bias.tolist()}, indent=2, default=convert),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    for method in ("caer", "majority", "selfrag_gate", "typed_yager", "mean"):
        for condition in ("source", "clean", "flip"):
            forced = report[method][condition]["forced"]
            selective = report[method][condition]["selective"]
            print(
                f"{method:14} {condition:8} acc={forced['accuracy']:.3f} f1={forced['macro_f1']:.3f} "
                f"harm={selective['harmful_error']:.3f} cov={selective['coverage']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
