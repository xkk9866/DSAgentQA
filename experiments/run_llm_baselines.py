"""Whole-bundle LLM baselines on the official PubMedQA split.

Same reader (Qwen2.5-3B-Instruct), same bundles, same calib/val/test split as
run_proposition_experiment.py.

  closed_book   question only, four-way log-probabilities
  vanilla_rag   all retrieved abstracts in one prompt, log-probabilities
  madam_debate  one agent per abstract; round 2 sees the other agents' round-1
                answers (Wang et al., 2025, MADAM-RAG); agents are averaged
  cot_rag       chain of thought over all abstracts, greedy
  self_consist  greedy chain plus four sampled chains, majority vote
                (Wang et al., 2023); answers only when 4 of 5 chains agree

Log-probability baselines get a calibration bias on calib and a confidence
threshold on val, exactly like the aggregation methods. Generative baselines
use no tuning on val; CoT answers every question.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prop", ROOT / "experiments" / "run_proposition_experiment.py")
prop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prop)
mod = prop.mod

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

LOG_CACHE = Path(r"D:\caer_cache\qwen3b_baseline_log.npz")
GEN_CACHE = Path(r"D:\caer_cache\qwen3b_baseline_gen.json")
LABELS = prop.LABELS
GEN_CONDITIONS = ("source", "clean", "flip", "real_conflict")
DEBATE_CONDITIONS = ("clean", "flip", "real_conflict", "irrelevant", "delete")
SC_SAMPLES = 4
SC_AGREE = 4


def docs_block(evidences: list[str]) -> str:
    return "\n\n".join(f"[Document {i + 1}]\n{text}" for i, text in enumerate(evidences))


def closed_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": prop.SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nAnswer from your medical knowledge. Reply yes, no, or maybe."},
    ]


def rag_prompt(evidences: list[str], question: str) -> list[dict]:
    return [
        {"role": "system", "content": prop.SYSTEM},
        {"role": "user", "content": prop.user_text(docs_block(evidences), question)},
    ]


def debate_prompt(evidence: str, others: list[str], question: str) -> list[dict]:
    seen = "; ".join(f"agent {i + 1} answered {answer}" for i, answer in enumerate(others)) or "none"
    text = (
        prop.user_text(evidence, question)
        + f"\n\nOther agents, each reading a different document: {seen}. "
        "Keep your answer if your document supports it; change it only if your document is uninformative."
    )
    return [{"role": "system", "content": prop.SYSTEM}, {"role": "user", "content": text}]


COT_SYSTEM = "You are a careful biomedical evidence reviewer."


def cot_prompt(evidences: list[str], question: str) -> list[dict]:
    text = (
        f"{docs_block(evidences)}\n\nQuestion: {question}\n\n"
        "The documents may disagree. Reason step by step in at most five sentences about which "
        "evidence answers this exact question. A nonsignificant or negative result is no; an "
        "inconclusive result is maybe. End with a final line 'Answer: yes', 'Answer: no', or 'Answer: maybe'."
    )
    return [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": text}]


def key_of(messages: list[dict]) -> str:
    return hashlib.sha1(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()


class Reader:
    def __init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(prop.QWEN, local_files_only=True)
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.groups = [
            self.tok.encode(w, add_special_tokens=False) + self.tok.encode(w[:1].upper() + w[1:], add_special_tokens=False)
            for w in prop.WORDS
        ]
        self.model = None
        self.log = {}
        if LOG_CACHE.exists():
            stored = np.load(LOG_CACHE, allow_pickle=False)
            self.log = dict(zip(stored["keys"].tolist(), stored["values"]))
        self.gen = json.loads(GEN_CACHE.read_text(encoding="utf-8")) if GEN_CACHE.exists() else {}

    def ensure(self) -> None:
        if self.model is None:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                prop.QWEN, dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True
            ).cuda().eval()

    def save(self) -> None:
        if self.log:
            keys = list(self.log)
            np.savez(LOG_CACHE, keys=np.array(keys), values=np.stack([self.log[k] for k in keys]))
        GEN_CACHE.write_text(json.dumps(self.gen), encoding="utf-8")

    def text(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def logprobs(self, prompts: list[list[dict]], tag: str) -> list[np.ndarray]:
        missing = [p for p in {key_of(p): p for p in prompts}.values() if key_of(p) not in self.log]
        print(f"[{tag}] logprob prompts={len(prompts)} missing={len(missing)}", flush=True)
        if missing:
            self.ensure()
        for index, messages in enumerate(missing, start=1):
            enc = self.tok(self.text(messages), return_tensors="pt", truncation=True, max_length=3072).to("cuda")
            with torch.no_grad():
                logp = torch.log_softmax(self.model(**enc).logits[0, -1].float(), dim=-1)
                scores = torch.stack([torch.logsumexp(logp[ids], dim=0) for ids in self.groups])
            self.log[key_of(messages)] = scores.cpu().numpy().astype(np.float32)
            if index % 50 == 0 or index == len(missing):
                print(f"  [{tag}] {index}/{len(missing)}", flush=True)
                self.save()
        return [self.log[key_of(p)] for p in prompts]

    def generate(self, prompts: list[list[dict]], tag: str, samples: int, batch: int) -> list[list[str]]:
        cache_tag = f"{tag}:{samples}"
        missing = [p for p in {key_of(p): p for p in prompts}.values() if f"{cache_tag}:{key_of(p)}" not in self.gen]
        print(f"[{tag}] generate prompts={len(prompts)} missing={len(missing)}", flush=True)
        if missing:
            self.ensure()
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            enc = self.tok([self.text(m) for m in chunk], return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
            kwargs = dict(max_new_tokens=192, pad_token_id=self.tok.pad_token_id)
            if samples == 0:
                kwargs.update(do_sample=False, num_return_sequences=1)
            else:
                kwargs.update(do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=samples)
            with torch.no_grad():
                out = self.model.generate(**enc, **kwargs)
            texts = self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            per = max(samples, 1)
            for j, messages in enumerate(chunk):
                self.gen[f"{cache_tag}:{key_of(messages)}"] = texts[j * per : (j + 1) * per]
            done = min(start + batch, len(missing))
            print(f"  [{tag}] {done}/{len(missing)}", flush=True)
            self.save()
        return [self.gen[f"{cache_tag}:{key_of(p)}"] for p in prompts]


ANSWER = re.compile(r"answer\s*[:：]?\s*\**\s*(yes|no|maybe)", re.IGNORECASE)


def parse(text: str) -> str | None:
    found = ANSWER.findall(text)
    if found:
        return found[-1].lower()
    tail = re.findall(r"\b(yes|no|maybe)\b", text[-80:].lower())
    return tail[-1] if tail else None


def three_way(rows: np.ndarray, bias: np.ndarray) -> np.ndarray:
    probabilities = prop.softmax(rows + bias)
    return probabilities[:, :3] / probabilities[:, :3].sum(axis=1, keepdims=True)


def threshold_output(probs: np.ndarray) -> dict:
    return {"probs": probs, "confidence": float(probs.max()), "gate": "threshold"}


def tune_threshold(clean_rows: list[dict], flip_rows: list[dict], method: str) -> dict:
    best = None
    for threshold in mod.THRESHOLDS:
        score = mod.utility(clean_rows, flip_rows, method, threshold, 0.0, 0.0)
        if best is None or score > best[0] + 1e-12:
            best = (score, threshold)
    return {"utility": best[0], "threshold": best[1], "conflict_slack": 0.0, "margin": 0.0, "gate": "threshold"}


def vote_summary(golds: list[str], answers: list[list[str | None]], agree: int | None) -> dict:
    forced, selective = [], []
    for votes in answers:
        valid = [v for v in votes if v is not None]
        if not valid:
            forced.append("none")
            selective.append(None)
            continue
        counts = {label: valid.count(label) for label in LABELS}
        top = max(LABELS, key=lambda label: (counts[label], label == votes[0]))
        forced.append(top)
        selective.append(top if agree is None or counts[top] >= agree else None)
    return {
        "forced": {"accuracy": mod.accuracy(golds, forced), "macro_f1": mod.macro_f1(golds, forced, LABELS),
                   "unparsed": float(np.mean([f == "none" for f in forced]))},
        "selective": mod.selective_stats(golds, selective),
    }


def main() -> None:
    records = mod.load_records()
    by_pmid = {row["pmid"]: index for index, row in enumerate(records)}
    official = json.loads((ROOT / "data" / "test_ground_truth.json").read_text(encoding="utf-8"))
    test_index = [by_pmid[pmid] for pmid in official if pmid in by_pmid]
    test_set = set(test_index)
    rest = [index for index in range(len(records)) if index not in test_set]
    calib_pos, val_pos = train_test_split(
        np.arange(len(rest)), test_size=0.5, random_state=42, stratify=[records[i]["gold"] for i in rest]
    )
    split = {
        "calib": [rest[int(i)] for i in calib_pos],
        "val": [rest[int(i)] for i in val_pos],
        "test": test_index,
    }
    question_emb, abstract_emb = mod.embed_corpus(records)
    bundles = prop.bundles_for(records, question_emb, abstract_emb)
    doc_cache = prop.load_cache()
    calib_logs = np.stack([doc_cache[prop.key_of(records[i]["abstract"], records[i]["question"])] for i in split["calib"]])
    doc_bias = prop.fit_bias(calib_logs, [records[i]["gold"] for i in split["calib"]])
    reader = Reader()
    gold = {name: [records[i]["gold"] for i in idx] for name, idx in split.items()}

    rows_by = {name: {c: [{"gold": records[i]["gold"], "outputs": {}} for i in idx] for c in mod.CONDITIONS} for name, idx in split.items()}
    params = {}

    # closed book
    logs = {name: np.stack(reader.logprobs([closed_prompt(records[i]["question"]) for i in idx], f"closed-{name}")) for name, idx in split.items()}
    bias = prop.fit_bias(logs["calib"], gold["calib"])
    for name in ("val", "test"):
        probs = three_way(logs[name], bias)
        for condition in mod.CONDITIONS:
            for row, p in zip(rows_by[name][condition], probs):
                row["outputs"]["closed_book"] = threshold_output(p)

    # vanilla RAG
    calib_rag = np.stack(reader.logprobs([rag_prompt(bundles[records[i]["pmid"]]["clean"], records[i]["question"]) for i in split["calib"]], "rag-calib"))
    bias = prop.fit_bias(calib_rag, gold["calib"])
    for name in ("val", "test"):
        for condition in mod.CONDITIONS:
            prompts = [rag_prompt(bundles[records[i]["pmid"]][condition], records[i]["question"]) for i in split[name]]
            probs = three_way(np.stack(reader.logprobs(prompts, f"rag-{name}-{condition}")), bias)
            for row, p in zip(rows_by[name][condition], probs):
                row["outputs"]["vanilla_rag"] = threshold_output(p)

    # MADAM-RAG style debate
    def first_round(evidence: str, question: str) -> tuple[str, np.ndarray]:
        p = prop.softmax((doc_cache[prop.key_of(evidence, question)] + doc_bias)[None, :])[0]
        return prop.WORDS[int(np.argmax(p))], p

    for name in ("val", "test"):
        for condition in mod.CONDITIONS:
            prompts, owners = [], []
            for row_index, i in enumerate(split[name]):
                question = records[i]["question"]
                evidences = bundles[records[i]["pmid"]][condition]
                answers = [first_round(e, question)[0] for e in evidences]
                for k, evidence in enumerate(evidences):
                    if condition in DEBATE_CONDITIONS and len(evidences) > 1:
                        prompts.append(debate_prompt(evidence, answers[:k] + answers[k + 1 :], question))
                        owners.append((row_index, None))
                    else:
                        owners.append((row_index, first_round(evidence, question)[1]))
            scored = iter(reader.logprobs(prompts, f"debate-{name}-{condition}")) if prompts else iter(())
            pooled = [np.zeros(3) for _ in split[name]]
            for row_index, ready in owners:
                p = ready if ready is not None else prop.softmax((next(scored) + doc_bias)[None, :])[0]
                pooled[row_index] += p[:3]
            for row, total in zip(rows_by[name][condition], pooled):
                row["outputs"]["madam_debate"] = threshold_output(total / total.sum())

    for method in ("closed_book", "vanilla_rag", "madam_debate"):
        params[method] = tune_threshold(rows_by["val"]["clean"], rows_by["val"]["flip"], method)
        print(f"tune {method}: {params[method]}", flush=True)

    report = {}
    for method in ("closed_book", "vanilla_rag", "madam_debate"):
        report[method] = {
            c: {"forced": mod.forced_summary(rows, method), "selective": mod.selective_summary(rows, method, params[method])}
            for c, rows in rows_by["test"].items()
        }

    # generative baselines on test
    report["cot_rag"], report["self_consistency"] = {}, {}
    for condition in GEN_CONDITIONS:
        prompts = [cot_prompt(bundles[records[i]["pmid"]][condition], records[i]["question"]) for i in split["test"]]
        greedy = reader.generate(prompts, f"cot-{condition}", 0, 8)
        sampled = reader.generate(prompts, f"sc-{condition}", SC_SAMPLES, 4)
        cot_votes = [[parse(g[0])] for g in greedy]
        sc_votes = [[parse(g[0])] + [parse(t) for t in s] for g, s in zip(greedy, sampled)]
        report["cot_rag"][condition] = vote_summary(gold["test"], cot_votes, None)
        report["self_consistency"][condition] = vote_summary(gold["test"], sc_votes, SC_AGREE)
    reader.save()

    out = ROOT / "results" / "metrics_llm_baselines.json"
    out.write_text(json.dumps({"params": params, "test": report}, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    for method, by_condition in report.items():
        for condition in GEN_CONDITIONS:
            f, s = by_condition[condition]["forced"], by_condition[condition]["selective"]
            print(f"{method:16} {condition:13} acc={f['accuracy']:.3f} f1={f['macro_f1']:.3f} "
                  f"harm={s['harmful_error']:.3f} cov={s['coverage']:.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
