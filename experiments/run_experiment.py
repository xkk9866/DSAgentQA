"""Run CAER against fusion and routing baselines on PubMedQA.

The evidence encoder is shared. Methods differ in how they discount sources,
combine mass functions, and turn residual conflict into an answer or an abstention.
Thresholds are chosen on the validation split and applied once to the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caer.fusion import LABELS
from caer.methods import PreparedItem, caer_output, leave_one_out_citation, method_outputs, selective_decision
from caer.metrics import (
    accuracy,
    aurc,
    bootstrap_delta,
    brier_score,
    citation_scores,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
    selective_stats,
)
from caer.safety import evaluate_crisis_gate

NLI_PATH = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--cross-encoder--nli-deberta-v3-base"
    r"\snapshots\6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
)
DATA_PATH = ROOT / "data" / "ori_pqal.json"
FALLBACK_DATA = Path(r"D:\caer_tmp_pq.json")
CACHE_DIR = Path(r"D:\caer_cache")
RESULT_DIR = ROOT / "results"

METHODS = (
    "mean",
    "top1",
    "majority",
    "dempster",
    "yager",
    "crag",
    "selfrag_gate",
    "kg_filter",
    "smets_cdg",
    "typed_yager",
    "caer",
)
CONDITIONS = ("source", "clean", "flip", "real_conflict", "delete", "irrelevant")
THRESHOLDS = (0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
SLACKS = (-0.05, 0.0, 0.05, 0.10, 0.15, 0.25)
MARGINS = (0.0, 0.05, 0.10, 0.20)
DECISIVES = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)
MAX_CONFLICTS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 1.0)
MH_PATTERN_TERMS = (
    "depression",
    "depressive",
    "antidepress",
    "anxiety",
    "anxious",
    "ssri",
    "snri",
    "schizophren",
    "bipolar",
    "psychiatr",
    "psychosis",
    "psychotic",
    "ptsd",
    "post-traumatic",
    "adhd",
    "obsessive",
    "suicid",
    "psychotherap",
    "mental health",
    "antipsychotic",
    "anxiolytic",
    "panic disorder",
    "anorexia",
    "bulimia",
    "lithium",
    "mood disorder",
    "autism",
    "cognitive behavioral",
    "cognitive behavioural",
)

RELATION_PROBES = [
    {
        "question": "Is combining phenelzine and sertraline safe in adults?",
        "gold": "no",
        "evidences": [
            "In adult patients, phenelzine combined with sertraline is contraindicated because of serotonin syndrome.",
            "A commentary states that the combination of phenelzine and sertraline is safe and recommended.",
        ],
    },
    {
        "question": "Is combining tranylcypromine and fluoxetine safe?",
        "gold": "no",
        "evidences": [
            "Tranylcypromine with fluoxetine must not be combined. The combination can cause serotonin syndrome.",
            "The combination of tranylcypromine and fluoxetine is safe and well tolerated.",
        ],
    },
    {
        "question": "Can adults take isocarboxazid together with paroxetine?",
        "gold": "no",
        "evidences": [
            "Isocarboxazid and paroxetine are contraindicated in combination.",
            "Combining isocarboxazid with paroxetine is safe and effective.",
        ],
    },
    {
        "question": "Is moclobemide with venlafaxine a safe combination?",
        "gold": "no",
        "evidences": [
            "Moclobemide coadministered with venlafaxine risks serotonin syndrome and is contraindicated.",
            "Moclobemide with venlafaxine is safe and recommended for adults.",
        ],
    },
    {
        "question": "Is clozapine with carbamazepine appropriate?",
        "gold": "no",
        "evidences": [
            "Clozapine and carbamazepine should not be combined because of hematologic risk.",
            "The combination of clozapine and carbamazepine is safe and recommended.",
        ],
    },
]


def load_records() -> list[dict]:
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for pmid, row in raw.items():
        contexts = row["CONTEXTS"]
        abstract = " ".join(contexts)
        records.append(
            {
                "pmid": pmid,
                "question": row["QUESTION"],
                "abstract": abstract,
                "findings": finding_sentences(contexts, row.get("LABELS") or []),
                "conclusion": contexts[-1],
                "gold": row["final_decision"],
                "meshes": row.get("MESHES") or [],
                "mental_health": is_mental_health(row["QUESTION"], abstract, row.get("MESHES") or []),
            }
        )
    return records


def finding_sentences(contexts: list[str], labels: list[str]) -> list[str]:
    chosen = [
        context
        for context, label in zip(contexts, labels)
        if str(label).upper() in {"RESULTS", "CONCLUSIONS", "CONCLUSION", "FINDINGS", "PRINCIPAL FINDINGS"}
    ]
    if not chosen:
        chosen = contexts[-2:]
    return [context[:1600] for context in chosen[:2]]


def is_mental_health(question: str, abstract: str, meshes: list[str]) -> bool:
    text = " ".join([question, abstract, *meshes]).lower()
    if any(term in text for term in MH_PATTERN_TERMS):
        if "depress" in text and ("st segment" in text or "st-segment" in text or "st depression" in text):
            if not any(term in text for term in ("antidepress", "depressive disorder", "major depression", "psychiatr")):
                return False
        return True
    return False


def split_records(records: list[dict], seed: int) -> dict[str, list[int]]:
    indices = np.arange(len(records))
    golds = [records[i]["gold"] for i in indices]
    trainval, test = train_test_split(indices, test_size=0.5, random_state=seed, stratify=golds)
    trainval_golds = [records[i]["gold"] for i in trainval]
    train, val = train_test_split(trainval, test_size=0.5, random_state=seed, stratify=trainval_golds)
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def embed_corpus(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "embeddings_minilm.npy"
    if cache.exists():
        packed = np.load(cache)
        dim = packed.shape[1] // 2
        return packed[:, :dim], packed[:, dim:]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda", local_files_only=True)
    questions = model.encode(
        [row["question"] for row in records],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    abstracts = model.encode(
        [row["abstract"] for row in records],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(cache, np.concatenate([questions, abstracts], axis=1).astype(np.float32))
    return np.asarray(questions), np.asarray(abstracts)


def opposite_label(label: str) -> str:
    return {"yes": "no", "no": "yes", "maybe": "yes"}[label]


def polarity_swap(text: str) -> str:
    swaps = (
        ("significantly increased", "significantly decreased"),
        ("significantly decreased", "significantly increased"),
        ("significant increase", "significant decrease"),
        ("significant decrease", "significant increase"),
        ("significantly higher", "significantly lower"),
        ("significantly lower", "significantly higher"),
        ("was associated", "was not associated"),
        ("were associated", "were not associated"),
        ("is associated", "is not associated"),
        ("improved", "did not improve"),
        ("reduced", "did not reduce"),
        ("effective", "not effective"),
    )
    lowered = text.lower()
    for source, target in swaps:
        position = lowered.find(source)
        if position >= 0:
            return text[:position] + target + text[position + len(source) :]
    return text


def counterfactual_text(record: dict) -> str:
    opposite = opposite_label(record["gold"])
    flipped = polarity_swap(record["conclusion"])
    return (
        f"Replication study of the question: {record['question']} "
        f"A matched cohort in the same population reported the opposite finding. {flipped} "
        f"Therefore the answer is {opposite}."
    )


def build_bundles(records: list[dict], question_emb: np.ndarray, abstract_emb: np.ndarray) -> dict[str, dict[str, list[str]]]:
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
        gold_block = " ".join(record["findings"])[:2000]

        def neighbor_block(other: int) -> str:
            return " ".join(records[other]["findings"])[:2000]

        def keep(parts: list[str]) -> list[str]:
            return [part for part in parts if part.strip()]

        bundles[record["pmid"]] = {
            "source": keep([gold_block]),
            "clean": keep([gold_block, neighbor_block(nearest[0]), neighbor_block(nearest[1])]),
            "flip": keep([gold_block, counterfactual_text(record)]),
            "real_conflict": keep([gold_block, neighbor_block(opposite[0]), neighbor_block(opposite[1])]),
            "delete": keep([neighbor_block(nearest[0]), neighbor_block(nearest[1]), neighbor_block(nearest[2])]),
            "irrelevant": keep([gold_block, neighbor_block(far[0]), neighbor_block(far[1])]),
        }
    return bundles


def claim_from_question(question: str) -> str:
    text = question.strip()
    if text.endswith("?"):
        text = text[:-1].strip()
    parts = text.split(" ", 1)
    first = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else text
    if first in {"do", "did", "can", "could", "should", "would", "does"}:
        statement = rest[:1].upper() + rest[1:]
    elif first in {"is", "are", "was", "were", "has", "have", "had"}:
        bits = rest.split(" ", 1)
        tail = bits[1] if len(bits) > 1 else ""
        statement = f"{bits[0]} {first} {tail}".strip()
    else:
        statement = text
    return statement.rstrip(".") + "."


def hypotheses_for(question: str) -> dict[str, str]:
    claim = claim_from_question(question)
    lowered = claim[:1].lower() + claim[1:]
    return {
        "yes": claim,
        "no": f"It is not the case that {lowered}",
        "maybe": f"It is uncertain whether {lowered}",
    }


def hypothesis(question: str, label: str) -> str:
    return hypotheses_for(question)[label]


def score_pairs(pairs: list[tuple[str, str]]) -> np.ndarray:
    import hashlib
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "nli_cache_doc.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        packed = np.load(cache_path, allow_pickle=True)
        keys = packed["keys"].tolist()
        values = packed["values"]
        cached = {key: values[i] for i, key in enumerate(keys)}

    def pair_key(premise: str, hypo: str) -> str:
        return hashlib.sha1(f"{premise}\n{hypo}".encode("utf-8")).hexdigest()

    missing = [pair for pair in pairs if pair_key(*pair) not in cached]
    print(f"nli pairs={len(pairs)} cached={len(pairs) - len(missing)} missing={len(missing)}", flush=True)
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(NLI_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_PATH, local_files_only=True)
        model.eval().cuda().half()
        id2label = {int(key): value.lower() for key, value in model.config.id2label.items()}
        entail_index = next(index for index, name in id2label.items() if "entail" in name)
        neutral_index = next(index for index, name in id2label.items() if "neutral" in name)
        contra_index = next(index for index, name in id2label.items() if "contra" in name)
        started = time.perf_counter()
        batch_size = 16
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            encoded = tokenizer(
                [premise for premise, _ in batch],
                [hypo for _, hypo in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.cuda() for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits.float()
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            ordered = np.stack(
                [probabilities[:, entail_index], probabilities[:, neutral_index], probabilities[:, contra_index]],
                axis=1,
            )
            for pair, row in zip(batch, ordered):
                cached[pair_key(*pair)] = row.astype(np.float32)
            done = start + len(batch)
            if start % (batch_size * 20) == 0:
                print(f"  nli {done}/{len(missing)}", flush=True)
            if done % 640 == 0 or done == len(missing):
                keys = list(cached.keys())
                values = np.stack([cached[key] for key in keys], axis=0)
                np.savez(cache_path, keys=np.array(keys), values=values)
        print(f"nli seconds={time.perf_counter() - started:.1f}", flush=True)
        keys = list(cached.keys())
        values = np.stack([cached[key] for key in keys], axis=0)
        np.savez(cache_path, keys=np.array(keys), values=values)

    return np.stack([cached[pair_key(*pair)] for pair in pairs], axis=0)


QWEN_PATH = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
)


def score_qwen(pairs: list[tuple[str, str]]) -> np.ndarray:
    """One forward per (sentence, question). Returns yes/no/maybe probabilities.

    Each row is the renormalized next-token mass on yes/Yes, no/No, and maybe/Maybe.
    The original question is used. A declarative rewrite drops the auxiliary verb and
    makes the reader answer "maybe" to an ungrammatical claim.
    """
    import hashlib
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "qwen25_7b_question.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=False)
        cached = {key: value for key, value in zip(stored["keys"].tolist(), stored["values"])}

    def group_key(premise: str, question: str) -> str:
        return hashlib.sha1(f"{premise}\n{question}".encode("utf-8")).hexdigest()

    groups: dict[str, tuple[str, str]] = {}
    for premise, question in pairs:
        groups.setdefault(group_key(premise, question), (premise, question))
    missing = [item for key, item in groups.items() if key not in cached]
    print(f"qwen groups={len(groups)} cached={len(groups) - len(missing)} missing={len(missing)}", flush=True)
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        label_ids = [
            tokenizer.encode("yes", add_special_tokens=False) + tokenizer.encode("Yes", add_special_tokens=False),
            tokenizer.encode("no", add_special_tokens=False) + tokenizer.encode("No", add_special_tokens=False),
            tokenizer.encode("maybe", add_special_tokens=False) + tokenizer.encode("Maybe", add_special_tokens=False),
        ]
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
        model.eval()
        started = time.perf_counter()
        batch_size = 2
        system = "Answer a biomedical question from the evidence only. Reply with one word: yes, no, or maybe."
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            prompts = []
            for premise, question in batch:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Evidence: {premise}\nQuestion: {question}"},
                ]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            encoded = tokenizer(prompts, padding=True, truncation=True, max_length=768, return_tensors="pt")
            encoded = {key: value.cuda() for key, value in encoded.items()}
            with torch.no_grad():
                logp = torch.log_softmax(model(**encoded).logits[:, -1, :].float(), dim=-1)
                chosen = torch.stack(
                    [torch.logsumexp(logp[:, ids], dim=-1) for ids in label_ids],
                    dim=-1,
                )
                probabilities = torch.softmax(chosen, dim=-1).detach().cpu().numpy()
            for (premise, question), row in zip(batch, probabilities):
                cached[group_key(premise, question)] = row.astype(np.float32)
            done = start + len(batch)
            if start % (batch_size * 10) == 0:
                print(f"  qwen {done}/{len(missing)}", flush=True)
            if done % 400 == 0 or done == len(missing):
                keys = list(cached.keys())
                values = np.stack([cached[key] for key in keys], axis=0)
                np.savez(cache_path, keys=np.array(keys), values=values)
        print(f"qwen seconds={time.perf_counter() - started:.1f}", flush=True)
        keys = list(cached.keys())
        values = np.stack([cached[key] for key in keys], axis=0)
        np.savez(cache_path, keys=np.array(keys), values=values)
        del model
        torch.cuda.empty_cache()

    rows = [cached[group_key(premise, question)] for premise, question in pairs]
    return np.stack(rows, axis=0)


def collect_pairs(records: list[dict], bundles: dict[str, dict[str, list[str]]], probes: list[dict]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []

    def add(premise: str, question: str) -> None:
        for label in LABELS:
            pair = (premise, hypothesis(question, label))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    for record in records:
        for evidences in bundles[record["pmid"]].values():
            for evidence in evidences:
                add(evidence, record["question"])
    for probe in probes:
        for evidence in probe["evidences"]:
            add(evidence, probe["question"])
    return pairs


def rows_for(question: str, evidences: list[str], lookup: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    rows = []
    for evidence in evidences:
        per_label = [lookup[(evidence, hypothesis(question, label))] for label in LABELS]
        rows.append(np.stack(per_label, axis=0))
    return np.stack(rows, axis=0)


def prepare_split(
    records: list[dict],
    indices: list[int],
    bundles: dict[str, dict[str, list[str]]],
    lookup: dict[tuple[str, str], np.ndarray],
) -> dict[str, list[dict]]:
    prepared: dict[str, list[dict]] = {condition: [] for condition in CONDITIONS}
    for index in indices:
        record = records[index]
        for condition in CONDITIONS:
            evidences = bundles[record["pmid"]][condition]
            item = PreparedItem.build(record["question"], evidences, rows_for(record["question"], evidences, lookup))
            prepared[condition].append(
                {
                    "pmid": record["pmid"],
                    "gold": record["gold"],
                    "mental_health": record["mental_health"],
                    "evidences": evidences,
                    "item": item,
                    "outputs": method_outputs(item, LABELS),
                }
            )
    return prepared


def decision_confidence(output: dict) -> float:
    if output["gate"] in {"cdg", "pool"}:
        state = output["state"]
        leading = float(state.singleton.max())
        return leading / (leading + float(state.conflict) + 1e-8)
    return float(output["confidence"])


def apply_gate(rows: list[dict], method: str, threshold: float, slack: float, margin: float) -> list[str | None]:
    predictions = []
    for row in rows:
        label, _ = selective_decision(row["outputs"][method], LABELS, threshold, slack, margin)
        predictions.append(label)
    return predictions


def utility(clean_rows: list[dict], flip_rows: list[dict], method: str, threshold: float, slack: float, margin: float) -> float:
    clean_pred = apply_gate(clean_rows, method, threshold, slack, margin)
    flip_pred = apply_gate(flip_rows, method, threshold, slack, margin)
    clean = selective_stats([row["gold"] for row in clean_rows], clean_pred)
    flip = selective_stats([row["gold"] for row in flip_rows], flip_pred)
    return clean["success"] - flip["harmful_error"]


def apply_caer_decisive(prepared: dict[str, list[dict]], decisive: float) -> None:
    for rows in prepared.values():
        for row in rows:
            row["outputs"]["caer"] = caer_output(row["item"], decisive)


def tune_caer(prepared: dict[str, list[dict]]) -> dict:
    best = None
    for decisive in DECISIVES:
        apply_caer_decisive(prepared, decisive)
        for threshold in THRESHOLDS:
            for max_conflict in MAX_CONFLICTS:
                for margin in MARGINS:
                    score = utility(
                        prepared["clean"],
                        prepared["flip"],
                        "caer",
                        threshold,
                        max_conflict,
                        margin,
                    )
                    candidate = (score, decisive, threshold, max_conflict, margin)
                    if best is None or candidate[0] > best[0] + 1e-12:
                        best = candidate
    assert best is not None
    apply_caer_decisive(prepared, best[1])
    chosen = {
        "utility": best[0],
        "threshold": best[2],
        "conflict_slack": best[3],
        "margin": best[4],
        "decisive": best[1],
        "gate": "pool",
        "max_conflict": best[3],
    }
    print(f"tune caer: {chosen}", flush=True)
    return chosen


def tune(prepared: dict[str, list[dict]]) -> dict[str, dict]:
    chosen = {}
    for method in METHODS:
        if method == "caer":
            continue
        best = None
        gate = prepared["clean"][0]["outputs"][method]["gate"]
        if gate == "cdg":
            grid = [(0.0, slack, margin) for slack in SLACKS for margin in MARGINS]
        else:
            grid = [(threshold, 0.0, 0.0) for threshold in THRESHOLDS]
        for threshold, slack, margin in grid:
            score = utility(prepared["clean"], prepared["flip"], method, threshold, slack, margin)
            candidate = (score, threshold, slack, margin)
            if best is None or candidate[0] > best[0] + 1e-12:
                best = candidate
        assert best is not None
        chosen[method] = {
            "utility": best[0],
            "threshold": best[1],
            "conflict_slack": best[2],
            "margin": best[3],
            "gate": gate,
        }
        print(f"tune {method}: {chosen[method]}", flush=True)
    chosen["caer"] = tune_caer(prepared)
    return chosen


def forced_summary(rows: list[dict], method: str) -> dict[str, float]:
    golds = [row["gold"] for row in rows]
    preds = [LABELS[int(np.argmax(row["outputs"][method]["probs"]))] for row in rows]
    probs = np.stack([row["outputs"][method]["probs"] for row in rows], axis=0)
    gold_index = np.asarray([LABELS.index(gold) for gold in golds])
    confidences = np.asarray([decision_confidence(row["outputs"][method]) for row in rows])
    correct = np.asarray([pred == gold for pred, gold in zip(preds, golds)], dtype=np.float64)
    conflicts = []
    for row in rows:
        state = row["outputs"][method].get("state")
        if state is not None:
            conflicts.append(float(state.conflict))
    return {
        "accuracy": accuracy(golds, preds),
        "macro_f1": macro_f1(golds, preds, LABELS),
        "ece": expected_calibration_error(confidences, correct),
        "brier": brier_score(probs, gold_index),
        "nll": negative_log_likelihood(probs, gold_index),
        "aurc": aurc(confidences, correct),
        "mean_conflict": float(np.mean(conflicts)) if conflicts else None,
    }


def selective_summary(rows: list[dict], method: str, params: dict) -> dict[str, float]:
    predictions = apply_gate(rows, method, params["threshold"], params["conflict_slack"], params["margin"])
    return selective_stats([row["gold"] for row in rows], predictions)


def citation_summary(rows: list[dict], method: str) -> dict[str, float]:
    cited_all = []
    relevant_pred = []
    relevant_gold = []
    for row in rows:
        output = row["outputs"][method]
        predicted = int(np.argmax(output["probs"]))
        gold_index = LABELS.index(row["gold"])
        if method == "caer":
            cited = leave_one_out_citation(row["item"], predicted, typed=True)
        elif method == "top1":
            strength = np.asarray([1.0 - ignorance for _, ignorance in row["item"].masses])
            cited = [int(np.argmax(strength))]
        else:
            cited = [index for index, score in enumerate(row["item"].support[:, predicted]) if score > 0.5]
        cited_all.append(cited)
        relevant_pred.append(supporting_indices(row["item"].support, predicted))
        relevant_gold.append(supporting_indices(row["item"].support, gold_index))
    precision = citation_scores(cited_all, relevant_pred)
    attribution = citation_scores(cited_all, relevant_gold)
    return {
        "citation_precision": precision["citation_precision"],
        "citation_recall": citation_scores(cited_all, relevant_gold)["citation_recall"],
        "attribution_correctness": attribution["citation_precision"],
    }


def supporting_indices(support: np.ndarray, label_index: int) -> list[int]:
    chosen = []
    for index in range(support.shape[0]):
        if support[index, label_index] > 0.5 and support[index, label_index] >= np.max(support[index]) - 1e-8:
            chosen.append(index)
    return chosen


def subset(rows: list[dict], mental: bool) -> list[dict]:
    return [row for row in rows if row["mental_health"] is mental]


def harmful_flags(rows: list[dict], method: str, params: dict) -> np.ndarray:
    predictions = apply_gate(rows, method, params["threshold"], params["conflict_slack"], params["margin"])
    return np.asarray([(pred is not None and pred != row["gold"]) for pred, row in zip(predictions, rows)], dtype=np.float64)


def render_table(title: str, rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(["method", *columns]) + " |"
    rule = "|" + "|".join(["---"] * (len(columns) + 1)) + "|"
    body = []
    for row in rows:
        cells = [row["method"]]
        for column in columns:
            value = row[column]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([f"### {title}", header, rule, *body])


def evaluate_relation_probe(lookup: dict[tuple[str, str], np.ndarray], params: dict[str, dict]) -> dict:
    rows = []
    for probe in RELATION_PROBES:
        item = PreparedItem.build(probe["question"], probe["evidences"], rows_for(probe["question"], probe["evidences"], lookup))
        rows.append(
            {
                "pmid": probe["question"],
                "gold": probe["gold"],
                "mental_health": True,
                "evidences": probe["evidences"],
                "item": item,
                "outputs": method_outputs(item, LABELS),
            }
        )
    for row in rows:
        row["outputs"]["caer"] = caer_output(row["item"], params["caer"]["decisive"])
    report = {}
    for method in METHODS:
        report[method] = {
            "forced": forced_summary(rows, method),
            "selective": selective_summary(rows, method, params[method]),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = load_records()
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_PATH.write_text(FALLBACK_DATA.read_text(encoding="utf-8"), encoding="utf-8")
    splits = split_records(records, args.seed)
    if args.limit:
        for key in splits:
            splits[key] = splits[key][: args.limit]
    print(
        {key: len(value) for key, value in splits.items()},
        "mental_health",
        sum(row["mental_health"] for row in records),
        flush=True,
    )
    question_emb, abstract_emb = embed_corpus(records)
    bundles = build_bundles(records, question_emb, abstract_emb)
    used = splits["val"] + splits["test"]
    used_records = [records[index] for index in used]
    pairs = collect_pairs(used_records, bundles, RELATION_PROBES)
    scored = score_pairs(pairs)
    lookup = {pair: scored[index] for index, pair in enumerate(pairs)}

    prepared = {
        "val": prepare_split(records, splits["val"], bundles, lookup),
        "test": prepare_split(records, splits["test"], bundles, lookup),
    }
    params = tune(prepared["val"])
    apply_caer_decisive(prepared["test"], params["caer"]["decisive"])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def pack(rows_by_condition: dict[str, list[dict]]) -> dict:
        packed = {}
        for method in METHODS:
            packed[method] = {
                "params": params[method],
                "conditions": {},
            }
            for condition, rows in rows_by_condition.items():
                packed[method]["conditions"][condition] = {
                    "n": len(rows),
                    "forced": forced_summary(rows, method),
                    "selective": selective_summary(rows, method, params[method]),
                }
            packed[method]["citation_clean"] = citation_summary(rows_by_condition["clean"], method)
        return packed

    test_report = pack(prepared["test"])
    mental_rows = {condition: subset(rows, True) for condition, rows in prepared["test"].items()}
    mental_report = pack(mental_rows) if mental_rows["clean"] else {}
    relation_report = evaluate_relation_probe(lookup, params)
    safety_report = evaluate_crisis_gate()

    comparisons = {}
    for condition in ("flip", "real_conflict", "delete"):
        caer_harm = harmful_flags(prepared["test"][condition], "caer", params["caer"])
        comparisons[condition] = {}
        for method in METHODS:
            if method == "caer":
                continue
            other = harmful_flags(prepared["test"][condition], method, params[method])
            comparisons[condition][method] = bootstrap_delta(other - caer_harm)

    payload = {
        "seed": args.seed,
        "n": {key: len(value) for key, value in splits.items()},
        "n_mental_health_all": sum(row["mental_health"] for row in records),
        "n_mental_health_test": len(mental_rows["clean"]),
        "labels": list(LABELS),
        "params": params,
        "test": test_report,
        "mental_health_test": mental_report,
        "relation_probe": relation_report,
        "safety": safety_report,
        "harm_reduction_bootstrap": comparisons,
        "notes": {
            "encoder": "cross-encoder/nli-deberta-v3-base on full results sections",
            "retriever": "sentence-transformers/all-MiniLM-L6-v2",
            "caer": "agreement pool with validation-chosen decisive, confidence floor, and max opposing-vote conflict",
            "tuning": "validation utility = clean success - flip harmful error",
            "flip": "counterfactual opposite conclusion inserted beside the source abstract",
            "real_conflict": "two nearest abstracts with a different PubMedQA label",
        },
    }
    out = RESULT_DIR / ("metrics_smoke.json" if args.limit else "metrics.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    for method in METHODS:
        for condition in CONDITIONS:
            block = test_report[method]["conditions"][condition]
            forced = block["forced"]
            selective = block["selective"]
            print(
                f"{method:14} {condition:14} acc={forced['accuracy']:.3f} f1={forced['macro_f1']:.3f} "
                f"harm={selective['harmful_error']:.3f} cov={selective['coverage']:.3f} "
                f"sel={selective['selective_accuracy']:.3f} aurc={forced['aurc']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
