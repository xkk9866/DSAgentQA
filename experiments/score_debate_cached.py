"""Test numbers for the cached multi-agent revision baseline.

Replays run_llm_baselines.madam_debate from the log-probability cache.
The threshold 0.80 is the one chosen on validation in that run.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prop", ROOT / "experiments" / "run_proposition_experiment.py")
prop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prop)
spec_b = importlib.util.spec_from_file_location("baselines", ROOT / "experiments" / "run_llm_baselines.py")
baselines = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(baselines)

LABELS = ("yes", "no", "maybe")
BIAS = np.array([1.5, 1.0, -1.5, 0.0])
THRESHOLD = 0.80


def main() -> None:
    records = prop.mod.load_records()
    by_pmid = {row["pmid"]: index for index, row in enumerate(records)}
    official = json.loads((ROOT / "data" / "test_ground_truth.json").read_text(encoding="utf-8"))
    test_index = [by_pmid[pmid] for pmid in official if pmid in by_pmid]
    question_emb, abstract_emb = prop.mod.embed_corpus(records)
    bundles = prop.bundles_for(records, question_emb, abstract_emb)
    doc_cache = prop.load_cache()
    stored = np.load(baselines.LOG_CACHE, allow_pickle=False)
    log_cache = dict(zip(stored["keys"].tolist(), stored["values"]))

    def first_round(evidence: str, question: str) -> tuple[str, np.ndarray]:
        probabilities = prop.softmax((doc_cache[prop.key_of(evidence, question)] + BIAS)[None, :])[0]
        return prop.WORDS[int(np.argmax(probabilities))], probabilities

    report = {}
    for condition in prop.mod.CONDITIONS:
        golds, preds, confs = [], [], []
        missing = 0
        for index in test_index:
            record = records[index]
            question = record["question"]
            evidences = bundles[record["pmid"]][condition]
            answers = [first_round(evidence, question)[0] for evidence in evidences]
            total = np.zeros(3)
            for k, evidence in enumerate(evidences):
                if condition in baselines.DEBATE_CONDITIONS and len(evidences) > 1:
                    messages = baselines.debate_prompt(evidence, answers[:k] + answers[k + 1 :], question)
                    scores = log_cache.get(baselines.key_of(messages))
                    if scores is None:
                        missing += 1
                        break
                    probabilities = prop.softmax((scores + BIAS)[None, :])[0]
                else:
                    probabilities = first_round(evidence, question)[1]
                total += probabilities[:3]
            else:
                total = total / total.sum()
                golds.append(record["gold"])
                preds.append(LABELS[int(np.argmax(total))])
                confs.append(float(total.max()))
        answered = [pred if conf >= THRESHOLD else None for pred, conf in zip(preds, confs)]
        selective = prop.mod.selective_stats(golds, answered)
        print(
            f"debate {condition:14} n={len(golds)} missing={missing} "
            f"acc={prop.mod.accuracy(golds, preds):.3f} f1={prop.mod.macro_f1(golds, preds, LABELS):.3f} "
            f"harm={selective['harmful_error']:.3f} cov={selective['coverage']:.3f}",
            flush=True,
        )
        report[condition] = {
            "n": len(golds),
            "accuracy": prop.mod.accuracy(golds, preds),
            "macro_f1": prop.mod.macro_f1(golds, preds, LABELS),
            "selective": selective,
        }
    out = ROOT / "results" / "metrics_debate.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
