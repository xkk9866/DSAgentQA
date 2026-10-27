"""Score cached whole-prompt baselines: vanilla RAG, Astute, Adaptive-RAG.

Uses the log-probability cache written by run_llm_baselines.py. Thresholds for
the harm budget are fit on the clean calibration split only.
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
spec_e = importlib.util.spec_from_file_location("enriched", ROOT / "experiments" / "run_enriched_eval.py")
enriched = importlib.util.module_from_spec(spec_e)
spec_e.loader.exec_module(enriched)

LABELS = prop.LABELS
CONDITIONS = prop.mod.CONDITIONS
BIAS = np.array([1.5, 1.0, -1.5, 0.0])


def main() -> None:
    records = prop.mod.load_records()
    by_pmid = {row["pmid"]: i for i, row in enumerate(records)}
    official = json.loads((ROOT / "data" / "test_ground_truth.json").read_text(encoding="utf-8"))
    test_index = [by_pmid[pmid] for pmid in official if pmid in by_pmid]
    rest = [i for i in range(len(records)) if i not in set(test_index)]
    calib_pos, val_pos = train_test_split(
        np.arange(len(rest)), test_size=0.5, random_state=42, stratify=[records[i]["gold"] for i in rest]
    )
    split = {
        "calib": [rest[int(i)] for i in calib_pos],
        "val": [rest[int(i)] for i in val_pos],
        "test": test_index,
    }
    question_emb, abstract_emb = prop.mod.embed_corpus(records)
    bundles = prop.bundles_for(records, question_emb, abstract_emb)
    stored = np.load(baselines.LOG_CACHE, allow_pickle=False)
    log_cache = {key: value for key, value in zip(stored["keys"].tolist(), stored["values"])}
    print(f"log cache {len(log_cache)}", flush=True)

    def scores_for(messages):
        return log_cache.get(baselines.key_of(messages))

    def fill(builder, name, conditions):
        out = {condition: [] for condition in conditions}
        missing = 0
        for index in split[name]:
            record = records[index]
            for condition in conditions:
                scores = scores_for(builder(bundles[record["pmid"]][condition], record["question"]))
                if scores is None:
                    missing += 1
                    continue
                probabilities = baselines.three_way(scores[None, :], BIAS)[0]
                out[condition].append(
                    {
                        "gold": record["gold"],
                        "pred": LABELS[int(np.argmax(probabilities))],
                        "conf": float(probabilities.max()),
                    }
                )
        print(f"missing prompts {missing}", flush=True)
        return {c: enriched.pack_method(rows) for c, rows in out.items() if rows}

    rag = {name: fill(baselines.rag_prompt, name, ("clean",) if name != "test" else CONDITIONS) for name in split}
    closed = {
        name: fill(lambda evidences, question: baselines.closed_prompt(question), name, ("clean",) if name != "test" else ("clean",))
        for name in split
    }
    report = {}
    for method, packs in (("vanilla_rag", rag), ("closed_book", closed)):
        report[method] = {}
        for condition, block in packs["test"].items():
            entry = {"forced": enriched.forced_of(block["golds"], block["preds"])}
            threshold = enriched.crc_threshold(packs["calib"]["clean"]["conf"], packs["calib"]["clean"]["golds"], packs["calib"]["clean"]["preds"], 0.10)
            entry["conformal_0.10"] = {"threshold": threshold, **enriched.selective_at(block["conf"], block["golds"], block["preds"], threshold)}
            report[method][condition] = entry
            forced, selective = entry["forced"], entry["conformal_0.10"]
            print(
                f"{method:16} {condition:14} acc={forced['accuracy']:.3f} f1={forced['macro_f1']:.3f} "
                f"harm={selective['harmful_error']:.3f} cov={selective['coverage']:.3f}",
                flush=True,
            )

    best_route, best_acc = 0.5, -1.0
    for route in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        preds = [
            pred_c if conf_c >= route else pred_r
            for pred_c, conf_c, pred_r in zip(
                closed["val"]["clean"]["preds"], closed["val"]["clean"]["conf"], rag["val"]["clean"]["preds"]
            )
        ]
        accuracy = prop.mod.accuracy(rag["val"]["clean"]["golds"], preds)
        if accuracy > best_acc:
            best_acc = accuracy
            best_route = route
    print(f"adaptive route {best_route} val acc {best_acc:.3f}", flush=True)

    def blend(name: str, condition: str) -> dict:
        closed_pack = closed[name]["clean"]
        rag_pack = rag[name][condition]
        rows = []
        for pred_c, conf_c, pred_r, conf_r, gold in zip(
            closed_pack["preds"], closed_pack["conf"], rag_pack["preds"], rag_pack["conf"], rag_pack["golds"]
        ):
            if pred_c == pred_r:
                pred, conf = pred_r, float(0.5 * (conf_c + conf_r))
            elif conf_r >= conf_c:
                pred, conf = pred_r, float(conf_r - conf_c)
            else:
                pred, conf = pred_c, float(conf_c - conf_r)
            rows.append({"gold": gold, "pred": pred, "conf": max(conf, 0.0)})
        return enriched.pack_method(rows)

    astute = {name: {condition: blend(name, condition) for condition in rag[name]} for name in split}
    adaptive = {name: {} for name in split}
    for name in split:
        for condition, rag_pack in rag[name].items():
            rows = []
            for pred_c, conf_c, pred_r, conf_r, gold in zip(
                closed[name]["clean"]["preds"],
                closed[name]["clean"]["conf"],
                rag_pack["preds"],
                rag_pack["conf"],
                rag_pack["golds"],
            ):
                if conf_c >= best_route:
                    rows.append({"gold": gold, "pred": pred_c, "conf": float(conf_c)})
                else:
                    rows.append({"gold": gold, "pred": pred_r, "conf": float(conf_r)})
            adaptive[name][condition] = enriched.pack_method(rows)
    for method, packs in (("astute", astute), ("adaptive", adaptive)):
        report[method] = {}
        for condition, block in packs["test"].items():
            threshold = enriched.crc_threshold(
                packs["calib"]["clean"]["conf"], packs["calib"]["clean"]["golds"], packs["calib"]["clean"]["preds"], 0.10
            )
            selective = enriched.selective_at(block["conf"], block["golds"], block["preds"], threshold)
            forced = enriched.forced_of(block["golds"], block["preds"])
            report[method][condition] = {"forced": forced, "conformal_0.10": {"threshold": threshold, **selective}}
            print(
                f"{method:16} {condition:14} acc={forced['accuracy']:.3f} f1={forced['macro_f1']:.3f} "
                f"harm={selective['harmful_error']:.3f} cov={selective['coverage']:.3f}",
                flush=True,
            )

    out = ROOT / "results" / "metrics_prompt_baselines.json"
    out.write_text(json.dumps({"adaptive_route": best_route, "test": report}, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
