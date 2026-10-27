"""Dependent-evidence fusion, conformal risk control, and cached RAG baselines.

Same official split and the same frozen four-way reader as
run_proposition_experiment.py. Nothing here is chosen on the test split.

  dependent     Jøsang cumulative/averaging blend. Dependence is the maximum
                similarity to an already counted paper with the same label.
                The decisive cutoff is chosen on val by coverage under a
                conformal threshold fit on calib at harm budget 0.10.
  replug        per-document probabilities weighted by retrieval similarity
                (Shi et al., 2023). Temperature chosen on val accuracy.
  chain_of_note documents whose leading class is "other" are dropped
                (Yu et al., 2023), then averaged.
  single_passage the single bundle document closest to the question.
  astute        closed-book vs whole-bundle RAG; disagreement lowers confidence
                (Wang et al., 2024).
  adaptive      closed-book when its confidence clears a val-chosen cutoff,
                otherwise whole-bundle RAG (Jeong et al., 2024).

Selective numbers use conformal risk control (Angelopoulos et al.): the most
permissive threshold whose finite-sample harmful-answer rate on calib is at
most alpha. Alpha is fixed at 0.10 and 0.05 before looking at test.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prop", ROOT / "experiments" / "run_proposition_experiment.py")
prop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prop)
mod = prop.mod

spec_b = importlib.util.spec_from_file_location("baselines", ROOT / "experiments" / "run_llm_baselines.py")
baselines = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(baselines)

from caer.fusion import dependent_consensus  # noqa: E402

LABELS = prop.LABELS
ALPHAS = (0.10, 0.05)
FROZEN_DECISIVE = 0.55
CONDITIONS = mod.CONDITIONS
REPORT_CONDITIONS = ("source", "clean", "flip", "real_conflict", "delete", "irrelevant")


def crc_threshold(confidences: np.ndarray, golds: list[str], preds: list[str], alpha: float) -> float:
    """Smallest threshold whose corrected harm rate is at most alpha."""
    n = len(golds)
    confidences = np.asarray(confidences, dtype=np.float64)
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), np.round(confidences, 5)]))
    for threshold in grid:
        harm = np.mean([(float(conf) >= threshold and pred != gold) for conf, pred, gold in zip(confidences, preds, golds)])
        if (harm * n + 1.0) / (n + 1.0) <= alpha:
            return float(threshold)
    return 1.01


def selective_at(confidences, golds, preds, threshold: float) -> dict:
    answered = [
        pred if float(conf) >= threshold else None
        for conf, pred in zip(confidences, preds)
    ]
    return mod.selective_stats(golds, answered)


def forced_of(golds, preds) -> dict:
    return {"accuracy": mod.accuracy(golds, preds), "macro_f1": mod.macro_f1(golds, preds, LABELS)}


def pack_method(rows: list[dict]) -> dict:
    return {
        "golds": [row["gold"] for row in rows],
        "preds": [row["pred"] for row in rows],
        "conf": np.asarray([row["conf"] for row in rows], dtype=np.float64),
    }


def main() -> None:
    records = mod.load_records()
    by_pmid = {row["pmid"]: index for index, row in enumerate(records)}
    official = json.loads((ROOT / "data" / "test_ground_truth.json").read_text(encoding="utf-8"))
    test_index = [by_pmid[pmid] for pmid in official if pmid in by_pmid]
    rest = [index for index in range(len(records)) if index not in set(test_index)]
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
    cached = prop.load_cache()
    bias = np.array([1.5, 1.0, -1.5, 0.0])
    prepared = {
        name: prop.prepare(records, indices, bundles, cached, bias) for name, indices in split.items()
    }
    mod.apply_caer_decisive(prepared["calib"], FROZEN_DECISIVE)
    mod.apply_caer_decisive(prepared["val"], FROZEN_DECISIVE)
    mod.apply_caer_decisive(prepared["test"], FROZEN_DECISIVE)

    known = {records[i]["abstract"]: abstract_emb[i] for i in range(len(records))}
    missing = []
    for indices in split.values():
        for index in indices:
            for evidences in bundles[records[index]["pmid"]].values():
                for text in evidences:
                    if text not in known:
                        missing.append(text)
    missing = list(dict.fromkeys(missing))
    print(f"encoding {len(missing)} extra evidence texts", flush=True)
    if missing:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True
        )
        vectors = encoder.encode(missing, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        for text, vector in zip(missing, vectors):
            known[text] = np.asarray(vector, dtype=np.float32)
        del encoder

    def vectors_for(texts: list[str]) -> np.ndarray:
        return np.stack([known[text] for text in texts], axis=0)

    def similarity_of(texts: list[str]) -> np.ndarray:
        matrix = vectors_for(texts)
        return matrix @ matrix.T

    log_cache = {}
    if baselines.LOG_CACHE.exists():
        stored = np.load(baselines.LOG_CACHE, allow_pickle=False)
        log_cache = {key: value for key, value in zip(stored["keys"].tolist(), stored["values"])}

    def cached_log(messages) -> np.ndarray | None:
        key = baselines.key_of(messages)
        return log_cache.get(key)

    # Per-split, per-condition, per-method rows of {gold, pred, conf}.
    tables: dict[str, dict[str, dict[str, list]]] = {
        name: {condition: {} for condition in CONDITIONS} for name in split
    }

    def add(split_name: str, condition: str, method: str, gold: str, probs: np.ndarray, conf: float) -> None:
        tables[split_name][condition].setdefault(method, []).append(
            {"gold": gold, "pred": LABELS[int(np.argmax(probs))], "conf": float(conf)}
        )

    frozen_methods = ("mean", "top1", "majority", "dempster", "yager", "selfrag_gate", "typed_yager", "caer")
    for name, by_condition in prepared.items():
        for condition, rows in by_condition.items():
            indices = split[name]
            for row_index, row in enumerate(rows):
                record = records[indices[row_index]]
                evidences = bundles[record["pmid"]][condition]
                question = record["question"]
                qvec = question_emb[indices[row_index]]
                doc_vec = vectors_for(evidences)
                doc_sim = doc_vec @ qvec
                for method in frozen_methods:
                    output = row["outputs"][method]
                    add(name, condition, method, row["gold"], output["probs"], mod.decision_confidence(output))
                state_rows = []
                four = []
                for evidence in evidences:
                    probabilities = prop.softmax((cached[prop.key_of(evidence, question)] + bias)[None, :])[0]
                    four.append(probabilities)
                    state_rows.append(probabilities[:3] / max(float(probabilities[:3].sum()), 1e-8))
                four = np.stack(four, axis=0)
                # REPLUG temperatures filled later; store the doc distributions via conf placeholder.
                sims = similarity_of(evidences)
                # single passage
                best = int(np.argmax(doc_sim))
                add(name, condition, "single_passage", row["gold"], state_rows[best], float(four[best, :3].max()))
                # chain of note: drop documents whose four-way label is other
                kept = [i for i in range(len(evidences)) if int(np.argmax(four[i])) != 3]
                if kept:
                    pooled = np.mean(np.stack([state_rows[i] for i in kept], axis=0), axis=0)
                    conf = float(pooled.max())
                else:
                    pooled = np.ones(3) / 3.0
                    conf = 0.0
                add(name, condition, "chain_of_note", row["gold"], pooled, conf)
                # dependent consensus is filled after decisive is chosen; store pieces
                add(name, condition, "_docsim", row["gold"], state_rows[0], float(doc_sim.max()))
                row["_sims"] = sims
                row["_support_ok"] = True

    # Dependent consensus: choose decisive on val at alpha 0.10.
    def dependent_rows(split_name: str, decisive: float) -> dict[str, dict]:
        out = {}
        for condition, rows in prepared[split_name].items():
            built = []
            for row in rows:
                state = dependent_consensus(row["item"].support, row["item"].reliabilities, row["_sims"], decisive)
                pred = LABELS[int(np.argmax(state.singleton))]
                conf = float(state.singleton.max() * (1.0 - state.ignorance))
                built.append({"gold": row["gold"], "pred": pred, "conf": conf})
            out[condition] = pack_method(built)
        return out

    best_decisive = None
    best_utility = -1e9
    dependent_by_decisive = {}
    for decisive in mod.DECISIVES:
        by_split = {name: dependent_rows(name, decisive) for name in split}
        threshold = crc_threshold(
            by_split["calib"]["clean"]["conf"],
            by_split["calib"]["clean"]["golds"],
            by_split["calib"]["clean"]["preds"],
            0.10,
        )
        clean = selective_at(by_split["val"]["clean"]["conf"], by_split["val"]["clean"]["golds"], by_split["val"]["clean"]["preds"], threshold)
        flip = selective_at(by_split["val"]["flip"]["conf"], by_split["val"]["flip"]["golds"], by_split["val"]["flip"]["preds"], threshold)
        utility = clean["success"] - flip["harmful_error"]
        print(f"dependent decisive={decisive:.2f} t={threshold:.3f} val_utility={utility:.3f}", flush=True)
        dependent_by_decisive[decisive] = (by_split, threshold, utility)
        if utility > best_utility + 1e-12:
            best_utility = utility
            best_decisive = decisive
    print(f"chosen decisive={best_decisive}", flush=True)

    # REPLUG: temperature on validation clean accuracy.
    def replug_rows(split_name: str, tau: float) -> dict[str, dict]:
        out = {}
        for condition, rows in prepared[split_name].items():
            indices = split[split_name]
            built = []
            for row_index, row in enumerate(rows):
                record = records[indices[row_index]]
                evidences = bundles[record["pmid"]][condition]
                qvec = question_emb[indices[row_index]]
                dists = []
                sims = []
                for evidence in evidences:
                    probabilities = prop.softmax((cached[prop.key_of(evidence, record["question"])] + bias)[None, :])[0]
                    three = probabilities[:3]
                    three = three / max(float(three.sum()), 1e-8)
                    dists.append(three)
                    sims.append(float(known[evidence] @ qvec))
                weights = np.exp(np.asarray(sims) / tau)
                weights = weights / weights.sum()
                pooled = np.average(np.stack(dists, axis=0), axis=0, weights=weights)
                built.append({"gold": row["gold"], "pred": LABELS[int(np.argmax(pooled))], "conf": float(pooled.max())})
            out[condition] = pack_method(built)
        return out

    best_tau = None
    best_acc = -1.0
    replug_by_tau = {}
    for tau in (0.05, 0.1, 0.2, 0.5, 1.0):
        by_split = {name: replug_rows(name, tau) for name in ("val", "calib", "test")}
        accuracy = forced_of(by_split["val"]["clean"]["golds"], by_split["val"]["clean"]["preds"])["accuracy"]
        print(f"replug tau={tau} val_clean_acc={accuracy:.3f}", flush=True)
        replug_by_tau[tau] = by_split
        if accuracy > best_acc + 1e-12:
            best_acc = accuracy
            best_tau = tau
    print(f"chosen tau={best_tau}", flush=True)

    chosen = {
        "dependent": dependent_by_decisive[best_decisive][0],
        "replug": replug_by_tau[best_tau],
        "chain_of_note": {},
        "single_passage": {},
    }
    for method in ("chain_of_note", "single_passage", *frozen_methods):
        chosen[method] = {}
        for name in split:
            chosen[method][name] = {condition: pack_method(tables[name][condition][method]) for condition in CONDITIONS}

    # Prompt baselines from the log-probability cache, when every prompt is present.
    def try_prompt_method(method: str, builder, needed: dict[str, tuple[str, ...]]) -> None:
        built = {name: {condition: [] for condition in CONDITIONS} for name in split}
        for name, conditions in needed.items():
            for condition in conditions:
                for index in split[name]:
                    record = records[index]
                    scores = cached_log(builder(bundles[record["pmid"]][condition], record["question"]))
                    if scores is None:
                        print(f"skip {method}: missing {name}/{condition}", flush=True)
                        return
                    probabilities = baselines.three_way(scores[None, :], bias)[0]
                    built[name][condition].append(
                        {
                            "gold": record["gold"],
                            "pred": LABELS[int(np.argmax(probabilities))],
                            "conf": float(probabilities.max()),
                        }
                    )
                if name == "test" and condition == "clean":
                    counts = {label: 0 for label in LABELS}
                    for row in built[name][condition]:
                        counts[row["pred"]] += 1
                    print(f"{method} test clean pred counts {counts}", flush=True)
        chosen[method] = {}
        for name, by_c in built.items():
            chosen[method][name] = {c: pack_method(rows) for c, rows in by_c.items() if rows}
        print(f"loaded {method}", flush=True)

    def rag_builder(evidences, question):
        return baselines.rag_prompt(evidences, question)

    def closed_builder(evidences, question):
        return baselines.closed_prompt(question)

    needed = {"calib": ("clean",), "val": ("clean",), "test": CONDITIONS}
    try_prompt_method("vanilla_rag", rag_builder, needed)
    try_prompt_method("closed_book", closed_builder, needed)

    if "vanilla_rag" in chosen and "closed_book" in chosen:
        # Astute: agree -> average; disagree -> keep the sharper one and cut confidence.
        astute = {name: {} for name in split}
        for name in split:
            for condition in CONDITIONS:
                rows = []
                for closed, retrieved, gold in zip(
                    chosen["closed_book"][name][condition]["preds"],
                    chosen["vanilla_rag"][name][condition]["preds"],
                    chosen["vanilla_rag"][name][condition]["golds"],
                ):
                    pass
                closed_pack = chosen["closed_book"][name][condition]
                rag_pack = chosen["vanilla_rag"][name][condition]
                # Rebuild probabilities are not stored. Confidence-only blend:
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
                astute[name][condition] = pack_method(rows)
        chosen["astute"] = astute
        # Adaptive route cutoff on validation clean accuracy.
        best_route, best_route_acc = 0.0, -1.0
        for route in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            preds = []
            golds = chosen["vanilla_rag"]["val"]["clean"]["golds"]
            for pred_c, conf_c, pred_r in zip(
                chosen["closed_book"]["val"]["clean"]["preds"],
                chosen["closed_book"]["val"]["clean"]["conf"],
                chosen["vanilla_rag"]["val"]["clean"]["preds"],
            ):
                preds.append(pred_c if conf_c >= route else pred_r)
            accuracy = mod.accuracy(golds, preds)
            if accuracy > best_route_acc + 1e-12:
                best_route_acc = accuracy
                best_route = route
        print(f"adaptive route={best_route} val_clean_acc={best_route_acc:.3f}", flush=True)
        adaptive = {name: {} for name in split}
        for name in split:
            for condition in CONDITIONS:
                rows = []
                closed_pack = chosen["closed_book"][name][condition]
                rag_pack = chosen["vanilla_rag"][name][condition]
                for pred_c, conf_c, pred_r, conf_r, gold in zip(
                    closed_pack["preds"], closed_pack["conf"], rag_pack["preds"], rag_pack["conf"], rag_pack["golds"]
                ):
                    if conf_c >= best_route:
                        rows.append({"gold": gold, "pred": pred_c, "conf": float(conf_c)})
                    else:
                        rows.append({"gold": gold, "pred": pred_r, "conf": float(conf_r)})
                adaptive[name][condition] = pack_method(rows)
        chosen["adaptive"] = adaptive
        chosen["adaptive_route"] = best_route

    report = {"decisive": best_decisive, "replug_tau": best_tau, "alphas": list(ALPHAS), "test": {}}
    methods = [method for method in chosen if method not in {"adaptive_route"} and "test" in chosen[method]]
    for method in methods:
        report["test"][method] = {}
        for condition in REPORT_CONDITIONS:
            block = chosen[method]["test"][condition]
            entry = {"forced": forced_of(block["golds"], block["preds"]), "conformal": {}}
            for alpha in ALPHAS:
                # Thresholds come from ordinary (clean) calibration questions and
                # are not refit on the conflict condition.
                threshold = crc_threshold(
                    chosen[method]["calib"]["clean"]["conf"],
                    chosen[method]["calib"]["clean"]["golds"],
                    chosen[method]["calib"]["clean"]["preds"],
                    alpha,
                )
                entry["conformal"][str(alpha)] = {
                    "threshold": threshold,
                    **selective_at(block["conf"], block["golds"], block["preds"], threshold),
                }
            report["test"][method][condition] = entry
            forced = entry["forced"]
            selective = entry["conformal"]["0.1"]
            if condition in {"source", "clean", "flip"}:
                print(
                    f"{method:16} {condition:8} acc={forced['accuracy']:.3f} f1={forced['macro_f1']:.3f} "
                    f"a10_harm={selective['harmful_error']:.3f} a10_cov={selective['coverage']:.3f}",
                    flush=True,
                )

    out = ROOT / "results" / "metrics_enriched.json"
    # Drop numpy arrays before writing by rebuilding from printed scalars only.
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
