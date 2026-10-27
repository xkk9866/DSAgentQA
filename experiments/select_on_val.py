"""Choose the agreement-pooling rule on the validation split only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_experiment import (  # noqa: E402
    LABELS,
    build_bundles,
    collect_pairs,
    embed_corpus,
    load_records,
    prepare_split,
    score_pairs,
    split_records,
)
from caer.metrics import accuracy, macro_f1, selective_stats  # noqa: E402


def cluster_weights(support: np.ndarray, reliability: np.ndarray, decisive: float, mode: str) -> tuple[np.ndarray, float]:
    weights = np.zeros(support.shape[1], dtype=np.float64)
    vacuous = 0.0
    for index in range(len(support)):
        strength = float(support[index].max())
        reliability_i = float(reliability[index])
        if mode == "count":
            vote_weight = 1.0
        elif mode == "reliability":
            vote_weight = reliability_i
        else:
            margin = float(support[index].max() - np.partition(support[index], -2)[-2])
            vote_weight = reliability_i * margin
        if strength < decisive or vote_weight <= 0:
            vacuous += max(reliability_i, 1e-6)
            continue
        weights[int(np.argmax(support[index]))] += vote_weight
    committed = float(weights.sum())
    if committed <= 0:
        return np.ones(support.shape[1]) / support.shape[1], 1.0
    conflict = float((committed - weights.max()) / committed)
    return weights / committed, conflict


def evaluate(rows_clean, rows_flip, decisive: float, mode: str, max_conflict: float) -> dict[str, float]:
    forced = []
    selective_clean = []
    selective_flip = []
    for row in rows_clean:
        weights, conflict = cluster_weights(row["item"].support, row["item"].reliabilities, decisive, mode)
        label = LABELS[int(np.argmax(weights))]
        forced.append(label)
        selective_clean.append(label if conflict <= max_conflict and float(weights.max()) > 0 else None)
    for row in rows_flip:
        weights, conflict = cluster_weights(row["item"].support, row["item"].reliabilities, decisive, mode)
        label = LABELS[int(np.argmax(weights))]
        selective_flip.append(label if conflict <= max_conflict and float(weights.max()) > 0 else None)
    golds = [row["gold"] for row in rows_clean]
    clean = selective_stats(golds, selective_clean)
    flip = selective_stats([row["gold"] for row in rows_flip], selective_flip)
    return {
        "acc": accuracy(golds, forced),
        "f1": macro_f1(golds, forced, LABELS),
        "utility": clean["success"] - flip["harmful_error"],
        "clean_harm": clean["harmful_error"],
        "clean_cov": clean["coverage"],
        "flip_harm": flip["harmful_error"],
        "flip_cov": flip["coverage"],
        "flip_sel": flip["selective_accuracy"],
    }


def main() -> None:
    records = load_records()
    splits = split_records(records, 42)
    question_emb, abstract_emb = embed_corpus(records)
    bundles = build_bundles(records, question_emb, abstract_emb)
    used = [records[index] for index in splits["val"]]
    pairs = collect_pairs(used, bundles, [])
    scored = score_pairs(pairs)
    lookup = {pair: scored[index] for index, pair in enumerate(pairs)}
    prepared = prepare_split(records, splits["val"], bundles, lookup)
    best = None
    print("val search", flush=True)
    for mode in ("count", "reliability", "margin"):
        for decisive in (0.2, 0.35, 0.45, 0.6, 0.75):
            for max_conflict in (0.0, 0.15, 0.25, 0.35, 0.5, 1.0):
                score = evaluate(prepared["clean"], prepared["flip"], decisive, mode, max_conflict)
                row = (score["utility"], score["acc"], mode, decisive, max_conflict, score)
                if best is None or row[0] > best[0] + 1e-12 or (abs(row[0] - best[0]) < 1e-12 and row[1] > best[1]):
                    best = row
                if max_conflict == 1.0:
                    print(
                        f"{mode:12} d={decisive:.2f} forced_acc={score['acc']:.3f} f1={score['f1']:.3f}",
                        flush=True,
                    )
    utility, acc, mode, decisive, max_conflict, score = best
    print("BEST", mode, decisive, max_conflict, score, flush=True)


if __name__ == "__main__":
    main()
