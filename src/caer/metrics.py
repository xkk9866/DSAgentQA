"""Selective-prediction and attribution metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy(golds: list[str], preds: list[str]) -> float:
    return float(accuracy_score(golds, preds))


def macro_f1(golds: list[str], preds: list[str], labels: tuple[str, ...]) -> float:
    if not golds:
        return float("nan")
    return float(f1_score(golds, preds, labels=list(labels), average="macro", zero_division=0))


def selective_stats(golds: list[str], preds: list[str | None]) -> dict[str, float]:
    if not golds:
        return {
            "coverage": float("nan"),
            "selective_accuracy": float("nan"),
            "success": float("nan"),
            "harmful_error": float("nan"),
            "abstain_rate": float("nan"),
        }
    answered = [pred is not None for pred in preds]
    coverage = float(np.mean(answered))
    correct_flags = [(pred == gold) if pred is not None else False for pred, gold in zip(preds, golds)]
    n_answered = int(sum(answered))
    return {
        "coverage": coverage,
        "selective_accuracy": float(sum(correct_flags) / n_answered) if n_answered else float("nan"),
        "success": float(np.mean(correct_flags)),
        "harmful_error": float(np.mean([(pred is not None and pred != gold) for pred, gold in zip(preds, golds)])),
        "abstain_rate": 1.0 - coverage,
    }


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(confidences) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return float(error)


def brier_score(probs: np.ndarray, gold_index: np.ndarray) -> float:
    if len(probs) == 0:
        return float("nan")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(gold_index)), gold_index] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, gold_index: np.ndarray) -> float:
    if len(probs) == 0:
        return float("nan")
    picked = np.clip(probs[np.arange(len(gold_index)), gold_index], 1e-8, 1.0)
    return float(-np.mean(np.log(picked)))


def aurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(confidences) == 0:
        return float("nan")
    order = np.argsort(-confidences)
    ranked = correct[order]
    steps = np.arange(1, len(ranked) + 1)
    risk = np.cumsum(1.0 - ranked) / steps
    coverage = steps / len(ranked)
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(risk, coverage))


def citation_scores(cited: list[list[int]], relevant: list[list[int]]) -> dict[str, float]:
    precisions = []
    recalls = []
    for cited_ids, relevant_ids in zip(cited, relevant):
        cited_set = set(cited_ids)
        relevant_set = set(relevant_ids)
        if cited_set:
            precisions.append(len(cited_set & relevant_set) / len(cited_set))
        if relevant_set:
            recalls.append(len(cited_set & relevant_set) / len(relevant_set))
    return {
        "citation_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "citation_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "n_precision": len(precisions),
        "n_recall": len(recalls),
    }


def bootstrap_delta(differences: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(differences) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    means = [float(rng.choice(differences, size=len(differences), replace=True).mean()) for _ in range(n_boot)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(np.mean(differences)), "low": float(low), "high": float(high)}
