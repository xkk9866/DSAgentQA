"""Baselines and CAER decision rules on a shared evidence encoder."""

from __future__ import annotations

import numpy as np

from caer.fusion import (
    FusedState,
    agreement_pool,
    conflict_dominance_answer,
    fuse_sources,
    nli_to_mass,
    shafer_discount,
    smets_fuse,
    source_pignistic,
)
from caer.typing import relation_violation, source_reliability


def masses_from_nli(entail: np.ndarray, neutral: np.ndarray, contra: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """entail/neutral/contra: shape (n_sources, n_labels)."""
    return [nli_to_mass(entail[row], neutral[row], contra[row]) for row in range(entail.shape[0])]


def _mean_probs(masses: list[tuple[np.ndarray, float]]) -> np.ndarray:
    stacked = [source_pignistic(singleton, ignorance) for singleton, ignorance in masses]
    return np.mean(np.stack(stacked, axis=0), axis=0)


def _vote_probs(support: np.ndarray) -> np.ndarray:
    width = support.shape[1]
    votes = np.zeros(width, dtype=np.float64)
    for row in range(support.shape[0]):
        if float(support[row].max()) <= 0.45:
            continue
        votes[int(np.argmax(support[row]))] += 1.0
    if votes.sum() == 0:
        return np.ones(width) / width
    return votes / votes.sum()


class PreparedItem:
    def __init__(
        self,
        entail: np.ndarray,
        neutral: np.ndarray,
        contra: np.ndarray,
        reliabilities: np.ndarray,
        relation_flags: np.ndarray,
    ) -> None:
        self.entail = entail
        self.neutral = neutral
        self.contra = contra
        self.reliabilities = reliabilities
        self.relation_flags = relation_flags
        self.masses = masses_from_nli(entail, neutral, contra)
        self.support = np.stack(
            [
                entail[:, 0] + contra[:, 1],
                entail[:, 1] + contra[:, 0],
                entail[:, 2],
            ],
            axis=1,
        )
        ones = np.full(len(self.masses), 0.85, dtype=np.float64)
        self.plain = fuse_sources(self.masses, ones)
        self.typed = fuse_sources(self.masses, reliabilities)
        self.decisive = 0.35
        self.pooled = agreement_pool(self.support, reliabilities, decisive=self.decisive)

    def set_pooled(self, decisive: float) -> FusedState:
        self.decisive = float(decisive)
        self.pooled = agreement_pool(self.support, self.reliabilities, decisive=self.decisive)
        return self.pooled

    @classmethod
    def build(cls, question: str, evidences: list[str], nli_rows: np.ndarray) -> "PreparedItem":
        """nli_rows: (n_sources, n_labels, 3) ordered entail, neutral, contradiction.

        Reliability is the product of scope/relation factors and the source's own
        bidirectional support. A sentence that does not support or oppose a label
        is discounted toward ignorance, so it cannot create residual conflict.
        """
        reliabilities = []
        for index, evidence in enumerate(evidences):
            structural = source_reliability(question, evidence) / 0.85
            entail = nli_rows[index, :, 0]
            contra = nli_rows[index, :, 2]
            opinion = max(
                float(entail[0] + contra[1]),
                float(entail[1] + contra[0]),
                float(entail[2]),
            )
            reliabilities.append(structural * float(np.clip(opinion, 0.0, 1.0)))
        flags = np.asarray([relation_violation(question, evidence) for evidence in evidences], dtype=bool)
        return cls(
            nli_rows[:, :, 0],
            nli_rows[:, :, 1],
            nli_rows[:, :, 2],
            np.asarray(reliabilities, dtype=np.float64),
            flags,
        )


def method_outputs(item: PreparedItem, labels: tuple[str, ...]) -> dict[str, dict]:
    """Forced probabilities plus the state used by selective rules."""
    outputs: dict[str, dict] = {}
    mean_probs = _mean_probs(item.masses)
    outputs["mean"] = {"probs": mean_probs, "confidence": float(mean_probs.max()), "gate": "threshold"}

    strength = np.asarray([1.0 - ignorance for _, ignorance in item.masses])
    top = int(np.argmax(strength))
    top_probs = source_pignistic(*item.masses[top])
    outputs["top1"] = {"probs": top_probs, "confidence": float(top_probs.max()), "gate": "threshold"}

    vote_probs = _vote_probs(item.support)
    outputs["majority"] = {"probs": vote_probs, "confidence": float(vote_probs.max()), "gate": "threshold"}

    dempster_probs = item.plain.dempster_probs()
    outputs["dempster"] = {
        "probs": dempster_probs,
        "confidence": float(dempster_probs.max()),
        "gate": "threshold",
    }
    yager_probs = item.plain.yager_probs()
    outputs["yager"] = {"probs": yager_probs, "confidence": float(yager_probs.max()), "gate": "threshold"}

    relevance = np.clip(item.support.max(axis=1), 0.0, 1.0)
    outputs["crag"] = {
        "probs": mean_probs,
        "confidence": float(relevance.max()) if len(relevance) else 0.0,
        "gate": "threshold",
    }

    relevant = np.where(relevance >= 0.45)[0]
    if len(relevant):
        selfrag_probs = _mean_probs([item.masses[i] for i in relevant])
        selfrag_conf = float(selfrag_probs.max())
    else:
        selfrag_probs = np.ones(len(labels)) / len(labels)
        selfrag_conf = 0.0
    outputs["selfrag_gate"] = {"probs": selfrag_probs, "confidence": selfrag_conf, "gate": "threshold"}

    kept = [item.masses[i] for i in range(len(item.masses)) if not item.relation_flags[i]]
    kg_probs = _mean_probs(kept) if kept else np.ones(len(labels)) / len(labels)
    outputs["kg_filter"] = {"probs": kg_probs, "confidence": float(kg_probs.max()), "gate": "threshold"}

    outputs["smets_cdg"] = {
        "probs": item.plain.yager_probs(),
        "confidence": float(item.plain.singleton.max()),
        "gate": "cdg",
        "state": item.plain,
    }
    outputs["typed_yager"] = {
        "probs": item.typed.yager_probs(),
        "confidence": float(item.typed.yager_probs().max()),
        "gate": "threshold",
    }
    outputs["caer"] = caer_output(item)
    return outputs


def caer_output(item: PreparedItem, decisive: float | None = None) -> dict:
    """Agreement pool for the forced answer; conflict cap for abstention.

    On a normalized vote, conflict = 1 - leading, so the Smets dominance
    test collapses to a majority threshold. CAER instead keeps a hard cap on
    the opposing vote share and a confidence floor.
    """
    state = item.set_pooled(decisive) if decisive is not None else item.pooled
    leading = float(state.singleton.max())
    return {
        "probs": state.singleton,
        "confidence": leading * (1.0 - float(state.ignorance)),
        "gate": "pool",
        "state": state,
        "decisive": float(item.decisive),
    }


def selective_decision(
    output: dict,
    labels: tuple[str, ...],
    threshold: float,
    conflict_slack: float,
    margin: float,
) -> tuple[str | None, str]:
    """Return the label, or None when the action is abstain."""
    if output["gate"] == "cdg":
        state: FusedState = output["state"]
        answer, index = conflict_dominance_answer(state, conflict_slack, margin)
        if not answer:
            return None, "abstain"
        return labels[index], "answer"
    if output["gate"] == "pool":
        state = output["state"]
        order = np.argsort(state.singleton)[::-1]
        leading = float(state.singleton[order[0]])
        second = float(state.singleton[order[1]]) if len(order) > 1 else 0.0
        # conflict_slack carries the maximum allowed opposing vote share.
        if float(state.conflict) > conflict_slack:
            return None, "abstain"
        if leading - second < margin:
            return None, "abstain"
        if float(output["confidence"]) < threshold:
            return None, "abstain"
        return labels[int(order[0])], "answer"
    if output["confidence"] < threshold:
        return None, "abstain"
    return labels[int(np.argmax(output["probs"]))], "answer"


def leave_one_out_citation(item: PreparedItem, predicted_index: int, typed: bool) -> list[int]:
    """Cite sources whose removal reduces the predicted singleton mass.

    The agreement pool cites the decisive sentences that voted for the label.
    """
    if typed:
        cited = []
        for index in range(len(item.support)):
            strength = float(item.support[index].max())
            if int(np.argmax(item.support[index])) == predicted_index and strength >= float(item.decisive):
                if float(item.reliabilities[index]) * strength >= 0.1:
                    cited.append(index)
        return cited
    reliabilities = item.reliabilities if typed else np.full(len(item.masses), 0.85)
    discounted = [
        shafer_discount(singleton, ignorance, float(reliabilities[i]))
        for i, (singleton, ignorance) in enumerate(item.masses)
    ]
    full, _, _ = smets_fuse(discounted)
    cited = []
    for index in range(len(item.masses)):
        kept = [mass for j, mass in enumerate(discounted) if j != index]
        if not kept:
            continue
        reduced, _, _ = smets_fuse(kept)
        if float(full[predicted_index] - reduced[predicted_index]) > 1e-4:
            cited.append(index)
    return cited
