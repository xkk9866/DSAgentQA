"""Typed residual fusion and the conflict-dominance gate.

The predictive step keeps Smets' unnormalized conjunctive rule, so the empty-set
mass stays observable. Scope mismatch and structured-relation violations change
source reliability by Shafer discounting before fusion. Stance conflict is not
discounted away and is not renormalized onto a singleton.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = ("yes", "no", "maybe")


def nli_to_mass(entail: np.ndarray, neutral: np.ndarray, contra: np.ndarray) -> tuple[np.ndarray, float]:
    """Map bidirectional NLI scores to a simple support function.

    Index 0 is the affirmative claim, index 1 its negation, index 2 an
    uncertainty claim. A finding supports yes when it entails the claim or
    contradicts the negation. Scientific abstracts often do the second without
    lexically entailing the question wording. Singleton masses and ignorance
    sum to 1. The source itself assigns no empty-set mass.
    """
    entail = np.asarray(entail, dtype=np.float64)
    neutral = np.asarray(neutral, dtype=np.float64)
    contra = np.asarray(contra, dtype=np.float64)
    support_yes = float(entail[0] + contra[1])
    support_no = float(entail[1] + contra[0])
    support_maybe = float(entail[2])
    if max(support_yes, support_no) < 0.45:
        support_maybe = max(support_maybe, float(0.5 * (neutral[0] + neutral[1])))
    support = np.array([support_yes, support_no, support_maybe], dtype=np.float64)
    if float(support.sum()) < 1e-8:
        return np.zeros(3, dtype=np.float64), 1.0
    neutral_pair = float(0.5 * (neutral[0] + neutral[1]))
    opinion = float(np.clip(1.0 - neutral_pair, 0.0, 1.0))
    peak = float(support.max() / support.sum())
    alpha = float(np.clip(opinion * peak, 0.0, 1.0))
    singleton = alpha * support / support.sum()
    return singleton, 1.0 - float(singleton.sum())


def shafer_discount(singleton: np.ndarray, ignorance: float, reliability: float) -> tuple[np.ndarray, float]:
    """Shafer reliability discount. reliability=1 leaves the source unchanged."""
    reliability = float(np.clip(reliability, 0.0, 1.0))
    singleton = np.asarray(singleton, dtype=np.float64) * reliability
    ignorance = 1.0 - reliability + reliability * float(ignorance)
    return singleton, ignorance


def smets_fuse(masses: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, float, float]:
    """Closed-form Smets combination for simple support functions.

    For each label y,
        m({y}) = prod_i (m_i({y}) + m_i(Theta)) - prod_i m_i(Theta).
    Ignorance is the product of ignorance masses. The residual is conflict
    m(empty) and is not divided out.
    """
    if not masses:
        raise ValueError("smets_fuse requires at least one mass")
    width = len(masses[0][0])
    prod_support = np.ones(width, dtype=np.float64)
    prod_ignorance = 1.0
    for singleton, ignorance in masses:
        singleton = np.asarray(singleton, dtype=np.float64)
        ignorance = float(ignorance)
        prod_support *= singleton + ignorance
        prod_ignorance *= ignorance
    singleton = np.clip(prod_support - prod_ignorance, 0.0, None)
    ignorance = float(prod_ignorance)
    nonempty = float(singleton.sum() + ignorance)
    conflict = float(max(0.0, 1.0 - nonempty))
    total = float(singleton.sum() + ignorance + conflict)
    if total <= 0:
        return np.ones(width) / width, 0.0, 0.0
    return singleton / total, ignorance / total, conflict / total


def yager_pignistic(singleton: np.ndarray, ignorance: float, conflict: float) -> np.ndarray:
    """Move conflict into ignorance, then apply the pignistic transform."""
    width = len(singleton)
    ignorance = ignorance + conflict
    probs = np.asarray(singleton, dtype=np.float64) + ignorance / width
    total = float(probs.sum())
    if total <= 0:
        return np.ones(width) / width
    return probs / total


def dempster_pignistic(singleton: np.ndarray, ignorance: float, conflict: float) -> np.ndarray:
    """Dempster normalization, then the pignistic transform.

    The uniform fallback is only for a vanishing normalizing constant. The
    classical Zadeh inputs remain normalized, which concentrates the residual
    mass on the singleton that neither source supported.
    """
    width = len(singleton)
    scale = 1.0 - conflict
    if scale <= 1e-15:
        return np.ones(width) / width
    masses = np.asarray(singleton, dtype=np.float64) / scale
    ignorance = ignorance / scale
    probs = masses + ignorance / width
    total = float(probs.sum())
    return probs / total


def source_pignistic(singleton: np.ndarray, ignorance: float) -> np.ndarray:
    width = len(singleton)
    probs = np.asarray(singleton, dtype=np.float64) + ignorance / width
    total = float(probs.sum())
    return probs / total


@dataclass
class FusedState:
    singleton: np.ndarray
    ignorance: float
    conflict: float
    reliabilities: np.ndarray

    @property
    def prediction_index(self) -> int:
        return int(np.argmax(self.singleton))

    def yager_probs(self) -> np.ndarray:
        return yager_pignistic(self.singleton, self.ignorance, self.conflict)

    def dempster_probs(self) -> np.ndarray:
        return dempster_pignistic(self.singleton, self.ignorance, self.conflict)


def _fuse_opinions(
    belief_a: np.ndarray,
    uncertainty_a: float,
    belief_b: np.ndarray,
    uncertainty_b: float,
    dependence: float,
) -> tuple[np.ndarray, float]:
    """Blend cumulative fusion (independent) with averaging fusion (dependent).

    Averaging fusion is idempotent: two copies of one opinion stay that opinion.
    Cumulative fusion (Jøsang) shrinks uncertainty when independent sources agree.
    ``dependence`` is 1 for a repeated source and 0 for an independent source.
    """
    dependence = float(np.clip(dependence, 0.0, 1.0))
    belief_a = np.asarray(belief_a, dtype=np.float64)
    belief_b = np.asarray(belief_b, dtype=np.float64)
    uncertainty_a = float(uncertainty_a)
    uncertainty_b = float(uncertainty_b)
    den_avg = uncertainty_a + uncertainty_b
    den_cum = den_avg - uncertainty_a * uncertainty_b
    if den_avg <= 1e-12:
        belief = 0.5 * (belief_a + belief_b)
        uncertainty = 0.0
    else:
        cross_a = belief_a * uncertainty_b + belief_b * uncertainty_a
        belief_avg = cross_a / den_avg
        uncertainty_avg = 2.0 * uncertainty_a * uncertainty_b / den_avg
        if den_cum <= 1e-12:
            belief_cum, uncertainty_cum = belief_avg, uncertainty_avg
        else:
            belief_cum = cross_a / den_cum
            uncertainty_cum = uncertainty_a * uncertainty_b / den_cum
        belief = (1.0 - dependence) * belief_cum + dependence * belief_avg
        uncertainty = (1.0 - dependence) * uncertainty_cum + dependence * uncertainty_avg
    total = float(belief.sum() + uncertainty)
    if total <= 0:
        width = len(belief_a)
        return np.ones(width) / width, 0.0
    return belief / total, float(uncertainty / total)


def dependent_consensus(
    support: np.ndarray,
    reliability: np.ndarray,
    similarity: np.ndarray,
    decisive: float = 0.35,
) -> FusedState:
    """Fuse same-proposition votes without double-counting near-duplicate papers.

    A source below ``decisive`` stays vacuous. Sources that share a label are
    fused from strongest to weakest; the dependence of a new source is its
    maximum similarity to a source already counted for that label. Opposite
    labels are then fused as independent opinions, so disagreement stays
    visible as split belief instead of being discounted away.
    """
    support = np.asarray(support, dtype=np.float64)
    reliability = np.asarray(reliability, dtype=np.float64)
    similarity = np.clip(np.asarray(similarity, dtype=np.float64), 0.0, 1.0)
    width = support.shape[1]
    opinions: list[tuple[int, np.ndarray, float]] = []
    vacuous = 0.0
    for index in range(len(support)):
        row = support[index]
        strength = float(row.max())
        if strength < decisive or float(reliability[index]) <= 0:
            vacuous += max(float(reliability[index]), 0.0)
            continue
        commitment = float(np.clip(float(reliability[index]) * strength, 0.0, 0.95))
        if commitment <= 0:
            vacuous += max(float(reliability[index]), 0.0)
            continue
        belief = np.zeros(width, dtype=np.float64)
        belief[int(np.argmax(row))] = commitment
        opinions.append((index, belief, 1.0 - commitment))
    if not opinions:
        return FusedState(np.ones(width) / width, 1.0, 1.0, reliability)

    clusters: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for index, belief, uncertainty in opinions:
        clusters.setdefault(int(np.argmax(belief)), []).append((index, belief, uncertainty))
    cluster_opinions: list[tuple[np.ndarray, float]] = []
    for members in clusters.values():
        members.sort(key=lambda item: float(item[1].max()), reverse=True)
        belief, uncertainty = members[0][1], members[0][2]
        seen = [members[0][0]]
        for index, belief_i, uncertainty_i in members[1:]:
            dependence = max(float(similarity[index, earlier]) for earlier in seen)
            belief, uncertainty = _fuse_opinions(belief, uncertainty, belief_i, uncertainty_i, dependence)
            seen.append(index)
        cluster_opinions.append((belief, uncertainty))
    belief, uncertainty = cluster_opinions[0]
    for belief_i, uncertainty_i in cluster_opinions[1:]:
        belief, uncertainty = _fuse_opinions(belief, uncertainty, belief_i, uncertainty_i, 0.0)
    committed = float(sum(item[1].sum() for item in opinions))
    ignorance_mix = float(vacuous / (vacuous + committed)) if vacuous + committed > 0 else 0.0
    belief = belief * (1.0 - ignorance_mix)
    uncertainty = uncertainty + (1.0 - uncertainty) * ignorance_mix
    total = float(belief.sum() + uncertainty)
    belief = belief / total
    uncertainty = float(uncertainty / total)
    belief_mass = float(belief.sum())
    conflict = float(1.0 - belief.max() / belief_mass) if belief_mass > 1e-8 else 1.0
    probs = belief + uncertainty / width
    return FusedState(probs / float(probs.sum()), uncertainty, conflict, reliability)


def agreement_pool(
    support: np.ndarray,
    reliability: np.ndarray,
    decisive: float = 0.35,
) -> FusedState:
    """Pool sources that agree, and keep disagreement as conflict.

    Smets multiplication treats every sentence as an independent veto, so
    agreeing findings manufacture empty-set mass. Here a source votes for its
    leading label with weight reliability times its margin. Conflict is the
    share of that weight sitting outside the leading label. Sources below the
    decisive threshold contribute only ignorance.
    """
    support = np.asarray(support, dtype=np.float64)
    reliability = np.asarray(reliability, dtype=np.float64)
    width = support.shape[1]
    weights = np.zeros(width, dtype=np.float64)
    vacuous = 0.0
    for index in range(len(support)):
        row = support[index]
        strength = float(row.max())
        margin = float(row.max() - np.partition(row, -2)[-2]) if width > 1 else strength
        vote_weight = float(reliability[index]) * margin
        if strength < decisive or vote_weight <= 0:
            vacuous += max(float(reliability[index]), 0.0)
            continue
        weights[int(np.argmax(row))] += vote_weight
    committed = float(weights.sum())
    if committed <= 0:
        return FusedState(np.ones(width) / width, 1.0, 1.0, reliability)
    singleton = weights / committed
    conflict = float(1.0 - singleton.max())
    ignorance = float(vacuous / (vacuous + committed))
    return FusedState(singleton, ignorance, conflict, reliability)


def fuse_sources(
    source_masses: list[tuple[np.ndarray, float]],
    reliabilities: np.ndarray,
) -> FusedState:
    discounted = [
        shafer_discount(singleton, ignorance, float(reliabilities[i]))
        for i, (singleton, ignorance) in enumerate(source_masses)
    ]
    singleton, ignorance, conflict = smets_fuse(discounted)
    return FusedState(singleton, ignorance, conflict, np.asarray(reliabilities, dtype=np.float64))


def conflict_dominance_answer(
    state: FusedState,
    conflict_slack: float,
    margin: float,
) -> tuple[bool, int]:
    """Answer only when the leading singleton dominates conflict and the runner-up.

    conflict_slack shifts the dominance test: a positive slack abstains earlier
    (conflict > leading mass - slack). The gate reads unnormalized conflict, so
    Dempster normalization cannot express the same test after the fact.
    """
    order = np.argsort(state.singleton)[::-1]
    top = int(order[0])
    leading = float(state.singleton[top])
    second = float(state.singleton[order[1]]) if len(order) > 1 else 0.0
    dominates_conflict = state.conflict <= leading - conflict_slack
    dominates_rival = leading - second >= margin
    dominates_ignorance = leading >= state.ignorance
    answer = bool(dominates_conflict and dominates_rival and dominates_ignorance and leading > 0)
    return answer, top
