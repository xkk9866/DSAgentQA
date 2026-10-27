"""Invariants that separate typed residual fusion from Dempster normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caer.fusion import (
    agreement_pool,
    conflict_dominance_answer,
    dempster_pignistic,
    dependent_consensus,
    fuse_sources,
    smets_fuse,
)
from caer.typing import relation_violation, scope_mismatch, source_reliability


def zadeh_case() -> dict:
    mass_a = np.array([0.99, 0.01, 0.0])
    mass_b = np.array([0.0, 0.01, 0.99])
    singleton, ignorance, conflict = smets_fuse([(mass_a, 0.0), (mass_b, 0.0)])
    dempster = dempster_pignistic(singleton, ignorance, conflict)
    state = fuse_sources([(mass_a, 0.0), (mass_b, 0.0)], np.ones(2))
    answer, index = conflict_dominance_answer(state, conflict_slack=0.0, margin=0.0)
    return {
        "conflict": float(conflict),
        "singleton": singleton.tolist(),
        "dempster_probs": dempster.tolist(),
        "dempster_picks_unsupported_middle": bool(int(np.argmax(dempster)) == 1 and dempster[1] > 0.9),
        "caer_abstains": not answer,
        "caer_index_if_forced": int(index),
    }


def pool_case() -> dict:
    support = np.array([[0.99, 0.01, 0.0], [0.0, 0.01, 0.99]])
    state = agreement_pool(support, np.ones(2), decisive=0.5)
    return {
        "middle_mass": float(state.singleton[1]),
        "conflict": float(state.conflict),
        "does_not_pick_middle": bool(state.singleton[1] < 0.05 and state.conflict > 0.4),
    }


def scope_case() -> dict:
    question = "Does the drug reduce symptoms in adult patients?"
    human = "In adult patients the drug reduced symptoms."
    mouse = "In mice the drug did not reduce symptoms."
    assert scope_mismatch(question, mouse)
    assert not scope_mismatch(question, human)
    human_mass = (np.array([0.8, 0.0, 0.0]), 0.2)
    mouse_mass = (np.array([0.0, 0.8, 0.0]), 0.2)
    plain = fuse_sources([human_mass, mouse_mass], np.array([0.85, 0.85]))
    typed = fuse_sources(
        [human_mass, mouse_mass],
        np.array(
            [source_reliability(question, human), source_reliability(question, mouse)],
            dtype=np.float64,
        ),
    )
    plain_answer, _ = conflict_dominance_answer(plain, 0.0, 0.0)
    typed_answer, typed_index = conflict_dominance_answer(typed, 0.0, 0.0)
    return {
        "plain_conflict": float(plain.conflict),
        "typed_conflict": float(typed.conflict),
        "typed_singleton": typed.singleton.tolist(),
        "plain_abstains": not plain_answer,
        "typed_answers_yes": bool(typed_answer and typed_index == 0),
    }


def relation_case() -> dict:
    question = "Is combining phenelzine and sertraline safe?"
    bad = "The combination of phenelzine and sertraline is safe and recommended."
    good = "Phenelzine with an SSRI is contraindicated because of serotonin syndrome."
    assert relation_violation(question, bad)
    assert not relation_violation(question, good)
    good_mass = (np.array([0.0, 0.8, 0.0]), 0.2)
    bad_mass = (np.array([0.8, 0.0, 0.0]), 0.2)
    typed = fuse_sources(
        [good_mass, bad_mass],
        np.array(
            [source_reliability(question, good), source_reliability(question, bad)],
            dtype=np.float64,
        ),
    )
    answer, index = conflict_dominance_answer(typed, 0.0, 0.0)
    return {
        "typed_singleton": typed.singleton.tolist(),
        "answers_no": bool(answer and index == 1),
        "bad_reliability": source_reliability(question, bad),
    }


def dependent_case() -> dict:
    one = np.array([[0.90, 0.05, 0.05]])
    duplicate = np.array([[0.90, 0.05, 0.05], [0.90, 0.05, 0.05]])
    opposite = np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]])
    same = np.ones((2, 2))
    single = dependent_consensus(one, np.ones(1), np.ones((1, 1)), decisive=0.5)
    copied = dependent_consensus(duplicate, np.ones(2), same, decisive=0.5)
    independent = dependent_consensus(duplicate, np.ones(2), np.eye(2), decisive=0.5)
    conflicted = dependent_consensus(opposite, np.ones(2), same, decisive=0.5)
    return {
        "duplicate_matches_single": bool(np.allclose(copied.singleton, single.singleton, atol=1e-6)),
        "independent_is_surer": bool(independent.ignorance < single.ignorance - 1e-4),
        "opposite_keeps_split": bool(
            conflicted.singleton[2] < 0.08
            and conflicted.singleton[0] > 0.3
            and conflicted.singleton[1] > 0.3
            and conflicted.conflict > 0.4
        ),
        "single_ignorance": float(single.ignorance),
        "independent_ignorance": float(independent.ignorance),
        "opposite_conflict": float(conflicted.conflict),
    }


def main() -> None:
    report = {
        "zadeh": zadeh_case(),
        "scope": scope_case(),
        "relation": relation_case(),
        "pool": pool_case(),
        "dependent": dependent_case(),
    }
    ok = (
        report["zadeh"]["dempster_picks_unsupported_middle"]
        and report["zadeh"]["caer_abstains"]
        and report["scope"]["typed_answers_yes"]
        and report["scope"]["plain_abstains"]
        and report["relation"]["answers_no"]
        and report["pool"]["does_not_pick_middle"]
        and report["dependent"]["duplicate_matches_single"]
        and report["dependent"]["independent_is_surer"]
        and report["dependent"]["opposite_keeps_split"]
    )
    report["passed"] = ok
    print(report)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
