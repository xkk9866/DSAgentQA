"""Scope mismatch and structured relation violations.

These checks set Shafer discounts. They do not see the gold label.
"""

from __future__ import annotations

import re

# Population groups are matched in this order. Child and animal markers are
# more specific than the generic human markers, so they are tested first.
_GROUP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "child",
        re.compile(r"\b(child|children|infant|infants|neonate|neonates|adolescent|adolescents|pediatric|paediatric)\b", re.I),
    ),
    (
        "animal",
        re.compile(r"\b(mouse|mice|rat|rats|murine|rodent|rodents|zebrafish|rabbit|rabbits|canine|porcine|primate|primates|animal model)\b", re.I),
    ),
    (
        "invitro",
        re.compile(r"\b(in vitro|cell line|cell lines|cultured cells|hek293|hela)\b", re.I),
    ),
    (
        "human",
        re.compile(r"\b(patient|patients|human|humans|adult|adults|clinical|woman|women|man|men|participant|participants)\b", re.I),
    ),
]

# Well-known contraindicated pairs used only as a relation-violation probe.
# A source that calls one of these combinations safe is discounted.
_CONTRAINDICATED: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("phenelzine", "tranylcypromine", "isocarboxazid", "moclobemide", "maoi", "monoamine oxidase inhibitor"),
        ("sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram", "fluvoxamine", "ssri", "venlafaxine", "duloxetine", "snri"),
    ),
    (
        ("clozapine",),
        ("carbamazepine",),
    ),
]

_SAFE_CLAIM = re.compile(
    r"(is safe|are safe|safe and (?:effective|recommended|well tolerated)|"
    r"can be (?:safely )?combined|combination is (?:safe|effective|recommended)|"
    r"no (?:clinically significant )?interaction|without serotonin syndrome)",
    re.I,
)


def population_groups(text: str) -> set[str]:
    found: set[str] = set()
    for name, pattern in _GROUP_PATTERNS:
        if pattern.search(text):
            found.add(name)
    if "child" in found or "animal" in found or "invitro" in found:
        found.discard("human")
    return found


def scope_mismatch(question: str, evidence: str) -> bool:
    """True when the question and the evidence name disjoint populations."""
    question_groups = population_groups(question)
    evidence_groups = population_groups(evidence)
    if not question_groups or not evidence_groups:
        return False
    return question_groups.isdisjoint(evidence_groups)


def relation_violation(question: str, evidence: str) -> bool:
    """True when evidence asserts safety for a contraindicated drug pair."""
    combined = f"{question} {evidence}".lower()
    if _SAFE_CLAIM.search(evidence) is None:
        return False
    for left, right in _CONTRAINDICATED:
        if any(token in combined for token in left) and any(token in combined for token in right):
            return True
    return False


def source_reliability(
    question: str,
    evidence: str,
    base: float = 0.85,
    scope_factor: float = 0.2,
    relation_factor: float = 0.25,
) -> float:
    reliability = base
    if scope_mismatch(question, evidence):
        reliability *= scope_factor
    if relation_violation(question, evidence):
        reliability *= relation_factor
    return float(reliability)
