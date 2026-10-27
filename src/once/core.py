"""ONCE: count every piece of evidence once.

Each passage is read in isolation. Its evidence for candidate c is a log Bayes factor
against the closed-book reading, so the model prior enters the posterior once instead of
once per passage. Passages that back the same answer and say nearly the same thing share
one vote. Each passage moves any log-odds by at most M nats.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import numpy as np

IDK_PATTERNS = ("i don't know", "i do not know", "unknown", "no information", "not mentioned",
                "not provided", "cannot be determined", "not specified", "none", "n/a", "unclear",
                "does not", "doesn't", "not stated", "no answer", "not available", "insufficient")


def normalize(s: str) -> str:
    s = s.lower().replace(" ", " ")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_idk(s: str) -> bool:
    t = s.lower().strip()
    return (not t) or any(p in t for p in IDK_PATTERNS)


def clean_answer(s: str) -> str:
    s = s.strip().split("\n")[0].strip()
    s = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", s, flags=re.I)
    return s.strip().strip(".").strip()


def build_candidates(answers: list[str], max_cands: int = 8) -> list[str]:
    """Distinct non-abstaining answers, most frequent first."""
    seen: dict[str, list] = {}
    for a in answers:
        a = clean_answer(a)
        if is_idk(a) or len(a) > 120:
            continue
        k = normalize(a)
        if not k:
            continue
        if k not in seen:
            seen[k] = [a, 0]
        seen[k][1] += 1
    items = sorted(seen.values(), key=lambda x: -x[1])
    return [a for a, _ in items[:max_cands]]


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


@dataclass
class Reading:
    """Cached isolated readings for one question.

    L[0] is the closed-book row, L[1:] are passages. Columns are candidates, plus a last
    column for "I don't know".
    """
    L: np.ndarray
    emb: np.ndarray  # (n, d) unit-norm passage embeddings

    @property
    def n_cand(self) -> int:
        return self.L.shape[1] - 1

    def parts(self):
        C = self.n_cand
        lp0 = log_softmax(self.L[0, :C])
        full = log_softmax(self.L[1:], axis=1)
        rel = 1.0 - np.exp(full[:, C])
        q = log_softmax(self.L[1:, :C], axis=1)
        return lp0, q, rel, full


def evidence(lp0: np.ndarray, q: np.ndarray, rel: np.ndarray, M: float) -> np.ndarray:
    """Clipped log Bayes factor of each passage; an irrelevant passage gives exactly 0."""
    p0 = np.exp(lp0)[None]
    mix = rel[:, None] * np.exp(q) + (1.0 - rel[:, None]) * p0
    e = np.log(np.clip(mix, 1e-12, None)) - lp0[None]
    return np.clip(e, -M, M)


def claims(full: np.ndarray, C: int) -> np.ndarray:
    """Answer each passage asserts; -1 when it abstains."""
    a = full.argmax(axis=1)
    return np.where(a == C, -1, a)


def redundancy_weights(emb: np.ndarray, claim: np.ndarray, tau: float) -> np.ndarray:
    """w_i = 1 / sum_j K_ij over passages making the same claim.

    K_ij rises linearly from 0 at cosine tau to 1 at cosine 1, so k exact copies split one
    vote. Passages with different claims never share weight, which keeps an injected
    passage from diluting passages that disagree with it.
    """
    n = len(claim)
    if n == 0:
        return np.zeros(0)
    if tau >= 1.0:
        return np.ones(n)
    cos = emb @ emb.T
    K = np.clip((cos - tau) / (1.0 - tau), 0.0, 1.0)
    same = (claim[:, None] == claim[None, :]) & (claim[:, None] >= 0)
    K = np.where(same, K, 0.0)
    np.fill_diagonal(K, 1.0)
    return 1.0 / K.sum(axis=1)


def once(r: Reading, beta: float = 1.0, M: float = 3.0, tau: float = 0.8,
         prior_once: bool = True, dedup: bool = True) -> dict:
    C = r.n_cand
    lp0, q, rel, full = r.parts()
    if prior_once:
        e = evidence(lp0, q, rel, M)
        base = beta * lp0
    else:  # naive product of experts: every passage carries the prior with it
        e = np.clip(q, -1e9, None)
        base = np.zeros(C)
    claim = claims(full, C)
    w = redundancy_weights(r.emb, claim, tau) if dedup else np.ones(len(claim))
    S = base + (w[:, None] * e).sum(axis=0)
    P = np.exp(log_softmax(S))
    order = np.argsort(-S)
    top, second = order[0], (order[1] if C > 1 else order[0])
    push = w[:, None] * e
    support = np.abs(push).sum(axis=0)
    # conflict: share of absolute evidence on the leader that pulls against it
    against = np.clip(-push[:, top], 0, None).sum()
    conflict = float(against / (support[top] + 1e-9))
    return {
        "probs": P, "pred": int(top),
        "margin": float(S[top] - S[second]) if C > 1 else float("inf"),
        "conflict": conflict, "weights": w, "claims": claim, "rel": rel,
        "n_eff": float(w[claim >= 0].sum()) if (claim >= 0).any() else 0.0,
    }


def certified_radius(margin: float, M: float) -> int:
    """Largest k such that any k injected passages that do not claim the prediction cannot flip it.

    Each such passage has weight at most 1 and moves the leader and any rival by at most M
    each, and displacing a clean passage removes at most 2M of margin; so 4kM < margin.
    """
    if not np.isfinite(margin):
        return 10 ** 6
    return max(0, int(np.ceil(margin / (4.0 * M))) - 1)
