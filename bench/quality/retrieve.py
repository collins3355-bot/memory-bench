"""Retrieval arms.

Every arm consumes precomputed per-chunk signals (BM25 scores, cosine scores,
ages) and emits a ranking. Fusion is reciprocal-rank (RRF, k=60) rather than
score interpolation, for one load-bearing reason: RRF is *parameter-free*.
Score interpolation needs a mixing weight, and with only 500 eval questions
any tuned weight would be fitted to the test set -- the resulting table would
be an overfitting exercise wearing a benchmark's clothes. RRF ranks are scale-
free, so lexical and dense scores need no calibration to be combined.

Recency enters the same way: as a third ranked list (newest first) fused in,
not as a decay multiplied into scores. Same reason -- a decay rate is a free
parameter; a rank list is not. The cost of this choice is that recency gets a
full vote even when the question has no temporal aspect, which is exactly the
trade the eval is meant to expose (does a recency vote pay for itself across
all question types, or only on knowledge-update?).
"""

from __future__ import annotations

import numpy as np


def _ranks(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    """rank[i] = 0-based rank of item i under `scores`, ties sharing their mean rank.

    Tie handling is not pedantry here -- it is most of the BM25 list. A typical
    query shares terms with a small fraction of chunks, so the majority of the
    corpus scores exactly 0.0 and forms one giant tie block. Plain argsort
    ranks that block by array index, and RRF then converts those arbitrary
    positions into real score differences: chunks that happen to sit early in
    the haystack outrank identical-scored later ones in every fused arm. The
    unit test that caught this had a fresher chunk with an identical content
    score losing to a staler one purely because of its index.

    Averaging each tie group's ranks (the standard treatment) makes fused
    scores invariant to haystack order.
    """
    s = -scores if descending else scores
    order = np.argsort(s, kind="stable")
    sorted_s = s[order]
    # Boundaries of runs of equal value in the sorted array; each run's items
    # all receive the run's mean position. Detected by *equality*, not by
    # np.diff: the age list is mostly-inf whenever dates are missing, and
    # inf - inf is NaN, which np.diff would read as a boundary -- silently
    # splitting the one tie group that most needs to stay tied.
    boundaries = np.flatnonzero(sorted_s[1:] != sorted_s[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(s)]])
    ranks = np.empty(len(s), dtype=np.float64)
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + b - 1) / 2.0
    return ranks


def rrf(rank_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Fused score per item: sum over lists of 1/(k + rank + 1)."""
    fused = np.zeros(len(rank_lists[0]), dtype=np.float64)
    for r in rank_lists:
        fused += 1.0 / (k + r + 1.0)
    return fused


def rank_arm(
    arm: str,
    bm25_scores: np.ndarray,
    cos_scores: np.ndarray,
    age_days: np.ndarray,
) -> np.ndarray:
    """Return chunk indices, best first.

    age_days: days between question date and chunk's session date; chunks with
    unknown dates carry +inf so a recency vote puts them last, not first.
    """
    if arm == "bm25":
        scores = bm25_scores
    elif arm == "vector":
        scores = cos_scores
    elif arm == "hybrid":
        scores = rrf([_ranks(bm25_scores), _ranks(cos_scores)])
    elif arm == "vector_time":
        scores = rrf([_ranks(cos_scores), _ranks(age_days, descending=False)])
    elif arm == "hybrid_time":
        scores = rrf(
            [
                _ranks(bm25_scores),
                _ranks(cos_scores),
                _ranks(age_days, descending=False),
            ]
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return np.argsort(-scores, kind="stable")


ARMS = ("bm25", "vector", "hybrid", "vector_time", "hybrid_time")
