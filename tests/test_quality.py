"""Unit checks for the quality-eval machinery.

Each test here is a regression guard for a bug that actually occurred while
building this, not a coverage exercise. The two ranking ones matter most:
plain argsort ranks turned tie blocks into index-position bias inside RRF, and
np.diff-based tie detection split all-inf age blocks because inf - inf is NaN.
Both produced plausible rankings and would have shipped wrong tables.

Run: .venv/bin/python -m pytest tests/ -q   (or plain `python tests/test_quality.py`)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.quality.chunk import Chunk, chunk_window  # noqa: E402
from bench.quality.data import Instance, Turn, _parse_lme_date, _truthy  # noqa: E402
from bench.quality.metrics import score_ranking  # noqa: E402
from bench.quality.retrieve import _ranks, rank_arm, rrf  # noqa: E402
from bench.quality.run import ages_for  # noqa: E402


def _chunk(dt=None, sid="s", turns=(0,), tokens=10):
    return Chunk("u", sid, tuple(turns), "x", dt, tokens)


# ---------------------------------------------------------------- ranking


def test_ranks_no_ties():
    assert list(_ranks(np.array([3.0, 1.0, 2.0]))) == [0.0, 2.0, 1.0]


def test_ranks_tie_block_shares_mean_rank():
    # Regression: argsort ranks gave tie blocks index-order positions, which
    # RRF converted into a systematic early-in-corpus advantage.
    assert list(_ranks(np.array([0.0, 0.0, 0.0]))) == [1.0, 1.0, 1.0]
    assert list(_ranks(np.array([2.0, 0.0, 0.0, 1.0]))) == [0.0, 2.5, 2.5, 1.0]


def test_ranks_all_inf_ties():
    # Regression: np.diff on [inf, inf] is NaN, which read as a boundary and
    # split the undated-chunk tie group.
    assert list(_ranks(np.array([np.inf, np.inf]), descending=False)) == [0.5, 0.5]


def test_recency_breaks_content_tie():
    bm = np.array([1.0, 1.0, 0.0])
    cos = np.array([1.0, 1.0, 0.0])
    ages = np.array([50.0, 1.0, np.inf])
    assert list(rank_arm("hybrid_time", bm, cos, ages)[:2]) == [1, 0]


def test_fused_ranking_is_corpus_order_invariant():
    rng = np.random.default_rng(0)
    bm = np.array([0.0, 0.0, 0.0, 0.0, 3.0])
    cos = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    ages = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    base = rank_arm("hybrid", bm, cos, ages)
    perm = rng.permutation(5)
    shuffled = rank_arm("hybrid", bm[perm], cos[perm], ages[perm])
    assert list(perm[shuffled]) == list(base)


def test_rrf_rewards_agreement():
    fused = rrf([np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 2.0])])
    assert fused[0] == fused[1] > fused[2]


# ---------------------------------------------------------------- metrics


def _instance(ev_sessions, ev_turns):
    return Instance(
        "q", "t", "?", None, "a", [], set(ev_sessions), set(ev_turns), False, "h"
    )


def test_score_ranking_session_and_turn_recall():
    chunks = [_chunk(sid=s) for s in ("s1", "s2", "s3")] + [
        _chunk(sid="s4", turns=(0, 1))
    ]
    inst = _instance({"s1", "s4"}, {("s1", 0), ("s4", 1)})
    s = score_ranking(np.array([0, 1, 2, 3]), chunks, inst, ks=(2, 4))
    assert s.session_recall[2] == 0.5 and s.session_recall[4] == 1.0
    assert s.complete[2] == 0.0 and s.complete[4] == 1.0
    assert s.turn_recall[2] == 0.5 and s.turn_recall[4] == 1.0
    assert s.mrr == 1.0 and s.tokens[2] == 20


def test_mrr_counts_rank_of_first_evidence():
    chunks = [_chunk(sid=s) for s in ("s1", "s2", "s3", "s4")]
    inst = _instance({"s1"}, {("s1", 0)})
    s = score_ranking(np.array([1, 2, 0, 3]), chunks, inst, ks=(2,))
    assert abs(s.mrr - 1 / 3) < 1e-9


# ---------------------------------------------------------------- data / ages


def test_has_answer_is_a_string_in_longmemeval():
    assert _truthy("True") and _truthy(True)
    assert not _truthy("False") and not _truthy(False) and not _truthy(None)


def test_lme_date_parse():
    assert _parse_lme_date("2023/04/10 (Mon) 17:50") == datetime(2023, 4, 10, 17, 50)
    assert _parse_lme_date("garbage") is None


def test_ages_undated_chunks_never_look_fresh():
    q = datetime(2023, 5, 10)
    ages = ages_for([_chunk(datetime(2023, 5, 1)), _chunk(None), _chunk(datetime(2023, 5, 9))], q)
    assert ages[0] == 9.0 and np.isinf(ages[1]) and ages[2] == 1.0


def test_ages_fall_back_to_newest_session_without_question_date():
    ages = ages_for([_chunk(datetime(2023, 3, 1)), _chunk(datetime(2023, 4, 1))], None)
    assert ages[1] == 0.0 and ages[0] == 31.0


def test_window_chunker_covers_short_sessions():
    inst = Instance(
        "q", "t", "?", None, "a",
        [__import__("bench.quality.data", fromlist=["Session"]).Session(
            "s1", None, [Turn("user", f"t{i}") for i in range(3)]
        )],
        set(), set(), False, "h",
    )
    chunks = chunk_window(inst, size=4, stride=2)
    assert len(chunks) == 1 and chunks[0].turn_idxs == (0, 1, 2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok {fn.__name__}")
    print(f"{len(fns)} tests passed")
