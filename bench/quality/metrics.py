"""Evidence-recall metrics.

The unit of credit is deliberately the *evidence*, not the chunk. A ranking is
good if the labelled evidence sessions/turns appear in the top-k chunks; how
many chunks that took and how much text they drag along are reported beside
it, because they are the cost side of the same decision.

Metrics:

  * session_recall@k  -- fraction of evidence *sessions* with >=1 chunk in top-k.
  * complete@k        -- 1 iff *every* evidence session is represented in top-k.
    This is the one that matters for multi-session questions: an answer
    assembled from half the evidence is usually wrong, so partial credit on
    session_recall overstates usefulness. LongMemEval's multi-session questions
    average >3 evidence sessions; complete@10 is a much harder bar than
    session_recall@10 and the gap between them is itself a finding.
  * turn_recall@k     -- fraction of labelled evidence turns covered by the
    union of top-k chunks' turn spans. Finer-grained than sessions; a session
    chunk that contains the turn gets credit, which is the correct accounting
    for coarse chunkings (they really did retrieve the evidence -- along with
    everything else, which tokens@k charges them for).
  * mrr               -- 1/rank of the first chunk from any evidence session.
  * tokens@k          -- total tokens of the top-k chunks: what the retrieval
    actually costs the context window downstream.

Abstention instances are excluded from all means; there is no evidence to
recall, so any score would be noise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .chunk import Chunk
from .data import Instance


@dataclass
class InstanceScore:
    qtype: str
    session_recall: dict[int, float]
    complete: dict[int, float]
    turn_recall: dict[int, float]
    mrr: float
    tokens: dict[int, int]


def score_ranking(
    ranking: np.ndarray, chunks: list[Chunk], inst: Instance, ks: tuple[int, ...] = (5, 10)
) -> InstanceScore:
    ev_sessions = inst.evidence_session_ids
    ev_turns = inst.evidence_turn_keys

    session_recall, complete, turn_recall, tokens = {}, {}, {}, {}
    mrr = 0.0
    for rank_pos, ci in enumerate(ranking):
        if chunks[ci].session_id in ev_sessions:
            mrr = 1.0 / (rank_pos + 1)
            break

    for k in ks:
        top = [chunks[i] for i in ranking[:k]]
        got_sessions = {c.session_id for c in top}
        got_turns = {
            (c.session_id, ti) for c in top for ti in c.turn_idxs
        }
        session_recall[k] = (
            len(ev_sessions & got_sessions) / len(ev_sessions) if ev_sessions else 0.0
        )
        complete[k] = 1.0 if ev_sessions and ev_sessions <= got_sessions else 0.0
        turn_recall[k] = (
            len(ev_turns & got_turns) / len(ev_turns) if ev_turns else 0.0
        )
        tokens[k] = sum(c.n_tokens for c in top)

    return InstanceScore(inst.qtype, session_recall, complete, turn_recall, mrr, tokens)


@dataclass
class Aggregate:
    n: int = 0
    n_abstention: int = 0
    by_metric: dict = field(default_factory=dict)
    by_qtype: dict = field(default_factory=dict)


def aggregate(scores: Iterable[InstanceScore], n_abstention: int, ks=(5, 10)) -> Aggregate:
    scores = list(scores)
    agg = Aggregate(n=len(scores), n_abstention=n_abstention)

    def summarise(subset: list[InstanceScore]) -> dict:
        out = {}
        for k in ks:
            out[f"session_recall@{k}"] = float(np.mean([s.session_recall[k] for s in subset]))
            out[f"complete@{k}"] = float(np.mean([s.complete[k] for s in subset]))
            out[f"turn_recall@{k}"] = float(np.mean([s.turn_recall[k] for s in subset]))
            out[f"tokens@{k}"] = float(np.mean([s.tokens[k] for s in subset]))
        out["mrr"] = float(np.mean([s.mrr for s in subset]))
        out["n"] = len(subset)
        return out

    agg.by_metric = summarise(scores) if scores else {}
    groups: dict[str, list[InstanceScore]] = defaultdict(list)
    for s in scores:
        groups[s.qtype].append(s)
    agg.by_qtype = {qt: summarise(g) for qt, g in sorted(groups.items())}
    return agg
