"""Chunking policies: what is the unit of memory?

This choice is upstream of every retrieval number and is itself one of the
experimental variables:

  * **turn** -- one chunk per utterance. Precise targets, cheap to embed, but
    a turn often lacks the context that makes it findable ("yes, that one" is
    unfindable on its own).
  * **window** -- a sliding window of turns (default 4, stride 2). The middle
    ground: local context without whole-session dilution. Stride < size means
    overlap, so a fact near a boundary appears in two chunks.
  * **session** -- one chunk per session. Maximum context, but two costs: the
    embedding sees only the first `max_len` tokens (a small encoder truncates;
    the tail of a long session is invisible to dense retrieval), and top-k
    returns far more tokens for the same k, which is a real context-budget
    cost to whatever consumes the retrieval.

Token counts are computed here, once per chunk, because "tokens returned at
k=10" is a first-class output of the eval -- an arm that wins recall by
returning 6x more text has not won.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from .data import Instance


@dataclass
class Chunk:
    uid: str  # sha1 of text -- the embedding-cache key
    session_id: str
    turn_idxs: tuple[int, ...]
    text: str
    dt: datetime | None
    n_tokens: int = 0  # stamped by run.py in one tokenizer pass


def _uid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _turn_text(role: str, content: str) -> str:
    return f"{role}: {content}"


def chunk_turn(inst: Instance) -> list[Chunk]:
    out = []
    for sess in inst.sessions:
        for i, t in enumerate(sess.turns):
            text = _turn_text(t.role, t.content)
            out.append(Chunk(_uid(text), sess.id, (i,), text, sess.dt))
    return out


def chunk_window(inst: Instance, size: int = 4, stride: int = 2) -> list[Chunk]:
    out = []
    for sess in inst.sessions:
        n = len(sess.turns)
        start = 0
        while start < n:
            idxs = tuple(range(start, min(start + size, n)))
            text = "\n".join(_turn_text(sess.turns[i].role, sess.turns[i].content) for i in idxs)
            out.append(Chunk(_uid(text), sess.id, idxs, text, sess.dt))
            if start + size >= n:
                break
            start += stride
    return out


def chunk_session(inst: Instance) -> list[Chunk]:
    out = []
    for sess in inst.sessions:
        idxs = tuple(range(len(sess.turns)))
        text = "\n".join(_turn_text(t.role, t.content) for t in sess.turns)
        out.append(Chunk(_uid(text), sess.id, idxs, text, sess.dt))
    return out


CHUNKERS = {
    "turn": chunk_turn,
    "window": chunk_window,
    "session": chunk_session,
}
