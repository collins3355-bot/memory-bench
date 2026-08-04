"""SQLite text store -- the step benchmarks usually leave out.

Vector search returns integer ids. A memory system has to turn those into text
before anything can be done with them, and that is a random-access read against
a table far too large for cache. It is a genuine line item in the latency budget
and omitting it is how "sub-millisecond retrieval" claims get made about systems
that take 20ms to answer.

Kept deliberately boring: SQLite with an integer primary key. Not a bottleneck
worth optimising, but worth *measuring* so it cannot be quietly dropped.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

WORDS = (
    "memory recall context vector index query latency embedding token session "
    "compression retrieval summary neural engine cache bandwidth quantise graph "
    "consolidate rerank chunk pipeline throughput residency scheduler"
).split()


def build(path: Path | str, n: int, *, chars: int = 600, seed: int = 3) -> Path:
    """Create a store of `n` synthetic chunks, if it does not already exist."""
    path = Path(path)
    if path.exists():
        with sqlite3.connect(path) as c:
            have = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if have == n:
                return path
        path.unlink()

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")

        per_row = max(1, chars // 7)
        batch, BATCH = [], 20_000
        for i in range(n):
            idx = rng.integers(0, len(WORDS), size=per_row)
            batch.append((i, " ".join(WORDS[j] for j in idx)))
            if len(batch) >= BATCH:
                conn.executemany("INSERT INTO chunks VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            conn.executemany("INSERT INTO chunks VALUES (?, ?)", batch)
        conn.commit()

    return path


class Store:
    """Read-only handle. `check_same_thread=False` so the contention benchmark
    can hold one open across threads."""

    def __init__(self, path: Path | str):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA query_only=ON")

    def hydrate(self, ids: np.ndarray) -> list[str]:
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({marks})",
            [int(i) for i in ids],
        ).fetchall()
        by_id = dict(rows)
        return [by_id.get(int(i), "") for i in ids]

    def close(self) -> None:
        self.conn.close()
