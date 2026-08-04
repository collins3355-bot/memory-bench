"""Retrieval backends, from "no index at all" upward.

The thesis under test: at the corpus sizes a personal memory system actually
reaches (10^5-10^6 vectors), a flat SIMD scan is fast enough, and a graph index
costs build time and memory it never earns back.

Each backend implements the same three things -- build, search, footprint -- so
the comparison is apples to apples, including the parts people leave out of
benchmarks (index construction time, resident bytes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class BuildStats:
    seconds: float
    index_bytes: int
    notes: dict


class Backend(Protocol):
    name: str

    def build(self, X: np.ndarray) -> BuildStats: ...
    def search(self, q: np.ndarray, k: int) -> np.ndarray: ...


# --------------------------------------------------------------------------
# Flat scans
# --------------------------------------------------------------------------


class FlatFP32:
    """Exact cosine by one BLAS GEMV per query.

    On unit-normalised vectors cosine is a plain dot product, so this is a
    single `X @ q`. It is memory-bandwidth bound, not compute bound: the whole
    matrix streams through the cores once per query. Predicted latency is simply
    `n * d * 4 bytes / achievable-bandwidth`, and on an M1 Max that is around
    200-350 GB/s in practice against a ~400 GB/s peak.
    """

    name = "flat-fp32"

    def __init__(self) -> None:
        self.X: np.ndarray | None = None

    def build(self, X: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        # Force residency: a memmap that has never been touched measures page
        # faults, not arithmetic. Cold-cache behaviour is measured separately.
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        return BuildStats(
            seconds=time.perf_counter() - t0,
            index_bytes=self.X.nbytes,
            notes={"structure": "none"},
        )

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        scores = self.X @ q
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]


class FlatFP32Threaded:
    """The same exact scan, split across worker threads.

    A single-query flat scan is one BLAS GEMV, and Accelerate does not
    parallelise GEMV -- it runs on one core. That caps throughput at *single-core*
    memory bandwidth (~35-50 GB/s on an M1 Max) rather than the ~400 GB/s the
    chip is quoted at, which is why the naive flat scan comes in several times
    slower than a back-of-envelope bandwidth estimate predicts.

    Splitting the matrix into row blocks and scanning them concurrently fixes
    this. numpy releases the GIL inside BLAS, so Python threads genuinely run in
    parallel here. The thread pool is created once at build time: spawning
    threads per query would cost more than the scan.
    """

    def __init__(self, threads: int = 8) -> None:
        self.threads = threads
        self.name = f"flat-fp32-t{threads}"
        self.X: np.ndarray | None = None
        self._blocks: list[tuple[int, np.ndarray]] = []
        self._pool = None

    def build(self, X: np.ndarray) -> BuildStats:
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.perf_counter()
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        n = self.X.shape[0]
        step = (n + self.threads - 1) // self.threads
        # Views, not copies -- the blocks share the parent buffer.
        self._blocks = [
            (s, self.X[s : min(s + step, n)]) for s in range(0, n, step)
        ]
        self._pool = ThreadPoolExecutor(max_workers=self.threads)
        return BuildStats(
            seconds=time.perf_counter() - t0,
            index_bytes=self.X.nbytes,
            notes={"structure": "none", "threads": self.threads, "blocks": len(self._blocks)},
        )

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        def part(item: tuple[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            offset, block = item
            scores = block @ q
            kk = min(k, scores.shape[0])
            loc = np.argpartition(-scores, kk - 1)[:kk]
            return scores[loc], loc + offset

        results = list(self._pool.map(part, self._blocks))
        scores = np.concatenate([r[0] for r in results])
        idx = np.concatenate([r[1] for r in results])
        top = np.argpartition(-scores, k - 1)[:k]
        return idx[top[np.argsort(-scores[top])]]


class FlatFP16:
    """Half-precision scan: half the bytes, so ideally half the time.

    Whether it delivers depends on the BLAS. If fp16 is upcast to fp32 before
    the GEMV, the bandwidth saving on the load is real but the arithmetic saving
    is not, and the result lands between the two.
    """

    name = "flat-fp16"

    def __init__(self) -> None:
        self.X: np.ndarray | None = None

    def build(self, X: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        self.X = np.ascontiguousarray(X, dtype=np.float16)
        return BuildStats(
            seconds=time.perf_counter() - t0,
            index_bytes=self.X.nbytes,
            notes={"structure": "none"},
        )

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        scores = (self.X @ q.astype(np.float16)).astype(np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]


# --------------------------------------------------------------------------
# Binary quantisation
# --------------------------------------------------------------------------


class BinaryRerank:
    """1-bit codes for the scan, full precision for the rerank.

    32x smaller than fp32, and Hamming distance is XOR plus popcount, which is
    both cheap and bandwidth-friendly. The scan returns `rerank_depth`
    candidates and those are rescored exactly, so final ranking quality depends
    only on whether the true top-k survived the coarse pass.

    `center` is the whole experiment. Embedding spaces are cones: most vectors
    share a dominant direction, so most of them agree on the sign of most
    coordinates, and uncentered sign codes end up nearly identical. Subtracting
    the corpus mean first restores per-coordinate balance. Same bit budget, same
    speed, materially different recall -- and it is one line of code.
    """

    def __init__(self, *, center: bool = True, rerank_depth: int = 1000) -> None:
        self.center = center
        self.rerank_depth = rerank_depth
        self.name = f"binary{'-centered' if center else '-raw'}-rr{rerank_depth}"
        self.X: np.ndarray | None = None
        self.codes: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self._scratch: np.ndarray | None = None

    def build(self, X: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.mean = (
            self.X.mean(axis=0)
            if self.center
            else np.zeros(self.X.shape[1], dtype=np.float32)
        )
        self.codes = np.packbits(self.X > self.mean, axis=1)
        self._scratch = np.empty_like(self.codes)
        return BuildStats(
            seconds=time.perf_counter() - t0,
            # Honest accounting: the fp32 copy must stay resident for reranking.
            # The bit codes alone are not a working system.
            index_bytes=self.codes.nbytes + self.X.nbytes,
            notes={
                "code_bytes": int(self.codes.nbytes),
                "fp32_bytes_for_rerank": int(self.X.nbytes),
                "centered": self.center,
                "rerank_depth": self.rerank_depth,
            },
        )

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        qcode = np.packbits(q > self.mean)
        np.bitwise_xor(self.codes, qcode, out=self._scratch)
        # uint8 popcounts must accumulate in a wider type: d=384 exceeds 255.
        dist = np.bitwise_count(self._scratch).sum(axis=1, dtype=np.int32)

        depth = min(self.rerank_depth, dist.shape[0])
        cand = np.argpartition(dist, depth - 1)[:depth]

        exact = self.X[cand] @ q
        take = min(k, depth)
        top = np.argpartition(-exact, take - 1)[:take]
        return cand[top[np.argsort(-exact[top])]]


class Int8Rerank:
    """Symmetric int8 scalar quantisation, 4x smaller than fp32.

    Included to document a trap rather than to win. The bandwidth argument says
    this should beat fp32, but numpy has no integer GEMV -- there is no BLAS
    path for int8 -- so it falls back to a generic loop and lands far slower
    than the fp32 baseline. Capturing the win needs a native kernel over ARM
    SDOT/SMMLA intrinsics. The number this backend prints is a measurement of
    numpy's dispatch, not of what int8 can do.
    """

    name = "int8-rerank"

    def __init__(self, rerank_depth: int = 1000) -> None:
        self.rerank_depth = rerank_depth
        self.X: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.scale: float = 1.0

    def build(self, X: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.scale = float(np.abs(self.X).max()) / 127.0
        self.Q = np.round(self.X / self.scale).astype(np.int8)
        return BuildStats(
            seconds=time.perf_counter() - t0,
            index_bytes=self.Q.nbytes + self.X.nbytes,
            notes={"code_bytes": int(self.Q.nbytes), "scale": self.scale},
        )

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        qq = np.round(q / self.scale).astype(np.int8)
        # int32 accumulation, not int16. A 384-dim dot product of two int8
        # vectors reaches 384 * 127 * 127 ~= 6.2e6, and int16 saturates at
        # 32767 -- so an int16 accumulator clips essentially every score and
        # correlates *negatively* (-0.32) with the true ranking. It does not
        # look broken from the outside: the search returns plausible ids at
        # plausible speed, and recall is quietly 0.02.
        scores = self.Q.astype(np.int32) @ qq.astype(np.int32)
        depth = min(self.rerank_depth, scores.shape[0])
        cand = np.argpartition(-scores, depth - 1)[:depth]
        exact = self.X[cand] @ q
        take = min(k, depth)
        top = np.argpartition(-exact, take - 1)[:take]
        return cand[top[np.argsort(-exact[top])]]


# --------------------------------------------------------------------------
# Graph index
# --------------------------------------------------------------------------


class HNSW:
    """usearch HNSW -- the "proper" answer, measured including build cost.

    Build time is the number that decides this for a consumer app. Query latency
    is only half the story: if first-run indexing pins eight cores for twenty
    minutes, the laptop gets hot and loud and the user notices, which is exactly
    the failure mode milestone 3 is meant to avoid.
    """

    def __init__(self, *, connectivity: int = 16, ef_construct: int = 128, ef: int = 64) -> None:
        self.connectivity = connectivity
        self.ef_construct = ef_construct
        self.ef = ef
        self.name = f"hnsw-m{connectivity}-ef{ef}"
        self.index = None

    def build(self, X: np.ndarray) -> BuildStats:
        from usearch.index import Index

        t0 = time.perf_counter()
        self.index = Index(
            ndim=int(X.shape[1]),
            metric="ip",  # inner product == cosine on normalised vectors
            dtype="f32",
            connectivity=self.connectivity,
            expansion_add=self.ef_construct,
            expansion_search=self.ef,
        )
        Xc = np.ascontiguousarray(X, dtype=np.float32)
        self.index.add(np.arange(Xc.shape[0], dtype=np.int64), Xc)
        return BuildStats(
            seconds=time.perf_counter() - t0,
            index_bytes=int(self.index.memory_usage),
            notes={
                "connectivity": self.connectivity,
                "ef_construct": self.ef_construct,
                "ef_search": self.ef,
            },
        )

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        m = self.index.search(q, k)
        return np.asarray(m.keys, dtype=np.int64)
