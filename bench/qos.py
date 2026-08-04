"""Does background indexing make the foreground feel slow?

Milestone 3 is usually framed as a hardware-damage problem -- "don't burn out
the GPU". That framing is wrong twice over. GPUs thermal-throttle rather than
burn out, and on Apple Silicon the thing that makes a laptop unpleasant is not
damage but *contention*: fans audible, battery draining, and the user's own work
stuttering because a background job took the performance cores.

macOS already solves this, and not with SIMD or low-level tuning. Quality of
Service is a scheduling hint: a thread marked QOS_CLASS_BACKGROUND is confined
to the efficiency cores, which on an M1 Max draw a fraction of a watt against
several for the performance cluster. It costs one ctypes call.

So the metric here is not indexer throughput. Throughput at background QoS is
*supposed* to be worse -- that is the trade being made. The metric that matters
is foreground query latency measured while the indexer runs, because that is
what the user actually perceives.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
import time
from dataclasses import dataclass

import numpy as np

# From <sys/qos.h>. Values are ABI-stable.
QOS_CLASS_USER_INTERACTIVE = 0x21
QOS_CLASS_USER_INITIATED = 0x19
QOS_CLASS_DEFAULT = 0x15
QOS_CLASS_UTILITY = 0x11
QOS_CLASS_BACKGROUND = 0x09

QOS_NAMES = {
    "user-interactive": QOS_CLASS_USER_INTERACTIVE,
    "user-initiated": QOS_CLASS_USER_INITIATED,
    "default": QOS_CLASS_DEFAULT,
    "utility": QOS_CLASS_UTILITY,
    "background": QOS_CLASS_BACKGROUND,
}


def _libc():
    path = ctypes.util.find_library("c")
    return ctypes.CDLL(path, use_errno=True)


def set_thread_qos(qos: int, relative_priority: int = 0) -> bool:
    """Apply a QoS class to the *calling* thread. Returns success."""
    try:
        libc = _libc()
        fn = libc.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        return fn(ctypes.c_uint(qos), ctypes.c_int(relative_priority)) == 0
    except Exception:
        return False


def current_thread_qos() -> int | None:
    """Read back the calling thread's QoS class, to confirm it took effect."""
    try:
        libc = _libc()
        fn = libc.pthread_get_qos_class_np
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        fn.restype = ctypes.c_int
        self_fn = libc.pthread_self
        self_fn.restype = ctypes.c_void_p

        qos = ctypes.c_uint(0)
        rel = ctypes.c_int(0)
        if fn(self_fn(), ctypes.byref(qos), ctypes.byref(rel)) == 0:
            return int(qos.value)
    except Exception:
        pass
    return None


@dataclass
class ContentionResult:
    condition: str
    query_p50_ms: float
    query_p99_ms: float
    indexer_ops: int
    indexer_qos: str | None


class _Indexer:
    """A stand-in for real index maintenance: dense linear algebra that releases
    the GIL, so the contention measured is genuine CPU contention rather than an
    artefact of Python's interpreter lock."""

    def __init__(self, qos: str | None, work_dim: int = 512):
        self.qos = qos
        self.work_dim = work_dim
        self.ops = 0
        self._stop = threading.Event()
        self.applied_qos: int | None = None
        self.thread: threading.Thread | None = None

    def _run(self) -> None:
        if self.qos is not None:
            set_thread_qos(QOS_NAMES[self.qos])
            self.applied_qos = current_thread_qos()

        rng = np.random.default_rng(0)
        a = rng.standard_normal((self.work_dim, self.work_dim)).astype(np.float32)
        b = rng.standard_normal((self.work_dim, self.work_dim)).astype(np.float32)
        while not self._stop.is_set():
            a @ b
            self.ops += 1

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.25)  # let it reach steady state before measuring

    def stop(self) -> None:
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=5)


def measure_contention(
    search_fn,
    queries: np.ndarray,
    *,
    condition: str,
    indexer_qos: str | None,
    iterations: int = 200,
    warmup: int = 20,
) -> ContentionResult:
    """Time foreground queries with an indexer running at `indexer_qos`.

    `indexer_qos=None` means no indexer at all -- the idle baseline.
    """
    indexer = _Indexer(indexer_qos) if indexer_qos is not None else None
    if indexer:
        indexer.start()

    try:
        nq = queries.shape[0]
        for i in range(warmup):
            search_fn(queries[i % nq])

        samples: list[float] = []
        for i in range(iterations):
            q = queries[i % nq]
            t0 = time.perf_counter_ns()
            search_fn(q)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        if indexer:
            indexer.stop()

    samples.sort()

    def pct(p: float) -> float:
        idx = min(len(samples) - 1, max(0, int(round(p / 100 * (len(samples) - 1)))))
        return samples[idx]

    applied = None
    if indexer and indexer.applied_qos is not None:
        applied = next(
            (k for k, v in QOS_NAMES.items() if v == indexer.applied_qos),
            hex(indexer.applied_qos),
        )

    return ContentionResult(
        condition=condition,
        query_p50_ms=pct(50),
        query_p99_ms=pct(99),
        indexer_ops=indexer.ops if indexer else 0,
        indexer_qos=applied,
    )
