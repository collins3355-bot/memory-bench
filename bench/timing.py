"""Measurement primitives.

Everything here exists to stop the benchmark from flattering itself. The rules:

  * Report percentiles, never a bare mean. Retrieval latency is a tail problem --
    a p50 of 3ms with a p99 of 180ms is a bad system, and a mean hides that.
  * Discard warmup iterations. First-call cost includes lazy weight loading,
    Metal shader compilation, and CoreML model specialization, none of which
    recur in steady state.
  * Report the timer floor so nobody quotes a number that is actually noise.
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Any


@dataclass
class Timing:
    """Latency distribution for one measured operation, in milliseconds."""

    label: str
    n: int
    p50: float
    p90: float
    p99: float
    minimum: float
    mean: float
    stdev: float
    total_s: float
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.label:<34} p50={self.p50:8.3f}ms  p90={self.p90:8.3f}ms  "
            f"p99={self.p99:8.3f}ms  min={self.minimum:8.3f}ms  n={self.n}"
        )


def timer_floor_ms(samples: int = 2000) -> float:
    """Empirical resolution of the clock, so we know what 'too fast to measure' is."""
    deltas = []
    for _ in range(samples):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        if b > a:
            deltas.append(b - a)
    return (min(deltas) if deltas else 0) / 1e6


def measure(
    label: str,
    fn: Callable[[], Any],
    *,
    iterations: int = 100,
    warmup: int = 10,
    min_seconds: float = 0.0,
    disable_gc: bool = True,
    notes: dict[str, Any] | None = None,
) -> Timing:
    """Run `fn` repeatedly and summarise the latency distribution.

    `min_seconds` keeps sampling past `iterations` until that much wall time has
    elapsed. Useful for operations near the timer floor, where 100 samples of a
    0.2ms call is mostly measuring the loop.
    """
    for _ in range(warmup):
        fn()

    was_enabled = gc.isenabled()
    if disable_gc:
        gc.collect()
        gc.disable()

    durations_ns: list[int] = []
    started = time.perf_counter()
    try:
        i = 0
        while i < iterations or (time.perf_counter() - started) < min_seconds:
            t0 = time.perf_counter_ns()
            fn()
            durations_ns.append(time.perf_counter_ns() - t0)
            i += 1
            # Guard against a pathologically slow op spinning here forever.
            if i > iterations and (time.perf_counter() - started) > max(min_seconds, 30.0):
                break
    finally:
        if disable_gc and was_enabled:
            gc.enable()

    total_s = time.perf_counter() - started
    ms = sorted(d / 1e6 for d in durations_ns)

    def pct(p: float) -> float:
        if not ms:
            return float("nan")
        idx = min(len(ms) - 1, max(0, int(round((p / 100.0) * (len(ms) - 1)))))
        return ms[idx]

    return Timing(
        label=label,
        n=len(ms),
        p50=pct(50),
        p90=pct(90),
        p99=pct(99),
        minimum=ms[0] if ms else float("nan"),
        mean=statistics.fmean(ms) if ms else float("nan"),
        stdev=statistics.pstdev(ms) if len(ms) > 1 else 0.0,
        total_s=total_s,
        notes=notes or {},
    )
