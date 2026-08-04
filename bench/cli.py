"""Benchmark driver.

Subcommands:
    search       retrieval backends across corpus scales, with recall
    embed        embedding backends and compute units, with ANE residency
    contention   foreground query latency while an indexer runs at each QoS
    pipeline     end-to-end tokenise -> embed -> search -> hydrate
    all          everything, written to results/

Every run records the machine fingerprint alongside the numbers, because a
latency figure without its hardware is not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import corpus as C
from . import qos as Q
from . import search as S
from . import sysinfo
from . import store as ST
from .timing import measure, timer_floor_ms

RESULTS = Path("results")
DEFAULT_SCALES = (60_000, 600_000, 6_000_000)


def _emit(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  -> {path}")
    return path


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ==========================================================================
# search
# ==========================================================================


def cmd_search(args) -> dict:
    out: dict[str, Any] = {"machine": sysinfo.collect(), "timer_floor_ms": timer_floor_ms(), "scales": {}}

    for n in args.scales:
        _rule(f"SEARCH  n={n:,}  d={args.dim}  k={args.k}")
        cor = C.generate(n, args.dim, out_dir=args.data)
        qs = C.queries(cor, count=args.queries)

        # A float64 query silently upcasts the whole float32 index on every
        # search -- 5x slower, with recall unchanged, so nothing in the results
        # would reveal it. Checked once here rather than per-search.
        assert qs.dtype == np.float32 and cor.vectors.dtype == np.float32, (
            f"dtype mismatch: queries={qs.dtype}, corpus={cor.vectors.dtype}"
        )

        print("  computing exact ground truth ...", flush=True)
        t0 = time.perf_counter()
        truth = C.exact_neighbors(cor, qs, k=args.k)
        print(f"  ground truth in {time.perf_counter() - t0:.1f}s")

        X = np.asarray(cor.vectors)  # materialise once, shared by all backends

        backends: list[S.Backend] = [
            S.FlatFP32(),
            S.FlatFP32Threaded(threads=args.threads),
            S.BinaryRerank(center=False, rerank_depth=args.rerank),
            S.BinaryRerank(center=True, rerank_depth=args.rerank),
        ]
        # fp16 and int8 have no BLAS path in numpy and run roughly 15x slower
        # than fp32. They are documented traps rather than candidates, so they
        # are opt-in: at n=6M a single fp16 query takes seconds, and the recall
        # pass alone would run for a quarter of an hour.
        if args.slow_backends:
            backends.insert(2, S.FlatFP16())
            backends.append(S.Int8Rerank(rerank_depth=args.rerank))
        if n <= args.hnsw_max:
            backends.append(S.HNSW(connectivity=args.hnsw_m, ef=args.hnsw_ef))
        else:
            print(f"  [skip] HNSW above n={args.hnsw_max:,} (build time); raise --hnsw-max to include")

        rows = []
        for b in backends:
            build = b.build(X)
            got = np.stack([b.search(qs[i], args.k) for i in range(qs.shape[0])])
            recall = C.recall_at_k(got, truth)

            counter = {"i": 0}

            def one() -> None:
                b.search(qs[counter["i"] % qs.shape[0]], args.k)
                counter["i"] += 1

            t = measure(b.name, one, iterations=args.iters, warmup=20, min_seconds=0.75)
            print(f"  {t}  recall@{args.k}={recall:.4f}  build={build.seconds:6.1f}s  idx={build.index_bytes / 1e6:8.1f}MB")

            rows.append({
                **t.as_dict(),
                "recall_at_k": recall,
                "build_seconds": build.seconds,
                "index_bytes": build.index_bytes,
                "build_notes": build.notes,
            })
            del b

        out["scales"][str(n)] = {"corpus": cor.meta, "backends": rows}

    return out


# ==========================================================================
# embed
# ==========================================================================


def cmd_embed(args) -> dict:
    from .embed import CoreMLBackend, TorchBackend, Tokenizer

    out: dict[str, Any] = {"machine": sysinfo.collect(), "backends": [], "residency": {}}
    tok = Tokenizer()

    text = (
        "The user asked about vector search latency on Apple Silicon and whether "
        "the Neural Engine is faster than the GPU for small embedding models "
        "running continuously in the background of a laptop."
    )

    for seq in args.seq:
        ids, mask, n_tok = tok.encode(text, seq_len=seq)
        _rule(f"EMBED  seq_len={seq}  (real tokens={n_tok})")

        cands: list[Any] = []
        for dev in ("cpu", "mps"):
            try:
                cands.append(TorchBackend(device=dev))
            except Exception as exc:
                print(f"  [skip] torch-{dev}: {exc}")

        for variant in args.variants:
            pkg = Path(args.models) / f"{variant}_s{seq}.mlpackage"
            if not pkg.exists():
                print(f"  [skip] {pkg.name} not found -- run scripts/convert_coreml.py")
                continue
            for unit in args.units:
                try:
                    b = CoreMLBackend(pkg, unit=unit)
                    b.name = f"coreml-{variant}-{unit}"
                    cands.append(b)
                except Exception as exc:
                    print(f"  [skip] coreml-{variant}-{unit}: {exc}")

        for b in cands:
            try:
                vec = b.encode(ids, mask)
                t = measure(b.name, lambda: b.encode(ids, mask), iterations=args.iters, warmup=15, min_seconds=0.75)
                print(f"  {t}  |v|={float(np.linalg.norm(vec)):.4f}")
                row = {**t.as_dict(), "seq_len": seq, "norm": float(np.linalg.norm(vec))}

                if isinstance(b, CoreMLBackend):
                    res = b.residency()
                    row["residency"] = res
                    out["residency"][f"{b.name}_s{seq}"] = res
                    if "ane_fraction" in res:
                        print(f"      residency: ANE={res['ane_fraction']:.0%} CPU={res['cpu_fraction']:.0%} GPU={res['gpu_fraction']:.0%}")

                out["backends"].append(row)
            except Exception as exc:
                print(f"  [fail] {b.name}: {type(exc).__name__}: {exc}")
            del b

    return out


# ==========================================================================
# contention
# ==========================================================================


def cmd_contention(args) -> dict:
    n = args.scales[0]
    _rule(f"CONTENTION  n={n:,}  -- foreground query latency while indexing")

    cor = C.generate(n, args.dim, out_dir=args.data)
    qs = C.queries(cor, count=128)
    backend = S.FlatFP32()
    backend.build(np.asarray(cor.vectors))
    fn = lambda q: backend.search(q, args.k)  # noqa: E731

    conditions = [
        ("idle (no indexer)", None),
        ("indexer @ user-initiated", "user-initiated"),
        ("indexer @ default", "default"),
        ("indexer @ utility", "utility"),
        ("indexer @ background", "background"),
    ]

    rows = []
    baseline = None
    for label, q in conditions:
        r = Q.measure_contention(fn, qs, condition=label, indexer_qos=q, iterations=args.iters)
        if baseline is None:
            baseline = r.query_p50_ms
        slow = r.query_p50_ms / baseline if baseline else float("nan")
        print(f"  {label:28} p50={r.query_p50_ms:7.3f}ms  p99={r.query_p99_ms:7.3f}ms  "
              f"({slow:4.2f}x baseline)  indexer_ops={r.indexer_ops:>6}  qos={r.indexer_qos}")
        rows.append({**r.__dict__, "slowdown_vs_idle": slow})

    return {"machine": sysinfo.collect(), "n": n, "conditions": rows}


# ==========================================================================
# pipeline
# ==========================================================================


def cmd_pipeline(args) -> dict:
    from .embed import CoreMLBackend, TorchBackend, Tokenizer

    n = args.scales[0]
    _rule(f"PIPELINE  n={n:,}  -- full budget, tokenise to text")

    cor = C.generate(n, args.dim, out_dir=args.data)
    # The threaded flat scan, not binary quantisation. Binary codes are 32x
    # smaller but measured *slower* than the threaded flat scan at every scale
    # here (16.1ms vs 12.3ms at 600k; 147ms vs 112ms at 6M) while giving up
    # exactness: the coarse scan is single-threaded, and the rerank still has to
    # gather rows from the full fp32 array, so the small codes never become the
    # working set. Flat wins on latency and returns exact results.
    backend = S.FlatFP32Threaded(threads=args.threads)
    backend.build(np.asarray(cor.vectors))

    db = ST.build(Path(args.data) / f"chunks_{n}.sqlite", n)
    store = ST.Store(db)
    tok = Tokenizer()

    text = "What did we decide about the retrieval latency budget last week?"
    seq = args.seq[0]

    embedder = None
    for variant in args.variants:
        pkg = Path(args.models) / f"{variant}_s{seq}.mlpackage"
        if pkg.exists():
            try:
                embedder = CoreMLBackend(pkg, unit="ane")
                embedder.name = f"coreml-{variant}-ane"
                break
            except Exception:
                pass
    if embedder is None:
        embedder = TorchBackend(device="cpu")
        print(f"  [note] no CoreML package found, falling back to {embedder.name}")

    print(f"  embedder: {embedder.name}")

    stages: dict[str, Any] = {}
    ids, mask, _ = tok.encode(text, seq_len=seq)
    vec = embedder.encode(ids, mask)
    hits = backend.search(vec, args.k)

    stages["1_tokenise"] = measure("tokenise", lambda: tok.encode(text, seq_len=seq), iterations=args.iters, warmup=20, min_seconds=0.5)
    stages["2_embed"] = measure("embed", lambda: embedder.encode(ids, mask), iterations=args.iters, warmup=15, min_seconds=0.5)
    stages["3_search"] = measure("search", lambda: backend.search(vec, args.k), iterations=args.iters, warmup=20, min_seconds=0.5)
    stages["4_hydrate"] = measure("hydrate (sqlite)", lambda: store.hydrate(hits), iterations=args.iters, warmup=20, min_seconds=0.5)

    def full() -> None:
        i, m, _ = tok.encode(text, seq_len=seq)
        v = embedder.encode(i, m)
        h = backend.search(v, args.k)
        store.hydrate(h)

    stages["5_end_to_end"] = measure("END TO END", full, iterations=args.iters, warmup=15, min_seconds=1.0)

    print()
    total_p50 = stages["5_end_to_end"].p50
    for key in sorted(stages):
        t = stages[key]
        share = t.p50 / total_p50 * 100 if total_p50 else float("nan")
        print(f"  {t}   {share:5.1f}% of e2e")

    budget = 50.0
    print(f"\n  Budget {budget:.0f}ms -> end-to-end p50={total_p50:.2f}ms, p99={stages['5_end_to_end'].p99:.2f}ms "
          f"({'PASS' if stages['5_end_to_end'].p99 < budget else 'FAIL'} at p99)")

    store.close()
    return {
        "machine": sysinfo.collect(),
        "n": n,
        "embedder": embedder.name,
        "seq_len": seq,
        "budget_ms": budget,
        "stages": {k: v.as_dict() for k, v in stages.items()},
    }


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(prog="memory-bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "embed", "contention", "pipeline", "all"])
    ap.add_argument("--scales", type=int, nargs="*", default=list(DEFAULT_SCALES))
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=256)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rerank", type=int, default=1000)
    ap.add_argument("--seq", type=int, nargs="*", default=[64])
    ap.add_argument("--variants", nargs="*", default=["reference", "ane"])
    ap.add_argument("--units", nargs="*", default=["cpu", "gpu", "ane"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--threads", type=int, default=8, help="workers for the threaded flat scan (M1 Max has 8 P-cores)")
    ap.add_argument("--slow-backends", action="store_true", help="include fp16 and int8, which have no numpy BLAS path (see search.py)")
    ap.add_argument("--hnsw-max", type=int, default=600_000, help="skip HNSW above this n")
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--hnsw-ef", type=int, default=64)
    args = ap.parse_args()

    print(json.dumps(sysinfo.collect(), indent=2))

    cmds = {"search": cmd_search, "embed": cmd_embed, "contention": cmd_contention, "pipeline": cmd_pipeline}
    todo = list(cmds) if args.command == "all" else [args.command]

    for name in todo:
        try:
            _emit(name, cmds[name](args))
        except Exception as exc:
            import traceback

            print(f"\n[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
