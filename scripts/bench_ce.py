#!/usr/bin/env python
"""Latency of the rerank stage: one batch of 50 (query, passage) pairs.

This is the number that decides whether the validated retrieval architecture
(wide net -> cross-encoder rerank) fits the 50ms budget on-device. The eval
already measured the *quality* of the reranked ranking; this measures its
*price* across execution paths, same protocol as the embedder benchmarks:
warmup discarded, percentiles not means, identical inputs everywhere.

Inputs are one real query paired with 50 distinct passages, tokenised once and
padded to the fixed shape the CoreML packages were converted at. The torch
paths run the same fixed shape so the comparison is execution-path-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from bench import encoder as E  # noqa: E402
from bench import sysinfo  # noqa: E402
from bench.timing import measure  # noqa: E402


def make_inputs(batch: int, seq_len: int):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E.DEFAULT_CE_MODEL)
    query = "What was the first issue I had with my new car after its first service?"
    passages = [
        f"user: I took the car in for service and afterwards the {w} started acting up, "
        "which was annoying because it had just been checked."
        for w in ("GPS", "radio", "bluetooth", "heater", "sunroof")
    ] * 10
    enc = tok(
        [query] * batch,
        passages[:batch],
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return (
        enc["input_ids"].astype(np.int32),
        enc["attention_mask"].astype(np.int32),
        enc["token_type_ids"].astype(np.int32),
    )


def bench_torch(device: str, ids, mask, tt, iters: int) -> dict | None:
    import torch

    try:
        m = E.build_ce("reference").to(device).eval()
    except Exception as exc:
        print(f"  [skip] torch-{device}: {exc}")
        return None

    t_ids = torch.from_numpy(ids.astype(np.int64)).to(device)
    t_mask = torch.from_numpy(mask.astype(np.int64)).to(device)
    t_tt = torch.from_numpy(tt.astype(np.int64)).to(device)

    def run():
        with torch.no_grad():
            out = m(t_ids, t_mask, t_tt)
        if device == "mps":
            torch.mps.synchronize()
        return out

    t = measure(f"torch-{device}", run, iterations=iters, warmup=8, min_seconds=1.0)
    print(f"  {t}")
    return t.as_dict()


def bench_coreml(pkg: Path, unit: str, ids, mask, tt, iters: int) -> dict | None:
    import coremltools as ct

    units = {"cpu": "CPU_ONLY", "gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE"}
    try:
        m = ct.models.MLModel(str(pkg), compute_units=getattr(ct.ComputeUnit, units[unit]))
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": tt}
        m.predict(feed)  # surface load/compile failures before timing
    except Exception as exc:
        print(f"  [skip] {pkg.stem}-{unit}: {str(exc)[:90]}")
        return None

    label = f"coreml-{pkg.stem.replace('ce_', '').split('_b')[0]}-{unit}"
    t = measure(label, lambda: m.predict(feed), iterations=iters, warmup=8, min_seconds=1.0)
    print(f"  {t}")
    return t.as_dict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--models", default=Path("models"), type=Path)
    ap.add_argument("--out", default=Path("results/rerank_perf.json"), type=Path)
    args = ap.parse_args()

    print(f"CE rerank batch: {args.batch} pairs x seq {args.seq}")
    ids, mask, tt = make_inputs(args.batch, args.seq)

    rows = []
    for dev in ("cpu", "mps"):
        r = bench_torch(dev, ids, mask, tt, args.iters)
        if r:
            rows.append(r)
    for variant in ("reference", "ane"):
        pkg = args.models / f"ce_{variant}_b{args.batch}_s{args.seq}.mlpackage"
        if not pkg.exists():
            print(f"  [skip] {pkg.name} not found -- run scripts/convert_ce.py")
            continue
        for unit in ("cpu", "gpu", "ane"):
            r = bench_coreml(pkg, unit, ids, mask, tt, args.iters)
            if r:
                rows.append(r)

    plan_file = args.models / "ce_conversion.json"
    plans = json.loads(plan_file.read_text()) if plan_file.exists() else {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "batch": args.batch,
                "seq_len": args.seq,
                "machine": sysinfo.collect(),
                "backends": rows,
                "conversion_plans": plans,
            },
            indent=2,
        )
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
