#!/usr/bin/env python
"""Convert the cross-encoder reranker to CoreML at its real workload shape.

The embedder conversions are batch-1: one query, one vector. The reranker's
workload is different in kind — it scores a *batch* of (query, passage) pairs
per retrieval, so the shape that matters is (batch=50, seq=128). Batch-50
tensors are ~50x larger per op than anything the embedder conversions put on
the Neural Engine, which makes ANE residency a genuinely open question here
rather than a foregone conclusion: the per-head attention splits that keep
batch-1 working sets resident may not survive a 50-row batch.

Everything else follows the embedder playbook: fp16, fixed shapes, int32
inputs, aten::Int-free graphs, per-op compute plan reported from a compiled
.mlmodelc under subprocess isolation (a bad package aborts in libc++, past any
Python except).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from bench import encoder as E  # noqa: E402
from convert_coreml import report_plan, _plan_via_subprocess  # noqa: E402

VARIANTS = ("reference", "ane")


def convert_one(variant: str, batch: int, seq_len: int, out_dir: Path, model_name: str) -> Path:
    import coremltools as ct

    module = E.build_ce(variant, model_name, seq_len=seq_len)
    ids = torch.randint(999, 20000, (batch, seq_len), dtype=torch.int32)
    mask = torch.ones((batch, seq_len), dtype=torch.int32)
    tt = torch.zeros((batch, seq_len), dtype=torch.int32)
    tt[:, seq_len // 2 :] = 1  # plausible segment split for tracing

    with torch.no_grad():
        traced = torch.jit.trace(module, (ids, mask, tt), strict=False)

    out = out_dir / f"ce_{variant}_b{batch}_s{seq_len}.mlpackage"
    if out.exists():
        shutil.rmtree(out)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=ids.shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=mask.shape, dtype=np.int32),
            ct.TensorType(name="token_type_ids", shape=tt.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(str(out))
    return out


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--plan-only":
        try:
            print("__PLAN__" + json.dumps(report_plan(Path(sys.argv[2]))))
        except Exception as exc:
            print("__PLAN__" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("models"), type=Path)
    ap.add_argument("--model", default=E.DEFAULT_CE_MODEL)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seq", type=int, nargs="*", default=[128])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for variant in args.variants:
        for s in args.seq:
            key = f"ce_{variant}_b{args.batch}_s{s}"
            print(f"[convert] {key} ...", flush=True)
            try:
                pkg = convert_one(variant, args.batch, s, args.out, args.model)
                entry = {"package": str(pkg), "ok": True, "plan": _plan_via_subprocess(pkg)}
                p = entry["plan"]
                print(f"[convert] {key} -> ANE {p.get('ane_fraction')} by-count, "
                      f"{p.get('ane_fraction_by_cost')} by-cost {p.get('error','')}")
            except Exception as exc:
                entry = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[convert] {key} FAILED: {exc}", file=sys.stderr)
            summary[key] = entry

    report = args.out / "ce_conversion.json"
    report.write_text(json.dumps(summary, indent=2))
    print(f"-> {report}")
    return 0 if any(v.get("ok") for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
