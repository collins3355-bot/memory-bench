#!/usr/bin/env python
"""Convert both encoder layouts to CoreML and report where the ops actually land.

Produces `{variant}_s{seq}.mlpackage` for variant in {reference, ane} and each
sequence bucket. The pair is the point: identical weights and identical outputs,
differing only in tensor layout, so the Neural Engine residency difference is
attributable to layout alone.

Conversion choices that matter for ANE placement:

  * **fp16 precision.** The ANE is fp16-native; an fp32 model is commonly
    refused and placed on CPU instead.
  * **Fixed input shapes.** Dynamic sequence length reliably causes ANE
    eviction, which is why real deployments pad to a handful of buckets.
  * **int32 inputs.** CoreML has no int64 tensor type.

`--report-plan` prints the per-operation device split. Without it, any claim
about "ANE latency" is an assumption rather than a measurement.
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

SEQ_BUCKETS = (32, 64, 128, 256)
VARIANTS = ("reference", "ane")


def convert_one(variant: str, seq_len: int, out_dir: Path, model_name: str) -> Path:
    import coremltools as ct

    # seq_len is baked in so the traced graph has no shape queries at all.
    module = E.build(variant, model_name, seq_len=seq_len)
    ids = torch.randint(999, 20000, (1, seq_len), dtype=torch.int32)
    mask = torch.ones((1, seq_len), dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(module, (ids, mask), strict=False)

    out = out_dir / f"{variant}_s{seq_len}.mlpackage"
    if out.exists():
        shutil.rmtree(out)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=ids.shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=mask.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(str(out))
    return out


def report_plan(package: Path, unit: str = "CPU_AND_NE") -> dict:
    """Per-op device assignment for a converted package.

    Two things make this fiddlier than it looks:

    * `MLComputePlan` requires a *compiled* .mlmodelc. Handing it an .mlpackage
      aborts inside libc++ rather than raising -- which takes the whole process
      down, past any Python `except`. Hence the compile step here, and the
      subprocess isolation in the caller.
    * Op *count* is a poor residency measure: a reshape and a 1536-wide matmul
      each count once. The plan exposes a per-op cost estimate, so residency is
      reported cost-weighted as well. When the two disagree, the cost-weighted
      figure is the one that predicts latency.
    """
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan  # type: ignore
    from coremltools.models.utils import compile_model

    compiled = package if package.suffix == ".mlmodelc" else Path(compile_model(str(package)))

    plan = MLComputePlan.load_from_path(
        path=str(compiled), compute_units=getattr(ct.ComputeUnit, unit)
    )

    counts: dict[str, int] = {}
    cost: dict[str, float] = {}
    for func in plan.model_structure.program.functions.values():
        for op in func.block.operations:
            info = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = (
                type(info.preferred_compute_device).__name__
                .replace("ML", "").replace("ComputeDevice", "")
                if info else "Unassigned"
            )
            counts[dev] = counts.get(dev, 0) + 1
            try:
                est = plan.get_estimated_cost_for_mlprogram_operation(op)
                cost[dev] = cost.get(dev, 0.0) + float(getattr(est, "weight", 0.0) or 0.0)
            except Exception:
                pass

    total = sum(counts.values()) or 1
    total_cost = sum(cost.values()) or 0.0

    # "Unassigned" ops are constants -- weights and literals that are never
    # executed, and which the cost model duly scores at zero. Counting them in
    # the denominator drags every residency figure toward a floor that has
    # nothing to do with placement: the ANE variant looks like 43% ANE when 97%
    # of the work it actually performs is on the Neural Engine. Fractions are
    # therefore reported over *executing* ops, with the raw counts kept for
    # anyone who wants to check the arithmetic.
    executing = {k: v for k, v in counts.items() if k != "Unassigned"}
    n_exec = sum(executing.values()) or 1

    result = {
        "compiled_path": str(compiled),
        "op_counts": counts,
        "total_ops": total,
        "executing_ops": n_exec,
        "ane_fraction": round(executing.get("NeuralEngine", 0) / n_exec, 3),
        "cpu_fraction": round(executing.get("CPU", 0) / n_exec, 3),
        "gpu_fraction": round(executing.get("GPU", 0) / n_exec, 3),
    }
    if total_cost > 0:
        result["cost_weighted"] = {
            k: round(v / total_cost, 3) for k, v in sorted(cost.items())
        }
        result["ane_fraction_by_cost"] = round(
            cost.get("NeuralEngine", 0.0) / total_cost, 3
        )
    return result


def _plan_via_subprocess(package: Path) -> dict:
    """Run report_plan out-of-process so a libc++ abort cannot kill the batch."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, __file__, "--plan-only", str(package)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__PLAN__"):
            return json.loads(line[len("__PLAN__"):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {"error": tail[-1] if tail else f"exit {proc.returncode} with no output"}


def main() -> int:
    # Subprocess entry point for plan reporting; see _plan_via_subprocess.
    if len(sys.argv) >= 3 and sys.argv[1] == "--plan-only":
        try:
            print("__PLAN__" + json.dumps(report_plan(Path(sys.argv[2]))))
        except Exception as exc:
            print("__PLAN__" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("models"), type=Path)
    ap.add_argument("--model", default=E.DEFAULT_MODEL)
    ap.add_argument("--seq", type=int, nargs="*", default=list(SEQ_BUCKETS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--report-plan", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for variant in args.variants:
        for s in args.seq:
            key = f"{variant}_s{s}"
            print(f"[convert] {key} ...", flush=True)
            try:
                pkg = convert_one(variant, s, args.out, args.model)
                entry = {"package": str(pkg), "ok": True}
                if args.report_plan:
                    entry["plan"] = _plan_via_subprocess(pkg)
                    p = entry["plan"]
                    print(f"[convert] {key} -> ANE {p.get('ane_fraction')} by-count, "
                          f"{p.get('ane_fraction_by_cost')} by-cost {p.get('error','')}")
                else:
                    print(f"[convert] {key} -> {pkg}")
            except Exception as exc:
                entry = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[convert] {key} FAILED: {exc}", file=sys.stderr)
            summary[key] = entry

    (args.out / "conversion.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for v in summary.values() if v.get("ok"))
    print(f"\n[convert] {ok}/{len(summary)} packages converted")
    for k, v in summary.items():
        if v.get("ok") and "plan" in v:
            p = v["plan"]
            if "error" in p:
                print(f"  {k:18} plan unavailable: {p['error'][:70]}")
            else:
                print(f"  {k:18} ops={p.get('total_ops'):>5}  "
                      f"ANE={p.get('ane_fraction'):.2f}/{p.get('ane_fraction_by_cost', float('nan')):.2f} "
                      f"CPU={p.get('cpu_fraction'):.2f} GPU={p.get('gpu_fraction'):.2f}   (by-count/by-cost)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
