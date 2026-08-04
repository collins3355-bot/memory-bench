"""Embedding backends: PyTorch CPU/MPS vs CoreML CPU/GPU/ANE.

This is the contended half of the retrieval budget. A flat scan over a personal
corpus costs single-digit milliseconds; encoding the query costs more than
everything else combined, so it is the only part of the pipeline where the
choice of compute unit changes the answer.

Every backend takes tokenised input and returns a normalised float32 embedding,
so differences are attributable to the execution path rather than to pre- or
post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EmbedResult:
    vector: np.ndarray
    backend: str


class TorchBackend:
    """PyTorch eager execution on CPU or MPS.

    MPS is included to make a point about milestone 3: the GPU is not the goal.
    It will likely win on batch throughput and lose on single-query latency,
    because a 22M-parameter model at batch 1 cannot fill a 32-core GPU and the
    kernel launch overhead dominates. Meanwhile it spins up the GPU -- power,
    heat, fan, and contention with whatever the user is actually doing.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str | None = None,
        threads: int | None = None,
        variant: str = "reference",
    ):
        import torch

        from . import encoder as E

        self.device = device
        self.name = f"torch-{device}"
        if threads is not None:
            torch.set_num_threads(threads)
            self.name = f"torch-{device}-t{threads}"

        self.torch = torch
        # Built from bench.encoder, the same module the CoreML packages are
        # converted from -- not the HuggingFace model. Benchmarking PyTorch on
        # the stock HF graph against CoreML on a reimplementation would compare
        # two different graphs and attribute the difference to the runtime.
        self.model = E.build(variant, model_name or E.DEFAULT_MODEL).to(device)
        self.model.eval()

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        t = self.torch
        ids = t.from_numpy(input_ids.astype(np.int64)).to(self.device)
        mask = t.from_numpy(attention_mask.astype(np.int64)).to(self.device)
        with t.no_grad():
            out = self.model(ids, mask)
        if self.device == "mps":
            t.mps.synchronize()  # else we time the dispatch, not the work
        return out.detach().to("cpu").numpy().astype(np.float32)


class CoreMLBackend:
    """CoreML with an explicit compute-unit request.

    `CPU_AND_NE` is a *request*, not a guarantee. CoreML partitions the graph
    per operation and silently falls back for anything the Neural Engine cannot
    take. `residency()` surfaces the resulting split so an "ANE number" that is
    really a CPU number is visible rather than assumed.
    """

    UNITS = {
        "cpu": "CPU_ONLY",
        "gpu": "CPU_AND_GPU",
        "ane": "CPU_AND_NE",
        "all": "ALL",
    }

    def __init__(self, package: Path | str, unit: str = "ane"):
        import coremltools as ct

        if unit not in self.UNITS:
            raise ValueError(f"unit must be one of {sorted(self.UNITS)}")

        self.ct = ct
        self.package = Path(package)
        self.unit = unit
        self.name = f"coreml-{unit}"
        self.model = ct.models.MLModel(
            str(self.package),
            compute_units=getattr(ct.ComputeUnit, self.UNITS[unit]),
        )

        spec_inputs = self.model.get_spec().description.input
        self.input_names = [i.name for i in spec_inputs]
        self.output_name = self.model.get_spec().description.output[0].name

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        out = self.model.predict(
            {
                "input_ids": input_ids.astype(np.int32),
                "attention_mask": attention_mask.astype(np.int32),
            }
        )
        return np.asarray(out[self.output_name], dtype=np.float32).reshape(-1)

    def residency(self) -> dict[str, Any]:
        """Per-op device split, read from the conversion report.

        Deliberately *not* recomputed here. `MLComputePlan.load_from_path`
        requires a compiled .mlmodelc, and handing it an .mlpackage aborts
        inside libc++ -- which terminates the process outright, sailing past any
        Python `except` and taking the benchmark run with it. Residency is a
        static property of a package anyway, so it is computed once at
        conversion time (under subprocess isolation, in scripts/convert_coreml.py)
        and simply looked up here.
        """
        import json

        report = self.package.parent / "conversion.json"
        if not report.exists():
            return {"error": "conversion.json not found -- run scripts/convert_coreml.py --report-plan"}
        try:
            data = json.loads(report.read_text())
            entry = data.get(self.package.stem, {})
            return entry.get("plan", {"error": "no plan recorded for this package"})
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


class Tokenizer:
    """Padding to fixed buckets, because CoreML packages are shape-specialised.

    Tokenisation is counted in the end-to-end budget even though it is small --
    the point of the exercise is that nothing gets to be free.
    """

    def __init__(self, buckets: tuple[int, ...] = (32, 64, 128, 256), model_name: str | None = None):
        from . import encoder as E

        self.tok = E.load_tokenizer(model_name or E.DEFAULT_MODEL)
        self.buckets = tuple(sorted(buckets))

    def bucket_for(self, n_tokens: int) -> int:
        for b in self.buckets:
            if n_tokens <= b:
                return b
        return self.buckets[-1]

    def encode(self, text: str, seq_len: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        raw = self.tok(text, add_special_tokens=True, truncation=True, max_length=self.buckets[-1])
        n = len(raw["input_ids"])
        target = seq_len or self.bucket_for(n)

        enc = self.tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=target,
            return_tensors="np",
        )
        return (
            enc["input_ids"].astype(np.int32),
            enc["attention_mask"].astype(np.int32),
            n,
        )
