"""Machine fingerprint recorded alongside every result set.

A latency number without the machine it was measured on is not a result.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _blas_backend() -> dict[str, Any]:
    """Which BLAS numpy is actually calling.

    This matters more than people expect: Accelerate reaches the AMX coprocessor
    on Apple Silicon, OpenBLAS does not. Same numpy call, very different GEMV
    throughput.
    """
    try:
        import numpy as np

        cfg = getattr(np, "__config__", None)
        info: dict[str, Any] = {"numpy": np.__version__}
        if cfg is not None and hasattr(cfg, "show"):
            try:
                blob = cfg.show(mode="dicts")  # numpy >= 1.25
                builds = blob.get("Build Dependencies", {})
                blas = builds.get("blas", {})
                info["blas_name"] = blas.get("name")
                info["blas_version"] = blas.get("version")
            except Exception:
                pass
        return info
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect() -> dict[str, Any]:
    mem = _sysctl("hw.memsize")
    info: dict[str, Any] = {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "arch": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "ram_gb": round(int(mem) / 1024**3, 1) if mem else None,
        "p_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "e_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "logical_cpus": _sysctl("hw.logicalcpu"),
    }
    info.update(_blas_backend())

    for mod in ("torch", "coremltools", "usearch", "transformers"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception:
            info[f"{mod}_version"] = None

    try:
        import torch

        info["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:
        info["mps_available"] = None

    return info


def banner() -> str:
    return json.dumps(collect(), indent=2)
