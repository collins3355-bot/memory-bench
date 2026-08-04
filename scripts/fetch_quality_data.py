#!/usr/bin/env python
"""Download the retrieval-quality datasets into data/quality/.

Sources:
  * LongMemEval (cleaned) -- the authors deprecated the original HF dataset in
    favour of `xiaowu0162/longmemeval-cleaned`; this fetches that.
    `_m` (2.7 GB) is deliberately not fetched by default.
  * LoCoMo -- fetched straight from the paper repo.

Both are research benchmarks published for evaluation use; see their repos for
licence terms (LongMemEval: MIT; LoCoMo: see snap-research/locomo).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

FILES = {
    "longmemeval_oracle.json": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
        15.4,
    ),
    "longmemeval_s.json": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        277.4,
    ),
    "locomo10.json": (
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        2.8,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("data/quality"), type=Path)
    ap.add_argument("--only", nargs="*", help="subset of filenames to fetch")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (url, mb) in FILES.items():
        if args.only and name not in args.only:
            continue
        dest = args.out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"[get]  {name} (~{mb} MB) from {url.split('/')[2]}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"       -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        except Exception as exc:
            print(f"[fail] {name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
