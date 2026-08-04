#!/usr/bin/env python
"""Render results/*.json into the README, replacing the <!-- RESULTS --> block.

Keeps the published numbers mechanically tied to the JSON the harness emitted,
so the README cannot drift from the measurements through hand-editing.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
README = Path("README.md")
MARKER = "<!-- RESULTS -->"
END = "<!-- /RESULTS -->"


def _load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _mb(b: float) -> str:
    return f"{b / 1e6:,.0f}"


def search_section(d: dict) -> list[str]:
    out = ["### Retrieval", ""]
    for n in sorted(d["scales"], key=int):
        block = d["scales"][n]
        out += [
            f"**n = {int(n):,} vectors, d=384, k=10** "
            f"({int(n) * 384 * 4 / 1e9:.2f} GB as fp32)",
            "",
            "| backend | p50 | p99 | recall@10 | build | resident |",
            "|---|--:|--:|--:|--:|--:|",
        ]
        for b in block["backends"]:
            out.append(
                f"| `{b['label']}` | {b['p50']:.2f} ms | {b['p99']:.2f} ms | "
                f"{b['recall_at_k']:.4f} | {b['build_seconds']:.1f} s | "
                f"{_mb(b['index_bytes'])} MB |"
            )
        out.append("")
    return out


def embed_section(d: dict) -> list[str]:
    out = [
        "### Query embedding",
        "",
        "| backend | seq | p50 | p99 | ANE residency |",
        "|---|--:|--:|--:|--:|",
    ]
    for b in sorted(d["backends"], key=lambda r: (r["seq_len"], r["p50"])):
        res = b.get("residency") or {}
        frac = res.get("ane_fraction")
        tag = "n/a" if frac is None else f"{frac:.0%}"
        out.append(
            f"| `{b['label']}` | {b['seq_len']} | {b['p50']:.2f} ms | "
            f"{b['p99']:.2f} ms | {tag} |"
        )
    out.append("")
    return out


def contention_section(d: dict) -> list[str]:
    out = [
        "### Background indexing vs foreground latency",
        "",
        f"Foreground query p50/p99 at n={d['n']:,} while an indexer saturates cores "
        "at each QoS class.",
        "",
        "| condition | query p50 | query p99 | vs idle | indexer work done |",
        "|---|--:|--:|--:|--:|",
    ]
    for c in d["conditions"]:
        out.append(
            f"| {c['condition']} | {c['query_p50_ms']:.2f} ms | "
            f"{c['query_p99_ms']:.2f} ms | {c['slowdown_vs_idle']:.2f}x | "
            f"{c['indexer_ops']:,} ops |"
        )
    out.append("")
    return out


def pipeline_section(d: dict) -> list[str]:
    stages = d["stages"]
    total = stages["5_end_to_end"]["p50"]
    out = [
        "### End-to-end budget",
        "",
        f"n={d['n']:,}, embedder `{d['embedder']}`, seq_len={d['seq_len']}.",
        "",
        "| stage | p50 | p99 | share of e2e |",
        "|---|--:|--:|--:|",
    ]
    for key in sorted(stages):
        s = stages[key]
        share = "" if key.startswith("5") else f"{s['p50'] / total * 100:.0f}%"
        out.append(f"| {s['label']} | {s['p50']:.2f} ms | {s['p99']:.2f} ms | {share} |")
    out.append("")
    verdict = "within" if stages["5_end_to_end"]["p99"] < d["budget_ms"] else "over"
    out.append(
        f"End-to-end p99 is **{stages['5_end_to_end']['p99']:.2f} ms**, "
        f"{verdict} the {d['budget_ms']:.0f} ms budget."
    )
    out.append("")
    return out


def main() -> int:
    parts: list[str] = [MARKER, ""]
    for name, fn in (
        ("search", search_section),
        ("embed", embed_section),
        ("contention", contention_section),
        ("pipeline", pipeline_section),
    ):
        d = _load(name)
        if d:
            parts += fn(d)
        else:
            print(f"[skip] results/{name}.json not found")
    parts.append(END)

    text = README.read_text()
    if MARKER not in text:
        print("[error] README has no <!-- RESULTS --> marker")
        return 1

    head = text.split(MARKER)[0]
    tail = text.split(END)[1] if END in text else text.split(MARKER)[1]
    README.write_text(head + "\n".join(parts) + tail)
    print(f"[ok] wrote {len(parts)} lines into {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
