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


QUALITY_DATASET_NOTES = {
    "longmemeval_s": "LongMemEval-S: 500 questions, ~50-session haystacks (~115k tokens each)",
    "locomo": "LoCoMo: 10 long conversations, ~2k questions with per-turn evidence labels",
}


def quality_section(d: dict, dataset: str) -> list[str]:
    note = QUALITY_DATASET_NOTES.get(dataset, dataset)
    out = [
        f"### Retrieval quality — {dataset}",
        "",
        f"{note}. {d['n_instances']} instances, {d['n_abstention']} abstention "
        "(excluded from means). Encoder: MiniLM-L6, the same model as the perf "
        "benchmarks.",
        "",
        "| chunking | arm | sess_r@10 | complete@10 | turn_r@10 | MRR | tokens@10 |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for chunking, arms in d["results"].items():
        for arm, agg in arms.items():
            m = agg["by_metric"]
            out.append(
                f"| {chunking} | `{arm}` | {m['session_recall@10']:.3f} | "
                f"{m['complete@10']:.3f} | {m['turn_recall@10']:.3f} | "
                f"{m['mrr']:.3f} | {m['tokens@10']:,.0f} |"
            )
    out.append("")

    # Per-question-type breakdown at one operating point per dataset: the
    # chunking whose best arm has the highest complete@10.
    best_chunking, best_score = None, -1.0
    for chunking, arms in d["results"].items():
        top = max(a["by_metric"]["complete@10"] for a in arms.values())
        if top > best_score:
            best_chunking, best_score = chunking, top
    arms = d["results"][best_chunking]
    qtypes = sorted(next(iter(arms.values()))["by_qtype"])
    out += [
        f"**complete@10 by question type** (chunking: `{best_chunking}`)",
        "",
        "| arm | " + " | ".join(qtypes) + " |",
        "|---|" + "--:|" * len(qtypes),
    ]
    for arm, agg in arms.items():
        cells = " | ".join(
            f"{agg['by_qtype'][qt]['complete@10']:.3f}" if qt in agg["by_qtype"] else "—"
            for qt in qtypes
        )
        out.append(f"| `{arm}` | {cells} |")
    out.append("")
    return out


def ksweep_section(d: dict) -> list[str]:
    """complete@k growth for the hybrid arm -- is k the lever for hard questions?"""
    ks = d.get("ks", [5, 10])
    if len(ks) < 3:
        return []
    out = [
        "### Does retrieving more fix the hard questions? (k-sweep)",
        "",
        f"complete@k for the `hybrid` arm on longmemeval_s, overall and for the "
        "hardest type (multi-session), as k grows.",
        "",
        "| chunking | slice | " + " | ".join(f"c@{k}" for k in ks) + " | tokens@" + str(ks[-1]) + " |",
        "|---|---|" + "--:|" * (len(ks) + 1),
    ]
    for chunking, arms in d["results"].items():
        agg = arms.get("hybrid")
        if not agg:
            continue
        m = agg["by_metric"]
        cells = " | ".join(f"{m[f'complete@{k}']:.3f}" for k in ks)
        out.append(f"| {chunking} | overall | {cells} | {m[f'tokens@{ks[-1]}']:,.0f} |")
        ms = agg["by_qtype"].get("multi-session")
        if ms:
            cells = " | ".join(f"{ms[f'complete@{k}']:.3f}" for k in ks)
            out.append(f"| {chunking} | multi-session | {cells} | {ms[f'tokens@{ks[-1]}']:,.0f} |")
    out.append("")
    return out


def encoder_section(base: dict, e5: dict, dataset: str) -> list[str]:
    """MiniLM vs e5-small, isolated on the arms where the encoder matters."""
    out = [
        f"### Encoder sweep — {dataset}",
        "",
        "Same grid, dense encoder swapped: MiniLM-L6 (22.7M, mean-pool) vs "
        "e5-small-v2 (33M, 12 layers, asymmetric query/passage prefixes). "
        "complete@10; Δ is e5 minus MiniLM.",
        "",
        "| chunking | arm | MiniLM | e5-small | Δ |",
        "|---|---|--:|--:|--:|",
    ]
    for chunking in base["results"]:
        for arm in ("vector", "hybrid"):
            b = base["results"][chunking][arm]["by_metric"]["complete@10"]
            e = e5["results"][chunking][arm]["by_metric"]["complete@10"]
            out.append(
                f"| {chunking} | `{arm}` | {b:.3f} | {e:.3f} | {e - b:+.3f} |"
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

    for dataset in ("longmemeval_s", "locomo"):
        d = _load(f"quality_{dataset}")
        if d:
            parts += quality_section(d, dataset)
        else:
            print(f"[skip] results/quality_{dataset}.json not found")

    lme = _load("quality_longmemeval_s")
    if lme:
        parts += ksweep_section(lme)
    for dataset in ("longmemeval_s", "locomo"):
        base, e5 = _load(f"quality_{dataset}"), _load(f"quality_{dataset}_e5-small")
        if base and e5:
            parts += encoder_section(base, e5, dataset)
        elif not e5:
            print(f"[skip] results/quality_{dataset}_e5-small.json not found")
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
