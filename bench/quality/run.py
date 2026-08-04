"""Quality-eval driver.

    python -m bench.quality.run --dataset longmemeval_s
    python -m bench.quality.run --dataset locomo --chunkings turn,window

Design notes:

  * Chunks, BM25 state and token counts are built once per *haystack*, not per
    instance. LoCoMo asks ~200 questions about each conversation; rebuilding
    per question would multiply all preprocessing by 200 for nothing.
  * All embedding goes through the content-addressed cache, so a rerun with a
    new arm embeds nothing and finishes in seconds. Arms are cheap; embeddings
    are the only expensive thing here.
  * Ages are computed against the question date when the dataset provides one
    (LongMemEval) and against the newest session in the haystack otherwise
    (LoCoMo). Chunks with unparseable dates get +inf age -- a recency vote
    should never *promote* an undated chunk.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from . import chunk as CH
from . import data as D
from . import metrics as M
from .bm25 import BM25
from .embedder import BulkEmbedder, EmbedCache
from .retrieve import ARMS, rank_arm


def _hash_uid(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def build_haystacks(
    instances: list[D.Instance], chunking: str
) -> dict[str, list[CH.Chunk]]:
    chunker = CH.CHUNKERS[chunking]
    by_key: dict[str, list[CH.Chunk]] = {}
    seen_instance_of: dict[str, D.Instance] = {}
    for inst in instances:
        if inst.haystack_key not in by_key:
            by_key[inst.haystack_key] = chunker(inst)
            seen_instance_of[inst.haystack_key] = inst
    return by_key


def stamp_tokens(chunks: list[CH.Chunk], tok) -> None:
    texts = [c.text for c in chunks]
    # Batch through the fast tokenizer; special tokens excluded because the
    # count models context cost of the *content*, not of encoder framing.
    enc = tok(texts, add_special_tokens=False, truncation=False, verbose=False)
    for c, ids in zip(chunks, enc["input_ids"]):
        c.n_tokens = len(ids)


def ages_for(chunks: list[CH.Chunk], q_dt: datetime | None) -> np.ndarray:
    dts = [c.dt for c in chunks]
    ref = q_dt
    if ref is None:
        known = [d for d in dts if d is not None]
        ref = max(known) if known else None
    out = np.full(len(chunks), np.inf, dtype=np.float64)
    if ref is None:
        return out
    for i, d in enumerate(dts):
        if d is not None:
            out[i] = (ref - d).total_seconds() / 86400.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    from .embedder import MODELS

    ap.add_argument("--dataset", default="longmemeval_s",
                    choices=["longmemeval_oracle", "longmemeval_s", "locomo"])
    ap.add_argument("--chunkings", default="turn,window,session")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--model", default="minilm", choices=sorted(MODELS))
    ap.add_argument("--ks", default="5,10,20,50",
                    help="k values for recall/complete/tokens metrics")
    ap.add_argument("--limit", type=int, default=0, help="cap instances, 0 = all")
    ap.add_argument("--data", default="data/quality")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rerank-arms", default="",
                    help="comma-separated base arms whose top --rerank-depth gets cross-encoder reranked")
    ap.add_argument("--rerank-depth", type=int, default=50)
    args = ap.parse_args()

    chunkings = [c for c in args.chunkings.split(",") if c]
    arms = [a for a in args.arms.split(",") if a]
    ks = tuple(int(k) for k in args.ks.split(",") if k)
    rerank_arms = [a for a in args.rerank_arms.split(",") if a]
    unknown = set(rerank_arms) - set(arms)
    if unknown:
        raise SystemExit(f"--rerank-arms must be a subset of --arms; unknown: {sorted(unknown)}")

    t0 = time.time()
    instances = D.load(args.dataset, args.data)
    if args.limit:
        instances = instances[: args.limit]
    n_abst = sum(1 for i in instances if i.abstention)
    scored_instances = [i for i in instances if not i.abstention]
    print(
        f"[load] {args.dataset}: {len(instances)} instances "
        f"({n_abst} abstention, excluded from means), "
        f"{len({i.haystack_key for i in instances})} haystacks, {time.time()-t0:.1f}s"
    )

    embedder = BulkEmbedder(model=args.model, device=args.device)
    cache = EmbedCache(Path(args.data) / "cache", embedder.spec["slug"])
    qp, pp = embedder.spec["query_prefix"], embedder.spec["passage_prefix"]
    print(f"[embed] model={args.model}, device={embedder.device}, cache has {len(cache.index)} texts")

    reranker = None
    ce_cache = None
    if rerank_arms:
        from .rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(device=args.device)
        # Pair scores depend only on the two texts, so the cache is shared
        # across first-stage encoders, chunkings and datasets.
        ce_cache = EmbedCache(Path(args.data) / "cache", "ce-msmarco-minilm-l6")
        print(f"[rerank] ce depth={args.rerank_depth}, device={reranker.device}, "
              f"cache has {len(ce_cache.index)} pairs")

    results: dict[str, dict] = {}
    for chunking in chunkings:
        t1 = time.time()
        haystacks = build_haystacks(instances, chunking)
        all_chunks = [c for chunks in haystacks.values() for c in chunks]
        print(f"\n[{chunking}] {len(all_chunks)} chunks over {len(haystacks)} haystacks")

        for chunks in haystacks.values():
            stamp_tokens(chunks, embedder.tok)

        # One embedding pass over everything new: chunk texts + questions. The
        # cache key hashes the *prefixed* string, because that is what gets
        # embedded -- e5's "query: "/"passage: " prefixes produce different
        # vectors for the same text, and a key that ignored them would serve
        # passage vectors where query vectors were meant.
        uid_text: dict[str, str] = {}
        for c in all_chunks:
            uid_text.setdefault(_hash_uid(pp + c.text), pp + c.text)
        for inst in instances:
            uid_text.setdefault(_hash_uid(qp + inst.question), qp + inst.question)
        todo = cache.missing(list(uid_text))
        if todo:
            print(f"  embedding {len(todo)} new texts ({len(uid_text)-len(todo)} cached)")
            vecs = embedder.encode([uid_text[u] for u in todo])
            cache.add(todo, vecs)
            cache.save()

        bm25_by_key = {key: BM25([c.text for c in chunks]) for key, chunks in haystacks.items()}
        vec_by_key = {
            key: cache.gather([_hash_uid(pp + c.text) for c in chunks])
            for key, chunks in haystacks.items()
        }

        per_arm_scores: dict[str, list[M.InstanceScore]] = defaultdict(list)
        # Pass 1: base rankings for every instance and arm. Rerank candidates
        # (top `rerank_depth` of each base ranking) are collected rather than
        # scored inline, so the cross-encoder sees one large batched workload
        # instead of 470 tiny ones.
        base_rankings: dict[tuple[str, str], np.ndarray] = {}
        for inst in scored_instances:
            chunks = haystacks[inst.haystack_key]
            qvec = cache.gather([_hash_uid(qp + inst.question)])[0]
            cos = vec_by_key[inst.haystack_key] @ qvec
            bm = np.asarray(bm25_by_key[inst.haystack_key].scores(inst.question), dtype=np.float64)
            ages = ages_for(chunks, inst.q_dt)
            for arm in arms:
                ranking = rank_arm(arm, bm, cos, ages)
                per_arm_scores[arm].append(M.score_ranking(ranking, chunks, inst, ks=ks))
                if arm in rerank_arms:
                    base_rankings[(inst.id, arm)] = ranking

        # Pass 2: cross-encoder rerank of the wide head of each base ranking.
        if rerank_arms:
            from .rerank import apply_rerank, pair_key

            depth = args.rerank_depth
            need: dict[str, tuple[str, str]] = {}
            for inst in scored_instances:
                chunks = haystacks[inst.haystack_key]
                for arm in rerank_arms:
                    for ci in base_rankings[(inst.id, arm)][:depth]:
                        k = pair_key(inst.question, chunks[ci].text)
                        need.setdefault(k, (inst.question, chunks[ci].text))
            todo = ce_cache.missing(list(need))
            if todo:
                print(f"  scoring {len(todo)} new CE pairs ({len(need)-len(todo)} cached)")
                scores = reranker.score_pairs([need[k] for k in todo])
                ce_cache.add(todo, scores.reshape(-1, 1))
                ce_cache.save()

            for inst in scored_instances:
                chunks = haystacks[inst.haystack_key]
                for arm in rerank_arms:
                    ranking = base_rankings[(inst.id, arm)]
                    keys = [pair_key(inst.question, chunks[ci].text) for ci in ranking[:depth]]
                    ce_scores = ce_cache.gather(keys).reshape(-1)
                    reranked = apply_rerank(ranking, depth, ce_scores)
                    per_arm_scores[f"{arm}+ce{depth}"].append(
                        M.score_ranking(reranked, chunks, inst, ks=ks)
                    )

        results[chunking] = {
            arm: M.aggregate(scores, n_abst, ks=ks).__dict__
            for arm, scores in per_arm_scores.items()
        }
        print(f"  scored {len(scored_instances)} instances x {len(arms)} arms in {time.time()-t1:.1f}s")

        hdr = f"  {'arm':14} {'sess_r@10':>9} {'complete@10':>11} {'turn_r@10':>9} {'mrr':>6} {'tok@10':>8}"
        print(hdr)
        for arm in results[chunking]:  # includes any +ce rerank arms
            m = results[chunking][arm]["by_metric"]
            print(
                f"  {arm:14} {m['session_recall@10']:9.3f} {m['complete@10']:11.3f} "
                f"{m['turn_recall@10']:9.3f} {m['mrr']:6.3f} {m['tokens@10']:8.0f}"
            )

    suffix = "" if args.model == "minilm" else f"_{args.model}"
    out_path = Path(args.out or f"results/quality_{args.dataset}{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from .. import sysinfo

    out_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "model": args.model,
                "ks": list(ks),
                "n_instances": len(instances),
                "n_abstention": n_abst,
                "machine": sysinfo.collect(),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n-> {out_path}  ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
