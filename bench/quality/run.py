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
    ap.add_argument("--dataset", default="longmemeval_s",
                    choices=["longmemeval_oracle", "longmemeval_s", "locomo"])
    ap.add_argument("--chunkings", default="turn,window,session")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0, help="cap instances, 0 = all")
    ap.add_argument("--data", default="data/quality")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    chunkings = [c for c in args.chunkings.split(",") if c]
    arms = [a for a in args.arms.split(",") if a]

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

    embedder = BulkEmbedder(device=args.device)
    cache = EmbedCache(Path(args.data) / "cache", "minilm-l6")
    print(f"[embed] device={embedder.device}, cache has {len(cache.index)} texts")

    results: dict[str, dict] = {}
    for chunking in chunkings:
        t1 = time.time()
        haystacks = build_haystacks(instances, chunking)
        all_chunks = [c for chunks in haystacks.values() for c in chunks]
        print(f"\n[{chunking}] {len(all_chunks)} chunks over {len(haystacks)} haystacks")

        for chunks in haystacks.values():
            stamp_tokens(chunks, embedder.tok)

        # One embedding pass over everything new: chunk texts + questions.
        uid_text: dict[str, str] = {}
        for c in all_chunks:
            uid_text.setdefault(c.uid, c.text)
        for inst in instances:
            uid_text.setdefault(_hash_uid(inst.question), inst.question)
        todo = cache.missing(list(uid_text))
        if todo:
            print(f"  embedding {len(todo)} new texts ({len(uid_text)-len(todo)} cached)")
            vecs = embedder.encode([uid_text[u] for u in todo])
            cache.add(todo, vecs)
            cache.save()

        bm25_by_key = {key: BM25([c.text for c in chunks]) for key, chunks in haystacks.items()}
        vec_by_key = {key: cache.gather([c.uid for c in chunks]) for key, chunks in haystacks.items()}

        per_arm_scores: dict[str, list[M.InstanceScore]] = defaultdict(list)
        for inst in scored_instances:
            chunks = haystacks[inst.haystack_key]
            qvec = cache.gather([_hash_uid(inst.question)])[0]
            cos = vec_by_key[inst.haystack_key] @ qvec
            bm = np.asarray(bm25_by_key[inst.haystack_key].scores(inst.question), dtype=np.float64)
            ages = ages_for(chunks, inst.q_dt)
            for arm in arms:
                ranking = rank_arm(arm, bm, cos, ages)
                per_arm_scores[arm].append(M.score_ranking(ranking, chunks, inst))

        results[chunking] = {
            arm: M.aggregate(scores, n_abst).__dict__ for arm, scores in per_arm_scores.items()
        }
        print(f"  scored {len(scored_instances)} instances x {len(arms)} arms in {time.time()-t1:.1f}s")

        hdr = f"  {'arm':14} {'sess_r@10':>9} {'complete@10':>11} {'turn_r@10':>9} {'mrr':>6} {'tok@10':>8}"
        print(hdr)
        for arm in arms:
            m = results[chunking][arm]["by_metric"]
            print(
                f"  {arm:14} {m['session_recall@10']:9.3f} {m['complete@10']:11.3f} "
                f"{m['turn_recall@10']:9.3f} {m['mrr']:6.3f} {m['tokens@10']:8.0f}"
            )

    out_path = Path(args.out or f"results/quality_{args.dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from .. import sysinfo

    out_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
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
