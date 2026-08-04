"""Cross-encoder reranking: turning wide-retrieval recall into small-k precision.

The k-sweep reframed the retrieval problem. At k=50 over turn chunks, evidence
recall is essentially solved (complete@50 = 0.98 on LongMemEval-S) — but 11.5k
tokens is too much context to hand an LLM per query. The open question is how
much of that recall survives compression back to k=5-10, and the standard tool
is a cross-encoder: score each (query, chunk) pair jointly, reorder the wide
candidate set, truncate.

A cross-encoder reads the query and passage *together* through one transformer,
so it models interactions a bi-encoder structurally cannot ("my car" in the
query binding to "the GPS" in the passage). The price is that scores cannot be
precomputed: 50 forward passes per query instead of one dot product. That price
is why it belongs at rerank depth 50, not as a first stage over 250k chunks.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (22.7M params) — same size class as
the bi-encoders under test, so the comparison stays "architecture vs
architecture" rather than "small model vs big model". Scores are cached by
(query, passage) content hash, reusing the embedding cache machinery with
1-dimensional "vectors": pairs are text, so the cache is shared across
first-stage encoders and reruns.
"""

from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_CE = "cross-encoder/ms-marco-MiniLM-L6-v2"

# \x1f (unit separator) cannot appear in tokenised chat text, so the pair key
# is unambiguous -- "ab" + "c" and "a" + "bc" hash differently.
_SEP = "\x1f"


def pair_key(query: str, passage: str) -> str:
    return hashlib.sha1((query + _SEP + passage).encode("utf-8")).hexdigest()[:20]


def apply_rerank(
    ranking: np.ndarray, depth: int, scores: np.ndarray
) -> np.ndarray:
    """Reorder the top `depth` of `ranking` by `scores` (desc); tail unchanged.

    `scores[i]` corresponds to `ranking[i]` for i < depth. Pure function so the
    reorder logic is unit-testable without a model.
    """
    depth = min(depth, len(ranking))
    if depth <= 1:
        return ranking
    head = ranking[:depth]
    order = np.argsort(-scores[:depth], kind="stable")
    return np.concatenate([head[order], ranking[depth:]])


class CrossEncoderReranker:
    def __init__(
        self,
        model: str = DEFAULT_CE,
        device: str | None = None,
        max_len: int = 256,
        batch: int = 128,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.max_len = max_len
        self.batch = batch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model)
            .to(self.device)
            .eval()
        )
        self.tok = AutoTokenizer.from_pretrained(model)

    def score_pairs(self, pairs: list[tuple[str, str]], progress: bool = True) -> np.ndarray:
        """Relevance score per (query, passage) pair; rows align with input."""
        t = self.torch
        # Sort by combined length so batch padding is minimal (same trick as
        # the bulk embedder; worth ~2x throughput at these sizes).
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = np.zeros(len(pairs), dtype=np.float32)

        done = 0
        with t.no_grad():
            for start in range(0, len(order), self.batch):
                idxs = order[start : start + self.batch]
                enc = self.tok(
                    [pairs[i][0] for i in idxs],
                    [pairs[i][1] for i in idxs],
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                    token_type_ids=enc.get("token_type_ids", None).to(self.device)
                    if enc.get("token_type_ids") is not None
                    else None,
                ).logits
                scores = logits.squeeze(-1).to("cpu").numpy()
                for s, i in zip(scores, idxs):
                    out[i] = s
                done += len(idxs)
                if progress and done % (self.batch * 40) < self.batch:
                    print(f"    reranked {done}/{len(pairs)} pairs", flush=True)
        return out
