"""Bulk embedding with a content-addressed disk cache.

Two decisions here, both about iteration speed rather than cleverness:

  * **Batched eager PyTorch, not CoreML.** The perf harness showed the ANE
    wins single-query *latency*; bulk indexing is a *throughput* problem, and
    batching on MPS (or CPU) beats feeding a batch-1 fixed-shape CoreML
    package 250k times. This is also the honest architecture split the perf
    results recommend: ANE for the query path, batched write-path elsewhere.

  * **Cache by content hash, not by dataset.** LoCoMo asks ~200 questions per
    conversation, so its haystack texts repeat ~200x; LongMemEval shares
    distractor sessions across instances. Hashing the text collapses all of
    that: each unique string is embedded exactly once, ever, across datasets,
    chunkings and reruns. Changing the chunker invalidates exactly the chunks
    it changes and nothing else.

Texts are sorted by length before batching so padding within a batch is
minimal -- at MiniLM sizes this is a ~2x throughput difference, free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EmbedCache:
    """vectors.npy + keys.json, append-only, loaded fully into RAM.

    At 384 float32 dims, a million cached texts is ~1.5 GB -- acceptable for
    an eval tool, and far simpler than an on-disk KV store.
    """

    def __init__(self, cache_dir: Path | str, model_slug: str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.dir / f"{model_slug}_vectors.npy"
        self.key_path = self.dir / f"{model_slug}_keys.json"
        if self.vec_path.exists() and self.key_path.exists():
            self.vectors = np.load(self.vec_path)
            keys = json.loads(self.key_path.read_text())
            self.index: dict[str, int] = {k: i for i, k in enumerate(keys)}
        else:
            self.vectors = np.zeros((0, 0), dtype=np.float32)
            self.index = {}

    def missing(self, uids: list[str]) -> list[str]:
        seen = set()
        out = []
        for u in uids:
            if u not in self.index and u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def add(self, uids: list[str], vectors: np.ndarray) -> None:
        if not uids:
            return
        if self.vectors.size == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        base = len(self.index)
        for i, u in enumerate(uids):
            self.index[u] = base + i

    def save(self) -> None:
        np.save(self.vec_path, self.vectors)
        keys = [None] * len(self.index)
        for k, i in self.index.items():
            keys[i] = k
        self.key_path.write_text(json.dumps(keys))

    def gather(self, uids: list[str]) -> np.ndarray:
        rows = [self.index[u] for u in uids]
        return self.vectors[rows]


# Encoders the eval can sweep. Only mean-pooling BERT-architecture models fit
# the shared encoder implementation -- a CLS-pooling model (bge-*) loaded into
# a mean-pooling graph would silently misrepresent the model, so it is not
# offered. e5 requires its documented asymmetric prefixes; retrieval quality
# degrades measurably without them, which would unfairly handicap the model.
MODELS = {
    "minilm": {
        "hf": "sentence-transformers/all-MiniLM-L6-v2",
        "slug": "minilm-l6",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5-small": {
        "hf": "intfloat/e5-small-v2",
        "slug": "e5-small-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


class BulkEmbedder:
    def __init__(
        self,
        model: str = "minilm",
        device: str | None = None,
        max_len: int = 256,
        batch: int = 128,
    ):
        import torch

        from .. import encoder as E

        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; options: {sorted(MODELS)}")
        self.spec = MODELS[model]
        self.model_key = model
        self.torch = torch
        self.max_len = max_len
        self.batch = batch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = E.build("reference", self.spec["hf"]).to(self.device).eval()
        self.dim = self.model.cfg.hidden_size
        self.tok = E.load_tokenizer(self.spec["hf"])

    def encode(self, texts: list[str], progress: bool = True) -> np.ndarray:
        """Embed `texts` in length-sorted batches; rows align with input order.

        Texts must arrive already carrying any model prefix -- prefixing is the
        caller's job because the cache key must hash the exact string embedded.
        """
        t = self.torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)

        done = 0
        with t.no_grad():
            for start in range(0, len(order), self.batch):
                idxs = order[start : start + self.batch]
                enc = self.tok(
                    [texts[i] for i in idxs],
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(self.device)
                mask = enc["attention_mask"].to(self.device)
                vec = self.model(ids.int(), mask.int()).to("cpu").numpy()
                for row, i in zip(vec, idxs):
                    out[i] = row
                done += len(idxs)
                if progress and done % (self.batch * 20) < self.batch:
                    print(f"    embedded {done}/{len(texts)}", flush=True)
        return out
