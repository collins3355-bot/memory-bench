"""Synthetic corpora with the geometry real embeddings actually have.

This module is the part of the benchmark most likely to produce a lie, so it is
worth being explicit about why it is built the way it is.

The naive move is `np.random.randn(n, d)` normalised to the sphere. Do not do
this. Uniform points on a high-dimensional sphere are the pathological worst
case for every technique under test:

  * ANN indexes look terrible, because uniform data has no cluster structure to
    exploit and every point is roughly equidistant from every other.
  * Quantisation looks terrific, because there is no structure to destroy.

Both errors point in the direction of "the fancy index is not worth it", which
happens to be the conclusion this project leans toward -- so the benchmark must
not be allowed to reach it for free.

Real sentence-embedding corpora have two properties we reproduce here:

1. **Cluster structure.** Documents fall into topics. Within-topic cosine
   similarity runs ~0.6-0.85, across-topic ~0.0-0.3.

2. **Anisotropy.** Embedding spaces are cones, not spheres. Nearly all vectors
   share a large common component, so random pairs have *positive* mean cosine
   similarity (typically 0.15-0.4 for MiniLM-class models) rather than ~0. This
   is not a nuisance detail: it is precisely what breaks naive sign-based binary
   quantisation, because if every vector shares a dominant direction then most
   of them agree on the sign of most coordinates and the codes collapse. The
   fix -- centering before binarising -- is measurable only if the synthetic
   data has the flaw in the first place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHUNK_ROWS = 250_000


@dataclass
class Corpus:
    vectors: np.ndarray  # (n, d) float32, L2-normalised, memory-mapped
    path: Path
    meta: dict

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def d(self) -> int:
        return int(self.vectors.shape[1])


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n


def generate(
    n: int,
    d: int = 384,
    *,
    clusters: int = 2_000,
    within_cluster: float = 0.72,
    anisotropy: float = 0.25,
    seed: int = 0,
    out_dir: Path | str = "data",
    force: bool = False,
) -> Corpus:
    """Build (or reuse) an `n` x `d` float32 corpus on disk.

    Both knobs are stated as the cosine similarity they produce, rather than as
    raw magnitudes, because raw magnitudes in high dimension are extremely easy
    to get wrong: an isotropic noise vector in d dimensions has norm ~sqrt(d),
    so a "large" cluster offset of 3.0 is in fact swamped by unit noise at
    d=384, and the corpus comes out isotropic while looking correctly written.

    Each vector is composed as three orthogonal-in-expectation parts:

        x = a * common + b * cluster_centre + c * noise,   a^2 + b^2 + c^2 = 1

    In high dimension random directions are near-orthogonal, so the dot product
    of two vectors collapses to just their shared components:

        different clusters -> a^2          == anisotropy
        same cluster       -> a^2 + b^2    == within_cluster

    which makes both parameters directly checkable against the generated output.

    within_cluster -- cosine between two documents on the same topic (~0.72).
    anisotropy     -- cosine between two *unrelated* documents. Real encoders
                      give 0.15-0.4 rather than 0, because the embedding space
                      is a cone. Set 0.0 for an isotropic sphere.
    """
    if not 0.0 <= anisotropy < within_cluster <= 1.0:
        raise ValueError("require 0 <= anisotropy < within_cluster <= 1")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"corpus_n{n}_d{d}_c{clusters}_w{within_cluster}_a{anisotropy}_s{seed}"
    vec_path = out_dir / f"{stem}.npy"
    meta_path = out_dir / f"{stem}.json"

    if vec_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        vectors = np.load(vec_path, mmap_mode="r")
        return Corpus(vectors=vectors, path=vec_path, meta=meta)

    rng = np.random.default_rng(seed)
    centers = _unit(rng.standard_normal((clusters, d)).astype(np.float32))
    common = _unit(rng.standard_normal((1, d)).astype(np.float32))

    a = float(np.sqrt(anisotropy))
    b = float(np.sqrt(within_cluster - anisotropy))
    c = float(np.sqrt(1.0 - within_cluster))

    vectors = np.lib.format.open_memmap(
        vec_path, mode="w+", dtype=np.float32, shape=(n, d)
    )

    written = 0
    while written < n:
        rows = min(CHUNK_ROWS, n - written)
        assign = rng.integers(0, clusters, size=rows)
        # Each part is unit-norm before weighting, so a/b/c are the actual
        # component magnitudes rather than something scaled by sqrt(d).
        noise = _unit(rng.standard_normal((rows, d)).astype(np.float32))
        block = a * common + b * centers[assign] + c * noise
        vectors[written : written + rows] = _unit(block)
        written += rows

    vectors.flush()

    meta = {
        "n": n,
        "d": d,
        "clusters": clusters,
        "within_cluster": within_cluster,
        "anisotropy": anisotropy,
        "components": {"common": a, "cluster": b, "noise": c},
        "seed": seed,
        "bytes_fp32": int(n) * int(d) * 4,
    }
    meta.update(_describe(np.load(vec_path, mmap_mode="r"), seed=seed))
    meta_path.write_text(json.dumps(meta, indent=2))

    return Corpus(
        vectors=np.load(vec_path, mmap_mode="r"), path=vec_path, meta=meta
    )


def _describe(vectors: np.ndarray, *, seed: int, sample: int = 4096) -> dict:
    """Sanity-check the geometry so a mis-tuned corpus is visible in the output."""
    rng = np.random.default_rng(seed + 991)
    n = vectors.shape[0]
    take = min(sample, n)
    idx = np.sort(rng.choice(n, size=take, replace=False))
    S = np.asarray(vectors[idx], dtype=np.float32)

    sims = S @ S.T
    iu = np.triu_indices(take, k=1)
    pair = sims[iu]

    return {
        "mean_pair_cosine": float(pair.mean()),
        "p99_pair_cosine": float(np.quantile(pair, 0.99)),
        "max_pair_cosine": float(pair.max()),
        "mean_abs_coord": float(np.abs(S).mean()),
        # Fraction of coordinates whose sign matches the corpus mean's sign.
        # 0.5 is isotropic; real encoders land ~0.6-0.75 and that is what
        # wrecks uncentered binary quantisation.
        "sign_agreement_with_mean": float(
            (np.sign(S) == np.sign(S.mean(axis=0))).mean()
        ),
    }


def queries(
    corpus: Corpus, count: int = 512, *, seed: int = 7, similarity: float = 0.65
) -> np.ndarray:
    """Queries drawn as perturbed corpus members.

    Real queries are neither corpus members (too easy -- exact match at rank 1)
    nor random vectors (too hard -- no meaningful neighbour exists, so top-k is
    noise and every recall number measures nothing). Perturbing a real document
    approximates "user asks about something they discussed".

    `similarity` is the expected cosine between the query and the document it
    was derived from, and is enforced directly:

        q = s * doc + sqrt(1 - s^2) * unit_noise

    Note the `_unit` on the noise. An un-normalised `standard_normal(d)` has
    norm ~sqrt(d) -- about 19.6 at d=384 -- so the intuitive-looking "add 0.35
    of noise" in fact buries a unit-norm document under 6.9 units of noise and
    yields a query with cosine ~0.14 to its own source. This is the same
    high-dimensional scaling trap documented in `generate`, and it is worth
    stating twice because it fails silently: the benchmark still runs, still
    reports recall, and every number is meaningless.
    """
    if not 0.0 < similarity <= 1.0:
        raise ValueError("similarity must be in (0, 1]")

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(corpus.n, size=count, replace=False))
    base = np.asarray(corpus.vectors[idx], dtype=np.float32)

    noise = _unit(rng.standard_normal(base.shape).astype(np.float32))
    s = np.float32(similarity)
    w = np.float32(np.sqrt(1.0 - float(s) * float(s)))

    # Both coefficients are forced to float32 explicitly. `np.sqrt()` returns a
    # float64 *numpy scalar*, and unlike a plain Python float that is not "weak"
    # under NEP 50 -- multiplying a float32 array by it silently promotes the
    # result to float64. A float64 query against a float32 index then makes
    # numpy upcast the entire index on every single search, which cost 5x here
    # (9.8ms vs 1.9ms at n=60k) while every recall number stayed perfect. Cheap
    # to cause, invisible in the output, so the dtype is asserted below.
    out = _unit(s * base + w * noise)
    assert out.dtype == np.float32, f"queries must stay float32, got {out.dtype}"
    return out


def exact_neighbors(
    corpus: Corpus, q: np.ndarray, k: int = 10, *, block: int = 200_000
) -> np.ndarray:
    """Ground truth top-k by exact cosine, computed in blocks to bound memory."""
    nq = q.shape[0]
    best_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    best_idx = np.zeros((nq, k), dtype=np.int64)

    for start in range(0, corpus.n, block):
        stop = min(start + block, corpus.n)
        chunk = np.asarray(corpus.vectors[start:stop], dtype=np.float32)
        scores = q @ chunk.T  # (nq, chunk)

        take = min(k, scores.shape[1])
        part = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        part_scores = np.take_along_axis(scores, part, axis=1)

        cand_scores = np.concatenate([best_scores, part_scores], axis=1)
        cand_idx = np.concatenate([best_idx, part + start], axis=1)
        keep = np.argsort(-cand_scores, axis=1)[:, :k]
        best_scores = np.take_along_axis(cand_scores, keep, axis=1)
        best_idx = np.take_along_axis(cand_idx, keep, axis=1)

    return best_idx


def recall_at_k(got: np.ndarray, truth: np.ndarray) -> float:
    """Mean fraction of the true top-k recovered. `got` may be wider than `truth`."""
    k = truth.shape[1]
    hits = [
        len(set(got[i, :k].tolist()) & set(truth[i].tolist())) / k
        for i in range(truth.shape[0])
    ]
    return float(np.mean(hits))
