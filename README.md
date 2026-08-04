# memory-bench

Where does the latency budget of a local AI long-term-memory system actually go?

This is a measurement harness, not a memory system. It exists to answer one
question with numbers instead of intuition: if you want an AI assistant that
remembers everything you have ever said to it, runs locally, and answers in
under 50ms without turning your laptop into a hairdryer — which part is
actually hard?

The short answer, on the hardware below: **the budget is comfortable, and almost
all of it goes to the vector scan — but the fix is eight threads, not a vector
database.**

## The claims under test

A local memory system is usually scoped around three milestones:

1. **The vector bottleneck** — search millions of past interaction tokens in
   under 50ms.
2. **Context compression** — summarise old conversations so the model does not
   choke on a huge history.
3. **Hardware optimisation** — run continuously in the background of a consumer
   machine without cooking the GPU.

Milestone 1 turns out to be correctly named and comfortably achievable: the
vector scan really does dominate the budget, and it still finishes well inside
50ms at realistic corpus sizes. Milestones 2 and 3 are aimed at the wrong
target, and this repo measures why.

**On scale.** Heavy daily use of an assistant runs about 50k tokens/day, so
roughly 18M tokens/year. At ~300 tokens per chunk that is **~60k vectors per
year of history**; a decade of heavy use lands near 600k. That is a small index.
The benchmark covers 60k / 600k / 6M so the 6M case is a decade of history with
an order of magnitude of headroom.

**On the GPU.** GPUs thermal-throttle rather than burn out, so milestone 3 is
not a hardware-damage problem. On Apple Silicon it is a *contention* problem —
fans, battery, and the user's own work stuttering — and the fix is a scheduling
hint, not SIMD.

## Results

Measured on the machine recorded in `results/*.json`:

| | |
|---|---|
| Chip | Apple M1 Max (8 performance + 2 efficiency cores, 32-core GPU, 16-core ANE) |
| Memory | 64 GB unified |
| numpy BLAS | Accelerate |
| Model | `all-MiniLM-L6-v2` — 22.7M params, 384 dims, 6 layers |

<!-- RESULTS -->

### Retrieval

**n = 60,000 vectors, d=384, k=10** (0.09 GB as fp32)

| backend | p50 | p99 | recall@10 | build | resident |
|---|--:|--:|--:|--:|--:|
| `flat-fp32` | 2.44 ms | 2.80 ms | 1.0000 | 0.0 s | 92 MB |
| `flat-fp32-t8` | 1.44 ms | 1.68 ms | 1.0000 | 0.0 s | 92 MB |
| `binary-raw-rr1000` | 1.77 ms | 1.95 ms | 0.9992 | 0.0 s | 95 MB |
| `binary-centered-rr1000` | 1.83 ms | 2.07 ms | 0.9992 | 0.0 s | 95 MB |
| `hnsw-m16-ef64` | 0.43 ms | 0.69 ms | 0.9836 | 8.2 s | 152 MB |

**n = 600,000 vectors, d=384, k=10** (0.92 GB as fp32)

| backend | p50 | p99 | recall@10 | build | resident |
|---|--:|--:|--:|--:|--:|
| `flat-fp32` | 24.86 ms | 26.21 ms | 1.0000 | 0.0 s | 922 MB |
| `flat-fp32-t8` | 12.32 ms | 13.81 ms | 1.0000 | 0.0 s | 922 MB |
| `binary-raw-rr1000` | 16.11 ms | 17.77 ms | 0.9930 | 0.1 s | 950 MB |
| `binary-centered-rr1000` | 16.08 ms | 26.88 ms | 0.9969 | 0.1 s | 950 MB |
| `hnsw-m16-ef64` | 0.39 ms | 0.79 ms | 0.7094 | 60.1 s | 1,202 MB |

**n = 6,000,000 vectors, d=384, k=10** (9.22 GB as fp32)

| backend | p50 | p99 | recall@10 | build | resident |
|---|--:|--:|--:|--:|--:|
| `flat-fp32` | 242.98 ms | 266.99 ms | 1.0000 | 0.0 s | 9,216 MB |
| `flat-fp32-t8` | 111.90 ms | 142.01 ms | 1.0000 | 0.0 s | 9,216 MB |
| `binary-raw-rr1000` | 146.57 ms | 165.02 ms | 0.8625 | 1.4 s | 9,504 MB |
| `binary-centered-rr1000` | 158.69 ms | 175.27 ms | 0.9004 | 1.4 s | 9,504 MB |

### Query embedding

| backend | seq | p50 | p99 | ANE residency |
|---|--:|--:|--:|--:|
| `coreml-ane-cpu` | 32 | 0.92 ms | 1.10 ms | 97% |
| `coreml-reference-cpu` | 32 | 0.94 ms | 2.03 ms | 0% |
| `coreml-ane-ane` | 32 | 0.98 ms | 2.17 ms | 97% |
| `coreml-reference-gpu` | 32 | 4.30 ms | 5.13 ms | 0% |
| `coreml-reference-ane` | 32 | 4.41 ms | 4.87 ms | 0% |
| `torch-cpu` | 32 | 4.62 ms | 5.15 ms | n/a |
| `torch-mps` | 32 | 5.33 ms | 16.07 ms | n/a |
| `coreml-ane-gpu` | 32 | 8.36 ms | 10.48 ms | 97% |
| `coreml-reference-ane` | 64 | 0.93 ms | 2.58 ms | 93% |
| `coreml-ane-ane` | 64 | 0.94 ms | 2.30 ms | 97% |
| `coreml-ane-cpu` | 64 | 1.40 ms | 1.75 ms | 97% |
| `coreml-reference-cpu` | 64 | 1.60 ms | 1.94 ms | 93% |
| `coreml-reference-gpu` | 64 | 4.47 ms | 5.26 ms | 93% |
| `coreml-ane-gpu` | 64 | 6.39 ms | 10.36 ms | 97% |
| `torch-cpu` | 64 | 6.97 ms | 7.45 ms | n/a |
| `torch-mps` | 64 | 7.63 ms | 28.66 ms | n/a |
| `coreml-ane-ane` | 128 | 0.93 ms | 4.26 ms | 97% |
| `coreml-reference-ane` | 128 | 1.17 ms | 4.89 ms | 93% |
| `coreml-reference-cpu` | 128 | 2.78 ms | 3.16 ms | 93% |
| `coreml-ane-cpu` | 128 | 3.02 ms | 3.51 ms | 97% |
| `coreml-reference-gpu` | 128 | 3.62 ms | 6.12 ms | 93% |
| `torch-mps` | 128 | 8.26 ms | 8.71 ms | n/a |
| `coreml-ane-gpu` | 128 | 8.58 ms | 10.41 ms | 97% |
| `torch-cpu` | 128 | 12.09 ms | 13.46 ms | n/a |
| `coreml-ane-ane` | 256 | 1.59 ms | 3.92 ms | 97% |
| `coreml-reference-gpu` | 256 | 2.72 ms | 3.75 ms | 93% |
| `coreml-reference-ane` | 256 | 2.74 ms | 4.45 ms | 93% |
| `coreml-ane-cpu` | 256 | 5.65 ms | 5.87 ms | 97% |
| `coreml-reference-cpu` | 256 | 5.73 ms | 5.99 ms | 93% |
| `torch-mps` | 256 | 7.69 ms | 10.48 ms | n/a |
| `coreml-ane-gpu` | 256 | 8.39 ms | 9.37 ms | 97% |
| `torch-cpu` | 256 | 15.76 ms | 17.65 ms | n/a |

### Background indexing vs foreground latency

Foreground query p50/p99 at n=600,000 while an indexer saturates cores at each QoS class.

| condition | query p50 | query p99 | vs idle | indexer work done |
|---|--:|--:|--:|--:|
| idle (no indexer) | 24.82 ms | 27.31 ms | 1.00x | 0 ops |
| indexer @ user-initiated | 33.72 ms | 37.10 ms | 1.36x | 29,247 ops |
| indexer @ default | 33.45 ms | 38.13 ms | 1.35x | 29,249 ops |
| indexer @ utility | 33.68 ms | 36.59 ms | 1.36x | 29,806 ops |
| indexer @ background | 25.10 ms | 53.90 ms | 1.01x | 3,612 ops |

### End-to-end budget

n=600,000, embedder `coreml-reference-ane`, seq_len=64.

| stage | p50 | p99 | share of e2e |
|---|--:|--:|--:|
| tokenise | 0.10 ms | 0.21 ms | 1% |
| embed | 0.91 ms | 2.58 ms | 6% |
| search | 12.03 ms | 18.39 ms | 85% |
| hydrate (sqlite) | 0.01 ms | 0.02 ms | 0% |
| END TO END | 14.14 ms | 15.84 ms |  |

End-to-end p99 is **15.84 ms**, within the 50 ms budget.

<!-- /RESULTS -->

> The ANE-residency column is a property of the converted *package*, measured
> once under `CPU_AND_NE`, so it repeats across the `-cpu` / `-gpu` / `-ane`
> rows of a variant. It describes where the graph *can* run, not where a given
> row ran.

## What the numbers say

**The 50ms budget is not tight.** At 600k vectors — roughly a decade of heavy
daily use — the whole pipeline runs in 14.1ms p50 / 15.8ms p99. Three times
headroom, on a laptop, with exact search.

**Search dominates the budget; embedding is a rounding error.** 85% search
versus 6% embedding. I predicted the reverse before measuring, and was wrong in
both directions: CoreML on the Neural Engine makes query embedding far cheaper
than expected (~0.9ms), while a single-query flat scan is slower than a
bandwidth estimate suggests, because Accelerate does not parallelise GEMV and
one core sees ~48 GB/s rather than the chip's ~400 GB/s.

**Threads are the highest-leverage optimisation, and they are eight lines.**
`flat-fp32-t8` is ~2x faster than `flat-fp32` at every scale. Not 8x — it is
memory-bandwidth bound, and 8 threads reach ~82 GB/s. Still the best
latency-per-unit-effort in the repo.

**Do not build an index below ~1M vectors.** A threaded flat scan is exact
(recall 1.0000), needs no build step, no maintenance, no incremental-update
story, and no extra memory. HNSW is ~30x faster per query but costs 60s of build
at 600k and — at default `ef=64` — collapses to **0.71 recall**, down from 0.98
at 60k. Fast retrieval of the wrong memories is worse than useless. If you do
go this route, tune `ef` and measure recall; the defaults lie at scale.

**Binary quantisation did not pay off, contrary to the usual advice.** 1-bit
codes are 32x smaller, but measured *slower* than the threaded flat scan at
every scale (16.1ms vs 12.3ms at 600k; 147ms vs 112ms at 6M) while giving up
exactness. The reason is that the small codes never become the working set: the
coarse Hamming scan is single-threaded, and reranking still gathers rows from
the full fp32 array, which has to stay resident. Worth revisiting only with a
threaded scan and a quantised rerank.

**Mean-centering before binarising helps, but only at scale.** Identical at 60k
(0.9992 both), then 0.9930 → 0.9969 at 600k and 0.8625 → 0.9004 at 6M. It is one
line of code and costs nothing, so it is worth having; it simply cannot be
validated on a small corpus.

**The real crossover is ~1M vectors.** At 6M the threaded flat scan takes 112ms
and blows the budget. That is where an index genuinely earns its complexity —
not before.

**The GPU is the wrong target, at every sequence length.** `coreml-*-gpu` runs
2.7-8.6ms against 0.93-1.59ms on the ANE, and `torch-mps` sits at 5-8ms with a
p99 as high as 28.7ms, nearly flat across sequence length — the signature of
dispatch overhead rather than compute. A 22M-parameter model at batch 1 cannot
fill a 32-core GPU. Using it costs power, heat, and contention with the user's
actual work, and buys nothing.

**Apple's ANE layout matters most where you would least expect it.** The
`ane` variant holds 97% Neural Engine residency at every sequence length. The
conventional `reference` layout holds 93% at seq_len 64-256 but **fails to
compile for the ANE entirely at seq_len 32** (`ANECCompile() FAILED`), falling
back to 100% CPU and taking **4.5x** longer (4.41ms vs 0.98ms). Short queries
are the common case for a memory system, so the layout rewrite pays off
precisely in the regime that matters. At seq_len 256 the ANE layout is 1.73x
faster (1.59ms vs 2.74ms); at seq_len 64 the two are indistinguishable.

**CPU and ANE are close at short sequences; the ANE pulls away as they grow.**
At seq_len 32 `coreml-ane-cpu` runs 0.92ms against 0.98ms on the ANE — a 6%
edge, well inside run-to-run variation, so treat them as tied. By seq_len 128
the ANE is 3.3x faster and by 256 it is 3.6x. The ANE results are also visibly
bimodal (min 0.82ms, p50 0.98ms, p90 2.07ms), which is power-state transition
rather than noise, and it is why the p90 column is worth reading here. If your
queries are consistently very short, measure both; otherwise use the ANE.

> An earlier run of this benchmark showed CPU beating the ANE 2:1 at seq_len 32
> (0.92ms vs 1.86ms). That gap did not survive re-measurement once `TorchBackend`
> was switched to the same graph as the CoreML packages and the sweep was rerun —
> it collapsed to 6%. The ANE's bimodal timing makes single runs at short
> sequence lengths unreliable; the 4.5x reference-layout gap above reproduced
> across both runs and is the more robust finding.

**Milestone 3 is a scheduling problem and it is already solved.** A background
indexer at default QoS inflates foreground query latency by 1.36x. The same
indexer at `QOS_CLASS_BACKGROUND` inflates it by **1.01x** — macOS confines the
thread to the efficiency cores. The cost is real and worth stating: the indexer
completes 8x less work (3,612 vs 29,249 ops), and foreground p99 gets *worse*
(53.9ms vs 27.3ms idle), so occasional stalls remain even when the median is
untouched. Note that `utility` behaves like `default`, not like `background` —
only `QOS_CLASS_BACKGROUND` gets E-core confinement. This is one `ctypes` call
(`bench/qos.py`), not SIMD and not a rewrite.

**Text hydration is free.** 0.012ms against a warm 600k-row SQLite store — 0.1%
of the budget. Measured because it is usually omitted, and it turns out not to
matter.

### What this does not measure

Retrieval *quality* — whether the returned memories are the right ones — is the
hard, unsolved part of this problem, and nothing here addresses it. These are
synthetic vectors with known ground truth, so recall@10 measures whether a
backend finds the true nearest neighbours, not whether nearest-neighbour
retrieval surfaces what a user actually meant. Milestone 2 (compression) is
likewise unmeasured: it belongs at write time and idle time, never in the query
path, where a single summarisation call would cost 500ms-5s and blow this entire
budget a hundred times over.

## What the harness measures

| Command | Question |
|---|---|
| `search` | Flat scan vs binary quantisation vs HNSW, with recall and build cost |
| `embed` | PyTorch CPU/MPS vs CoreML CPU/GPU/ANE, with real ANE residency |
| `contention` | Does background indexing make foreground queries slow? |
| `pipeline` | Full budget: tokenise → embed → search → hydrate text |

Two encoder implementations are compared, both loading the same MiniLM weights
and verified against HuggingFace to float32 roundoff (1e-7):

* `reference` — conventional `(B, S, C)` layout with `nn.Linear`.
* `ane` — Apple's `ml-ane-transformers` layout: `(B, C, 1, S)`, 1×1 `Conv2d`
  instead of `Linear`, per-head split attention, channel-axis LayerNorm.

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
```

Convert the encoders to CoreML (needed for the ANE paths):

```bash
.venv/bin/python scripts/convert_coreml.py --out models --report-plan
```

Then run any subcommand:

```bash
.venv/bin/python -m bench.cli all
```

Individual runs, if you want to skip the 6M scale:

```bash
.venv/bin/python -m bench.cli search --scales 60000 600000 --iters 60
```

Results land in `results/*.json`, each tagged with the machine fingerprint. A
latency number without its hardware is not a result.

## Methodology notes

Benchmarks mostly lie by accident. Four bugs found while building this one, all
of which produced *plausible* numbers, are documented in the source where they
occurred:

**Synthetic corpora must not be uniform on the sphere.** Uniform random vectors
are the pathological worst case: ANN indexes look terrible because nothing
clusters, and quantisation looks perfect because there is no structure to
destroy. Both errors flatter the conclusion this repo happens to reach, so the
generator produces clustered, *anisotropic* data instead — real embedding spaces
are cones, not spheres. See `bench/corpus.py`.

**High-dimensional noise has norm √d.** "Add 0.35 of noise" to a unit vector in
384 dimensions adds noise of norm ~6.9, not 0.35 — burying the signal and
producing queries with cosine ~0.14 to their own source document. The first run
of this benchmark measured recall against effectively random ground truth and
reported confident numbers for all of it.

**A float64 query silently upcasts a float32 index.** `np.sqrt()` returns a
float64 numpy scalar, which under NEP 50 is not "weak" — it promotes the whole
array. numpy then upcasts the entire index on *every search*: 5× slower
(9.8ms → 1.9ms at n=60k) with recall unchanged at 1.0000, so nothing in the
output revealed it. Now asserted in `bench/cli.py`.

**int8 dot products overflow int16.** A 384-dim int8 dot product reaches ~6.2e6;
int16 saturates at 32767. Every score clipped, ranking correlating **−0.32**
with truth — and the search still returned plausible ids at plausible speed,
with recall quietly at 0.02.

The measurement primitives in `bench/timing.py` discard warmup iterations,
report p50/p90/p99 rather than a mean, and record the empirical timer floor.
Index build time and resident bytes are reported alongside query latency,
because an index that answers in 0.4ms but takes minutes to build and cannot be
maintained incrementally is not free.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Deliberately permissive: a benchmark's value is in being run, cited, and
reproduced on hardware other than mine. Restricting that would cost reach and
buy nothing.

Results from other machines are the most useful contribution you can make —
particularly non-Max Apple Silicon (base M-series have narrower memory buses,
which should move the flat-scan numbers), and Intel or CUDA hosts, where the
Neural Engine findings do not apply at all.
