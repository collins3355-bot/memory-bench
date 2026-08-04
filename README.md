# memory-bench

Where does the latency budget of a local AI long-term-memory system actually go?

This is a measurement harness, not a memory system. It exists to answer one
question with numbers instead of intuition: if you want an AI assistant that
remembers everything you have ever said to it, runs locally, and answers in
under 50ms without turning your laptop into a hairdryer — which part is
actually hard?

The short answer, on the hardware below: **the budget is comfortable, and almost
all of it goes to the vector scan — but the fix is eight threads, not a vector
database.** And on the follow-up question — does fast retrieval find the *right*
memories? — the quality eval below says: a retrieval-trained encoder over small
chunks reaches 0.91 evidence-complete@10 on LongMemEval-S, retrieving *wider*
(k=50 over turn-sized chunks) solves even multi-session assembly at 0.98, a
22M-parameter cross-encoder rerank compresses that wide net back to 0.93
complete@10 in a ~2.3k-token context, and naive recency weighting is actively
destructive with every encoder tested. What survives as genuinely open:
multi-hop questions, and running the reranker inside the latency budget.

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
| Perf model | `all-MiniLM-L6-v2` — 22.7M params, 384 dims, 6 layers |
| Quality encoders | MiniLM-L6 and `e5-small-v2` (33M, 12 layers, query/passage prefixes) |

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

### Retrieval quality — longmemeval_s

LongMemEval-S: 500 questions, ~50-session haystacks (~115k tokens each). 500 instances, 30 abstention (excluded from means). Encoder: MiniLM-L6, the same model as the perf benchmarks.

| chunking | arm | sess_r@10 | complete@10 | turn_r@10 | MRR | tokens@10 |
|---|---|--:|--:|--:|--:|--:|
| turn | `bm25` | 0.905 | 0.830 | 0.754 | 0.894 | 2,240 |
| turn | `vector` | 0.909 | 0.832 | 0.734 | 0.903 | 2,153 |
| turn | `hybrid` | 0.945 | 0.881 | 0.813 | 0.921 | 2,359 |
| turn | `vector_time` | 0.513 | 0.300 | 0.387 | 0.439 | 2,085 |
| turn | `hybrid_time` | 0.916 | 0.834 | 0.785 | 0.836 | 2,333 |
| window | `bm25` | 0.897 | 0.809 | 0.841 | 0.885 | 10,307 |
| window | `vector` | 0.931 | 0.868 | 0.849 | 0.900 | 9,877 |
| window | `hybrid` | 0.946 | 0.887 | 0.888 | 0.920 | 10,270 |
| window | `vector_time` | 0.522 | 0.302 | 0.466 | 0.446 | 9,497 |
| window | `hybrid_time` | 0.914 | 0.823 | 0.857 | 0.763 | 10,234 |
| session | `bm25` | 0.949 | 0.902 | 0.953 | 0.902 | 28,536 |
| session | `vector` | 0.928 | 0.883 | 0.924 | 0.834 | 26,009 |
| session | `hybrid` | 0.964 | 0.932 | 0.964 | 0.903 | 28,549 |
| session | `vector_time` | 0.723 | 0.536 | 0.726 | 0.518 | 25,105 |
| session | `hybrid_time` | 0.954 | 0.909 | 0.957 | 0.748 | 28,011 |

**complete@10 by question type** (chunking: `session`)

| arm | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|--:|--:|--:|--:|--:|--:|
| `bm25` | 0.986 | 0.818 | 1.000 | 0.867 | 1.000 | 0.850 |
| `vector` | 0.875 | 0.860 | 0.982 | 0.967 | 0.891 | 0.843 |
| `hybrid` | 0.972 | 0.876 | 0.982 | 0.967 | 0.984 | 0.906 |
| `vector_time` | 0.403 | 0.397 | 0.786 | 0.800 | 0.750 | 0.465 |
| `hybrid_time` | 0.944 | 0.835 | 1.000 | 0.933 | 0.984 | 0.874 |

### Retrieval quality — locomo

LoCoMo: 10 long conversations, ~2k questions with per-turn evidence labels. 1986 instances, 455 abstention (excluded from means). Encoder: MiniLM-L6, the same model as the perf benchmarks.

| chunking | arm | sess_r@10 | complete@10 | turn_r@10 | MRR | tokens@10 |
|---|---|--:|--:|--:|--:|--:|
| turn | `bm25` | 0.821 | 0.759 | 0.512 | 0.664 | 319 |
| turn | `vector` | 0.746 | 0.690 | 0.455 | 0.536 | 271 |
| turn | `hybrid` | 0.845 | 0.790 | 0.561 | 0.652 | 300 |
| turn | `vector_time` | 0.345 | 0.310 | 0.207 | 0.175 | 276 |
| turn | `hybrid_time` | 0.762 | 0.705 | 0.498 | 0.484 | 303 |
| window | `bm25` | 0.828 | 0.771 | 0.752 | 0.710 | 1,326 |
| window | `vector` | 0.747 | 0.690 | 0.634 | 0.574 | 1,146 |
| window | `hybrid` | 0.837 | 0.779 | 0.759 | 0.695 | 1,245 |
| window | `vector_time` | 0.359 | 0.320 | 0.304 | 0.199 | 1,163 |
| window | `hybrid_time` | 0.767 | 0.711 | 0.697 | 0.484 | 1,260 |
| session | `bm25` | 0.886 | 0.826 | 0.887 | 0.721 | 7,224 |
| session | `vector` | 0.619 | 0.556 | 0.620 | 0.406 | 6,567 |
| session | `hybrid` | 0.851 | 0.798 | 0.851 | 0.550 | 6,883 |
| session | `vector_time` | 0.557 | 0.500 | 0.557 | 0.269 | 6,862 |
| session | `hybrid_time` | 0.774 | 0.715 | 0.775 | 0.436 | 7,023 |

**complete@10 by question type** (chunking: `session`)

| arm | locomo-cat1 | locomo-cat2 | locomo-cat3 | locomo-cat4 |
|---|--:|--:|--:|--:|
| `bm25` | 0.413 | 0.887 | 0.551 | 0.970 |
| `vector` | 0.299 | 0.691 | 0.348 | 0.614 |
| `hybrid` | 0.470 | 0.878 | 0.506 | 0.908 |
| `vector_time` | 0.210 | 0.575 | 0.213 | 0.598 |
| `hybrid_time` | 0.335 | 0.800 | 0.404 | 0.843 |

### Does retrieving more fix the hard questions? (k-sweep)

complete@k for the `hybrid` arm on longmemeval_s, overall and for the hardest type (multi-session), as k grows.

| chunking | slice | c@5 | c@10 | c@20 | c@50 | tokens@50 |
|---|---|--:|--:|--:|--:|--:|
| turn | overall | 0.774 | 0.881 | 0.943 | 0.981 | 11,465 |
| turn | multi-session | 0.579 | 0.785 | 0.901 | 0.975 | 12,142 |
| window | overall | 0.777 | 0.887 | 0.938 | 0.987 | 48,660 |
| window | multi-session | 0.529 | 0.793 | 0.884 | 0.967 | 49,450 |
| session | overall | 0.860 | 0.932 | 0.981 | 1.000 | 109,649 |
| session | multi-session | 0.777 | 0.876 | 0.959 | 1.000 | 109,926 |

### Cross-encoder rerank — compressing k=50 into k=5-10

First stage retrieves 50 turn chunks; `ms-marco-MiniLM-L6` reranks them. `ceiling` is the base arm's complete@50 — the recall available to the reranker. Δ@5 is reranked minus base at complete@5.

| dataset / encoder | base arm | c@5 → +ce | Δ@5 | c@10 → +ce | ceiling | tok@10 → +ce |
|---|---|--:|--:|--:|--:|--:|
| LongMemEval-S / MiniLM | `vector` | 0.681 → **0.817** | +0.136 | 0.832 → **0.913** | 0.972 | 2,153 → 2,203 |
| LongMemEval-S / MiniLM | `hybrid` | 0.774 → **0.828** | +0.053 | 0.881 → **0.921** | 0.981 | 2,359 → 2,250 |
| LongMemEval-S / e5-small | `vector` | 0.802 → **0.836** | +0.034 | 0.906 → **0.923** | 0.985 | 1,963 → 2,091 |
| LongMemEval-S / e5-small | `hybrid` | 0.815 → **0.830** | +0.015 | 0.898 → **0.928** | 0.991 | 2,222 → 2,200 |
| LoCoMo / MiniLM | `vector` | 0.572 → **0.722** | +0.151 | 0.690 → **0.782** | 0.927 | 271 → 317 |
| LoCoMo / MiniLM | `hybrid` | 0.681 → **0.775** | +0.095 | 0.790 → **0.839** | 0.952 | 300 → 344 |
| LoCoMo / e5-small | `vector` | 0.745 → **0.784** | +0.039 | 0.817 → **0.848** | 0.975 | 387 → 381 |
| LoCoMo / e5-small | `hybrid` | 0.748 → **0.788** | +0.040 | 0.828 → **0.849** | 0.964 | 372 → 381 |

### Encoder sweep — longmemeval_s

Same grid, dense encoder swapped: MiniLM-L6 (22.7M, mean-pool) vs e5-small-v2 (33M, 12 layers, asymmetric query/passage prefixes). complete@10; Δ is e5 minus MiniLM.

| chunking | arm | MiniLM | e5-small | Δ |
|---|---|--:|--:|--:|
| turn | `vector` | 0.832 | 0.906 | +0.074 |
| turn | `hybrid` | 0.881 | 0.898 | +0.017 |
| window | `vector` | 0.868 | 0.891 | +0.023 |
| window | `hybrid` | 0.887 | 0.866 | -0.021 |
| session | `vector` | 0.883 | 0.877 | -0.006 |
| session | `hybrid` | 0.932 | 0.947 | +0.015 |

### Encoder sweep — locomo

Same grid, dense encoder swapped: MiniLM-L6 (22.7M, mean-pool) vs e5-small-v2 (33M, 12 layers, asymmetric query/passage prefixes). complete@10; Δ is e5 minus MiniLM.

| chunking | arm | MiniLM | e5-small | Δ |
|---|---|--:|--:|--:|
| turn | `vector` | 0.690 | 0.817 | +0.127 |
| turn | `hybrid` | 0.790 | 0.828 | +0.037 |
| window | `vector` | 0.690 | 0.813 | +0.123 |
| window | `hybrid` | 0.779 | 0.812 | +0.033 |
| session | `vector` | 0.556 | 0.586 | +0.029 |
| session | `hybrid` | 0.798 | 0.805 | +0.007 |

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

## What the quality numbers say

The perf section above answers "how fast"; `bench/quality/` answers whether
retrieval finds the *labelled evidence* on LongMemEval-S and LoCoMo. Scores are
evidence recall, and the strictest column — complete@10, all evidence sessions
present in the top 10 chunks — is the one quoted below unless stated otherwise.

**Fusion failures are a small-k phenomenon.** With the weaker encoder
(MiniLM), hybrid RRF beats both parents at every LongMemEval-S chunking
(0.881 / 0.887 / 0.932 vs 0.832-0.902 for the best single retriever). With
clearly unequal parents at k≤10, fusion is fragile — it costs recall against
the stronger e5 parent on LongMemEval-S turn chunks (0.898 fused vs 0.906
dense-only at k=10), and the truncation-crippled dense parent drags BM25 down
on LoCoMo sessions (0.798 vs 0.826 at k=10; a 17-point gap at complete@5) —
though not reliably: on LoCoMo turn chunks e5 outguns BM25 by a similar margin
and fusion still helps (0.828). What *is* consistent is the k-dependence: by
k=20, hybrid beats both parents at every operating point in the grid but one
(on LoCoMo turn chunks with e5, dense-only stays ahead by a noise-level
0.003), including over the broken parent (LoCoMo session: 0.942 vs BM25's
0.931) and over the dominant one on LongMemEval-S (e5 turn: 0.955 vs 0.949,
extending to 0.991 vs 0.985 at k=50). Practical rule: *at small k, fusion
needs validating against the stronger parent; at k≥20 it is close to safe to
fuse everything measured here — except recency, below.*

**"Dense-only never wins" was mostly an artifact of the weak encoder — but the
chunk-shape constraint is not.** With MiniLM, vectors took first place at
almost nothing (one cell: window complete@20). e5-small-v2 (33M params, still
tiny) changes the picture: at LongMemEval-S turn granularity it is the best
k=10 arm — 0.906 complete@10 at ~2.0k retrieved tokens, beating every MiniLM
configuration at turn or window chunking (hybrid retakes the lead at other
k) — and e5+hybrid at session granularity sets the overall ceiling (0.947).
The encoder swap also does half the k-sweep's job at fixed cost: multi-session
at k=10 jumps 0.785 (MiniLM hybrid) → 0.876 (e5 dense-only) with *fewer*
tokens. What does *not* flip is the truncation collapse: on LoCoMo session
chunks e5 still loses to BM25 by 24 points (0.586 vs 0.826), because a
256-token encoder window cannot represent a long session no matter how well it
was trained. Small encoders want small chunks; that constraint survives the
encoder upgrade. The cost side: e5 is 12 layers to MiniLM's 6, roughly double
the encode time — a trade the perf harness can now quantify on the ANE.

**Un-gated recency fusion is a catastrophe, including on the questions it was
meant to help.** Giving recency an equal RRF vote (`vector_time`) drops
complete@10 from 0.832 to 0.300 overall — and on *knowledge-update* questions,
the type recency intuitively targets, from **0.972 to 0.097**. The mechanism is
visible in the labels: knowledge-update questions average exactly 2.0 evidence
sessions (the stale fact and its update), and a "newest first" vote starves the
older half of the evidence pair of rank. Distractor flooding is not even
required: on the oracle split, where *only* evidence sessions are indexed,
recency fusion still drops complete@5 from 0.750 to 0.475. Diluting recency to
1/3 of the vote (`hybrid_time`) does not rescue it: it never beats plain
`hybrid` on any overall metric at any chunking with either encoder — verified
exhaustively — and at the top of the ranking it loses to plain BM25 on
complete@5 (0.555-0.715 vs 0.713-0.834 across chunkings) and MRR, recovering
only at deeper k. And unlike every other pathology in this section, retrieving
wider does not fix it. The result is encoder-robust: with e5, `vector_time`
still craters to 0.315 overall and **0.083** on knowledge-update. The design
conclusion is sharp: temporal signals must be *gated by the query* (detect that
a question is temporal, then apply time logic), never fused unconditionally.
This contradicts the intuitive advice — "add recency weighting" — that this
project started with; measured, it subtracts, with every encoder tested.

**Retrieving wider dissolves most of the "hard question" problem — width over
coarseness is the architecture.** At k=10, multi-session (0.785) and
temporal-reasoning (0.787) look like open research problems. The k-sweep says
otherwise: with MiniLM hybrid over turn chunks, multi-session climbs
0.579 → 0.785 → 0.901 → **0.975** at k=5/10/20/50, and temporal-reasoning
reaches 0.961 at k=50 — inside ~11.5k retrieved tokens. Even LoCoMo's brutal
multi-hop slice (0.377 complete@10, the hardest number in either benchmark)
rises to 0.819 at k=50 with MiniLM and 0.865 with e5. Width also rescues
fusion: the small-k fusion failures above all invert by k=20, leaving un-gated
recency as the *only* pathology retrieving wider cannot fix. Coarser chunks
were never the answer: turn chunks at
k=20 beat session chunks at k=10 on *both* recall (0.943 vs 0.932) *and*
tokens (4.9k vs 28.5k). Retrieval recall is largely a solved problem if you
retrieve wide over small chunks; what k=50 leaves unsolved is that 11.5k tokens
is still too much to hand an LLM per query. **The open problem in local memory
is the precision stage — reranking ~50 wide-net candidates down to a small
context — plus query-gated time logic.** That is where a memory engine gets to
differentiate.

**A tiny cross-encoder converts wide-net recall into small-context precision.**
Reranking the top-50 turn chunks with `ms-marco-MiniLM-L6` (22.7M params, the
same size class as the retrievers) improves every overall metric in all eight
configurations measured (vector and hybrid bases × two encoders × two
datasets; BM25 bases and coarser chunkings were not given a reranker). On
LongMemEval-S with the MiniLM first stage: complete@5 goes 0.774 → 0.828
within ~1k retrieved tokens, and complete@10 goes 0.881 → 0.921 within ~2.3k —
recovering ~40% of the gap between the base ranking and the 0.981 ceiling the
wide net makes available. With e5, hybrid+ce50 posts the best LongMemEval-S
number on the board, **0.928 complete@10** (on LoCoMo, vector+ce50 is the
better e5 arm past k=10). The largest wins are where first-stage ranking was
weakest — LoCoMo MRR jumps 0.652 → 0.806 — and knowledge-update closes from
0.972 to a clean **1.000 complete@10 with both encoders** (plain retrieval had
already recovered the update/stale-fact pairs recency fusion buried; the CE
finishes the last two questions). "Improves every overall metric" is not
"free", though: two question types consistently regress — LoCoMo's
open-domain cat3 (0.573 → 0.528 with e5, same direction in all four cells)
and single-session-preference MRR — so a shipped reranker wants per-type
evaluation, not just the aggregate.

**The reranker nearly erases the encoder gap — which changes what to ship.**
MiniLM+ce50 beats *plain e5* on both datasets (0.921 vs 0.898 on
LongMemEval-S; 0.839 vs 0.828 on LoCoMo), and trails e5+ce50 by under a point.
The +7.4-point encoder upgrade the sweep celebrated shrinks to +0.6 once both
sides get a reranker. Engine implication: if you ship a rerank stage, the
first-stage encoder barely matters — and MiniLM is the encoder with the
validated 0.9ms ANE path. Spend the model budget on the reranker, not the
retriever. Two costs temper the whole rerank story. Multi-hop is *not* a
ranking problem: LoCoMo cat1 improves (0.377 → 0.473 at k=10 with MiniLM) but
stays the worst number on the board — those questions need query
decomposition, not better scoring. And the latency bill is real: a batch of
50 pairs costs 54.5ms in eager PyTorch on CPU — alone exceeding the 50ms
budget — or 17.6ms on MPS. The reranker is a 6-layer BERT, the exact
architecture class this repo already converts to the Neural Engine at ~7-13x
eager-CPU speed at these sequence lengths, so CE-on-ANE is the obvious next
perf milestone rather than an open question.

**LoCoMo is the stress test, not an echo.** Its shape (10 very long two-speaker
histories, per-turn evidence, 455 of 1,986 questions adversarial and excluded)
is what surfaced the broken-parent fusion failure at session granularity and
the multi-hop floor (0.377 at k=10 — recoverable to 0.865 at k=50 under e5,
but still the worst wide-retrieval residual anywhere). Its noise-level window
result (hybrid 0.779 vs BM25 0.771) is also a reminder that not every tabled
difference is a finding.

### What this still does not measure

Answer accuracy (whether an LLM given the retrieved chunks answers correctly) —
evidence recall is the retrieval system's own scoreboard, deliberately isolated
from the downstream model. The remaining arms the results point at: query
decomposition for multi-hop (the one question type reranking cannot fix),
query-gated time logic, and the cross-encoder's CoreML/ANE conversion so the
rerank stage fits the latency budget; the harness embeds ~300k chunks and
caches ~200k reranker pair scores by content hash, so each additional arm
costs seconds, not hours. Milestone 2 (compression/consolidation) remains
unmeasured: it belongs
at write time and idle time, never in the query path, where a single
summarisation call would cost 500ms-5s and blow the entire latency budget a
hundred times over.

## What the harness measures

| Command | Question |
|---|---|
| `search` | Flat scan vs binary quantisation vs HNSW, with recall and build cost |
| `embed` | PyTorch CPU/MPS vs CoreML CPU/GPU/ANE, with real ANE residency |
| `contention` | Does background indexing make foreground queries slow? |
| `pipeline` | Full budget: tokenise → embed → search → hydrate text |
| `quality` | Does retrieval surface the *right* memories? (separate module, below) |

### Retrieval quality (`bench/quality/`)

The perf harness answers "how fast"; this answers the harder question it
deliberately left open — whether nearest-neighbour retrieval finds the labelled
evidence on real assistant-memory benchmarks:

* **LongMemEval** (cleaned) — 500 questions over ~115k-token multi-session chat
  histories, evidence labelled at session *and turn* level, with question types
  that map to memory-system failure modes (multi-session assembly, temporal
  reasoning, knowledge updates, abstention).
* **LoCoMo** — ~2k questions over 10 very long two-speaker conversations with
  per-dialog-turn evidence.

Experimental grid: chunking (turn / 4-turn window / session) × retrieval arm
(BM25 / dense MiniLM vectors / RRF hybrid / recency-fused variants). Fusion is
reciprocal-rank throughout because RRF is parameter-free — with only 500 eval
questions, any tuned mixing weight or decay rate would be fitted to the test
set. Scored on evidence recall (session and turn), completeness (all evidence
present — the bar that matters for multi-session questions), MRR, and tokens
retrieved at k (the context-window cost the arm imposes downstream).

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

The quality eval fetches its datasets (LongMemEval-cleaned ~293 MB, LoCoMo
~3 MB) and then runs the full grid; embeddings are cached by content hash, so
only the first run pays the encoding cost:

```bash
.venv/bin/python scripts/fetch_quality_data.py
.venv/bin/python -m bench.quality.run --dataset longmemeval_s
.venv/bin/python -m bench.quality.run --dataset locomo
```

Tests (each one is a regression guard for a bug that actually happened):

```bash
.venv/bin/python tests/test_quality.py
```

Results land in `results/*.json`, each tagged with the machine fingerprint. A
latency number without its hardware is not a result.

## Methodology notes

Benchmarks mostly lie by accident. Six bugs found while building this one, all
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

**Tie blocks turn rank fusion into index-position bias.** Plain `argsort` ranks
give tied scores index-order positions, and in RRF those arbitrary positions
become real score differences. This is not a corner case: most chunks share no
term with a given query, so the majority of the BM25 list is one tie block at
0.0 — and chunks early in the haystack systematically outranked identical-scored
later ones in every fused arm. Caught by a unit test in which a fresher chunk
with an identical content score lost to a staler one. Fix: tied items share
their mean rank (`bench/quality/retrieve.py`), verified by a corpus-order-
invariance test.

**`inf − inf` is NaN, which un-ties the one block that must stay tied.** Tie
boundaries detected via `np.diff` break on the all-`inf` age block that undated
chunks form, silently reintroducing the same index bias for exactly the chunks
a recency vote must not discriminate among. Boundaries are detected by equality
instead.

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
