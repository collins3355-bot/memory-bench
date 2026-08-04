"""Retrieval *quality* on real assistant-memory benchmarks.

The perf harness (bench/) established that speed is a solved problem at
personal-memory scale: 14ms end-to-end against a 50ms budget. It also
established what it deliberately did not measure: whether nearest-neighbour
search over embeddings surfaces the memories a user actually meant. recall@10
against synthetic ground truth answers "did you find the closest vectors",
which is a different question from "did you find the evidence".

This package measures the second question, on two published benchmarks with
human-labelled evidence:

  * **LongMemEval** (Wu et al., 2024) -- 500 questions over multi-session chat
    histories, labelled with the evidence sessions *and turns* required to
    answer. Question types cover the failure modes that matter for a memory
    system: multi-session assembly, temporal reasoning, knowledge updates,
    abstention.
  * **LoCoMo** (Maharana et al., 2024) -- very long two-speaker conversations
    with per-dialog-turn evidence labels.

The unit under test is the *retrieval arm*: BM25, dense vectors (the same
MiniLM encoder the perf harness validated), reciprocal-rank fusion of the two,
and recency-aware variants. The output is evidence recall -- not answer
accuracy, which would measure the downstream LLM as much as the retrieval.
"""
