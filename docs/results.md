# Results

## Overview

The repository includes stored evaluation artifacts for multiple retrieval scales.

These are useful because they show:

- the system was actually executed
- retrieval quality was measured
- latency was observed
- explanation cost is visible separately from search cost

## Retrieval Results

### 1k image run

Source:

- `benchmarks/run_1000_metrics_eval.json`

Results:

- sample size: 200 queries
- average query latency: 6.92 ms
- recall@1: 0.46
- recall@5: 0.755
- recall@10: 0.88

### 5k image run

Source:

- `benchmarks/run_5000_metrics_eval.json`

Results:

- sample size: 200 queries
- average query latency: 6.9 ms
- recall@1: 0.315
- recall@5: 0.48
- recall@10: 0.57

## Explanation Cost

Source:

- `benchmarks/run_5000_metrics_summary.json`

Results:

- logged events: 209
- average search time: 68.07 ms
- average LLM time: 13737.42 ms

## Interpretation

The benchmark pattern is useful for discussion:

- retrieval remains fast even as the index scales
- retrieval quality declines as the search space grows
- the explanation layer becomes the main UX bottleneck

That tradeoff is valuable because it turns the demo into a systems discussion, not only a model demo.
