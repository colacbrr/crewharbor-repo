# crewharbor-repo

This is the showcase project built from the original `Job-offer/tem-project` codebase.

It keeps the strongest parts of that system in a cleaner presentation format for job applications:

- CLIP-based multimodal embeddings
- FAISS vector indexing
- semantic text-to-image retrieval
- caption-aware reranking
- grounded explanation generation through a local LLM
- benchmark and metrics reporting

The goal is not to publish the whole academic workspace. The goal is to show a clear, runnable engineering project.

## What The System Does

Given a natural-language query such as `a dog playing in the park`, the system:

1. encodes the query into the same semantic space as the indexed images
2. retrieves the nearest visual matches through FAISS
3. optionally reranks results using caption information
4. generates a grounded explanation from the top retrieved items

This makes the system more useful than pure vector search alone because it adds a traceable explanation layer on top of retrieval.

## Architecture

The core pipeline has four layers:

- Embedding layer: CLIP encodes text and images into a shared vector space
- Retrieval layer: FAISS serves fast nearest-neighbor search
- Explanation layer: a local Ollama-based LLM generates grounded responses
- Evaluation layer: benchmark artifacts and runtime metrics are recorded

More detail:

- [Architecture](docs/architecture.md)
- [Demo Guide](docs/demo-guide.md)
- [Startup](docs/startup.md)
- [Results](docs/results.md)
- [Workflow](docs/workflow.md)
- [Research Notes](docs/research-notes.md)

## Stack

- Python
- FastAPI
- CLIP
- FAISS
- Ollama
- React
- Vite

## Repository Layout

```text
backend/
frontend/
docs/
assets/
benchmarks/
```

Source material remains available in `../tem-project/` and `../TEM/`, but those folders are intentionally kept outside this repo's main story.

## Benchmark Snapshot

The repository includes stored benchmark artifacts for both smaller and larger index sizes.

### 1k image run

- average query latency: 6.92 ms
- recall@1: 0.46
- recall@5: 0.755
- recall@10: 0.88

### 5k image run

- average query latency: 6.9 ms
- recall@1: 0.315
- recall@5: 0.48
- recall@10: 0.57

### Explanation summary

- average search time: 68.07 ms
- average LLM time: 13737.42 ms

These numbers are useful because they show both retrieval quality and the latency tradeoff introduced by the explanation layer.

## Current Scope

Implemented:

- image retrieval
- text-to-image search
- reranking
- grounded explanation
- benchmark logging

Planned:

- video retrieval
- audio retrieval
- stronger reranking strategies
- cloud comparison experiments

## Screenshots

The `assets/` directory contains interface screenshots and the current architecture diagram.

## Positioning

This repo is strongest when presented as:

`A semantic multimedia retrieval system with a local explanation layer, benchmark artifacts, and a modular path toward production-scale retrieval workflows.`
