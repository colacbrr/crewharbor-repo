# Semantic Multimedia Retrieval with Grounded Explanations

This project is a local-first multimedia retrieval system that turns natural-language queries into image search results, then generates a grounded explanation over the retrieved evidence.

It combines:

- CLIP embeddings for shared text-image semantic space
- FAISS indexing for fast nearest-neighbor retrieval
- caption-aware reranking for better result ordering
- a local Ollama explanation layer over retrieved items
- benchmark artifacts for latency and retrieval quality analysis

## What Problem This Solves

Traditional keyword search is a poor fit for images. Users ask for meaning, scenes, actions, and concepts, while image files usually expose only weak metadata.

This project addresses that semantic gap by:

1. encoding text queries and images into the same embedding space
2. retrieving semantically similar images with vector search
3. optionally reranking results using caption similarity
4. generating an explanation from the retrieved evidence rather than from free-form hallucinated context

The result is a system that is more useful than raw vector search alone because it returns both ranked matches and a grounded explanation of what was found.

## What The System Does

Given a query such as `a dog playing in the park`, the system:

1. encodes the query with CLIP
2. searches a FAISS index built from image embeddings
3. attaches captions and metadata to the retrieved images
4. optionally reranks results using caption embeddings
5. sends the final retrieved context to a local Ollama model
6. returns a structured explanation with a summary, uncertainty hint, and referenced retrieved items

## Why This Is Interesting

This is not only a UI demo. It is an end-to-end retrieval workflow with measurable tradeoffs:

- retrieval is fast enough for interactive use
- explanation is slower and becomes the dominant latency cost
- explanation quality depends directly on retrieval quality and caption coverage
- the system is modular enough to extend toward richer multimodal pipelines

## Architecture

```text
query
  -> CLIP text encoder
  -> FAISS vector retrieval
  -> optional caption reranking
  -> retrieved context assembly
  -> local Ollama explanation
  -> structured response + metrics logging
```

Main layers:

- Embedding layer: CLIP text and image encoders
- Retrieval layer: FAISS flat or HNSW index
- Reranking layer: caption similarity fusion
- Explanation layer: local retrieval-grounded generation
- Evaluation layer: stored benchmarks and runtime metrics

More detail:

- [Architecture](docs/architecture.md)
- [Startup](docs/startup.md)
- [Demo Guide](docs/demo-guide.md)
- [Results](docs/results.md)
- [Workflow](docs/workflow.md)
- [Research Notes](docs/research-notes.md)

## Current Scope

Implemented:

- image retrieval
- text-to-image semantic search
- FAISS flat and HNSW support
- caption-aware reranking
- local explanation generation with Ollama
- prompt versioning
- explanation caching
- fallback handling for malformed model output
- benchmark and runtime metrics

Planned:

- video retrieval
- audio retrieval
- richer hybrid search
- stronger reranking
- broader evaluation

## Complexity and Tradeoffs

This project sits at the intersection of multimodal retrieval and generation, so its complexity is mostly systems complexity rather than UI complexity.

Main complexity points:

- embedding generation is compute-heavy and front-loaded
- vector retrieval is fast but quality-sensitive to encoder choice and index size
- caption reranking improves precision but adds extra inference work
- explanation depends on retrieved context quality
- local LLM generation is much slower than retrieval and needs careful timeout, caching, and fallback behavior

The explanation layer is retrieval-grounded, but it is not a full document-chunk RAG system yet. There is no separate text chunk store, no citation graph, and no long-context retrieval stage.

## Benchmark Snapshot

Stored benchmark artifacts currently show:

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

These numbers matter because they make the tradeoff visible: retrieval remains interactive, while generation is the expensive stage.

## Repository Layout

```text
backend/     FastAPI retrieval and explanation service
frontend/    React + Vite interface
docs/        architecture, startup, results, workflow
assets/      screenshots and diagrams
benchmarks/  stored evaluation outputs
tests/       focused helper tests
```

## Installation

Full setup instructions live in [docs/startup.md](docs/startup.md). Short version:

### Requirements

- Python 3.10+
- Node.js 18+
- Ollama

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python search_server.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Ollama

Install Ollama, start the local service, then pull a model such as:

```bash
ollama pull llama3.1:8b
```

The explanation layer will run only if Ollama is available and the selected model exists locally.

### Dataset

The repository does not include the COCO image set or captions. The backend expects:

```text
backend/val2017/
backend/annotations/captions_val2017.json
```

## API Quickstart

Search:

```bash
curl "http://localhost:8000/search?query=a%20dog%20playing%20in%20the%20park&top_k=5"
```

Generate a retrieval-grounded explanation:

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a dog playing in the park",
    "model": "llama3.1:8b",
    "results": [
      { "file_name": "000000000139.jpg", "caption": "A dog running through a grassy park.", "score": 0.51 }
    ]
  }'
```

Useful endpoints:

- `/status`
- `/search`
- `/rag`
- `/explain`
- `/metrics`
- `/metrics/summary`
- `/benchmarks`
- `/ollama/models`

## Limitations

- current modality support is image-only
- explanation quality is bounded by retrieval quality
- dataset assets are not bundled in the repo
- local generation is significantly slower than retrieval
- the current explanation layer is not yet a full citation-rich RAG stack

## Screenshots

Screenshots and diagrams are available under `assets/`.
