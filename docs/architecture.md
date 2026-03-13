# Architecture

## System Summary

The demo is an end-to-end pipeline for semantic multimedia retrieval with grounded response generation.

The currently implemented modality is image search. The structure already anticipates future expansion toward video and audio.

## Pipeline

### 1. Data Source

The current dataset is based on MS COCO validation images and captions.

This gives the project:

- a stable image collection
- caption metadata for reranking and explanation
- a practical testbed for semantic search experiments

### 2. Embedding Generation

Images are encoded through CLIP into normalized vector embeddings.

Queries are encoded with the same CLIP text encoder, which places both text and images in a shared semantic space.

This is what makes text-to-image retrieval possible without a fixed class list.

### 3. Vector Index

The embeddings are added to a FAISS index.

The implementation supports:

- flat inner-product search
- HNSW-based approximate nearest-neighbor search

This allows direct discussion of speed and quality tradeoffs.

### 4. Retrieval

At query time:

1. the query is embedded
2. FAISS returns the top-k nearest items
3. captions and metadata are attached to the results

The returned set is semantic retrieval output rather than simple keyword matching.

### 5. Reranking

The system includes a reranking phase based on caption similarity.

This improves control over result quality and creates a better bridge between visual retrieval and language-based explanation.

### 6. Grounded Explanation

The top retrieved results are passed into a local Ollama-based LLM workflow.

The model receives:

- the original query
- the top retrieved items
- their captions
- supporting metadata

The output is a grounded explanation rather than unconstrained free-form text.

### 7. Metrics and Observability

The project records:

- retrieval latency
- query counts
- search metrics
- benchmark outputs
- logs and event streams

This gives the project a stronger engineering profile than a UI-only prototype.

## Frontend Role

The frontend is an operator-facing demo layer.

It exposes:

- live search
- quick queries
- reranking controls
- modality controls
- explanation generation
- benchmark panels
- research/demo toggles

This makes the demo suitable for interviews and technical walkthroughs.
