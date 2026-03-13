# Research Notes

## Problem

Multimedia retrieval is difficult because human queries are semantic, while raw images are only pixels.

This mismatch is often described as the semantic gap: users search for concepts, scenes, actions, and relationships, but traditional visual features do not map cleanly to those meanings.

## Why CLIP

CLIP is useful here because it maps images and text into a shared semantic space.

That means:

- text queries and images can be compared directly
- search is no longer restricted to a fixed label set
- the system can support natural-language retrieval without retraining for every query type

This makes it a strong baseline for multimodal retrieval.

## Why FAISS

Once images are turned into embeddings, the system still needs fast nearest-neighbor search.

FAISS is used because it provides:

- efficient vector indexing
- support for exact and approximate search
- a practical path from small controlled experiments to larger collections

This turns semantic embeddings into an actually usable search layer.

## Why Add Grounded Generation

Retrieval alone gives ranked results, but not always explanation.

Adding a grounded generation layer makes the system more useful because it can:

- summarize why the top results match the query
- use captions and metadata as evidence
- create a more interpretable interaction model

This is especially helpful for demos and for discussing product-facing use cases.

## Practical Tradeoffs

The project makes a few tradeoffs visible:

- retrieval can stay fast while explanation remains expensive
- larger indexes reduce recall unless retrieval strategy improves
- reranking helps quality but adds complexity
- future multimodal expansion requires stronger evaluation

## Direction

The longer-term direction of the project is to evolve from a local image-search baseline toward:

- richer multimodal support
- stronger reranking
- better observability
- more formal evaluation
- cloud-based comparison experiments
