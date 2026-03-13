# Backend

This folder contains the FastAPI service that powers the retrieval demo.

Main responsibilities:

- load or build CLIP embeddings
- create the FAISS index
- expose retrieval endpoints
- expose grounded explanation endpoints
- publish status, logs, and metrics

Main file:

- `search_server.py`

Dataset note:

The dataset is intentionally not included in this cleaned repo skeleton.

To run the full demo locally, you need:

- `val2017/`
- `annotations/captions_val2017.json`

Generated artifacts should live in:

- `outputs/`

Those folders are excluded from version control in this public-ready structure.
