# Backend

This folder contains the FastAPI service that powers the retrieval system.

Main responsibilities:

- load or build CLIP embeddings
- create the FAISS index
- expose retrieval endpoints
- expose retrieval-grounded explanation endpoints
- publish status, logs, and metrics
- enforce timeout, fallback, and cache behavior for local generation

Main file:

- `search_server.py`
- `config.py`
- `rag.py`
- `metrics_utils.py`

Dataset note:

The dataset is not included in the repository.

To run the full demo locally, you need:

- `val2017/`
- `annotations/captions_val2017.json`

Generated artifacts should live in:

- `outputs/`

Those folders are excluded from version control.
