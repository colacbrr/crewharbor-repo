# Startup

## Overview

This project has two main parts:

- `backend/` for the FastAPI retrieval service
- `frontend/` for the React + Vite demo UI

The repository is a cleaned public-facing skeleton, so the dataset and generated outputs are not included by default.

## Prerequisites

Install locally:

- Python 3.10+
- Node.js 18+
- Ollama

You also need the dataset files expected by the backend:

- `backend/val2017/`
- `backend/annotations/captions_val2017.json`

And generated artifacts will be written under:

- `backend/outputs/`

## 1. Prepare The Backend

From the repo root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want grounded explanations, make sure Ollama is available and a model is installed. Example:

```bash
ollama pull llama3.1:8b
```

## 2. Dataset Placement

Place the COCO files like this:

```text
backend/
  val2017/
    000000000139.jpg
    ...
  annotations/
    captions_val2017.json
```

If these folders are missing, the backend can start, but retrieval will not work correctly.

## 3. Start The Backend

From `backend/`:

```bash
source .venv/bin/activate
python search_server.py
```

Expected default API:

```text
http://localhost:8000
```

Useful endpoints:

- `/status`
- `/search`
- `/explain`
- `/metrics/summary`
- `/benchmarks`

## 4. Start The Frontend

Open a new terminal and run:

```bash
cd frontend
npm install
npm run dev
```

The frontend will usually start at:

```text
http://localhost:5173
```

If needed, set the API base explicitly:

```bash
VITE_SEARCH_API_BASE=http://localhost:8000 npm run dev
```

## 5. First Demo Check

Once both services are running:

1. open the frontend
2. wait for the backend status to become ready
3. run a query such as `a dog playing in the park`
4. inspect the returned images
5. trigger `Generate explanation`

## 6. Optional Environment Variables

The backend supports configuration through environment variables.

Examples:

```bash
MIR_MAX_IMAGES=1000
MIR_INDEX_TYPE=hnsw
MIR_HNSW_M=32
MIR_HNSW_EF=64
MIR_RERANK=true
MIR_RERANK_ALPHA=0.25
MIR_RAG_MODEL=llama3.1:8b
```

Example startup:

```bash
MIR_MAX_IMAGES=1000 MIR_INDEX_TYPE=hnsw python search_server.py
```

## 7. Common Issues

### Backend starts but search returns no results

Check:

- `backend/val2017/` exists
- `backend/annotations/captions_val2017.json` exists
- the backend has permission to create `backend/outputs/`

### Frontend loads but cannot reach the API

Check:

- backend is running on `http://localhost:8000`
- `VITE_SEARCH_API_BASE` points to the right address

### Explanations fail

Check:

- Ollama is installed
- the selected model is available
- the backend can access the Ollama service

### First startup is slow

That is expected if embeddings or the index need to be created for the first time.
