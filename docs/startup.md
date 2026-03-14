# Startup

## Overview

This project has two main parts:

- `backend/` for the FastAPI retrieval service
- `frontend/` for the React + Vite demo UI

The dataset and generated outputs are not included by default.

## Prerequisites

Install locally:

- Python 3.10+
- Node.js 18+
- Ollama

Ollama installation:

- macOS: install from the official Ollama app or package manager, then open the app once
- Linux: install the Ollama service, then verify with `ollama list`
- Windows: install Ollama Desktop, make sure the local service is running, then verify with `ollama list`

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
ollama list
ollama pull llama3.1:8b
```

If `ollama list` fails, the local Ollama service is not available yet.

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

Official download example:

```bash
cd backend

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

rm val2017.zip
rm annotations_trainval2017.zip
```

This project only needs:

- `val2017/`
- `annotations/captions_val2017.json`

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
- `/rag`
- `/explain`
- `/metrics/summary`
- `/benchmarks`
- `/ollama/models`

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

For the first full run, expect startup to be slower because embeddings and the index may need to be built.

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
MIR_RAG_TIMEOUT_SEC=45
MIR_RAG_CACHE_TTL_SEC=300
```

Example startup:

```bash
MIR_MAX_IMAGES=1000 MIR_INDEX_TYPE=hnsw python search_server.py
```

## 7. API Examples

Search:

```bash
curl "http://localhost:8000/search?query=sunset%20over%20the%20ocean&top_k=5&rerank=true"
```

Explain from retrieved results:

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sunset over the ocean",
    "model": "llama3.1:8b",
    "results": [
      { "file_name": "000000000139.jpg", "caption": "A red sunset over the sea.", "score": 0.47 }
    ]
  }'
```

Direct retrieval-plus-generation in one request:

```bash
curl "http://localhost:8000/rag?query=sunset%20over%20the%20ocean&top_k=5&rerank=true"
```

## 8. Common Issues

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
- `ollama list` returns available local models
- the selected model is available
- the backend can access the Ollama service

### First startup is slow

That is expected if embeddings or the index need to be created for the first time.
