# Demo Guide

## Recommended Flow

The strongest live demo is short and structured.

Suggested order:

1. show the architecture briefly
2. launch the backend
3. launch the frontend
4. run a few semantic queries
5. show the explanation flow
6. discuss benchmark artifacts
7. close with next steps and tradeoffs

## Backend

Main file:

- `backend/search_server.py`

Expected capabilities:

- image retrieval
- status reporting
- metrics endpoints
- logs/events streaming
- grounded explanation endpoint
- model discovery endpoint

## Frontend

Main file:

- `frontend/src/App.jsx`

What to highlight:

- live semantic query input
- quick-query presets
- rerank controls
- benchmark visibility
- grounded explanation panel

## Suggested Demo Queries

- `a dog playing in the park`
- `sunset over the ocean`
- `city street at night with neon lights`
- `a child holding a balloon`
- `a bicycle near a wall`

## Benchmark Talking Points

- retrieval remains interactive at both 1k and 5k scale
- quality drops as the search space expands, which is expected and useful to analyze
- the explanation layer is much slower than raw retrieval
- the cache/fallback path makes the explanation flow more resilient during repeated demos
