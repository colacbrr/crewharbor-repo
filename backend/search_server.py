import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import torch
import clip
from PIL import Image
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from config import (
    CLIP_MODEL_NAME,
    FUTURE_MODALITIES,
    HNSW_EF,
    HNSW_M,
    INDEX_TYPE,
    MAX_IMAGES,
    PORT,
    PROMPT_VERSION,
    RAG_CACHE_TTL_SEC,
    RAG_MAX_CAPTION_CHARS,
    RAG_MAX_CONTEXT_ITEMS,
    RAG_MODEL,
    RAG_NUM_PREDICT,
    RAG_TEMPERATURE,
    RAG_TIMEOUT_SEC,
    RAG_TOP_P,
    RERANK_ALPHA,
    RERANK_ENABLED,
    SUPPORTED_MODALITIES,
)
from metrics_utils import load_jsonl_events, summarize_events
from rag import (
    build_context_lines,
    build_prompt,
    build_structured_fallback,
    extract_usage,
    normalize_results,
    parse_structured_response,
    stable_cache_key,
)

try:
    import ollama
except Exception:
    ollama = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
COCO_IMAGES = BASE_DIR / "val2017"
COCO_ANNOTATIONS = BASE_DIR / "annotations" / "captions_val2017.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
EMBEDDINGS_FILE = OUTPUTS_DIR / f"coco_embeddings_{MAX_IMAGES}.npy"
IMAGE_FILES_FILE = OUTPUTS_DIR / f"coco_image_files_{MAX_IMAGES}.json"
MANIFEST_FILE = OUTPUTS_DIR / "manifest.json"
METRICS_LOG_FILE = OUTPUTS_DIR / "metrics.jsonl"
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
BENCHMARK_1K = BASE_DIR.parent / "benchmarks" / "run_1000_metrics_eval.json"
BENCHMARK_5K = BASE_DIR.parent / "benchmarks" / "run_5000_metrics_eval.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_or_build_index, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if COCO_IMAGES.exists():
    app.mount("/images", StaticFiles(directory=str(COCO_IMAGES)), name="images")
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

_init_lock = threading.Lock()
_model = None
_index = None
_image_files = []
_captions_by_filename = {}
_captions_all_by_filename = {}
_preprocess = None
_ready = False
_init_error = None
_progress = {
    "stage": "idle",
    "message": "Idle",
    "current": 0,
    "total": 0,
    "percent": 0,
}
_log_lines = []
_log_lock = threading.Lock()
_metrics_lock = threading.Lock()
_metrics = {
    "embeddings_source": None,
    "encoding_time_sec": None,
    "index_build_time_sec": None,
    "image_count": 0,
    "query_count": 0,
    "query_total_ms": 0.0,
    "last_query_ms": None,
    "rag_query_count": 0,
    "rag_cache_hits": 0,
    "rag_failures": 0,
}
_rag_cache = {}
_rag_cache_lock = threading.Lock()


class ExplainResultItem(BaseModel):
    file_name: str
    caption: str | None = None
    score: float | None = None
    image_url: str | None = None


class ExplainRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str | None = None
    results: list[ExplainResultItem] = Field(..., min_length=1)


def _log(message):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    with _log_lock:
        _log_lines.append(line)
        if len(_log_lines) > 200:
            _log_lines.pop(0)


def _load_captions():
    if not COCO_ANNOTATIONS.exists():
        return {}, {}

    with COCO_ANNOTATIONS.open("r", encoding="utf-8") as handle:
        coco_data = json.load(handle)

    captions_by_id = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        captions_by_id.setdefault(img_id, []).append(ann.get("caption", ""))

    filenames_by_id = {
        image.get("id"): image.get("file_name")
        for image in coco_data.get("images", [])
        if image.get("id") is not None
    }

    captions_by_filename = {}
    captions_all_by_filename = {}
    for image_id, filename in filenames_by_id.items():
        captions = captions_by_id.get(image_id)
        if not captions:
            continue
        captions_by_filename[filename] = captions[0]
        captions_all_by_filename[filename] = captions

    return captions_by_filename, captions_all_by_filename


def _set_progress(stage, message, current=0, total=0):
    percent = int((current / total) * 100) if total else 0
    _progress.update(
        {
            "stage": stage,
            "message": message,
            "current": current,
            "total": total,
            "percent": percent,
        }
    )
    _log(f"{stage}: {message} ({current}/{total})")


def _append_metrics_event(event):
    event["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    line = json.dumps(event, ensure_ascii=False)
    with _metrics_lock:
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            with METRICS_LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass


def _encode_images(image_paths, batch_size=32, progress_cb=None):
    embeddings = []
    valid_paths = []
    total = len(image_paths)
    processed = 0

    _model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            batch_valid_paths = []
            for path in batch_paths:
                try:
                    img = _preprocess(Image.open(path).convert("RGB"))
                    images.append(img)
                    batch_valid_paths.append(path)
                except Exception:
                    continue

            if not images:
                continue

            images = torch.stack(images).to(DEVICE)
            features = _model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
            valid_paths.extend(batch_valid_paths)
            processed += len(batch_valid_paths)
            if progress_cb:
                progress_cb(processed, total)

    if not embeddings:
        raise ValueError("No valid images were encoded. Check COCO images folder.")
    return np.vstack(embeddings), valid_paths


def _encode_texts(texts):
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    tokens = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = _model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)


def _encode_query(query_text):
    return _encode_texts([query_text])


def _build_index(embeddings):
    if INDEX_TYPE == "hnsw":
        index = faiss.IndexHNSWFlat(embeddings.shape[1], HNSW_M)
        index.hnsw.efSearch = HNSW_EF
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def _write_manifest(embeddings_shape):
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE,
        "clip_model": CLIP_MODEL_NAME,
        "index_type": INDEX_TYPE,
        "hnsw_m": HNSW_M,
        "hnsw_ef": HNSW_EF,
        "max_images": MAX_IMAGES,
        "embeddings_shape": list(embeddings_shape),
        "image_count": len(_image_files),
        "embeddings_file": EMBEDDINGS_FILE.name,
        "image_files_file": IMAGE_FILES_FILE.name,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_or_build_index():
    global _model, _index, _image_files, _preprocess, _ready, _init_error
    if _index is not None:
        return

    with _init_lock:
        if _index is not None:
            return

        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            if not COCO_IMAGES.exists():
                raise FileNotFoundError(f"Missing images folder: {COCO_IMAGES}")

            _set_progress("init", "Loading CLIP model")
            _log(f"Device: {DEVICE}")
            _model, _preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
            _model.eval()

            _set_progress("init", "Loading image list")
            if IMAGE_FILES_FILE.exists():
                _image_files = json.loads(IMAGE_FILES_FILE.read_text(encoding="utf-8"))
            else:
                _image_files = sorted([p.name for p in COCO_IMAGES.glob("*.jpg")])[
                    :MAX_IMAGES
                ]
                IMAGE_FILES_FILE.write_text(
                    json.dumps(_image_files, indent=2), encoding="utf-8"
                )
            if not _image_files:
                raise FileNotFoundError(
                    f"No .jpg files found in images folder: {COCO_IMAGES}"
                )

            _set_progress("init", "Loading embeddings")
            if EMBEDDINGS_FILE.exists():
                embeddings = np.load(EMBEDDINGS_FILE)
                _log(f"Embeddings loaded: {embeddings.shape}")
                _metrics["embeddings_source"] = "loaded"
                _metrics["encoding_time_sec"] = None
            else:
                image_paths = [COCO_IMAGES / name for name in _image_files]
                _set_progress("encode", "Encoding images", 0, len(image_paths))
                encode_start = time.perf_counter()
                embeddings, valid_paths = _encode_images(
                    image_paths,
                    batch_size=64,
                    progress_cb=lambda current, total: _set_progress(
                        "encode", "Encoding images", current, total
                    ),
                )
                _metrics["encoding_time_sec"] = round(
                    time.perf_counter() - encode_start, 3
                )
                _metrics["embeddings_source"] = "computed"
                _image_files = [path.name for path in valid_paths]
                IMAGE_FILES_FILE.write_text(
                    json.dumps(_image_files, indent=2), encoding="utf-8"
                )
                np.save(EMBEDDINGS_FILE, embeddings)
                _log(f"Embeddings saved: {embeddings.shape}")

            if embeddings.shape[0] != len(_image_files):
                image_paths = [COCO_IMAGES / name for name in _image_files]
                _set_progress("encode", "Re-encoding images", 0, len(image_paths))
                encode_start = time.perf_counter()
                embeddings, valid_paths = _encode_images(
                    image_paths,
                    batch_size=64,
                    progress_cb=lambda current, total: _set_progress(
                        "encode", "Re-encoding images", current, total
                    ),
                )
                _metrics["encoding_time_sec"] = round(
                    time.perf_counter() - encode_start, 3
                )
                _metrics["embeddings_source"] = "computed"
                _image_files = [path.name for path in valid_paths]
                IMAGE_FILES_FILE.write_text(
                    json.dumps(_image_files, indent=2), encoding="utf-8"
                )
                np.save(EMBEDDINGS_FILE, embeddings)

            _set_progress("index", "Building FAISS index", 0, len(_image_files))
            build_start = time.perf_counter()
            _index = _build_index(embeddings)
            _metrics["index_build_time_sec"] = round(
                time.perf_counter() - build_start, 3
            )
            captions_single, captions_all = _load_captions()
            _captions_by_filename.update(captions_single)
            _captions_all_by_filename.update(captions_all)
            _metrics["image_count"] = len(_image_files)
            _write_manifest(embeddings.shape)
            _ready = True
            _set_progress("ready", "Index ready", len(_image_files), len(_image_files))
            _log("Index ready")
        except Exception as exc:
            _init_error = str(exc)
            _ready = False
            _set_progress("error", _init_error or "Initialization failed")


def _search_images(query_text, top_k=5):
    query_vector = _encode_query(query_text)
    similarities, indices = _index.search(query_vector, top_k)
    return query_vector[0], indices[0], similarities[0]


def _update_query_metrics(latency_ms):
    _metrics["query_count"] += 1
    _metrics["query_total_ms"] += latency_ms
    _metrics["last_query_ms"] = round(latency_ms, 2)


def _summarize_metrics_from_log(limit=1000):
    return summarize_events(load_jsonl_events(METRICS_LOG_FILE, limit=limit))


def _evaluate_recall(sample_size=200, ks=(1, 5, 10), seed=42):
    if not _captions_all_by_filename:
        return {"error": "Captions not available for evaluation"}
    filenames = [
        name for name in _image_files if name in _captions_all_by_filename
    ]
    if not filenames:
        return {"error": "No captioned images available"}

    rng = random.Random(seed)
    sample = filenames[:]
    rng.shuffle(sample)
    sample = sample[: min(sample_size, len(sample))]
    max_k = max(ks)

    hits = {k: 0 for k in ks}
    total = 0
    total_ms = 0.0

    for filename in sample:
        captions = _captions_all_by_filename.get(filename, [])
        if not captions:
            continue
        query = captions[0]
        start = time.perf_counter()
        _, indices, _ = _search_images(query, top_k=max_k)
        total_ms += (time.perf_counter() - start) * 1000.0
        retrieved = [ _image_files[i] for i in indices ]
        total += 1
        for k in ks:
            if filename in retrieved[:k]:
                hits[k] += 1

    if total == 0:
        return {"error": "No valid samples for evaluation"}

    recall = {f"recall@{k}": round(hits[k] / total, 4) for k in ks}
    return {
        "sample_size": total,
        "avg_query_ms": round(total_ms / total, 2),
        "recall": recall,
    }


def _list_ollama_models():
    if ollama is None:
        return {"available": False, "models": [], "error": "Ollama client not available"}
    try:
        data = ollama.list()
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}

    models = []
    if isinstance(data, dict):
        items = data.get("models") or []
        for item in items:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model") or item.get("id")
                if name:
                    models.append(name)
            elif isinstance(item, str):
                models.append(item)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model") or item.get("id")
                if name:
                    models.append(name)
            elif isinstance(item, str):
                models.append(item)

    unique_models = sorted(set(models))
    if not unique_models:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines:
                    parsed = []
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            parsed.append(parts[0])
                    unique_models = sorted(set(parsed))
        except Exception:
            pass

    return {"available": True, "models": unique_models, "default": RAG_MODEL}


def _read_json_file(path: Path):
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _api_error(message, code, **extra):
    payload = {"ok": False, "error": message, "error_code": code}
    payload.update(extra)
    return payload


def _avg_query_ms():
    if not _metrics["query_count"]:
        return None
    return round(_metrics["query_total_ms"] / _metrics["query_count"], 2)


def _cache_get(cache_key):
    now = time.time()
    with _rag_cache_lock:
        entry = _rag_cache.get(cache_key)
        if not entry:
            return None
        if entry["expires_at"] < now:
            _rag_cache.pop(cache_key, None)
            return None
        return entry["value"]


def _cache_set(cache_key, payload):
    with _rag_cache_lock:
        _rag_cache[cache_key] = {
            "expires_at": time.time() + RAG_CACHE_TTL_SEC,
            "value": payload,
        }


def _generate_with_timeout(selected_model, prompt):
    def _run():
        return ollama.generate(
            model=selected_model,
            prompt=prompt,
            options={
                "temperature": RAG_TEMPERATURE,
                "top_p": RAG_TOP_P,
                "num_predict": RAG_NUM_PREDICT,
            },
        )

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run)
    try:
        return future.result(timeout=RAG_TIMEOUT_SEC)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"Ollama timed out after {RAG_TIMEOUT_SEC:.0f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _build_rag_payload(query, results, selected_model, raw_text, llm_ms, usage, cache_hit):
    normalized = normalize_results(
        results,
        max_items=RAG_MAX_CONTEXT_ITEMS,
        max_caption_chars=RAG_MAX_CAPTION_CHARS,
    )
    fallback = build_structured_fallback(query, normalized)
    parsed = parse_structured_response(raw_text, fallback)
    return {
        "ok": True,
        "query": query,
        "model": selected_model,
        "prompt_version": PROMPT_VERSION,
        "llm_ms": round(llm_ms, 2),
        "duration_ms": round(llm_ms, 2),
        "cache_hit": cache_hit,
        "summary": parsed["summary"],
        "uncertainty": parsed.get("uncertainty"),
        "items": parsed["items"],
        "answer": parsed["formatted"],
        "explanation": parsed["formatted"],
        "used_fallback": parsed.get("used_fallback", False),
        "raw_response": raw_text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "context": {
            "query": query,
            "results": normalized,
            "lines": build_context_lines(normalized),
        },
    }


def _run_rag(query, results, selected_model):
    normalized = normalize_results(
        results,
        max_items=RAG_MAX_CONTEXT_ITEMS,
        max_caption_chars=RAG_MAX_CAPTION_CHARS,
    )
    cache_key = stable_cache_key(query, selected_model, normalized, PROMPT_VERSION)
    cached = _cache_get(cache_key)
    if cached is not None:
        _metrics["rag_query_count"] += 1
        _metrics["rag_cache_hits"] += 1
        return {**cached, "cache_hit": True}

    context_lines = build_context_lines(normalized)
    prompt = build_prompt(query, context_lines, PROMPT_VERSION)
    llm_start = time.perf_counter()
    response = _generate_with_timeout(selected_model, prompt)
    llm_ms = (time.perf_counter() - llm_start) * 1000.0
    payload = _build_rag_payload(
        query=query,
        results=normalized,
        selected_model=selected_model,
        raw_text=response.get("response", ""),
        llm_ms=llm_ms,
        usage=extract_usage(response),
        cache_hit=False,
    )
    _cache_set(cache_key, payload)
    _metrics["rag_query_count"] += 1
    return payload


def _build_results(query, top_k, rerank, rerank_alpha):
    candidate_k = max(top_k * 3, top_k)
    candidate_k = min(candidate_k, len(_image_files))
    query_vector, indices, scores = _search_images(query, top_k=candidate_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        filename = _image_files[idx]
        caption_list = _captions_all_by_filename.get(filename) or []
        results.append(
            {
                "rank": rank,
                "file_name": filename,
                "image_score": float(score),
                "score": float(score),
                "image_url": f"/images/{filename}",
                "caption": _captions_by_filename.get(filename),
                "caption_count": len(caption_list),
            }
        )

    if rerank and results:
        captions = [item["caption"] or "" for item in results]
        if any(captions):
            caption_embeddings = _encode_texts(captions)
            if caption_embeddings.shape[0] == len(results):
                caption_scores = caption_embeddings @ query_vector
                reranked = []
                for item, caption_score in zip(results, caption_scores.tolist()):
                    combined = (1.0 - rerank_alpha) * item["image_score"] + (
                        rerank_alpha * caption_score
                    )
                    item["caption_score"] = float(caption_score)
                    item["score"] = float(combined)
                    reranked.append(item)
                results = sorted(reranked, key=lambda x: x["score"], reverse=True)
                for rank, item in enumerate(results, start=1):
                    item["rank"] = rank
    return results[:top_k]


@app.get("/", response_class=HTMLResponse)
def root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return HTMLResponse(
        """
        <html>
          <head><title>MIR Search API</title></head>
          <body style="font-family: Arial, sans-serif; margin: 40px;">
            <h1>MIR Search API</h1>
            <p>Backend is running. Useful endpoints:</p>
            <ul>
              <li><a href="/status">/status</a></li>
              <li><a href="/docs">/docs</a></li>
            </ul>
            <p>Frontend lives in <code>frontend/</code>.</p>
            <pre>cd ../frontend
npm install
npm run dev</pre>
          </body>
        </html>
        """
    )

@app.get("/status")
def status():
    manifest = None
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "ready": _ready,
        "device": DEVICE,
        "index_size": len(_image_files),
        "error": _init_error,
        "progress": _progress,
        "index_type": INDEX_TYPE,
        "clip_model": CLIP_MODEL_NAME,
        "max_images": MAX_IMAGES,
        "modalities_supported": sorted(SUPPORTED_MODALITIES),
        "modalities_planned": sorted(FUTURE_MODALITIES),
        "rerank": RERANK_ENABLED,
        "rerank_alpha_default": RERANK_ALPHA,
        "rag_model": RAG_MODEL,
        "rag_timeout_sec": RAG_TIMEOUT_SEC,
        "rag_cache_ttl_sec": RAG_CACHE_TTL_SEC,
        "prompt_version": PROMPT_VERSION,
        "manifest": manifest,
        "metrics": {
            **_metrics,
            "avg_query_ms": _avg_query_ms(),
        },
    }


@app.get("/health")
def health():
    return {"ok": True, "ready": _ready, "error": _init_error}


@app.get("/capabilities")
def capabilities():
    return {
        "ok": True,
        "modalities_supported": sorted(SUPPORTED_MODALITIES),
        "modalities_planned": sorted(FUTURE_MODALITIES),
        "index_types": ["flat", "hnsw"],
        "clip_model": CLIP_MODEL_NAME,
        "rag_models_available": _list_ollama_models(),
    }


@app.get("/config")
def config():
    return {
        "ok": True,
        "device": DEVICE,
        "index_type": INDEX_TYPE,
        "hnsw_m": HNSW_M,
        "hnsw_ef": HNSW_EF,
        "rerank_default": RERANK_ENABLED,
        "rerank_alpha_default": RERANK_ALPHA,
        "clip_model": CLIP_MODEL_NAME,
        "rag_model": RAG_MODEL,
        "rag_temperature": RAG_TEMPERATURE,
        "rag_top_p": RAG_TOP_P,
        "rag_num_predict": RAG_NUM_PREDICT,
        "rag_timeout_sec": RAG_TIMEOUT_SEC,
        "rag_cache_ttl_sec": RAG_CACHE_TTL_SEC,
        "prompt_version": PROMPT_VERSION,
        "max_images": MAX_IMAGES,
    }


@app.get("/benchmarks")
def benchmarks():
    return {
        "ok": True,
        "run_1000": _read_json_file(BENCHMARK_1K),
        "run_5000": _read_json_file(BENCHMARK_5K),
    }


@app.get("/events")
def events():
    def event_stream():
        last_payload = None
        while True:
            payload = dict(_progress)
            payload["ready"] = _ready
            payload["error"] = _init_error
            if payload != last_payload:
                yield f"data: {json.dumps(payload)}\n\n"
                last_payload = payload
            time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/logs")
def logs():
    def log_stream():
        last_len = 0
        while True:
            with _log_lock:
                current_len = len(_log_lines)
                if current_len > last_len:
                    chunk = _log_lines[last_len:current_len]
                    last_len = current_len
                else:
                    chunk = []
            for line in chunk:
                yield f"data: {json.dumps({'line': line})}\n\n"
            time.sleep(1)

    return StreamingResponse(log_stream(), media_type="text/event-stream")


@app.get("/search")
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    rerank: bool = Query(RERANK_ENABLED),
    rerank_alpha: float = Query(RERANK_ALPHA, ge=0.0, le=1.0),
    modality: str = Query("image"),
):
    if modality not in SUPPORTED_MODALITIES:
        return _api_error(
            f"Unsupported modality '{modality}'.",
            "unsupported_modality",
            supported=sorted(SUPPORTED_MODALITIES),
            planned=sorted(FUTURE_MODALITIES),
        )
    if not _ready:
        return _api_error(_init_error or "Index initializing", "index_initializing")

    if _index is None:
        _load_or_build_index()

    if _index is None:
        return _api_error(_init_error or "Index unavailable", "index_unavailable")

    start = time.perf_counter()
    results = _build_results(query, top_k, rerank, rerank_alpha)
    latency_ms = (time.perf_counter() - start) * 1000.0
    _update_query_metrics(latency_ms)

    payload = {
        "ok": True,
        "query": query,
        "modality": modality,
        "top_k": top_k,
        "rerank": rerank,
        "rerank_alpha": rerank_alpha,
        "latency_ms": round(latency_ms, 2),
        "results": results,
    }
    _append_metrics_event(
        {
            "type": "search",
            "query": query,
            "modality": modality,
            "top_k": top_k,
            "rerank": rerank,
            "rerank_alpha": rerank_alpha,
            "latency_ms": round(latency_ms, 2),
            "results": len(results),
        }
    )
    return payload


@app.get("/rag")
def rag(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    rerank: bool = Query(RERANK_ENABLED),
    rerank_alpha: float = Query(RERANK_ALPHA, ge=0.0, le=1.0),
    model: str | None = None,
    modality: str = Query("image"),
):
    if modality not in SUPPORTED_MODALITIES:
        return _api_error(
            f"Unsupported modality '{modality}'.",
            "unsupported_modality",
            supported=sorted(SUPPORTED_MODALITIES),
            planned=sorted(FUTURE_MODALITIES),
        )
    if not _ready:
        return _api_error(_init_error or "Index initializing", "index_initializing")

    if ollama is None:
        return _api_error("Ollama client not available", "ollama_unavailable")

    start = time.perf_counter()
    results = _build_results(query, top_k, rerank, rerank_alpha)
    latency_ms = (time.perf_counter() - start) * 1000.0
    _update_query_metrics(latency_ms)
    if not results:
        return _api_error("No results available", "no_results")

    selected_model = model or RAG_MODEL
    try:
        rag_payload = _run_rag(query, results, selected_model)
    except TimeoutError as exc:
        _metrics["rag_failures"] += 1
        _append_metrics_event(
            {
                "type": "rag",
                "query": query,
                "modality": modality,
                "top_k": top_k,
                "model": selected_model,
                "status": "error",
                "error": str(exc),
            }
        )
        return _api_error(str(exc), "ollama_timeout")
    except Exception as exc:
        _metrics["rag_failures"] += 1
        _append_metrics_event(
            {
                "type": "rag",
                "query": query,
                "modality": modality,
                "top_k": top_k,
                "model": selected_model,
                "status": "error",
                "error": str(exc),
            }
        )
        return _api_error(f"Ollama error: {exc}", "ollama_error")

    payload = {
        **rag_payload,
        "modality": modality,
        "top_k": top_k,
        "rerank": rerank,
        "rerank_alpha": rerank_alpha,
        "latency_ms": round(latency_ms, 2),
        "results": results,
    }
    _append_metrics_event(
        {
            "type": "rag",
            "query": query,
            "modality": modality,
            "top_k": top_k,
            "model": selected_model,
            "latency_ms": round(latency_ms, 2),
            "llm_ms": payload["llm_ms"],
            "results": len(results),
            "status": "ok",
            "cache_hit": payload["cache_hit"],
            "prompt_tokens": payload.get("prompt_tokens"),
            "completion_tokens": payload.get("completion_tokens"),
        }
    )
    return payload


@app.post("/explain")
def explain(payload: ExplainRequest):
    if not _ready:
        return _api_error(_init_error or "Index initializing", "index_initializing")

    if ollama is None:
        return _api_error("Ollama client not available", "ollama_unavailable")

    query = payload.query
    results = [item.model_dump() for item in payload.results]
    selected_model = payload.model or RAG_MODEL
    try:
        response_payload = _run_rag(query, results, selected_model)
    except TimeoutError as exc:
        _metrics["rag_failures"] += 1
        _append_metrics_event(
            {
                "type": "explain",
                "query": query,
                "model": selected_model,
                "status": "error",
                "error": str(exc),
            }
        )
        return _api_error(str(exc), "ollama_timeout")
    except Exception as exc:
        _metrics["rag_failures"] += 1
        _append_metrics_event(
            {
                "type": "explain",
                "query": query,
                "model": selected_model,
                "status": "error",
                "error": str(exc),
            }
        )
        return _api_error(f"Ollama error: {exc}", "ollama_error")

    _append_metrics_event(
        {
            "type": "explain",
            "query": query,
            "model": selected_model,
            "llm_ms": response_payload["llm_ms"],
            "results": len(results),
            "status": "ok",
            "cache_hit": response_payload["cache_hit"],
            "prompt_tokens": response_payload.get("prompt_tokens"),
            "completion_tokens": response_payload.get("completion_tokens"),
        }
    )
    return response_payload


@app.get("/metrics")
def metrics(evaluate: bool = False, sample_size: int = 200):
    if not _ready:
        return _api_error(_init_error or "Index initializing", "index_initializing")

    response = {
        "ok": True,
        "metrics": {
            **_metrics,
            "avg_query_ms": _avg_query_ms(),
        }
    }

    if evaluate:
        response["evaluation"] = _evaluate_recall(sample_size=sample_size)
    return response


@app.get("/metrics/summary")
def metrics_summary(limit: int = 1000):
    return {"ok": True, **_summarize_metrics_from_log(limit=limit)}


@app.get("/metrics/log")
def metrics_log(limit: int = 200):
    return {"ok": True, "events": load_jsonl_events(METRICS_LOG_FILE, limit=limit)}


@app.get("/ollama/models")
def ollama_models():
    return {"ok": True, **_list_ollama_models()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("search_server:app", host="0.0.0.0", port=PORT, reload=False)
