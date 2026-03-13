import os


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEVICE = "cuda"
CLIP_MODEL_NAME = os.getenv("MIR_CLIP_MODEL", "ViT-B/32")
INDEX_TYPE = os.getenv("MIR_INDEX_TYPE", "flat").lower()
HNSW_M = int(os.getenv("MIR_HNSW_M", "32"))
HNSW_EF = int(os.getenv("MIR_HNSW_EF", "64"))
RERANK_ENABLED = env_bool("MIR_RERANK", True)
RERANK_ALPHA = float(os.getenv("MIR_RERANK_ALPHA", "0.25"))
RAG_MODEL = os.getenv("MIR_RAG_MODEL", "llama3.1:8b")
RAG_TEMPERATURE = float(os.getenv("MIR_RAG_TEMPERATURE", "0.2"))
RAG_TOP_P = float(os.getenv("MIR_RAG_TOP_P", "0.9"))
RAG_NUM_PREDICT = int(os.getenv("MIR_RAG_NUM_PREDICT", "256"))
RAG_TIMEOUT_SEC = float(os.getenv("MIR_RAG_TIMEOUT_SEC", "45"))
RAG_CACHE_TTL_SEC = float(os.getenv("MIR_RAG_CACHE_TTL_SEC", "300"))
RAG_MAX_CONTEXT_ITEMS = int(os.getenv("MIR_RAG_MAX_CONTEXT_ITEMS", "5"))
RAG_MAX_CAPTION_CHARS = int(os.getenv("MIR_RAG_MAX_CAPTION_CHARS", "180"))
MAX_IMAGES = int(os.getenv("MIR_MAX_IMAGES", "1000"))
PORT = int(os.getenv("MIR_PORT", "8000"))
SUPPORTED_MODALITIES = {"image"}
FUTURE_MODALITIES = {"video", "audio"}
PROMPT_VERSION = "v2"
