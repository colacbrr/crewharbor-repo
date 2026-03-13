import json


def avg(values):
    return round(sum(values) / len(values), 2) if values else None


def summarize_events(events):
    if not events:
        return {"events": 0}

    search_lat = [e.get("latency_ms") for e in events if e.get("type") == "search"]
    search_lat = [v for v in search_lat if isinstance(v, (int, float))]
    rag_llm = [e.get("llm_ms") for e in events if e.get("type") in {"rag", "explain"}]
    rag_llm = [v for v in rag_llm if isinstance(v, (int, float))]
    failures = [e for e in events if e.get("status") == "error"]
    cache_hits = [e for e in events if e.get("cache_hit")]

    return {
        "events": len(events),
        "avg_search_ms": avg(search_lat),
        "avg_llm_ms": avg(rag_llm),
        "failure_count": len(failures),
        "cache_hits": len(cache_hits),
    }


def load_jsonl_events(path, limit=1000):
    if not path.exists():
        return []

    events = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []

    if limit and len(events) > limit:
        return events[-limit:]
    return events
