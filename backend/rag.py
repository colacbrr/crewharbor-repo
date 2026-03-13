import hashlib
import json
import re
from typing import Any


PROMPT_INSTRUCTIONS = """Respond in English. Use only the provided retrieval context.
Do not invent objects, actions, or relationships that are not supported by the retrieved items.
If the retrieved context is weak or ambiguous, say so briefly.
Return STRICT valid JSON only, with no markdown and no extra text, using this schema:
{
  "summary": string,
  "uncertainty": string,
  "items": [
    { "file_name": string, "caption": string, "score_pct": number }
  ]
}"""


def truncate_text(text: str | None, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return "no caption"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_results(results: list[dict[str, Any]], max_items: int, max_caption_chars: int):
    normalized = []
    seen = set()
    for idx, item in enumerate(results, start=1):
        file_name = item.get("file_name") or f"item_{idx}"
        if file_name in seen:
            continue
        seen.add(file_name)
        score = float(item.get("score", item.get("image_score", 0.0)) or 0.0)
        normalized.append(
            {
                "rank": len(normalized) + 1,
                "file_name": file_name,
                "caption": truncate_text(item.get("caption"), max_caption_chars),
                "score": score,
                "score_pct": round(score * 100.0, 1),
                "image_url": item.get("image_url"),
            }
        )
        if len(normalized) >= max_items:
            break
    return normalized


def build_context_lines(results: list[dict[str, Any]]):
    lines = []
    for item in results:
        lines.append(
            f"{item['rank']}. {item['file_name']} | score {item['score_pct']:.1f}% | "
            f"caption: {item['caption']}"
        )
    return lines


def build_prompt(query: str, context_lines: list[str], prompt_version: str):
    return (
        f"Prompt version: {prompt_version}\n"
        f"{PROMPT_INSTRUCTIONS}\n\n"
        "Example:\n"
        '{ "summary": "3 relevant results.", "uncertainty": "low", "items": ['
        '{ "file_name": "0001.jpg", "caption": "dogs running", "score_pct": 41.2 }'
        "] }\n\n"
        f"Query: {query}\n"
        "Context:\n"
        + "\n".join(context_lines)
        + "\n\nAnswer:"
    )


def build_structured_fallback(query: str, results: list[dict[str, Any]]):
    items = []
    for item in results:
        items.append(
            {
                "file_name": item["file_name"],
                "caption": item["caption"],
                "score_pct": item["score_pct"],
            }
        )
    return {
        "summary": f"The query returned {len(results)} retrieved results.",
        "uncertainty": "medium",
        "items": items,
        "used_fallback": True,
        "formatted": format_structured_answer(
            {
                "summary": f"The query returned {len(results)} retrieved results.",
                "uncertainty": "medium",
                "items": items,
            }
        ),
        "query": query,
    }


def parse_structured_response(text: str, fallback: dict[str, Any]):
    if not text:
        return fallback

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return fallback

    try:
        data = json.loads(match.group(0))
    except Exception:
        return fallback

    summary = (data.get("summary") or "").strip()
    uncertainty = (data.get("uncertainty") or "unspecified").strip()
    items = data.get("items") if isinstance(data.get("items"), list) else []
    if not summary or not items:
        return fallback

    parsed_items = []
    for idx, item in enumerate(items, start=1):
        parsed_items.append(
            {
                "file_name": item.get("file_name") or f"item_{idx}",
                "caption": item.get("caption") or "no caption",
                "score_pct": float(item.get("score_pct", 0.0) or 0.0),
            }
        )

    payload = {
        "summary": summary,
        "uncertainty": uncertainty,
        "items": parsed_items,
        "used_fallback": False,
    }
    payload["formatted"] = format_structured_answer(payload)
    return payload


def format_structured_answer(payload: dict[str, Any]):
    lines = [f"Summary: {payload.get('summary', '').strip()}"]
    uncertainty = payload.get("uncertainty")
    if uncertainty:
        lines.append(f"Uncertainty: {uncertainty}")
    for idx, item in enumerate(payload.get("items") or [], start=1):
        lines.append(
            f"{idx}) {item.get('file_name', f'item_{idx}')} — "
            f"{item.get('caption', 'no caption')}. "
            f"Score: {float(item.get('score_pct', 0.0)):.1f}%"
        )
    return "\n".join(lines)


def stable_cache_key(query: str, model: str, results: list[dict[str, Any]], prompt_version: str):
    payload = {
        "query": query,
        "model": model,
        "prompt_version": prompt_version,
        "results": [
            {
                "file_name": item.get("file_name"),
                "caption": item.get("caption"),
                "score_pct": item.get("score_pct"),
            }
            for item in results
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extract_usage(response: dict[str, Any]):
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")
    return {
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else None,
    }
