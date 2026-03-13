import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from rag import (  # noqa: E402
    build_context_lines,
    build_prompt,
    build_structured_fallback,
    normalize_results,
    parse_structured_response,
    stable_cache_key,
)


class RagHelpersTests(unittest.TestCase):
    def test_normalize_results_deduplicates_and_truncates(self):
        results = [
            {"file_name": "a.jpg", "caption": "x" * 40, "score": 0.5},
            {"file_name": "a.jpg", "caption": "duplicate", "score": 0.4},
            {"file_name": "b.jpg", "caption": "short", "score": 0.3},
        ]
        normalized = normalize_results(results, max_items=5, max_caption_chars=12)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["file_name"], "a.jpg")
        self.assertTrue(normalized[0]["caption"].endswith("..."))

    def test_parse_structured_response_uses_json_when_valid(self):
        fallback = build_structured_fallback("dogs", [{"file_name": "a.jpg", "caption": "dog", "score_pct": 50.0}])
        parsed = parse_structured_response(
            '{"summary":"Two matches","uncertainty":"low","items":[{"file_name":"a.jpg","caption":"dog","score_pct":50.0}]}',
            fallback,
        )
        self.assertFalse(parsed["used_fallback"])
        self.assertEqual(parsed["summary"], "Two matches")
        self.assertIn("Uncertainty: low", parsed["formatted"])

    def test_parse_structured_response_falls_back(self):
        fallback = build_structured_fallback("dogs", [{"file_name": "a.jpg", "caption": "dog", "score_pct": 50.0}])
        parsed = parse_structured_response("not json", fallback)
        self.assertTrue(parsed["used_fallback"])

    def test_cache_key_is_stable(self):
        results = [{"file_name": "a.jpg", "caption": "dog", "score_pct": 50.0}]
        first = stable_cache_key("dogs", "llama3.1:8b", results, "v2")
        second = stable_cache_key("dogs", "llama3.1:8b", results, "v2")
        self.assertEqual(first, second)

    def test_build_prompt_contains_prompt_version_and_context(self):
        prompt = build_prompt("dogs", build_context_lines([{"rank": 1, "file_name": "a.jpg", "caption": "dog", "score_pct": 50.0}]), "v2")
        self.assertIn("Prompt version: v2", prompt)
        self.assertIn("a.jpg", prompt)


if __name__ == "__main__":
    unittest.main()
