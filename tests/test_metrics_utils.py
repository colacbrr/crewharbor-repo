import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from metrics_utils import summarize_events  # noqa: E402


class MetricsUtilsTests(unittest.TestCase):
    def test_summarize_events_aggregates_latency_failures_and_cache_hits(self):
        summary = summarize_events(
            [
                {"type": "search", "latency_ms": 10.0},
                {"type": "rag", "llm_ms": 100.0, "cache_hit": True},
                {"type": "explain", "llm_ms": 200.0, "status": "error"},
            ]
        )
        self.assertEqual(summary["events"], 3)
        self.assertEqual(summary["avg_search_ms"], 10.0)
        self.assertEqual(summary["avg_llm_ms"], 150.0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
