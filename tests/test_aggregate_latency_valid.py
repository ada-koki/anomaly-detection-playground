"""latency_valid を考慮した集計のテスト。"""

from __future__ import annotations

import unittest
from typing import Any

from adbench import aggregate


def _report(
    *,
    p50: float,
    p95: float,
    model_p50: float,
    model_p95: float,
    latency_valid: bool | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "ok",
        "per_image_inference_latency": {
            "end_to_end_ms": {"p50": p50, "p95": p95},
            "model_ms": {"p50": model_p50, "p95": model_p95},
        },
    }
    if latency_valid is not None:
        report["latency_valid"] = latency_valid
    return report


class AggregateLatencyValidTest(unittest.TestCase):
    def _entries(self) -> list[dict[str, Any]]:
        return [
            {
                "run_key": "r-valid",
                "model": "padim",
                "category": "bottle",
                "image_size": 256,
                "cpu_threads": 8,
                "repeat": 0,
                "status": "ok",
                "report_path": "valid.json",
            },
            {
                "run_key": "r-invalid",
                "model": "padim",
                "category": "bottle",
                "image_size": 256,
                "cpu_threads": 8,
                "repeat": 1,
                "status": "ok",
                "report_path": "invalid.json",
            },
            {
                "run_key": "r-unjudged",
                "model": "padim",
                "category": "bottle",
                "image_size": 256,
                "cpu_threads": 8,
                "repeat": 2,
                "status": "ok",
                "report_path": "unjudged.json",
            },
        ]

    def _reports(self) -> dict[str, Any]:
        return {
            "valid.json": _report(
                p50=10.0, p95=12.0, model_p50=5.0, model_p95=6.0,
                latency_valid=True,
            ),
            "invalid.json": _report(
                p50=999.0, p95=1500.0, model_p50=500.0, model_p95=800.0,
                latency_valid=False,
            ),
            "unjudged.json": _report(
                p50=20.0, p95=24.0, model_p50=8.0, model_p95=9.0,
                latency_valid=None,
            ),
        }

    def test_summary_excludes_latency_invalid(self) -> None:
        reports = self._reports()
        rows = aggregate.build_runs_table(
            self._entries(), lambda path: reports.get(path)
        )
        summary = aggregate.summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        item = summary[0]
        self.assertEqual(item["n_runs"], 3)
        self.assertEqual(item["n_ok"], 3)
        self.assertEqual(item["n_latency_invalid"], 1)
        self.assertEqual(item["n_latency_unjudged"], 1)
        # invalid (999) は除外、(10 + 20) / 2 = 15
        self.assertAlmostEqual(item["e2e_p50_ms_mean"], 15.0)
        # (5 + 8) / 2 = 6.5
        self.assertAlmostEqual(item["model_p50_ms_mean"], 6.5)

    def test_report_md_has_latency_column(self) -> None:
        reports = self._reports()
        rows = aggregate.build_runs_table(
            self._entries(), lambda path: reports.get(path)
        )
        summary = aggregate.summarize_rows(rows)
        markdown = aggregate.render_report_md(rows, summary)
        self.assertIn("レイテンシ無効/未判定", markdown)
        self.assertIn("1/1", markdown)


if __name__ == "__main__":
    unittest.main()
