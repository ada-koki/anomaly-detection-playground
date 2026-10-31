"""Issue #30: チャンク coreset の run を既定条件と別グループに集計するテスト。"""

import unittest

from adbench.aggregate import (
    build_runs_table,
    render_report_md,
    summarize_rows,
    variant_of_report,
)


def _report(auroc: float, mitigation: dict | None) -> dict:
    report = {"status": "ok", "test_metrics": [{"image_AUROC": auroc}]}
    if mitigation is not None:
        report["training_memory_mitigation"] = mitigation
    return report


class AggregateVariantTests(unittest.TestCase):
    def test_variant_of_report(self) -> None:
        self.assertIsNone(variant_of_report(None))
        self.assertIsNone(variant_of_report(_report(0.9, {"coreset_mode": "global"})))
        self.assertEqual(
            variant_of_report(_report(0.9, {"coreset_mode": "chunked", "chunk_images": 100})),
            "chunked_coreset_100img",
        )

    def test_chunked_runs_are_not_mixed_with_global(self) -> None:
        reports = {
            "g": _report(0.90, {"coreset_mode": "global"}),
            "c": _report(0.80, {"coreset_mode": "chunked", "chunk_images": 20}),
        }
        entries = [
            {"run_key": key, "model": "patchcore", "category": "bottle", "image_size": 256,
             "cpu_threads": 8, "repeat": 0, "status": "ok", "report_path": key}
            for key in reports
        ]
        rows = build_runs_table(entries, lambda path: reports.get(path))
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 2)
        by_variant = {row["variant"]: row for row in summary}
        self.assertAlmostEqual(by_variant[None]["image_AUROC_mean"], 0.90)
        self.assertAlmostEqual(by_variant["chunked_coreset_20img"]["image_AUROC_mean"], 0.80)
        markdown = render_report_md(rows, summary)
        self.assertIn("patchcore [chunked_coreset_20img]", markdown)


if __name__ == "__main__":
    unittest.main()
