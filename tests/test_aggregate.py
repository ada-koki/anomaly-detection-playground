"""adbench.aggregate のテスト（標準ライブラリのみ）。"""

from __future__ import annotations

import csv
import io
import unittest
from typing import Any

from adbench.aggregate import (
    build_runs_table,
    render_report_md,
    runs_csv_text,
    summarize_rows,
    summary_csv_text,
)


def make_report(
    *,
    image_auroc: Any = 0.9,
    pixel_auroc: Any = 0.8,
    image_f1: Any = 0.7,
    pixel_f1: Any = 0.6,
    image_raw: Any = 0.85,
    pixel_raw: Any = 0.75,
    train_seconds: Any = 10.0,
    test_seconds: Any = 4.0,
    e2e_p50: Any = 40.0,
    e2e_p95: Any = 42.0,
    model_p50: Any = 22.0,
    model_p95: Any = 24.0,
    train_rss: Any = 100.0,
    process_rss: Any = 120.0,
    protocol_name: str = "mvtec_ad_v1",
    manifest: str = "m1",
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "status": status,
        "test_metrics": [
            {
                "image_AUROC": image_auroc,
                "pixel_AUROC": pixel_auroc,
                "image_F1Score": image_f1,
                "pixel_F1Score": pixel_f1,
            }
        ],
        "raw_score_metrics": {
            "image_AUROC_raw": image_raw,
            "pixel_AUROC_raw": pixel_raw,
        },
        "train_stage_seconds": train_seconds,
        "test_stage_seconds": test_seconds,
        "per_image_inference_latency": {
            "end_to_end_ms": {"p50": e2e_p50, "p95": e2e_p95},
            "model_ms": {"p50": model_p50, "p95": model_p95},
        },
        "memory": {
            "train_peak_rss_mib_sampled": train_rss,
            "process_lifetime_peak_rss_mib": process_rss,
        },
        "protocol": {"name": protocol_name},
        "data_split": {"manifest_sha256": manifest},
        "error_type": "RuntimeError" if status != "ok" else None,
        "error_message": "failed" if status != "ok" else None,
    }


def make_entry(run_key: str, report_path: str | None, status: str = "ok") -> dict[str, Any]:
    repeat = 1
    if "_r" in run_key:
        try:
            repeat = int(run_key.rsplit("_r", 1)[1])
        except ValueError:
            repeat = 1
    return {
        "run_key": run_key,
        "model": "padim",
        "category": "bottle",
        "image_size": 256,
        "cpu_threads": 8,
        "repeat": repeat,
        "status": status,
        "report_path": report_path,
        "error": None,
    }


def csv_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


class AggregateTest(unittest.TestCase):
    def test_missing_is_empty_in_csv_and_dash_in_markdown(self) -> None:
        entry = make_entry("padim__bottle__s256__t8__r1", "missing.json", "failed")
        entry["error"] = "returncode 1"
        rows = build_runs_table([entry], lambda path: None)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIsNone(rows[0]["image_AUROC"])
        self.assertEqual(rows[0]["error"], "returncode 1")

        table = csv_rows(runs_csv_text(rows))
        header = table[0]
        row = table[1]
        value = row[header.index("image_AUROC")]
        self.assertEqual(value, "")

        summary = summarize_rows(rows)
        markdown = render_report_md(
            rows,
            summary,
            {
                "config": {
                    "name": "test",
                    "protocol": "p.json",
                    "models": ["padim"],
                    "categories": ["bottle"],
                    "image_sizes": [256],
                    "cpu_threads": [8],
                }
            },
            {},
        )
        self.assertIn("—", markdown)
        self.assertNotIn("nan", markdown.lower())

    def test_nan_is_missing_and_noted(self) -> None:
        entry = make_entry("padim__bottle__s256__t8__r1", "report.json")
        report = make_report(image_auroc=float("nan"))
        rows = build_runs_table([entry], lambda path: report)
        self.assertIsNone(rows[0]["image_AUROC"])
        self.assertIn("nonfinite:image_AUROC", rows[0]["notes"])

        table = csv_rows(runs_csv_text(rows))
        header = table[0]
        self.assertEqual(table[1][header.index("image_AUROC")], "")
        self.assertNotIn("nan", runs_csv_text(rows).lower())

    def test_failed_is_not_included_in_mean(self) -> None:
        entries = [
            make_entry("padim__bottle__s256__t8__r1", "r1.json", "ok"),
            make_entry("padim__bottle__s256__t8__r2", "r2.json", "ok"),
            make_entry("padim__bottle__s256__t8__r3", "r3.json", "failed"),
        ]
        reports = {
            "r1.json": make_report(image_auroc=0.8),
            "r2.json": make_report(image_auroc=1.0),
            "r3.json": make_report(image_auroc=100.0, status="failed"),
        }
        rows = build_runs_table(entries, lambda path: reports.get(str(path)))
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        item = summary[0]
        self.assertEqual(item["n_runs"], 3)
        self.assertEqual(item["n_ok"], 2)
        self.assertEqual(item["n_failed"], 1)
        self.assertAlmostEqual(item["image_AUROC_mean"], 0.9)
        self.assertAlmostEqual(item["image_AUROC_std"], 0.1414213562373095)

    def test_std_is_missing_when_n_ok_is_one(self) -> None:
        entry = make_entry("padim__bottle__s256__t8__r1", "r1.json", "ok")
        rows = build_runs_table([entry], lambda path: make_report())
        summary = summarize_rows(rows)
        self.assertEqual(summary[0]["n_ok"], 1)
        self.assertIsNone(summary[0]["image_AUROC_std"])
        self.assertIsNotNone(summary[0]["image_AUROC_mean"])

    def test_shuffle_does_not_change_csv(self) -> None:
        entries = [
            make_entry("padim__bottle__s256__t8__r1", "r1.json", "ok"),
            make_entry("padim__bottle__s256__t8__r2", "r2.json", "failed"),
            make_entry("padim__bottle__s256__t8__r3", "r3.json", "ok"),
        ]
        reports = {
            "r1.json": make_report(image_auroc=0.8, manifest="m1"),
            "r2.json": make_report(image_auroc=0.5, manifest="m1", status="failed"),
            "r3.json": make_report(image_auroc=0.9, manifest="m1"),
        }
        rows_a = build_runs_table(entries, lambda path: reports.get(str(path)))
        rows_b = build_runs_table(
            list(reversed(entries)), lambda path: reports.get(str(path))
        )
        self.assertEqual(runs_csv_text(rows_a), runs_csv_text(rows_b))
        self.assertEqual(
            summary_csv_text(summarize_rows(rows_a)),
            summary_csv_text(summarize_rows(rows_b)),
        )

    def test_manifest_inconsistent(self) -> None:
        entries = [
            make_entry("padim__bottle__s256__t8__r1", "r1.json", "ok"),
            make_entry("padim__bottle__s256__t8__r2", "r2.json", "ok"),
        ]
        reports = {
            "r1.json": make_report(manifest="m1"),
            "r2.json": make_report(manifest="m2"),
        }
        rows = build_runs_table(entries, lambda path: reports.get(str(path)))
        summary = summarize_rows(rows)
        self.assertFalse(summary[0]["manifest_consistent"])

    def test_missing_report_failure_row(self) -> None:
        entry = make_entry("padim__bottle__s256__t8__r1", "missing.json", "failed")
        entry["error"] = "returncode 1; log_tail: boom"
        rows = build_runs_table([entry], lambda path: None)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["error"], "returncode 1; log_tail: boom")
        summary = summarize_rows(rows)
        self.assertEqual(summary[0]["n_runs"], 1)
        self.assertEqual(summary[0]["n_ok"], 0)
        self.assertEqual(summary[0]["n_failed"], 1)
        self.assertIsNone(summary[0]["image_AUROC_mean"])


if __name__ == "__main__":
    unittest.main()
