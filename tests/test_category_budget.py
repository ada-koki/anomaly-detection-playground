"""adbench.category_budget と scripts/summarize_categories.py のテスト。"""

from __future__ import annotations

import csv as _csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.category_budget import (
    CategoryBudget,
    budget_from_dict,
    judge_budget,
    run_row_from_report,
    summarize_by_dataset,
)


def _load_cli() -> Any:
    """scripts/summarize_categories.py をファイルパスから読み込む。

    `from scripts.summarize_categories import main` を避けるため importlib を使う。
    """
    spec = importlib.util.spec_from_file_location(
        "summarize_categories_under_test",
        _REPO_ROOT / "scripts" / "summarize_categories.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ok_report(**overrides: Any) -> dict[str, Any]:
    """最小構成の status=ok な report.json を返す。"""
    report: dict[str, Any] = {
        "status": "ok",
        "run_id": "run-001",
        "test_metrics": [{"image_AUROC": 0.9, "pixel_AUROC": 0.85}],
        "memory": {"process_lifetime_peak_rss_mib": 500.0},
        "per_image_inference_latency": {"end_to_end_ms": {"p95": 12.5}},
        "data_split": {"roles": {"train": {"count": 10}, "test": {"count": 5}}},
        "train_stage_seconds": 10.0,
        "test_stage_seconds": 2.0,
        "latency_valid": True,
    }
    report.update(overrides)
    return report


def _row(**kw: Any) -> dict[str, Any]:
    """集計・判定テスト用の run 行のデフォルト値を返す。"""
    base: dict[str, Any] = dict(
        dataset="ds", model="m", variant="", category="a", status="ok",
        image_auroc=0.8, pixel_auroc=0.7, train_stage_seconds=10.0,
        peak_rss_mib=100.0, within_budget=True,
    )
    base.update(kw)
    return base


def _budget(**kw: Any) -> CategoryBudget:
    """予算オブジェクトを組み立てるヘルパー（閾値は既定 None）。"""
    base: dict[str, Any] = dict(
        name="b", max_train_stage_seconds=None,
        max_peak_rss_mib=None, max_download_mib=None,
    )
    base.update(kw)
    return CategoryBudget(**base)


class BudgetFromDictTests(unittest.TestCase):
    def test_valid(self) -> None:
        b = budget_from_dict({
            "name": "x",
            "max_train_stage_seconds": 60.0,
            "max_peak_rss_mib": 1024,
            "is_provisional": False,
            "notes": "テスト",
        })
        self.assertEqual(b.name, "x")
        self.assertEqual(b.max_train_stage_seconds, 60.0)
        self.assertEqual(b.max_peak_rss_mib, 1024.0)
        self.assertIsNone(b.max_download_mib)
        self.assertFalse(b.is_provisional)
        self.assertEqual(b.notes, "テスト")

    def test_unknown_key(self) -> None:
        with self.assertRaises(ValueError):
            budget_from_dict({"name": "x", "foo": 1})

    def test_negative(self) -> None:
        with self.assertRaises(ValueError):
            budget_from_dict({"name": "x", "max_train_stage_seconds": -1.0})

    def test_bool_rejected(self) -> None:
        with self.assertRaises(ValueError):
            budget_from_dict({"name": "x", "max_peak_rss_mib": True})


class RunRowFromReportTests(unittest.TestCase):
    def test_ok(self) -> None:
        row = run_row_from_report(
            _ok_report(), condition="c", dataset="d", category="cat", model="m"
        )
        self.assertEqual(row["status"], "ok")
        self.assertAlmostEqual(row["image_auroc"], 0.9)
        self.assertAlmostEqual(row["pixel_auroc"], 0.85)
        self.assertEqual(row["peak_rss_mib"], 500.0)
        self.assertEqual(row["e2e_p95_ms"], 12.5)
        self.assertEqual(row["train_images"], 10)
        self.assertEqual(row["test_images"], 5)
        self.assertEqual(row["variant"], "")

    def test_chunked_variant(self) -> None:
        report = _ok_report(
            training_memory_mitigation={"coreset_mode": "chunked", "chunk_images": 200}
        )
        row = run_row_from_report(report, condition="c", dataset="d", category="cat", model="m")
        self.assertEqual(row["variant"], "chunked_coreset_200img")

    def test_none_report_oom(self) -> None:
        row = run_row_from_report(
            None, condition="c", dataset="d", category="cat",
            model="m", failure_status="oom_killed",
        )
        self.assertEqual(row["status"], "oom_killed")
        self.assertIsNone(row["image_auroc"])
        self.assertIsNone(row["train_stage_seconds"])

    def test_missing_nested(self) -> None:
        row = run_row_from_report(
            {"status": "ok"}, condition="c", dataset="d", category="cat", model="m"
        )
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["image_auroc"])
        self.assertIsNone(row["peak_rss_mib"])
        self.assertIsNone(row["e2e_p95_ms"])
        self.assertIsNone(row["train_images"])


class JudgeBudgetTests(unittest.TestCase):
    def test_exceed(self) -> None:
        result = judge_budget(
            _row(train_stage_seconds=10.0), _budget(max_train_stage_seconds=5.0)
        )
        self.assertFalse(result["within_budget"])
        self.assertTrue(any("train_stage_seconds" in r for r in result["reasons"]))

    def test_within(self) -> None:
        result = judge_budget(
            _row(train_stage_seconds=10.0), _budget(max_train_stage_seconds=20.0)
        )
        self.assertTrue(result["within_budget"])
        self.assertEqual(result["reasons"], [])

    def test_failed_run(self) -> None:
        result = judge_budget(
            _row(status="failed", train_stage_seconds=None, image_auroc=None,
                 pixel_auroc=None, peak_rss_mib=None),
            _budget(max_train_stage_seconds=5.0),
        )
        self.assertFalse(result["within_budget"])
        self.assertTrue(any("status=" in r for r in result["reasons"]))

    def test_download_exceed(self) -> None:
        result = judge_budget(
            _row(train_stage_seconds=None, peak_rss_mib=None),
            _budget(max_download_mib=100.0),
            download_mib=150.0,
        )
        self.assertFalse(result["within_budget"])
        self.assertTrue(any("download_mib" in r for r in result["reasons"]))

    def test_all_unavailable(self) -> None:
        result = judge_budget(
            _row(train_stage_seconds=None), _budget(max_train_stage_seconds=5.0)
        )
        self.assertIsNone(result["within_budget"])
        self.assertIn("train_stage_seconds", result["unavailable"])

    def test_no_thresholds(self) -> None:
        result = judge_budget(_row(), _budget())
        self.assertIsNone(result["within_budget"])
        self.assertEqual(result["unavailable"], [])


class SummarizeTests(unittest.TestCase):
    def test_two_categories(self) -> None:
        rows = [
            _row(category="a", image_auroc=0.8, pixel_auroc=0.7,
                 train_stage_seconds=10.0, peak_rss_mib=100.0),
            _row(category="b", image_auroc=0.9, pixel_auroc=0.85,
                 train_stage_seconds=20.0, peak_rss_mib=200.0),
        ]
        summary = summarize_by_dataset(rows)
        self.assertEqual(len(summary), 1)
        s = summary[0]
        self.assertEqual(s["n_runs"], 2)
        self.assertEqual(s["n_ok"], 2)
        self.assertAlmostEqual(s["image_auroc_mean"], 0.85)
        self.assertAlmostEqual(s["image_auroc_std"], 0.070710678, places=5)
        self.assertAlmostEqual(s["image_auroc_min"], 0.8)
        self.assertAlmostEqual(s["image_auroc_max"], 0.9)
        self.assertEqual(s["categories"], "a,b")
        self.assertAlmostEqual(s["train_stage_seconds_total"], 30.0)
        self.assertEqual(s["peak_rss_mib_max"], 200.0)

    def test_failed_run_excluded(self) -> None:
        rows = [
            _row(category="a", image_auroc=0.8, within_budget=True),
            _row(category="b", status="failed", image_auroc=None, pixel_auroc=None,
                 train_stage_seconds=None, peak_rss_mib=None, within_budget=False),
        ]
        s = summarize_by_dataset(rows)[0]
        self.assertEqual(s["n_runs"], 2)
        self.assertEqual(s["n_ok"], 1)
        self.assertEqual(s["n_failed"], 1)
        self.assertEqual(s["n_within_budget"], 1)
        # 失敗行の数値（None）は平均に含まれない
        self.assertAlmostEqual(s["image_auroc_mean"], 0.8)

    def test_variant_separate(self) -> None:
        rows = [
            _row(category="a", variant=""),
            _row(category="a", variant="chunked_coreset_100img"),
        ]
        summary = summarize_by_dataset(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {s["variant"] for s in summary}, {"", "chunked_coreset_100img"}
        )


class CliTests(unittest.TestCase):
    def test_cli_end_to_end(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            reports = base / "reports"
            reports.mkdir()
            (reports / "r1.json").write_text(json.dumps(_ok_report()), encoding="utf-8")
            (reports / "r2.json").write_text(
                json.dumps(_ok_report(
                    training_memory_mitigation={
                        "coreset_mode": "chunked", "chunk_images": 100,
                    }
                )),
                encoding="utf-8",
            )
            entries = [
                {"condition": "c1", "dataset": "ds", "category": "cat1",
                 "model": "m", "report_path": "reports/r1.json",
                 "failure_status": None, "download_mib": 10.0},
                {"condition": "c1", "dataset": "ds", "category": "cat2",
                 "model": "m", "report_path": "reports/r2.json",
                 "failure_status": None, "download_mib": 10.0},
                {"condition": "c1", "dataset": "ds", "category": "cat3",
                 "model": "m", "report_path": "reports/missing.json",
                 "failure_status": None, "download_mib": None},
            ]
            runs_path = base / "runs.jsonl"
            runs_path.write_text(
                "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
            )
            budget_path = base / "budget.json"
            budget_path.write_text(json.dumps({
                "name": "test-budget",
                "max_train_stage_seconds": 60.0,
                "max_peak_rss_mib": 2048.0,
                "max_download_mib": 100.0,
            }), encoding="utf-8")
            out_dir = base / "out"
            ret = cli.main([
                "--runs-jsonl", str(runs_path),
                "--budget", str(budget_path),
                "--out-dir", str(out_dir),
            ])
            self.assertEqual(ret, 0)
            self.assertTrue((out_dir / "category_runs.csv").exists())
            self.assertTrue((out_dir / "dataset_summary.csv").exists())
            self.assertTrue((out_dir / "category_report.md").exists())
            with open(out_dir / "category_runs.csv", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            statuses = {r["category"]: r["status"] for r in rows}
            self.assertEqual(statuses["cat1"], "ok")
            self.assertEqual(statuses["cat3"], "missing_report")


if __name__ == "__main__":
    unittest.main()
