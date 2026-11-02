"""adbench.aggregate の execution_mode グループ化のテスト（依存なし）。

同じ (model, category, image_size, cpu_threads) でも実行モードが違えば
別グループ・別表になることを確認する。
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

from adbench import aggregate, devices

MODEL = "padim"
CATEGORY = "bottle"
IMAGE_SIZE = 256
CPU_THREADS = 1


def _report(**payload: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "ok"}
    report.update(payload)
    return report


REPORTS: dict[str, Mapping[str, Any]] = {
    # 新形式（execution_mode あり）
    "r1.json": _report(
        execution_mode=devices.EXECUTION_MODE_CPU_ONLY,
        test_metrics=[{"image_AUROC": 0.90, "pixel_AUROC": 0.80}],
        data_split={"manifest_sha256": "m1"},
    ),
    "r2.json": _report(
        execution_mode=devices.EXECUTION_MODE_CPU_ONLY,
        test_metrics=[{"image_AUROC": 0.92, "pixel_AUROC": 0.82}],
        data_split={"manifest_sha256": "m1"},
    ),
    "r3.json": _report(
        execution_mode=devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
        test_metrics=[{"image_AUROC": 0.95, "pixel_AUROC": 0.85}],
        data_split={"manifest_sha256": "m1"},
    ),
    # 旧形式（execution_mode なし）。training_mode から推定される。
    "r4.json": _report(
        training_mode="external_checkpoint_cpu_infer",
        test_metrics=[{"image_AUROC": 0.70, "pixel_AUROC": 0.60}],
    ),
    # 旧形式（training_mode もなし）。device が cuda なので gpu_reference と推定。
    "r5.json": _report(
        device="cuda",
        test_metrics=[{"image_AUROC": 0.99, "pixel_AUROC": 0.98}],
    ),
}


def _loader(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        raise KeyError("report_path がありません")
    return REPORTS[path]


def _entry(
    run_key: str,
    report_path: str | None,
    repeat: int = 0,
    status: str | None = "ok",
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "run_key": run_key,
        "model": MODEL,
        "category": CATEGORY,
        "image_size": IMAGE_SIZE,
        "cpu_threads": CPU_THREADS,
        "repeat": repeat,
        "report_path": report_path,
    }
    if status is not None:
        entry["status"] = status
    return entry


ENTRIES = [
    _entry("r1", "r1.json", 0),
    _entry("r2", "r2.json", 1),
    _entry("r3", "r3.json", 0),
    _entry("r4", "r4.json", 0),
    _entry("r5", "r5.json", 0),
    # report が読めない行。execution_mode は捏造せず None のままにする。
    _entry("r6", "missing.json", 0, status=None),
]


class BuildRunsTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = aggregate.build_runs_table(ENTRIES, _loader)
        self.by_key = {row["run_key"]: row for row in self.rows}

    def test_execution_mode_is_a_known_column(self) -> None:
        self.assertIn("execution_mode", aggregate.RUN_COLUMNS)
        self.assertIn("execution_mode", aggregate.SUMMARY_COLUMNS)

    def test_new_format_reports_keep_execution_mode(self) -> None:
        self.assertEqual(
            self.by_key["r1"]["execution_mode"],
            devices.EXECUTION_MODE_CPU_ONLY,
        )
        self.assertEqual(
            self.by_key["r3"]["execution_mode"],
            devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
        )

    def test_old_format_reports_are_estimated(self) -> None:
        self.assertEqual(
            self.by_key["r4"]["execution_mode"],
            devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
        )
        self.assertEqual(
            self.by_key["r5"]["execution_mode"],
            devices.EXECUTION_MODE_GPU_REFERENCE,
        )

    def test_unreadable_report_leaves_execution_mode_none(self) -> None:
        self.assertIsNone(self.by_key["r6"]["execution_mode"])
        self.assertEqual(self.by_key["r6"]["status"], "failed")

    def test_runs_csv_contains_execution_mode_column(self) -> None:
        text = aggregate.runs_csv_text(self.rows)
        header = text.splitlines()[0].split(",")
        self.assertIn("execution_mode", header)


class SummarizeRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = aggregate.build_runs_table(ENTRIES, _loader)
        self.summary = aggregate.summarize_rows(self.rows)
        self.by_mode = {row["execution_mode"]: row for row in self.summary}

    def test_different_modes_are_separate_groups(self) -> None:
        self.assertEqual(len(self.summary), 5)
        self.assertEqual(
            set(self.by_mode),
            {
                devices.EXECUTION_MODE_CPU_ONLY,
                devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
                devices.EXECUTION_MODE_GPU_REFERENCE,
                devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
                None,
            },
        )

    def test_cpu_only_group_aggregates_two_ok_runs(self) -> None:
        row = self.by_mode[devices.EXECUTION_MODE_CPU_ONLY]
        self.assertEqual(row["n_runs"], 2)
        self.assertEqual(row["n_ok"], 2)
        self.assertEqual(row["n_failed"], 0)
        self.assertAlmostEqual(float(row["image_AUROC_mean"]), 0.91, places=6)
        self.assertIsNotNone(row["image_AUROC_std"])
        self.assertTrue(row["manifest_consistent"])

    def test_gpu_train_cpu_infer_group_is_not_mixed(self) -> None:
        row = self.by_mode[devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER]
        self.assertEqual(row["n_runs"], 1)
        self.assertEqual(row["n_ok"], 1)
        self.assertAlmostEqual(float(row["image_AUROC_mean"]), 0.95, places=6)

    def test_estimated_old_format_groups_are_kept_apart(self) -> None:
        external = self.by_mode[
            devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE
        ]
        gpu_reference = self.by_mode[devices.EXECUTION_MODE_GPU_REFERENCE]
        self.assertEqual(external["n_runs"], 1)
        self.assertEqual(gpu_reference["n_runs"], 1)
        self.assertAlmostEqual(
            float(external["image_AUROC_mean"]), 0.70, places=6
        )
        self.assertAlmostEqual(
            float(gpu_reference["image_AUROC_mean"]), 0.99, places=6
        )

    def test_group_without_execution_mode_counts_as_failed(self) -> None:
        row = self.by_mode[None]
        self.assertEqual(row["n_runs"], 1)
        self.assertEqual(row["n_ok"], 0)
        self.assertEqual(row["n_failed"], 1)


class RenderReportMdTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = aggregate.build_runs_table(ENTRIES, _loader)
        self.summary = aggregate.summarize_rows(self.rows)
        self.text = aggregate.render_report_md(
            self.rows,
            self.summary,
            {
                "config": {
                    "name": "dummy-matrix",
                    "categories": [CATEGORY],
                    "models": [MODEL],
                }
            },
            {},
        )

    def test_each_execution_mode_has_own_table_heading(self) -> None:
        for mode in (
            devices.EXECUTION_MODE_CPU_ONLY,
            devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
            devices.EXECUTION_MODE_GPU_REFERENCE,
            devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
        ):
            with self.subTest(mode=mode):
                self.assertIn(f"#### 実行モード: {mode}", self.text)

    def test_comparison_caution_note_is_rendered(self) -> None:
        self.assertIn("実行モードが異なる行は同一表で比較しない", self.text)

    def test_single_mode_still_has_heading(self) -> None:
        only_cpu = [
            row
            for row in self.rows
            if row["execution_mode"] == devices.EXECUTION_MODE_CPU_ONLY
        ]
        summary = aggregate.summarize_rows(only_cpu)
        text = aggregate.render_report_md(only_cpu, summary, {}, {})
        self.assertIn(
            f"#### 実行モード: {devices.EXECUTION_MODE_CPU_ONLY}", text
        )
        self.assertNotIn(devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER, text)
        self.assertNotIn(devices.EXECUTION_MODE_GPU_REFERENCE, text)


if __name__ == "__main__":
    unittest.main()
