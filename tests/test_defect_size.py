"""``adbench.defect_size`` と ``scripts/defect_size_report.py`` のテスト。"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.defect_size import (  # noqa: E402
    SIZE_BANDS_DEFAULT,
    PerImageRecord,
    classify_size,
    load_records_jsonl,
    mask_stats,
    records_to_jsonl_lines,
    summarize,
)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_report_script() -> Any:
    """``scripts/defect_size_report.py`` をファイルパスから読み込む。"""
    path = _REPO_ROOT / "scripts" / "defect_size_report.py"
    spec = importlib.util.spec_from_file_location("defect_size_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"モジュールを読み込めません: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("defect_size_report", module)
    spec.loader.exec_module(module)
    return module


def _record(
    image_path: str,
    *,
    defect_type: str = "broken",
    label: int = 1,
    score: float = 1.0,
    mask_available: bool = True,
    mask_stats_value: dict[str, Any] | None = None,
    size_band: str | None = None,
    map_max_in_mask: float | None = None,
    map_max_outside_mask: float | None = None,
    normal_fp_pixel_ratio: float | None = None,
    mask_px_at_map_resolution: int | None = None,
) -> PerImageRecord:
    return PerImageRecord(
        image_path=image_path,
        defect_type=defect_type,
        label=label,
        score=score,
        mask_available=mask_available,
        mask_stats=mask_stats_value,
        size_band=size_band,
        map_max_in_mask=map_max_in_mask,
        map_max_outside_mask=map_max_outside_mask,
        normal_fp_pixel_ratio=normal_fp_pixel_ratio,
        mask_px_at_map_resolution=mask_px_at_map_resolution,
    )


def _sample_records() -> list[PerImageRecord]:
    """手計算できる小さな合成データ（image_threshold=1.0, pixel_threshold=0.5）。"""
    return [
        _record(
            "/data/test/broken/000.png",
            defect_type="broken",
            score=2.0,
            mask_stats_value={"area_ratio": 5e-4},
            size_band="tiny",
            map_max_in_mask=0.9,
            map_max_outside_mask=0.1,
            mask_px_at_map_resolution=12,
        ),
        _record(
            "/data/test/broken/001.png",
            defect_type="broken",
            score=0.5,
            mask_stats_value={"area_ratio": 5e-3},
            size_band="small",
            map_max_in_mask=0.2,
            map_max_outside_mask=0.3,
            mask_px_at_map_resolution=40,
        ),
        _record(
            "/data/test/broken/002.png",
            defect_type="broken",
            score=3.0,
            mask_available=False,
            size_band=None,
            map_max_in_mask=None,
            map_max_outside_mask=0.4,
            mask_px_at_map_resolution=None,
        ),
        _record(
            "/data/test/hole/000.png",
            defect_type="hole",
            score=1.0,
            mask_stats_value={"area_ratio": 5e-2},
            size_band="large",
            map_max_in_mask=0.5,
            map_max_outside_mask=0.0,
            mask_px_at_map_resolution=300,
        ),
        _record(
            "/data/test/good/000.png",
            defect_type="good",
            label=0,
            score=0.1,
            mask_available=False,
            normal_fp_pixel_ratio=0.01,
        ),
        _record(
            "/data/test/good/001.png",
            defect_type="good",
            label=0,
            score=2.0,
            mask_available=False,
            normal_fp_pixel_ratio=0.03,
        ),
    ]


class ClassifySizeTest(unittest.TestCase):
    def test_default_bands_boundaries(self) -> None:
        self.assertEqual(classify_size(0.0), "tiny")
        self.assertEqual(classify_size(5e-4), "tiny")
        self.assertEqual(classify_size(1e-3), "small")  # 下限を含む
        self.assertEqual(classify_size(9.99e-3), "small")
        self.assertEqual(classify_size(1e-2), "large")
        self.assertEqual(classify_size(0.5), "large")
        self.assertEqual(classify_size(1.0), "large")  # 最後の帯は上限を含む

    def test_out_of_range_and_invalid(self) -> None:
        self.assertIsNone(classify_size(None))
        self.assertIsNone(classify_size(float("nan")))
        self.assertIsNone(classify_size(-1e-6))
        self.assertIsNone(classify_size(1.0001))
        self.assertIsNone(classify_size("abc"))

    def test_custom_bands(self) -> None:
        bands = (("a", 0.0, 0.5), ("b", 0.5, 1.0))
        self.assertEqual(classify_size(0.5, bands), "b")
        self.assertEqual(classify_size(1.0, bands), "b")


class SummarizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = _sample_records()
        self.summary = summarize(self.records, 1.0, 0.5, min_samples=2)

    def test_thresholds_block(self) -> None:
        thresholds = self.summary["thresholds"]
        self.assertEqual(thresholds["image"], 1.0)
        self.assertEqual(thresholds["pixel"], 0.5)
        self.assertIn("F1AdaptiveThreshold", thresholds["source"])
        self.assertTrue(thresholds["available"]["image"])
        self.assertTrue(thresholds["available"]["pixel"])
        self.assertEqual(
            [band["name"] for band in self.summary["bands"]],
            [name for name, _lower, _upper in SIZE_BANDS_DEFAULT],
        )
        self.assertIn("detection_rate", self.summary["denominators"])

    def test_overall_denominator_is_abnormal_only(self) -> None:
        overall = self.summary["overall"]
        self.assertEqual(overall["n"], 4)  # 正常2枚は含まない
        self.assertEqual(overall["detected"], 3)
        self.assertEqual(overall["missed"], 1)
        self.assertAlmostEqual(overall["detection_rate"], 0.75, places=6)
        self.assertAlmostEqual(overall["miss_rate"], 0.25, places=6)
        self.assertEqual(overall["region_hit"], 2)
        self.assertEqual(overall["n_region_evaluable"], 3)
        self.assertAlmostEqual(overall["region_hit_rate"], 2 / 3, places=6)
        self.assertEqual(overall["n_without_mask"], 1)

    def test_by_size_band(self) -> None:
        cells = self.summary["by_size_band"]
        self.assertEqual(cells["tiny"]["n"], 1)
        self.assertAlmostEqual(cells["tiny"]["detection_rate"], 1.0, places=6)
        self.assertAlmostEqual(cells["tiny"]["region_hit_rate"], 1.0, places=6)

        self.assertEqual(cells["small"]["n"], 1)
        self.assertAlmostEqual(cells["small"]["detection_rate"], 0.0, places=6)
        self.assertAlmostEqual(cells["small"]["miss_rate"], 1.0, places=6)
        self.assertEqual(cells["small"]["region_hit"], 0)

        self.assertEqual(cells["large"]["n"], 1)
        self.assertAlmostEqual(cells["large"]["region_hit_rate"], 1.0, places=6)

    def test_by_defect_type(self) -> None:
        cells = self.summary["by_defect_type"]
        self.assertEqual(cells["broken"]["n"], 3)
        self.assertEqual(cells["broken"]["detected"], 2)
        self.assertAlmostEqual(cells["broken"]["detection_rate"], 2 / 3, places=6)
        self.assertEqual(cells["hole"]["n"], 1)
        self.assertEqual(cells["hole"]["detected"], 1)

    def test_by_defect_type_and_size_band(self) -> None:
        nested = self.summary["by_defect_type_and_size_band"]
        self.assertEqual(nested["broken"]["tiny"]["n"], 1)
        self.assertEqual(nested["broken"]["small"]["n"], 1)
        self.assertEqual(nested["broken"]["large"]["n"], 0)
        self.assertIsNone(nested["broken"]["large"]["detection_rate"])
        self.assertEqual(nested["hole"]["large"]["n"], 1)
        self.assertEqual(nested["hole"]["tiny"]["n"], 0)

    def test_empty_cell_note(self) -> None:
        cell = self.summary["by_defect_type_and_size_band"]["hole"]["small"]
        self.assertEqual(cell["n"], 0)
        self.assertIn("参考値", cell["note"])

    def test_normal_summary(self) -> None:
        normal = self.summary["normal"]
        self.assertEqual(normal["n_normal"], 2)
        self.assertEqual(normal["false_positive"], 1)  # score 2.0 のみ閾値以上
        self.assertAlmostEqual(normal["false_positive_rate"], 0.5, places=6)
        self.assertAlmostEqual(normal["mean_normal_fp_pixel_ratio"], 0.02, places=6)

    def test_counts(self) -> None:
        counts = self.summary["counts"]
        self.assertEqual(counts["n_images"], 6)
        self.assertEqual(counts["n_abnormal"], 4)
        self.assertEqual(counts["n_normal"], 2)
        self.assertEqual(counts["n_abnormal_without_size_band"], 1)

    def test_min_samples_note(self) -> None:
        summary = summarize(self.records, 1.0, 0.5, min_samples=5)
        self.assertIn("参考値", summary["overall"]["note"])

    def test_min_samples_not_needed(self) -> None:
        overall = self.summary["overall"]  # min_samples=2, n=4
        self.assertNotIn("参考値", overall["note"])


class ThresholdMissingTest(unittest.TestCase):
    def test_missing_thresholds(self) -> None:
        summary = summarize(_sample_records(), None, float("nan"), min_samples=2)
        thresholds = summary["thresholds"]
        self.assertIsNone(thresholds["image"])
        self.assertIsNone(thresholds["pixel"])
        self.assertFalse(thresholds["available"]["image"])
        self.assertFalse(thresholds["available"]["pixel"])

        overall = summary["overall"]
        self.assertIsNone(overall["detected"])
        self.assertIsNone(overall["missed"])
        self.assertIsNone(overall["detection_rate"])
        self.assertIsNone(overall["miss_rate"])
        self.assertIsNone(overall["region_hit"])
        self.assertIsNone(overall["region_hit_rate"])
        self.assertIn("閾値未計算", overall["note"])

        normal = summary["normal"]
        self.assertIsNone(normal["false_positive"])
        self.assertIsNone(normal["false_positive_rate"])


class NanScoreTest(unittest.TestCase):
    def test_nan_score_is_counted_as_missed(self) -> None:
        records = [
            _record("/data/test/broken/000.png", score=float("nan")),
            _record(
                "/data/test/broken/001.png",
                score=2.0,
                mask_stats_value={"area_ratio": 1e-3},
                size_band="small",
                map_max_in_mask=0.9,
            ),
        ]
        summary = summarize(records, 1.0, 0.5, min_samples=1)
        overall = summary["overall"]
        self.assertEqual(overall["detected"], 1)
        self.assertEqual(overall["missed"], 1)
        self.assertIn("NaN", overall["note"])


class JsonlRoundTripTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        records = _sample_records()[:2]
        lines = records_to_jsonl_lines(records)
        self.assertEqual(len(lines), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_image.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            loaded = load_records_jsonl(path)
        self.assertEqual(loaded, records)

    def test_none_fields_become_null(self) -> None:
        records = [
            _record(
                "/data/test/good/000.png",
                defect_type="good",
                label=0,
                score=0.1,
                mask_available=False,
                normal_fp_pixel_ratio=None,
            )
        ]
        data = json.loads(records_to_jsonl_lines(records)[0])
        self.assertIsNone(data["map_max_in_mask"])
        self.assertIsNone(data["mask_stats"])
        self.assertIsNone(data["score"] if data["score"] is None else None)
        self.assertFalse(data["mask_available"])

    def test_nan_score_is_written_as_null(self) -> None:
        records = [_record("/data/test/broken/000.png", score=float("nan"))]
        data = json.loads(records_to_jsonl_lines(records)[0])
        self.assertIsNone(data["score"])
        loaded = PerImageRecord.from_dict(data)
        self.assertTrue(math.isnan(loaded.score))

    def test_blank_lines_are_ignored(self) -> None:
        records = _sample_records()[:1]
        text = "\n".join(records_to_jsonl_lines(records)) + "\n\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_image.jsonl"
            path.write_text(text, encoding="utf-8")
            loaded = load_records_jsonl(path)
        self.assertEqual(len(loaded), 1)

    def test_invalid_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_image.jsonl"
            path.write_text("{broken json\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_records_jsonl(path)


@unittest.skipUnless(_has_module("numpy"), "numpy が無いため skip")
class MaskStatsTest(unittest.TestCase):
    def test_bbox_area_and_components(self) -> None:
        import numpy as np

        mask = np.zeros((10, 8), dtype=np.uint8)
        mask[1:3, 1:4] = 255  # 2x3 = 6画素
        mask[6, 5] = 255  # 孤立した1画素
        stats = mask_stats(mask)
        self.assertEqual(stats["area_px"], 7)
        self.assertEqual(stats["bbox"], [1, 1, 7, 6])
        self.assertEqual(stats["bbox_h"], 6)
        self.assertEqual(stats["bbox_w"], 5)
        self.assertAlmostEqual(stats["area_ratio"], 7 / 80, places=12)
        if _has_module("scipy"):
            self.assertEqual(stats["component_count"], 2)
            self.assertEqual(stats["largest_component_px"], 6)
        else:
            self.assertIsNone(stats["component_count"])
            self.assertIsNone(stats["largest_component_px"])

    def test_empty_mask(self) -> None:
        import numpy as np

        stats = mask_stats(np.zeros((4, 5), dtype=np.uint8))
        self.assertEqual(stats["area_px"], 0)
        self.assertIsNone(stats["bbox"])
        self.assertEqual(stats["bbox_h"], 0)
        self.assertEqual(stats["bbox_w"], 0)
        self.assertEqual(stats["area_ratio"], 0.0)
        self.assertEqual(stats["component_count"], 0)

    @unittest.skipUnless(_has_module("scipy"), "scipy が無いため skip")
    def test_diagonal_is_one_component_with_8_connectivity(self) -> None:
        import numpy as np

        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 1] = 1
        stats = mask_stats(mask)
        self.assertEqual(stats["component_count"], 1)
        self.assertEqual(stats["largest_component_px"], 2)

    def test_invalid_shape_raises(self) -> None:
        import numpy as np

        with self.assertRaises(ValueError):
            mask_stats(np.zeros((3,), dtype=np.uint8))


class DefectSizeReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_report_script()

    def _write_run(self, run_dir: Path, *, with_report: bool = True) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        records = [
            _record(
                "/data/test/broken/000.png",
                defect_type="broken",
                score=2.0,
                mask_stats_value={"area_ratio": 5e-4},
                size_band="tiny",
                map_max_in_mask=0.9,
                mask_px_at_map_resolution=12,
            ),
            _record(
                "/data/test/broken/001.png",
                defect_type="broken",
                score=0.5,
                mask_stats_value={"area_ratio": 5e-3},
                size_band="small",
                map_max_in_mask=0.1,
                mask_px_at_map_resolution=40,
            ),
            _record(
                "/data/test/good/000.png",
                defect_type="good",
                label=0,
                score=2.0,
                mask_available=False,
                normal_fp_pixel_ratio=0.02,
            ),
        ]
        (run_dir / "per_image.jsonl").write_text(
            "\n".join(records_to_jsonl_lines(records)) + "\n", encoding="utf-8"
        )
        if not with_report:
            return
        report = {
            "model": "fake_model",
            "train_stage_seconds": 1.5,
            "memory": {"train_peak_rss_mib_sampled": 123.4},
            "per_image_inference_latency": {
                "end_to_end_ms": {"p50": 10.0, "p95": 20.0}
            },
            "raw_score_metrics": {"image_AUROC_raw": 0.9, "pixel_AUROC_raw": 0.8},
            "spatial_strategy": {
                "spatial_strategy": "global_resize",
                "working_resolution": [256, 256],
                "tile_size": None,
                "stride": None,
                "tiles_per_image": 1,
            },
            "defect_size": {
                "thresholds": {
                    "image": 1.0,
                    "pixel": 0.5,
                    "source": "validation（PostProcessor F1AdaptiveThreshold, 生スケール）",
                },
                "min_samples": 5,
            },
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )

    def _run_main(self, argv: list[str]) -> int:
        stdout = []
        stderr = []
        with redirect_stdout(_Writer(stdout)), redirect_stderr(_Writer(stderr)):
            return int(self.module.main(argv))

    def test_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run1"
            self._write_run(run_dir)
            out_dir = root / "out"
            code = self._run_main(
                [
                    "--runs",
                    str(run_dir),
                    "--labels",
                    "runA",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(code, 0)

            csv_path = out_dir / "defect_size_summary.csv"
            self.assertTrue(csv_path.is_file())
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader.fieldnames or []), list(self.module.CSV_COLUMNS))
                rows = list(reader)

            overall = [
                row for row in rows if row["group"] == "overall" and row["key"] == "overall"
            ]
            self.assertEqual(len(overall), 1)
            self.assertEqual(overall[0]["run_label"], "runA")
            self.assertEqual(overall[0]["n"], "2")
            self.assertEqual(overall[0]["detected"], "1")
            self.assertAlmostEqual(float(overall[0]["detection_rate"]), 0.5, places=6)
            self.assertAlmostEqual(float(overall[0]["normal_fpr"]), 1.0, places=6)
            self.assertEqual(overall[0]["spatial_strategy"], "global_resize")
            self.assertEqual(overall[0]["working_resolution"], "256x256")
            self.assertAlmostEqual(float(overall[0]["e2e_p50_ms"]), 10.0, places=6)
            self.assertAlmostEqual(
                float(overall[0]["train_peak_rss_mib"]), 123.4, places=6
            )

            groups = {row["group"] for row in rows}
            self.assertIn("size_band", groups)
            self.assertIn("defect_type", groups)
            self.assertIn("defect_type_x_size_band", groups)

            tiny = [
                row for row in rows if row["group"] == "size_band" and row["key"] == "tiny"
            ]
            self.assertEqual(len(tiny), 1)
            self.assertAlmostEqual(float(tiny[0]["detection_rate"]), 1.0, places=6)

            md_path = out_dir / "defect_size_summary.md"
            self.assertTrue(md_path.is_file())
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("サイズ帯", markdown)
            self.assertIn("欠陥サイズ", markdown)
            self.assertIn("runA", markdown)

    def test_missing_report_json_keeps_rates_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_no_report"
            self._write_run(run_dir, with_report=False)
            out_dir = root / "out"
            code = self._run_main(
                ["--runs", str(run_dir), "--out-dir", str(out_dir)]
            )
            self.assertEqual(code, 0)
            with (out_dir / "defect_size_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            overall = [
                row for row in rows if row["group"] == "overall" and row["key"] == "overall"
            ]
            self.assertEqual(len(overall), 1)
            self.assertEqual(overall[0]["detection_rate"], "")
            self.assertIn("閾値未計算", overall[0]["note"])

    def test_missing_per_image_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "empty_run"
            run_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                self._run_main(
                    ["--runs", str(run_dir), "--out-dir", str(root / "out")]
                )


class _Writer:
    """``redirect_stdout`` / ``redirect_stderr`` 用の簡易ライタ。"""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def write(self, text: str) -> int:
        self._sink.append(text)
        return len(text)

    def flush(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
