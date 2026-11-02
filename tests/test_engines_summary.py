"""scripts/engines_summary.py のテスト。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("engines_summary", ROOT / "scripts/engines_summary.py")
assert _spec and _spec.loader
es = importlib.util.module_from_spec(_spec)
sys.modules["engines_summary"] = es
_spec.loader.exec_module(es)


def _report() -> dict:
    ms = {"p50": 10.0, "p95": 20.0, "mean": 12.0, "std": 3.0}
    eng = {"status": "ok", "model_ms": ms, "repeats": 3, "latency_samples": 6, "load_seconds": 1.0,
           "latency_peak_rss_mib_sampled": 100.0, "process_peak_rss_mib": 120.0,
           "process_peak_rss_source": "VmHWM(/proc/self/status)"}
    return {
        "run_id": "r1", "model": "padim", "threads": 8,
        "reference": {"image_AUROC_raw": 0.99, "pixel_AUROC_raw": 0.97},
        "export": {"onnxruntime": {"method": "dynamo", "opset": 18, "file_size_bytes": 1000, "seconds": 2.0,
                                   "onnx_path": "/abs/model.onnx"},
                   "openvino": {"method": "dynamo", "opset": 18, "file_size_bytes": 1000,
                                "ir_size_bytes": 1100, "seconds": 2.0}},
        "engines": {"torch": dict(eng), "onnxruntime": dict(eng), "openvino": dict(eng),
                    "broken": {"status": "error"}},
        "agreement": {"onnxruntime": {"decision_agreement": 1.0, "image_AUROC_raw": 0.99,
                                      "pixel_AUROC_raw": 0.97, "diffs": {"map_max_abs_diff": 1e-4}}},
    }


class EnginesSummaryTests(unittest.TestCase):
    def test_rows(self) -> None:
        rows = es.rows_from_report(_report(), "padim_t8")
        by = {r["engine"]: r for r in rows}
        self.assertEqual(by["torch"]["image_AUROC_raw"], 0.99)
        self.assertIsNone(by["torch"]["model_size_bytes"])
        self.assertEqual(by["onnxruntime"]["model_size_bytes"], 1000)
        self.assertEqual(by["openvino"]["model_size_bytes"], 1100)
        self.assertEqual(by["onnxruntime"]["map_max_abs_diff"], 1e-4)
        self.assertNotIn("model_ms_p50", {k for k, v in by["broken"].items() if v is not None})

    def test_cli_writes_no_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rp = Path(tmp) / "engine_report.json"
            rp.write_text(json.dumps(_report()), encoding="utf-8")
            out = Path(tmp) / "out"
            self.assertEqual(es.main(["--reports", str(rp), "--labels", "x", "--out-dir", str(out)]), 0)
            text = (out / "engines_summary.csv").read_text(encoding="utf-8")
            self.assertNotIn("/abs/", text)
            self.assertNotIn(tmp, text)
            rows = list(csv.DictReader(text.splitlines()))
            self.assertEqual(len(rows), 4)
            self.assertTrue((out / "engines_summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
