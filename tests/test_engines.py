"""adbench.engines と scripts/compare_engines.py の軽量テスト。

重い依存（torch, onnx, onnxruntime, onnxscript, openvino, sklearn, PIL）を
必要としないテストは依存無しで動作する。torch/onnxruntime/onnxscript が
揃っている場合のみ、小さな nn.Module の export→onnxruntime 一致テストを実行する。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adbench import engines


def _has(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


class TestParseEngines(unittest.TestCase):
    def test_default(self) -> None:
        self.assertEqual(engines.parse_engines(None), list(engines.ENGINE_NAMES))
        self.assertEqual(engines.parse_engines(""), list(engines.ENGINE_NAMES))
        self.assertEqual(engines.parse_engines("   "), list(engines.ENGINE_NAMES))

    def test_basic(self) -> None:
        self.assertEqual(
            engines.parse_engines("torch,onnxruntime"),
            ["torch", "onnxruntime"],
        )
        self.assertEqual(
            engines.parse_engines("torch, ONNXRuntime , openvino"),
            ["torch", "onnxruntime", "openvino"],
        )

    def test_duplicate_raises(self) -> None:
        with self.assertRaises(ValueError):
            engines.parse_engines("torch,torch,openvino")

    def test_torch_required(self) -> None:
        with self.assertRaises(ValueError):
            engines.parse_engines("onnxruntime")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError):
            engines.parse_engines("torch,cuda")


@unittest.skipUnless(_has("numpy"), "numpy が必要")
class TestCompareOutputs(unittest.TestCase):
    def test_identical(self) -> None:
        import numpy as np

        scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        maps = (np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 1.0).astype(np.float32)
        result = engines.compare_outputs(scores, maps, scores, maps)
        self.assertEqual(result["score_max_abs_diff"], 0.0)
        self.assertEqual(result["score_mean_abs_diff"], 0.0)
        self.assertEqual(result["map_max_abs_diff"], 0.0)
        self.assertEqual(result["map_mean_abs_diff"], 0.0)
        self.assertEqual(result["map_max_rel_diff"], 0.0)

    def test_known_diff(self) -> None:
        import numpy as np

        ref_scores = np.array([1.0, 2.0], dtype=np.float32)
        tgt_scores = np.array([1.1, 2.0], dtype=np.float32)
        ref_maps = np.array([[[1.0, 2.0]]], dtype=np.float32)
        tgt_maps = np.array([[[1.5, 2.0]]], dtype=np.float32)
        result = engines.compare_outputs(ref_scores, ref_maps, tgt_scores, tgt_maps)
        self.assertAlmostEqual(result["score_max_abs_diff"], 0.1, places=5)
        self.assertAlmostEqual(result["map_max_abs_diff"], 0.5, places=5)
        # 相対差は |a - b| / max(|b|, eps): ref_maps[0,0,0]=1.0, tgt=1.5 → 0.5
        self.assertAlmostEqual(result["map_max_rel_diff"], 0.5, places=5)

    def test_shape_mismatch(self) -> None:
        import numpy as np

        with self.assertRaises(ValueError):
            engines.compare_outputs(
                np.zeros(3), np.zeros((1, 2, 2)),
                np.zeros(4), np.zeros((1, 2, 2)),
            )
        with self.assertRaises(ValueError):
            engines.compare_outputs(
                np.zeros(2), np.zeros((1, 2, 2)),
                np.zeros(2), np.zeros((1, 2, 3)),
            )


@unittest.skipUnless(_has("numpy"), "numpy が必要")
class TestDecisionAgreement(unittest.TestCase):
    def test_perfect(self) -> None:
        import numpy as np

        scores = np.array([0.1, 0.9, 0.5])
        self.assertEqual(engines.decision_agreement(scores, scores, 0.5), 1.0)

    def test_none_and_nan_threshold(self) -> None:
        import numpy as np

        s = np.array([0.1, 0.9])
        self.assertIsNone(engines.decision_agreement(s, s, None))
        self.assertIsNone(engines.decision_agreement(s, s, float("nan")))
        self.assertIsNone(engines.decision_agreement(s, s, float("inf")))

    def test_disagreement(self) -> None:
        import numpy as np

        ref = np.array([0.1, 0.9, 0.1, 0.9])
        tgt = np.array([0.1, 0.1, 0.9, 0.9])
        self.assertAlmostEqual(engines.decision_agreement(ref, tgt, 0.5), 0.5)


class TestFileSizeBytes(unittest.TestCase):
    def test_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.bin"
            p.write_bytes(b"x" * 100)
            self.assertEqual(engines.file_size_bytes([p]), 100)
            self.assertEqual(
                engines.file_size_bytes([p, Path(tmp) / "missing.bin"]), 100
            )


class TestLoadManifestTestSamples(unittest.TestCase):
    def _write(self, data: object) -> Path:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        return Path(path)

    def test_basic(self) -> None:
        manifest = {
            "schema_version": 1,
            "dataset_root": "/data/mvtec",
            "splits": {
                "test": [
                    {
                        "image_path": "test/bottle/good/000.png",
                        "label": "good",
                        "label_index": 0,
                        "mask_path": None,
                    },
                    {
                        "image_path": "test/bottle/broken_small/001.png",
                        "label": "broken_small",
                        "label_index": 1,
                        "mask_path": "ground_truth/bottle/broken_small/001_mask.png",
                    },
                ]
            },
        }
        path = self._write(manifest)
        try:
            info = engines.load_manifest_test_samples(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(info["dataset_root"], "/data/mvtec")
        self.assertEqual(len(info["samples"]), 2)
        self.assertTrue(info["samples"][0]["image_path"].endswith("000.png"))
        self.assertIsNone(info["samples"][0]["mask_path"])
        self.assertEqual(info["samples"][0]["label_index"], 0)
        self.assertTrue(info["samples"][1]["mask_path"].endswith("001_mask.png"))
        self.assertEqual(info["samples"][1]["label_index"], 1)

    def test_missing_dataset_root(self) -> None:
        path = self._write({"splits": {"test": [{"image_path": "a.png"}]}})
        try:
            with self.assertRaises(ValueError):
                engines.load_manifest_test_samples(path)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_test_split(self) -> None:
        path = self._write({"dataset_root": "/x", "splits": {"test": []}})
        try:
            with self.assertRaises(ValueError):
                engines.load_manifest_test_samples(path)
        finally:
            path.unlink(missing_ok=True)


class TestEngineNames(unittest.TestCase):
    def test_engine_names(self) -> None:
        self.assertEqual(engines.ENGINE_NAMES, ("torch", "onnxruntime", "openvino"))


class TestCliHelp(unittest.TestCase):
    def test_help_runs(self) -> None:
        script = REPO_ROOT / "scripts" / "compare_engines.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compare_engines", result.stdout + result.stderr)
        self.assertIn("--checkpoint", result.stdout)


@unittest.skipUnless(_has("onnx"), "onnx が必要")
class TestActualDefaultOpset(unittest.TestCase):
    def test_reads_default_domain(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("com.microsoft", 1),
                helper.make_opsetid("", 18),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.onnx"
            onnx.save(model, str(path))
            self.assertEqual(engines.actual_default_opset(path), 18)

    def test_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(engines.actual_default_opset(Path(tmp) / "none.onnx"))


@unittest.skipUnless(
    _has("torch") and _has("onnxruntime") and _has("onnxscript"),
    "torch / onnxruntime / onnxscript が必要",
)
class TestExportRoundTrip(unittest.TestCase):
    def test_simple_module(self) -> None:
        from collections import namedtuple

        import numpy as np
        import torch

        Batch = namedtuple("InferenceBatch", ["pred_score", "anomaly_map"])

        class Toy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

            def forward(self, x: "torch.Tensor") -> "Batch":
                y = self.conv(x)
                score = y.mean(dim=(1, 2, 3))
                return Batch(pred_score=score, anomaly_map=y)

        model = Toy().eval()
        example = torch.randn(1, 3, 8, 8)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.onnx"
            result = engines.export_onnx(model, example, path, opset=17)
            self.assertEqual(result["status"], "ok", msg=str(result))
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            # 要求 opset とは別に、ファイルに書かれた実際の opset を記録する
            self.assertEqual(result["opset_requested"], 17)
            self.assertEqual(result["opset"], engines.actual_default_opset(path))
            self.assertIsInstance(result["opset"], int)
            data_path = Path(str(path) + ".data")
            expected = path.stat().st_size + (
                data_path.stat().st_size if data_path.exists() else 0
            )
            self.assertEqual(result["file_size_bytes"], expected)
            self.assertEqual(result["external_data"], data_path.exists())

            import onnxruntime as ort

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            names = [o.name for o in session.get_outputs()]
            x = np.ascontiguousarray(example.numpy().astype(np.float32))
            outs = session.run(names, {session.get_inputs()[0].name: x})
            by_name = dict(zip(names, outs))
            self.assertIn("pred_score", by_name)
            self.assertIn("anomaly_map", by_name)


if __name__ == "__main__":
    unittest.main()
