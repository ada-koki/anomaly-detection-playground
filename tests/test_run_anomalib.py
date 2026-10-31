"""外部依存ライブラリなしで実行できる、CLIとJSON変換の基本テスト。"""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_anomalib.py"
spec = importlib.util.spec_from_file_location("run_anomalib", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BenchmarkCliTests(unittest.TestCase):
    def test_cpu_default(self):
        args = module.build_parser().parse_args([])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.model, "padim")

    def test_new_options(self):
        args = module.build_parser().parse_args(
            [
                "--backbone",
                "resnet18",
                "--coreset-sampling-ratio",
                "0.05",
                "--prefetch-weights",
            ]
        )
        self.assertEqual(args.backbone, "resnet18")
        self.assertAlmostEqual(args.coreset_sampling_ratio, 0.05)
        self.assertTrue(args.prefetch_weights)

    def test_new_option_defaults(self):
        args = module.build_parser().parse_args([])
        self.assertIsNone(args.backbone)
        self.assertAlmostEqual(args.coreset_sampling_ratio, 0.1)
        self.assertFalse(args.prefetch_weights)

    def test_mvtec_4k_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "MVTec-4K"
            for split in ("train", "test"):
                (root / "bottle" / split / "good").mkdir(parents=True)
            args = module.build_parser().parse_args(
                ["--dataset", "mvtec_4k", "--dataset-root", str(root)]
            )
            self.assertEqual(module.validate_args(args, module.build_parser()), root)

    def test_prefetch_weights_skips_dataset_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing"
            args = module.build_parser().parse_args(
                ["--dataset-root", str(root), "--prefetch-weights"]
            )
            self.assertEqual(module.validate_args(args, module.build_parser()), root)

    def test_invalid_split(self):
        args = module.build_parser().parse_args(["--val-fraction", "1"])
        with self.assertRaises(SystemExit):
            module.validate_args(args, module.build_parser())

    def test_json_safe_scalars(self):
        class TensorLike:
            def item(self):
                return 0.975

        self.assertEqual(
            module.json_safe([{"image_AUROC": TensorLike()}]),
            [{"image_AUROC": 0.975}],
        )

    def test_build_failed_report(self):
        base = {"status": "running", "run_id": "example"}
        report = module.build_failed_report(base, ValueError("boom"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "ValueError")
        self.assertEqual(report["error_message"], "boom")
        self.assertEqual(base["status"], "running")

    def test_collect_environment_keys(self):
        environment = module.collect_environment()
        self.assertIn("host", environment)
        self.assertIn("cpu_logical_cores", environment)
        self.assertIn("versions", environment)
        self.assertIn("os", environment["host"])
        self.assertIn("machine", environment["host"])
        self.assertIn("processor", environment["host"])
        self.assertIn("python", environment["versions"])
        for name in module.DEPENDENCY_NAMES:
            self.assertIn(name, environment["versions"])


if __name__ == "__main__":
    unittest.main()
