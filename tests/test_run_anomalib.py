"""外部依存ライブラリなしで実行できる、CLIとJSON変換の基本テスト。

CLIの既定値はプロトコルJSONで決まるため、``build_parser().parse_args()`` 直後ではなく
``parse_and_resolve()`` 後の引数で検証する。
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from adbench import report as report_module


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

    def test_latency_defaults(self):
        args = module.build_parser().parse_args([])
        self.assertEqual(args.latency_warmup, 5)
        self.assertEqual(args.latency_max_images, 0)
        self.assertEqual(args.latency_repeats, 1)
        self.assertFalse(args.skip_latency)

    def test_new_options(self):
        args = module.build_parser().parse_args(
            [
                "--backbone",
                "resnet18",
                "--coreset-sampling-ratio",
                "0.05",
                "--prefetch-weights",
                "--latency-warmup",
                "2",
                "--latency-max-images",
                "10",
                "--latency-repeats",
                "3",
                "--skip-latency",
            ]
        )
        self.assertEqual(args.backbone, "resnet18")
        self.assertAlmostEqual(args.coreset_sampling_ratio, 0.05)
        self.assertTrue(args.prefetch_weights)
        self.assertEqual(args.latency_warmup, 2)
        self.assertEqual(args.latency_max_images, 10)
        self.assertEqual(args.latency_repeats, 3)
        self.assertTrue(args.skip_latency)

    def test_new_option_defaults(self):
        """プロトコル mvtec_ad_v1 適用後の既定値を検証する。"""
        args, protocol, protocol_path, overrides = module.parse_and_resolve([])

        self.assertTrue(Path(protocol_path).is_file())
        self.assertEqual(protocol.protocol_name, "mvtec_ad_v1")
        self.assertEqual(overrides, {})

        self.assertEqual(args.dataset, "mvtec_ad")
        self.assertEqual(args.image_size, 256)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.cpu_threads, 0)
        self.assertEqual(args.val_split_mode, "from_test")
        self.assertAlmostEqual(args.val_split_ratio, 0.2)

        self.assertEqual(args.backbone, "resnet18")
        self.assertEqual(list(args.layers), ["layer1", "layer2", "layer3"])
        self.assertEqual(args.backbone_source, "protocol")
        self.assertFalse(args.backbone_is_default)
        # padim には coreset 系の設定がないため None のまま。
        self.assertIsNone(args.coreset_sampling_ratio)
        self.assertIsNone(args.num_neighbors)

        self.assertFalse(args.prefetch_weights)
        self.assertFalse(args.manifest_only)

        # patchcore はプロトコル既定値（coreset 0.1, num_neighbors 9）を使う。
        patch_args, _, _, _ = module.parse_and_resolve(["--model", "patchcore"])
        self.assertEqual(patch_args.backbone, "wide_resnet50_2")
        self.assertEqual(list(patch_args.layers), ["layer2", "layer3"])
        self.assertAlmostEqual(patch_args.coreset_sampling_ratio, 0.1)
        self.assertEqual(patch_args.num_neighbors, 9)

    def test_mvtec_4k_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "MVTec-4K"
            for split in ("train", "test"):
                (root / "bottle" / split / "good").mkdir(parents=True)
            args, _, _, overrides = module.parse_and_resolve(
                ["--dataset", "mvtec_4k", "--dataset-root", str(root)]
            )
            self.assertEqual(args.dataset, "mvtec_4k")
            self.assertEqual(overrides["dataset"], "mvtec_4k")
            self.assertEqual(module.validate_args(args, module.build_parser()), root)

    def test_prefetch_weights_skips_dataset_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing"
            args, _, _, _ = module.parse_and_resolve(
                ["--dataset-root", str(root), "--prefetch-weights"]
            )
            self.assertEqual(module.validate_args(args, module.build_parser()), root)

    def test_invalid_split(self):
        with self.assertRaises(SystemExit):
            args, _, _, _ = module.parse_and_resolve(
                ["--val-fraction", "1", "--prefetch-weights"]
            )
            module.validate_args(args, module.build_parser())

    def test_invalid_latency_options(self):
        invalid_options = (
            ["--latency-warmup", "-1"],
            ["--latency-max-images", "-1"],
            ["--latency-repeats", "0"],
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(SystemExit):
                    args, _, _, _ = module.parse_and_resolve(
                        ["--prefetch-weights", *options]
                    )
                    module.validate_args(args, module.build_parser())

    def test_json_safe_scalars(self):
        class TensorLike:
            def item(self):
                return 0.975

        self.assertEqual(
            report_module.json_safe([{"image_AUROC": TensorLike()}]),
            [{"image_AUROC": 0.975}],
        )

    def test_build_failed_report(self):
        base = {"status": "running", "run_id": "example"}
        report = report_module.build_failed_report(base, ValueError("boom"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "ValueError")
        self.assertEqual(report["error_message"], "boom")
        self.assertEqual(base["status"], "running")

    def test_collect_environment_keys(self):
        environment = report_module.collect_environment()
        self.assertIn("host", environment)
        self.assertIn("cpu_logical_cores", environment)
        self.assertIn("versions", environment)
        self.assertIn("os", environment["host"])
        self.assertIn("machine", environment["host"])
        self.assertIn("processor", environment["host"])
        self.assertIn("python", environment["versions"])
        for name in report_module.DEPENDENCY_NAMES:
            self.assertIn(name, environment["versions"])

    def test_build_base_report_schema_and_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets" / "MVTecAD"
            args, protocol, protocol_path, overrides = module.parse_and_resolve(
                [
                    "--dataset-root",
                    str(root),
                    "--prefetch-weights",
                    "--latency-warmup",
                    "2",
                ]
            )
            protocol_report = module._build_protocol_report(
                protocol, protocol_path, overrides
            )
            report = report_module.build_base_report(
                args, root, "example", [], protocol_report
            )
        self.assertEqual(report_module.SCHEMA_VERSION, 4)
        self.assertEqual(report["schema_version"], 4)
        self.assertIsNone(report["per_image_inference_latency"])
        self.assertIn("test_stage_seconds", report["timing_notes"])
        self.assertIn("engine.test", report["timing_notes"]["test_stage_seconds"])
        self.assertIn("画像1枚", report["timing_notes"]["test_stage_seconds"])
        self.assertEqual(report["latency_settings"]["warmup"], 2)
        memory = report["memory"]
        for key in (
            "train_peak_rss_mib_sampled",
            "test_peak_rss_mib_sampled",
            "latency_peak_rss_mib_sampled",
            "process_lifetime_peak_rss_mib",
            "sampling_interval_s",
            "notes",
        ):
            self.assertIn(key, memory)
        self.assertIn("train_peak_host_rss_mib_sampled", report)
        self.assertIn("test_peak_host_rss_mib_sampled", report)


if __name__ == "__main__":
    unittest.main()

