"""Dependency-free tests for the benchmark runner CLI and JSON conversion."""

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

    def test_mvtec_4k_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "MVTec-4K"
            for split in ("train", "test"):
                (root / "bottle" / split / "good").mkdir(parents=True)
            args = module.build_parser().parse_args(
                ["--dataset", "mvtec_4k", "--dataset-root", str(root)]
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


if __name__ == "__main__":
    unittest.main()
