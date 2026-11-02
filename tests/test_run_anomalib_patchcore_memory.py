"""Issue #30: PatchCore 学習メモリ対策の CLI 検証（重い依存なし）。"""

import importlib.util
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_anomalib.py"
spec = importlib.util.spec_from_file_location("run_anomalib_i30", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _validate(argv: list[str]) -> object:
    args, _, _, _ = module.parse_and_resolve(argv + ["--prefetch-weights"])
    module.validate_args(args, module.build_parser())
    return args


class PatchcoreMemoryCliTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = module.build_parser().parse_args([])
        self.assertEqual(args.patchcore_coreset_mode, "global")
        self.assertIsNone(args.patchcore_chunk_images)
        self.assertIsNone(args.max_train_feature_mib)

    def test_chunked_valid(self) -> None:
        args = _validate(
            [
                "--model", "patchcore",
                "--patchcore-coreset-mode", "chunked",
                "--patchcore-chunk-images", "100",
                "--max-train-feature-mib", "4000",
            ]
        )
        self.assertEqual(args.patchcore_chunk_images, 100)

    def test_invalid_combinations(self) -> None:
        cases = [
            ["--model", "patchcore", "--patchcore-coreset-mode", "chunked"],
            ["--model", "patchcore", "--patchcore-coreset-mode", "chunked", "--patchcore-chunk-images", "0"],
            ["--model", "patchcore", "--patchcore-chunk-images", "10"],
            ["--model", "patchcore", "--max-train-feature-mib", "0"],
            ["--model", "padim", "--patchcore-coreset-mode", "chunked", "--patchcore-chunk-images", "10"],
            ["--model", "padim", "--max-train-feature-mib", "100"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    _validate(argv)

    def test_feature_dim_only_for_default_backbone(self) -> None:
        args = _validate(["--model", "patchcore"])
        self.assertEqual(module._patchcore_feature_dim(args), 1536)
        args.layers = ["layer3"]
        self.assertIsNone(module._patchcore_feature_dim(args))


if __name__ == "__main__":
    unittest.main()
