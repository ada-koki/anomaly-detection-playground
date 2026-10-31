"""run_anomalib のタイル推論オプション（--tile-size 等）の引数検証テスト。

torch / anomalib は import しない。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "mvtec_ad_v1.json"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"モジュールを読み込めません: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN_ANOMALIB = _load_module(
    "run_anomalib_for_tiling_test",
    REPO_ROOT / "scripts" / "run_anomalib.py",
)


class TilingCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        for split in ("train", "test"):
            (self.root / "bottle" / split / "good").mkdir(parents=True)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _resolve(self, extra: list[str]):
        argv = [
            "--protocol",
            str(PROTOCOL_PATH),
            "--dataset-root",
            str(self.root),
            "--category",
            "bottle",
            *extra,
        ]
        args, _, _, _ = RUN_ANOMALIB.parse_and_resolve(argv)
        RUN_ANOMALIB.validate_args(args, RUN_ANOMALIB.build_parser())
        return args

    def test_no_tiling_is_global_resize(self) -> None:
        args = self._resolve(["--image-size", "256"])
        self.assertIsNone(RUN_ANOMALIB.build_tiling_config(args))
        strategy = RUN_ANOMALIB.describe_spatial_strategy(args)
        self.assertEqual(strategy["spatial_strategy"], "global_resize")
        self.assertEqual(strategy["tiles_per_image"], 1)

    def test_stride_defaults_to_tile_size(self) -> None:
        args = self._resolve(["--image-size", "512", "--tile-size", "256"])
        config = RUN_ANOMALIB.build_tiling_config(args)
        self.assertEqual(config.stride, 256)
        strategy = RUN_ANOMALIB.describe_spatial_strategy(args)
        self.assertEqual(strategy["spatial_strategy"], "tiled")
        self.assertEqual(strategy["tiles_per_image"], 4)

    def test_overlapping_tiles(self) -> None:
        args = self._resolve(
            ["--image-size", "512", "--tile-size", "256", "--tile-stride", "128"]
        )
        strategy = RUN_ANOMALIB.describe_spatial_strategy(args)
        self.assertEqual(strategy["grid"], [3, 3])
        self.assertAlmostEqual(strategy["overlap_ratio"], 0.5)

    def test_rejections(self) -> None:
        cases = [
            # パディングが必要（(600-256) % 256 != 0）で座標ずれの恐れがある。
            ["--image-size", "600", "--tile-size", "256"],
            # タイルが作業解像度より大きい。
            ["--image-size", "256", "--tile-size", "512"],
            # stride > tile。
            ["--image-size", "512", "--tile-size", "128", "--tile-stride", "256"],
            # --tile-size なしの --tile-stride。
            ["--image-size", "512", "--tile-stride", "128"],
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit) as ctx:
                    self._resolve(extra)
                self.assertEqual(ctx.exception.code, 2)

    def test_efficientad_rejected_for_tiler_reason(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            self._resolve(
                [
                    "--model",
                    "efficient_ad",
                    "--manifest-only",
                    "--image-size",
                    "512",
                    "--tile-size",
                    "256",
                ]
            )
        self.assertIn("tiler", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
