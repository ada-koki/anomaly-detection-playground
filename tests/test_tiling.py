"""adbench.tiling のテスト。

torch / anomalib が無い環境でも純粋ロジックのテストは実行できる。
Anomalib Tiler との互換性テストは依存がある場合のみ実行する。
"""

from __future__ import annotations

import importlib.util
import unittest

from adbench.tiling import (
    TILE_MODES,
    TilingConfig,
    axis_coverage,
    coverage_summary,
    describe_global,
    describe_tiling,
    padded_length,
    plan_tiles,
    requires_padding,
    tile_starts,
    validate_for_model,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_ANOMALIB = importlib.util.find_spec("anomalib") is not None


class TestPureTilingLogic(unittest.TestCase):
    """依存なしで実行できる純粋ロジックのテスト。"""

    def test_tile_starts_divisible(self) -> None:
        self.assertEqual(tile_starts(512, 256, 256), [0, 256])

    def test_tile_starts_with_overlap(self) -> None:
        self.assertEqual(tile_starts(512, 256, 128), [0, 128, 256])

    def test_tile_starts_length_equals_tile(self) -> None:
        self.assertEqual(tile_starts(256, 256, 128), [0])

    def test_tile_starts_raises_when_image_smaller_than_tile(self) -> None:
        with self.assertRaises(ValueError):
            tile_starts(255, 256, 128)

    def test_padded_length_matches_anomalib_formula(self) -> None:
        # 512 - 256 = 256 で stride 256 の倍数なので変化しない。
        self.assertEqual(padded_length(512, 256, 256), 512)

        # 512 - 256 = 256 で stride 128 の倍数なので変化しない。
        self.assertEqual(padded_length(512, 256, 128), 512)

        # 555 - 256 = 299。299 / 128 = 2.335... なので ceil は 3。
        # 3 * 128 + 256 = 640。
        self.assertEqual(padded_length(555, 256, 128), 640)

        # 600 - 256 = 344。344 / 256 = 1.343... なので ceil は 2。
        # 2 * 256 + 256 = 768。
        self.assertEqual(padded_length(600, 256, 256), 768)

        # 513 - 256 = 257。257 / 128 = 2.007... なので ceil は 3。
        # 3 * 128 + 256 = 640。
        self.assertEqual(padded_length(513, 256, 128), 640)

    def test_overlap_ratio(self) -> None:
        self.assertEqual(TilingConfig(256, 256).overlap_ratio, 0.0)
        self.assertEqual(TilingConfig(256, 128).overlap_ratio, 0.5)
        self.assertEqual(TilingConfig(256, 192).overlap_ratio, 0.25)

    def test_validate_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "tile_size"):
            TilingConfig(0, 1).validate()

        with self.assertRaisesRegex(ValueError, "stride"):
            TilingConfig(256, 0).validate()

        with self.assertRaisesRegex(ValueError, "stride"):
            TilingConfig(256, 257).validate()

        with self.assertRaisesRegex(ValueError, "mode"):
            TilingConfig(256, 128, mode="nearest").validate()
        self.assertIn("padding", TILE_MODES)
        self.assertIn("interpolation", TILE_MODES)

        with self.assertRaisesRegex(ValueError, "remove_border_count"):
            TilingConfig(256, 128, remove_border_count=-1).validate()

        with self.assertRaisesRegex(ValueError, "remove_border_count"):
            TilingConfig(10, 5, remove_border_count=5).validate()

    def test_plan_tiles_with_overlap(self) -> None:
        config = TilingConfig(256, 128)
        plan = plan_tiles((512, 512), config)

        self.assertEqual(plan.image_hw, (512, 512))
        self.assertEqual(plan.padded_hw, (512, 512))
        self.assertEqual(plan.grid_hw, (3, 3))
        self.assertEqual(len(plan.tiles), 9)
        self.assertEqual(plan.overlap_ratio, 0.5)
        self.assertFalse(plan.requires_padding)

        self.assertEqual(plan.tiles[0], (0, 0, 256, 256))
        self.assertEqual(plan.tiles[1], (0, 128, 256, 384))
        self.assertEqual(plan.tiles[-1], (256, 256, 512, 512))

    def test_plan_tiles_with_padding(self) -> None:
        config = TilingConfig(256, 256)
        plan = plan_tiles((600, 500), config)

        # 高さ: 600 -> 768、幅: 500 -> 512。
        self.assertEqual(plan.padded_hw, (768, 512))
        self.assertEqual(plan.grid_hw, (3, 2))
        self.assertEqual(len(plan.tiles), 6)
        self.assertTrue(plan.requires_padding)
        self.assertEqual(plan.tiles[-1], (512, 256, 768, 512))

    def test_plan_tiles_rejects_image_smaller_than_tile(self) -> None:
        with self.assertRaises(ValueError):
            plan_tiles((255, 512), TilingConfig(256, 128))

    def test_requires_padding(self) -> None:
        self.assertFalse(
            requires_padding((512, 512), TilingConfig(256, 128))
        )
        self.assertTrue(
            requires_padding((600, 600), TilingConfig(256, 256))
        )

    def test_axis_coverage_sum_equals_tiles_times_tile_without_padding(self) -> None:
        coverage = axis_coverage(512, 256, 128)
        expected = [1] * 128 + [2] * 128 + [2] * 128 + [1] * 128
        self.assertEqual(coverage, expected)
        self.assertEqual(sum(coverage), 3 * 256)

        coverage_no_overlap = axis_coverage(512, 256, 256)
        self.assertEqual(coverage_no_overlap, [1] * 512)
        self.assertEqual(sum(coverage_no_overlap), 2 * 256)

    def test_coverage_summary_has_no_uncovered_pixels(self) -> None:
        plan = plan_tiles((512, 512), TilingConfig(256, 128))
        coverage = coverage_summary(plan)
        self.assertEqual(coverage["min"], 1)
        self.assertEqual(coverage["max"], 4)
        self.assertEqual(coverage["uncovered"], 0)

        padded_plan = plan_tiles((600, 500), TilingConfig(256, 256))
        padded_coverage = coverage_summary(padded_plan)
        self.assertEqual(padded_coverage["uncovered"], 0)

    def test_validate_for_model_accepts_padding_free_config(self) -> None:
        self.assertEqual(
            validate_for_model(512, TilingConfig(256, 128)),
            [],
        )
        self.assertEqual(
            validate_for_model(512, TilingConfig(256, 256)),
            [],
        )

    def test_validate_for_model_rejects_padding_required_config(self) -> None:
        errors = validate_for_model(600, TilingConfig(256, 256))
        self.assertTrue(errors)
        self.assertTrue(any("パディング" in message for message in errors))
        self.assertTrue(any("座標" in message for message in errors))

    def test_validate_for_model_rejects_tile_larger_than_image(self) -> None:
        errors = validate_for_model(256, TilingConfig(512, 256))
        self.assertTrue(errors)
        self.assertTrue(any("タイルサイズ" in message for message in errors))

    def test_describe_tiling_keys_and_values(self) -> None:
        described = describe_tiling(512, TilingConfig(256, 128))
        expected_keys = {
            "spatial_strategy",
            "working_resolution",
            "tile_size",
            "stride",
            "overlap_ratio",
            "mode",
            "remove_border_count",
            "grid",
            "tiles_per_image",
            "padded_resolution",
            "requires_padding",
            "coverage",
            "merge_rule",
            "notes",
        }
        self.assertTrue(expected_keys.issubset(described.keys()))
        self.assertEqual(described["spatial_strategy"], "tiled")
        self.assertEqual(described["working_resolution"], [512, 512])
        self.assertEqual(described["tile_size"], 256)
        self.assertEqual(described["stride"], 128)
        self.assertEqual(described["overlap_ratio"], 0.5)
        self.assertEqual(described["mode"], "padding")
        self.assertEqual(described["remove_border_count"], 0)
        self.assertEqual(described["grid"], [3, 3])
        self.assertEqual(described["tiles_per_image"], 9)
        self.assertEqual(described["padded_resolution"], [512, 512])
        self.assertFalse(described["requires_padding"])
        self.assertEqual(described["coverage"]["uncovered"], 0)
        self.assertIn("untile", described["merge_rule"])

    def test_describe_global(self) -> None:
        described = describe_global(512)
        self.assertEqual(described["spatial_strategy"], "global_resize")
        self.assertEqual(described["working_resolution"], [512, 512])
        self.assertEqual(described["tiles_per_image"], 1)


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ANOMALIB,
    "torch / anomalib が無いため Anomalib Tiler 互換性テストを skip します",
)
class TestAnomalibTilerCompatibility(unittest.TestCase):
    """Anomalib Tiler の実装との互換性テスト。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - 依存なし環境用
            raise unittest.SkipTest("torch が無いため skip します") from exc

        tiler_class = None
        import_errors: list[str] = []
        for module_name in (
            "anomalib.data.utils.tiler",
            "anomalib.data.utils.tiling",
            "anomalib.data.utils",
        ):
            try:
                module = __import__(module_name, fromlist=["Tiler"])
                tiler_class = getattr(module, "Tiler")
                break
            except (ImportError, AttributeError) as exc:
                import_errors.append(f"{module_name}: {exc}")

        if tiler_class is None:
            raise unittest.SkipTest(
                "Anomalib Tiler を import できないため skip します: "
                + " / ".join(import_errors)
            )

        cls.torch = torch
        cls.Tiler = tiler_class

    def test_tile_untile_preserves_embedded_coordinates(self) -> None:
        torch = self.torch
        height = 512
        width = 512
        rows = (
            torch.arange(height, dtype=torch.float32)
            .view(height, 1)
            .expand(height, width)
        )
        cols = (
            torch.arange(width, dtype=torch.float32)
            .view(1, width)
            .expand(height, width)
        )
        image = torch.stack([rows, cols], dim=0).unsqueeze(0)

        for stride in (256, 128):
            with self.subTest(stride=stride):
                tiler = self.Tiler(256, stride)
                tiles = tiler.tile(image)
                restored = tiler.untile(tiles)
                self.assertEqual(tuple(restored.shape), tuple(image.shape))
                self.assertTrue(
                    torch.allclose(restored, image, atol=1e-6, rtol=0.0)
                )

    def test_tile_order_and_contents_match_plan_tiles(self) -> None:
        torch = self.torch
        config = TilingConfig(256, 128)
        plan = plan_tiles((512, 512), config)
        image = torch.rand(1, 1, 512, 512)

        tiler = self.Tiler(config.tile_size, config.stride)
        tiles = tiler.tile(image)

        self.assertEqual(tiles.shape[0], len(plan.tiles))
        for index, (y0, x0, y1, x1) in enumerate(plan.tiles):
            with self.subTest(index=index):
                expected = image[:, :, y0:y1, x0:x1]
                self.assertTrue(
                    torch.allclose(tiles[index], expected, atol=1e-7, rtol=0.0)
                )

    def test_feature_untile_matches_global_pool_without_padding(self) -> None:
        torch = self.torch
        torch.manual_seed(0)
        image = torch.rand(1, 1, 512, 512)

        tiler = self.Tiler(256, 128)
        tiles = tiler.tile(image)
        feature_tiles = torch.nn.functional.avg_pool2d(tiles, kernel_size=4)
        merged = tiler.untile(feature_tiles)

        expected = torch.nn.functional.avg_pool2d(image, kernel_size=4)
        self.assertEqual(tuple(merged.shape), tuple(expected.shape))
        self.assertTrue(
            torch.allclose(merged, expected, atol=1e-5, rtol=1e-5)
        )

    def test_feature_untile_shape_with_padding_is_padded_quarter(self) -> None:
        torch = self.torch
        image = torch.zeros(1, 1, 600, 600)
        config = TilingConfig(256, 256)
        plan = plan_tiles((600, 600), config)
        self.assertTrue(plan.requires_padding)

        tiler = self.Tiler(config.tile_size, config.stride)
        tiles = tiler.tile(image)
        feature_tiles = torch.nn.functional.avg_pool2d(tiles, kernel_size=4)
        merged = tiler.untile(feature_tiles)

        padded_quarter = (
            plan.padded_hw[0] // 4,
            plan.padded_hw[1] // 4,
        )
        input_quarter = (600 // 4, 600 // 4)
        self.assertEqual(tuple(merged.shape[-2:]), padded_quarter)
        self.assertNotEqual(tuple(merged.shape[-2:]), input_quarter)

    def test_single_pixel_defect_keeps_coordinate_with_overlap(self) -> None:
        torch = self.torch
        image = torch.zeros(1, 1, 512, 512)
        defect_y = 300
        defect_x = 410
        image[0, 0, defect_y, defect_x] = 1.0

        tiler = self.Tiler(256, 128)
        restored = tiler.untile(tiler.tile(image))
        flat_index = int(torch.argmax(restored[0, 0]).item())
        row = flat_index // 512
        col = flat_index % 512

        self.assertEqual((row, col), (defect_y, defect_x))
        self.assertAlmostEqual(
            float(restored[0, 0, defect_y, defect_x].item()),
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
