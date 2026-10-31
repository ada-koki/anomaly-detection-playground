"""adbench.raw_eval の純粋関数テスト。

scikit-learn / numpy は環境にある場合だけ検証する。
無い場合でも、依存を必要としない分岐（空入力・単一クラス・件数不一致）は検証する。
"""

from __future__ import annotations

import importlib
import importlib.util
import unittest

from adbench.raw_eval import (
    binary_auroc_from_values,
    count_tied_scores,
    pixel_auroc_from_arrays,
    resize_mask_nearest,
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_HAS_NUMPY = _module_available("numpy")
_HAS_SKLEARN = _module_available("sklearn")


class TiedScoreTest(unittest.TestCase):
    def test_unique_scores_are_not_tied(self) -> None:
        self.assertEqual(count_tied_scores([0.1, 0.2, 0.3]), 0)

    def test_tied_scores_count_all_members(self) -> None:
        self.assertEqual(count_tied_scores([0.1, 0.2, 0.2, 0.3]), 2)

    def test_all_equal_scores_are_all_tied(self) -> None:
        self.assertEqual(count_tied_scores([1.0, 1.0, 1.0]), 3)

    def test_empty_scores(self) -> None:
        self.assertEqual(count_tied_scores([]), 0)


class ResizeMaskTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_NUMPY, "numpy が必要です")
    def test_upscale_uses_nearest(self) -> None:
        np = importlib.import_module("numpy")
        mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        resized = resize_mask_nearest(mask, 4, 4)
        self.assertEqual(resized.shape, (4, 4))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertEqual(int(resized[0, 0]), 0)
        self.assertEqual(int(resized[0, 3]), 1)
        self.assertEqual(int(resized[3, 0]), 1)

    @unittest.skipUnless(_HAS_NUMPY, "numpy が必要です")
    def test_identical_size_is_binarized(self) -> None:
        np = importlib.import_module("numpy")
        mask = np.array([[0, 2], [3, 0]], dtype=np.uint8)
        resized = resize_mask_nearest(mask, 2, 2)
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(resized, expected))

    @unittest.skipUnless(_HAS_NUMPY, "numpy が必要です")
    def test_non_2d_mask_raises(self) -> None:
        np = importlib.import_module("numpy")
        with self.assertRaises(ValueError):
            resize_mask_nearest(np.zeros((1, 2, 3), dtype=np.uint8), 2, 2)


class BinaryAurocTest(unittest.TestCase):
    def test_empty_scores_returns_none(self) -> None:
        self.assertIsNone(binary_auroc_from_values([], []))

    def test_single_class_returns_none(self) -> None:
        self.assertIsNone(binary_auroc_from_values([0.1, 0.2], [0, 0]))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            binary_auroc_from_values([0.1], [0, 1])

    @unittest.skipUnless(_HAS_SKLEARN, "scikit-learn が必要です")
    def test_perfect_separation(self) -> None:
        value = binary_auroc_from_values([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertAlmostEqual(value, 1.0, places=6)

    @unittest.skipUnless(_HAS_SKLEARN, "scikit-learn が必要です")
    def test_reversed_scores(self) -> None:
        value = binary_auroc_from_values([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
        self.assertAlmostEqual(value, 0.0, places=6)

    @unittest.skipUnless(_HAS_SKLEARN, "scikit-learn が必要です")
    def test_raw_scores_outside_unit_interval(self) -> None:
        # 回帰テスト: 生スコアは [0, 1] 外を取る。シグモイドで飽和させず順位で評価すること。
        value = binary_auroc_from_values([20.0, 35.0, 120.0, 280.0], [0, 0, 1, 1])
        self.assertAlmostEqual(value, 1.0, places=6)


class PixelAurocTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_NUMPY, "numpy が必要です")
    def test_all_zero_mask_returns_none(self) -> None:
        # 画素が空の場合は None。
        self.assertIsNone(pixel_auroc_from_arrays([], []))

    @unittest.skipUnless(_HAS_NUMPY and _HAS_SKLEARN, "numpy と scikit-learn が必要です")
    def test_pixel_auroc_perfect_separation(self) -> None:
        np = importlib.import_module("numpy")
        maps = np.array([0.1, 0.2, 0.9], dtype=np.float32)
        masks = np.array([0, 0, 1], dtype=np.uint8)
        self.assertAlmostEqual(pixel_auroc_from_arrays(maps, masks), 1.0, places=6)

    @unittest.skipUnless(_HAS_NUMPY and _HAS_SKLEARN, "numpy と scikit-learn が必要です")
    def test_pixel_raw_values_outside_unit_interval(self) -> None:
        np = importlib.import_module("numpy")
        maps = np.array([10.0, 30.0, 55.0, 90.0], dtype=np.float32)
        masks = np.array([0, 0, 1, 1], dtype=np.uint8)
        self.assertAlmostEqual(pixel_auroc_from_arrays(maps, masks), 1.0, places=6)

    @unittest.skipUnless(_HAS_NUMPY, "numpy が必要です")
    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            pixel_auroc_from_arrays([0.1, 0.2], [0])


if __name__ == "__main__":
    unittest.main()
