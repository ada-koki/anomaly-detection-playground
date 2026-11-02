"""torch に依存しないレイテンシ計測ヘルパーのテスト。"""

import importlib.util
import unittest

from adbench.latency import _tensor_hw, select_latency_images


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class LatencySelectionTests(unittest.TestCase):
    def test_warmup_and_measurement_split(self):
        warmup, measured = select_latency_images(
            ["a", "b", "c", "d"],
            warmup=2,
            max_images=1,
        )
        self.assertEqual(warmup, ["a", "b"])
        self.assertEqual(measured, ["c"])

    def test_max_images_none_uses_all_remaining(self):
        warmup, measured = select_latency_images(
            ["a", "b", "c"],
            warmup=1,
            max_images=None,
        )
        self.assertEqual(warmup, ["a"])
        self.assertEqual(measured, ["b", "c"])

    def test_max_images_zero_uses_all_remaining(self):
        warmup, measured = select_latency_images(
            ["a", "b", "c"],
            warmup=1,
            max_images=0,
        )
        self.assertEqual(warmup, ["a"])
        self.assertEqual(measured, ["b", "c"])

    def test_warmup_larger_than_input(self):
        warmup, measured = select_latency_images(
            ["a"],
            warmup=3,
            max_images=None,
        )
        self.assertEqual(warmup, ["a"])
        self.assertEqual(measured, [])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            select_latency_images([], warmup=-1, max_images=None)
        with self.assertRaises(ValueError):
            select_latency_images([], warmup=0, max_images=-1)


class TensorShapeHelperTests(unittest.TestCase):
    def test_tensor_hw_with_shape_like_object(self):
        class ShapeLike:
            shape = (1, 3, 8, 16)

        self.assertEqual(_tensor_hw(ShapeLike()), [8, 16])

    def test_tensor_hw_with_list_shape(self):
        class ShapeLike:
            shape = [2, 3, 4, 5]

        self.assertEqual(_tensor_hw(ShapeLike()), [4, 5])

    def test_tensor_hw_with_too_few_dims(self):
        class ShapeLike:
            shape = (4,)

        self.assertIsNone(_tensor_hw(ShapeLike()))

    def test_tensor_hw_without_shape_attribute(self):
        self.assertIsNone(_tensor_hw(object()))
        self.assertIsNone(_tensor_hw(None))


@unittest.skipUnless(TORCH_AVAILABLE, "torch がインストールされていません")
class TorchTensorHelperTests(unittest.TestCase):
    def test_tensor_hw_with_torch_tensor(self):
        import torch

        tensor = torch.zeros(1, 3, 8, 16)
        self.assertEqual(_tensor_hw(tensor), [8, 16])


if __name__ == "__main__":
    unittest.main()

