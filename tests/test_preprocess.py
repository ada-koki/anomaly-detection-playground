"""adbench.preprocess のテスト（torch 不要の部分と、torchvision があれば実変換の確認）。"""

from __future__ import annotations

import unittest

from adbench.preprocess import (
    DEFAULT_PREPROCESS_MODE,
    PREPROCESS_MODES,
    apply_pre_processor,
    describe_preprocess,
    select_transform,
)


class _FakePreProcessor:
    def __init__(self) -> None:
        self.transform = lambda x: ("eval", x)
        self.export_transform = lambda x: ("export", x)

    def __call__(self, x: object) -> object:  # forward() 相当
        return self.export_transform(x)


class PreprocessTest(unittest.TestCase):
    def test_default_is_eval_transform(self) -> None:
        self.assertEqual(DEFAULT_PREPROCESS_MODE, "eval_transform")
        self.assertEqual(apply_pre_processor(_FakePreProcessor(), 1), ("eval", 1))

    def test_export_mode(self) -> None:
        self.assertEqual(apply_pre_processor(_FakePreProcessor(), 1, "export_transform"), ("export", 1))

    def test_none_passthrough(self) -> None:
        self.assertEqual(apply_pre_processor(None, 5), 5)
        self.assertIsNone(select_transform(object(), "eval_transform"))

    def test_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            apply_pre_processor(_FakePreProcessor(), 1, "forward")

    def test_describe(self) -> None:
        info = describe_preprocess(None, "export_transform")
        self.assertEqual(info, {"mode": "export_transform", "transform": None})
        self.assertEqual(set(PREPROCESS_MODES), {"eval_transform", "export_transform"})


class AnomalibAntialiasTest(unittest.TestCase):
    """Anomalib が入っている環境でのみ、transform と export_transform の差を確認する。"""

    def test_export_transform_disables_antialias(self) -> None:
        try:
            import torch
            from anomalib.models import Padim
        except ImportError:
            self.skipTest("anomalib/torch が無い環境")
        pre = Padim.configure_pre_processor(image_size=(32, 32))
        self.assertIn("antialias=True", describe_preprocess(pre, "eval_transform")["transform"])
        self.assertIn("antialias=False", describe_preprocess(pre, "export_transform")["transform"])
        # 一様乱数画像を 16 倍縮小すると、antialias 有りでは平均化されて分散が大きく下がる。
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(1, 3, 512, 512, generator=generator)
        eval_out = apply_pre_processor(pre, image, "eval_transform")
        export_out = apply_pre_processor(pre, image, "export_transform")
        self.assertEqual(tuple(eval_out.shape), (1, 3, 32, 32))
        self.assertEqual(tuple(export_out.shape), (1, 3, 32, 32))
        self.assertLess(float(eval_out.std()) * 3.0, float(export_out.std()))

if __name__ == "__main__":
    unittest.main()
