"""adbench.experiment のテスト（標準ライブラリのみ）。"""

from __future__ import annotations

import unittest

from adbench.experiment import expand, parse_config


def make_config(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "name": "test",
        "protocol": "configs/protocols/mvtec_ad_v1.json",
        "dataset_root": "datasets/MVTecAD",
        "models": ["padim", "patchcore"],
        "categories": ["bottle"],
        "image_sizes": [256],
        "cpu_threads": [8],
        "repeats": 2,
        "extra_args": [],
        "model_extra_args": {},
        "keep_checkpoints": False,
    }
    data.update(overrides)
    return data


class ExperimentExpandTest(unittest.TestCase):
    def test_expand_count_and_order(self) -> None:
        config = parse_config(
            make_config(
                models=["patchcore", "padim"],
                categories=["bottle", "grid"],
                image_sizes=[256, 192],
                cpu_threads=[8, 4],
                repeats=2,
            )
        )
        specs = expand(config)
        self.assertEqual(len(specs), 32)
        self.assertEqual(specs[0].run_key, "patchcore__bottle__s256__t8__r1")
        self.assertEqual(specs[1].run_key, "patchcore__bottle__s256__t8__r2")
        self.assertEqual(specs[2].run_key, "patchcore__bottle__s256__t4__r1")
        self.assertEqual(specs[3].run_key, "patchcore__bottle__s256__t4__r2")
        self.assertEqual(specs[4].run_key, "patchcore__bottle__s192__t8__r1")
        self.assertEqual(specs[8].run_key, "patchcore__grid__s256__t8__r1")
        self.assertEqual(specs[16].run_key, "padim__bottle__s256__t8__r1")

    def test_run_key_and_argv(self) -> None:
        config = parse_config(
            make_config(
                models=["padim"],
                categories=["metal_nut"],
                image_sizes=[320],
                cpu_threads=[0],
                repeats=1,
                extra_args=["--eval-batch-size", "2"],
                model_extra_args={"padim": ["--num-neighbors", "5"]},
            )
        )
        spec = expand(config)[0]
        self.assertEqual(spec.run_key, "padim__metal_nut__s320__t0__r1")
        argv = list(spec.argv)
        self.assertEqual(argv[0:2], ["--protocol", "configs/protocols/mvtec_ad_v1.json"])
        self.assertIn("--dataset-root", argv)
        self.assertIn("datasets/MVTecAD", argv)
        self.assertIn("--category", argv)
        self.assertIn("metal_nut", argv)
        self.assertIn("--model", argv)
        self.assertIn("padim", argv)
        self.assertIn("--image-size", argv)
        self.assertIn("320", argv)
        self.assertIn("--cpu-threads", argv)
        self.assertIn("0", argv)
        self.assertIn("--eval-batch-size", argv)
        self.assertIn("2", argv)
        self.assertIn("--num-neighbors", argv)
        self.assertIn("5", argv)

    def test_invalid_values(self) -> None:
        invalid_overrides: list[dict[str, object]] = [
            {"models": []},
            {"models": ["padim", "padim"]},
            {"models": ["padim", "unknown"]},
            {"categories": []},
            {"categories": ["bottle", "bottle"]},
            {"categories": ["../bottle"]},
            {"image_sizes": []},
            {"image_sizes": [0]},
            {"image_sizes": [True]},
            {"cpu_threads": []},
            {"cpu_threads": [-1]},
            {"repeats": 0},
            {"repeats": True},
            {"extra_args": ["--output-dir", "x"]},
            {"extra_args": ["--model=patchcore"]},
            {"model_extra_args": {"unknown": ["--x"]}},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    parse_config(make_config(**overrides))


if __name__ == "__main__":
    unittest.main()
