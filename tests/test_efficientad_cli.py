"""EfficientAD の CLI 検証に関する依存なしテスト。"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from adbench.training_mode import (
    TRAINING_MODE_CPU_TRAIN_CPU_INFER,
    TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
    determine_training_mode,
)
from scripts.run_anomalib import build_parser, parse_and_resolve, validate_args


def _validate(argv: list[str]) -> Path:
    args, _, _, _ = parse_and_resolve(argv)
    return validate_args(args, build_parser())


def _expect_parser_error(testcase: unittest.TestCase, argv: list[str]) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with testcase.assertRaises(SystemExit) as context:
            _validate(argv)
    testcase.assertEqual(context.exception.code, 2)
    testcase.assertTrue(stderr.getvalue())


class EfficientAdCliTest(unittest.TestCase):
    def test_training_mode_pure_function(self) -> None:
        self.assertEqual(determine_training_mode(None), TRAINING_MODE_CPU_TRAIN_CPU_INFER)
        self.assertEqual(
            determine_training_mode(Path("model.ckpt")),
            TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )

    def test_non_efficient_ad_rejects_non_one_max_epochs(self) -> None:
        _expect_parser_error(self, ["--model", "padim", "--max-epochs", "2"])

    def test_non_efficient_ad_rejects_max_steps(self) -> None:
        _expect_parser_error(self, ["--model", "padim", "--max-steps", "10"])

    def test_efficient_ad_rejects_train_batch_size_not_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            imagenet_dir = Path(tmp) / "imagenette2"
            imagenet_dir.mkdir()
            _expect_parser_error(
                self,
                [
                    "--model",
                    "efficient_ad",
                    "--train-batch-size",
                    "2",
                    "--imagenet-dir",
                    str(imagenet_dir),
                ],
            )

    def test_efficient_ad_rejects_missing_imagenet_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-imagenette2"
            _expect_parser_error(
                self,
                [
                    "--model",
                    "efficient_ad",
                    "--imagenet-dir",
                    str(missing),
                ],
            )

    def test_checkpoint_missing_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "missing.ckpt"
            _expect_parser_error(self, ["--checkpoint", str(checkpoint)])

    def test_checkpoint_and_prefetch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.ckpt"
            checkpoint.write_bytes(b"dummy")
            _expect_parser_error(
                self,
                ["--checkpoint", str(checkpoint), "--prefetch-weights"],
            )

    def test_efficient_ad_resolves_protocol_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            imagenet_dir = Path(tmp) / "imagenette2"
            imagenet_dir.mkdir()
            args, _, _, _ = parse_and_resolve(
                [
                    "--model",
                    "efficient_ad",
                    "--imagenet-dir",
                    str(imagenet_dir),
                ]
            )
        self.assertEqual(args.efficientad_model_size, "small")
        self.assertEqual(args.max_epochs, 1)
        self.assertEqual(args.max_steps, -1)

    def test_efficient_ad_accepts_max_epochs_and_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            imagenet_dir = Path(tmp) / "imagenette2"
            imagenet_dir.mkdir()
            args, protocol, _, overrides = parse_and_resolve(
                [
                    "--model",
                    "efficient_ad",
                    "--imagenet-dir",
                    str(imagenet_dir),
                    "--max-epochs",
                    "2",
                    "--max-steps",
                    "5",
                ]
            )
        self.assertEqual(args.max_epochs, 2)
        self.assertEqual(args.max_steps, 5)
        self.assertEqual(protocol.models["efficient_ad"].max_epochs, 2)
        self.assertEqual(protocol.models["efficient_ad"].max_steps, 5)
        self.assertEqual(overrides["models.efficient_ad.max_epochs"], 2)
        self.assertEqual(overrides["models.efficient_ad.max_steps"], 5)

    def test_valid_fake_dataset_layout_passes_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "datasets" / "MVTecAD"
            for split in ("train", "test"):
                (root / "bottle" / split / "good").mkdir(parents=True)
            imagenet_dir = Path(tmp) / "imagenette2"
            imagenet_dir.mkdir()
            resolved_root = _validate(
                [
                    "--model",
                    "efficient_ad",
                    "--dataset-root",
                    str(root),
                    "--category",
                    "bottle",
                    "--imagenet-dir",
                    str(imagenet_dir),
                ]
            )
            self.assertEqual(Path(resolved_root), root)


if __name__ == "__main__":
    unittest.main()
