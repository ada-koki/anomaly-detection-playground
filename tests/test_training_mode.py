"""adbench.training_mode の単体テスト（依存なし）。"""

from __future__ import annotations

import unittest
from pathlib import Path

from adbench import training_mode


class DetermineTrainingModeTest(unittest.TestCase):
    def test_without_checkpoint_means_cpu_train(self) -> None:
        self.assertEqual(
            training_mode.determine_training_mode(None),
            training_mode.TRAINING_MODE_CPU_TRAIN_CPU_INFER,
        )

    def test_checkpoint_without_provenance_is_external(self) -> None:
        # 既存呼び出し（第 2 引数なし）の後方互換。
        self.assertEqual(
            training_mode.determine_training_mode("model.ckpt"),
            training_mode.TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )
        self.assertEqual(
            training_mode.determine_training_mode(Path("dir/model.ckpt")),
            training_mode.TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )

    def test_checkpoint_with_none_provenance_is_external(self) -> None:
        self.assertEqual(
            training_mode.determine_training_mode(
                "model.ckpt", provenance_train_device=None
            ),
            training_mode.TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )

    def test_provenance_cpu_stays_external(self) -> None:
        self.assertEqual(
            training_mode.determine_training_mode(
                "model.ckpt", provenance_train_device="cpu"
            ),
            training_mode.TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )
        self.assertEqual(
            training_mode.determine_training_mode(
                "model.ckpt", provenance_train_device="CPU:0"
            ),
            training_mode.TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )

    def test_provenance_cuda_is_gpu_train_cpu_infer(self) -> None:
        for value in ("cuda", "cuda:0", "CUDA:1"):
            with self.subTest(value=value):
                self.assertEqual(
                    training_mode.determine_training_mode(
                        "model.ckpt", provenance_train_device=value
                    ),
                    training_mode.TRAINING_MODE_GPU_TRAIN_CPU_INFER,
                )

    def test_provenance_with_unknown_value_is_external(self) -> None:
        self.assertEqual(
            training_mode.determine_training_mode(
                "model.ckpt", provenance_train_device="unknown"
            ),
            training_mode.TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )
        self.assertEqual(
            training_mode.determine_training_mode(
                "model.ckpt", provenance_train_device="tpu"
            ),
            training_mode.TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER,
        )

    def test_checkpoint_none_takes_precedence_over_provenance(self) -> None:
        # このプロセスで学習した場合、来歴引数より「学習した」事実を優先する。
        self.assertEqual(
            training_mode.determine_training_mode(
                None, provenance_train_device="cuda"
            ),
            training_mode.TRAINING_MODE_CPU_TRAIN_CPU_INFER,
        )


if __name__ == "__main__":
    unittest.main()
