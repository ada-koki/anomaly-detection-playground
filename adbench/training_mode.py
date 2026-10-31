"""学習・推論モードの判定。

torch / anomalib などの重い依存を持たず、unittest からそのまま利用できる。
"""

from __future__ import annotations

from pathlib import Path

TRAINING_MODE_CPU_TRAIN_CPU_INFER = "cpu_train_cpu_infer"
TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER = "external_checkpoint_cpu_infer"


def determine_training_mode(checkpoint: str | Path | None) -> str:
    """checkpoint の有無からレポート用の training_mode を返す。

    Args:
        checkpoint: 外部学習済みチェックポイント。``None`` の場合は学習を実行する。

    Returns:
        ``cpu_train_cpu_infer`` または ``external_checkpoint_cpu_infer``。
    """
    if checkpoint is None:
        return TRAINING_MODE_CPU_TRAIN_CPU_INFER
    return TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER
