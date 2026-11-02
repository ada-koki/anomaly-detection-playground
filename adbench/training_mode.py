"""学習・推論モードの判定。

torch / anomalib などの重い依存を持たず、unittest からそのまま利用できる。
"""

from __future__ import annotations

from pathlib import Path

from adbench.devices import normalize_device

TRAINING_MODE_CPU_TRAIN_CPU_INFER = "cpu_train_cpu_infer"
TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER = "external_checkpoint_cpu_infer"
# Issue #11: 来歴サイドカーで GPU 学習が証明されたチェックポイントを CPU 推論する場合。
TRAINING_MODE_GPU_TRAIN_CPU_INFER = "gpu_train_cpu_infer"


def determine_training_mode(
    checkpoint: str | Path | None,
    provenance_train_device: str | None = None,
) -> str:
    """checkpoint の有無と来歴からレポート用の training_mode を返す。

    Args:
        checkpoint: 外部学習済みチェックポイント。``None`` の場合は学習を実行する。
        provenance_train_device: 来歴サイドカーで証明された学習デバイス。
            未指定（``None``）または判定不能な場合は従来どおり
            ``external_checkpoint_cpu_infer`` を返す。後方互換のため追加した
            省略可能な引数であり、既存の ``determine_training_mode(checkpoint)``
            呼び出しはそのまま動く。

    Returns:
        ``cpu_train_cpu_infer``、``external_checkpoint_cpu_infer``、
        または ``gpu_train_cpu_infer``。
    """
    if checkpoint is None:
        # このプロセスで学習した。学習後にチェックポイントを保存した場合も同じ。
        return TRAINING_MODE_CPU_TRAIN_CPU_INFER
    if provenance_train_device is not None:
        try:
            train_device = normalize_device(provenance_train_device)
        except ValueError:
            train_device = None
        if train_device == "cuda":
            return TRAINING_MODE_GPU_TRAIN_CPU_INFER
    return TRAINING_MODE_EXTERNAL_CHECKPOINT_CPU_INFER
