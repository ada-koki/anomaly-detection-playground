"""Anomalib PreProcessor の適用方法を明示的に選ぶためのヘルパー。

Anomalib 2.7 の ``PreProcessor.forward()`` はエクスポート用の
``export_transform`` を使い、そこでは ``Resize`` の antialias が無効化される
（ONNX 互換のため）。一方、Lightning の test ループ（``engine.test`` の精度指標）は
``on_test_batch_start`` で ``transform``（antialias 有効）を使う。

4096px→256px のような大きな縮小では両者の出力が大きく異なり、生スコア評価や
レイテンシ計測を ``forward()`` で行うと engine の精度指標と別条件になる。
そのため本リポジトリでは既定で ``transform`` を使い、どちらを使ったかを report に残す。
"""

from __future__ import annotations

from typing import Any

PREPROCESS_MODES = ("eval_transform", "export_transform")
DEFAULT_PREPROCESS_MODE = "eval_transform"


def select_transform(pre_processor: Any, mode: str = DEFAULT_PREPROCESS_MODE) -> Any:
    """mode に対応する変換（呼び出し可能オブジェクト）を返す。無ければ ``None``。

    Args:
        pre_processor: Anomalib の PreProcessor（``transform`` / ``export_transform`` を持つ）。
        mode: ``"eval_transform"``（Lightning の test ループと同じ）または
            ``"export_transform"``（``PreProcessor.forward()`` と同じ、antialias 無効）。

    Raises:
        ValueError: mode が不正な場合。
    """
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"preprocess mode は {PREPROCESS_MODES} のいずれかです: {mode}")
    if pre_processor is None:
        return None
    if mode == "eval_transform":
        return getattr(pre_processor, "transform", None)
    return getattr(pre_processor, "export_transform", None)


def apply_pre_processor(
    pre_processor: Any,
    batch: Any,
    mode: str = DEFAULT_PREPROCESS_MODE,
) -> Any:
    """選んだ変換をバッチテンソルに適用する。変換が無ければ入力をそのまま返す。"""
    transform = select_transform(pre_processor, mode)
    if transform is None:
        return batch
    return transform(batch)


def describe_preprocess(pre_processor: Any, mode: str = DEFAULT_PREPROCESS_MODE) -> dict[str, Any]:
    """report 用に、使った変換の種類と表現（repr）を返す。"""
    transform = select_transform(pre_processor, mode)
    return {
        "mode": mode,
        "transform": None if transform is None else " ".join(repr(transform).split()),
    }
