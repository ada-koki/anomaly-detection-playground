"""正規化に依存しない生スコア評価（検証用）。

Anomalib 2.7.0 の PostProcessor は検証集合（val）から ``image_min`` /
``image_max`` と閾値を推定し、生スコアを
``((pred - threshold) / (max - min) + 0.5).clamp(0, 1)`` で正規化する。
検証集合が空になる条件（``val_split_mode=none``）ではこれらの値が NaN のままに
なり、``engine.test`` のメトリクスが実行ごとに変動する現象が実機で確認されている
（Issue #7）。正規化の範囲外のスコアは 0/1 に飽和して同値が生じ得るため、
``from_test`` でも AUROC が正規化の影響を受ける可能性がある。

本モジュールは PostProcessor を通さない生の ``pred_score`` と ``anomaly_map`` から
image AUROC と pixel AUROC を計算し、正規化の影響を排除した検証用の値を得る。

重い依存（torch, scikit-learn, anomalib, numpy, PIL）はすべて関数内で遅延 import
するため、本モジュールの import 自体は軽い。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

# 256px×100枚相当の画素数。メモリ見積りの目安として notes に記録する。
REFERENCE_PIXEL_COUNT = 100 * 256 * 256


def count_tied_scores(scores: Sequence[float]) -> int:
    """同一値を持つスコアの件数を返す。

    同値グループに属する全要素を数える（例: ``[0.1, 0.2, 0.2]`` は 2）。
    NaN は同値とみなさない（float の NaN は自身と等しくないため）。
    """
    counts: dict[float, int] = {}
    for score in scores:
        value = float(score)
        counts[value] = counts.get(value, 0) + 1
    return sum(count for count in counts.values() if count > 1)


def binary_auroc_from_values(
    scores: Sequence[float],
    targets: Sequence[int],
) -> float | None:
    """生スコアと0/1ラベルからAUROCを計算する。

    scikit-learn は関数内で遅延 import する。空入力またはラベルが
    単一クラスの場合は AUROC が定義できないため ``None`` を返す。

    Args:
        scores: 画像ごとの生スコア。
        targets: 画像ごとの正解ラベル（0=正常, 1=異常）。

    Returns:
        AUROC。計算できない場合は ``None``。

    Raises:
        ValueError: scores と targets の件数が一致しない場合。
    """
    score_list = [float(score) for score in scores]
    target_list = [int(target) for target in targets]
    if len(score_list) != len(target_list):
        raise ValueError("scores と targets の件数が一致しません")
    if not score_list:
        return None
    if len(set(target_list)) < 2:
        return None

    # torchmetrics の binary_auroc は [0, 1] 外の値にシグモイドを掛けるため、
    # 生スコア（例: 20〜300）では飽和して 0.5 になる。順位に基づく sklearn を使う。
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(target_list, score_list))


def resize_mask_nearest(mask: Any, height: int, width: int) -> Any:
    """2値マスクを最近傍で ``(height, width)`` へリサイズする。

    numpy は関数内で遅延 import する。

    Args:
        mask: 2次元のマスク配列（0より大きい値を異常とみなす）。
        height: 出力の高さ。
        width: 出力の幅。

    Returns:
        uint8（0/1）の2次元配列。

    Raises:
        ValueError: mask が2次元でない、空、またはサイズ指定が不正な場合。
    """
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask は2次元配列で指定してください: shape={array.shape}")
    if height < 1 or width < 1:
        raise ValueError("height と width には1以上を指定してください")

    src_h, src_w = int(array.shape[0]), int(array.shape[1])
    if src_h == 0 or src_w == 0:
        raise ValueError("mask が空です")

    binary = (array > 0).astype(np.uint8)
    if (src_h, src_w) == (height, width):
        return binary

    rows = np.floor((np.arange(height) + 0.5) * src_h / height).astype(np.int64)
    cols = np.floor((np.arange(width) + 0.5) * src_w / width).astype(np.int64)
    rows = np.clip(rows, 0, src_h - 1)
    cols = np.clip(cols, 0, src_w - 1)
    return binary[np.ix_(rows, cols)]


def pixel_auroc_from_arrays(anomaly_maps: Any, masks: Any) -> float | None:
    """全画像分を連結した画素値とマスクから画素AUROCを1回だけ計算する。

    Args:
        anomaly_maps: float32 の1次元配列（全画像分を連結済み）。
        masks: 0/1 の1次元配列。anomaly_maps と同じ長さ。

    Returns:
        pixel AUROC。画素が無い、または異常画素が1つも無い場合は ``None``。

    Raises:
        ValueError: 長さが一致しない場合。
    """
    import numpy as np

    maps_array = np.asarray(anomaly_maps, dtype=np.float32).reshape(-1)
    masks_array = np.asarray(masks).reshape(-1)
    if maps_array.size != masks_array.size:
        raise ValueError("anomaly_maps と masks の長さが一致しません")
    if maps_array.size == 0 or not bool(np.any(masks_array)):
        return None

    # 生の anomaly_map は [0, 1] 外の値を取るため、シグモイド変換をしない sklearn を使う。
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(masks_array > 0, maps_array))


def _sample_records(samples: Any) -> list[dict[str, Any]]:
    """Anomalibのsamplesをレコード一覧へ変換する。"""
    if samples is None:
        raise ValueError("テスト画像一覧（samples）がありません")

    to_dict = getattr(samples, "to_dict", None)
    if callable(to_dict):
        return [dict(record) for record in to_dict("records")]

    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
        return [dict(record) for record in samples]

    raise ValueError("samples をレコード一覧へ変換できません")


def _text_or_none(value: Any) -> str | None:
    """空文字・NaN・None 相当を ``None`` に正規化する。"""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if text in {"", "nan", "None", "<NA>"}:
        return None
    return text


def _field(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _pred_score_scalar(inference: Any) -> float:
    """モデル出力から生の pred_score（スカラー）を取り出す。"""
    import torch

    value = _field(inference, "pred_score")
    if value is None:
        raise RuntimeError("model.model の出力に pred_score がありません")
    flat = torch.as_tensor(value).detach().float().reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError("pred_score が空です")
    return float(flat[0])


def _anomaly_map_tensor(inference: Any) -> Any:
    """モデル出力から anomaly_map を2次元テンソルとして取り出す。"""
    import torch

    value = _field(inference, "anomaly_map")
    if value is None:
        raise RuntimeError("model.model の出力に anomaly_map がありません")
    tensor = torch.as_tensor(value).detach().float().cpu()
    while tensor.dim() > 2:
        tensor = tensor[0]
    if tensor.dim() != 2:
        raise RuntimeError(f"anomaly_map を2次元に変換できません: shape={tuple(tensor.shape)}")
    return tensor


def _label_index(record: Mapping[str, Any]) -> int:
    """レコードの label_index を0/1に正規化する。"""
    value = record.get("label_index")
    try:
        return 1 if int(value) else 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        label = str(record.get("label", "")).strip()
        return 0 if label == "good" else 1


def evaluate_raw_score_metrics(
    model: Any,
    samples: Any,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """学習済みモデルでtest画像を1枚ずつ処理し、生スコアのメトリクスを返す。

    ``post_processor`` を通さない ``pred_score`` / ``anomaly_map`` を使うため、
    検証集合に依存する正規化の影響を受けない。画素AUROCはメモリ節約のため、
    全画像分の uint8 マスクと float32 マップを連結して1回だけ計算する。

    Args:
        model: 学習済みのAnomalibモデル。
        samples: ``datamodule.test_data.samples``（pandas DataFrame など）。
        device: 推論デバイス（``"cpu"`` / ``"cuda"``）。

    Returns:
        image/pixel AUROC、件数、同値スコア数、注意書きを含む辞書。

    Raises:
        ValueError: samples が空、またはレコードへ変換できない場合。
        RuntimeError: モデル出力に必要な項目が無い場合。
    """
    records = _sample_records(samples)
    if not records:
        raise ValueError("テスト画像一覧が空です")

    import numpy as np
    import torch
    from anomalib.data.utils import read_image
    from PIL import Image as PILImage

    torch_device = torch.device(device)
    model.eval()
    pre_processor = getattr(model, "pre_processor", None)

    image_scores: list[float] = []
    image_labels: list[int] = []
    map_tensors: list[Any] = []
    mask_chunks: list[Any] = []
    map_size: tuple[int, int] | None = None

    with torch.inference_mode():
        for record in records:
            image_path = _text_or_none(record.get("image_path"))
            if image_path is None:
                raise ValueError("samples に image_path がありません")

            image = read_image(image_path, as_tensor=True)
            input_tensor = image.unsqueeze(0)
            preprocessed = (
                pre_processor(input_tensor) if pre_processor is not None else input_tensor
            )
            if isinstance(preprocessed, Mapping):
                preprocessed = next(iter(preprocessed.values()))
            preprocessed = preprocessed.to(torch_device)

            inference = model.model(preprocessed)
            image_scores.append(_pred_score_scalar(inference))
            image_labels.append(_label_index(record))

            anomaly_map = _anomaly_map_tensor(inference)
            height, width = int(anomaly_map.shape[0]), int(anomaly_map.shape[1])
            if map_size is None:
                map_size = (height, width)
            elif map_size != (height, width):
                raise RuntimeError(
                    "anomaly_map の空間サイズが画像ごとに異なります: "
                    f"{map_size} と {(height, width)}"
                )

            mask_path = _text_or_none(record.get("mask_path"))
            if mask_path is None:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                with PILImage.open(mask_path) as mask_image:
                    mask_array = np.asarray(mask_image.convert("L"))
                mask = resize_mask_nearest(mask_array, height, width)

            # inference_mode 内では .numpy() が使えないため、テンソルのまま溜める。
            map_tensors.append(anomaly_map.reshape(-1))
            mask_chunks.append(mask.reshape(-1))

    if map_tensors:
        maps_tensor = torch.cat(map_tensors)
        maps = maps_tensor.detach().cpu().numpy()
        del maps_tensor
    else:
        maps = np.zeros((0,), dtype=np.float32)
    masks = (
        np.concatenate(mask_chunks) if mask_chunks else np.zeros((0,), dtype=np.uint8)
    )
    del map_tensors, mask_chunks

    pixel_count = int(maps.size)
    image_auroc = binary_auroc_from_values(image_scores, image_labels)
    pixel_auroc = pixel_auroc_from_arrays(maps, masks)
    tied_image_scores = count_tied_scores(image_scores)
    nan_scores = sum(1 for score in image_scores if math.isnan(score))
    finite_scores = [score for score in image_scores if not math.isnan(score)]

    notes_parts = [
        "post_processorによる正規化を通さない生のpred_score/anomaly_mapで計算しています。",
        "pixel AUROCのマスクはanomaly_mapの空間サイズへ最近傍リサイズし、"
        "正常画像はゼロマスクとしています。",
        "画素は全画像分を連結して1回だけAUROCを計算しています。",
    ]
    if pixel_count > REFERENCE_PIXEL_COUNT:
        notes_parts.append(
            f"想定（{REFERENCE_PIXEL_COUNT}画素 = 256px×100枚相当）を超える"
            f"{pixel_count}画素を処理しました。"
        )
    if image_auroc is None:
        notes_parts.append("ラベルが単一クラスのためimage AUROCを計算できませんでした。")
    elif math.isnan(image_auroc):
        image_auroc = None
        notes_parts.append("image AUROCがNaNになったためnullとしています。")
    if pixel_auroc is None:
        notes_parts.append("異常画素が無いためpixel AUROCを計算できませんでした。")
    elif math.isnan(pixel_auroc):
        pixel_auroc = None
        notes_parts.append("pixel AUROCがNaNになったためnullとしています。")
    if tied_image_scores:
        notes_parts.append(
            f"同値のimageスコアが{tied_image_scores}件あります（丸め・飽和・重複の可能性）。"
        )
    if nan_scores:
        notes_parts.append(f"NaNのimageスコアが{nan_scores}件あります。")

    return {
        "image_AUROC_raw": None if image_auroc is None else round(image_auroc, 6),
        "pixel_AUROC_raw": None if pixel_auroc is None else round(pixel_auroc, 6),
        "image_count": len(image_scores),
        "image_positive_count": int(sum(image_labels)),
        "image_negative_count": int(len(image_labels) - sum(image_labels)),
        "pixel_count": pixel_count,
        "positive_pixel_count": int(np.count_nonzero(masks)),
        "tied_image_scores": tied_image_scores,
        "nan_image_scores": nan_scores,
        "image_score_min": min(finite_scores) if finite_scores else None,
        "image_score_max": max(finite_scores) if finite_scores else None,
        "anomaly_map_size": list(map_size) if map_size is not None else None,
        "device": str(torch_device),
        "normalization": "none(post_processor を通さない生スコア)",
        "notes": " ".join(notes_parts),
    }
