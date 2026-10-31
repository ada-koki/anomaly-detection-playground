"""欠陥サイズ別の検出率・見逃し率・過検出率の集計。

本モジュールは標準ライブラリだけで import でき、numpy / scipy は
``mask_stats`` の呼び出し時にのみ遅延 import する。CLI の ``--help`` や
純粋ロジックのテストは重い依存なしで動く。

用語:
    - 欠陥サイズ: 元画像解像度のマスク面積率（マスク画素数 / 元画像画素数）。
    - サイズ帯: 上記面積率を ``SIZE_BANDS_DEFAULT`` で分類したもの。
    - 画像単位検出: ``score >= image_threshold``。
    - 領域単位ヒット: ``map_max_in_mask >= pixel_threshold``。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# サイズ帯の既定定義。(name, 下限面積率, 上限面積率)。
# 下限を含み上限を含まない。最後の帯のみ上限を含む。
SIZE_BANDS_DEFAULT: tuple[tuple[str, float, float], ...] = (
    ("tiny", 0.0, 1e-3),
    ("small", 1e-3, 1e-2),
    ("large", 1e-2, 1.0),
)

# 閾値の出所（report.json に記録する文字列と同じ表現）。
THRESHOLD_SOURCE_DEFAULT = "validation（PostProcessor F1AdaptiveThreshold, 生スケール）"

# サンプル数が少ないセルに付ける注記。
SMALL_SAMPLE_NOTE = "サンプル数が少ないため参考値"

# 集計値の分母の説明。レポートにそのまま載せる。
DENOMINATORS: dict[str, str] = {
    "detection_rate": (
        "異常画像（label=1）のみを分母とする。detected は "
        "score >= image_threshold の件数。スコアが NaN の画像は未検出として数える。"
    ),
    "miss_rate": "同じ分母を使う。missed = n - detected。",
    "region_hit_rate": (
        "マスクが有り、かつ anomaly_map の解像度へ最近傍リサイズしてもマスクが残った"
        "異常画像のみを分母とする。map_max_in_mask >= pixel_threshold を領域ヒットとする。"
    ),
    "false_positive_rate": (
        "正常画像（label=0）のみを分母とする。score >= image_threshold を過検出とする。"
    ),
    "mean_normal_fp_pixel_ratio": (
        "正常画像のうち normal_fp_pixel_ratio を計算できた画像の単純平均。"
        "分母は画素ではなく画像。"
    ),
}


def is_finite_number(value: Any) -> bool:
    """有限の数値として扱えるかを返す（``None`` と NaN/Inf は False）。"""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_or_none(value: Any) -> float | None:
    if not is_finite_number(value):
        return None
    return float(value)  # type: ignore[arg-type]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    """数値化する。変換できない場合は NaN。"""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_int(value: Any) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _rate(numerator: int | None, denominator: int | None) -> float | None:
    """割合を返す。分母が 0 または None なら ``None``。"""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def classify_size(
    area_ratio: Any,
    bands: Sequence[tuple[str, float, float]] = SIZE_BANDS_DEFAULT,
) -> str | None:
    """面積率をサイズ帯名へ分類する。

    各帯は下限を含み上限を含まない。最後の帯のみ上限を含む。
    ``None`` / NaN / 範囲外（負値や 1.0 超）は ``None`` を返す。

    Args:
        area_ratio: 元画像面積に対する欠陥面積の割合。
        bands: ``(name, lower, upper)`` の並び。

    Returns:
        サイズ帯名。分類できない場合は ``None``。
    """
    if area_ratio is None:
        return None
    try:
        value = float(area_ratio)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    band_items = list(bands)
    for index, (name, lower, upper) in enumerate(band_items):
        is_last = index == len(band_items) - 1
        if is_last:
            if float(lower) <= value <= float(upper):
                return str(name)
        elif float(lower) <= value < float(upper):
            return str(name)
    return None


def mask_stats(mask_array: Any) -> dict[str, Any]:
    """マスク配列から面積・外接矩形・連結成分数を計算する。

    numpy と scipy は関数内で遅延 import する。scipy が無い場合は
    ``component_count`` と ``largest_component_px`` を ``None`` にする。

    Args:
        mask_array: 2次元のマスク配列（0 より大きい値を異常とみなす）。

    Returns:
        次のキーを持つ辞書。

        - ``mask_height`` / ``mask_width``: マスクの解像度。
        - ``area_px``: 異常画素数。
        - ``area_ratio``: ``area_px / (mask_height * mask_width)``。
        - ``bbox``: ``[y0, x0, y1, x1]``（``y1`` / ``x1`` は排他的）。空なら ``None``。
        - ``bbox_h`` / ``bbox_w``: 外接矩形の高さ・幅。空なら 0。
        - ``component_count``: 8連結の連結成分数。空なら 0、scipy 無しは ``None``。
        - ``largest_component_px``: 最大連結成分の画素数。scipy 無しは ``None``。

    Raises:
        ValueError: mask が2次元でない、または空の場合。
    """
    import numpy as np

    array = np.asarray(mask_array)
    if array.ndim != 2:
        raise ValueError(f"mask は2次元配列で指定してください: shape={array.shape}")
    height, width = int(array.shape[0]), int(array.shape[1])
    if height == 0 or width == 0:
        raise ValueError("mask が空です")

    binary = array > 0
    area_px = int(np.count_nonzero(binary))
    total = height * width
    stats: dict[str, Any] = {
        "mask_height": height,
        "mask_width": width,
        "area_px": area_px,
        "area_ratio": round(area_px / total, 12),
        "bbox": None,
        "bbox_h": 0,
        "bbox_w": 0,
        "component_count": 0 if area_px == 0 else None,
        "largest_component_px": 0 if area_px == 0 else None,
    }
    if area_px == 0:
        return stats

    rows = np.flatnonzero(binary.any(axis=1))
    cols = np.flatnonzero(binary.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    stats["bbox"] = [y0, x0, y1, x1]
    stats["bbox_h"] = y1 - y0
    stats["bbox_w"] = x1 - x0

    try:
        from scipy import ndimage
    except ImportError:
        stats["component_count"] = None
        stats["largest_component_px"] = None
        return stats

    # メモリ節約のため外接矩形の内側だけでラベリングする。
    cropped = binary[y0:y1, x0:x1]
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label(cropped, structure=structure)
    stats["component_count"] = int(count)
    if count:
        sizes = np.bincount(labels.reshape(-1), minlength=int(count) + 1)[1:]
        stats["largest_component_px"] = int(sizes.max()) if sizes.size else 0
    else:
        stats["largest_component_px"] = 0
    return stats


@dataclass
class PerImageRecord:
    """1画像分の評価結果。"""

    image_path: str
    defect_type: str
    label: int
    score: float
    mask_available: bool = False
    mask_stats: dict[str, Any] | None = None
    size_band: str | None = None
    map_max_in_mask: float | None = None
    map_max_outside_mask: float | None = None
    normal_fp_pixel_ratio: float | None = None
    mask_px_at_map_resolution: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON 化できる辞書に変換する。

        非有限値（NaN/Inf）は標準 JSON に合わせて ``None`` にする。
        """
        return {
            "image_path": self.image_path,
            "defect_type": self.defect_type,
            "label": int(self.label),
            "score": _finite_or_none(self.score),
            "mask_available": bool(self.mask_available),
            "mask_stats": None if self.mask_stats is None else dict(self.mask_stats),
            "size_band": self.size_band,
            "map_max_in_mask": _finite_or_none(self.map_max_in_mask),
            "map_max_outside_mask": _finite_or_none(self.map_max_outside_mask),
            "normal_fp_pixel_ratio": _finite_or_none(self.normal_fp_pixel_ratio),
            "mask_px_at_map_resolution": (
                None if self.mask_px_at_map_resolution is None
                else int(self.mask_px_at_map_resolution)
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerImageRecord":
        """``to_dict`` の出力から復元する。欠けているキーは既定値で補う。"""
        mask_stats_value = data.get("mask_stats")
        if mask_stats_value is not None and not isinstance(mask_stats_value, dict):
            mask_stats_value = dict(mask_stats_value)
        size_band = data.get("size_band")
        mask_px = data.get("mask_px_at_map_resolution")
        return cls(
            image_path=str(data.get("image_path") or ""),
            defect_type=str(data.get("defect_type") or "unknown"),
            label=1 if _as_int(data.get("label")) else 0,
            score=_as_float(data.get("score")),
            mask_available=bool(data.get("mask_available", False)),
            mask_stats=mask_stats_value,
            size_band=None if size_band in (None, "") else str(size_band),
            map_max_in_mask=_optional_float(data.get("map_max_in_mask")),
            map_max_outside_mask=_optional_float(data.get("map_max_outside_mask")),
            normal_fp_pixel_ratio=_optional_float(data.get("normal_fp_pixel_ratio")),
            mask_px_at_map_resolution=None if mask_px is None else _as_int(mask_px),
        )


def _score_of(record: PerImageRecord) -> float:
    return _as_float(record.score)


def _defect_key(record: PerImageRecord) -> str:
    text = str(record.defect_type or "").strip()
    return text or "unknown"


def _cell(
    records: Sequence[PerImageRecord],
    image_threshold: float | None,
    pixel_threshold: float | None,
    min_samples: int,
) -> dict[str, Any]:
    """異常画像の集合を1セル分に集計する。"""
    n = len(records)
    image_ok = is_finite_number(image_threshold)
    pixel_ok = is_finite_number(pixel_threshold)
    notes: list[str] = []

    if n == 0:
        notes.append(SMALL_SAMPLE_NOTE)
        notes.append("n=0 のため率は算出できません")
    elif n < int(min_samples):
        notes.append(SMALL_SAMPLE_NOTE)

    nan_scores = sum(1 for record in records if not math.isfinite(_score_of(record)))
    if nan_scores:
        notes.append(f"スコアが NaN の画像が{nan_scores}件あり、未検出として数えています")

    if image_ok:
        threshold = float(image_threshold)  # type: ignore[arg-type]
        detected: int | None = sum(1 for record in records if _score_of(record) >= threshold)
        missed: int | None = n - detected
        detection_rate = _rate(detected, n)
        miss_rate = _rate(missed, n)
    else:
        detected = None
        missed = None
        detection_rate = None
        miss_rate = None
        notes.append("閾値未計算（image_threshold）")

    mask_present = [record for record in records if bool(record.mask_available)]
    n_without_mask = n - len(mask_present)
    evaluable_region = [
        record for record in mask_present if record.map_max_in_mask is not None
    ]
    n_mask_vanished = len(mask_present) - len(evaluable_region)

    if pixel_ok:
        pixel_limit = float(pixel_threshold)  # type: ignore[arg-type]
        region_hit: int | None = sum(
            1
            for record in evaluable_region
            if float(record.map_max_in_mask) >= pixel_limit  # type: ignore[arg-type]
        )
        region_hit_rate = _rate(region_hit, len(evaluable_region))
    else:
        region_hit = None
        region_hit_rate = None
        notes.append("閾値未計算（pixel_threshold）")

    if n_without_mask:
        notes.append(
            f"マスクが無い異常画像が{n_without_mask}件あり、領域単位指標の分母から除外しています"
        )
    if n_mask_vanished:
        notes.append(
            "作業解像度への最近傍リサイズでマスクが消失した異常画像が"
            f"{n_mask_vanished}件あり、領域単位指標の分母から除外しています"
        )

    return {
        "n": n,
        "detected": detected,
        "missed": missed,
        "detection_rate": detection_rate,
        "miss_rate": miss_rate,
        "region_hit": region_hit,
        "region_hit_rate": region_hit_rate,
        "n_region_evaluable": len(evaluable_region),
        "n_without_mask": n_without_mask,
        "n_mask_vanished": n_mask_vanished,
        "note": " ".join(notes),
    }


def _normal_summary(
    normal: Sequence[PerImageRecord],
    image_threshold: float | None,
    min_samples: int,
) -> dict[str, Any]:
    """正常画像の過検出（画像単位・画素単位）を集計する。"""
    n = len(normal)
    notes: list[str] = []
    if n == 0:
        notes.append("正常画像がありません")
    elif n < int(min_samples):
        notes.append(SMALL_SAMPLE_NOTE)

    if is_finite_number(image_threshold):
        threshold = float(image_threshold)  # type: ignore[arg-type]
        false_positive: int | None = sum(
            1 for record in normal if _score_of(record) >= threshold
        )
        false_positive_rate = _rate(false_positive, n)
    else:
        false_positive = None
        false_positive_rate = None
        notes.append("閾値未計算（image_threshold）")

    ratios = [
        float(record.normal_fp_pixel_ratio)
        for record in normal
        if is_finite_number(record.normal_fp_pixel_ratio)
    ]
    mean_ratio = round(sum(ratios) / len(ratios), 6) if ratios else None
    if ratios and len(ratios) < n:
        notes.append(f"normal_fp_pixel_ratio を計算できた正常画像は{len(ratios)}件です")
    if not ratios and n:
        notes.append("normal_fp_pixel_ratio を計算できた正常画像がありません")

    return {
        "n_normal": n,
        "false_positive": false_positive,
        "false_positive_rate": false_positive_rate,
        "mean_normal_fp_pixel_ratio": mean_ratio,
        "n_with_pixel_fp_ratio": len(ratios),
        "note": " ".join(notes),
    }


def summarize(
    records: Sequence[PerImageRecord],
    image_threshold: float | None,
    pixel_threshold: float | None,
    *,
    min_samples: int = 5,
    bands: Sequence[tuple[str, float, float]] = SIZE_BANDS_DEFAULT,
) -> dict[str, Any]:
    """画像レコードからサイズ帯別・異常種別別の集計を作る。

    Args:
        records: 画像ごとのレコード。
        image_threshold: 画像スコアの閾値（検証集合由来の生スケール）。
        pixel_threshold: anomaly_map の閾値（検証集合由来の生スケール）。
        min_samples: これ未満のセルに「参考値」の注記を付ける。
        bands: サイズ帯の定義。

    Returns:
        閾値・帯定義・分母説明・各集計セル・正常画像集計を含む辞書。
        閾値が NaN / None の場合は該当する率を ``None`` にし、注記に
        「閾値未計算」を含める。
    """
    record_list = list(records)
    band_items = [
        (str(name), float(lower), float(upper)) for name, lower, upper in bands
    ]
    band_names = [name for name, _lower, _upper in band_items]

    abnormal = [record for record in record_list if int(record.label) == 1]
    normal = [record for record in record_list if int(record.label) != 1]

    overall = _cell(abnormal, image_threshold, pixel_threshold, min_samples)

    by_size_band = {
        name: _cell(
            [record for record in abnormal if record.size_band == name],
            image_threshold,
            pixel_threshold,
            min_samples,
        )
        for name in band_names
    }

    defect_types = sorted({_defect_key(record) for record in abnormal})
    by_defect_type = {
        defect: _cell(
            [record for record in abnormal if _defect_key(record) == defect],
            image_threshold,
            pixel_threshold,
            min_samples,
        )
        for defect in defect_types
    }
    by_defect_and_band = {
        defect: {
            name: _cell(
                [
                    record
                    for record in abnormal
                    if _defect_key(record) == defect and record.size_band == name
                ],
                image_threshold,
                pixel_threshold,
                min_samples,
            )
            for name in band_names
        }
        for defect in defect_types
    }

    normal_summary = _normal_summary(normal, image_threshold, min_samples)
    without_band = sum(1 for record in abnormal if record.size_band not in band_names)

    notes: list[str] = [
        "画像単位の検出は score >= image_threshold で判定しています。",
        "領域単位ヒットは map_max_in_mask >= pixel_threshold で判定しています。",
        "サイズは元解像度マスクの面積率で分類し、anomaly_map 側の解像度には依存しません。",
        "anomaly_map は作業解像度のため、微小欠陥は最近傍リサイズで消失し得ます。",
        "閾値は検証集合から求め、テスト集合では調整していません。",
    ]
    if not is_finite_number(image_threshold):
        notes.append(
            "image_threshold が未計算です。画像単位の検出率・見逃し率は null になります。"
        )
    if not is_finite_number(pixel_threshold):
        notes.append(
            "pixel_threshold が未計算です。領域単位ヒット率と normal_fp_pixel_ratio は null になります。"
        )
    if without_band:
        notes.append(
            "サイズ帯に分類できない（マスクが無い）異常画像が"
            f"{without_band}件あります。overall と異常種別別には含まれます。"
        )

    return {
        "thresholds": {
            "image": _finite_or_none(image_threshold),
            "pixel": _finite_or_none(pixel_threshold),
            "source": THRESHOLD_SOURCE_DEFAULT,
            "available": {
                "image": is_finite_number(image_threshold),
                "pixel": is_finite_number(pixel_threshold),
            },
        },
        "bands": [
            {"name": name, "min_area_ratio": lower, "max_area_ratio": upper}
            for name, lower, upper in band_items
        ],
        "min_samples": int(min_samples),
        "counts": {
            "n_images": len(record_list),
            "n_abnormal": len(abnormal),
            "n_normal": len(normal),
            "n_abnormal_without_size_band": without_band,
        },
        "denominators": dict(DENOMINATORS),
        "overall": overall,
        "by_size_band": by_size_band,
        "by_defect_type": by_defect_type,
        "by_defect_type_and_size_band": by_defect_and_band,
        "normal": normal_summary,
        "notes": " ".join(notes),
    }


def records_to_jsonl_lines(records: Sequence[PerImageRecord]) -> list[str]:
    """レコードを JSON Lines の行（改行なし）へ変換する。"""
    return [json.dumps(record.to_dict(), ensure_ascii=False) for record in records]


def load_records_jsonl(path: str | Path) -> list[PerImageRecord]:
    """JSON Lines ファイルを読み、空行は無視する。

    Raises:
        ValueError: JSON として解析できない行がある場合。
    """
    result: list[PerImageRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} の JSON を解析できません: {exc}"
                ) from exc
            result.append(PerImageRecord.from_dict(data))
    return result
