"""計測値の要約統計。

numpy などの重い依存を使わず、線形補間で百分位を計算する。
"""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Sequence


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """昇順に並んだ値から、線形補間で百分位を計算する。"""
    if not sorted_values:
        raise ValueError("百分位を計算する値がありません")

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])

    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_latencies(values_ms: Sequence[float]) -> dict[str, float | int | None]:
    """ミリ秒単位の計測値を要約する。

    空の入力では count だけを 0 とし、それ以外は None を返す。
    """
    values = [float(value) for value in values_ms]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }

    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 50),
        "p95": _percentile(sorted_values, 95),
        "max": sorted_values[-1],
    }
