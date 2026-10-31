"""PatchCore 学習時の特徴量バンクのメモリ見積り（標準ライブラリのみ）。

Anomalib 2.7 の PatchCore は学習中に特徴量を ``embedding_store``（list）へ
append し、学習終了時に ``torch.vstack`` して k-center-greedy で coreset を選ぶ。
256px・wide_resnet50_2・layer2+layer3 では 1 枚あたり 32x32=1024 パッチ、
特徴次元 1536 になり、枚数が多いとホストメモリを圧迫する。

本モジュールは特徴量テンソルのバイト数だけを数えた**概算**を返す。実測値ではない
ため、レポートでは見積り（``training_memory_estimate``）と実測
（``memory.train_peak_rss_mib_sampled`` / ``memory.process_lifetime_peak_rss_mib``）
を別キーで扱うこと。torch / anomalib には依存しない。
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FEATURE_DIM = 1536
DEFAULT_STRIDE = 8
DEFAULT_DTYPE_BYTES = 4
CORESET_MODES: tuple[str, ...] = ("global", "chunked")
JL_EPS = 0.9

_MEMINFO_PATH = Path("/proc/meminfo")

_COMMON_NOTE = (
    "特徴量テンソルのバイト数だけを数えた概算であり、実測値ではありません。"
    "PyTorch アロケータの断片化、DataLoader・前処理の作業領域、"
    "KCenterGreedy の距離配列、sklearn 射影の作業領域は含みません。"
    "実測側は memory.train_peak_rss_mib_sampled と "
    "memory.process_lifetime_peak_rss_mib を参照してください。"
)

_MITIGATION_NOTE = (
    "学習時のメモリ対策の設定と事後の情報です。実測ピークは "
    "memory.train_peak_rss_mib_sampled と memory.process_lifetime_peak_rss_mib を"
    "参照してください。training_memory_estimate は特徴量テンソルのみを数えた概算であり"
    "実測値ではありません。chunked は coreset 選択をチャンク単位で行うため、選択点集合は"
    "global と一致せず、精度指標を global と同一条件として比較できません。"
)

_APPROXIMATION_CHUNKED = (
    "チャンクごとの KCenterGreedy。全体 KCenterGreedy と選択点集合は一致しない"
)


@dataclass(frozen=True)
class FeatureBankEstimate:
    """PatchCore の特徴量バンクのメモリ見積り（バイト数）。

    すべて概算であり実測値ではない。``note`` に前提と限界を日本語で残す。
    """

    n_images: int
    patches_per_image: int
    feature_dim: int
    dtype_bytes: int
    coreset_mode: str
    chunk_images: int | None
    feature_bytes: int
    peak_bytes_estimate: int
    note: str


def bytes_to_mib(value: int) -> float:
    """バイト数を MiB（小数第2位）へ変換する。"""
    return round(value / (1024.0 * 1024.0), 2)


def patches_per_image(image_size: int, stride: int = DEFAULT_STRIDE) -> int:
    """1 枚あたりのパッチ数を返す（``(image_size // stride) ** 2``）。

    Args:
        image_size: モデル入力の一辺（正方を仮定）。
        stride: 特徴マップのストライド。

    Raises:
        ValueError: 引数が不正、またはパッチ数が 0 になる場合。
    """
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise ValueError("image_size には int を指定してください")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise ValueError("stride には int を指定してください")
    if image_size < 1:
        raise ValueError("image_size には1以上を指定してください")
    if stride < 1:
        raise ValueError("stride には1以上を指定してください")
    side = image_size // stride
    if side < 1:
        raise ValueError(
            f"image_size ({image_size}) が stride ({stride}) より小さいため"
            "パッチ数を計算できません"
        )
    return side * side


def johnson_lindenstrauss_min_dim(n_samples: int, eps: float = JL_EPS) -> int:
    """Johnson-Lindenstrauss の最小次元を返す（sklearn と同じ式・同じ切り捨て）。

    ``int(4 * ln(n_samples) / (eps**2 / 2 - eps**3 / 3))``。
    sklearn の ``johnson_lindenstrauss_min_dim`` は ``astype(int64)`` で
    切り捨てるため、同じく切り捨てる。

    Raises:
        ValueError: 引数が不正な場合。
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples には1以上の int を指定してください")
    if not 0 < eps < 1:
        raise ValueError("eps には0より大きく1より小さい値を指定してください")
    denominator = (eps**2) / 2 - (eps**3) / 3
    return int(4 * math.log(n_samples) / denominator)


def _projection_dim(n_rows: int, feature_dim: int) -> int:
    """KCenterGreedy が使う SparseRandomProjection の出力次元の見積り。"""
    return max(1, min(johnson_lindenstrauss_min_dim(n_rows), feature_dim))


def plan_chunks(n_rows_total: int, chunk_rows: int) -> list[int]:
    """``n_rows_total`` 行を ``chunk_rows`` 行ずつに分割した各チャンクの行数を返す。

    端数は最後のチャンクとして 1 件返す。``n_rows_total`` が 0 なら空リスト。

    Raises:
        ValueError: 引数が不正な場合。
    """
    if isinstance(n_rows_total, bool) or not isinstance(n_rows_total, int):
        raise ValueError("n_rows_total には int を指定してください")
    if n_rows_total < 0:
        raise ValueError("n_rows_total には0以上を指定してください")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int):
        raise ValueError("chunk_rows には int を指定してください")
    if chunk_rows < 1:
        raise ValueError("chunk_rows には1以上を指定してください")
    if n_rows_total == 0:
        return []
    full, remainder = divmod(n_rows_total, chunk_rows)
    sizes = [chunk_rows] * full
    if remainder:
        sizes.append(remainder)
    return sizes


def chunk_coreset_size(n_rows: int, ratio: float) -> int:
    """チャンク内で選ぶ行数を返す。

    Anomalib の ``KCenterGreedy`` と同じく ``int(n_rows * ratio)`` で切り捨てるが、
    0 件になると coreset が空になるため 1 件を下限とする。

    Raises:
        ValueError: 引数が不正な場合。
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ValueError("n_rows には1以上の int を指定してください")
    if not 0 < ratio <= 1:
        raise ValueError("ratio には0より大きく1以下の値を指定してください")
    return min(n_rows, max(1, int(n_rows * ratio)))


def estimate_patchcore_feature_memory(
    n_images: int,
    image_size: int,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    stride: int = DEFAULT_STRIDE,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    coreset_mode: str = "global",
    chunk_images: int | None = None,
    sampling_ratio: float = 0.1,
    projection_dim: int | None = None,
) -> FeatureBankEstimate:
    """PatchCore 学習時の特徴量バンクのピークメモリを概算する。

    ``coreset_mode="global"`` は「embedding_store の全量 + vstack 後の全量（2 倍）
    + 射影特徴」、``coreset_mode="chunked"`` は「1 チャンク分の vstack（2 倍）+
    チャンク射影 + 選択済みバンク（全行数 × sampling_ratio 相当）」を数える。

    射影特徴の次元は ``projection_dim`` が None のとき Johnson-Lindenstrauss の
    最小次元（``eps=0.9``、上限 ``feature_dim``）で見積もる。chunked では
    チャンクごとに射影を作り直すため、チャンク行数で計算する。

    本関数の戻り値は**概算**であり実測値ではない（``note`` を参照）。

    Args:
        n_images: 学習画像枚数。
        image_size: モデル入力の一辺（正方を仮定）。
        feature_dim: パッチ特徴の次元。既定は wide_resnet50_2 の layer2+layer3
            （512+1024=1536）。
        stride: 特徴マップのストライド。既定は 8。
        dtype_bytes: 特徴量の要素バイト数。float32 なら 4。
        coreset_mode: ``"global"`` または ``"chunked"``。
        chunk_images: chunked のときの 1 チャンクあたり画像枚数。
        sampling_ratio: coreset のサンプリング比率。
        projection_dim: 射影次元の明示指定。None なら自動見積り。

    Returns:
        見積り結果。

    Raises:
        ValueError: 引数が不正な場合。
    """
    if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 1:
        raise ValueError("n_images には1以上の int を指定してください")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("feature_dim には1以上の int を指定してください")
    if isinstance(dtype_bytes, bool) or not isinstance(dtype_bytes, int) or dtype_bytes < 1:
        raise ValueError("dtype_bytes には1以上の int を指定してください")
    if coreset_mode not in CORESET_MODES:
        raise ValueError(
            f"coreset_mode は {list(CORESET_MODES)} のいずれかを指定してください: "
            f"{coreset_mode!r}"
        )
    if not 0 < sampling_ratio <= 1:
        raise ValueError("sampling_ratio には0より大きく1以下の値を指定してください")
    if projection_dim is not None:
        if (
            isinstance(projection_dim, bool)
            or not isinstance(projection_dim, int)
            or projection_dim < 1
        ):
            raise ValueError("projection_dim には1以上の int を指定してください")

    per_image = patches_per_image(image_size, stride)
    total_rows = n_images * per_image
    feature_bytes = total_rows * feature_dim * dtype_bytes
    explicit_projection = (
        min(projection_dim, feature_dim) if projection_dim is not None else None
    )

    if coreset_mode == "global":
        projected_dim = (
            explicit_projection
            if explicit_projection is not None
            else _projection_dim(total_rows, feature_dim)
        )
        projection_bytes = total_rows * projected_dim * dtype_bytes
        peak_bytes = 2 * feature_bytes + projection_bytes
        note = (
            _COMMON_NOTE
            + "global は embedding_store の全量と vstack 後の全量が同時に存在する状態"
            "（feature_bytes の2倍）に、KCenterGreedy が作る射影特徴"
            f"（{projected_dim} 次元）を加えた概算です。"
        )
        return FeatureBankEstimate(
            n_images=n_images,
            patches_per_image=per_image,
            feature_dim=feature_dim,
            dtype_bytes=dtype_bytes,
            coreset_mode=coreset_mode,
            chunk_images=None,
            feature_bytes=feature_bytes,
            peak_bytes_estimate=peak_bytes,
            note=note,
        )

    if (
        isinstance(chunk_images, bool)
        or not isinstance(chunk_images, int)
        or chunk_images < 1
    ):
        raise ValueError(
            "coreset_mode='chunked' では chunk_images に1以上の int を指定してください"
        )

    chunk_sizes = plan_chunks(total_rows, chunk_images * per_image)
    if not chunk_sizes:
        raise ValueError("チャンクに分割できる行数がありません")
    max_chunk_rows = max(chunk_sizes)
    selected_rows = sum(
        chunk_coreset_size(size, sampling_ratio) for size in chunk_sizes
    )
    projected_dim = (
        explicit_projection
        if explicit_projection is not None
        else _projection_dim(max_chunk_rows, feature_dim)
    )
    chunk_bytes = max_chunk_rows * feature_dim * dtype_bytes
    projection_bytes = max_chunk_rows * projected_dim * dtype_bytes
    selected_bytes = selected_rows * feature_dim * dtype_bytes
    note = (
        _COMMON_NOTE
        + "chunked は 1 チャンク分の vstack（チャンク行数の2倍）とチャンク射影"
        f"（{projected_dim} 次元）、選択済みバンク（{selected_rows} 行）の合計を"
        "概算した値です。チャンク境界の端数（最大1ステップ分）と、選択済みバンクを"
        "list で保持する間の断片化は含みません。"
    )
    return FeatureBankEstimate(
        n_images=n_images,
        patches_per_image=per_image,
        feature_dim=feature_dim,
        dtype_bytes=dtype_bytes,
        coreset_mode=coreset_mode,
        chunk_images=chunk_images,
        feature_bytes=feature_bytes,
        peak_bytes_estimate=2 * chunk_bytes + projection_bytes + selected_bytes,
        note=note,
    )


def check_memory_budget(
    estimate_bytes: int,
    limit_mib: float | None,
) -> tuple[bool, str]:
    """見積りバイト数が上限（MiB）以内かを判定する。

    Args:
        estimate_bytes: 見積りバイト数。
        limit_mib: 上限（MiB）。None なら判定しない。

    Returns:
        ``(上限以内か, 日本語の説明)``。

    Raises:
        ValueError: 引数が不正な場合。
    """
    if isinstance(estimate_bytes, bool) or not isinstance(estimate_bytes, int):
        raise ValueError("estimate_bytes には int を指定してください")
    if estimate_bytes < 0:
        raise ValueError("estimate_bytes には0以上を指定してください")
    if limit_mib is None:
        return True, "上限（limit_mib）が指定されていないため判定しません"
    if limit_mib <= 0:
        raise ValueError("limit_mib には0より大きい値を指定してください")

    limit_bytes = int(limit_mib * 1024 * 1024)
    estimate_value_mib = bytes_to_mib(estimate_bytes)
    if estimate_bytes <= limit_bytes:
        return True, (
            f"見積りピーク {estimate_value_mib} MiB は上限 {limit_mib} MiB 以内です（概算）。"
        )
    return False, (
        f"見積りピーク {estimate_value_mib} MiB が上限 {limit_mib} MiB を超えます（概算）。"
        " --patchcore-coreset-mode chunked と --patchcore-chunk-images で分割するか、"
        " --max-train-feature-mib を緩めてください。OOM Kill を避けるため学習を"
        "開始しません。"
    )


def _parse_meminfo_kib(text: str, key: str) -> int | None:
    """``/proc/meminfo`` 形式のテキストから指定キーの値（KiB）を取り出す。"""
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def read_mem_available_mib(meminfo_text: str | None = None) -> float | None:
    """``MemAvailable`` を MiB で返す。取得できない環境では None。

    Args:
        meminfo_text: ``/proc/meminfo`` の内容。None なら実ファイルを読む。
    """
    text = meminfo_text
    if text is None:
        try:
            text = _MEMINFO_PATH.read_text(encoding="utf-8")
        except OSError:
            return None
    kib = _parse_meminfo_kib(text, "MemAvailable")
    if kib is None:
        return None
    return round(kib / 1024.0, 2)


def build_mitigation_record(
    coreset_mode: str,
    chunk_images: int | None,
    sampling_ratio: float,
    *,
    image_size: int | None = None,
    feature_dim: int | None = None,
    stride: int | None = None,
    projection_dim: int | None = None,
    train_image_count: int | None = None,
    mem_available_mib_at_start: float | None = None,
    max_train_feature_mib: float | None = None,
    chunk_stats: Sequence[Mapping[str, Any]] | None = None,
    memory_bank_rows: int | None = None,
    memory_bank_dtype: str | None = None,
    memory_bank_bytes: int | None = None,
) -> dict[str, Any]:
    """report.json の ``training_memory_mitigation`` を作る。

    ``is_protocol_deviation`` は ``coreset_mode != "global"`` のとき True。
    chunked のときは ``approximation`` に「全体 KCenterGreedy と選択点集合が
    一致しない」旨を残す。

    Raises:
        ValueError: 引数が不正な場合。
    """
    if coreset_mode not in CORESET_MODES:
        raise ValueError(
            f"coreset_mode は {list(CORESET_MODES)} のいずれかを指定してください: "
            f"{coreset_mode!r}"
        )
    if not 0 < sampling_ratio <= 1:
        raise ValueError("sampling_ratio には0より大きく1以下の値を指定してください")
    if coreset_mode == "chunked":
        if (
            isinstance(chunk_images, bool)
            or not isinstance(chunk_images, int)
            or chunk_images < 1
        ):
            raise ValueError(
                "coreset_mode='chunked' では chunk_images に1以上の int を指定してください"
            )
    elif chunk_images is not None:
        raise ValueError(
            "coreset_mode='global' では chunk_images を指定できません"
        )

    stats = [dict(stat) for stat in (chunk_stats or [])]
    bank_mib = (
        bytes_to_mib(memory_bank_bytes) if memory_bank_bytes is not None else None
    )
    return {
        "coreset_mode": coreset_mode,
        "chunk_images": chunk_images,
        "sampling_ratio": sampling_ratio,
        "is_protocol_deviation": coreset_mode != "global",
        "approximation": None if coreset_mode == "global" else _APPROXIMATION_CHUNKED,
        "image_size": image_size,
        "feature_dim": feature_dim,
        "stride": stride,
        "projection_dim": projection_dim,
        "train_image_count": train_image_count,
        "mem_available_mib_at_start": mem_available_mib_at_start,
        "max_train_feature_mib": max_train_feature_mib,
        "chunk_stats": stats,
        "chunk_count": len(stats),
        "memory_bank": {
            "rows": memory_bank_rows,
            "dtype": memory_bank_dtype,
            "bytes": memory_bank_bytes,
            "mib": bank_mib,
        },
        "measured_peak_keys": [
            "memory.train_peak_rss_mib_sampled",
            "memory.process_lifetime_peak_rss_mib",
        ],
        "note": _MITIGATION_NOTE,
    }


def platform_supports_meminfo() -> bool:
    """``/proc/meminfo`` を読めるプラットフォームかを返す。"""
    return sys.platform.startswith("linux")
