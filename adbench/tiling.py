"""タイル分割の設定・検証ロジック。

このモジュールは torch / numpy / anomalib に依存しない純粋ロジックのみを扱う。
Anomalib Tiler のタイル開始位置・パディング長・行優先のタイル座標に合わせ、
run_anomalib.py から利用できる設定検証とレポート用情報を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TILE_MODES",
    "TilingConfig",
    "TilePlan",
    "tile_starts",
    "padded_length",
    "requires_padding",
    "plan_tiles",
    "axis_coverage",
    "coverage_summary",
    "validate_for_model",
    "describe_tiling",
    "describe_global",
]

TILE_MODES = ("padding", "interpolation")

Tile = tuple[int, int, int, int]

_MERGE_RULE = (
    "Anomalib Tiler の untile により、重複領域の特徴（埋め込み）を"
    "被覆数で平均してから位置ごとに採点"
)


@dataclass(frozen=True)
class TilingConfig:
    """タイル分割の設定。

    Attributes:
        tile_size: 1 タイルの一辺の画素数。
        stride: タイル開始位置の間隔。
        mode: パディング方式。``TILE_MODES`` のいずれか。
        remove_border_count: タイル境界から除去する画素数。
    """

    tile_size: int
    stride: int
    mode: str = "padding"
    remove_border_count: int = 0

    def validate(self) -> None:
        """設定値を検証する。

        Raises:
            ValueError: いずれかの値が不正な場合。
        """
        if self.tile_size < 1:
            raise ValueError(
                f"tile_size は 1 以上でなければなりません: {self.tile_size}"
            )
        if self.stride < 1 or self.stride > self.tile_size:
            raise ValueError(
                "stride は 1 以上 tile_size 以下でなければなりません: "
                f"stride={self.stride}, tile_size={self.tile_size}"
            )
        if self.mode not in TILE_MODES:
            raise ValueError(
                f"mode は {TILE_MODES} のいずれかでなければなりません: "
                f"{self.mode!r}"
            )
        if self.remove_border_count < 0:
            raise ValueError(
                "remove_border_count は 0 以上でなければなりません: "
                f"{self.remove_border_count}"
            )
        if 2 * self.remove_border_count >= self.tile_size:
            raise ValueError(
                "2 * remove_border_count は tile_size より小さくなければ"
                "なりません: "
                f"remove_border_count={self.remove_border_count}, "
                f"tile_size={self.tile_size}"
            )

    @property
    def overlap_ratio(self) -> float:
        """タイルの重なり率（0.0 以上 1.0 未満）を返す。"""
        if self.tile_size <= 0:
            raise ValueError(
                f"tile_size は 1 以上でなければなりません: {self.tile_size}"
            )
        return 1.0 - (self.stride / self.tile_size)


@dataclass(frozen=True)
class TilePlan:
    """1 枚の画像に対するタイル配置。

    ``tiles`` の座標はパディング後の作業解像度上での
    ``(y0, x0, y1, x1)`` であり、Anomalib Tiler と同じ行優先順である。
    """

    image_hw: tuple[int, int]
    padded_hw: tuple[int, int]
    grid_hw: tuple[int, int]
    tiles: tuple[Tile, ...]
    overlap_ratio: float
    requires_padding: bool


def _validate_tile_and_stride(tile_size: int, stride: int) -> None:
    if tile_size < 1:
        raise ValueError(
            f"tile_size は 1 以上でなければなりません: {tile_size}"
        )
    if stride < 1:
        raise ValueError(f"stride は 1 以上でなければなりません: {stride}")


def _validate_image_hw(image_hw: tuple[int, int]) -> tuple[int, int]:
    if len(image_hw) != 2:
        raise ValueError(f"image_hw は (height, width) である必要があります: {image_hw}")
    height, width = image_hw
    if height < 1 or width < 1:
        raise ValueError(
            "画像の高さ・幅は 1 以上でなければなりません: "
            f"image_hw={image_hw}"
        )
    return height, width


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """1 辺分のタイル開始位置を返す。

    Anomalib Tiler と同じく ``range(0, length - tile_size + 1, stride)``
    の値を返す。パディングは行わない。

    Args:
        length: 対象の辺の長さ。
        tile_size: タイルの一辺。
        stride: タイル開始位置の間隔。

    Returns:
        タイル開始位置のリスト。

    Raises:
        ValueError: 引数が不正、または ``length < tile_size`` の場合。
    """
    _validate_tile_and_stride(tile_size, stride)
    if length < tile_size:
        raise ValueError(
            "length は tile_size 以上でなければなりません: "
            f"length={length}, tile_size={tile_size}"
        )
    return list(range(0, length - tile_size + 1, stride))


def padded_length(length: int, tile_size: int, stride: int) -> int:
    """Anomalib の ``compute_new_image_size`` の 1 辺分の長さを返す。

    ``(length - tile_size) % stride != 0`` の場合、
    ``ceil((length - tile_size) / stride) * stride + tile_size`` に拡張する。

    Args:
        length: 元の辺の長さ。
        tile_size: タイルの一辺。
        stride: タイル開始位置の間隔。

    Returns:
        パディング後の辺の長さ。

    Raises:
        ValueError: 引数が不正、または ``length < tile_size`` の場合。
    """
    _validate_tile_and_stride(tile_size, stride)
    if length < tile_size:
        raise ValueError(
            "length は tile_size 以上でなければなりません: "
            f"length={length}, tile_size={tile_size}"
        )

    delta = length - tile_size
    if delta % stride == 0:
        return length
    return ((delta + stride - 1) // stride) * stride + tile_size


def requires_padding(
    image_hw: tuple[int, int],
    config: TilingConfig,
) -> bool:
    """指定画像・設定で右端・下端のパディングが必要かを返す。

    Args:
        image_hw: ``(height, width)``。
        config: タイル設定。

    Returns:
        いずれかの辺で ``padded_length`` が元の長さを超えるなら True。

    Raises:
        ValueError: 設定不正、画像サイズ不正、または画像がタイルより小さい場合。
    """
    config.validate()
    height, width = _validate_image_hw(image_hw)
    if height < config.tile_size or width < config.tile_size:
        raise ValueError(
            "画像の高さ・幅は tile_size 以上でなければなりません: "
            f"image_hw={image_hw}, tile_size={config.tile_size}"
        )
    padded_h = padded_length(height, config.tile_size, config.stride)
    padded_w = padded_length(width, config.tile_size, config.stride)
    return padded_h != height or padded_w != width


def plan_tiles(image_hw: tuple[int, int], config: TilingConfig) -> TilePlan:
    """タイル配置を計算する。

    タイル座標はパディング後の作業解像度上で ``(y0, x0, y1, x1)``。
    Anomalib Tiler と同じく、行方向を外側、列方向を内側にした行優先順で返す。

    Args:
        image_hw: ``(height, width)``。
        config: タイル設定。

    Returns:
        タイル配置。

    Raises:
        ValueError: 設定不正、画像サイズ不正、または画像がタイルより小さい場合。
    """
    config.validate()
    height, width = _validate_image_hw(image_hw)
    if height < config.tile_size or width < config.tile_size:
        raise ValueError(
            "画像の高さ・幅は tile_size 以上でなければなりません: "
            f"image_hw={image_hw}, tile_size={config.tile_size}"
        )

    padded_h = padded_length(height, config.tile_size, config.stride)
    padded_w = padded_length(width, config.tile_size, config.stride)
    starts_h = tile_starts(padded_h, config.tile_size, config.stride)
    starts_w = tile_starts(padded_w, config.tile_size, config.stride)

    tiles = tuple(
        (
            y,
            x,
            y + config.tile_size,
            x + config.tile_size,
        )
        for y in starts_h
        for x in starts_w
    )

    return TilePlan(
        image_hw=(height, width),
        padded_hw=(padded_h, padded_w),
        grid_hw=(len(starts_h), len(starts_w)),
        tiles=tiles,
        overlap_ratio=config.overlap_ratio,
        requires_padding=(padded_h != height or padded_w != width),
    )


def _coverage_from_starts(
    length: int,
    tile_size: int,
    starts: list[int],
) -> list[int]:
    """開始位置リストから 1 辺分の画素被覆数を計算する。"""
    if length < 1:
        return []
    diff = [0] * (length + 1)
    for start in starts:
        begin = max(0, start)
        end = min(length, start + tile_size)
        if begin >= end:
            continue
        diff[begin] += 1
        diff[end] -= 1

    coverage: list[int] = []
    current = 0
    for index in range(length):
        current += diff[index]
        coverage.append(current)
    return coverage


def axis_coverage(length: int, tile_size: int, stride: int) -> list[int]:
    """パディングなしの 1 辺分について、各画素を覆うタイル数を返す。

    2 次元の被覆数は、高さ方向と幅方向の ``axis_coverage`` の外積になる。
    端数がある場合は末尾の画素が 0 になり得る。

    Args:
        length: 対象の辺の長さ。
        tile_size: タイルの一辺。
        stride: タイル開始位置の間隔。

    Returns:
        長さ ``length`` の被覆数リスト。

    Raises:
        ValueError: 引数が不正、または ``length < tile_size`` の場合。
    """
    starts = tile_starts(length, tile_size, stride)
    return _coverage_from_starts(length, tile_size, starts)


def coverage_summary(plan: TilePlan) -> dict[str, int]:
    """タイル配置の被覆状況を要約する。

    元画像の範囲内だけを対象に、最小・最大の被覆数と未被覆画素数を返す。
    タイルは直交グリッド状に並ぶため、2 次元の被覆数は各辺の被覆数の積になる。

    Args:
        plan: タイル配置。

    Returns:
        ``{"min": 最小被覆数, "max": 最大被覆数, "uncovered": 未被覆画素数}``。
    """
    height, width = plan.image_hw
    if not plan.tiles:
        return {"min": 0, "max": 0, "uncovered": height * width}

    first_y0, first_x0, first_y1, first_x1 = plan.tiles[0]
    tile_h = first_y1 - first_y0
    tile_w = first_x1 - first_x0

    starts_h = sorted({tile[0] for tile in plan.tiles})
    starts_w = sorted({tile[1] for tile in plan.tiles})
    coverage_h = _coverage_from_starts(height, tile_h, starts_h)
    coverage_w = _coverage_from_starts(width, tile_w, starts_w)

    if not coverage_h or not coverage_w:
        return {"min": 0, "max": 0, "uncovered": height * width}

    min_h = min(coverage_h)
    max_h = max(coverage_h)
    min_w = min(coverage_w)
    max_w = max(coverage_w)

    zero_h = sum(1 for value in coverage_h if value == 0)
    zero_w = sum(1 for value in coverage_w if value == 0)
    uncovered = zero_h * width + zero_w * height - zero_h * zero_w

    return {
        "min": min_h * min_w,
        "max": max_h * max_w,
        "uncovered": uncovered,
    }


def validate_for_model(image_size: int, config: TilingConfig) -> list[str]:
    """モデル実行に使えるタイル設定かを検証し、エラー文の一覧を返す。

    ``image_size`` は正方形の作業解像度。空のリストなら利用可能。

    Args:
        image_size: 作業解像度（正方形の一辺）。
        config: タイル設定。

    Returns:
        エラー文のリスト。問題がなければ空リスト。
    """
    errors: list[str] = []

    if image_size < 1:
        errors.append(
            f"作業解像度は 1 以上でなければなりません: image_size={image_size}"
        )
        return errors

    try:
        config.validate()
    except ValueError as exc:
        errors.append(f"タイル設定が不正です: {exc}")
        return errors

    if config.tile_size > image_size:
        errors.append(
            f"タイルサイズ {config.tile_size} が作業解像度 {image_size} を"
            "超えています。--image-size をタイルサイズ以上にするか、"
            "--tile-size を作業解像度以下にしてください。"
        )
        return errors

    if not requires_padding((image_size, image_size), config):
        return errors

    delta = image_size - config.tile_size
    valid_strides = [
        stride
        for stride in range(1, config.tile_size + 1)
        if delta % stride == 0
    ]
    non_trivial_strides = [stride for stride in valid_strides if stride > 1]
    stride_examples = (
        non_trivial_strides[-2:]
        if non_trivial_strides
        else valid_strides[-1:]
    )

    k = delta // config.stride
    lower_size = config.tile_size + k * config.stride
    upper_size = lower_size + config.stride
    image_size_examples = [
        value
        for value in (lower_size, upper_size)
        if value >= config.tile_size
    ]

    message = (
        "パディングが必要です: "
        f"作業解像度 {image_size} と tile_size {config.tile_size} の差 "
        f"{delta} は stride {config.stride} で割り切れません。"
        "Anomalib Tiler の untile は入力を画素サイズで切り出すため、"
        "特徴マップでは座標がずれます。このリポジトリでは各辺で "
        "(作業解像度 - tile_size) % stride == 0 となる構成のみ許可します。"
    )
    if stride_examples:
        message += (
            " stride は差の約数（例: "
            + ", ".join(str(value) for value in stride_examples)
            + "）に変更してください。"
        )
    if image_size_examples:
        message += (
            " または作業解像度を "
            + " / ".join(str(value) for value in image_size_examples)
            + " に変更してください。"
        )
    errors.append(message)
    return errors


def describe_tiling(image_size: int, config: TilingConfig) -> dict:
    """タイル実行時のレポート用情報を返す。

    Args:
        image_size: 作業解像度（正方形の一辺）。
        config: タイル設定。

    Returns:
        report に保存する辞書。
    """
    plan = plan_tiles((image_size, image_size), config)
    coverage = coverage_summary(plan)

    notes: list[str] = [
        "タイル座標はパディング後の作業解像度上の (y0, x0, y1, x1)。"
    ]
    if plan.requires_padding:
        notes.append(
            "この構成はパディングが必要なため、validate_for_model では拒否される。"
        )
    if config.remove_border_count:
        notes.append(
            "remove_border_count は Anomalib Tiler に渡す想定。"
            "本リポジトリでの座標・精度への影響は未検証。"
        )

    return {
        "spatial_strategy": "tiled",
        "working_resolution": [image_size, image_size],
        "tile_size": config.tile_size,
        "stride": config.stride,
        "overlap_ratio": round(config.overlap_ratio, 6),
        "mode": config.mode,
        "remove_border_count": config.remove_border_count,
        "grid": [plan.grid_hw[0], plan.grid_hw[1]],
        "tiles_per_image": len(plan.tiles),
        "padded_resolution": [plan.padded_hw[0], plan.padded_hw[1]],
        "requires_padding": plan.requires_padding,
        "coverage": coverage,
        "merge_rule": _MERGE_RULE,
        "notes": notes,
    }


def describe_global(image_size: int) -> dict:
    """全体縮小実行時のレポート用情報を返す。

    Args:
        image_size: 作業解像度（正方形の一辺）。

    Returns:
        report に保存する辞書。
    """
    return {
        "spatial_strategy": "global_resize",
        "working_resolution": [image_size, image_size],
        "tiles_per_image": 1,
    }
