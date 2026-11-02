"""画像1枚単位のCPU推論レイテンシ計測。

Anomalib 2.7.0 の ``PreProcessor.forward`` はテンソルを受け取りテンソルを返すため、
本モジュールは「バッチ次元付き画像テンソル」を前処理に渡し、その出力テンソルを
``model.model`` に渡す。
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from .memory import DEFAULT_SAMPLING_INTERVAL_S, sampled_rss
from .stats import summarize_latencies

LATENCY_STAGE_KEYS = (
    "decode_ms",
    "preprocess_ms",
    "model_ms",
    "postprocess_ms",
    "end_to_end_ms",
)


def select_latency_images(
    image_paths: Sequence[Any],
    *,
    warmup: int,
    max_images: int | None,
) -> tuple[list[Any], list[Any]]:
    """ウォームアップ用画像と計測対象画像に分割する。

    max_images が None または 0 の場合は、ウォームアップ後の全画像を計測対象にする。
    """
    if warmup < 0:
        raise ValueError("warmup には0以上を指定してください")
    if max_images is not None and max_images < 0:
        raise ValueError("max_images には0以上を指定してください")

    paths = list(image_paths)
    warmup_count = min(warmup, len(paths))
    warmup_paths = paths[:warmup_count]
    remaining = paths[warmup_count:]

    if max_images is None or max_images == 0:
        measured_paths = remaining
    else:
        measured_paths = remaining[:max_images]

    return warmup_paths, measured_paths


def _tensor_hw(value: Any) -> list[int] | None:
    """テンソルまたは配列の末尾2次元を [H, W] として返す。"""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        if len(shape) < 2:
            return None
        return [int(shape[-2]), int(shape[-1])]
    except (TypeError, ValueError):
        return None


def measure_per_image_latency(
    model: Any,
    image_paths: Sequence[Any],
    *,
    warmup: int,
    max_images: int | None,
    repeats: int,
) -> dict[str, Any]:
    """画像1枚ごとの推論レイテンシを計測する。

    torch と anomalib は関数内で遅延 import する。
    """
    if repeats < 1:
        raise ValueError("repeats には1以上を指定してください")

    warmup_paths, measured_paths = select_latency_images(
        image_paths,
        warmup=warmup,
        max_images=max_images,
    )

    import torch
    from anomalib.data.utils import read_image

    model.eval()
    pre_processor = getattr(model, "pre_processor", None)
    stage_values: dict[str, list[float]] = {key: [] for key in LATENCY_STAGE_KEYS}
    input_resolution: list[int] | None = None
    warmup_executed = 0
    latency_samples = 0

    def run_once(path: Any, *, collect: bool) -> None:
        nonlocal input_resolution, latency_samples

        end_to_end_start = time.perf_counter()

        decode_start = time.perf_counter()
        image = read_image(path, as_tensor=True)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        preprocess_start = time.perf_counter()
        # PreProcessor はテンソルを受け取りテンソルを返す（Anomalib 2.7.0）。
        input_tensor = image.unsqueeze(0)
        preprocessed = (
            pre_processor(input_tensor) if pre_processor is not None else input_tensor
        )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        current_hw = _tensor_hw(preprocessed)
        if current_hw is not None:
            input_resolution = current_hw

        model_start = time.perf_counter()
        inference = model.model(preprocessed)
        model_ms = (time.perf_counter() - model_start) * 1000.0

        postprocess_start = time.perf_counter()
        model.post_processor(inference)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0

        if collect:
            stage_values["decode_ms"].append(decode_ms)
            stage_values["preprocess_ms"].append(preprocess_ms)
            stage_values["model_ms"].append(model_ms)
            stage_values["postprocess_ms"].append(postprocess_ms)
            stage_values["end_to_end_ms"].append(end_to_end_ms)
            latency_samples += 1

    with torch.inference_mode():
        for path in warmup_paths:
            run_once(path, collect=False)
            warmup_executed += 1

        with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as rss:
            for _ in range(repeats):
                for path in measured_paths:
                    run_once(path, collect=True)

    summaries = {
        key: summarize_latencies(stage_values[key])
        for key in LATENCY_STAGE_KEYS
    }
    notes = (
        "decode は read_image のみ、preprocess はバッチ次元付与と pre_processor（画像テンソルを入力）、"
        "model は model.model に前処理済みテンソルを渡す時間です。"
        "postprocess は post_processor の時間です。"
        "end_to_end は decode から postprocess までの合計です。"
        "ウォームアップ画像は集計から除外しています。"
        "RSSはサンプリングに基づく推定値で、サンプリング間隔より短いピークは取り逃す可能性があります。"
    )
    if not measured_paths:
        notes += " 計測対象の画像がありませんでした。"

    return {
        "model_class": type(model).__name__,
        "torch_num_threads": torch.get_num_threads(),
        "batch_size": 1,
        "input_resolution": input_resolution,
        "warmup_requested": warmup,
        "warmup_images": warmup_executed,
        "warmup_excluded_from_summary": True,
        "max_images": max_images,
        "measured_images": len(measured_paths),
        "repeats": repeats,
        "latency_samples": latency_samples,
        "sampling_interval_s": DEFAULT_SAMPLING_INTERVAL_S,
        "latency_peak_rss_mib_sampled": rss["peak_rss_mib"],
        "decode_ms": summaries["decode_ms"],
        "preprocess_ms": summaries["preprocess_ms"],
        "model_ms": summaries["model_ms"],
        "postprocess_ms": summaries["postprocess_ms"],
        "end_to_end_ms": summaries["end_to_end_ms"],
        "notes": notes,
    }

