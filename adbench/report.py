"""ベンチマーク結果レポートの生成と実行環境収集。"""

from __future__ import annotations

import json
import math
import os
import platform
import shlex
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from .memory import DEFAULT_SAMPLING_INTERVAL_S

SCHEMA_VERSION = 4
DEPENDENCY_NAMES = ("torch", "torchvision", "anomalib", "lightning", "timm", "numpy")

MEMORY_NOTES = (
    "RSSはpsutilでプロセスツリーをサンプリングした推定値です。"
    "サンプリング間隔より短いピークは取り逃す可能性があります。"
    "process_lifetime_peak_rss_mib は resource.getrusage(RUSAGE_SELF).ru_maxrss に基づく"
    "自プロセス生涯最大RSSであり、学習・テスト・推論などの工程別の値ではありません。"
    "num_workers=0 の場合、通常はデータ読み込み用の子プロセスは発生しません。"
)

TIMING_NOTES = {
    "train_stage_seconds": "engine.fit 全体の所要時間です。",
    "test_stage_seconds": (
        "engine.test 全体（テストデータ読み込み・評価指標計算を含む）の所要時間です。"
        "画像1枚あたりの推論時間ではなく、per_image_inference_latency.end_to_end_ms とは"
        "意味が異なるため混同しないでください。"
    ),
    "per_image_inference_latency": (
        "画像1枚ごとに decode / preprocess / model / postprocess / end_to_end を"
        "time.perf_counter で計測した結果です。ウォームアップ画像は集計から除外します。"
    ),
}

RANDOMNESS_NOTES = (
    "lightning.seed_everything(seed, workers=True) と torch.manual_seed(seed) を実行します。"
    "torch.are_deterministic_algorithms_enabled() は実行時点のPyTorch設定を記録したもので、"
    "本スクリプトが決定論的アルゴリズムを有効化したことを意味しません。"
    "PYTHONHASHSEED はプロセス開始時点の環境変数の有無と値を記録します。"
)


def json_safe(value: Any) -> Any:
    """Lightningの評価指標などをJSONに保存できる形式へ変換する。"""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if callable(getattr(value, "item", None)):
        try:
            return json_safe(value.item())
        except (ValueError, RuntimeError):
            pass
    return str(value)


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """重い依存ライブラリをimportせずに取得できる実行環境情報を返す。"""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in DEPENDENCY_NAMES:
        versions[name] = _package_version(name)
    return {
        "host": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_logical_cores": os.cpu_count(),
        "versions": versions,
    }


def format_command(argv: Sequence[str]) -> str:
    return shlex.join(["python", "scripts/run_anomalib.py", *argv])


def build_base_report(
    args: Any,
    root: Path,
    run_id: str,
    argv: Sequence[str],
    protocol_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """実行開始時点の共通レポートを作る。"""
    environment = collect_environment()
    pythonhashseed_value = os.environ.get("PYTHONHASHSEED")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "run_id": run_id,
        "dataset": args.dataset,
        "dataset_root": str(root.resolve()),
        "category": args.category,
        "model": args.model,
        "backbone": args.backbone,
        "backbone_is_default": getattr(args, "backbone_is_default", args.backbone is None),
        "backbone_source": getattr(args, "backbone_source", None),
        "layers": list(args.layers) if getattr(args, "layers", None) is not None else None,
        "coreset_sampling_ratio": (
            args.coreset_sampling_ratio if args.model == "patchcore" else None
        ),
        "num_neighbors": getattr(args, "num_neighbors", None),
        "prefetch_weights": args.prefetch_weights,
        "manifest_only": getattr(args, "manifest_only", False),
        "device": args.device,
        "image_size": [args.image_size, args.image_size],
        "image_handling": "whole_image_resized_not_tiled",
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "requested_cpu_threads": args.cpu_threads,
        "torch_num_threads": None,
        "val_split": {
            "mode": args.val_split_mode,
            "fraction": args.val_split_ratio,
            "seed": args.seed,
        },
        "protocol": protocol_report,
        "data_split": None,
        "randomness": {
            "seed": args.seed,
            "seed_everything_called": False,
            "torch_manual_seed_called": False,
            "torch_are_deterministic_algorithms_enabled": None,
            "pythonhashseed_present": "PYTHONHASHSEED" in os.environ,
            "pythonhashseed_value": pythonhashseed_value,
            "notes": RANDOMNESS_NOTES,
        },
        "latency_settings": {
            "warmup": getattr(args, "latency_warmup", None),
            "max_images": getattr(args, "latency_max_images", None),
            "repeats": getattr(args, "latency_repeats", None),
            "skip_latency": getattr(args, "skip_latency", None),
        },
        "train_stage_seconds": None,
        "test_stage_seconds": None,
        "timing_notes": dict(TIMING_NOTES),
        "memory": {
            "train_peak_rss_mib_sampled": None,
            "test_peak_rss_mib_sampled": None,
            "latency_peak_rss_mib_sampled": None,
            "process_lifetime_peak_rss_mib": None,
            "sampling_interval_s": DEFAULT_SAMPLING_INTERVAL_S,
            "notes": MEMORY_NOTES,
        },
        "train_peak_host_rss_mib_sampled": None,
        "test_peak_host_rss_mib_sampled": None,
        "test_metrics": None,
        "per_image_inference_latency": None,
        "host": environment["host"],
        "cpu_logical_cores": environment["cpu_logical_cores"],
        "versions": environment["versions"],
        "argv": list(argv),
        "command": format_command(argv),
    }


def build_failed_report(base_report: dict[str, Any], error: BaseException) -> dict[str, Any]:
    report = dict(base_report)
    report["status"] = "failed"
    report["error_type"] = type(error).__name__
    report["error_message"] = str(error)
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
