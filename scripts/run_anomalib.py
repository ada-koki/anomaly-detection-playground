"""AnomalibによるCPU優先の異常検知ベンチマーク。

学習・テスト工程全体の時間、画像1枚あたりの推論レイテンシ、
サンプリングしたホストメモリ（RSS）を計測する。
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.latency import measure_per_image_latency
from adbench.memory import DEFAULT_SAMPLING_INTERVAL_S, peak_rss_lifetime_mib, sampled_rss
from adbench.report import (
    DEPENDENCY_NAMES,
    SCHEMA_VERSION,
    build_base_report,
    build_failed_report,
    collect_environment,
    format_command,
    json_safe,
    write_report,
)

# サンプリングで記録した各段階のピークRSSキー。
_SAMPLED_PEAK_KEYS = (
    "train_peak_rss_mib_sampled",
    "test_peak_rss_mib_sampled",
    "latency_peak_rss_mib_sampled",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anomalibを使ったCPU優先の異常検知ベンチマーク")
    parser.add_argument("--dataset", choices=("mvtec_ad", "mvtec_4k"), default="mvtec_ad")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--model", choices=("padim", "patchcore"), default="padim")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0, help="0の場合はPyTorchの既定値を使用")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone",
        default=None,
        help="backbone名。省略時はモデル既定（PaDiM: resnet18, PatchCore: wide_resnet50_2）",
    )
    parser.add_argument(
        "--coreset-sampling-ratio",
        type=float,
        default=0.1,
        help="PatchCore用のcoresetサンプリング比率（0より大きく1以下）",
    )
    parser.add_argument(
        "--prefetch-weights",
        action="store_true",
        help="モデルを生成してbackbone重みを取得するだけで終了する",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=5,
        help="レイテンシ計測前に実行するウォームアップ画像数",
    )
    parser.add_argument(
        "--latency-max-images",
        type=int,
        default=0,
        help="レイテンシ計測に使う最大画像数。0はテスト画像全体",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=1,
        help="計測対象画像を繰り返す回数",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="画像1枚あたりの推論レイテンシ計測を行わない",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    for key in ("image_size", "train_batch_size", "eval_batch_size"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} には1以上を指定してください")
    if args.num_workers < 0 or args.cpu_threads < 0:
        parser.error("--num-workers と --cpu-threads には0以上を指定してください")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction には0より大きく1より小さい値を指定してください")
    if not 0 < args.coreset_sampling_ratio <= 1:
        parser.error("--coreset-sampling-ratio には0より大きく1以下の値を指定してください")
    if args.latency_warmup < 0:
        parser.error("--latency-warmup には0以上を指定してください")
    if args.latency_max_images < 0:
        parser.error("--latency-max-images には0以上を指定してください")
    if args.latency_repeats < 1:
        parser.error("--latency-repeats には1以上を指定してください")

    default_root = "MVTecAD" if args.dataset == "mvtec_ad" else "MVTec-4K"
    root = args.dataset_root or Path("datasets") / default_root

    # 重み取得だけを行う場合はデータセット配置を要求しない。
    if not args.prefetch_weights:
        # ダウンロード時間を計測に含めないため、実行前にデータセットの配置を確認する。
        if not all((root / args.category / split / "good").is_dir() for split in ("train", "test")):
            parser.error(
                f"{root / args.category} に train/good または test/good が見つかりません。"
                "READMEを参照してデータセットをダウンロード・展開してください。"
            )
    return root


def _create_model(args: argparse.Namespace) -> Any:
    from anomalib.models import Padim, Patchcore

    model_cls = {"padim": Padim, "patchcore": Patchcore}[args.model]
    model_kwargs: dict[str, Any] = {
        "pre_processor": model_cls.configure_pre_processor(
            image_size=(args.image_size, args.image_size)
        ),
        "visualizer": False,
    }
    if args.backbone:
        model_kwargs["backbone"] = args.backbone
    if args.model == "patchcore":
        model_kwargs["coreset_sampling_ratio"] = args.coreset_sampling_ratio
    return model_cls(**model_kwargs)


def _create_datamodule(args: argparse.Namespace, root: Path) -> Any:
    from anomalib.data import MVTecAD
    from anomalib.data.utils import ValSplitMode

    return MVTecAD(
        root=root,
        category=args.category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=args.val_fraction,
        seed=args.seed,
    )


def _collect_test_image_paths(datamodule: Any) -> list[Path]:
    """engine.test 後に setup 済みのテスト画像パスを取得する。"""
    test_data = getattr(datamodule, "test_data", None)
    samples = getattr(test_data, "samples", None)
    if samples is None:
        raise RuntimeError("テスト画像一覧（datamodule.test_data.samples）を取得できませんでした")

    image_paths = samples.image_path
    if hasattr(image_paths, "tolist"):
        image_paths = image_paths.tolist()
    return [Path(path) for path in image_paths]


def _max_peak_rss_mib(memory: dict[str, Any]) -> float:
    """プロセス寿命ピークと各段階のサンプリングピークの最大値をMiBで返す。

    例外発生時にも「記録済みのサンプリング値より小さいピーク」を残さないため、
    既存の記録値も候補に含める。
    """
    candidates: list[float] = [float(peak_rss_lifetime_mib())]
    for key in ("process_lifetime_peak_rss_mib",) + _SAMPLED_PEAK_KEYS:
        value = memory.get(key)
        if isinstance(value, (int, float)):
            candidates.append(float(value))
    return max(candidates)


def _rebuild_datamodule(datamodule: Any) -> Any:
    """既存のDataModuleから同じ設定で作り直す必要がある場合のフック。

    現状は作り直し不要のため、そのまま返す。
    """
    return datamodule


def _run(
    args: argparse.Namespace,
    root: Path,
    output: Path,
    report: dict[str, Any],
) -> None:
    """ベンチマーク本体を実行する。

    例外が発生しても finally でプロセス寿命ピークRSSを最新値に更新し、
    失敗レポートに学習時などのサンプリングピークが反映されるようにする。
    """
    memory = report.setdefault("memory", {})
    memory["sampling_interval_s"] = DEFAULT_SAMPLING_INTERVAL_S
    memory["process_lifetime_peak_rss_mib"] = round(peak_rss_lifetime_mib(), 2)

    try:
        _run_stages(args, root, output, report, memory)
    finally:
        # 例外時も最新のピーク値に更新する（成功時はほぼ同値）。
        memory["process_lifetime_peak_rss_mib"] = round(_max_peak_rss_mib(memory), 2)


def _run_stages(
    args: argparse.Namespace,
    root: Path,
    output: Path,
    report: dict[str, Any],
    memory: dict[str, Any],
) -> None:
    import torch
    from anomalib.engine import Engine

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    report["torch_num_threads"] = torch.get_num_threads()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAを利用できません。--device cpu を指定してください。")

    torch.manual_seed(args.seed)
    model = _create_model(args)

    if args.prefetch_weights:
        report["status"] = "ok"
        return

    datamodule = _create_datamodule(args, root)
    engine = Engine(
        accelerator=args.device,
        devices=1,
        max_epochs=1,
        default_root_dir=output,
        logger=False,
        num_sanity_val_steps=0,
    )

    with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as fit_memory:
        fit_start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = time.perf_counter() - fit_start
    memory["train_peak_rss_mib_sampled"] = fit_memory["peak_rss_mib"]

    with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as test_memory:
        test_start = time.perf_counter()
        test_metrics = engine.test(model=model, datamodule=datamodule)
        test_seconds = time.perf_counter() - test_start
    memory["test_peak_rss_mib_sampled"] = test_memory["peak_rss_mib"]

    report.update(
        {
            "train_stage_seconds": round(fit_seconds, 3),
            "test_stage_seconds": round(test_seconds, 3),
            "test_metrics": json_safe(test_metrics),
        }
    )
    # 旧キーは互換のため残す。
    report["train_peak_host_rss_mib_sampled"] = fit_memory["peak_rss_mib"]
    report["test_peak_host_rss_mib_sampled"] = test_memory["peak_rss_mib"]

    if args.skip_latency:
        report["per_image_inference_latency"] = {
            "skipped": True,
            "reason": "--skip-latency",
        }
    else:
        image_paths = _collect_test_image_paths(datamodule)
        latency_result = measure_per_image_latency(
            model,
            image_paths,
            warmup=args.latency_warmup,
            max_images=(None if args.latency_max_images == 0 else args.latency_max_images),
            repeats=args.latency_repeats,
        )
        report["per_image_inference_latency"] = latency_result
        memory["latency_peak_rss_mib_sampled"] = latency_result[
            "latency_peak_rss_mib_sampled"
        ]

    report["status"] = "ok"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = validate_args(args, parser)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = build_base_report(args, root, run_id, argv_list)

    try:
        _run(args, root, output, report)
    except Exception as error:
        failed_report = build_failed_report(report, error)
        write_report(report_path, failed_report)
        print(f"実行に失敗しました: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        print(f"結果を保存しました: {report_path}", file=sys.stderr)
        return 1

    write_report(report_path, report)
    print(f"結果を保存しました: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

