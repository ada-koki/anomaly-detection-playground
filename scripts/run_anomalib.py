"""AnomalibによるCPU優先の異常検知ベンチマーク。

学習・テスト工程全体の時間と、サンプリングしたホストメモリ（RSS）を計測する。
画像1枚当たりの推論時間は計測しない。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator


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

    default_root = "MVTecAD" if args.dataset == "mvtec_ad" else "MVTec-4K"
    root = args.dataset_root or Path("datasets") / default_root

    # ダウンロード時間を計測に含めないため、実行前にデータセットの配置を確認する。
    if not all((root / args.category / split / "good").is_dir() for split in ("train", "test")):
        parser.error(
            f"{root / args.category} に train/good または test/good が見つかりません。"
            "READMEを参照してデータセットをダウンロード・展開してください。"
        )
    return root


def _process_rss_bytes(process: Any) -> int:
    total = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except Exception:
        processes = [process]
    for child in processes:
        try:
            total += child.memory_info().rss
        except Exception:
            continue
    return total


@contextmanager
def sampled_rss() -> Iterator[dict[str, float]]:
    import psutil

    process = psutil.Process(os.getpid())
    stopped = threading.Event()
    summary = {"peak_rss_mib": 0.0}

    def poll() -> None:
        while True:
            current = _process_rss_bytes(process) / (1024 * 1024)
            summary["peak_rss_mib"] = max(summary["peak_rss_mib"], current)
            if stopped.wait(0.1):
                break

    thread = threading.Thread(target=poll, name="rss-sampler", daemon=True)
    thread.start()
    try:
        yield summary
    finally:
        stopped.set()
        thread.join()
        summary["peak_rss_mib"] = round(
            max(summary["peak_rss_mib"], _process_rss_bytes(process) / (1024 * 1024)), 2
        )


def json_safe(value: Any) -> Any:
    """Lightningの評価指標をJSONに保存できる形式へ変換する。"""
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = validate_args(args, parser)

    # --helpや基本テストで重い依存ライブラリを読み込まないよう、importを遅延する。
    import torch
    from anomalib.data import MVTecAD
    from anomalib.data.utils import ValSplitMode
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore

    torch.manual_seed(args.seed)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDAを利用できません。--device cpu を指定してください")

    model_cls = {"padim": Padim, "patchcore": Patchcore}[args.model]
    model = model_cls(
        pre_processor=model_cls.configure_pre_processor(image_size=(args.image_size, args.image_size)),
        visualizer=False,
    )
    datamodule = MVTecAD(
        root=root,
        category=args.category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=args.val_fraction,
        seed=args.seed,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    engine = Engine(
        accelerator=args.device,
        devices=1,
        max_epochs=1,
        default_root_dir=output,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )

    with sampled_rss() as fit_memory:
        fit_start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = time.perf_counter() - fit_start
    with sampled_rss() as test_memory:
        test_start = time.perf_counter()
        test_metrics = engine.test(model=model, datamodule=datamodule)
        test_seconds = time.perf_counter() - test_start

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": args.dataset,
        "dataset_root": str(root.resolve()),
        "category": args.category,
        "model": args.model,
        "device": args.device,
        "image_size": [args.image_size, args.image_size],
        "image_handling": "whole_image_resized_not_tiled",
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "cpu_threads": torch.get_num_threads(),
        "val_split": {"mode": "from_test", "fraction": args.val_fraction, "seed": args.seed},
        "train_stage_seconds": round(fit_seconds, 3),
        "test_stage_seconds": round(test_seconds, 3),
        "train_peak_host_rss_mib_sampled": fit_memory["peak_rss_mib"],
        "test_peak_host_rss_mib_sampled": test_memory["peak_rss_mib"],
        "test_metrics": json_safe(test_metrics),
        "per_image_inference_latency": None,
        "host": {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "versions": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "anomalib": version("anomalib"),
        },
        "argv": list(sys.argv[1:] if argv is None else argv),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"結果を保存しました: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
