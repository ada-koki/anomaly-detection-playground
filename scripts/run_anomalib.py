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
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.latency import measure_per_image_latency
from adbench.manifest import (
    build_data_split_report,
    collect_manifest_rows,
    write_manifest,
)
from adbench.memory import DEFAULT_SAMPLING_INTERVAL_S, peak_rss_lifetime_mib, sampled_rss
from adbench.protocol import (
    DEFAULT_PROTOCOL_PATH,
    ModelConfig,
    Protocol,
    ProtocolError,
    apply_overrides,
    load_protocol,
    protocol_file_sha256,
    protocol_to_dict,
)
from adbench.raw_eval import evaluate_raw_score_metrics
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
    "raw_eval_peak_rss_mib_sampled",
    "latency_peak_rss_mib_sampled",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anomalibを使ったCPU優先の異常検知ベンチマーク")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="標準プロトコルJSON。省略時は configs/protocols/mvtec_ad_v1.json",
    )
    parser.add_argument("--dataset", choices=("mvtec_ad", "mvtec_4k"), default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--model", choices=("padim", "patchcore"), default="padim")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="0の場合はPyTorchの既定値を使用",
    )
    parser.add_argument(
        "--val-split-ratio",
        "--val-fraction",
        dest="val_split_ratio",
        type=float,
        default=None,
        help="検証集合の割合。プロトコル設定を明示的に上書きする場合に指定",
    )
    parser.add_argument(
        "--val-split-mode",
        choices=("none", "same_as_test", "from_test"),
        default=None,
        help=(
            "検証集合の作り方。プロトコル設定を明示的に上書きする場合に指定。"
            "none は engine.test のメトリクスが信頼できないため拒否します"
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--backbone",
        default=None,
        help="backbone名。省略時はプロトコルのモデル設定を使用",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="特徴量抽出に使う層。省略時はプロトコルのモデル設定を使用",
    )
    parser.add_argument(
        "--coreset-sampling-ratio",
        type=float,
        default=None,
        help="PatchCore用のcoresetサンプリング比率。プロトコル設定を上書きする場合に指定",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=None,
        help="PatchCore用の近傍数。プロトコル設定を上書きする場合に指定",
    )
    parser.add_argument(
        "--prefetch-weights",
        action="store_true",
        help="モデルを生成してbackbone重みを取得するだけで終了する",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="学習せずdatamodule setupとmanifest作成・保存だけを行う",
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
    parser.add_argument(
        "--skip-raw-eval",
        action="store_true",
        help="engine.test後の生スコア評価（raw_score_metrics）を行わない",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def _collect_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLIで明示指定された値だけを上書き辞書へ入れる。"""
    overrides: dict[str, Any] = {}
    for name in (
        "dataset",
        "seed",
        "image_size",
        "train_batch_size",
        "eval_batch_size",
        "num_workers",
        "cpu_threads",
        "val_split_mode",
        "val_split_ratio",
    ):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value

    model_prefix = f"models.{args.model}"
    if args.backbone is not None:
        overrides[f"{model_prefix}.backbone"] = args.backbone
    if args.layers is not None:
        overrides[f"{model_prefix}.layers"] = list(args.layers)
    if args.coreset_sampling_ratio is not None:
        overrides[f"{model_prefix}.coreset_sampling_ratio"] = args.coreset_sampling_ratio
    if args.num_neighbors is not None:
        overrides[f"{model_prefix}.num_neighbors"] = args.num_neighbors
    return overrides


def _resolve_args(
    args: argparse.Namespace,
    protocol: Protocol,
    overrides: dict[str, Any],
) -> argparse.Namespace:
    """プロトコル適用後の値でCLI名前空間を埋める。"""
    args.dataset = protocol.dataset
    args.val_split_mode = protocol.val_split_mode
    args.val_split_ratio = protocol.val_split_ratio
    args.seed = protocol.seed
    args.image_size = protocol.image_size
    args.train_batch_size = protocol.train_batch_size
    args.eval_batch_size = protocol.eval_batch_size
    args.num_workers = protocol.num_workers
    args.cpu_threads = protocol.cpu_threads

    model_config = protocol.models.get(args.model, ModelConfig())
    args.backbone = model_config.backbone
    args.layers = list(model_config.layers) if model_config.layers is not None else None
    args.coreset_sampling_ratio = model_config.coreset_sampling_ratio
    args.num_neighbors = model_config.num_neighbors

    backbone_override_key = f"models.{args.model}.backbone"
    if backbone_override_key in overrides:
        args.backbone_source = "cli"
    elif model_config.backbone is not None:
        args.backbone_source = "protocol"
    else:
        args.backbone_source = "model_default"
    args.backbone_is_default = model_config.backbone is None
    args.protocol_overrides = dict(overrides)
    return args


def parse_and_resolve(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, Protocol, Path, dict[str, Any]]:
    """CLI引数を解析し、プロトコル読み込み・上書き適用・既定値解決まで行う。

    引数解析直後の ``argparse.Namespace`` はプロトコル由来の値（``image_size`` など）が
    ``None`` のままなので、値の検証やレポート生成には本関数の戻り値を使う。

    Args:
        argv: CLI引数。``None`` の場合は ``sys.argv[1:]`` を使う。

    Returns:
        ``(解決済み引数, 上書き適用後のプロトコル, プロトコルファイルのパス,
        CLI上書き辞書)``。

    Raises:
        SystemExit: プロトコル読み込み・上書き適用に失敗した場合（argparse経由）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    protocol_path = args.protocol if args.protocol is not None else DEFAULT_PROTOCOL_PATH
    overrides: dict[str, Any] = {}
    try:
        protocol = load_protocol(protocol_path)
        overrides = _collect_cli_overrides(args)
        protocol = apply_overrides(protocol, overrides)
    except ProtocolError as error:
        # argparse の作法に合わせて終了コード2で終了する。
        parser.error(str(error))

    args = _resolve_args(args, protocol, overrides)
    args.protocol = protocol_path
    return args, protocol, protocol_path, overrides


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    for key in ("image_size", "train_batch_size", "eval_batch_size"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} には1以上を指定してください")
    if args.num_workers < 0 or args.cpu_threads < 0:
        parser.error("--num-workers と --cpu-threads には0以上を指定してください")
    if args.val_split_mode == "none":
        # プロトコル検証でも拒否しているが、CLI単体でも理由付きで落とす。
        parser.error(
            "val_split_mode=none は使用できません。AnomalibのPostProcessorは検証集合から"
            "正規化パラメータ（image_min/image_max）と閾値を求めるため、検証集合が空だと"
            "engine.testのメトリクスが実行ごとに変動して信頼できません。"
            "same_as_test または from_test を指定してください。"
        )
    if args.val_split_mode == "from_test":
        if args.val_split_ratio is None or not 0 < args.val_split_ratio < 1:
            parser.error(
                "val_split_mode=from_test では val_split_ratio に"
                "0より大きく1より小さい値を指定してください"
            )
    elif args.val_split_ratio is not None and not 0 < args.val_split_ratio < 1:
        parser.error("--val-split-ratio には0より大きく1より小さい値を指定してください")
    if args.model == "patchcore":
        if (
            args.coreset_sampling_ratio is None
            or not 0 < args.coreset_sampling_ratio <= 1
        ):
            parser.error(
                "PatchCoreでは coreset_sampling_ratio に"
                "0より大きく1以下の値を指定してください"
            )
        if args.num_neighbors is None or args.num_neighbors < 1:
            parser.error("PatchCoreでは num_neighbors に1以上を指定してください")
    if args.latency_warmup < 0:
        parser.error("--latency-warmup には0以上を指定してください")
    if args.latency_max_images < 0:
        parser.error("--latency-max-images には0以上を指定してください")
    if args.latency_repeats < 1:
        parser.error("--latency-repeats には1以上を指定してください")

    default_root = "MVTecAD" if args.dataset == "mvtec_ad" else "MVTec-4K"
    root = args.dataset_root or Path("datasets") / default_root

    # 重み取得だけを行う場合はデータセット配置を要求しない。
    if args.manifest_only or not args.prefetch_weights:
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
    if args.layers:
        model_kwargs["layers"] = list(args.layers)
    if args.model == "patchcore":
        if args.coreset_sampling_ratio is not None:
            model_kwargs["coreset_sampling_ratio"] = args.coreset_sampling_ratio
        if args.num_neighbors is not None:
            model_kwargs["num_neighbors"] = args.num_neighbors
    return model_cls(**model_kwargs)


def _create_datamodule(args: argparse.Namespace, root: Path) -> Any:
    from anomalib.data import MVTecAD
    from anomalib.data.utils import ValSplitMode

    kwargs: dict[str, Any] = {
        "root": root,
        "category": args.category,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "val_split_mode": ValSplitMode(args.val_split_mode),
        "seed": args.seed,
    }
    if args.val_split_ratio is not None:
        kwargs["val_split_ratio"] = args.val_split_ratio
    return MVTecAD(**kwargs)


def _test_samples(datamodule: Any) -> Any:
    """setup済みのDataModuleからテストサンプル一覧を取得する。"""
    test_data = getattr(datamodule, "test_data", None)
    samples = getattr(test_data, "samples", None)
    if samples is None:
        raise RuntimeError("テスト画像一覧（datamodule.test_data.samples）を取得できませんでした")
    return samples


def _collect_test_image_paths(datamodule: Any) -> list[Path]:
    """engine.test 後に setup 済みのテスト画像パスを取得する。"""
    samples = _test_samples(datamodule)
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


def _threshold_selection(args: argparse.Namespace) -> dict[str, Any]:
    """閾値調整に使う集合の説明を返す。"""
    if args.val_split_mode == "from_test":
        return {
            "source": "val",
            "val_split_mode": args.val_split_mode,
            "val_split_ratio": args.val_split_ratio,
            "uses_test_for_threshold": False,
            "optimistic": False,
            "threshold_dependent_metrics": "tuned_on_val",
            "note": (
                "検証集合は公式testからseed固定・label_awareで抽出した閾値調整専用集合です。"
                "AUROC評価にはtest集合のみを使い、valを精度評価に使いません。"
            ),
        }
    if args.val_split_mode == "same_as_test":
        return {
            "source": "test",
            "val_split_mode": args.val_split_mode,
            "val_split_ratio": args.val_split_ratio,
            "uses_test_for_threshold": True,
            "optimistic": True,
            "threshold_dependent_metrics": "tuned_on_test_optimistic",
            "note": (
                "valはテスト集合の複製です。閾値依存指標（F1・FPR・FNRなど）は"
                "テスト集合で調整済み（楽観的）であり、他プロトコルとの比較に"
                "使わないでください。"
            ),
        }
    return {
        "source": "none",
        "val_split_mode": args.val_split_mode,
        "val_split_ratio": args.val_split_ratio,
        "uses_test_for_threshold": None,
        "optimistic": None,
        "threshold_dependent_metrics": None,
        "note": (
            "val_split_mode=none は拒否されるため通常は到達しません。"
            "検証集合なしの場合、閾値依存指標は比較に使えません。"
        ),
    }


def _record_manifest(
    report: dict[str, Any],
    datamodule: Any,
    root: Path,
    output: Path,
    args: argparse.Namespace,
) -> None:
    rows = collect_manifest_rows(datamodule, root)
    # same_as_test のときだけ val と test の完全一致重複を許容する。
    allow_overlap = args.val_split_mode == "same_as_test"
    summary = write_manifest(
        output / "manifest.json",
        rows,
        root,
        allow_val_test_overlap=allow_overlap,
    )
    report["data_split"] = build_data_split_report(summary, _threshold_selection(args))


def _build_protocol_report(
    protocol: Protocol,
    protocol_path: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": protocol.protocol_name,
        "version": protocol.protocol_version,
        "config_path": str(protocol_path),
        "sha256": protocol_file_sha256(protocol_path),
        "sha256_scope": "protocol_file_bytes",
        "config": protocol_to_dict(protocol),
        "overrides": dict(sorted(overrides.items())),
    }


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
    from lightning.pytorch import seed_everything

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    report["torch_num_threads"] = torch.get_num_threads()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAを利用できません。--device cpu を指定してください。")

    seed_everything(args.seed, workers=True)
    torch.manual_seed(args.seed)
    randomness = report.setdefault("randomness", {})
    randomness.update(
        {
            "seed": args.seed,
            "seed_everything_called": True,
            "torch_manual_seed_called": True,
            "torch_are_deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        }
    )

    if args.manifest_only:
        datamodule = _create_datamodule(args, root)
        datamodule.setup("fit")
        _record_manifest(report, datamodule, root, output, args)
        report["status"] = "ok"
        return

    model = _create_model(args)

    if args.prefetch_weights:
        report["status"] = "ok"
        return

    from anomalib.engine import Engine

    datamodule = _create_datamodule(args, root)
    engine_kwargs: dict[str, Any] = {
        "accelerator": args.device,
        "devices": 1,
        "max_epochs": 1,
        "default_root_dir": output,
        "logger": False,
        "num_sanity_val_steps": 0,
    }
    engine = Engine(**engine_kwargs)

    with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as fit_memory:
        fit_start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = time.perf_counter() - fit_start
    memory["train_peak_rss_mib_sampled"] = fit_memory["peak_rss_mib"]

    _record_manifest(report, datamodule, root, output, args)

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

    # engine.test のメトリクスは PostProcessor の正規化（val由来）に依存するため、
    # 生スコアによる検証用メトリクスを別途記録する（Issue #7）。
    if args.skip_raw_eval:
        report["raw_score_metrics"] = {
            "skipped": True,
            "reason": "--skip-raw-eval",
        }
    else:
        with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as raw_memory:
            raw_start = time.perf_counter()
            try:
                raw_metrics = evaluate_raw_score_metrics(
                    model,
                    _test_samples(datamodule),
                    device=args.device,
                )
            except Exception as error:  # 補助指標のため実行全体は止めない
                raw_metrics = {
                    "error": f"{type(error).__name__}: {error}",
                    "notes": (
                        "生スコア評価に失敗しました。engine.test のメトリクスのみ"
                        "記録しています。"
                    ),
                }
                print(
                    f"生スコア評価に失敗しました: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )
            raw_seconds = time.perf_counter() - raw_start
        memory["raw_eval_peak_rss_mib_sampled"] = raw_memory["peak_rss_mib"]
        raw_metrics["elapsed_seconds"] = round(raw_seconds, 3)
        report["raw_score_metrics"] = raw_metrics

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
    # 解析からプロトコル既定値の解決までを一箇所にまとめる。
    args, protocol, protocol_path, overrides = parse_and_resolve(argv)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    # validate_args はエラー表示にパーサを使うだけなので、同じ定義から生成する。
    parser = build_parser()
    root = validate_args(args, parser)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    protocol_report = _build_protocol_report(protocol, protocol_path, overrides)
    report = build_base_report(args, root, run_id, argv_list, protocol_report)

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
