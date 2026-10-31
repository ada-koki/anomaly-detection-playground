"""AnomalibによるCPU優先の異常検知ベンチマーク。

学習・テスト工程全体の時間、画像1枚あたりの推論レイテンシ、
サンプリングしたホストメモリ（RSS）を計測する。

GPU 学習＋CPU 推論は 1 プロセス内でデバイスを切り替えず、
``--save-checkpoint`` で保存したチェックポイントを ``--checkpoint`` で
読み込む 2 段階実行で実現する（Issue #11）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import shutil
import sys
import time
import traceback
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.devices import (
    EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
    build_provenance,
    classify_execution_mode,
    detect_device_mismatch,
    effective_device,
    provenance_matches,
    provenance_path_for,
    read_provenance,
    write_provenance,
)
from adbench.latency import measure_per_image_latency
from adbench.manifest import (
    build_data_split_report,
    collect_manifest_rows,
    write_manifest,
)
from adbench.memory import DEFAULT_SAMPLING_INTERVAL_S, peak_rss_lifetime_mib, sampled_rss
from adbench.mvtec4k import build_resolution_condition, has_foreground_dir
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
from adbench.training_mode import determine_training_mode

# サンプリングで記録した各段階のピークRSSキー。
_SAMPLED_PEAK_KEYS = (
    "train_peak_rss_mib_sampled",
    "validate_peak_rss_mib_sampled",
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
    parser.add_argument(
        "--model",
        choices=("padim", "patchcore", "efficient_ad"),
        default="padim",
    )
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
        "--imagenet-dir",
        type=Path,
        default=Path("datasets/imagenette2"),
        help=(
            "EfficientADの教師学習に使うImageNette展開先。"
            "自動ダウンロードさせないため、実行前に配置を確認します"
        ),
    )
    parser.add_argument(
        "--efficientad-model-size",
        choices=("small", "medium"),
        default=None,
        help="EfficientADのモデルサイズ。省略時はプロトコル値（既定 small）",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=(
            "EfficientADの最大エポック数。省略時はプロトコル値。"
            "PaDiM/PatchCoreでは1のみ許可"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "EfficientADの最大ステップ数。省略時はプロトコル値。"
            "-1は無制限。PaDiM/PatchCoreでは指定不可"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "学習済みチェックポイントを読み込み、学習をスキップして"
            "test・レイテンシ・raw評価のみ行う"
        ),
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "学習後にチェックポイントを DIR/model.ckpt として保存し、"
            "学習デバイスの来歴サイドカー（model.ckpt.provenance.json）を書き出す。"
            "GPU学習＋CPU推論の2段階実行に使う。DIR はリポジトリ外を推奨"
            "（重みはコミットしない）。--checkpoint とは併用不可"
        ),
    )
    parser.add_argument(
        "--allow-device-mismatch",
        action="store_true",
        help=(
            "要求デバイスと観測デバイスが一致しない場合でも警告のみで続行する。"
            "既定では report を保存した上で終了コード 3 で終了する"
        ),
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

    if args.model == "efficient_ad":
        if args.efficientad_model_size is not None:
            overrides[f"{model_prefix}.model_size"] = args.efficientad_model_size
        if args.max_epochs is not None:
            overrides[f"{model_prefix}.max_epochs"] = args.max_epochs
        if args.max_steps is not None:
            overrides[f"{model_prefix}.max_steps"] = args.max_steps
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

    if args.model == "efficient_ad":
        args.efficientad_model_size = (
            model_config.model_size
            or args.efficientad_model_size
            or "small"
        )
        if args.max_epochs is None:
            args.max_epochs = (
                model_config.max_epochs
                if model_config.max_epochs is not None
                else 1
            )
        if args.max_steps is None:
            args.max_steps = (
                model_config.max_steps
                if model_config.max_steps is not None
                else -1
            )
    else:
        # メモリバンク系は1エポックで十分であり、条件差を作らない。
        args.max_epochs = 1
        args.max_steps = -1

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

    # 解決後に既定値へ上書きされるため、明示指定の有無とCLI値を退避する。
    args.max_epochs_cli = args.max_epochs
    args.max_steps_cli = args.max_steps
    args.efficientad_model_size_cli = args.efficientad_model_size

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
    max_epochs_cli = getattr(args, "max_epochs_cli", None)
    max_steps_cli = getattr(args, "max_steps_cli", None)

    if args.model != "efficient_ad":
        if max_epochs_cli is not None and max_epochs_cli != 1:
            parser.error(
                "--max-epochs は efficient_ad でのみ、1以外の値を指定できます。"
                "PaDiM/PatchCore は 1 エポック固定です。"
            )
        if max_steps_cli is not None:
            parser.error(
                "--max-steps は efficient_ad でのみ指定できます。"
                "PaDiM/PatchCore では条件差を作らないため指定できません。"
            )
    else:
        if getattr(args, "efficientad_model_size", None) not in (None, "small", "medium"):
            parser.error("--efficientad-model-size が不正です")
        if args.max_epochs < 1:
            parser.error("--max-epochs には1以上を指定してください")
        if args.max_steps == 0 or args.max_steps < -1:
            parser.error("--max-steps には -1（無制限）または1以上を指定してください")
        if args.train_batch_size != 1:
            parser.error(
                "EfficientAD は train_batch_size=1 を要求します。"
                "プロトコルまたは --train-batch-size を 1 にしてください。"
            )
        if (
            args.checkpoint is None
            and not args.prefetch_weights
            and not args.manifest_only
            and not Path(args.imagenet_dir).is_dir()
        ):
            parser.error(
                f"{args.imagenet_dir} が見つかりません。EfficientADの自動ダウンロードを"
                "計測に含めないため、ImageNette (imagenette2) を事前に配置してください。"
                "取得手順は docs/efficientad.md を参照してください。"
            )

    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            parser.error(f"--checkpoint のファイルが見つかりません: {args.checkpoint}")
        if args.prefetch_weights:
            parser.error("--checkpoint と --prefetch-weights は同時に指定できません")
        if args.save_checkpoint is not None:
            parser.error(
                "--checkpoint と --save-checkpoint は同時に指定できません。"
                "学習する実行で保存し、推論する実行で読み込む2段階に分けてください。"
            )

    if args.save_checkpoint is not None:
        if args.prefetch_weights:
            parser.error(
                "--save-checkpoint と --prefetch-weights は同時に指定できません"
            )
        if args.manifest_only:
            parser.error(
                "--save-checkpoint と --manifest-only は同時に指定できません"
            )

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
        if args.dataset == "mvtec_4k" and has_foreground_dir(root, args.category):
            parser.error(
                "MVTec-4K の生配置には foreground/ があり、Anomalib 2.7 の "
                "MVTecAD は foreground 配下の画像を train/test の重複行として"
                "取り込みます。scripts/validate_mvtec4k.py --make-view OUTPUT で"
                "clean view を作成し、--dataset-root にそのディレクトリを指定して"
                "ください。"
            )
    return root


def _create_model(args: argparse.Namespace) -> Any:
    from anomalib.models import EfficientAd, Padim, Patchcore

    model_cls = {
        "padim": Padim,
        "patchcore": Patchcore,
        "efficient_ad": EfficientAd,
    }[args.model]

    if args.model == "efficient_ad":
        model_kwargs: dict[str, Any] = {
            "imagenet_dir": str(args.imagenet_dir),
            "model_size": args.efficientad_model_size,
            "pre_processor": model_cls.configure_pre_processor(
                image_size=(args.image_size, args.image_size)
            ),
            "visualizer": False,
        }
        return model_cls(**model_kwargs)

    model_kwargs = {
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


def _load_checkpoint_model(args: argparse.Namespace) -> Any:
    """学習済みチェックポイントをCPUへ読み込む。"""
    from anomalib.models import EfficientAd, Padim, Patchcore

    model_cls = {
        "padim": Padim,
        "patchcore": Patchcore,
        "efficient_ad": EfficientAd,
    }[args.model]
    return model_cls.load_from_checkpoint(str(args.checkpoint), map_location="cpu")


def _prepare_efficientad_pretrained_model(model: Any) -> None:
    prepare = getattr(model, "prepare_pretrained_model", None)
    if not callable(prepare):
        raise RuntimeError(
            "EfficientAd に prepare_pretrained_model() が見つかりません。"
            "Anomalib のバージョンを確認してください。"
        )
    prepare()


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_repo_path(path: Path) -> str:
    """リポジトリルートからの相対表記を返す。

    リポジトリ外を指す場合は、絶対パスやユーザー名をレポートへ残さないため
    ファイル名だけを返す。
    """
    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(path)
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        relative = os.path.relpath(resolved, _REPO_ROOT)
        if relative.startswith(".."):
            return resolved.name
        return relative


def _module_version(name: str) -> str | None:
    """モジュールの ``__version__`` を返す。取得できなければ ``None``。"""
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def _observe_model_device(model: Any) -> str:
    """モデルパラメータの実デバイス種別を返す。

    パラメータを持たないモデルでは ``"unknown"`` を返す。
    """
    try:
        parameters = getattr(model, "parameters", None)
        if not callable(parameters):
            return "unknown"
        first = next(parameters(), None)
    except Exception:
        return "unknown"
    if first is None:
        return "unknown"
    device_type = getattr(getattr(first, "device", None), "type", None)
    return str(device_type) if device_type else "unknown"


def _cuda_device_name() -> str | None:
    """CUDA が利用可能な場合に 0 番デバイスの名前を返す。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


def _check_observed_device(
    requested: str, observed: str | None
) -> tuple[list[str], list[str]]:
    """観測デバイスを確認し、``(不一致の警告, その他の警告)`` を返す。

    不一致の警告がある場合、呼び出し側は report 保存後に終了コード 3 で
    終了する（``--allow-device-mismatch`` 指定時は警告のみ）。
    """
    if observed is None or observed == "" or observed == "unknown":
        return [], [
            "モデルのパラメータからデバイスを観測できませんでした"
            f"（要求: {requested}）。"
        ]
    try:
        mismatch = detect_device_mismatch(requested, [observed])
    except ValueError as error:
        return [], [f"観測デバイスを判定できませんでした: {error}"]
    return mismatch, []


def _ensure_devices_report(report: dict[str, Any], requested: str) -> dict[str, Any]:
    """report の ``devices`` セクションを用意して返す。"""
    devices = report.get("devices")
    if not isinstance(devices, dict):
        devices = {}
        report["devices"] = devices
    devices.setdefault("train", None)
    infer = devices.get("infer")
    if not isinstance(infer, dict):
        devices["infer"] = {"requested": requested, "observed": None}
    warnings = devices.get("warnings")
    if not isinstance(warnings, list):
        devices["warnings"] = []
    return devices


def _best_checkpoint_path(trainer: Any) -> str | None:
    """Lightning trainer が保持する best チェックポイントのパスを返す。"""
    if trainer is None:
        return None
    callback = getattr(trainer, "checkpoint_callback", None)
    if callback is None:
        return None
    best = getattr(callback, "best_model_path", None)
    if isinstance(best, str) and best:
        return best
    return None


def _save_checkpoint(
    engine: Any,
    args: argparse.Namespace,
    output: Path,
    report: dict[str, Any],
    devices_report: Mapping[str, Any],
) -> None:
    """学習後のチェックポイントと来歴サイドカーを保存し、report に記録する。

    現在のモデル状態を ``trainer.save_checkpoint`` で ``DIR/model.ckpt`` に保存し、
    trainer が無い場合のみ best チェックポイントを複製する。重みはコミットしない前提で、
    ``DIR`` はリポジトリ外を推奨する（help に明記）。
    """
    target_dir = Path(args.save_checkpoint)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "model.ckpt"

    trainer = getattr(engine, "trainer", None)
    # Lightning の best チェックポイントは validation 前（PostProcessor 統計や
    # EfficientAD の quantile 確定前）の状態の可能性があるため、まず現在の
    # 状態（fit・追加 validate 後）を trainer.save_checkpoint で保存する。
    # trainer が使えない場合に限り best チェックポイントを複製する。
    if trainer is not None and callable(getattr(trainer, "save_checkpoint", None)):
        trainer.save_checkpoint(str(target))
    else:
        source = _best_checkpoint_path(trainer)
        if source is None or not Path(source).is_file():
            raise RuntimeError(
                "--save-checkpoint を使うには Lightning の trainer が必要です。"
                "engine の初期化を確認してください。"
            )
        if Path(source).resolve() != target.resolve():
            shutil.copyfile(source, target)

    checkpoint_sha256 = _sha256_file(target)

    train_info = devices_report.get("train")
    train_info = train_info if isinstance(train_info, Mapping) else {}
    observed_train = train_info.get("observed")
    observed_train_device = (
        observed_train
        if isinstance(observed_train, str) and observed_train != "unknown"
        else None
    )
    train_device = observed_train_device or str(args.device)
    cuda_device_name = train_info.get("cuda_device_name")

    protocol_report = report.get("protocol")
    protocol_report = (
        protocol_report if isinstance(protocol_report, Mapping) else {}
    )
    protocol_name = protocol_report.get("name")
    protocol_sha256 = protocol_report.get("sha256")

    provenance = build_provenance(
        checkpoint_sha256=checkpoint_sha256,
        train_device=train_device,
        observed_train_device=observed_train_device,
        cuda_device_name=(
            cuda_device_name if isinstance(cuda_device_name, str) else None
        ),
        torch_version=_module_version("torch"),
        anomalib_version=_module_version("anomalib"),
        model=str(args.model),
        category=str(args.category),
        protocol_name=protocol_name if isinstance(protocol_name, str) else None,
        protocol_sha256=(
            protocol_sha256 if isinstance(protocol_sha256, str) else None
        ),
        run_id=str(report.get("run_id") or output.name),
    )
    sidecar = provenance_path_for(target)
    write_provenance(sidecar, provenance)

    report["saved_checkpoint"] = {
        "path": _relative_repo_path(target),
        "sha256": checkpoint_sha256,
        "provenance_path": _relative_repo_path(sidecar),
    }


def _load_checkpoint_provenance(
    checkpoint: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """チェックポイントの来歴サイドカーを読み、sha256 を照合する。

    Returns:
        ``(来歴データまたは None, 警告文のリスト, 信頼できる学習デバイスまたは None)``。
        サイドカーが無ければ ``(None, [], None)``。sha256 が一致しない場合は
        来歴データは返すが、学習デバイスは信頼せず ``None`` にする。
    """
    sidecar = provenance_path_for(checkpoint)
    try:
        provenance = read_provenance(sidecar)
    except ValueError as error:
        return (
            None,
            [f"来歴サイドカーを読めません（{sidecar.name}）: {error}"],
            None,
        )
    if provenance is None:
        return None, [], None
    if not provenance_matches(provenance, checkpoint_sha256):
        return (
            provenance,
            [
                "来歴サイドカーの checkpoint_sha256 が実ファイルと一致しません。"
                "学習デバイスの来歴を信頼せず、"
                "external_checkpoint_unknown_train_device として扱います。"
            ],
            None,
        )
    train_device = provenance.get("train_device")
    return (
        provenance,
        [],
        train_device if isinstance(train_device, str) else None,
    )


def _get_pretrained_weights_dir() -> Path | None:
    """Anomalib の教師重みキャッシュディレクトリを取得する。"""
    candidates = (
        "anomalib.models.image.efficient_ad.lightning_model",
        "anomalib.models.image.efficient_ad.utils",
        "anomalib.models.image.efficient_ad",
        "anomalib.utils.path",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        function = getattr(module, "get_pretrained_weights_dir", None)
        if callable(function):
            try:
                return Path(function())
            except Exception:
                continue
    return None


def _teacher_weights_path(model_size: str) -> Path | None:
    base = _get_pretrained_weights_dir()
    if base is None:
        return None
    return (
        base
        / "efficientad_pretrained_weights"
        / f"pretrained_teacher_{model_size}.pth"
    )


def _count_imagenet_images(imagenet_dir: Path) -> int | None:
    if not Path(imagenet_dir).is_dir():
        return None
    try:
        from torchvision.datasets import ImageFolder

        return len(ImageFolder(str(imagenet_dir)))
    except Exception:
        return None


def _build_efficientad_model_specific(
    args: argparse.Namespace,
    *,
    train_stage_seconds: float | None,
    trained_steps: int | None,
    validation_called: bool,
) -> dict[str, Any]:
    "#### EfficientAD固有の前提条件・外部依存をレポート用にまとめる。"
    weights_path = _teacher_weights_path(args.efficientad_model_size)
    weights_sha256: str | None = None
    if weights_path is not None and weights_path.is_file():
        try:
            weights_sha256 = _sha256_file(weights_path)
        except OSError:
            weights_sha256 = None

    train_seconds_per_step: float | None = None
    if (
        train_stage_seconds is not None
        and trained_steps is not None
        and trained_steps > 0
    ):
        train_seconds_per_step = round(train_stage_seconds / trained_steps, 6)

    return {
        "model_size": args.efficientad_model_size,
        "imagenet_dir": _relative_repo_path(Path(args.imagenet_dir)),
        "imagenet_image_count": _count_imagenet_images(Path(args.imagenet_dir)),
        "teacher_weights_path": (
            weights_path.name if weights_path is not None else None
        ),
        "teacher_weights_sha256": weights_sha256,
        "max_epochs": args.max_epochs,
        "max_steps": args.max_steps,
        "trained_steps": trained_steps,
        "train_seconds_per_step": train_seconds_per_step,
        "train_seconds_per_step_note": (
            "train_stage_seconds を trained_steps で割った実測由来の計算値です。"
            "ステップ数・モデルサイズ・学習条件が異なる値と同等比較しないでください。"
        ),
        "paper_default_steps": 70000,
        "validation_called_after_fit": validation_called,
        "external_data_dependencies": {
            "imagenette": {
                "name": "ImageNette (imagenette2)",
                "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
                "note": (
                    "ImageNet のサブセットです。ImageNet の利用条件に従う必要があり、"
                    "配布元・原データのライセンスを確認してください。"
                ),
            },
            "teacher_weights": {
                "source": "anomalib GitHub Releases (efficientad_pretrained_weights)",
                "url": (
                    "https://github.com/open-edge-platform/anomalib/releases/download/"
                    "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
                ),
                "note": (
                    "ImageNet 由来の蒸留を含みます。配布元で利用条件・ライセンスを"
                    "確認してください。"
                ),
            },
        },
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

    devices_report = _ensure_devices_report(report, args.device)
    device_warnings: list[str] = devices_report["warnings"]
    mismatch_warnings: list[str] = []

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

    checkpoint_mode = args.checkpoint is not None
    provenance_train_device: str | None = None

    if checkpoint_mode:
        model = _load_checkpoint_model(args)
        checkpoint_sha256 = report.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
            checkpoint_sha256 = _sha256_file(Path(args.checkpoint))
            report["checkpoint_sha256"] = checkpoint_sha256
        (
            provenance,
            provenance_warnings,
            provenance_train_device,
        ) = _load_checkpoint_provenance(
            Path(args.checkpoint), checkpoint_sha256
        )
        device_warnings.extend(provenance_warnings)
        report["checkpoint_provenance"] = (
            None
            if provenance is None
            else {
                "path": provenance_path_for(Path(args.checkpoint)).name,
                "sha256_matches": provenance_matches(
                    provenance, checkpoint_sha256
                ),
                "data": provenance,
            }
        )
        # 来歴で GPU 学習が証明された場合だけ training_mode を切り替える。
        report["training_mode"] = determine_training_mode(
            args.checkpoint,
            provenance_train_device=provenance_train_device,
        )
    else:
        model = _create_model(args)

    if args.prefetch_weights:
        if args.model == "efficient_ad":
            _prepare_efficientad_pretrained_model(model)
            report["model_specific"] = _build_efficientad_model_specific(
                args,
                train_stage_seconds=None,
                trained_steps=None,
                validation_called=False,
            )
        report["status"] = "ok"
        return

    from anomalib.engine import Engine

    datamodule = _create_datamodule(args, root)
    engine_kwargs: dict[str, Any] = {
        "accelerator": args.device,
        "devices": 1,
        "max_epochs": args.max_epochs if args.model == "efficient_ad" else 1,
        "default_root_dir": output,
        "logger": False,
        "num_sanity_val_steps": 0,
    }
    if args.model == "efficient_ad" and args.max_steps != -1:
        engine_kwargs["max_steps"] = args.max_steps
    engine = Engine(**engine_kwargs)

    fit_seconds: float | None = None
    trained_steps: int | None = None

    if checkpoint_mode:
        # 学習をスキップする。ckpt に PostProcessor 統計が保存されている前提で、
        # 追加の validate は呼ばず、チェックポイント内の統計を使う。
        datamodule.setup("fit")
        datamodule.setup("test")
        report["train_stage_seconds"] = None
        report["train_peak_host_rss_mib_sampled"] = None
        memory["train_peak_rss_mib_sampled"] = None
        report["validation_after_fit"] = False
        report["validation_note"] = (
            "--checkpoint 指定時は学習も追加の validate も実行せず、"
            "チェックポイントに保存された PostProcessor 統計を使用します。"
        )
        report["checkpoint_inference"] = {
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "validation_called": False,
            "uses_statistics_from_checkpoint": True,
            "note": (
                "チェックポイント内の統計に依存します。ckpt 作成時の検証集合・"
                "前処理条件が異なる場合は engine.test の閾値依存指標を比較できません。"
            ),
        }
        _record_manifest(report, datamodule, root, output, args)
        if args.model == "efficient_ad":
            report["model_specific"] = _build_efficientad_model_specific(
                args,
                train_stage_seconds=None,
                trained_steps=None,
                validation_called=False,
            )
    else:
        with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as fit_memory:
            fit_start = time.perf_counter()
            engine.fit(model=model, datamodule=datamodule)
            fit_seconds = time.perf_counter() - fit_start
        memory["train_peak_rss_mib_sampled"] = fit_memory["peak_rss_mib"]

        # 学習直後の実デバイスを観測する（意図しないフォールバック検知）。
        observed_train = _observe_model_device(model)
        devices_report["train"] = {
            "requested": args.device,
            "observed": observed_train,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                _cuda_device_name() if args.device == "cuda" else None
            ),
        }
        mismatch, other = _check_observed_device(args.device, observed_train)
        mismatch_warnings.extend(mismatch)
        device_warnings.extend(mismatch)
        device_warnings.extend(other)

        trainer = getattr(engine, "trainer", None)
        if trainer is not None:
            trained_steps = int(getattr(trainer, "global_step", 0))

        report["train_stage_seconds"] = round(fit_seconds, 3)
        report["train_peak_host_rss_mib_sampled"] = fit_memory["peak_rss_mib"]

        if args.model == "efficient_ad":
            # max_steps 指定時にエポック末の validation が走らない場合でも、
            # quantile と PostProcessor 統計を必ず計算するため追加で validate する。
            with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as validate_memory:
                validate_start = time.perf_counter()
                engine.validate(model=model, datamodule=datamodule)
                validate_seconds = time.perf_counter() - validate_start
            memory["validate_peak_rss_mib_sampled"] = validate_memory["peak_rss_mib"]
            report["validate_stage_seconds"] = round(validate_seconds, 3)
            report["validation_after_fit"] = True
            report["validation_note"] = (
                "EfficientAD では max_steps 指定時に validation が走らない場合に備え、"
                "fit 後に engine.validate を明示的に1回実行し、good 画像から"
                "quantile と PostProcessor 統計を確保しています。"
            )
            report["model_specific"] = _build_efficientad_model_specific(
                args,
                train_stage_seconds=fit_seconds,
                trained_steps=trained_steps,
                validation_called=True,
            )

        _record_manifest(report, datamodule, root, output, args)

        if args.save_checkpoint is not None:
            # PostProcessor 統計が確定した後（validate 後）に保存する。
            _save_checkpoint(engine, args, output, report, devices_report)

    if checkpoint_mode and args.device != "cpu":
        # チェックポイントは CPU に読み込まれるため、推論デバイスへ明示的に移す。
        model = model.to(args.device)

    # 推論（engine.test）直前の実デバイスを観測する。
    observed_infer = _observe_model_device(model)
    devices_report["infer"] = {
        "requested": args.device,
        "observed": observed_infer,
    }
    mismatch, other = _check_observed_device(args.device, observed_infer)
    mismatch_warnings.extend(mismatch)
    device_warnings.extend(mismatch)
    device_warnings.extend(other)

    if checkpoint_mode:
        train_device_for_mode = provenance_train_device
    else:
        train_info = devices_report.get("train")
        train_observed = (
            train_info.get("observed")
            if isinstance(train_info, Mapping)
            else None
        )
        train_device_for_mode = effective_device(args.device, train_observed)
    infer_device_for_mode = effective_device(args.device, observed_infer)
    try:
        report["execution_mode"] = classify_execution_mode(
            train_device_for_mode, infer_device_for_mode
        )
    except ValueError as error:
        # 来歴が手編集などで壊れていても実行は継続し、保守的に不明として扱う。
        report["execution_mode"] = (
            EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE
        )
        device_warnings.append(f"実行モードを判定できませんでした: {error}")

    with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as test_memory:
        test_start = time.perf_counter()
        test_metrics = engine.test(model=model, datamodule=datamodule)
        test_seconds = time.perf_counter() - test_start
    memory["test_peak_rss_mib_sampled"] = test_memory["peak_rss_mib"]

    report.update(
        {
            "test_stage_seconds": round(test_seconds, 3),
            "test_metrics": json_safe(test_metrics),
        }
    )
    # 旧キーは互換のため残す。
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

    report["device_mismatch"] = {
        "detected": bool(mismatch_warnings),
        "warnings": list(mismatch_warnings),
        "allowed": bool(args.allow_device_mismatch),
    }
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
    report["training_mode"] = determine_training_mode(args.checkpoint)
    # 元画像解像度とモデル入力解像度は別条件であるため、レポートで区別できるようにする。
    report["resolution_condition"] = build_resolution_condition(
        root,
        args.category,
        args.image_size,
    )
    if args.checkpoint is not None:
        report["checkpoint_sha256"] = _sha256_file(args.checkpoint)
        report["train_stage_seconds"] = None

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

    devices = report.get("devices")
    if isinstance(devices, Mapping):
        warnings = devices.get("warnings")
        if isinstance(warnings, list):
            for warning in warnings:
                print(f"警告: {warning}", file=sys.stderr)

    mismatch = report.get("device_mismatch")
    if (
        isinstance(mismatch, Mapping)
        and mismatch.get("detected")
        and not args.allow_device_mismatch
    ):
        print(
            "要求デバイスと観測デバイスが一致しません。意図しないフォールバックの"
            "可能性があるため終了コード 3 で終了します。警告のみで続行する場合は "
            "--allow-device-mismatch を指定してください。",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
