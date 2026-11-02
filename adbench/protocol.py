"""標準プロトコル設定の読み込み・検証。

torch / anomalib などの重い依存を持たず、CLI の --help や unittest から
そのまま利用できるようにする。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = _REPO_ROOT / "configs" / "protocols" / "mvtec_ad_v1.json"

_ALLOWED_DATASETS = {"mvtec_ad", "mvtec_4k"}
# val_split_mode=none は PostProcessor の正規化パラメータが NaN になり
# engine.test のメトリクスが信頼できないため、許可しない（Issue #7）。
_ALLOWED_VAL_SPLIT_MODES = {"same_as_test", "from_test"}
_ALLOWED_EFFICIENTAD_MODEL_SIZES = {"small", "medium"}
_TOP_LEVEL_FIELDS = {
    "protocol_name",
    "protocol_version",
    "dataset",
    "val_split_mode",
    "val_split_ratio",
    "seed",
    "image_size",
    "train_batch_size",
    "eval_batch_size",
    "num_workers",
    "cpu_threads",
    "models",
}
_BASE_MODEL_FIELDS = {
    "backbone",
    "layers",
    "coreset_sampling_ratio",
    "num_neighbors",
}
_EFFICIENTAD_MODEL_FIELDS = {
    "model_size",
    "max_epochs",
    "max_steps",
}
_MODEL_FIELDS = _BASE_MODEL_FIELDS | _EFFICIENTAD_MODEL_FIELDS

VAL_SPLIT_MODE_NONE_REASON = (
    "val_split_mode=none は使用できません。AnomalibのPostProcessorは検証集合から"
    "正規化パラメータ（image_min/image_max）と閾値を求めるため、検証集合が空だと"
    "これらの値がNaNのままになり、engine.testのメトリクスが実行ごとに変動して"
    "信頼できません。same_as_test または from_test を指定してください。"
)


class ProtocolError(ValueError):
    """プロトコル設定が不正な場合に送出する例外。"""


@dataclass(frozen=True)
class ModelConfig:
    """モデル別の実験条件。"""

    backbone: str | None = None
    layers: tuple[str, ...] | None = None
    coreset_sampling_ratio: float | None = None
    num_neighbors: int | None = None
    model_size: str | None = None
    max_epochs: int | None = None
    max_steps: int | None = None


@dataclass(frozen=True)
class Protocol:
    """標準プロトコルの解決済み設定。"""

    protocol_name: str
    protocol_version: int
    dataset: str
    val_split_mode: str
    val_split_ratio: float | None
    seed: int
    image_size: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    cpu_threads: int
    models: dict[str, ModelConfig]


def _expect_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{where} はJSONオブジェクトで指定してください")
    return value


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{where} には空でない文字列を指定してください")
    return value


def _expect_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{where} には整数を指定してください")
    return value


def _expect_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{where} には数値を指定してください")
    return float(value)


def _model_config_from_dict(model_name: str, value: Any) -> ModelConfig:
    raw = _expect_mapping(value, f"models.{model_name}")
    is_efficient_ad = model_name == "efficient_ad"
    allowed_fields = _MODEL_FIELDS if is_efficient_ad else _BASE_MODEL_FIELDS
    unknown = set(raw) - allowed_fields
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ProtocolError(f"models.{model_name} に未知のキーがあります: {joined}")

    backbone = raw.get("backbone")
    if backbone is not None:
        backbone = _expect_str(backbone, f"models.{model_name}.backbone")

    layers_value = raw.get("layers")
    layers: tuple[str, ...] | None
    if layers_value is None:
        layers = None
    else:
        if not isinstance(layers_value, (list, tuple)):
            raise ProtocolError(f"models.{model_name}.layers には文字列配列を指定してください")
        if not layers_value:
            raise ProtocolError(f"models.{model_name}.layers を空にしないでください")
        layers = tuple(
            _expect_str(item, f"models.{model_name}.layers[]") for item in layers_value
        )

    coreset_value = raw.get("coreset_sampling_ratio")
    coreset_sampling_ratio: float | None
    if coreset_value is None:
        coreset_sampling_ratio = None
    else:
        coreset_sampling_ratio = _expect_number(
            coreset_value, f"models.{model_name}.coreset_sampling_ratio"
        )
        if not 0 < coreset_sampling_ratio <= 1:
            raise ProtocolError(
                f"models.{model_name}.coreset_sampling_ratio には"
                "0より大きく1以下の値を指定してください"
            )

    neighbors_value = raw.get("num_neighbors")
    num_neighbors: int | None
    if neighbors_value is None:
        num_neighbors = None
    else:
        num_neighbors = _expect_int(neighbors_value, f"models.{model_name}.num_neighbors")
        if num_neighbors < 1:
            raise ProtocolError(
                f"models.{model_name}.num_neighbors には1以上を指定してください"
            )

    model_size: str | None = None
    max_epochs: int | None = None
    max_steps: int | None = None
    if is_efficient_ad:
        model_size_value = raw.get("model_size")
        if model_size_value is not None:
            model_size = _expect_str(
                model_size_value, f"models.{model_name}.model_size"
            ).lower()
            if model_size not in _ALLOWED_EFFICIENTAD_MODEL_SIZES:
                raise ProtocolError(
                    f"models.{model_name}.model_size は "
                    f"{sorted(_ALLOWED_EFFICIENTAD_MODEL_SIZES)} のいずれかを"
                    "指定してください"
                )

        max_epochs_value = raw.get("max_epochs")
        if max_epochs_value is not None:
            max_epochs = _expect_int(
                max_epochs_value, f"models.{model_name}.max_epochs"
            )
            if max_epochs < 1:
                raise ProtocolError(
                    f"models.{model_name}.max_epochs には1以上を指定してください"
                )

        max_steps_value = raw.get("max_steps")
        if max_steps_value is not None:
            max_steps = _expect_int(
                max_steps_value, f"models.{model_name}.max_steps"
            )
            if max_steps == 0 or max_steps < -1:
                raise ProtocolError(
                    f"models.{model_name}.max_steps には -1（無制限）または"
                    "1以上を指定してください"
                )

    return ModelConfig(
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        model_size=model_size,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def protocol_from_dict(data: Mapping[str, Any]) -> Protocol:
    """辞書からプロトコルを構築し、値を検証する。"""
    raw = _expect_mapping(data, "プロトコル設定")
    unknown = set(raw) - _TOP_LEVEL_FIELDS
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ProtocolError(f"プロトコル設定に未知のキーがあります: {joined}")

    protocol_name = _expect_str(raw.get("protocol_name"), "protocol_name")
    protocol_version = _expect_int(raw.get("protocol_version"), "protocol_version")
    if protocol_version < 1:
        raise ProtocolError("protocol_version には1以上を指定してください")

    dataset = _expect_str(raw.get("dataset"), "dataset")
    if dataset not in _ALLOWED_DATASETS:
        raise ProtocolError(
            f"dataset は {sorted(_ALLOWED_DATASETS)} のいずれかを指定してください"
        )

    val_split_mode = _expect_str(raw.get("val_split_mode"), "val_split_mode").lower()
    if val_split_mode == "none":
        raise ProtocolError(VAL_SPLIT_MODE_NONE_REASON)
    if val_split_mode not in _ALLOWED_VAL_SPLIT_MODES:
        raise ProtocolError(
            "val_split_mode は same_as_test / from_test のいずれかを"
            "指定してください（none は拒否されます）"
        )

    ratio_value = raw.get("val_split_ratio")
    val_split_ratio: float | None
    if ratio_value is None:
        val_split_ratio = None
    else:
        val_split_ratio = _expect_number(ratio_value, "val_split_ratio")
        if not 0 < val_split_ratio < 1:
            raise ProtocolError("val_split_ratio には0より大きく1より小さい値を指定してください")

    if val_split_mode == "from_test" and val_split_ratio is None:
        raise ProtocolError("val_split_mode=from_test では val_split_ratio が必須です")

    seed = _expect_int(raw.get("seed"), "seed")
    if seed < 0:
        raise ProtocolError("seed には0以上を指定してください")

    image_size = _expect_int(raw.get("image_size"), "image_size")
    if image_size < 1:
        raise ProtocolError("image_size には1以上を指定してください")

    train_batch_size = _expect_int(raw.get("train_batch_size"), "train_batch_size")
    if train_batch_size < 1:
        raise ProtocolError("train_batch_size には1以上を指定してください")

    eval_batch_size = _expect_int(raw.get("eval_batch_size"), "eval_batch_size")
    if eval_batch_size < 1:
        raise ProtocolError("eval_batch_size には1以上を指定してください")

    num_workers = _expect_int(raw.get("num_workers"), "num_workers")
    if num_workers < 0:
        raise ProtocolError("num_workers には0以上を指定してください")

    cpu_threads = _expect_int(raw.get("cpu_threads", 0), "cpu_threads")
    if cpu_threads < 0:
        raise ProtocolError("cpu_threads には0以上を指定してください")

    models_raw = _expect_mapping(raw.get("models"), "models")
    if not models_raw:
        raise ProtocolError("models には1つ以上のモデル設定を指定してください")

    models: dict[str, ModelConfig] = {}
    for model_name, model_value in models_raw.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ProtocolError("models のキーには空でない文字列を指定してください")
        models[model_name] = _model_config_from_dict(model_name, model_value)

    return Protocol(
        protocol_name=protocol_name,
        protocol_version=protocol_version,
        dataset=dataset,
        val_split_mode=val_split_mode,
        val_split_ratio=val_split_ratio,
        seed=seed,
        image_size=image_size,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        cpu_threads=cpu_threads,
        models=models,
    )


def load_protocol(path: Path | None = None) -> Protocol:
    """JSONファイルからプロトコルを読み込む。"""
    protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    try:
        text = protocol_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ProtocolError(f"プロトコル設定を読み込めません: {protocol_path}: {error}") from error

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"プロトコル設定のJSONが不正です: {protocol_path}: {error}") from error

    if not isinstance(data, Mapping):
        raise ProtocolError("プロトコル設定のルートはJSONオブジェクトにしてください")

    return protocol_from_dict(data)


def protocol_file_sha256(path: Path) -> str:
    """プロトコル設定ファイルのバイト列に対するSHA-256を返す。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_config_to_dict(model_config: ModelConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if model_config.backbone is not None:
        result["backbone"] = model_config.backbone
    if model_config.layers is not None:
        result["layers"] = list(model_config.layers)
    if model_config.coreset_sampling_ratio is not None:
        result["coreset_sampling_ratio"] = model_config.coreset_sampling_ratio
    if model_config.num_neighbors is not None:
        result["num_neighbors"] = model_config.num_neighbors
    if model_config.model_size is not None:
        result["model_size"] = model_config.model_size
    if model_config.max_epochs is not None:
        result["max_epochs"] = model_config.max_epochs
    if model_config.max_steps is not None:
        result["max_steps"] = model_config.max_steps
    return result


def protocol_to_dict(protocol: Protocol) -> dict[str, Any]:
    """レポートへ保存できるJSON互換の辞書へ変換する。"""
    return {
        "protocol_name": protocol.protocol_name,
        "protocol_version": protocol.protocol_version,
        "dataset": protocol.dataset,
        "val_split_mode": protocol.val_split_mode,
        "val_split_ratio": protocol.val_split_ratio,
        "seed": protocol.seed,
        "image_size": protocol.image_size,
        "train_batch_size": protocol.train_batch_size,
        "eval_batch_size": protocol.eval_batch_size,
        "num_workers": protocol.num_workers,
        "cpu_threads": protocol.cpu_threads,
        "models": {
            name: _model_config_to_dict(protocol.models[name])
            for name in sorted(protocol.models)
        },
    }


def apply_overrides(protocol: Protocol, overrides: Mapping[str, Any]) -> Protocol:
    """CLIで明示指定された項目だけを適用したプロトコルを返す。

    トップレベルの項目と ``models.<name>.<field>`` 形式のモデル別項目を
    受け付ける。適用後に全項目を再検証する。
    """
    if not overrides:
        return protocol

    data = protocol_to_dict(protocol)
    models = dict(data.get("models", {}))
    data["models"] = models

    for key, value in overrides.items():
        if not isinstance(key, str):
            raise ProtocolError("overrides のキーには文字列を指定してください")

        if key.startswith("models."):
            parts = key.split(".")
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise ProtocolError(f"モデル別上書きのキーが不正です: {key}")
            _, model_name, field = parts
            if field not in _MODEL_FIELDS:
                raise ProtocolError(f"未知のモデル別項目です: {field}")
            model_values = dict(models.get(model_name, {}))
            model_values[field] = list(value) if isinstance(value, tuple) else value
            models[model_name] = model_values
            continue

        if key not in _TOP_LEVEL_FIELDS:
            raise ProtocolError(f"未知の上書き項目です: {key}")
        if key == "models":
            data["models"] = value
        else:
            data[key] = list(value) if isinstance(value, tuple) else value

    return protocol_from_dict(data)
