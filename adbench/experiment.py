"""実験マトリクス設定の読み込み・検証・展開。

設定は JSON で記述し、`expand()` で run_anomalib.py に渡す
1 実行分の `RunSpec` に展開する。標準ライブラリのみを使う。

repeat 間で seed は変えない。seed はプロトコル JSON 側で固定し、
繰り返しは同一条件での計測ばらつき評価だけを目的とする。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

KNOWN_MODELS = ("padim", "patchcore")

# run_key と argv の対応を壊すため、設定側から上書きできない引数。
FORBIDDEN_EXTRA_ARGS = {
    "--output-dir",
    "--protocol",
    "--dataset-root",
    "--category",
    "--model",
    "--image-size",
    "--cpu-threads",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """実験マトリクス設定。"""

    name: str
    protocol: str
    dataset_root: str
    models: list[str]
    categories: list[str]
    image_sizes: list[int]
    cpu_threads: list[int]
    repeats: int
    extra_args: list[str] = field(default_factory=list)
    model_extra_args: dict[str, list[str]] = field(default_factory=dict)
    keep_checkpoints: bool = False


@dataclass(frozen=True)
class RunSpec:
    """1 実行分の展開結果。`argv` に `--output-dir` は含めない。"""

    run_key: str
    model: str
    category: str
    image_size: int
    cpu_threads: int
    repeat: int
    argv: tuple[str, ...]


def load_config(path: str | Path) -> ExperimentConfig:
    """JSON ファイルを読み込んで設定を返す。"""

    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"設定 JSON を解析できません: {config_path}: {error}") from error
    if not isinstance(data, Mapping):
        raise ValueError("設定 JSON のトップレベルはオブジェクトである必要があります")
    return parse_config(data)


def parse_config(data: Mapping[str, Any]) -> ExperimentConfig:
    """辞書から `ExperimentConfig` を作り、値を検証する。"""

    if not isinstance(data, Mapping):
        raise ValueError("設定はオブジェクトである必要があります")

    required = (
        "name",
        "protocol",
        "dataset_root",
        "models",
        "categories",
        "image_sizes",
        "cpu_threads",
        "repeats",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"設定項目がありません: {key}")

    name = data["name"]
    protocol = data["protocol"]
    dataset_root = data["dataset_root"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name は空でない文字列である必要があります")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("protocol は空でない文字列である必要があります")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError("dataset_root は空でない文字列である必要があります")

    models = _string_list(data["models"], "models")
    categories = _string_list(data["categories"], "categories")
    image_sizes = _int_list(data["image_sizes"], "image_sizes")
    cpu_threads = _int_list(data["cpu_threads"], "cpu_threads")

    repeats = data["repeats"]
    if not _is_int(repeats):
        raise ValueError("repeats は整数である必要があります")

    extra_args = _string_list(data.get("extra_args", []), "extra_args")

    raw_model_extra = data.get("model_extra_args", {})
    if not isinstance(raw_model_extra, Mapping):
        raise ValueError("model_extra_args はオブジェクトである必要があります")
    model_extra_args: dict[str, list[str]] = {}
    for model_name, args in raw_model_extra.items():
        if not isinstance(model_name, str):
            raise ValueError("model_extra_args のキーは文字列である必要があります")
        model_extra_args[model_name] = _string_list(
            args, f"model_extra_args.{model_name}"
        )

    keep_checkpoints = data.get("keep_checkpoints", False)
    if not isinstance(keep_checkpoints, bool):
        raise ValueError("keep_checkpoints は真偽値である必要があります")

    config = ExperimentConfig(
        name=name,
        protocol=protocol,
        dataset_root=dataset_root,
        models=models,
        categories=categories,
        image_sizes=image_sizes,
        cpu_threads=cpu_threads,
        repeats=repeats,
        extra_args=extra_args,
        model_extra_args=model_extra_args,
        keep_checkpoints=keep_checkpoints,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """設定値を検証し、不正なら ValueError を送出する。"""

    if not isinstance(config.name, str) or not config.name.strip():
        raise ValueError("name は空でない文字列である必要があります")
    if not isinstance(config.protocol, str) or not config.protocol.strip():
        raise ValueError("protocol は空でない文字列である必要があります")
    if not isinstance(config.dataset_root, str) or not config.dataset_root.strip():
        raise ValueError("dataset_root は空でない文字列である必要があります")

    if not config.models:
        raise ValueError("models は 1 件以上必要です")
    seen_models: set[str] = set()
    for model in config.models:
        if not isinstance(model, str):
            raise ValueError("models の要素は文字列である必要があります")
        if model not in KNOWN_MODELS:
            raise ValueError(f"未知のモデルです: {model}")
        if model in seen_models:
            raise ValueError(f"models に重複があります: {model}")
        seen_models.add(model)

    if not config.categories:
        raise ValueError("categories は 1 件以上必要です")
    seen_categories: set[str] = set()
    for category in config.categories:
        if not isinstance(category, str) or not category:
            raise ValueError("categories の要素は空でない文字列である必要があります")
        if "/" in category or "\\" in category or category in {".", ".."}:
            raise ValueError(f"categories にパスとして危険な値があります: {category}")
        if category in seen_categories:
            raise ValueError(f"categories に重複があります: {category}")
        seen_categories.add(category)

    if not config.image_sizes:
        raise ValueError("image_sizes は 1 件以上必要です")
    for image_size in config.image_sizes:
        if not _is_int(image_size) or image_size <= 0:
            raise ValueError("image_sizes は 1 以上の整数である必要があります")

    if not config.cpu_threads:
        raise ValueError("cpu_threads は 1 件以上必要です")
    for threads in config.cpu_threads:
        if not _is_int(threads) or threads < 0:
            raise ValueError("cpu_threads は 0 以上の整数である必要があります")

    if not _is_int(config.repeats) or config.repeats < 1:
        raise ValueError("repeats は 1 以上の整数である必要があります")

    if not isinstance(config.extra_args, list):
        raise ValueError("extra_args はリストである必要があります")
    _validate_extra_args(config.extra_args, "extra_args")

    if not isinstance(config.model_extra_args, Mapping):
        raise ValueError("model_extra_args はオブジェクトである必要があります")
    for model_name, args in config.model_extra_args.items():
        if model_name not in seen_models:
            raise ValueError(
                f"model_extra_args のキーが models にありません: {model_name}"
            )
        if not isinstance(args, list):
            raise ValueError(
                f"model_extra_args.{model_name} はリストである必要があります"
            )
        _validate_extra_args(args, f"model_extra_args.{model_name}")

    if not isinstance(config.keep_checkpoints, bool):
        raise ValueError("keep_checkpoints は真偽値である必要があります")


def expand(config: ExperimentConfig) -> list[RunSpec]:
    """設定から実行リストを決定的な順序で展開する。

    順序は model, category, image_size, cpu_threads, repeat の
    設定に書かれた順のネストである。repeat 間で seed は変えない。
    """

    validate_config(config)
    specs: list[RunSpec] = []
    for model in config.models:
        for category in config.categories:
            for image_size in config.image_sizes:
                for cpu_threads in config.cpu_threads:
                    for repeat in range(1, config.repeats + 1):
                        run_key = (
                            f"{model}__{category}__s{image_size}"
                            f"__t{cpu_threads}__r{repeat}"
                        )
                        argv = _build_argv(
                            config=config,
                            model=model,
                            category=category,
                            image_size=image_size,
                            cpu_threads=cpu_threads,
                        )
                        specs.append(
                            RunSpec(
                                run_key=run_key,
                                model=model,
                                category=category,
                                image_size=image_size,
                                cpu_threads=cpu_threads,
                                repeat=repeat,
                                argv=tuple(argv),
                            )
                        )
    return specs


def _build_argv(
    *,
    config: ExperimentConfig,
    model: str,
    category: str,
    image_size: int,
    cpu_threads: int,
) -> list[str]:
    argv = [
        "--protocol",
        config.protocol,
        "--dataset-root",
        config.dataset_root,
        "--category",
        category,
        "--model",
        model,
        "--image-size",
        str(image_size),
        "--cpu-threads",
        str(cpu_threads),
    ]
    argv.extend(config.extra_args)
    argv.extend(config.model_extra_args.get(model, []))
    return argv


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} はリストである必要があります")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} の要素は文字列である必要があります")
        result.append(item)
    return result


def _int_list(value: Any, key: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{key} はリストである必要があります")
    result: list[int] = []
    for item in value:
        if not _is_int(item):
            raise ValueError(f"{key} の要素は整数である必要があります")
        result.append(item)
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_extra_args(args: list[str], label: str) -> None:
    for arg in args:
        if not isinstance(arg, str):
            raise ValueError(f"{label} の要素は文字列である必要があります")
        name = arg.split("=", 1)[0]
        if name in FORBIDDEN_EXTRA_ARGS:
            raise ValueError(f"{label} に上書き禁止の引数があります: {name}")
