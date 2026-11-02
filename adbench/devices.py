"""実行デバイス・実行モード・チェックポイント来歴（provenance）の判定。

torch / anomalib などの重い依存を持たず、unittest からそのまま利用できる
純粋関数だけを置く。GPU 学習＋CPU 推論は 1 プロセス内でデバイスを切り替えず、
「学習してチェックポイントを保存する実行」と「チェックポイントを CPU で
推論する実行」の 2 段階に分けて実現する（Issue #11）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

EXECUTION_MODE_CPU_ONLY = "cpu_only"
EXECUTION_MODE_GPU_TRAIN_CPU_INFER = "gpu_train_cpu_infer"
EXECUTION_MODE_GPU_REFERENCE = "gpu_reference"
EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE = (
    "external_checkpoint_unknown_train_device"
)
EXECUTION_MODE_CPU_TRAIN_GPU_INFER = "cpu_train_gpu_infer"

#: 集計・表示で使う実行モードの定義順。
EXECUTION_MODES: tuple[str, ...] = (
    EXECUTION_MODE_CPU_ONLY,
    EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
    EXECUTION_MODE_GPU_REFERENCE,
    EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
    EXECUTION_MODE_CPU_TRAIN_GPU_INFER,
)

#: 来歴サイドカーのスキーマ版。
PROVENANCE_SCHEMA_VERSION = 1

# 正規化後に受け付けるデバイス種別。実際に使えるかは実行環境に依存する。
_KNOWN_DEVICE_TYPES = ("cpu", "cuda", "mps")


def normalize_device(device: str) -> str:
    """デバイス文字列を種別（``cpu`` / ``cuda`` / ``mps``）へ正規化する。

    ``"cuda:0"`` のようなインデックス付き指定は ``"cuda"`` に落とす。

    Args:
        device: ``"cpu"``、``"cuda"``、``"cuda:0"`` などの文字列。

    Returns:
        正規化後のデバイス種別。

    Raises:
        ValueError: 文字列でない、空文字、未知の種別、インデックスが数値でない場合。
    """
    if not isinstance(device, str):
        raise ValueError(f"デバイス文字列が不正です: {device!r}")
    text = device.strip().lower()
    if not text:
        raise ValueError("デバイス文字列が空です")
    base, separator, index = text.partition(":")
    if separator and not index.isdigit():
        raise ValueError(f"デバイスインデックスが不正です: {device!r}")
    if base not in _KNOWN_DEVICE_TYPES:
        raise ValueError(f"未知のデバイス種別です: {device!r}")
    return base


def classify_execution_mode(train_device: str | None, infer_device: str) -> str:
    """学習・推論デバイスから実行モードを分類する。

    Args:
        train_device: 学習を実行したデバイス。学習していない（外部チェックポイント）
            場合や、来歴が不明・信頼できない場合は ``None``。
        infer_device: 推論を実行したデバイス。

    Returns:
        ``EXECUTION_MODES`` のいずれか。

    Raises:
        ValueError: デバイス文字列が不正、または対応する実行モードが無い組合せの場合。
    """
    infer = normalize_device(infer_device)
    if train_device is None:
        return EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE
    train = normalize_device(train_device)
    if train == "cpu" and infer == "cpu":
        return EXECUTION_MODE_CPU_ONLY
    if train == "cuda" and infer == "cpu":
        return EXECUTION_MODE_GPU_TRAIN_CPU_INFER
    if train == "cuda" and infer == "cuda":
        return EXECUTION_MODE_GPU_REFERENCE
    if train == "cpu" and infer == "cuda":
        return EXECUTION_MODE_CPU_TRAIN_GPU_INFER
    raise ValueError(
        "実行モードを判定できないデバイスの組合せです: "
        f"train={train!r}, infer={infer!r}"
    )


def detect_device_mismatch(
    requested: str, observed: Iterable[str]
) -> list[str]:
    """要求デバイスと観測デバイスの不一致を警告文のリストにする。

    同じ組み合わせの警告は重複させない。``requested`` と ``observed`` の
    いずれかに不正なデバイス文字列が含まれる場合は :class:`ValueError` を送出する。

    Args:
        requested: CLI などで要求したデバイス。
        observed: 実際に観測したデバイス種別の列。

    Returns:
        日本語の警告文のリスト。一致していれば空リスト。
    """
    requested_norm = normalize_device(requested)
    warnings: list[str] = []
    seen: set[str] = set()
    for item in observed:
        if item is None:  # 型ヒント上は str だが、呼び出し側の保険。
            continue
        observed_norm = normalize_device(item)
        if observed_norm == requested_norm:
            continue
        message = (
            f"要求デバイス {requested_norm} と観測デバイス {observed_norm} が"
            "一致しません。意図しないデバイスフォールバックの可能性があります。"
        )
        if message in seen:
            continue
        seen.add(message)
        warnings.append(message)
    return warnings


def effective_device(requested: str, observed: str | None) -> str:
    """観測デバイスが妥当ならそれを、そうでなければ要求デバイスを返す。

    パラメータを持たないモデルなどで ``observed`` が ``"unknown"`` や ``None``
    になる場合に、レポート用の代表デバイスを決めるために使う。
    """
    if isinstance(observed, str) and observed and observed != "unknown":
        try:
            return normalize_device(observed)
        except ValueError:
            pass
    return normalize_device(requested)


def build_provenance(
    *,
    checkpoint_sha256: str,
    train_device: str | None,
    observed_train_device: str | None,
    cuda_device_name: str | None,
    torch_version: str | None,
    anomalib_version: str | None,
    model: str,
    category: str,
    protocol_name: str | None,
    protocol_sha256: str | None,
    run_id: str,
) -> dict[str, Any]:
    """チェックポイントの来歴（学習デバイス等）を辞書で組み立てる。

    ``train_device`` と ``observed_train_device`` は正規化して保存する。
    観測値が ``"unknown"`` など判定できない値の場合は ``None`` にする。
    """
    train: str | None = None
    if train_device is not None:
        train = normalize_device(train_device)

    observed: str | None = None
    if isinstance(observed_train_device, str):
        try:
            observed = normalize_device(observed_train_device)
        except ValueError:
            observed = None

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_sha256": str(checkpoint_sha256),
        "train_device": train,
        "observed_train_device": observed,
        "cuda_device_name": cuda_device_name,
        "torch_version": torch_version,
        "anomalib_version": anomalib_version,
        "model": model,
        "category": category,
        "protocol_name": protocol_name,
        "protocol_sha256": protocol_sha256,
        "run_id": run_id,
    }


def provenance_path_for(checkpoint: str | Path) -> Path:
    """チェックポイントに対応する来歴サイドカーのパスを返す。

    ``model.ckpt`` に対して ``model.ckpt.provenance.json`` を返す。
    """
    path = Path(checkpoint)
    return path.with_suffix(path.suffix + ".provenance.json")


def write_provenance(path: str | Path, data: Mapping[str, Any]) -> None:
    """来歴辞書を UTF-8 JSON として書き出す。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        dict(data), ensure_ascii=False, indent=2, sort_keys=True
    )
    target.write_text(text + "\n", encoding="utf-8")


def read_provenance(path: str | Path) -> dict[str, Any] | None:
    """来歴サイドカーを読み込む。

    Args:
        path: サイドカーのパス。

    Returns:
        来歴辞書。ファイルが存在しない場合は ``None``。

    Raises:
        ValueError: ファイルは存在するが JSON として読めない、または
            トップレベルがオブジェクトでない場合。
    """
    target = Path(path)
    if not target.is_file():
        return None
    text = target.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"来歴ファイルを JSON として読めません: {target.name}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError(
            f"来歴ファイルのトップレベルがオブジェクトではありません: {target.name}"
        )
    return data


def provenance_matches(
    provenance: Mapping[str, Any] | None, checkpoint_sha256: str | None
) -> bool:
    """来歴の ``checkpoint_sha256`` が実ファイルの sha256 と一致するかを返す。

    来歴が ``None``、必須キーが無い、値が文字列でない場合は ``False``。
    大文字小文字と前後の空白は無視する。
    """
    if not isinstance(provenance, Mapping):
        return False
    recorded = provenance.get("checkpoint_sha256")
    if not isinstance(recorded, str) or not recorded.strip():
        return False
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256.strip():
        return False
    return recorded.strip().lower() == checkpoint_sha256.strip().lower()


def execution_mode_of_report(report: Mapping[str, Any] | None) -> str:
    """report から実行モードを決める。

    新形式（``execution_mode`` あり）はその値をそのまま使う。旧形式
    （Issue #10 以前）は ``training_mode`` と ``device`` から推定する。
    """
    if not isinstance(report, Mapping):
        return EXECUTION_MODE_CPU_ONLY

    mode = report.get("execution_mode")
    if isinstance(mode, str) and mode:
        return mode

    training_mode = report.get("training_mode")
    if training_mode == "external_checkpoint_cpu_infer":
        return EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE
    if training_mode == "cpu_train_cpu_infer":
        return EXECUTION_MODE_CPU_ONLY

    # training_mode が無い旧形式。device が cuda なら GPU 参考実行とみなす。
    device = _report_device(report)
    if device is not None:
        try:
            if normalize_device(device) == "cuda":
                return EXECUTION_MODE_GPU_REFERENCE
        except ValueError:
            pass
    return EXECUTION_MODE_CPU_ONLY


def _report_device(report: Mapping[str, Any]) -> str | None:
    """旧形式 report に記録されたデバイス文字列を探す。"""
    candidates: list[Any] = [report.get("device")]
    for key in ("conditions", "args", "environment"):
        nested = report.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested.get("device"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None
