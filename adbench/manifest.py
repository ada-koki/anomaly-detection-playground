"""データ分割マニフェストの作成と検証。

pandas DataFrame を直接importせず、``to_dict("records")`` などの
最小限のインターフェースだけを利用する。純粋関数部分はunittestで検証できる。

重複検出は既定で厳格（同一 image_path が複数行に現れたらエラー）だが、
``val_split_mode=same_as_test`` 用に「val と test が完全一致する場合のみ」
重複を許容するモードを持つ（Issue #7）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLES = ("train", "val", "test")
MANIFEST_SCHEMA_VERSION = 1

OVERLAP_REASON_SAME_AS_TEST = (
    "val_split_mode=same_as_test では val が test の複製になるため、"
    "val と test が完全一致し、各画像が役割ごとに1回だけ現れる場合に限り"
    "重複を許容しています。閾値はテスト集合で調整されるため、"
    "閾値依存指標は楽観的な値になります。"
)


class ManifestError(ValueError):
    """マニフェスト作成・検証に失敗した場合に送出する例外。"""


@dataclass(frozen=True)
class ManifestRow:
    """マニフェスト1行分の情報。"""

    role: str
    image_path: str
    label: str
    label_index: int
    mask_path: str | None = None


def _row_to_dict(row: ManifestRow) -> dict[str, Any]:
    return {
        "role": row.role,
        "image_path": row.image_path,
        "label": row.label,
        "label_index": row.label_index,
        "mask_path": row.mask_path,
    }


def _row_sort_key(row: ManifestRow) -> tuple[str, str, str, int, str]:
    return (
        row.role,
        row.image_path,
        row.label,
        row.label_index,
        row.mask_path or "",
    )


def manifest_sha256(rows: Sequence[ManifestRow]) -> str:
    """行を正規化してソートしたJSONのSHA-256を返す。"""
    payload = [_row_to_dict(row) for row in sorted(rows, key=_row_sort_key)]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def find_duplicate_image_paths(rows: Sequence[ManifestRow]) -> dict[str, list[str]]:
    """複数行に現れる image_path と、その役割一覧を返す。"""
    roles_by_path: dict[str, list[str]] = {}
    for row in rows:
        roles_by_path.setdefault(row.image_path, []).append(row.role)

    duplicates: dict[str, list[str]] = {}
    for image_path, roles in roles_by_path.items():
        if len(roles) > 1:
            duplicates[image_path] = sorted(set(roles))
    return dict(sorted(duplicates.items()))


def evaluate_manifest_overlap(
    rows: Sequence[ManifestRow],
    *,
    allow_val_test_overlap: bool = False,
) -> dict[str, Any]:
    """役割間・同一役割内の画像パス重複を評価する。

    Args:
        rows: マニフェスト行。
        allow_val_test_overlap: ``val_split_mode=same_as_test`` 用の許可。
            True のとき、val と test の画像集合が完全一致し、各画像が
            どちらの役割にも1回だけ現れる場合に限り重複を許容する。

    Returns:
        ``overlap_detected`` / ``overlap_allowed`` / ``overlap_image_count`` /
        ``overlap_reason`` / ``duplicate_roles`` を含む辞書。

    Raises:
        ManifestError: 許容できない重複がある場合。
    """
    row_list = list(rows)
    duplicates = find_duplicate_image_paths(row_list)

    role_paths: dict[str, set[str]] = {role: set() for role in ROLES}
    counts: dict[tuple[str, str], int] = {}
    for row in row_list:
        role_paths[row.role].add(row.image_path)
        key = (row.role, row.image_path)
        counts[key] = counts.get(key, 0) + 1

    if not duplicates:
        return {
            "overlap_detected": False,
            "overlap_allowed": False,
            "overlap_image_count": 0,
            "overlap_reason": None,
            "duplicate_roles": {},
        }

    if allow_val_test_overlap:
        only_val_test = all(roles == ["test", "val"] for roles in duplicates.values())
        single_row_per_role = all(count == 1 for count in counts.values())
        roles_identical = role_paths["val"] == role_paths["test"]
        if only_val_test and single_row_per_role and roles_identical:
            return {
                "overlap_detected": True,
                "overlap_allowed": True,
                "overlap_image_count": len(duplicates),
                "overlap_reason": OVERLAP_REASON_SAME_AS_TEST,
                "duplicate_roles": duplicates,
            }

    preview = ", ".join(
        f"{path} ({'/'.join(roles)})" for path, roles in list(duplicates.items())[:5]
    )
    message = f"同一 image_path が複数の役割または重複して含まれています: {preview}"
    if allow_val_test_overlap:
        message += (
            "（val_split_mode=same_as_test では、val と test が完全一致し、"
            "各画像が役割ごとに1回だけ現れる場合のみ重複を許容します）"
        )
    raise ManifestError(message)


def summarize_manifest(
    rows: Sequence[ManifestRow],
    *,
    allow_val_test_overlap: bool = False,
) -> dict[str, Any]:
    """役割別件数・label別件数・重複有無・SHA-256を返す。

    既定では同一 image_path が複数行に現れる場合に ``ManifestError`` を送出する。
    ``allow_val_test_overlap=True`` のときだけ、val と test が完全一致する
    重複を許容し、``overlap_allowed`` と理由を返す。
    """
    row_list = list(rows)
    for row in row_list:
        if not isinstance(row, ManifestRow):
            raise ManifestError("ManifestRow 以外の要素が含まれています")
        if row.role not in ROLES:
            raise ManifestError(f"未知の役割です: {row.role}")

    overlap = evaluate_manifest_overlap(
        row_list,
        allow_val_test_overlap=allow_val_test_overlap,
    )

    roles: dict[str, dict[str, Any]] = {
        role: {"count": 0, "label_counts": {}} for role in ROLES
    }
    for row in row_list:
        role_summary = roles[row.role]
        role_summary["count"] += 1
        label_counts = role_summary["label_counts"]
        label_counts[row.label] = label_counts.get(row.label, 0) + 1

    for role_summary in roles.values():
        role_summary["label_counts"] = dict(sorted(role_summary["label_counts"].items()))

    return {
        "roles": roles,
        "overlap_detected": overlap["overlap_detected"],
        "overlap_allowed": overlap["overlap_allowed"],
        "overlap_image_count": overlap["overlap_image_count"],
        "overlap_reason": overlap["overlap_reason"],
        "manifest_sha256": manifest_sha256(row_list),
    }


def build_manifest(
    rows: Sequence[ManifestRow],
    dataset_root: Path,
    *,
    allow_val_test_overlap: bool = False,
) -> dict[str, Any]:
    """保存用の完全なマニフェスト辞書を構築する。"""
    row_list = list(rows)
    summary = summarize_manifest(
        row_list,
        allow_val_test_overlap=allow_val_test_overlap,
    )

    splits: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for row in sorted(row_list, key=_row_sort_key):
        splits[row.role].append(_row_to_dict(row))

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_root": str(Path(dataset_root).resolve()),
        "splits": splits,
        "summary": summary,
    }


def write_manifest(
    path: Path,
    rows: Sequence[ManifestRow],
    dataset_root: Path,
    *,
    allow_val_test_overlap: bool = False,
) -> dict[str, Any]:
    """マニフェストをJSONで保存し、サマリーを返す。"""
    manifest = build_manifest(
        rows,
        dataset_root,
        allow_val_test_overlap=allow_val_test_overlap,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest["summary"]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if text in {"", "nan", "None", "<NA>"}:
        return None
    return text


def _relative_path(value: Any, dataset_root: Path) -> str:
    text = _optional_text(value)
    if text is None:
        raise ManifestError("image_path または mask_path が空です")

    path = Path(text)
    root = Path(dataset_root)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(os.path.relpath(path, root))
    return relative.as_posix()


def _records_from_samples(samples: Any) -> list[dict[str, Any]]:
    if samples is None:
        return []

    to_dict = getattr(samples, "to_dict", None)
    if callable(to_dict):
        records = to_dict("records")
        return [dict(record) for record in records]

    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
        return [dict(record) for record in samples]

    raise ManifestError("samples をレコード一覧へ変換できません")


def rows_from_samples(
    samples: Any,
    role: str,
    dataset_root: Path,
) -> list[ManifestRow]:
    """Anomalibのsamplesからマニフェスト行を作る。"""
    if role not in ROLES:
        raise ManifestError(f"未知の役割です: {role}")

    rows: list[ManifestRow] = []
    for record in _records_from_samples(samples):
        image_value = record.get("image_path")
        if image_value is None:
            raise ManifestError("samples に image_path がありません")
        image_path = _relative_path(image_value, dataset_root)

        mask_text = _optional_text(record.get("mask_path"))
        mask_path = _relative_path(mask_text, dataset_root) if mask_text is not None else None

        label = str(record.get("label", ""))
        label_index_value = record.get("label_index", 0)
        try:
            label_index = int(label_index_value)
        except (TypeError, ValueError) as error:
            raise ManifestError(f"label_index を整数に変換できません: {label_index_value}") from error

        rows.append(
            ManifestRow(
                role=role,
                image_path=image_path,
                label=label,
                label_index=label_index,
                mask_path=mask_path,
            )
        )
    return rows


def collect_manifest_rows(datamodule: Any, dataset_root: Path) -> list[ManifestRow]:
    """setup済みDataModuleのtrain/val/testからマニフェスト行を集める。"""
    rows: list[ManifestRow] = []
    for role, attribute in (
        ("train", "train_data"),
        ("val", "val_data"),
        ("test", "test_data"),
    ):
        data = getattr(datamodule, attribute, None)
        samples = getattr(data, "samples", None)
        if samples is None:
            continue
        rows.extend(rows_from_samples(samples, role, dataset_root))
    return rows


def build_data_split_report(
    summary: Mapping[str, Any],
    threshold_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """レポートの data_split セクションを作る。"""
    roles = {
        role: {
            "count": int(summary["roles"][role]["count"]),
            "label_counts": dict(summary["roles"][role]["label_counts"]),
        }
        for role in ROLES
    }
    return {
        "roles": roles,
        "overlap_detected": bool(summary["overlap_detected"]),
        "overlap_allowed": bool(summary.get("overlap_allowed", False)),
        "overlap_image_count": int(summary.get("overlap_image_count", 0)),
        "overlap_reason": summary.get("overlap_reason"),
        "manifest_sha256": summary["manifest_sha256"],
        "threshold_selection": dict(threshold_selection),
    }
