"""カテゴリ単位の計算予算の定義・判定と、データセット単位の要約。

予算は「1 カテゴリ・1 run あたり」の上限として扱い、report.json から
1 行の run レコードを組み立てて判定する。値は未検証の暫定値を含むため、
`is_provisional` を明示できるようにしている。
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CategoryBudget",
    "budget_from_dict",
    "budget_to_dict",
    "load_budget",
    "run_row_from_report",
    "judge_budget",
    "summarize_by_dataset",
    "write_csv",
    "render_markdown",
    "RUN_COLUMNS",
    "SUMMARY_COLUMNS",
]


RUN_COLUMNS: list[str] = [
    "condition",
    "dataset",
    "category",
    "model",
    "variant",
    "status",
    "image_auroc",
    "pixel_auroc",
    "train_stage_seconds",
    "test_stage_seconds",
    "peak_rss_mib",
    "e2e_p95_ms",
    "latency_valid",
    "train_images",
    "test_images",
    "run_id",
]

SUMMARY_COLUMNS: list[str] = [
    "dataset",
    "model",
    "variant",
    "n_runs",
    "n_ok",
    "n_failed",
    "n_within_budget",
    "categories",
    "image_auroc_mean",
    "image_auroc_std",
    "image_auroc_min",
    "image_auroc_max",
    "pixel_auroc_mean",
    "pixel_auroc_std",
    "pixel_auroc_min",
    "pixel_auroc_max",
    "train_stage_seconds_total",
    "train_stage_seconds_max",
    "peak_rss_mib_max",
]

_KNOWN_KEYS = {
    "name",
    "max_train_stage_seconds",
    "max_peak_rss_mib",
    "max_download_mib",
    "is_provisional",
    "notes",
}

_NUMERIC_KEYS = (
    "max_train_stage_seconds",
    "max_peak_rss_mib",
    "max_download_mib",
)


@dataclass(frozen=True)
class CategoryBudget:
    """1 カテゴリ・1 run あたりの計算予算。"""

    name: str
    max_train_stage_seconds: float | None
    max_peak_rss_mib: float | None
    max_download_mib: float | None
    is_provisional: bool = True
    notes: str = ""


def _optional_number(data: Mapping[str, Any], key: str) -> float | None:
    if key not in data:
        return None
    value = data[key]
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} に bool は指定できません: {value!r}")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} は数値で指定してください: {value!r}")
    if value < 0:
        raise ValueError(f"{key} に負値は指定できません: {value!r}")
    return float(value)


def budget_from_dict(data: Mapping[str, Any]) -> CategoryBudget:
    """辞書から `CategoryBudget` を組み立てる。不正な入力は `ValueError`。"""
    if not isinstance(data, Mapping):
        raise ValueError("予算の定義はマッピング（dict）で指定してください")
    unknown = set(data.keys()) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"未知のキーが含まれています: {sorted(unknown)}")
    if "name" not in data:
        raise ValueError("name は必須です")
    name = data["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"name は空でない文字列で指定してください: {name!r}")

    numbers = {k: _optional_number(data, k) for k in _NUMERIC_KEYS}

    is_provisional = data.get("is_provisional", True)
    if not isinstance(is_provisional, bool):
        raise ValueError(f"is_provisional は bool で指定してください: {is_provisional!r}")

    notes = data.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError(f"notes は文字列で指定してください: {notes!r}")

    return CategoryBudget(
        name=name,
        max_train_stage_seconds=numbers["max_train_stage_seconds"],
        max_peak_rss_mib=numbers["max_peak_rss_mib"],
        max_download_mib=numbers["max_download_mib"],
        is_provisional=is_provisional,
        notes=notes,
    )


def load_budget(path: str | Path) -> CategoryBudget:
    """JSON ファイルから予算を読み込む。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"予算ファイルの最上位は JSON オブジェクトである必要があります: {path}")
    return budget_from_dict(data)


def budget_to_dict(b: CategoryBudget) -> dict:
    """`CategoryBudget` を辞書へ変換する。"""
    return asdict(b)


def _sequence_of_mapping(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _variant_from_report(report: Mapping[str, Any]) -> str:
    mitigation = report.get("training_memory_mitigation")
    if not isinstance(mitigation, Mapping):
        return ""
    if mitigation.get("coreset_mode") != "chunked":
        return ""
    chunk_images = mitigation.get("chunk_images")
    return f"chunked_coreset_{chunk_images}img"


def run_row_from_report(
    report: Mapping[str, Any] | None,
    *,
    condition: str,
    dataset: str,
    category: str,
    model: str,
    failure_status: str | None = None,
) -> dict:
    """report.json から run 1 行分のレコードを作る。

    report が None のとき、または status が "ok" でないときは失敗扱いとし、
    数値列はすべて None にする。ネストしたキーが欠けても例外にしない。
    """
    base = {
        "condition": condition,
        "dataset": dataset,
        "category": category,
        "model": model,
    }
    empty = {
        "variant": "",
        "image_auroc": None,
        "pixel_auroc": None,
        "train_stage_seconds": None,
        "test_stage_seconds": None,
        "peak_rss_mib": None,
        "e2e_p95_ms": None,
        "latency_valid": None,
        "train_images": None,
        "test_images": None,
        "run_id": None,
    }

    if report is None:
        status = failure_status if failure_status is not None else "failed"
        return {**base, "status": status, **empty}

    status = report.get("status")
    if status != "ok":
        if failure_status is not None:
            resolved = failure_status
        elif status is not None:
            resolved = status
        else:
            resolved = "failed"
        return {**base, "status": resolved, **empty}

    image_auroc: Any = None
    pixel_auroc: Any = None
    for item in _sequence_of_mapping(report.get("test_metrics")):
        if isinstance(item, Mapping):
            image_auroc = item.get("image_AUROC")
            pixel_auroc = item.get("pixel_AUROC")
        break

    memory = report.get("memory")
    peak_rss: Any = None
    if isinstance(memory, Mapping):
        lifetime = memory.get("process_lifetime_peak_rss_mib")
        if lifetime is not None:
            peak_rss = lifetime
        else:
            peak_rss = memory.get("train_peak_rss_mib_sampled")

    latency = report.get("per_image_inference_latency")
    e2e_p95: Any = None
    if isinstance(latency, Mapping):
        e2e = latency.get("end_to_end_ms")
        if isinstance(e2e, Mapping):
            e2e_p95 = e2e.get("p95")

    roles = report.get("data_split")
    train_images: Any = None
    test_images: Any = None
    if isinstance(roles, Mapping):
        roles_map = roles.get("roles")
        if isinstance(roles_map, Mapping):
            train_role = roles_map.get("train")
            if isinstance(train_role, Mapping):
                train_images = train_role.get("count")
            test_role = roles_map.get("test")
            if isinstance(test_role, Mapping):
                test_images = test_role.get("count")

    return {
        **base,
        "variant": _variant_from_report(report),
        "status": status,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "train_stage_seconds": report.get("train_stage_seconds"),
        "test_stage_seconds": report.get("test_stage_seconds"),
        "peak_rss_mib": peak_rss,
        "e2e_p95_ms": e2e_p95,
        "latency_valid": report.get("latency_valid"),
        "train_images": train_images,
        "test_images": test_images,
        "run_id": report.get("run_id"),
    }


def judge_budget(
    row: Mapping[str, Any],
    budget: CategoryBudget,
    *,
    download_mib: float | None = None,
) -> dict:
    """run 1 行を予算と比較する。

    戻り値は ``within_budget``（判定不能なら None）、``reasons``、
    ``unavailable``（閾値はあるが値が None の項目名）を持つ辞書。
    """
    reasons: list[str] = []
    unavailable: list[str] = []
    within_budget: bool | None = True

    status = row.get("status")
    status_ok = status == "ok"
    if not status_ok:
        reasons.append(f"status={status}")
        within_budget = False

    checks = [
        ("train_stage_seconds", budget.max_train_stage_seconds, row.get("train_stage_seconds")),
        ("peak_rss_mib", budget.max_peak_rss_mib, row.get("peak_rss_mib")),
        ("download_mib", budget.max_download_mib, download_mib),
    ]

    judged = 0
    for name, threshold, value in checks:
        if threshold is None:
            continue
        if value is None:
            unavailable.append(name)
            continue
        judged += 1
        if value > threshold:
            within_budget = False
            reasons.append(f"{name}={value:g} > {threshold:g}")

    if status_ok and judged == 0:
        within_budget = None

    return {
        "within_budget": within_budget,
        "reasons": reasons,
        "unavailable": unavailable,
    }


def _stats(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return (None, None, None, None)
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return (mean, std, min(values), max(values))


def summarize_by_dataset(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """(dataset, model, variant) 単位で集計した要約行を返す。"""
    order: list[tuple[Any, Any, str]] = []
    groups: dict[tuple[Any, Any, str], dict[str, Any]] = {}

    for row in rows:
        variant = row.get("variant")
        key = (row.get("dataset"), row.get("model"), variant if variant is not None else "")
        if key not in groups:
            order.append(key)
            groups[key] = {
                "n_runs": 0,
                "n_ok": 0,
                "n_failed": 0,
                "n_within_budget": 0,
                "categories": [],
                "image_auroc": [],
                "pixel_auroc": [],
                "train_stage_seconds": [],
                "peak_rss_mib": [],
            }
        group = groups[key]
        group["n_runs"] += 1

        status = row.get("status")
        if status == "ok":
            group["n_ok"] += 1
            category = row.get("category")
            if category is not None:
                group["categories"].append(str(category))
            for src, bucket in (
                ("image_auroc", "image_auroc"),
                ("pixel_auroc", "pixel_auroc"),
                ("train_stage_seconds", "train_stage_seconds"),
                ("peak_rss_mib", "peak_rss_mib"),
            ):
                value = row.get(src)
                if value is not None:
                    group[bucket].append(float(value))
        else:
            group["n_failed"] += 1

        if row.get("within_budget") is True:
            group["n_within_budget"] += 1

    summaries: list[dict] = []
    for key in order:
        group = groups[key]
        img_mean, img_std, img_min, img_max = _stats(group["image_auroc"])
        pix_mean, pix_std, pix_min, pix_max = _stats(group["pixel_auroc"])
        train_values = group["train_stage_seconds"]
        rss_values = group["peak_rss_mib"]
        summaries.append(
            {
                "dataset": key[0],
                "model": key[1],
                "variant": key[2],
                "n_runs": group["n_runs"],
                "n_ok": group["n_ok"],
                "n_failed": group["n_failed"],
                "n_within_budget": group["n_within_budget"],
                "categories": ",".join(group["categories"]),
                "image_auroc_mean": img_mean,
                "image_auroc_std": img_std,
                "image_auroc_min": img_min,
                "image_auroc_max": img_max,
                "pixel_auroc_mean": pix_mean,
                "pixel_auroc_std": pix_std,
                "pixel_auroc_min": pix_min,
                "pixel_auroc_max": pix_max,
                "train_stage_seconds_total": sum(train_values) if train_values else None,
                "train_stage_seconds_max": max(train_values) if train_values else None,
                "peak_rss_mib_max": max(rss_values) if rss_values else None,
            }
        )
    return summaries


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    """指定した列順で CSV を書き出す。None は空文字になる。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: ("" if row.get(column) is None else row.get(column)) for column in columns}
            )


_AUROC_COLUMNS = {
    "image_auroc",
    "image_auroc_mean",
    "image_auroc_std",
    "image_auroc_min",
    "image_auroc_max",
    "pixel_auroc",
    "pixel_auroc_mean",
    "pixel_auroc_std",
    "pixel_auroc_min",
    "pixel_auroc_max",
}

_SECONDS_OR_MIB_COLUMNS = {
    "train_stage_seconds",
    "test_stage_seconds",
    "peak_rss_mib",
    "e2e_p95_ms",
    "train_stage_seconds_total",
    "train_stage_seconds_max",
    "peak_rss_mib_max",
}


def _fmt_plain(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _fmt_by_column(column: str, value: Any) -> str:
    if value is None:
        return "—"
    if column in _AUROC_COLUMNS:
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return f"{value:.6f}"
        return str(value)
    if column in _SECONDS_OR_MIB_COLUMNS:
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)
    return _fmt_plain(value)


def _md_table(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt_by_column(column, row.get(column)) for column in columns) + " |"
        )
    return lines


def render_markdown(
    run_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    budget: CategoryBudget,
) -> str:
    """予算定義・run 一覧・データセット要約を含む Markdown を生成する。"""
    lines: list[str] = ["# カテゴリ予算レポート", ""]

    if budget.is_provisional:
        lines.append("> **注意**: 本レポートの予算は暫定値です（`is_provisional=true`）。")
        lines.append("")

    lines.append(f"予算名: `{budget.name}`")
    lines.append("")

    lines.append("## 予算の定義")
    lines.append("")
    budget_row = {
        "name": budget.name,
        "max_train_stage_seconds": budget.max_train_stage_seconds,
        "max_peak_rss_mib": budget.max_peak_rss_mib,
        "max_download_mib": budget.max_download_mib,
        "is_provisional": budget.is_provisional,
        "notes": budget.notes,
    }
    budget_columns = [
        "name",
        "max_train_stage_seconds",
        "max_peak_rss_mib",
        "max_download_mib",
        "is_provisional",
        "notes",
    ]
    lines.extend(_md_table(budget_columns, [budget_row]))
    lines.append("")

    lines.append("## run 一覧")
    lines.append("")
    run_columns = [
        "condition",
        "dataset",
        "category",
        "model",
        "variant",
        "status",
        "image_auroc",
        "pixel_auroc",
        "train_stage_seconds",
        "test_stage_seconds",
        "peak_rss_mib",
        "e2e_p95_ms",
        "latency_valid",
        "train_images",
        "test_images",
        "run_id",
        "within_budget",
        "reasons",
    ]
    prepared_rows: list[dict[str, Any]] = []
    for raw in run_rows:
        row = dict(raw)
        reasons_value = row.get("reasons")
        if reasons_value is None:
            reasons_value = row.get("budget_reasons")
        if isinstance(reasons_value, list):
            reasons_value = ";".join(str(item) for item in reasons_value)
        if reasons_value is None:
            reasons_value = ""
        row["reasons"] = reasons_value
        prepared_rows.append(row)
    lines.extend(_md_table(run_columns, prepared_rows))
    lines.append("")

    lines.append("## データセット要約")
    lines.append("")
    lines.extend(_md_table(SUMMARY_COLUMNS, summary_rows))
    lines.append("")

    return "\n".join(lines)
