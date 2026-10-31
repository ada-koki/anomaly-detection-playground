"""実験結果の集計。

`runs.jsonl` の 1 行と report.json から runs テーブルを作り、
(model, category, image_size, cpu_threads, execution_mode) ごとの summary を作る。
標準ライブラリのみを使い、欠損値を 0 や NaN で埋めない。
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from adbench.devices import EXECUTION_MODES, execution_mode_of_report

RUN_COLUMNS = [
    "run_key",
    "model",
    "category",
    "image_size",
    "cpu_threads",
    "execution_mode",
    "repeat",
    "status",
    "image_AUROC",
    "pixel_AUROC",
    "image_F1Score",
    "pixel_F1Score",
    "image_AUROC_raw",
    "pixel_AUROC_raw",
    "train_stage_seconds",
    "test_stage_seconds",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "model_p50_ms",
    "model_p95_ms",
    "train_peak_rss_mib",
    "process_peak_rss_mib",
    "protocol_name",
    "manifest_sha256",
    "error",
    "notes",
]

NUMERIC_COLUMNS = [
    "image_AUROC",
    "pixel_AUROC",
    "image_F1Score",
    "pixel_F1Score",
    "image_AUROC_raw",
    "pixel_AUROC_raw",
    "train_stage_seconds",
    "test_stage_seconds",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "model_p50_ms",
    "model_p95_ms",
    "train_peak_rss_mib",
    "process_peak_rss_mib",
]

SUMMARY_COLUMNS = [
    "model",
    "category",
    "image_size",
    "cpu_threads",
    "execution_mode",
    "n_runs",
    "n_ok",
    "n_failed",
    "manifest_consistent",
]
for _column in NUMERIC_COLUMNS:
    SUMMARY_COLUMNS.extend(
        [
            f"{_column}_mean",
            f"{_column}_std",
            f"{_column}_min",
            f"{_column}_max",
        ]
    )


def extract_metrics(
    report: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """report から runs テーブル用の値を取り出す。

    NaN/inf は欠損として扱い、notes 用の文字列を返す。
    """

    if not isinstance(report, Mapping):
        return {}, []

    metrics: dict[str, Any] = {}
    notes: list[str] = []

    def take(column: str, value: Any) -> None:
        number, nonfinite = _finite_number(value)
        if nonfinite:
            notes.append(f"nonfinite:{column}")
            return
        if number is not None:
            metrics[column] = number

    test_metrics = report.get("test_metrics")
    test0: Mapping[str, Any] = {}
    if isinstance(test_metrics, Sequence) and test_metrics:
        candidate = test_metrics[0]
        if isinstance(candidate, Mapping):
            test0 = candidate

    take("image_AUROC", test0.get("image_AUROC"))
    take("pixel_AUROC", test0.get("pixel_AUROC"))
    take("image_F1Score", test0.get("image_F1Score"))
    take("pixel_F1Score", test0.get("pixel_F1Score"))

    raw = report.get("raw_score_metrics")
    if isinstance(raw, Mapping):
        take("image_AUROC_raw", raw.get("image_AUROC_raw"))
        take("pixel_AUROC_raw", raw.get("pixel_AUROC_raw"))

    take("train_stage_seconds", report.get("train_stage_seconds"))
    take("test_stage_seconds", report.get("test_stage_seconds"))

    latency = report.get("per_image_inference_latency")
    if isinstance(latency, Mapping):
        take("e2e_p50_ms", _get(latency, "end_to_end_ms", "p50"))
        take("e2e_p95_ms", _get(latency, "end_to_end_ms", "p95"))
        take("model_p50_ms", _get(latency, "model_ms", "p50"))
        take("model_p95_ms", _get(latency, "model_ms", "p95"))

    memory = report.get("memory")
    if isinstance(memory, Mapping):
        train_peak = memory.get("train_peak_rss_mib_sampled")
        if train_peak is None:
            train_peak = memory.get("train_peak_host_rss_mib_sampled")
        take("train_peak_rss_mib", train_peak)
        take(
            "process_peak_rss_mib",
            memory.get("process_lifetime_peak_rss_mib"),
        )

    protocol_name = _get(report, "protocol", "name")
    if isinstance(protocol_name, str) and protocol_name:
        metrics["protocol_name"] = protocol_name

    manifest = _get(report, "data_split", "manifest_sha256")
    if isinstance(manifest, str) and manifest:
        metrics["manifest_sha256"] = manifest

    return metrics, notes


def build_runs_table(
    entries: Iterable[Mapping[str, Any]],
    report_loader: Callable[[str | None], Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    """runs.jsonl 相当の entries と report loader から runs テーブルを作る。

    ``execution_mode`` は report から決める。report が読めない場合は
    値を捏造せず ``None`` のままにする。
    """

    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue

        report_path = entry.get("report_path")
        report: Mapping[str, Any] | None = None
        if isinstance(report_path, str) and report_path:
            try:
                loaded = report_loader(report_path)
            except Exception:
                loaded = None
            if isinstance(loaded, Mapping):
                report = loaded

        metrics, notes = extract_metrics(report)
        row: dict[str, Any] = {column: None for column in RUN_COLUMNS}
        for key in (
            "run_key",
            "model",
            "category",
            "image_size",
            "cpu_threads",
            "repeat",
        ):
            row[key] = entry.get(key)

        status = entry.get("status")
        if not status and report is not None:
            status = report.get("status")
        if not status:
            status = "failed"
        row["status"] = str(status)

        if isinstance(report, Mapping):
            row["execution_mode"] = execution_mode_of_report(report)
        else:
            row["execution_mode"] = None

        row.update(metrics)

        error = entry.get("error")
        if not error:
            error = _report_error(report)
        row["error"] = error
        row["notes"] = ",".join(notes)
        rows.append(row)

    rows.sort(key=_run_sort_key)
    return rows


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """runs テーブルから summary 行を作る。

    平均等は ok の実行だけから計算する。実行モードが異なる行は
    別グループとして集計する（CPU のみの結果と GPU 実行の結果を混ぜない）。
    """

    groups: dict[tuple[Any, Any, Any, Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("model"),
            row.get("category"),
            row.get("image_size"),
            row.get("cpu_threads"),
            row.get("execution_mode"),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (
        model,
        category,
        image_size,
        cpu_threads,
        execution_mode,
    ), group in groups.items():
        ok_rows = [row for row in group if row.get("status") == "ok"]
        item: dict[str, Any] = {
            "model": model,
            "category": category,
            "image_size": image_size,
            "cpu_threads": cpu_threads,
            "execution_mode": execution_mode,
            "n_runs": len(group),
            "n_ok": len(ok_rows),
            "n_failed": len(group) - len(ok_rows),
        }

        manifest_values = {
            row.get("manifest_sha256")
            for row in ok_rows
            if isinstance(row.get("manifest_sha256"), str)
            and row.get("manifest_sha256")
        }
        item["manifest_consistent"] = len(manifest_values) <= 1

        for column in NUMERIC_COLUMNS:
            values = [
                float(row[column])
                for row in ok_rows
                if _is_finite_number(row.get(column))
            ]
            if values:
                item[f"{column}_mean"] = sum(values) / len(values)
                item[f"{column}_std"] = (
                    statistics.stdev(values) if len(values) >= 2 else None
                )
                item[f"{column}_min"] = min(values)
                item[f"{column}_max"] = max(values)
            else:
                item[f"{column}_mean"] = None
                item[f"{column}_std"] = None
                item[f"{column}_min"] = None
                item[f"{column}_max"] = None

        summary.append(item)

    summary.sort(key=_summary_sort_key)
    return summary


def runs_csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """runs テーブルを CSV 文字列にする。"""

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(RUN_COLUMNS)
    for row in rows:
        writer.writerow([_csv_cell(row.get(column)) for column in RUN_COLUMNS])
    return buffer.getvalue()


def summary_csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """summary テーブルを CSV 文字列にする。"""

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SUMMARY_COLUMNS)
    for row in rows:
        writer.writerow([_csv_cell(row.get(column)) for column in SUMMARY_COLUMNS])
    return buffer.getvalue()


def write_runs_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """runs テーブルを UTF-8 CSV として書き出す。"""

    Path(path).write_text(runs_csv_text(rows), encoding="utf-8")


def write_summary_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """summary テーブルを UTF-8 CSV として書き出す。"""

    Path(path).write_text(summary_csv_text(rows), encoding="utf-8")


def write_report_md(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    matrix_meta: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> None:
    """Markdown 比較レポートを書き出す。"""

    Path(path).write_text(
        render_report_md(rows, summary_rows, matrix_meta, environment),
        encoding="utf-8",
    )


def render_report_md(
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    matrix_meta: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> str:
    """Markdown 比較レポートの文字列を返す。"""

    meta: Mapping[str, Any] = matrix_meta if isinstance(matrix_meta, Mapping) else {}
    env: Mapping[str, Any] = environment if isinstance(environment, Mapping) else {}
    config_value = meta.get("config")
    config: Mapping[str, Any] = (
        config_value if isinstance(config_value, Mapping) else {}
    )

    name = config.get("name") or meta.get("name") or "—"
    protocol = config.get("protocol") or "—"

    lines: list[str] = []
    lines.append("# PaDiM / PatchCore 比較レポート")
    lines.append("")
    lines.append(
        "このレポートの数値は各 run の report.json からのみ取得しています。"
        "欠損は `—` で表し、0 などの仮の値で補いません。"
    )
    lines.append("")

    lines.append("## 1. 実行条件")
    lines.append("")
    lines.append(f"- 設定名: {_md_cell(name)}")
    lines.append(f"- プロトコル: {_md_cell(protocol)}")
    protocol_name = env.get("protocol_name")
    if protocol_name:
        lines.append(f"- プロトコル名: {_md_cell(protocol_name)}")

    versions = env.get("versions")
    host = env.get("host")
    cpu_logical_cores = env.get("cpu_logical_cores")
    if isinstance(versions, Mapping) or isinstance(host, Mapping):
        lines.append("- 環境:")
        if isinstance(versions, Mapping):
            for key in (
                "python",
                "torch",
                "torchvision",
                "anomalib",
                "lightning",
                "timm",
                "numpy",
            ):
                if key in versions:
                    lines.append(f"  - {key}: {_md_cell(versions.get(key))}")
        if isinstance(host, Mapping):
            if "os" in host:
                lines.append(f"  - os: {_md_cell(host.get('os'))}")
            if "machine" in host:
                lines.append(f"  - machine: {_md_cell(host.get('machine'))}")
        if cpu_logical_cores is not None:
            lines.append(f"  - cpu_logical_cores: {_md_cell(cpu_logical_cores)}")
    else:
        lines.append("- 環境: —（成功した report がありません）")
    lines.append("")

    lines.append("## 2. カテゴリ別比較")
    lines.append("")
    lines.append(
        "実行モードが異なる行は同一表で比較しない。"
        "`cpu_only` 以外の結果は CPU のみの実用性の数値と混同しない。"
    )
    lines.append("")

    raw_categories: list[Any] = []
    raw_models: list[Any] = []
    config_categories = config.get("categories")
    config_models = config.get("models")
    if isinstance(config_categories, list):
        raw_categories.extend(config_categories)
    if isinstance(config_models, list):
        raw_models.extend(config_models)
    raw_categories.extend(row.get("category") for row in summary_rows)
    raw_models.extend(row.get("model") for row in summary_rows)
    raw_categories.extend(row.get("category") for row in rows)
    raw_models.extend(row.get("model") for row in rows)

    categories = [
        value
        for value in _ordered_unique(raw_categories)
        if isinstance(value, str)
    ]
    models = [
        value for value in _ordered_unique(raw_models) if isinstance(value, str)
    ]

    summary_map: dict[tuple[str, str, int, int, str], Mapping[str, Any]] = {}
    for summary_row in summary_rows:
        model = summary_row.get("model")
        category = summary_row.get("category")
        image_size = _norm_int(summary_row.get("image_size"))
        cpu_threads = _norm_int(summary_row.get("cpu_threads"))
        mode = _execution_mode_label(summary_row.get("execution_mode"))
        if isinstance(model, str) and isinstance(category, str):
            summary_map[(model, category, image_size, cpu_threads, mode)] = (
                summary_row
            )

    wrote_comparison_table = False
    for category in categories:
        modes = _ordered_unique(
            _execution_mode_label(row.get("execution_mode"))
            for row in summary_rows
            if row.get("category") == category
        )
        modes.sort(key=_execution_mode_sort_key)

        wrote_category_header = False
        for mode in modes:
            conditions = sorted(
                {
                    (image_size, cpu_threads)
                    for (
                        _model_name,
                        cat,
                        image_size,
                        cpu_threads,
                        row_mode,
                    ) in summary_map
                    if cat == category and row_mode == mode
                }
            )
            if not conditions:
                continue

            wrote_comparison_table = True
            if not wrote_category_header:
                lines.append(f"### {_md_cell(category)}")
                lines.append("")
                wrote_category_header = True

            lines.append(f"#### 実行モード: {_md_cell(mode)}")
            lines.append("")
            lines.append(
                "| 条件 | モデル | n_ok/n_runs | Image AUROC engine | "
                "Image AUROC raw | Pixel AUROC engine | Pixel AUROC raw | "
                "学習時間 s | e2e P50 ms | e2e P95 ms | ピーク RSS MiB |"
            )
            lines.append(
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            )
            for image_size, cpu_threads in conditions:
                for model in models:
                    summary_row = summary_map.get(
                        (model, category, image_size, cpu_threads, mode)
                    )
                    if summary_row is None:
                        # 実行記録が無い組合せは表に出さない（0/0 行は情報を持たない）。
                        continue
                    n_runs = summary_row.get("n_runs")
                    n_ok = summary_row.get("n_ok")
                    cells = [
                        f"s{image_size} / t{cpu_threads}",
                        model,
                        f"{n_ok}/{n_runs}",
                        _mean_std_cell(summary_row, "image_AUROC"),
                        _mean_std_cell(summary_row, "image_AUROC_raw"),
                        _mean_std_cell(summary_row, "pixel_AUROC"),
                        _mean_std_cell(summary_row, "pixel_AUROC_raw"),
                        _mean_std_cell(summary_row, "train_stage_seconds"),
                        _mean_std_cell(summary_row, "e2e_p50_ms"),
                        _mean_std_cell(summary_row, "e2e_p95_ms"),
                        _mean_std_cell(summary_row, "process_peak_rss_mib"),
                    ]
                    lines.append(
                        "| " + " | ".join(_md_cell(cell) for cell in cells) + " |"
                    )
            lines.append("")

    if not wrote_comparison_table:
        lines.append("比較可能な実行がありません。")
        lines.append("")

    lines.append("## 3. 失敗・タイムアウト一覧")
    lines.append("")
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        lines.append("| run_key | status | error |")
        lines.append("|---|---|---|")
        for row in failures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(row.get("run_key")),
                        _md_cell(row.get("status")),
                        _md_cell(row.get("error")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("失敗・タイムアウトはありません。")
    lines.append("")

    lines.append("## 4. 指標の意味")
    lines.append("")
    lines.append(
        "- engine 指標は val 由来の閾値・正規化を経た後の metrics（test_metrics）です。"
    )
    lines.append(
        "- raw 指標は post_processor による正規化を通さない生スコアの順位ベース指標です。"
    )
    lines.append(
        "- 学習時間（train_stage_seconds）は engine.fit 工程全体の所要時間です。"
    )
    lines.append(
        "- レイテンシはバッチ 1 で計測し、ウォームアップ画像を集計から除外しています。"
    )
    lines.append(
        "- RSS は psutil によるサンプリング最大値であり、短いピークを取り逃し得ます。"
    )
    lines.append(
        "- repeat は同一 seed で実行しており、モデル条件の違いではなく計測ばらつきの評価です。"
    )
    lines.append("")
    return "\n".join(lines)


def _finite_number(value: Any) -> tuple[float | None, bool]:
    if value is None or isinstance(value, bool):
        return None, False
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number, False
        return None, True
    return None, False


def _is_finite_number(value: Any) -> bool:
    return _finite_number(value)[0] is not None


def _get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _report_error(report: Mapping[str, Any] | None) -> str | None:
    if not isinstance(report, Mapping):
        return None
    error_type = report.get("error_type")
    error_message = report.get("error_message")
    if error_type is None and error_message is None:
        return None
    label = str(error_type) if error_type is not None else "Error"
    message = str(error_message) if error_message is not None else ""
    return f"{label}: {message}".strip()


def _run_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("model") or ""),
        str(row.get("category") or ""),
        _norm_int(row.get("image_size")),
        _norm_int(row.get("cpu_threads")),
        _norm_int(row.get("repeat")),
        str(row.get("run_key") or ""),
    )


def _summary_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("model") or ""),
        str(row.get("category") or ""),
        _norm_int(row.get("image_size")),
        _norm_int(row.get("cpu_threads")),
        _execution_mode_sort_key(
            _execution_mode_label(row.get("execution_mode"))
        ),
    )


def _execution_mode_label(value: Any) -> str:
    """summary/report の execution_mode を表示用ラベルにする。"""
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _execution_mode_sort_key(mode: str) -> tuple[int, str]:
    """実行モードの表示順。定義順を優先し、未知の値は末尾へ回す。"""
    try:
        return (EXECUTION_MODES.index(mode), "")
    except ValueError:
        return (len(EXECUTION_MODES), mode)


def _norm_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return -1


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    return str(value)


def _digits_for(column: str) -> int:
    if "AUROC" in column or "F1" in column:
        return 4
    if "rss_mib" in column:
        return 1
    return 2


def _format_number(value: Any, digits: int) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _mean_std_cell(
    summary_row: Mapping[str, Any] | None, column: str
) -> str:
    if not isinstance(summary_row, Mapping):
        return "—"
    mean = summary_row.get(f"{column}_mean")
    std = summary_row.get(f"{column}_std")
    if mean is None:
        return "—"
    digits = _digits_for(column)
    mean_text = _format_number(mean, digits)
    if std is None:
        return mean_text
    return f"{mean_text} ± {_format_number(std, digits)}"


def _md_cell(value: Any) -> str:
    if value is None:
        return "—"
    if value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
