#!/usr/bin/env python3
"""欠陥サイズ別レポートを生成する CLI。

各 run ディレクトリの ``report.json`` と ``per_image.jsonl`` を読み、
``adbench.defect_size.summarize`` で集計し直して CSV / Markdown（任意で PNG）を
出力する。``--help`` と CSV 生成は torch / anomalib / matplotlib なしで動く。

例::

    python scripts/defect_size_report.py \
        --runs runs/run_global_resize runs/run_tiled_256 \
        --labels global_resize tiled_256 \
        --out-dir reports/defect_size
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.defect_size import (  # noqa: E402  (sys.path 調整の後に import する)
    SIZE_BANDS_DEFAULT,
    load_records_jsonl,
    summarize,
)

CSV_COLUMNS: tuple[str, ...] = (
    "run_label",
    "spatial_strategy",
    "working_resolution",
    "tile_size",
    "stride",
    "model",
    "group",
    "key",
    "n",
    "detected",
    "detection_rate",
    "miss_rate",
    "region_hit_rate",
    "normal_fpr",
    "image_AUROC_raw",
    "train_stage_seconds",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "train_peak_rss_mib",
    "note",
)

_GROUP_LABELS = {
    "overall": "全体",
    "size_band": "サイズ帯",
    "defect_type": "異常種別",
    "defect_type_x_size_band": "異常種別×サイズ帯",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "per_image.jsonl から欠陥サイズ別の検出率・見逃し率・正常画像の過検出率を"
            "再集計し、CSV / Markdown を出力する。"
        ),
        epilog=(
            "閾値は各 run の report.json にある defect_size.thresholds を使う。"
            "存在しない場合は率を空欄にし、注記に「閾値未計算」を残す。"
        ),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="DIR",
        help="run ディレクトリ（report.json と per_image.jsonl を含む）。複数指定可。",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        metavar="NAME",
        help="run の表示名。件数は --runs と一致させる。",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIR",
        help="出力先ディレクトリ（存在しなければ作成する）。",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON のトップレベルがオブジェクトではありません: {path}")
    return data


def _nested(mapping: Any, *keys: str, default: Any = None) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def load_run(run_dir: Path, label: str | None = None) -> dict[str, Any]:
    """run ディレクトリを読み、レコードと集計結果を返す。

    Raises:
        FileNotFoundError: per_image.jsonl が見つからない場合。
        ValueError: JSON を解析できない場合。
    """
    records_path = run_dir / "per_image.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"per_image.jsonl が見つかりません: {records_path}")

    report_path = run_dir / "report.json"
    if report_path.is_file():
        report = _read_json(report_path)
    else:
        report = {}
        sys.stderr.write(
            f"警告: report.json が見つかりません（閾値未計算として扱います）: {report_path}\n"
        )

    records = load_records_jsonl(records_path)
    defect_size_report = report.get("defect_size")
    if not isinstance(defect_size_report, Mapping):
        defect_size_report = {}
    thresholds = defect_size_report.get("thresholds")
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    min_samples_value = defect_size_report.get("min_samples", 5)
    try:
        min_samples = int(min_samples_value)
    except (TypeError, ValueError):
        min_samples = 5

    summary = summarize(
        records,
        _maybe_float(thresholds.get("image")),
        _maybe_float(thresholds.get("pixel")),
        min_samples=min_samples,
    )
    return {
        "run_dir": str(run_dir),
        "label": label or run_dir.name,
        "report": report,
        "records": records,
        "summary": summary,
    }


def _run_meta(report: Mapping[str, Any]) -> dict[str, Any]:
    spatial = report.get("spatial_strategy")
    if not isinstance(spatial, Mapping):
        spatial = {}
    latency = _nested(report, "per_image_inference_latency", "end_to_end_ms", default={})
    if not isinstance(latency, Mapping):
        latency = {}
    return {
        "model": _format_value(report.get("model")),
        "spatial_strategy": _format_value(spatial.get("spatial_strategy")),
        "working_resolution": _format_value(spatial.get("working_resolution")),
        "tile_size": _format_value(spatial.get("tile_size")),
        "stride": _format_value(spatial.get("stride")),
        "train_stage_seconds": _maybe_float(report.get("train_stage_seconds")),
        "train_peak_rss_mib": _maybe_float(
            _nested(report, "memory", "train_peak_rss_mib_sampled")
        ),
        "e2e_p50_ms": _maybe_float(latency.get("p50")),
        "e2e_p95_ms": _maybe_float(latency.get("p95")),
        "image_AUROC_raw": _maybe_float(
            _nested(report, "raw_score_metrics", "image_AUROC_raw")
        ),
    }


def iter_cells(summary: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    """集計結果を ``(group, key, cell)`` の並びに平坦化する。"""
    cells: list[tuple[str, str, Mapping[str, Any]]] = []
    overall = summary.get("overall")
    if isinstance(overall, Mapping):
        cells.append(("overall", "overall", overall))
    by_size_band = summary.get("by_size_band")
    if isinstance(by_size_band, Mapping):
        for band, cell in by_size_band.items():
            if isinstance(cell, Mapping):
                cells.append(("size_band", str(band), cell))
    by_defect_type = summary.get("by_defect_type")
    if isinstance(by_defect_type, Mapping):
        for defect, cell in by_defect_type.items():
            if isinstance(cell, Mapping):
                cells.append(("defect_type", str(defect), cell))
    nested = summary.get("by_defect_type_and_size_band")
    if isinstance(nested, Mapping):
        for defect, bands in nested.items():
            if not isinstance(bands, Mapping):
                continue
            for band, cell in bands.items():
                if isinstance(cell, Mapping):
                    cells.append(
                        ("defect_type_x_size_band", f"{defect}/{band}", cell)
                    )
    return cells


def build_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """1 run 分の CSV 行を作る。"""
    summary = run.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    meta = _run_meta(run.get("report") or {})
    normal = summary.get("normal")
    if not isinstance(normal, Mapping):
        normal = {}

    base: dict[str, Any] = {
        "run_label": run.get("label", ""),
        "spatial_strategy": meta["spatial_strategy"],
        "working_resolution": meta["working_resolution"],
        "tile_size": meta["tile_size"],
        "stride": meta["stride"],
        "model": meta["model"],
        "normal_fpr": normal.get("false_positive_rate"),
        "image_AUROC_raw": meta["image_AUROC_raw"],
        "train_stage_seconds": meta["train_stage_seconds"],
        "e2e_p50_ms": meta["e2e_p50_ms"],
        "e2e_p95_ms": meta["e2e_p95_ms"],
        "train_peak_rss_mib": meta["train_peak_rss_mib"],
    }
    rows: list[dict[str, Any]] = []
    for group, key, cell in iter_cells(summary):
        row = dict(base)
        row.update(
            {
                "group": group,
                "key": key,
                "n": cell.get("n"),
                "detected": cell.get("detected"),
                "detection_rate": cell.get("detection_rate"),
                "miss_rate": cell.get("miss_rate"),
                "region_hit_rate": cell.get("region_hit_rate"),
                "note": cell.get("note", ""),
            }
        )
        rows.append(row)
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """CSV を書き出す（None は空欄）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in CSV_COLUMNS})


def _md_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:.4f}"
    text = str(value)
    return text if text else "-"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_value(value) for value in row) + " |")
    return lines


def _band_names(summary: Mapping[str, Any]) -> list[str]:
    bands = summary.get("bands")
    if isinstance(bands, Sequence) and not isinstance(bands, (str, bytes)):
        names = [str(band.get("name")) for band in bands if isinstance(band, Mapping)]
        if names:
            return names
    return [name for name, _lower, _upper in SIZE_BANDS_DEFAULT]


def write_png(runs: Sequence[Mapping[str, Any]], path: Path) -> tuple[bool, str]:
    """サイズ帯別検出率の棒グラフを書き出す。

    Returns:
        ``(書き出したか, 注記)``。matplotlib が無い場合は ``(False, 注記)``。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False, (
            "matplotlib が無いため detection_by_size.png をスキップしました"
            "（CSV / Markdown は出力済み）。"
        )

    if not runs:
        return False, "run が無いため detection_by_size.png をスキップしました。"

    band_names = _band_names(runs[0].get("summary") or {})
    positions = np.arange(len(band_names), dtype=float)
    run_count = max(1, len(runs))
    width = 0.8 / run_count

    figure, axis = plt.subplots(
        figsize=(max(6.0, 1.2 * len(band_names) + 3.0), 4.0)
    )
    for index, run in enumerate(runs):
        summary = run.get("summary") or {}
        cells = summary.get("by_size_band") or {}
        values: list[float] = []
        for name in band_names:
            cell = cells.get(name) if isinstance(cells, Mapping) else None
            rate = cell.get("detection_rate") if isinstance(cell, Mapping) else None
            values.append(float("nan") if rate is None else float(rate))

        meta = _run_meta(run.get("report") or {})
        p50 = meta["e2e_p50_ms"]
        rss = meta["train_peak_rss_mib"]
        p50_text = "n/a" if p50 is None else f"{p50:.1f}"
        rss_text = "n/a" if rss is None else f"{rss:.0f}"
        legend = (
            f"{run.get('label', '')} "
            f"(e2e P50 {p50_text} ms, train peak RSS {rss_text} MiB)"
        )
        offset = (index - (run_count - 1) / 2.0) * width
        axis.bar(positions + offset, values, width=width, label=legend)

    # 日本語フォントの有無に依存しないよう、軸ラベルは英語にする。
    axis.set_xticks(positions)
    axis.set_xticklabels(band_names)
    axis.set_xlabel("Defect size band (mask area ratio, original resolution)")
    axis.set_ylabel("Image-level detection rate (abnormal images only)")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True, ""


def render_markdown(
    runs: Sequence[Mapping[str, Any]],
    *,
    png_note: str | None = None,
) -> str:
    """Markdown レポートを組み立てる（日本語）。"""
    lines: list[str] = ["# 欠陥サイズ別レポート", ""]
    lines.append(
        "`scripts/defect_size_report.py` の生成物。閾値は各 run の `report.json` に"
        "ある `defect_size.thresholds`（検証集合由来）を使い、テスト集合では調整していない。"
    )
    lines.append("")

    lines.extend(["## 定義", ""])
    lines.append(
        "- 欠陥サイズ: 元画像解像度のマスク面積率（マスク画素数 / 元画像画素数）。"
        "外接矩形は `[y0, x0, y1, x1]`（`y1` / `x1` は排他的）。"
    )
    lines.append(
        "- サイズ帯: "
        + " / ".join(
            f"{name} [{lower:g}, {upper:g})"
            for name, lower, upper in SIZE_BANDS_DEFAULT
        )
        + "。下限を含み上限を含まない（最後の帯のみ上限を含む）。"
    )
    lines.append(
        "- 閾値: 検証集合から求めた `image_threshold`（画像スコア）と "
        "`pixel_threshold`（anomaly_map）。PostProcessor F1AdaptiveThreshold の生スケール値。"
    )
    lines.append(
        "- 検出率・見逃し率の分母: 異常画像（label=1）のみ。"
        "正常画像は「正常画像の過検出」表で別に扱う。"
    )
    lines.append("- 率が `-` の場合は、分母が 0 または閾値が未計算であることを示す。")
    lines.append("")

    lines.extend(["## 実行条件", ""])
    headers = [
        "run",
        "model",
        "空間方式",
        "作業解像度",
        "tile",
        "stride",
        "train stage [s]",
        "e2e P50 [ms]",
        "e2e P95 [ms]",
        "train peak RSS [MiB]",
        "image AUROC (raw)",
    ]
    body: list[list[Any]] = []
    for run in runs:
        meta = _run_meta(run.get("report") or {})
        body.append(
            [
                run.get("label", ""),
                meta["model"],
                meta["spatial_strategy"],
                meta["working_resolution"],
                meta["tile_size"],
                meta["stride"],
                meta["train_stage_seconds"],
                meta["e2e_p50_ms"],
                meta["e2e_p95_ms"],
                meta["train_peak_rss_mib"],
                meta["image_AUROC_raw"],
            ]
        )
    lines.extend(_md_table(headers, body))
    lines.append("")

    lines.extend(["## 閾値", ""])
    body = []
    for run in runs:
        summary = run.get("summary") or {}
        thresholds = summary.get("thresholds") or {}
        body.append(
            [
                run.get("label", ""),
                thresholds.get("image"),
                thresholds.get("pixel"),
                thresholds.get("source", ""),
            ]
        )
    lines.extend(_md_table(["run", "image_threshold", "pixel_threshold", "出所"], body))
    lines.append("")

    lines.extend(["## 正常画像の過検出", ""])
    body = []
    for run in runs:
        summary = run.get("summary") or {}
        normal = summary.get("normal") or {}
        body.append(
            [
                run.get("label", ""),
                normal.get("n_normal"),
                normal.get("false_positive"),
                normal.get("false_positive_rate"),
                normal.get("mean_normal_fp_pixel_ratio"),
                normal.get("note", ""),
            ]
        )
    lines.extend(
        _md_table(
            [
                "run",
                "正常 n",
                "FP 件数",
                "FPR",
                "正常画素の平均過検出率",
                "注記",
            ],
            body,
        )
    )
    lines.append("")

    cell_headers = [
        "run",
        "n",
        "検出",
        "検出率",
        "見逃し率",
        "領域ヒット率",
        "マスク無し",
        "注記",
    ]

    lines.extend(["## サイズ帯別（異常画像のみ）", ""])
    body = []
    for run in runs:
        summary = run.get("summary") or {}
        cells = summary.get("by_size_band") or {}
        for band in _band_names(summary):
            cell = cells.get(band) or {}
            body.append(
                [
                    run.get("label", ""),
                    band,
                    cell.get("n"),
                    cell.get("detected"),
                    cell.get("detection_rate"),
                    cell.get("miss_rate"),
                    cell.get("region_hit_rate"),
                    cell.get("n_without_mask"),
                    cell.get("note", ""),
                ]
            )
    lines.extend(
        _md_table(["run", "サイズ帯"] + cell_headers[1:], body)
    )
    lines.append("")

    lines.extend(["## 異常種別別（異常画像のみ）", ""])
    body = []
    for run in runs:
        summary = run.get("summary") or {}
        cells = summary.get("by_defect_type") or {}
        for defect, cell in cells.items():
            if not isinstance(cell, Mapping):
                continue
            body.append(
                [
                    run.get("label", ""),
                    defect,
                    cell.get("n"),
                    cell.get("detected"),
                    cell.get("detection_rate"),
                    cell.get("miss_rate"),
                    cell.get("region_hit_rate"),
                    cell.get("n_without_mask"),
                    cell.get("note", ""),
                ]
            )
    lines.extend(_md_table(["run", "異常種別"] + cell_headers[1:], body))
    lines.append("")

    lines.extend(["## 異常種別×サイズ帯（異常画像のみ）", ""])
    body = []
    for run in runs:
        summary = run.get("summary") or {}
        nested = summary.get("by_defect_type_and_size_band") or {}
        band_names = _band_names(summary)
        for defect, bands in nested.items():
            if not isinstance(bands, Mapping):
                continue
            for band in band_names:
                cell = bands.get(band) or {}
                body.append(
                    [
                        run.get("label", ""),
                        defect,
                        band,
                        cell.get("n"),
                        cell.get("detected"),
                        cell.get("detection_rate"),
                        cell.get("miss_rate"),
                        cell.get("region_hit_rate"),
                        cell.get("n_without_mask"),
                        cell.get("note", ""),
                    ]
                )
    lines.extend(_md_table(["run", "異常種別", "サイズ帯"] + cell_headers[1:], body))
    lines.append("")

    lines.extend(["## 推論時間・メモリとのトレードオフ", ""])
    body = []
    for run in runs:
        summary = run.get("summary") or {}
        meta = _run_meta(run.get("report") or {})
        overall = summary.get("overall") or {}
        cells = summary.get("by_size_band") or {}
        row: list[Any] = [
            run.get("label", ""),
            meta["e2e_p50_ms"],
            meta["e2e_p95_ms"],
            meta["train_peak_rss_mib"],
            overall.get("detection_rate"),
        ]
        for band in _band_names(summary):
            cell = cells.get(band) or {}
            row.append(cell.get("detection_rate"))
        body.append(row)
    headers = ["run", "e2e P50 [ms]", "e2e P95 [ms]", "train peak RSS [MiB]", "overall 検出率"]
    headers.extend(_band_names(runs[0].get("summary") or {}) if runs else [])
    lines.extend(_md_table(headers, body))
    lines.append("")

    lines.extend(["## 注記", ""])
    for run in runs:
        summary = run.get("summary") or {}
        note = summary.get("notes")
        if note:
            lines.append(f"- {run.get('label', '')}: {note}")
    if png_note:
        lines.append(f"- {png_note}")
    lines.append(
        "- 推論時間・メモリの数値は各 run の report.json に記録された値をそのまま転記している。"
    )
    lines.append(
        "- 検出率の差が統計的に有意かどうかは評価していない（未検証）。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_dirs = [Path(path) for path in args.runs]
    labels = list(args.labels) if args.labels else []
    if labels and len(labels) != len(run_dirs):
        parser.error(
            f"--labels の件数（{len(labels)}）が --runs の件数（{len(run_dirs)}）と一致しません"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs):
        label = labels[index] if labels else run_dir.name
        runs.append(load_run(run_dir, label))

    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(build_rows(run))

    csv_path = out_dir / "defect_size_summary.csv"
    write_csv(rows, csv_path)

    png_path = out_dir / "detection_by_size.png"
    png_written, png_note = write_png(runs, png_path)

    md_path = out_dir / "defect_size_summary.md"
    md_path.write_text(
        render_markdown(runs, png_note=None if png_written else png_note),
        encoding="utf-8",
    )

    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")
    if png_written:
        print(f"PNG: {png_path}")
    else:
        print(png_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
