"""Issue #15: compare_engines.py の engine_report.json を CSV / Markdown に集計する。

絶対パスや重み・ONNX ファイルは出力しない（パスは run_id とラベルのみ）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

COLUMNS = [
    "label",
    "model",
    "engine",
    "threads",
    "repeats",
    "latency_samples",
    "model_ms_p50",
    "model_ms_p95",
    "model_ms_mean",
    "model_ms_std",
    "load_seconds",
    "latency_peak_rss_mib_sampled",
    "process_peak_rss_mib",
    "process_peak_rss_source",
    "image_AUROC_raw",
    "pixel_AUROC_raw",
    "decision_agreement",
    "score_max_abs_diff",
    "map_max_abs_diff",
    "map_p999_rel_diff",
    "export_method",
    "opset",
    "model_size_bytes",
    "export_seconds",
    "run_id",
]


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def rows_from_report(report: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """1 つの engine_report.json からエンジンごとの行を作る。"""
    rows: list[dict[str, Any]] = []
    ref = report.get("reference") or {}
    for engine, res in (report.get("engines") or {}).items():
        if res.get("status") != "ok":
            rows.append({"label": label, "model": report.get("model"), "engine": engine,
                         "run_id": report.get("run_id")})
            continue
        ms = res.get("model_ms") or {}
        exp = (report.get("export") or {}).get(engine) or {}
        agr = (report.get("agreement") or {}).get(engine) or {}
        diffs = agr.get("diffs") or {}
        if engine == "torch":
            auroc_img = ref.get("image_AUROC_raw")
            auroc_px = ref.get("pixel_AUROC_raw")
            size = None
        else:
            auroc_img = agr.get("image_AUROC_raw")
            auroc_px = agr.get("pixel_AUROC_raw")
            size = exp.get("ir_size_bytes") if engine == "openvino" else exp.get("file_size_bytes")
        rows.append(
            {
                "label": label,
                "model": report.get("model"),
                "engine": engine,
                "threads": report.get("threads"),
                "repeats": res.get("repeats"),
                "latency_samples": res.get("latency_samples"),
                "model_ms_p50": _round(ms.get("p50"), 3),
                "model_ms_p95": _round(ms.get("p95"), 3),
                "model_ms_mean": _round(ms.get("mean"), 3),
                "model_ms_std": _round(ms.get("std"), 3),
                "load_seconds": res.get("load_seconds"),
                "latency_peak_rss_mib_sampled": res.get("latency_peak_rss_mib_sampled"),
                "process_peak_rss_mib": res.get("process_peak_rss_mib"),
                "process_peak_rss_source": res.get("process_peak_rss_source", "getrusage"),
                "image_AUROC_raw": auroc_img,
                "pixel_AUROC_raw": auroc_px,
                "decision_agreement": agr.get("decision_agreement") if engine != "torch" else 1.0,
                "score_max_abs_diff": diffs.get("score_max_abs_diff"),
                "map_max_abs_diff": diffs.get("map_max_abs_diff"),
                "map_p999_rel_diff": diffs.get("map_p999_rel_diff"),
                "export_method": exp.get("method"),
                "opset": exp.get("opset"),
                "model_size_bytes": size,
                "export_seconds": exp.get("seconds"),
                "run_id": report.get("run_id"),
            }
        )
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    cols = ["label", "engine", "threads", "latency_samples", "model_ms_p50", "model_ms_p95",
            "load_seconds", "process_peak_rss_mib", "image_AUROC_raw", "pixel_AUROC_raw",
            "decision_agreement", "map_max_abs_diff"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", nargs="+", type=Path, required=True,
                   help="engine_report.json または それを含む run ディレクトリ")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if len(a.reports) != len(a.labels):
        p.error("--reports と --labels の数を揃えてください")
    rows: list[dict[str, Any]] = []
    for path, label in zip(a.reports, a.labels):
        f = path / "engine_report.json" if path.is_dir() else path
        rows.extend(rows_from_report(json.loads(f.read_text(encoding="utf-8")), label))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "engines_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (a.out_dir / "engines_summary.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"{len(rows)} 行を書き出しました: {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
