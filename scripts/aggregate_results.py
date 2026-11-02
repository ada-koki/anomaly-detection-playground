#!/usr/bin/env python3
"""run_matrix.py の出力から runs.csv / summary.csv / report.md を生成する。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adbench.aggregate import (  # noqa: E402
    build_runs_table,
    summarize_rows,
    write_report_md,
    write_runs_csv,
    write_summary_csv,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="実験マトリクス結果を集計して CSV と Markdown を生成します。"
    )
    parser.add_argument(
        "--matrix-dir",
        required=True,
        help="run_matrix.py の出力ディレクトリ",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="集計結果の出力先。省略時は matrix-dir と同じ",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSON Lines を読み、壊れた行は無視する。"""

    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                entries.append(value)
    return entries


def read_json_object(path: Path) -> dict[str, Any]:
    """JSON オブジェクトを読む。存在しない・壊れている場合は空辞書。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else matrix_dir

    entries = read_jsonl(matrix_dir / "runs.jsonl")
    meta = read_json_object(matrix_dir / "matrix_meta.json")

    report_cache: dict[str, Mapping[str, Any] | None] = {}

    def report_loader(relative_path: str | None) -> Mapping[str, Any] | None:
        if not relative_path:
            return None
        if relative_path in report_cache:
            return report_cache[relative_path]
        path = matrix_dir / relative_path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        report = value if isinstance(value, Mapping) else None
        report_cache[relative_path] = report
        return report

    environment: dict[str, Any] = {}
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        report = report_loader(entry.get("report_path"))
        if not isinstance(report, Mapping) or report.get("status") != "ok":
            continue
        protocol = report.get("protocol")
        protocol_name = (
            protocol.get("name") if isinstance(protocol, Mapping) else None
        )
        environment = {
            "versions": report.get("versions"),
            "host": report.get("host"),
            "cpu_logical_cores": report.get("cpu_logical_cores"),
            "protocol_name": protocol_name,
        }
        break

    rows = build_runs_table(entries, report_loader)
    summary_rows = summarize_rows(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_runs_csv(out_dir / "runs.csv", rows)
    write_summary_csv(out_dir / "summary.csv", summary_rows)
    write_report_md(
        out_dir / "report.md",
        rows,
        summary_rows,
        matrix_meta=meta,
        environment=environment,
    )

    print(
        f"集計を保存しました: {out_dir} "
        f"(runs={len(rows)}, summary={len(summary_rows)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
