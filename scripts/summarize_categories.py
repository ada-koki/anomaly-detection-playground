"""カテゴリ単位の予算判定とデータセット要約を生成する CLI。

入力は 1 行 1 JSON の runs.jsonl。各行は少なくとも
``condition``, ``dataset``, ``category``, ``model``, ``report_path``,
``failure_status``, ``download_mib`` を持つ。``report_path`` は
runs.jsonl からの相対パスまたは絶対パスで、``null`` も許容する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.category_budget import (  # noqa: E402
    RUN_COLUMNS,
    SUMMARY_COLUMNS,
    judge_budget,
    load_budget,
    render_markdown,
    run_row_from_report,
    summarize_by_dataset,
    write_csv,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON の解析に失敗しました: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_no}: 各行は JSON オブジェクトである必要があります")
            entries.append(data)
    return entries


def _resolve_report_path(raw: Any, base: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _read_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"レポート {path} の JSON 解析に失敗しました: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"レポート {path} の最上位は JSON オブジェクトである必要があります")
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="カテゴリ単位の予算判定とデータセット要約を生成します。",
    )
    parser.add_argument("--runs-jsonl", required=True, help="1 行 1 JSON の runs ファイル")
    parser.add_argument("--budget", required=True, help="予算定義の JSON ファイル")
    parser.add_argument("--out-dir", required=True, help="出力ディレクトリ")
    args = parser.parse_args(argv)

    runs_path = Path(args.runs_jsonl)
    jsonl_dir = runs_path.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = load_budget(args.budget)
    entries = _load_jsonl(runs_path)

    run_rows: list[dict[str, Any]] = []
    for entry in entries:
        condition = str(entry.get("condition") or "")
        dataset = str(entry.get("dataset") or "")
        category = str(entry.get("category") or "")
        model = str(entry.get("model") or "")
        failure_status = entry.get("failure_status")
        download_mib = entry.get("download_mib")

        report_path = _resolve_report_path(entry.get("report_path"), jsonl_dir)
        report = _read_report(report_path) if report_path is not None else None

        if report is None:
            fallback = failure_status if failure_status is not None else "missing_report"
            row = run_row_from_report(
                None,
                condition=condition,
                dataset=dataset,
                category=category,
                model=model,
                failure_status=fallback,
            )
        else:
            row = run_row_from_report(
                report,
                condition=condition,
                dataset=dataset,
                category=category,
                model=model,
                failure_status=failure_status,
            )

        judged = judge_budget(row, budget, download_mib=download_mib)
        row["within_budget"] = judged["within_budget"]
        row["budget_reasons"] = ";".join(judged["reasons"])
        run_rows.append(row)

    summary_rows = summarize_by_dataset(run_rows)

    runs_csv = out_dir / "category_runs.csv"
    summary_csv = out_dir / "dataset_summary.csv"
    report_md = out_dir / "category_report.md"

    run_columns = list(RUN_COLUMNS) + ["within_budget", "budget_reasons"]
    write_csv(runs_csv, run_rows, run_columns)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    report_md.write_text(render_markdown(run_rows, summary_rows, budget), encoding="utf-8")

    print(f"カテゴリ runs CSV: {runs_csv}")
    print(f"データセット要約 CSV: {summary_csv}")
    print(f"レポート Markdown: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
