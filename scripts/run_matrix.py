#!/usr/bin/env python3
"""複数の run_anomalib.py を逐次実行し、runs.jsonl に結果を追記する。

計測干渉を避けるため並列実行はしない。各実行は子プロセスで行い、
stdout/stderr をログファイルに保存する。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adbench.experiment import (  # noqa: E402
    ExperimentConfig,
    RunSpec,
    expand,
    load_config,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="実験マトリクスを逐次実行して runs.jsonl に記録します。"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="実験マトリクス設定 JSON のパス",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="出力ディレクトリ。省略時は results/matrix/<name>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行せずに run_key と argv を表示します",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PARTIAL_RUN_KEY",
        help="run_key の部分一致で実行対象を絞ります。複数指定可。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="runs.jsonl に記録済みの run_key をスキップします",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0,
        help="1 実行あたりのタイムアウト秒（既定 3600）",
    )
    return parser.parse_args(argv)


def recorded_run_keys(lines: Sequence[str]) -> set[str]:
    """runs.jsonl の行から run_key の集合を作る。壊れた行は無視する。"""

    keys: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        run_key = value.get("run_key")
        if isinstance(run_key, str) and run_key:
            keys.add(run_key)
    return keys


def matches_only(run_key: str, only_patterns: Sequence[str]) -> bool:
    """--only の部分一致判定。空なら常に True。"""

    if not only_patterns:
        return True
    return any(pattern in run_key for pattern in only_patterns)


def should_skip_run(
    run_key: str,
    recorded_keys: set[str],
    only_patterns: Sequence[str],
    resume: bool,
) -> bool:
    """--only と --resume を考慮したスキップ判定。"""

    if not matches_only(run_key, only_patterns):
        return True
    return bool(resume and run_key in recorded_keys)


def summarize_error(
    returncode: int | None,
    log_tail: Sequence[str],
    error_type: Any = None,
    error_message: Any = None,
    *,
    timed_out: bool = False,
) -> str:
    """report のエラー、または終了コードとログ末尾からエラー要約を作る。"""

    if error_type is not None or error_message is not None:
        label = str(error_type) if error_type is not None else "Error"
        message = str(error_message) if error_message is not None else ""
        return f"{label}: {message}".strip()

    if timed_out:
        head = "timeout"
    elif returncode is None:
        head = "returncode unknown"
    elif returncode < 0:
        head = f"signal {_signal_name(-returncode)}"
    else:
        head = f"returncode {returncode}"

    tail = " | ".join(
        line.strip() for line in log_tail[-20:] if line.strip()
    )
    if tail:
        return f"{head}; log_tail: {tail}"
    return head


def tail_lines(path: Path, count: int = 20, max_bytes: int = 65536) -> list[str]:
    """テキストファイルの末尾行を読む。巨大ログでもメモリを節約する。"""

    if not path.exists():
        return []
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        start = max(0, size - max_bytes)
        file.seek(start)
        data = file.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    return lines[-count:]


def find_latest_report(run_dir: Path) -> Path | None:
    """runs/<run_key>/*/report.json のうち最新のものを返す。"""

    candidates = [path for path in run_dir.glob("*/report.json") if path.is_file()]
    direct = run_dir / "report.json"
    if direct.is_file():
        candidates.append(direct)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def delete_checkpoints(run_dir: Path) -> tuple[int, int]:
    """run_dir 以下の *.ckpt を削除し、(件数, 削除バイト数) を返す。"""

    count = 0
    total_bytes = 0
    for path in sorted(run_dir.rglob("*.ckpt")):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError as error:
            print(f"チェックポイントを削除できません: {path}: {error}", file=sys.stderr)
            continue
        count += 1
        total_bytes += size
    return count, total_bytes


def execute_spec(
    spec: RunSpec,
    out_dir: Path,
    run_anomalib_path: Path,
    timeout_s: float,
    keep_checkpoints: bool,
) -> dict[str, Any]:
    """1 つの RunSpec を子プロセスとして実行し、runs.jsonl 用の行を返す。"""

    run_dir = out_dir / "runs" / spec.run_key
    log_dir = out_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.run_key}.log"

    command = [
        sys.executable,
        str(run_anomalib_path),
        *spec.argv,
        "--output-dir",
        str(run_dir),
    ]

    timed_out = False
    spawn_error: str | None = None
    returncode: int | None = None
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("command: " + " ".join(shlex.quote(arg) for arg in command))
        log_file.write("\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
        except OSError as error:
            spawn_error = f"{type(error).__name__}: {error}"
            returncode = None
    elapsed_s = time.perf_counter() - started

    report_path = find_latest_report(run_dir)
    report = _load_report(report_path) if report_path is not None else None
    log_tail = tail_lines(log_path, 20)

    if timed_out:
        status = "timeout"
    elif spawn_error is not None:
        status = "failed"
    elif report is not None:
        status = str(report.get("status") or "failed")
    else:
        status = "failed"

    error: str | None = None
    if status != "ok":
        error_type = report.get("error_type") if isinstance(report, Mapping) else None
        error_message = (
            report.get("error_message") if isinstance(report, Mapping) else None
        )
        error = summarize_error(
            returncode,
            log_tail,
            error_type,
            error_message,
            timed_out=timed_out,
        )
        if spawn_error is not None:
            error = spawn_error

    if report_path is not None:
        try:
            report_rel = report_path.relative_to(out_dir).as_posix()
        except ValueError:
            report_rel = str(report_path)
    else:
        report_rel = None

    row: dict[str, Any] = {
        "run_key": spec.run_key,
        "model": spec.model,
        "category": spec.category,
        "image_size": spec.image_size,
        "cpu_threads": spec.cpu_threads,
        "repeat": spec.repeat,
        "returncode": returncode,
        "elapsed_s": round(elapsed_s, 6),
        "report_path": report_rel,
        "status": status,
        "error": error,
    }

    if not keep_checkpoints:
        removed_count, removed_bytes = delete_checkpoints(run_dir)
        _append_matrix_log(
            out_dir,
            f"checkpoint削除: {spec.run_key}: "
            f"{removed_count} files, {removed_bytes} bytes",
        )

    return row


def write_matrix_meta(
    out_dir: Path,
    config_path: Path,
    config: ExperimentConfig,
    started_at: datetime,
    argv: Sequence[str],
) -> dict[str, Any]:
    """matrix_meta.json を書き出し、保存した内容を返す。"""

    meta: dict[str, Any] = {
        "schema_version": 1,
        "name": config.name,
        "config_path": config_path.as_posix(),
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "config": asdict(config),
        "started_at_utc": started_at.isoformat(),
        "git_commit": _git_commit(),
        "python_executable": sys.executable,
        "argv": list(argv),
    }
    path = out_dir / "matrix_meta.json"
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def _load_report(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _append_matrix_log(out_dir: Path, message: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "matrix.log").open("a", encoding="utf-8") as file:
        file.write(message + "\n")
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s は正の値である必要があります")

    config_path = Path(args.config)
    config = load_config(config_path)
    specs = expand(config)

    if args.output_dir is None:
        out_dir = Path("results/matrix") / config.name
    else:
        out_dir = Path(args.output_dir)

    run_anomalib_path = ROOT / "scripts" / "run_anomalib.py"

    if args.dry_run:
        for spec in specs:
            if not matches_only(spec.run_key, args.only):
                continue
            command = [
                sys.executable,
                str(run_anomalib_path),
                *spec.argv,
                "--output-dir",
                str(out_dir / "runs" / spec.run_key),
            ]
            print(
                f"{spec.run_key}: "
                + " ".join(shlex.quote(arg) for arg in command)
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    recorded: set[str] = set()
    if args.resume and runs_path.exists():
        recorded = recorded_run_keys(
            runs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        )

    started_at = datetime.now(timezone.utc)
    write_matrix_meta(
        out_dir,
        config_path,
        config,
        started_at,
        sys.argv[1:] if argv is None else list(argv),
    )
    _append_matrix_log(
        out_dir,
        f"matrix開始: name={config.name} runs={len(specs)} "
        f"started_at_utc={started_at.isoformat()}",
    )

    for spec in specs:
        selected = matches_only(spec.run_key, args.only)
        if should_skip_run(spec.run_key, recorded, args.only, args.resume):
            if selected and args.resume and spec.run_key in recorded:
                _append_matrix_log(out_dir, f"skip（記録済み）: {spec.run_key}")
            continue

        _append_matrix_log(out_dir, f"run開始: {spec.run_key}")
        row = execute_spec(
            spec=spec,
            out_dir=out_dir,
            run_anomalib_path=run_anomalib_path,
            timeout_s=args.timeout_s,
            keep_checkpoints=config.keep_checkpoints,
        )
        with runs_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        _append_matrix_log(
            out_dir,
            f"run終了: {spec.run_key} status={row['status']} "
            f"elapsed_s={row['elapsed_s']}",
        )

    _append_matrix_log(out_dir, "matrix終了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
