#!/usr/bin/env python3
"""干渉の再現用に、指定した数のプロセスで CPU を一定時間占有する（Issue #31）。

レイテンシ計測の「干渉あり」条件を意図的に作るための補助スクリプト。
計測対象とは別プロセスとして起動する（例: ``python scripts/cpu_load.py --procs 4 --seconds 120 &``）。
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from collections.abc import Sequence


def _spin(seconds: float) -> None:
    """``seconds`` 秒間ビジーループする。"""
    deadline = time.monotonic() + seconds
    value = 0
    while time.monotonic() < deadline:
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU 負荷を意図的に発生させます（干渉の再現用）。")
    parser.add_argument("--procs", type=int, default=4, help="ビジーループするプロセス数")
    parser.add_argument("--seconds", type=float, default=60.0, help="負荷を掛ける秒数")
    args = parser.parse_args(argv)
    if args.procs < 1:
        parser.error("--procs には1以上を指定してください")
    if args.seconds <= 0:
        parser.error("--seconds には0より大きい値を指定してください")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workers = [
        multiprocessing.Process(target=_spin, args=(args.seconds,), daemon=True)
        for _ in range(args.procs)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
