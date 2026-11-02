"""プロセスメモリ（RSS）の計測ユーティリティ。"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Iterator

DEFAULT_SAMPLING_INTERVAL_S = 0.05


def _process_rss_bytes(process: Any) -> int:
    """プロセスとその子プロセスのRSS合計を返す。"""
    total = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except Exception:
        processes = [process]
    for child in processes:
        try:
            total += child.memory_info().rss
        except Exception:
            continue
    return total


@contextmanager
def sampled_rss(interval_s: float = DEFAULT_SAMPLING_INTERVAL_S) -> Iterator[dict[str, float]]:
    """プロセスツリーのRSSを一定間隔でサンプリングする。

    得られる peak_rss_mib はサンプリングに基づく推定値であり、
    サンプリング間隔より短いピークは取り逃す可能性がある。
    """
    if interval_s <= 0:
        raise ValueError("interval_s には0より大きい値を指定してください")

    import psutil

    process = psutil.Process(os.getpid())
    stopped = threading.Event()
    summary: dict[str, float] = {"peak_rss_mib": 0.0}

    def poll() -> None:
        while True:
            current = _process_rss_bytes(process) / (1024 * 1024)
            summary["peak_rss_mib"] = max(summary["peak_rss_mib"], current)
            if stopped.wait(interval_s):
                break

    thread = threading.Thread(target=poll, name="rss-sampler", daemon=True)
    thread.start()
    try:
        yield summary
    finally:
        stopped.set()
        thread.join()
        final_rss = _process_rss_bytes(process) / (1024 * 1024)
        summary["peak_rss_mib"] = round(max(summary["peak_rss_mib"], final_rss), 2)


def peak_rss_lifetime_mib() -> float | None:
    """自プロセスの生涯最大RSSをMiBで返す。

    Linux では resource.getrusage(RUSAGE_SELF).ru_maxrss を KiB として扱う。
    macOS ではバイト単位として扱う。取得できない環境では None を返す。
    """
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform.startswith("linux"):
        return round(usage.ru_maxrss / 1024.0, 2)
    if sys.platform == "darwin":
        return round(usage.ru_maxrss / (1024.0 * 1024.0), 2)
    return None
