"""プロセスメモリ（RSS）の計測ユーティリティ。"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
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


def _parse_vmhwm_kib(status_text: str) -> int | None:
    """/proc/<pid>/status のテキストから VmHWM（KiB）を取り出す。"""
    for line in status_text.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def peak_rss_vmhwm_mib() -> float | None:
    """自プロセスのアドレス空間のピーク RSS（VmHWM）を MiB で返す。

    Linux の ``getrusage(RUSAGE_SELF).ru_maxrss`` は fork 時の親プロセスの値を
    引き継ぐため、subprocess で起動したワーカーでは親の RSS が下限として混入する。
    ``/proc/self/status`` の VmHWM は exec 後のアドレス空間ごとに管理されるので、
    ワーカー自身のピークを測るにはこちらを使う。Linux 以外や取得失敗時は None。
    """
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    kib = _parse_vmhwm_kib(text)
    if kib is None:
        return None
    return round(kib / 1024.0, 2)
