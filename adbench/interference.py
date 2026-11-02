"""レイテンシ計測中の干渉（同一ホストの別プロセス）を判定するユーティリティ。

標準ライブラリのみで動作する。psutil は関数内で遅延 import する。
``other_busy_cores`` はホスト全体の busy コア数から自プロセス木の CPU 使用分を
引いた推定値であり、カーネル処理や自プロセスの未計上分を含みうる。
PSI (``/proc/pressure/cpu``) は自プロセス由来と区別できないため既定では
判定に使わず、記録のみ行う。閾値は暫定値であり実測に応じて見直すこと。
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

_SUMMARY_METRICS = (
    "other_busy_cores",
    "host_busy_cores",
    "self_busy_cores",
    "psi_cpu_some_ratio",
)

_THRESHOLD_KEYS = {
    "max_other_busy_cores_mean",
    "max_other_busy_cores_p95",
    "max_psi_cpu_some_ratio_mean",
    "sample_interval_s",
    "is_provisional",
}

# (InterferenceThresholds の属性名, summary のキー, reasons 用ラベル)
_THRESHOLD_TO_METRIC = (
    ("max_other_busy_cores_mean", "other_busy_cores_mean", "other_busy_cores_mean"),
    ("max_other_busy_cores_p95", "other_busy_cores_p95", "other_busy_cores_p95"),
    (
        "max_psi_cpu_some_ratio_mean",
        "psi_cpu_some_ratio_mean",
        "psi_cpu_some_ratio_mean",
    ),
)

_NOTE_OTHER_BUSY = (
    "other_busy_cores はホスト全体の busy コア数から自プロセス木の CPU 使用分を"
    "引いた推定値。カーネル処理・自プロセスの未計上分を含みうる。"
    "PSI は自プロセス由来と区別できない。"
)


def parse_proc_stat_cpu(text: str) -> tuple[int, int] | None:
    """``/proc/stat`` の先頭 "cpu " 行から (total_jiffies, idle_jiffies) を返す。

    total は user, nice, system, idle, iowait, irq, softirq, steal の 8 項目合計
    （guest は user に含まれるため足さない）。idle は idle + iowait。
    集約行が見つからない、または値が不正なら None。
    """
    for raw_line in text.splitlines():
        parts = raw_line.split()
        if not parts or parts[0] != "cpu":
            continue
        values = parts[1:]
        if len(values) < 8:
            return None
        try:
            numbers = [int(value) for value in values[:8]]
        except ValueError:
            return None
        user, nice, system, idle, iowait, irq, softirq, steal = numbers
        total = user + nice + system + idle + iowait + irq + softirq + steal
        idle_jiffies = idle + iowait
        return total, idle_jiffies
    return None


def parse_psi_some_total_us(text: str) -> int | None:
    """/proc/pressure/cpu の "some" 行の total=（マイクロ秒）を返す。"""
    for raw_line in text.splitlines():
        parts = raw_line.split()
        if not parts or parts[0] != "some":
            continue
        for token in parts[1:]:
            if token.startswith("total="):
                try:
                    return int(token[len("total="):])
                except ValueError:
                    return None
    return None


def parse_loadavg(text: str) -> tuple[float, float, float] | None:
    """/proc/loadavg から (1分, 5分, 15分) の load average を返す。"""
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


@dataclass(frozen=True)
class CpuSample:
    """1 時点の CPU 使用状況サンプル。取得できなかった項目は None。"""

    monotonic_s: float
    host_total_jiffies: int | None
    host_idle_jiffies: int | None
    self_cpu_s: float | None
    psi_some_total_us: int | None
    loadavg_1m: float | None


def window_metrics(
    prev: CpuSample, cur: CpuSample, *, clk_tck: int
) -> dict[str, float | None]:
    """2 サンプル間の window 指標を計算する。

    elapsed_s <= 0 の場合はすべて None を返す。
    """
    result: dict[str, float | None] = {
        "elapsed_s": None,
        "host_busy_cores": None,
        "self_busy_cores": None,
        "other_busy_cores": None,
        "psi_cpu_some_ratio": None,
    }

    elapsed_s = cur.monotonic_s - prev.monotonic_s
    if elapsed_s <= 0:
        return result
    result["elapsed_s"] = elapsed_s

    host_busy_cores: float | None = None
    if (
        prev.host_total_jiffies is not None
        and cur.host_total_jiffies is not None
        and prev.host_idle_jiffies is not None
        and cur.host_idle_jiffies is not None
        and clk_tck > 0
    ):
        delta_total = cur.host_total_jiffies - prev.host_total_jiffies
        delta_idle = cur.host_idle_jiffies - prev.host_idle_jiffies
        host_busy_cores = (delta_total - delta_idle) / clk_tck / elapsed_s
    result["host_busy_cores"] = host_busy_cores

    self_busy_cores: float | None = None
    if prev.self_cpu_s is not None and cur.self_cpu_s is not None:
        self_busy_cores = (cur.self_cpu_s - prev.self_cpu_s) / elapsed_s
    result["self_busy_cores"] = self_busy_cores

    if host_busy_cores is not None and self_busy_cores is not None:
        result["other_busy_cores"] = max(0.0, host_busy_cores - self_busy_cores)

    psi_ratio: float | None = None
    if (
        prev.psi_some_total_us is not None
        and cur.psi_some_total_us is not None
    ):
        denom = elapsed_s * 1e6
        if denom > 0:
            psi_ratio = (cur.psi_some_total_us - prev.psi_some_total_us) / denom
    result["psi_cpu_some_ratio"] = psi_ratio

    return result


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    """最近傍順位法によるパーセンタイル。"""
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = int(math.ceil(fraction * n))
    if rank <= 0:
        rank = 1
    if rank > n:
        rank = n
    return ordered[rank - 1]


def summarize_windows(
    windows: Sequence[Mapping[str, float | None]],
    overall: Mapping[str, float | None] | None,
) -> dict[str, Any]:
    """window 指標の要約（mean/max/p95）を返す。None 値は集計から除外する。"""
    result: dict[str, Any] = {"n_windows": len(windows)}

    for metric in _SUMMARY_METRICS:
        values = [
            float(window[metric])
            for window in windows
            if window.get(metric) is not None
        ]
        if values:
            result[f"{metric}_mean"] = sum(values) / len(values)
            result[f"{metric}_max"] = max(values)
            result[f"{metric}_p95"] = _nearest_rank(values, 0.95)
        else:
            result[f"{metric}_mean"] = None
            result[f"{metric}_max"] = None
            result[f"{metric}_p95"] = None

    result["overall"] = dict(overall) if isinstance(overall, Mapping) else None
    return result


@dataclass(frozen=True)
class InterferenceThresholds:
    """干渉判定の閾値。閾値が None の指標は判定に使わない。"""

    max_other_busy_cores_mean: float | None = 1.5
    max_other_busy_cores_p95: float | None = 3.0
    max_psi_cpu_some_ratio_mean: float | None = None
    sample_interval_s: float = 0.5
    is_provisional: bool = True


def _expect_optional_threshold(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} には None または数値を指定してください")
    if value < 0:
        raise ValueError(f"{name} には0以上の値を指定してください")
    return float(value)


def thresholds_from_dict(data: Mapping[str, Any] | None) -> InterferenceThresholds:
    """辞書から閾値を構築する。None なら既定値。未知キー・負値などは ValueError。"""
    if data is None:
        return InterferenceThresholds()
    if not isinstance(data, Mapping):
        raise ValueError("latency_interference はJSONオブジェクトで指定してください")

    unknown = set(data) - _THRESHOLD_KEYS
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"latency_interference に未知のキーがあります: {joined}")

    kwargs: dict[str, Any] = {}
    for key in (
        "max_other_busy_cores_mean",
        "max_other_busy_cores_p95",
        "max_psi_cpu_some_ratio_mean",
    ):
        if key in data:
            kwargs[key] = _expect_optional_threshold(data[key], key)

    if "sample_interval_s" in data:
        value = data["sample_interval_s"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("sample_interval_s には数値を指定してください")
        if value <= 0:
            raise ValueError("sample_interval_s には0より大きい値を指定してください")
        kwargs["sample_interval_s"] = float(value)

    if "is_provisional" in data:
        value = data["is_provisional"]
        if not isinstance(value, bool):
            raise ValueError("is_provisional には真偽値を指定してください")
        kwargs["is_provisional"] = value

    return InterferenceThresholds(**kwargs)


def thresholds_to_dict(thresholds: InterferenceThresholds) -> dict[str, Any]:
    """閾値をJSON互換の辞書に変換する。"""
    return {
        "max_other_busy_cores_mean": thresholds.max_other_busy_cores_mean,
        "max_other_busy_cores_p95": thresholds.max_other_busy_cores_p95,
        "max_psi_cpu_some_ratio_mean": thresholds.max_psi_cpu_some_ratio_mean,
        "sample_interval_s": thresholds.sample_interval_s,
        "is_provisional": thresholds.is_provisional,
    }


def judge_latency_validity(
    summary: Mapping[str, Any], thresholds: InterferenceThresholds
) -> dict[str, Any]:
    """summary と閾値からレイテンシ計測の有効性を判定する。

    閾値 None の指標は判定しない。閾値があるが値が None の指標は unavailable。
    判定対象が 1 つも無ければ latency_valid=None（未判定）。1 つでも超過すれば
    False、それ以外は True。比較は「超える（>）」で無効。
    """
    reasons: list[str] = []
    checked: list[str] = []
    unavailable: list[str] = []
    exceeded = False

    mapping: Mapping[str, Any] = summary if isinstance(summary, Mapping) else {}

    for threshold_attr, metric_key, label in _THRESHOLD_TO_METRIC:
        threshold = getattr(thresholds, threshold_attr)
        if threshold is None:
            continue
        value = mapping.get(metric_key)
        if value is None:
            unavailable.append(label)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            unavailable.append(label)
            continue
        checked.append(label)
        if number > float(threshold):
            exceeded = True
            reasons.append(f"{label}={number:g} > {float(threshold):g}")

    if not checked:
        latency_valid: bool | None = None
    elif exceeded:
        latency_valid = False
    else:
        latency_valid = True

    return {
        "latency_valid": latency_valid,
        "reasons": reasons,
        "checked": checked,
        "unavailable": unavailable,
    }


def _default_clk_tck() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        return 100


def _self_cpu_seconds(process: Any | None) -> float | None:
    """自プロセス木の累積 CPU 秒を返す。psutil があればそれを優先する。"""
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            proc = process if process is not None else psutil.Process()
            times = proc.cpu_times()
            total = float(times.user) + float(times.system)
            for child in proc.children(recursive=True):
                try:
                    child_times = child.cpu_times()
                except psutil.Error:
                    continue
                total += float(child_times.user) + float(child_times.system)
            return total
        except psutil.Error:
            pass

    try:
        times = os.times()
    except Exception:
        return None
    return float(
        times.user + times.system + times.children_user + times.children_system
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_cpu_sample(
    *,
    proc_root: str = "/proc",
    process: Any | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CpuSample:
    """現在の CPU 使用状況サンプルを取得する。読めない項目は None。"""
    root = Path(proc_root)

    host_total: int | None = None
    host_idle: int | None = None
    stat_text = _read_text(root / "stat")
    if stat_text is not None:
        parsed = parse_proc_stat_cpu(stat_text)
        if parsed is not None:
            host_total, host_idle = parsed

    psi_total: int | None = None
    psi_text = _read_text(root / "pressure" / "cpu")
    if psi_text is not None:
        psi_total = parse_psi_some_total_us(psi_text)

    loadavg_1m: float | None = None
    loadavg_text = _read_text(root / "loadavg")
    if loadavg_text is not None:
        parsed_load = parse_loadavg(loadavg_text)
        if parsed_load is not None:
            loadavg_1m = parsed_load[0]

    return CpuSample(
        monotonic_s=clock(),
        host_total_jiffies=host_total,
        host_idle_jiffies=host_idle,
        self_cpu_s=_self_cpu_seconds(process),
        psi_some_total_us=psi_total,
        loadavg_1m=loadavg_1m,
    )


class InterferenceMonitor:
    """バックグラウンドで CPU 使用状況をサンプリングするコンテキストマネージャ。"""

    def __init__(
        self,
        interval_s: float = 0.5,
        *,
        sampler: Callable[[], CpuSample] | None = None,
        clk_tck: int | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s には0より大きい値を指定してください")
        self._interval_s = float(interval_s)
        self._sampler = sampler if sampler is not None else read_cpu_sample
        self._clk_tck = clk_tck if clk_tck is not None else _default_clk_tck()
        self._windows: list[dict[str, float | None]] = []
        self._start_sample: CpuSample | None = None
        self._end_sample: CpuSample | None = None
        self._prev_sample: CpuSample | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._platform_supported = False

    def __enter__(self) -> "InterferenceMonitor":
        start = self._sampler()
        self._start_sample = start
        self._prev_sample = start
        self._platform_supported = start.host_total_jiffies is not None
        self._stop_event.clear()
        thread = threading.Thread(target=self._run, daemon=True)
        self._thread = thread
        thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                sample = self._sampler()
            except Exception:
                continue
            with self._lock:
                prev = self._prev_sample
                if prev is not None:
                    self._windows.append(
                        window_metrics(prev, sample, clk_tck=self._clk_tck)
                    )
                self._prev_sample = sample

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=max(1.0, self._interval_s * 2))

        if thread is None or not thread.is_alive():
            try:
                end_sample: CpuSample | None = self._sampler()
            except Exception:
                end_sample = None
            with self._lock:
                self._end_sample = end_sample
                prev = self._prev_sample
                if end_sample is not None and prev is not None:
                    self._windows.append(
                        window_metrics(prev, end_sample, clk_tck=self._clk_tck)
                    )
                    self._prev_sample = end_sample
        return False

    def result(self) -> dict[str, Any]:
        """蓄積したサンプルから干渉判定用のレコードを返す。"""
        with self._lock:
            windows = list(self._windows)
            start = self._start_sample
            end = self._end_sample

        overall: dict[str, float | None] | None = None
        if start is not None and end is not None:
            overall = window_metrics(start, end, clk_tck=self._clk_tck)

        return {
            "platform_supported": self._platform_supported,
            "sample_interval_s": self._interval_s,
            "summary": summarize_windows(windows, overall),
            "loadavg_1m_start": start.loadavg_1m if start is not None else None,
            "loadavg_1m_end": end.loadavg_1m if end is not None else None,
            "notes": _NOTE_OTHER_BUSY,
        }


def measure_with_interference_check(
    measure: Callable[[], T],
    thresholds: InterferenceThresholds,
    *,
    max_retries: int = 0,
    monitor_factory: Callable[[float], Any] | None = None,
) -> tuple[T, dict[str, Any]]:
    """measure を干渉モニタ付きで実行し、判定結果とともに返す。

    latency_valid が False の試行が続く限り再計測する。戻り値は
    （採用した measure の結果, レコード）。採用規則は「最初に
    latency_valid が False でない試行、全部 False なら最後の試行」。
    """
    if max_retries < 0:
        raise ValueError("max_retries には0以上を指定してください")

    if monitor_factory is not None:
        factory = monitor_factory
    else:
        factory = lambda interval_s: InterferenceMonitor(interval_s)  # noqa: E731

    attempts: list[dict[str, Any]] = []
    adopted_index: int | None = None
    adopted_result: T | None = None
    adopted_judgement: dict[str, Any] | None = None

    for index in range(max_retries + 1):
        with factory(thresholds.sample_interval_s) as monitor:
            result = measure()
        monitor_result = monitor.result()
        summary = monitor_result.get("summary")
        if not isinstance(summary, Mapping):
            summary = {}
        judgement = judge_latency_validity(summary, thresholds)

        attempts.append(
            {
                "attempt": index,
                "latency_valid": judgement["latency_valid"],
                "reasons": judgement["reasons"],
                "unavailable": judgement["unavailable"],
                "summary": dict(summary),
                "platform_supported": monitor_result.get("platform_supported"),
            }
        )

        adopted_index = index
        adopted_result = result
        adopted_judgement = judgement

        if judgement["latency_valid"] is not False:
            break

    assert adopted_index is not None
    assert adopted_judgement is not None

    return adopted_result, {  # type: ignore[return-value]
        "thresholds": thresholds_to_dict(thresholds),
        "max_retries": max_retries,
        "attempts": attempts,
        "adopted_attempt": adopted_index,
        "latency_valid": adopted_judgement["latency_valid"],
        "reasons": adopted_judgement["reasons"],
        "thresholds_are_provisional": bool(thresholds.is_provisional),
        "notes": (
            "閾値は暫定値（Issue #31）。latency_valid=false は同一ホストの"
            "別プロセス干渉によりレイテンシ計測が不正確な可能性を示す。"
            "環境や負荷に応じて閾値を見直すこと。"
        ),
    }
