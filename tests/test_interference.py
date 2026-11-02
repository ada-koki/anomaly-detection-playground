"""adbench.interference のテスト。torch など重い依存は使わない。"""

from __future__ import annotations

import time
import unittest

from adbench import interference


def _sample(**kwargs: object) -> "interference.CpuSample":
    base: dict[str, object] = {
        "monotonic_s": 0.0,
        "host_total_jiffies": None,
        "host_idle_jiffies": None,
        "self_cpu_s": None,
        "psi_some_total_us": None,
        "loadavg_1m": None,
    }
    base.update(kwargs)
    return interference.CpuSample(**base)  # type: ignore[arg-type]


class ParseProcStatCpuTest(unittest.TestCase):
    def test_parse_first_cpu_line(self) -> None:
        text = (
            "cpu  100 20 30 400 50 5 6 7 0 0\n"
            "cpu0 10 2 3 40 5 0 0 0 0 0\n"
        )
        total, idle = interference.parse_proc_stat_cpu(text)
        self.assertEqual(total, 100 + 20 + 30 + 400 + 50 + 5 + 6 + 7)
        self.assertEqual(idle, 400 + 50)

    def test_skips_per_cpu_lines(self) -> None:
        self.assertIsNone(
            interference.parse_proc_stat_cpu("cpu0 1 2 3 4 5 6 7 8 0 0\n")
        )

    def test_invalid(self) -> None:
        self.assertIsNone(interference.parse_proc_stat_cpu(""))
        self.assertIsNone(interference.parse_proc_stat_cpu("cpu 1 2 3\n"))
        self.assertIsNone(
            interference.parse_proc_stat_cpu("cpu a b c d e f g h\n")
        )


class ParsePsiTest(unittest.TestCase):
    def test_some_total(self) -> None:
        text = (
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=12345\n"
            "full avg10=0.00 avg60=0.00 avg300=0.00 total=678\n"
        )
        self.assertEqual(interference.parse_psi_some_total_us(text), 12345)

    def test_missing_some(self) -> None:
        self.assertIsNone(
            interference.parse_psi_some_total_us("full total=1\n")
        )
        self.assertIsNone(interference.parse_psi_some_total_us(""))

    def test_bad_total(self) -> None:
        self.assertIsNone(
            interference.parse_psi_some_total_us("some total=abc\n")
        )


class ParseLoadavgTest(unittest.TestCase):
    def test_ok(self) -> None:
        parsed = interference.parse_loadavg("0.50 0.25 0.10 1/100 1234\n")
        self.assertEqual(parsed, (0.5, 0.25, 0.1))

    def test_short(self) -> None:
        self.assertIsNone(interference.parse_loadavg("0.5 0.2\n"))

    def test_bad(self) -> None:
        self.assertIsNone(interference.parse_loadavg("a b c\n"))


class WindowMetricsTest(unittest.TestCase):
    def test_basic(self) -> None:
        prev = _sample(
            monotonic_s=0.0,
            host_total_jiffies=1000,
            host_idle_jiffies=500,
            self_cpu_s=10.0,
            psi_some_total_us=0,
        )
        cur = _sample(
            monotonic_s=1.0,
            host_total_jiffies=1800,
            host_idle_jiffies=900,
            self_cpu_s=11.5,
            psi_some_total_us=200000,
        )
        metrics = interference.window_metrics(prev, cur, clk_tck=100)
        self.assertAlmostEqual(metrics["elapsed_s"], 1.0)
        self.assertAlmostEqual(metrics["host_busy_cores"], 4.0)
        self.assertAlmostEqual(metrics["self_busy_cores"], 1.5)
        self.assertAlmostEqual(metrics["other_busy_cores"], 2.5)
        self.assertAlmostEqual(metrics["psi_cpu_some_ratio"], 0.2)

    def test_nonpositive_elapsed(self) -> None:
        prev = _sample(monotonic_s=1.0)
        cur = _sample(monotonic_s=1.0)
        metrics = interference.window_metrics(prev, cur, clk_tck=100)
        self.assertIsNone(metrics["elapsed_s"])
        self.assertIsNone(metrics["host_busy_cores"])
        self.assertIsNone(metrics["other_busy_cores"])

    def test_missing_host(self) -> None:
        prev = _sample(monotonic_s=0.0, self_cpu_s=1.0)
        cur = _sample(monotonic_s=1.0, self_cpu_s=2.0)
        metrics = interference.window_metrics(prev, cur, clk_tck=100)
        self.assertIsNone(metrics["host_busy_cores"])
        self.assertIsNone(metrics["other_busy_cores"])
        self.assertAlmostEqual(metrics["self_busy_cores"], 1.0)


class SummarizeWindowsTest(unittest.TestCase):
    def test_stats_and_none_excluded(self) -> None:
        windows = [
            {
                "other_busy_cores": 1.0,
                "host_busy_cores": 2.0,
                "self_busy_cores": 1.0,
                "psi_cpu_some_ratio": None,
            },
            {
                "other_busy_cores": 3.0,
                "host_busy_cores": 4.0,
                "self_busy_cores": 1.0,
                "psi_cpu_some_ratio": None,
            },
            {
                "other_busy_cores": None,
                "host_busy_cores": 3.0,
                "self_busy_cores": 1.0,
                "psi_cpu_some_ratio": None,
            },
            {
                "other_busy_cores": 2.0,
                "host_busy_cores": 3.0,
                "self_busy_cores": 1.0,
                "psi_cpu_some_ratio": None,
            },
        ]
        summary = interference.summarize_windows(windows, None)
        self.assertEqual(summary["n_windows"], 4)
        self.assertAlmostEqual(summary["other_busy_cores_mean"], 2.0)
        self.assertAlmostEqual(summary["other_busy_cores_max"], 3.0)
        self.assertAlmostEqual(summary["other_busy_cores_p95"], 3.0)
        self.assertIsNone(summary["psi_cpu_some_ratio_mean"])
        self.assertIsNone(summary["psi_cpu_some_ratio_max"])
        self.assertIsNone(summary["psi_cpu_some_ratio_p95"])
        self.assertIsNone(summary["overall"])


class ThresholdsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        thresholds = interference.thresholds_from_dict(None)
        self.assertEqual(thresholds.max_other_busy_cores_mean, 1.5)
        self.assertEqual(thresholds.max_other_busy_cores_p95, 3.0)
        self.assertIsNone(thresholds.max_psi_cpu_some_ratio_mean)
        self.assertEqual(thresholds.sample_interval_s, 0.5)
        self.assertTrue(thresholds.is_provisional)

    def test_partial_overrides_keep_defaults(self) -> None:
        thresholds = interference.thresholds_from_dict(
            {"max_other_busy_cores_mean": 2, "sample_interval_s": 0.25}
        )
        self.assertEqual(thresholds.max_other_busy_cores_mean, 2.0)
        self.assertEqual(thresholds.max_other_busy_cores_p95, 3.0)
        self.assertEqual(thresholds.sample_interval_s, 0.25)

    def test_unknown_key(self) -> None:
        with self.assertRaises(ValueError):
            interference.thresholds_from_dict({"foo": 1})

    def test_negative_threshold(self) -> None:
        with self.assertRaises(ValueError):
            interference.thresholds_from_dict({"max_other_busy_cores_mean": -1})

    def test_bool_rejected(self) -> None:
        with self.assertRaises(ValueError):
            interference.thresholds_from_dict({"max_other_busy_cores_mean": True})

    def test_nonpositive_interval(self) -> None:
        with self.assertRaises(ValueError):
            interference.thresholds_from_dict({"sample_interval_s": 0})

    def test_roundtrip(self) -> None:
        thresholds = interference.thresholds_from_dict(
            {"max_psi_cpu_some_ratio_mean": 0.1}
        )
        data = interference.thresholds_to_dict(thresholds)
        self.assertEqual(data["max_psi_cpu_some_ratio_mean"], 0.1)
        self.assertEqual(data["sample_interval_s"], 0.5)


class JudgeTest(unittest.TestCase):
    def test_exceed_marks_invalid(self) -> None:
        thresholds = interference.InterferenceThresholds()
        result = interference.judge_latency_validity(
            {"other_busy_cores_mean": 2.31, "other_busy_cores_p95": 1.0},
            thresholds,
        )
        self.assertIs(result["latency_valid"], False)
        self.assertTrue(
            any("other_busy_cores_mean" in reason for reason in result["reasons"])
        )

    def test_all_missing_is_unjudged(self) -> None:
        thresholds = interference.InterferenceThresholds()
        result = interference.judge_latency_validity({}, thresholds)
        self.assertIsNone(result["latency_valid"])
        self.assertEqual(result["checked"], [])
        self.assertIn("other_busy_cores_mean", result["unavailable"])

    def test_threshold_none_skips(self) -> None:
        thresholds = interference.InterferenceThresholds(
            max_other_busy_cores_mean=None, max_other_busy_cores_p95=None
        )
        result = interference.judge_latency_validity({}, thresholds)
        self.assertIsNone(result["latency_valid"])
        self.assertEqual(result["checked"], [])
        self.assertEqual(result["unavailable"], [])

    def test_within_threshold_valid(self) -> None:
        thresholds = interference.InterferenceThresholds()
        result = interference.judge_latency_validity(
            {"other_busy_cores_mean": 0.5, "other_busy_cores_p95": 1.0},
            thresholds,
        )
        self.assertIs(result["latency_valid"], True)
        self.assertEqual(result["reasons"], [])


class InterferenceMonitorTest(unittest.TestCase):
    def test_monitor_with_fake_sampler(self) -> None:
        state = {"t": 0.0, "total": 0, "idle": 0}

        def sampler() -> "interference.CpuSample":
            state["t"] += 0.05
            state["total"] += 50
            state["idle"] += 25
            return interference.CpuSample(
                monotonic_s=state["t"],
                host_total_jiffies=state["total"],
                host_idle_jiffies=state["idle"],
                self_cpu_s=state["t"] * 0.1,
                psi_some_total_us=0,
                loadavg_1m=0.5,
            )

        monitor = interference.InterferenceMonitor(
            interval_s=0.01, sampler=sampler, clk_tck=100
        )
        with monitor:
            time.sleep(0.05)
        result = monitor.result()
        self.assertTrue(result["platform_supported"])
        self.assertEqual(result["sample_interval_s"], 0.01)
        self.assertGreaterEqual(result["summary"]["n_windows"], 1)
        self.assertIn("other_busy_cores", result["notes"])
        self.assertGreater(result["loadavg_1m_start"], 0.0)


def _monitor_factory(
    summaries: list[dict[str, object]],
) -> "interference.Callable[[float], object]":
    iterator = iter(summaries)

    class _FakeMonitor:
        def __init__(self, interval_s: float) -> None:
            self._interval_s = interval_s
            self._summary = next(iterator)

        def __enter__(self) -> "_FakeMonitor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def result(self) -> dict[str, object]:
            return {
                "platform_supported": True,
                "sample_interval_s": self._interval_s,
                "summary": self._summary,
            }

    return lambda interval_s: _FakeMonitor(interval_s)


class MeasureWithInterferenceCheckTest(unittest.TestCase):
    def test_retry_adopts_valid_attempt(self) -> None:
        bad = {"other_busy_cores_mean": 5.0, "other_busy_cores_p95": 5.0}
        good = {"other_busy_cores_mean": 0.1, "other_busy_cores_p95": 0.2}
        factory = _monitor_factory([bad, good])
        result, record = interference.measure_with_interference_check(
            lambda: "measured",
            interference.InterferenceThresholds(),
            max_retries=1,
            monitor_factory=factory,
        )
        self.assertEqual(result, "measured")
        self.assertEqual(record["adopted_attempt"], 1)
        self.assertIs(record["latency_valid"], True)
        self.assertEqual(len(record["attempts"]), 2)
        self.assertIs(record["attempts"][0]["latency_valid"], False)

    def test_all_invalid_adopts_last_attempt(self) -> None:
        bad = {"other_busy_cores_mean": 5.0, "other_busy_cores_p95": 5.0}
        factory = _monitor_factory([dict(bad), dict(bad), dict(bad)])
        result, record = interference.measure_with_interference_check(
            lambda: "x",
            interference.InterferenceThresholds(),
            max_retries=2,
            monitor_factory=factory,
        )
        self.assertEqual(result, "x")
        self.assertEqual(record["adopted_attempt"], 2)
        self.assertIs(record["latency_valid"], False)
        self.assertEqual(len(record["attempts"]), 3)

    def test_negative_retries_raises(self) -> None:
        with self.assertRaises(ValueError):
            interference.measure_with_interference_check(
                lambda: 1,
                interference.InterferenceThresholds(),
                max_retries=-1,
            )


if __name__ == "__main__":
    unittest.main()
