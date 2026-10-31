"""メモリ計測ユーティリティのテスト。"""

import importlib.util
import sys
import time
import unittest

import subprocess

from adbench.memory import (
    _parse_vmhwm_kib,
    peak_rss_lifetime_mib,
    peak_rss_vmhwm_mib,
    sampled_rss,
)


PSUTIL_AVAILABLE = importlib.util.find_spec("psutil") is not None


@unittest.skipUnless(PSUTIL_AVAILABLE, "psutil がインストールされていません")
class SampledRssTests(unittest.TestCase):
    def test_sampled_rss_returns_positive_value(self):
        with sampled_rss(interval_s=0.01) as summary:
            time.sleep(0.03)
        self.assertGreater(summary["peak_rss_mib"], 0.0)

    def test_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            with sampled_rss(interval_s=0):
                pass


class LifetimeRssTests(unittest.TestCase):
    def test_peak_rss_lifetime_mib(self):
        value = peak_rss_lifetime_mib()
        if sys.platform.startswith("linux"):
            self.assertIsInstance(value, float)
            self.assertGreater(value, 0.0)
        else:
            self.assertTrue(value is None or value > 0.0)


class VmHwmTests(unittest.TestCase):
    def test_parse(self):
        text = "Name:\tpython\nVmPeak:\t  2048 kB\nVmHWM:\t  1536 kB\nVmRSS:\t 1000 kB\n"
        self.assertEqual(_parse_vmhwm_kib(text), 1536)
        self.assertIsNone(_parse_vmhwm_kib("Name:\tpython\n"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux 専用")
    def test_child_does_not_inherit_parent_peak(self):
        # 親で大きな領域を確保してから子プロセスを起動し、VmHWM が親の値を
        # 引き継がない（getrusage とは異なる）ことを確認する。
        blob = bytearray(300 * 1024 * 1024)
        for i in range(0, len(blob), 4096):
            blob[i] = 1
        parent_peak = peak_rss_vmhwm_mib()
        self.assertIsNotNone(parent_peak)
        code = (
            "from adbench.memory import peak_rss_vmhwm_mib, peak_rss_lifetime_mib;"
            "print(peak_rss_vmhwm_mib(), peak_rss_lifetime_mib())"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.split()
        child_vmhwm = float(out[0])
        self.assertLess(child_vmhwm, parent_peak - 200)
        del blob


if __name__ == "__main__":
    unittest.main()
