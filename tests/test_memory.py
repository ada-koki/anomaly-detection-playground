"""メモリ計測ユーティリティのテスト。"""

import importlib.util
import sys
import time
import unittest

from adbench.memory import peak_rss_lifetime_mib, sampled_rss


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


if __name__ == "__main__":
    unittest.main()
