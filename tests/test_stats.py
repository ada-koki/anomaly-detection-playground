"""summarize_latencies のテスト。"""

import math
import unittest

from adbench.stats import summarize_latencies


class SummarizeLatenciesTests(unittest.TestCase):
    def test_known_values(self):
        summary = summarize_latencies([1, 2, 3, 4, 5])
        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["mean"], 3.0)
        self.assertAlmostEqual(summary["std"], math.sqrt(2.0))
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["p50"], 3.0)
        self.assertAlmostEqual(summary["p95"], 4.8)
        self.assertAlmostEqual(summary["max"], 5.0)

    def test_linear_interpolation(self):
        summary = summarize_latencies([0, 10])
        self.assertAlmostEqual(summary["p50"], 5.0)
        self.assertAlmostEqual(summary["p95"], 9.5)

    def test_single_value(self):
        summary = summarize_latencies([7.5])
        self.assertEqual(summary["count"], 1)
        self.assertAlmostEqual(summary["mean"], 7.5)
        self.assertAlmostEqual(summary["std"], 0.0)
        self.assertAlmostEqual(summary["p50"], 7.5)
        self.assertAlmostEqual(summary["p95"], 7.5)

    def test_empty_input(self):
        summary = summarize_latencies([])
        self.assertEqual(summary["count"], 0)
        for key in ("mean", "std", "min", "p50", "p95", "max"):
            self.assertIsNone(summary[key])


if __name__ == "__main__":
    unittest.main()
