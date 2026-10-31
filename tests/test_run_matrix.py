"""scripts.run_matrix の純粋関数のテスト（標準ライブラリのみ）。"""

from __future__ import annotations

import unittest

from scripts.run_matrix import (
    recorded_run_keys,
    should_skip_run,
    summarize_error,
)


class RunMatrixPureTest(unittest.TestCase):
    def test_recorded_run_keys_ignores_broken_lines(self) -> None:
        lines = [
            '{"run_key": "a"}',
            "not json",
            '{"run_key": "b"}',
            '{"other": 1}',
            "",
        ]
        self.assertEqual(recorded_run_keys(lines), {"a", "b"})

    def test_should_skip_run(self) -> None:
        recorded = {"padim__bottle__s256__t8__r1"}
        self.assertFalse(should_skip_run("x", recorded, [], False))
        self.assertTrue(should_skip_run("x", recorded, ["y"], False))
        self.assertFalse(should_skip_run("x", recorded, ["x"], False))
        self.assertFalse(should_skip_run("x", recorded, [], True))
        self.assertTrue(
            should_skip_run("padim__bottle__s256__t8__r1", recorded, [], True)
        )

    def test_summarize_error_signal(self) -> None:
        message = summarize_error(-9, ["line1", "line2"])
        self.assertIn("signal SIGKILL", message)
        self.assertIn("line2", message)

    def test_summarize_error_returncode_and_report(self) -> None:
        message = summarize_error(1, ["line1", "boom"])
        self.assertIn("returncode 1", message)
        self.assertIn("boom", message)

        report_message = summarize_error(
            1,
            [],
            error_type="RuntimeError",
            error_message="failed",
        )
        self.assertEqual(report_message, "RuntimeError: failed")

        timeout_message = summarize_error(None, [], timed_out=True)
        self.assertIn("timeout", timeout_message)


if __name__ == "__main__":
    unittest.main()
