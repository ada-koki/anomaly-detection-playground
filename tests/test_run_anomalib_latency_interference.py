"""Issue #31: run_anomalib のレイテンシ干渉判定まわりの CLI 検証（重い依存なし）。"""

import importlib.util
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_anomalib.py"
spec = importlib.util.spec_from_file_location("run_anomalib_i31", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _validate(argv: list[str]) -> object:
    args, _, _, _ = module.parse_and_resolve(argv + ["--prefetch-weights"])
    module.validate_args(args, module.build_parser())
    return args


class LatencyInterferenceCliTests(unittest.TestCase):
    def test_default_retries(self) -> None:
        args = _validate(["--model", "patchcore"])
        self.assertEqual(args.latency_max_retries, 1)

    def test_protocol_thresholds_are_resolved(self) -> None:
        args = _validate(["--model", "patchcore"])
        # 既定プロトコルに latency_interference が無ければ None（暫定既定値を使う）
        value = args.latency_interference
        self.assertTrue(value is None or isinstance(value, dict))
        thresholds = module.thresholds_from_dict(value)
        self.assertGreater(thresholds.sample_interval_s, 0)

    def test_negative_retries_rejected(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            _validate(["--model", "patchcore", "--latency-max-retries", "-1"])


if __name__ == "__main__":
    unittest.main()
