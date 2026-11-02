"""プロトコルの latency_interference 読み込み・検証のテスト。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from adbench import protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
VISA_PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "visa_v1.json"


def _minimal_protocol() -> dict[str, Any]:
    return json.loads(VISA_PROTOCOL_PATH.read_text(encoding="utf-8"))


class ProtocolLatencyInterferenceTest(unittest.TestCase):
    def test_absent_key_not_emitted(self) -> None:
        data = _minimal_protocol()
        loaded = protocol.protocol_from_dict(data)
        self.assertIsNone(loaded.latency_interference)
        emitted = protocol.protocol_to_dict(loaded)
        self.assertNotIn("latency_interference", emitted)

    def test_present_key_is_normalized(self) -> None:
        data = _minimal_protocol()
        data["latency_interference"] = {
            "max_other_busy_cores_mean": 2,
            "sample_interval_s": 0.25,
        }
        loaded = protocol.protocol_from_dict(data)
        self.assertIsNotNone(loaded.latency_interference)
        assert loaded.latency_interference is not None
        self.assertEqual(loaded.latency_interference["max_other_busy_cores_mean"], 2.0)
        self.assertEqual(loaded.latency_interference["sample_interval_s"], 0.25)
        self.assertEqual(loaded.latency_interference["max_other_busy_cores_p95"], 3.0)
        emitted = protocol.protocol_to_dict(loaded)
        self.assertIn("latency_interference", emitted)

    def test_unknown_key_raises_protocol_error(self) -> None:
        data = _minimal_protocol()
        data["latency_interference"] = {"foo": 1}
        with self.assertRaises(protocol.ProtocolError):
            protocol.protocol_from_dict(data)

    def test_apply_overrides_top_level(self) -> None:
        loaded = protocol.protocol_from_dict(_minimal_protocol())
        updated = protocol.apply_overrides(
            loaded,
            {"latency_interference": {"max_other_busy_cores_mean": 3}},
        )
        self.assertIsNotNone(updated.latency_interference)
        assert updated.latency_interference is not None
        self.assertEqual(
            updated.latency_interference["max_other_busy_cores_mean"], 3.0
        )


if __name__ == "__main__":
    unittest.main()
