"""val_split_mode の受け付け方針テスト（Issue #7）。"""

from __future__ import annotations

import unittest
from pathlib import Path

from adbench.protocol import ProtocolError, load_protocol, protocol_from_dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FULL_TEST_PROTOCOL = _REPO_ROOT / "configs" / "protocols" / "mvtec_ad_full_test.json"


def _base(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "protocol_name": "test_protocol",
        "protocol_version": 1,
        "dataset": "mvtec_ad",
        "val_split_mode": "from_test",
        "val_split_ratio": 0.2,
        "seed": 0,
        "image_size": 256,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_workers": 0,
        "cpu_threads": 0,
        "models": {"padim": {}},
    }
    data.update(overrides)
    return data


class ValSplitModePolicyTest(unittest.TestCase):
    def test_none_is_rejected_with_reason(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            protocol_from_dict(_base(val_split_mode="none", val_split_ratio=None))
        message = str(context.exception)
        self.assertIn("none", message)
        self.assertIn("PostProcessor", message)
        self.assertIn("same_as_test", message)

    def test_same_as_test_is_accepted(self) -> None:
        protocol = protocol_from_dict(
            _base(val_split_mode="same_as_test", val_split_ratio=None)
        )
        self.assertEqual(protocol.val_split_mode, "same_as_test")
        self.assertIsNone(protocol.val_split_ratio)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            protocol_from_dict(_base(val_split_mode="from_train"))

    def test_full_test_protocol_uses_same_as_test(self) -> None:
        protocol = load_protocol(_FULL_TEST_PROTOCOL)
        self.assertEqual(protocol.val_split_mode, "same_as_test")
        self.assertIsNone(protocol.val_split_ratio)


if __name__ == "__main__":
    unittest.main()
