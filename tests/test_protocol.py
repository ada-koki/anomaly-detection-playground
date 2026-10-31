"""adbench.protocol の依存なしテスト。"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from adbench.protocol import (
    DEFAULT_PROTOCOL_PATH,
    ProtocolError,
    apply_overrides,
    load_protocol,
    protocol_file_sha256,
    protocol_to_dict,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FULL_TEST_PATH = _REPO_ROOT / "configs" / "protocols" / "mvtec_ad_full_test.json"


class ProtocolTest(unittest.TestCase):
    def _write_json(self, directory: str, data: dict) -> Path:
        path = Path(directory) / "protocol.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_load_default_protocol(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol.protocol_name, "mvtec_ad_v1")
        self.assertEqual(protocol.protocol_version, 1)
        self.assertEqual(protocol.dataset, "mvtec_ad")
        self.assertEqual(protocol.val_split_mode, "from_test")
        self.assertIsNotNone(protocol.val_split_ratio)
        self.assertIn("padim", protocol.models)
        self.assertIn("patchcore", protocol.models)
        self.assertEqual(protocol.models["padim"].backbone, "resnet18")

    def test_load_full_test_protocol(self) -> None:
        protocol = load_protocol(_FULL_TEST_PATH)
        self.assertEqual(protocol.protocol_name, "mvtec_ad_full_test")
        self.assertEqual(protocol.val_split_mode, "same_as_test")
        self.assertIsNone(protocol.val_split_ratio)

    def test_apply_top_level_and_model_overrides(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        original_seed = protocol.seed
        updated = apply_overrides(
            protocol,
            {
                "seed": 7,
                "image_size": 128,
                "models.padim.backbone": "resnet50",
                "models.padim.layers": ["layer2"],
            },
        )
        self.assertEqual(updated.seed, 7)
        self.assertEqual(updated.image_size, 128)
        self.assertEqual(updated.models["padim"].backbone, "resnet50")
        self.assertEqual(updated.models["padim"].layers, ("layer2",))
        self.assertEqual(protocol.seed, original_seed)

    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(
                json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
            )
            data["unknown_key"] = 1
            path = self._write_json(directory, data)
            with self.assertRaises(ProtocolError):
                load_protocol(path)

    def test_invalid_val_split_ratio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(
                json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
            )
            data["val_split_ratio"] = 1.5
            path = self._write_json(directory, data)
            with self.assertRaises(ProtocolError):
                load_protocol(path)

    def test_from_test_requires_val_split_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(
                json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
            )
            data["val_split_ratio"] = None
            path = self._write_json(directory, data)
            with self.assertRaises(ProtocolError):
                load_protocol(path)

    def test_invalid_patchcore_ratio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(
                json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
            )
            data["models"]["patchcore"]["coreset_sampling_ratio"] = 0
            path = self._write_json(directory, data)
            with self.assertRaises(ProtocolError):
                load_protocol(path)

    def test_protocol_to_dict_round_trip(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        data = protocol_to_dict(protocol)
        self.assertEqual(data["protocol_name"], protocol.protocol_name)
        self.assertEqual(data["models"]["padim"]["backbone"], "resnet18")
        self.assertIn("layers", data["models"]["patchcore"])

    def test_protocol_file_sha256_is_stable(self) -> None:
        first = protocol_file_sha256(DEFAULT_PROTOCOL_PATH)
        second = protocol_file_sha256(DEFAULT_PROTOCOL_PATH)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
