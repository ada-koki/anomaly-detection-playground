"""EfficientAD のプロトコル設定に関する依存なしテスト。"""

from __future__ import annotations

import copy
import unittest

from adbench.protocol import (
    ProtocolError,
    apply_overrides,
    load_protocol,
    protocol_from_dict,
)


def _base_protocol(models: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "protocol_name": "test_protocol",
        "protocol_version": 1,
        "dataset": "mvtec_ad",
        "val_split_mode": "from_test",
        "val_split_ratio": 0.2,
        "seed": 42,
        "image_size": 256,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_workers": 0,
        "cpu_threads": 0,
        "models": models if models is not None else {"padim": {}},
    }


class EfficientAdProtocolTest(unittest.TestCase):
    def test_default_protocol_has_efficient_ad_settings(self) -> None:
        protocol = load_protocol()
        config = protocol.models["efficient_ad"]
        self.assertEqual(config.model_size, "small")
        self.assertEqual(config.max_epochs, 1)
        self.assertEqual(config.max_steps, -1)

    def test_protocol_accepts_valid_efficient_ad_settings(self) -> None:
        data = _base_protocol(
            {
                "efficient_ad": {
                    "model_size": "medium",
                    "max_epochs": 3,
                    "max_steps": 100,
                }
            }
        )
        protocol = protocol_from_dict(data)
        config = protocol.models["efficient_ad"]
        self.assertEqual(config.model_size, "medium")
        self.assertEqual(config.max_epochs, 3)
        self.assertEqual(config.max_steps, 100)

    def test_protocol_rejects_invalid_model_size(self) -> None:
        data = _base_protocol({"efficient_ad": {"model_size": "large"}})
        with self.assertRaises(ProtocolError):
            protocol_from_dict(data)

    def test_protocol_rejects_invalid_max_epochs(self) -> None:
        data = _base_protocol({"efficient_ad": {"max_epochs": 0}})
        with self.assertRaises(ProtocolError):
            protocol_from_dict(data)

    def test_protocol_rejects_invalid_max_steps(self) -> None:
        for value in (-2, 0):
            data = _base_protocol({"efficient_ad": {"max_steps": value}})
            with self.assertRaises(ProtocolError):
                protocol_from_dict(data)

    def test_efficient_ad_fields_are_rejected_for_other_models(self) -> None:
        data = _base_protocol({"padim": {"max_steps": 10}})
        with self.assertRaises(ProtocolError):
            protocol_from_dict(data)

    def test_apply_overrides_updates_efficient_ad_fields(self) -> None:
        protocol = load_protocol()
        updated = apply_overrides(
            protocol,
            {
                "models.efficient_ad.model_size": "medium",
                "models.efficient_ad.max_epochs": 2,
                "models.efficient_ad.max_steps": 5,
            },
        )
        config = updated.models["efficient_ad"]
        self.assertEqual(config.model_size, "medium")
        self.assertEqual(config.max_epochs, 2)
        self.assertEqual(config.max_steps, 5)

    def test_existing_model_config_is_not_changed(self) -> None:
        protocol = load_protocol()
        # 既存モデルの設定が EfficientAD 追加で変わっていないことを確認する。
        self.assertEqual(protocol.models["padim"].backbone, "resnet18")
        self.assertEqual(protocol.models["patchcore"].backbone, "wide_resnet50_2")
        data = copy.deepcopy(protocol.models["padim"])
        self.assertIsNone(data.model_size)
        self.assertIsNone(data.max_epochs)
        self.assertIsNone(data.max_steps)


if __name__ == "__main__":
    unittest.main()
