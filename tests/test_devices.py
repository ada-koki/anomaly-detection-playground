"""adbench.devices の純粋関数の単体テスト。

torch / anomalib を import しないため、標準ライブラリだけで実行できる。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from adbench import devices


def _provenance(**overrides: Any) -> dict[str, Any]:
    """build_provenance 用の既定値。テストごとに必要な項目だけ上書きする。"""
    values: dict[str, Any] = {
        "checkpoint_sha256": "a" * 64,
        "train_device": "cuda:0",
        "observed_train_device": "cuda",
        "cuda_device_name": "dummy-gpu-name",
        "torch_version": "dummy-torch",
        "anomalib_version": "dummy-anomalib",
        "model": "padim",
        "category": "bottle",
        "protocol_name": "dummy-protocol",
        "protocol_sha256": "b" * 64,
        "run_id": "run-1",
    }
    values.update(overrides)
    return devices.build_provenance(**values)


class ExecutionModeConstantsTest(unittest.TestCase):
    def test_modes_are_unique_and_contain_expected_values(self) -> None:
        self.assertEqual(
            len(devices.EXECUTION_MODES), len(set(devices.EXECUTION_MODES))
        )
        for mode in (
            devices.EXECUTION_MODE_CPU_ONLY,
            devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
            devices.EXECUTION_MODE_GPU_REFERENCE,
            devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
            devices.EXECUTION_MODE_CPU_TRAIN_GPU_INFER,
        ):
            with self.subTest(mode=mode):
                self.assertIn(mode, devices.EXECUTION_MODES)


class NormalizeDeviceTest(unittest.TestCase):
    def test_plain_types(self) -> None:
        self.assertEqual(devices.normalize_device("cpu"), "cpu")
        self.assertEqual(devices.normalize_device("cuda"), "cuda")
        self.assertEqual(devices.normalize_device("mps"), "mps")

    def test_index_and_case_are_normalized(self) -> None:
        self.assertEqual(devices.normalize_device("cuda:0"), "cuda")
        self.assertEqual(devices.normalize_device("CUDA:1"), "cuda")
        self.assertEqual(devices.normalize_device("CPU"), "cpu")
        self.assertEqual(devices.normalize_device("  cuda:2  "), "cuda")
        self.assertEqual(devices.normalize_device("Mps:0"), "mps")

    def test_invalid_strings_raise_value_error(self) -> None:
        for value in ("", "   ", "tpu", "cuda:x", "cuda:", ":0"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    devices.normalize_device(value)

    def test_non_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            devices.normalize_device(123)  # type: ignore[arg-type]


class ClassifyExecutionModeTest(unittest.TestCase):
    def test_all_combinations(self) -> None:
        cases = [
            (
                None,
                "cpu",
                devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
            ),
            (
                None,
                "cuda",
                devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
            ),
            ("cpu", "cpu", devices.EXECUTION_MODE_CPU_ONLY),
            ("cpu", "cuda", devices.EXECUTION_MODE_CPU_TRAIN_GPU_INFER),
            ("cuda", "cpu", devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER),
            ("cuda", "cuda", devices.EXECUTION_MODE_GPU_REFERENCE),
        ]
        for train, infer, expected in cases:
            with self.subTest(train=train, infer=infer):
                actual = devices.classify_execution_mode(train, infer)
                self.assertEqual(actual, expected)
                # 返り値は必ず定義済みのモードであること。
                self.assertIn(actual, devices.EXECUTION_MODES)

    def test_device_index_is_normalized_before_classification(self) -> None:
        self.assertEqual(
            devices.classify_execution_mode("cuda:0", "cpu"),
            devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
        )
        self.assertEqual(
            devices.classify_execution_mode("cuda:0", "CUDA:1"),
            devices.EXECUTION_MODE_GPU_REFERENCE,
        )

    def test_unsupported_combination_raises_value_error(self) -> None:
        for train, infer in (
            ("mps", "cpu"),
            ("cpu", "mps"),
            ("mps", "mps"),
        ):
            with self.subTest(train=train, infer=infer):
                with self.assertRaises(ValueError):
                    devices.classify_execution_mode(train, infer)

    def test_invalid_device_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            devices.classify_execution_mode(None, "tpu")
        with self.assertRaises(ValueError):
            devices.classify_execution_mode("tpu", "cpu")


class DetectDeviceMismatchTest(unittest.TestCase):
    def test_matching_devices_produce_no_warning(self) -> None:
        self.assertEqual(
            devices.detect_device_mismatch("cuda", ["cuda", "cuda:0", "CUDA"]),
            [],
        )
        self.assertEqual(devices.detect_device_mismatch("cpu", []), [])

    def test_mismatch_produces_japanese_warning(self) -> None:
        warnings = devices.detect_device_mismatch("cuda", ["cpu"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("cuda", warnings[0])
        self.assertIn("cpu", warnings[0])
        self.assertIn("フォールバック", warnings[0])

    def test_duplicate_mismatch_is_reported_once(self) -> None:
        warnings = devices.detect_device_mismatch("cuda", ["cpu", "cpu:0", "CPU"])
        self.assertEqual(len(warnings), 1)

    def test_multiple_distinct_mismatches(self) -> None:
        warnings = devices.detect_device_mismatch("cuda", ["cpu", "mps"])
        self.assertEqual(len(warnings), 2)

    def test_invalid_device_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            devices.detect_device_mismatch("cuda", ["tpu"])
        with self.assertRaises(ValueError):
            devices.detect_device_mismatch("bogus", ["cpu"])


class EffectiveDeviceTest(unittest.TestCase):
    def test_observed_device_wins_when_valid(self) -> None:
        self.assertEqual(devices.effective_device("cuda", "cpu"), "cpu")
        self.assertEqual(devices.effective_device("cpu", "cuda:0"), "cuda")

    def test_requested_device_is_fallback(self) -> None:
        self.assertEqual(devices.effective_device("cuda", "unknown"), "cuda")
        self.assertEqual(devices.effective_device("cuda", None), "cuda")
        self.assertEqual(devices.effective_device("cuda", ""), "cuda")


class BuildProvenanceTest(unittest.TestCase):
    def test_normalizes_devices_and_sets_schema_version(self) -> None:
        data = _provenance(train_device="cuda:0", observed_train_device="cuda:1")
        self.assertEqual(data["schema_version"], devices.PROVENANCE_SCHEMA_VERSION)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["train_device"], "cuda")
        self.assertEqual(data["observed_train_device"], "cuda")

    def test_unknown_observed_device_becomes_none(self) -> None:
        data = _provenance(observed_train_device="unknown")
        self.assertIsNone(data["observed_train_device"])

    def test_none_train_device_is_kept_as_none(self) -> None:
        data = _provenance(train_device=None, observed_train_device=None)
        self.assertIsNone(data["train_device"])
        self.assertIsNone(data["observed_train_device"])

    def test_cpu_train_device_is_kept(self) -> None:
        data = _provenance(train_device="CPU", observed_train_device="cpu:0")
        self.assertEqual(data["train_device"], "cpu")
        self.assertEqual(data["observed_train_device"], "cpu")

    def test_invalid_train_device_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            _provenance(train_device="tpu")

    def test_other_fields_are_preserved(self) -> None:
        data = _provenance(run_id="run-xyz", model="patchcore", category="screw")
        self.assertEqual(data["run_id"], "run-xyz")
        self.assertEqual(data["model"], "patchcore")
        self.assertEqual(data["category"], "screw")
        self.assertEqual(data["checkpoint_sha256"], "a" * 64)


class ProvenancePathTest(unittest.TestCase):
    def test_sidecar_name_for_checkpoint_file(self) -> None:
        self.assertEqual(
            devices.provenance_path_for("model.ckpt"),
            Path("model.ckpt.provenance.json"),
        )
        self.assertEqual(
            devices.provenance_path_for(Path("ckpt/model.ckpt")),
            Path("ckpt/model.ckpt.provenance.json"),
        )


class ProvenanceFileTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        data = _provenance()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt.provenance.json"
            devices.write_provenance(path, data)
            self.assertTrue(path.is_file())
            loaded = devices.read_provenance(path)
        self.assertEqual(loaded, data)

    def test_parent_directories_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dir" / "model.ckpt.provenance.json"
            devices.write_provenance(path, {"schema_version": 1})
            self.assertTrue(path.is_file())

    def test_written_file_is_valid_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            devices.write_provenance(path, _provenance(category="ボトル"))
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["category"], "ボトル")
        self.assertEqual(payload["schema_version"], 1)

    def test_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                devices.read_provenance(Path(tmp) / "missing.json")
            )

    def test_broken_json_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                devices.read_provenance(path)

    def test_non_object_json_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ValueError):
                devices.read_provenance(path)


class ProvenanceMatchesTest(unittest.TestCase):
    def test_matching_sha_is_case_and_space_insensitive(self) -> None:
        provenance = {"checkpoint_sha256": "  ABC123  "}
        self.assertTrue(devices.provenance_matches(provenance, "abc123"))
        self.assertTrue(devices.provenance_matches(provenance, " ABC123 "))

    def test_mismatch_returns_false(self) -> None:
        self.assertFalse(
            devices.provenance_matches({"checkpoint_sha256": "aaa"}, "bbb")
        )

    def test_missing_or_invalid_values_return_false(self) -> None:
        self.assertFalse(devices.provenance_matches(None, "aaa"))
        self.assertFalse(devices.provenance_matches({}, "aaa"))
        self.assertFalse(
            devices.provenance_matches({"checkpoint_sha256": None}, "aaa")
        )
        self.assertFalse(
            devices.provenance_matches({"checkpoint_sha256": ""}, "aaa")
        )
        self.assertFalse(
            devices.provenance_matches({"checkpoint_sha256": "aaa"}, None)
        )
        self.assertFalse(
            devices.provenance_matches({"checkpoint_sha256": "aaa"}, "")
        )

    def test_matches_built_provenance(self) -> None:
        data = _provenance(checkpoint_sha256="c" * 64)
        self.assertTrue(devices.provenance_matches(data, "c" * 64))
        self.assertFalse(devices.provenance_matches(data, "d" * 64))


class ExecutionModeOfReportTest(unittest.TestCase):
    def test_new_format_is_used_as_is(self) -> None:
        self.assertEqual(
            devices.execution_mode_of_report(
                {"execution_mode": devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER}
            ),
            devices.EXECUTION_MODE_GPU_TRAIN_CPU_INFER,
        )
        self.assertEqual(
            devices.execution_mode_of_report(
                {"execution_mode": "some_future_mode"}
            ),
            "some_future_mode",
        )

    def test_old_format_training_mode_is_estimated(self) -> None:
        self.assertEqual(
            devices.execution_mode_of_report(
                {"training_mode": "cpu_train_cpu_infer"}
            ),
            devices.EXECUTION_MODE_CPU_ONLY,
        )
        self.assertEqual(
            devices.execution_mode_of_report(
                {"training_mode": "external_checkpoint_cpu_infer"}
            ),
            devices.EXECUTION_MODE_EXTERNAL_CHECKPOINT_UNKNOWN_TRAIN_DEVICE,
        )

    def test_old_format_without_training_mode_uses_device(self) -> None:
        self.assertEqual(
            devices.execution_mode_of_report({"device": "cuda"}),
            devices.EXECUTION_MODE_GPU_REFERENCE,
        )
        self.assertEqual(
            devices.execution_mode_of_report({"device": "cuda:1"}),
            devices.EXECUTION_MODE_GPU_REFERENCE,
        )
        self.assertEqual(
            devices.execution_mode_of_report({"device": "cpu"}),
            devices.EXECUTION_MODE_CPU_ONLY,
        )

    def test_nested_device_is_looked_up(self) -> None:
        self.assertEqual(
            devices.execution_mode_of_report({"conditions": {"device": "cuda"}}),
            devices.EXECUTION_MODE_GPU_REFERENCE,
        )
        self.assertEqual(
            devices.execution_mode_of_report({"args": {"device": "cuda:0"}}),
            devices.EXECUTION_MODE_GPU_REFERENCE,
        )
        self.assertEqual(
            devices.execution_mode_of_report(
                {"environment": {"device": "cuda"}}
            ),
            devices.EXECUTION_MODE_GPU_REFERENCE,
        )

    def test_empty_report_falls_back_to_cpu_only(self) -> None:
        self.assertEqual(
            devices.execution_mode_of_report({}),
            devices.EXECUTION_MODE_CPU_ONLY,
        )
        self.assertEqual(
            devices.execution_mode_of_report(None),
            devices.EXECUTION_MODE_CPU_ONLY,
        )

    def test_unknown_device_string_falls_back_to_cpu_only(self) -> None:
        self.assertEqual(
            devices.execution_mode_of_report({"device": "tpu"}),
            devices.EXECUTION_MODE_CPU_ONLY,
        )

    def test_empty_execution_mode_is_ignored(self) -> None:
        self.assertEqual(
            devices.execution_mode_of_report({"execution_mode": ""}),
            devices.EXECUTION_MODE_CPU_ONLY,
        )


if __name__ == "__main__":
    unittest.main()
