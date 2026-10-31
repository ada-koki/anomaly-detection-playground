"""adbench.manifest の純粋関数テスト。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adbench.manifest import (
    ManifestError,
    ManifestRow,
    build_data_split_report,
    build_manifest,
    find_duplicate_image_paths,
    manifest_sha256,
    rows_from_samples,
    summarize_manifest,
)


class ManifestTest(unittest.TestCase):
    def _rows(self) -> list[ManifestRow]:
        return [
            ManifestRow("train", "bottle/train/good/000.png", "good", 0, None),
            ManifestRow("train", "bottle/train/good/001.png", "good", 0, None),
            ManifestRow("val", "bottle/test/broken_large/000.png", "broken_large", 1, "bottle/ground_truth/broken_large/000_mask.png"),
            ManifestRow("test", "bottle/test/good/000.png", "good", 0, None),
            ManifestRow("test", "bottle/test/broken_small/000.png", "broken_small", 1, "bottle/ground_truth/broken_small/000_mask.png"),
        ]

    def _same_as_test_rows(self) -> list[ManifestRow]:
        return [
            ManifestRow("train", "bottle/train/good/000.png", "good", 0, None),
            ManifestRow("val", "bottle/test/good/000.png", "good", 0, None),
            ManifestRow("val", "bottle/test/broken_large/000.png", "broken_large", 1, "bottle/ground_truth/broken_large/000_mask.png"),
            ManifestRow("test", "bottle/test/good/000.png", "good", 0, None),
            ManifestRow("test", "bottle/test/broken_large/000.png", "broken_large", 1, "bottle/ground_truth/broken_large/000_mask.png"),
        ]

    def test_summary_counts_and_label_counts(self) -> None:
        summary = summarize_manifest(self._rows())
        self.assertEqual(summary["roles"]["train"]["count"], 2)
        self.assertEqual(summary["roles"]["val"]["count"], 1)
        self.assertEqual(summary["roles"]["test"]["count"], 2)
        self.assertEqual(summary["roles"]["train"]["label_counts"], {"good": 2})
        self.assertEqual(
            summary["roles"]["test"]["label_counts"],
            {"broken_small": 1, "good": 1},
        )
        self.assertFalse(summary["overlap_detected"])
        self.assertFalse(summary["overlap_allowed"])

    def test_duplicate_image_path_is_error(self) -> None:
        rows = self._rows()
        rows.append(ManifestRow("test", "bottle/train/good/000.png", "good", 0, None))
        with self.assertRaises(ManifestError):
            summarize_manifest(rows)

    def test_duplicate_detection_returns_roles(self) -> None:
        rows = [
            ManifestRow("train", "same.png", "good", 0, None),
            ManifestRow("test", "same.png", "good", 0, None),
        ]
        duplicates = find_duplicate_image_paths(rows)
        self.assertEqual(duplicates, {"same.png": ["test", "train"]})
        with self.assertRaises(ManifestError):
            summarize_manifest(rows)

    def test_same_as_test_overlap_is_allowed(self) -> None:
        summary = summarize_manifest(
            self._same_as_test_rows(),
            allow_val_test_overlap=True,
        )
        self.assertTrue(summary["overlap_detected"])
        self.assertTrue(summary["overlap_allowed"])
        self.assertEqual(summary["overlap_image_count"], 2)
        self.assertIn("same_as_test", summary["overlap_reason"])
        self.assertEqual(summary["roles"]["val"]["count"], 2)
        self.assertEqual(summary["roles"]["test"]["count"], 2)

    def test_same_as_test_overlap_is_error_without_permission(self) -> None:
        with self.assertRaises(ManifestError):
            summarize_manifest(self._same_as_test_rows())

    def test_partial_val_test_overlap_is_error_even_when_allowed(self) -> None:
        rows = [
            ManifestRow("val", "bottle/test/good/000.png", "good", 0, None),
            ManifestRow("test", "bottle/test/good/000.png", "good", 0, None),
            ManifestRow("test", "bottle/test/good/001.png", "good", 0, None),
        ]
        with self.assertRaises(ManifestError):
            summarize_manifest(rows, allow_val_test_overlap=True)

    def test_train_test_overlap_is_error_even_when_allowed(self) -> None:
        rows = self._same_as_test_rows()
        rows.append(
            ManifestRow("test", "bottle/train/good/000.png", "good", 0, None)
        )
        with self.assertRaises(ManifestError):
            summarize_manifest(rows, allow_val_test_overlap=True)

    def test_sha256_is_order_independent(self) -> None:
        rows = self._rows()
        reversed_rows = list(reversed(rows))
        self.assertEqual(manifest_sha256(rows), manifest_sha256(reversed_rows))

    def test_sha256_changes_when_content_changes(self) -> None:
        rows = self._rows()
        changed = list(rows)
        changed[0] = ManifestRow("train", "bottle/train/good/000.png", "good", 0, "mask.png")
        self.assertNotEqual(manifest_sha256(rows), manifest_sha256(changed))

    def test_build_manifest_has_roles_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(self._rows(), Path(directory))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["splits"]["train"]), 2)
        self.assertIn("manifest_sha256", manifest["summary"])
        self.assertFalse(manifest["summary"]["overlap_detected"])
        self.assertFalse(manifest["summary"]["overlap_allowed"])

    def test_build_manifest_allows_same_as_test_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest(
                self._same_as_test_rows(),
                Path(directory),
                allow_val_test_overlap=True,
            )
        self.assertTrue(manifest["summary"]["overlap_allowed"])
        self.assertEqual(len(manifest["splits"]["val"]), 2)
        self.assertEqual(len(manifest["splits"]["test"]), 2)

    def test_data_split_report_marks_allowed_overlap(self) -> None:
        summary = summarize_manifest(
            self._same_as_test_rows(),
            allow_val_test_overlap=True,
        )
        report = build_data_split_report(summary, {"source": "test"})
        self.assertTrue(report["overlap_detected"])
        self.assertTrue(report["overlap_allowed"])
        self.assertEqual(report["overlap_image_count"], 2)
        self.assertIn("same_as_test", report["overlap_reason"])
        self.assertEqual(report["threshold_selection"], {"source": "test"})

    def test_rows_from_samples_uses_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                {
                    "image_path": str(root / "bottle" / "train" / "good" / "000.png"),
                    "label": "good",
                    "label_index": 0,
                    "mask_path": None,
                }
            ]
            rows = rows_from_samples(samples, "train", root)
        self.assertEqual(rows[0].image_path, "bottle/train/good/000.png")
        self.assertEqual(rows[0].label, "good")
        self.assertIsNone(rows[0].mask_path)


if __name__ == "__main__":
    unittest.main()
