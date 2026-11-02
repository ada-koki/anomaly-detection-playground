"""adbench.mvtec4k の検証ロジックのテスト。

Pillow が無い環境でも純粋ロジックのテストは動くようにし、画像を使う
テストだけを skip する。numpy、torch、anomalib は import しない。
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from adbench.mvtec4k import (
    PAPER_COUNTS,
    Finding,
    Mvtec4kSample,
    build_clean_view,
    check_pixels,
    classify_resolution_condition,
    compare_with_paper,
    count_by_condition,
    load_manifest,
    scan_disk,
    validate_category,
    validate_entries,
)


HAS_PIL = importlib.util.find_spec("PIL") is not None


def _sample(**overrides: object) -> Mvtec4kSample:
    values: dict[str, object] = {
        "split": "test",
        "filename": "bottle/test/broken_small/000.png",
        "mask": "bottle/ground_truth/broken_small/000_mask.png",
        "foreground": "bottle/foreground/test/broken_small/000.png",
        "clsname": "bottle",
        "label": 1,
        "label_name": "broken_small",
        "line": 1,
    }
    values.update(overrides)
    return Mvtec4kSample(**values)  # type: ignore[arg-type]


def _train_good(**overrides: object) -> Mvtec4kSample:
    values: dict[str, object] = {
        "split": "train",
        "filename": "bottle/train/good/000.png",
        "mask": None,
        "foreground": "bottle/foreground/train/good/000.png",
        "clsname": "bottle",
        "label": 0,
        "label_name": "good",
        "line": 1,
    }
    values.update(overrides)
    return Mvtec4kSample(**values)  # type: ignore[arg-type]


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class ValidateEntriesTest(unittest.TestCase):
    def test_accepts_valid_abnormal_and_train_good(self) -> None:
        self.assertEqual(validate_entries([_sample()], "bottle"), [])
        self.assertEqual(validate_entries([_train_good()], "bottle"), [])

    def test_detects_each_violation(self) -> None:
        cases: list[tuple[str, Mvtec4kSample, list[Mvtec4kSample], str]] = [
            (
                "category",
                _sample(clsname="wood"),
                [],
                "category_mismatch",
            ),
            (
                "invalid_label",
                _sample(label=2),
                [],
                "invalid_label",
            ),
            (
                "label_name_relation",
                _sample(
                    label=0,
                    label_name="broken_small",
                    filename="bottle/test/broken_small/000.png",
                    mask=None,
                    foreground="bottle/foreground/test/broken_small/000.png",
                ),
                [],
                "label_name_mismatch",
            ),
            (
                "train_not_good",
                _sample(split="train", filename="bottle/train/broken_small/000.png"),
                [],
                "train_label_not_good",
            ),
            (
                "filename_parent",
                _sample(filename="bottle/test/other/000.png"),
                [],
                "filename_path_mismatch",
            ),
            (
                "mask_required",
                _sample(mask=None),
                [],
                "mask_required",
            ),
            (
                "mask_mismatch",
                _sample(mask="bottle/ground_truth/broken_small/999_mask.png"),
                [],
                "mask_path_mismatch",
            ),
            (
                "unexpected_mask",
                _sample(
                    label=0,
                    label_name="good",
                    filename="bottle/test/good/000.png",
                    mask="bottle/ground_truth/good/000_mask.png",
                    foreground="bottle/foreground/test/good/000.png",
                ),
                [],
                "unexpected_mask",
            ),
            (
                "foreground_mismatch",
                _sample(foreground="bottle/foreground/test/broken_small/999.png"),
                [],
                "foreground_path_mismatch",
            ),
            (
                "duplicate",
                _sample(),
                [_sample()],
                "duplicate_filename",
            ),
        ]

        for name, sample, extra, expected_code in cases:
            with self.subTest(name=name):
                findings = validate_entries([sample, *extra], "bottle")
                self.assertIn(expected_code, {finding.code for finding in findings})


class CountsAndPaperTest(unittest.TestCase):
    def test_count_by_condition(self) -> None:
        samples = [
            _train_good(filename="bottle/train/good/000.png", line=1),
            _train_good(filename="bottle/train/good/001.png", line=2),
            _sample(
                filename="bottle/test/good/000.png",
                mask=None,
                label=0,
                label_name="good",
                foreground="bottle/foreground/test/good/000.png",
            ),
            _sample(filename="bottle/test/broken_small/000.png"),
            _sample(filename="bottle/test/broken_small/001.png"),
            _sample(
                filename="bottle/test/tiny/000.png",
                mask="bottle/ground_truth/tiny/000_mask.png",
                label_name="tiny",
                foreground="bottle/foreground/test/tiny/000.png",
            ),
            _sample(
                filename="bottle/test/tiny/001.png",
                mask="bottle/ground_truth/tiny/001_mask.png",
                label_name="tiny",
                foreground="bottle/foreground/test/tiny/001.png",
            ),
            _sample(
                filename="bottle/test/tiny/002.png",
                mask="bottle/ground_truth/tiny/002_mask.png",
                label_name="tiny",
                foreground="bottle/foreground/test/tiny/002.png",
            ),
        ]
        counts = count_by_condition(samples)
        self.assertEqual(counts["train"], 2)
        self.assertEqual(counts["test_good"], 1)
        self.assertEqual(counts["large"], 2)
        self.assertEqual(counts["tiny"], 3)
        self.assertEqual(counts["by_label_name"]["train"]["good"], 2)
        self.assertEqual(counts["by_label_name"]["test"]["tiny"], 3)

    def test_compare_with_paper(self) -> None:
        counts = dict(PAPER_COUNTS["bottle"])
        self.assertEqual(compare_with_paper("bottle", counts), [])

        counts["train"] += 1
        findings = compare_with_paper("bottle", counts)
        self.assertEqual([finding.code for finding in findings], ["paper_count_mismatch"])
        self.assertEqual(findings[0].severity, "warning")

        unknown = compare_with_paper("unknown", counts)
        self.assertEqual(unknown[0].code, "unknown_category")


class ResolutionConditionTest(unittest.TestCase):
    def test_all_branches(self) -> None:
        cases = [
            (None, (256, 256), "unknown"),
            ((4096, 4096), None, "unknown"),
            ((256, 256), (256, 256), "native"),
            ((4096, 4096), (256, 256), "downscaled"),
            ((4096, 2048), (512, 512), "downscaled"),
            ((512, 512), (4096, 4096), "upscaled"),
            ((512, 512), (512, 1024), "upscaled"),
            ((2048, 2048), (4096, 512), "mixed"),
        ]
        for source, model_input, expected in cases:
            with self.subTest(source=source, model_input=model_input):
                self.assertEqual(
                    classify_resolution_condition(source, model_input),
                    expected,
                )


class LoadManifestTest(unittest.TestCase):
    def test_missing_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples, findings = load_manifest(root, "bottle")
            self.assertEqual(samples, [])
            self.assertEqual(
                [finding.code for finding in findings],
                ["manifest_missing", "manifest_missing"],
            )

    def test_valid_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_jsonl(
                root / "bottle" / "train.jsonl",
                [
                    {
                        "filename": "bottle/train/good/000.png",
                        "foreground": "bottle/foreground/train/good/000.png",
                        "clsname": "bottle",
                        "label": 0,
                        "label_name": "good",
                    }
                ],
            )
            _write_jsonl(
                root / "bottle" / "test.jsonl",
                [
                    {
                        "filename": "bottle/test/broken_small/000.png",
                        "mask": "bottle/ground_truth/broken_small/000_mask.png",
                        "foreground": "bottle/foreground/test/broken_small/000.png",
                        "clsname": "bottle",
                        "label": 1,
                        "label_name": "broken_small",
                    }
                ],
            )
            samples, findings = load_manifest(root, "bottle")
            self.assertEqual(findings, [])
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].line, 1)
            self.assertIsNone(samples[0].mask)
            self.assertEqual(samples[1].label_name, "broken_small")

    def test_broken_json_and_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "bottle" / "train.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(
                "not-json\n"
                + json.dumps({"filename": "bottle/train/good/000.png"})
                + "\n",
                encoding="utf-8",
            )
            _write_jsonl(root / "bottle" / "test.jsonl", [])
            samples, findings = load_manifest(root, "bottle")
            self.assertEqual(samples, [])
            codes = {finding.code for finding in findings}
            self.assertIn("invalid_json", codes)
            self.assertIn("missing_key", codes)


class ScanDiskTest(unittest.TestCase):
    def test_missing_file_and_unlisted_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write(root / "bottle" / "train" / "good" / "000.png")
            _write(root / "bottle" / "test" / "good" / "extra.png")

            existing = _train_good(foreground=None)
            missing = _sample(
                filename="bottle/test/good/missing.png",
                mask=None,
                label=0,
                label_name="good",
                foreground=None,
            )
            findings = scan_disk(root, "bottle", [existing, missing])
            codes = {finding.code for finding in findings}
            self.assertIn("missing_file", codes)
            self.assertIn("unlisted_file", codes)
            unlisted = [
                finding
                for finding in findings
                if finding.code == "unlisted_file"
            ]
            self.assertEqual(
                unlisted[0].path,
                "bottle/test/good/extra.png",
            )


class BuildCleanViewTest(unittest.TestCase):
    def test_hardlinks_are_idempotent_and_foreground_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "root"
            out = base / "view"
            _write(root / "bottle" / "train" / "good" / "000.png")
            _write(root / "bottle" / "test" / "broken_small" / "000.png")
            _write(
                root
                / "bottle"
                / "ground_truth"
                / "broken_small"
                / "000_mask.png"
            )
            _write(
                root
                / "bottle"
                / "foreground"
                / "test"
                / "broken_small"
                / "000.png"
            )

            samples = [
                _train_good(foreground=None),
                _sample(),
            ]
            created = build_clean_view(root, "bottle", samples, out)
            self.assertEqual(created, 3)
            for rel in (
                "bottle/train/good/000.png",
                "bottle/test/broken_small/000.png",
                "bottle/ground_truth/broken_small/000_mask.png",
            ):
                # Anomalib は symlink を解決して root 外を除外するため、通常ファイル（同一 inode）であること。
                self.assertFalse((out / rel).is_symlink())
                self.assertTrue(os.path.samefile(out / rel, root / rel))
            self.assertFalse((out / "bottle/foreground").exists())

            self.assertEqual(build_clean_view(root, "bottle", samples, out), 0)

    def test_symlink_and_copy_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "root"
            _write(root / "bottle" / "train" / "good" / "000.png")
            samples = [_train_good(foreground=None)]
            rel = "bottle/train/good/000.png"
            self.assertEqual(
                build_clean_view(root, "bottle", samples, base / "s", link_mode="symlink"), 1
            )
            self.assertTrue((base / "s" / rel).is_symlink())
            self.assertEqual(
                build_clean_view(root, "bottle", samples, base / "s", link_mode="symlink"), 0
            )
            self.assertEqual(
                build_clean_view(root, "bottle", samples, base / "c", link_mode="copy"), 1
            )
            self.assertFalse((base / "c" / rel).is_symlink())
            self.assertFalse(os.path.samefile(base / "c" / rel, root / rel))
            with self.assertRaises(ValueError):
                build_clean_view(root, "bottle", samples, base / "x", link_mode="bogus")

    def test_regular_file_collision_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "root"
            out = base / "view"
            _write(root / "bottle" / "test" / "broken_small" / "001.png")
            _write(out / "bottle" / "test" / "broken_small" / "001.png", "collision")
            sample = _sample(
                filename="bottle/test/broken_small/001.png",
                mask=None,
            )
            with self.assertRaises(FileExistsError):
                build_clean_view(root, "bottle", [sample], out)
            self.assertEqual(
                (out / "bottle/test/broken_small/001.png").read_text(
                    encoding="utf-8"
                ),
                "collision",
            )


@unittest.skipUnless(HAS_PIL, "Pillow が無いためスキップ")
class PixelCheckTest(unittest.TestCase):
    def _save_l_image(
        self,
        path: Path,
        size: tuple[int, int],
        pixels: dict[tuple[int, int], int] | None = None,
    ) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("L", size, 0)
        for point, value in (pixels or {}).items():
            image.putpixel(point, value)
        image.save(path)
        image.close()

    def test_valid_mask_and_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_l_image(root / "bottle/test/broken_small/000.png", (4, 4))
            self._save_l_image(
                root / "bottle/ground_truth/broken_small/000_mask.png",
                (4, 4),
                {(0, 0): 255, (3, 0): 255},
            )
            self._save_l_image(
                root / "bottle/foreground/test/broken_small/000.png",
                (4, 4),
                {(x, y): 255 for x in range(4) for y in range(4)},
            )

            info, findings = check_pixels(root, _sample())
            self.assertEqual(findings, [])
            self.assertEqual(info["image_size"], [4, 4])
            self.assertAlmostEqual(info["mask_area_fraction"], 2 / 16)
            self.assertEqual(info["mask_bbox"], [0, 0, 4, 1])
            self.assertAlmostEqual(info["mask_in_foreground_fraction"], 1.0)

    def test_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_l_image(root / "bottle/test/broken_small/000.png", (4, 4))
            self._save_l_image(
                root / "bottle/ground_truth/broken_small/000_mask.png",
                (2, 2),
                {(0, 0): 255},
            )
            _, findings = check_pixels(
                root,
                _sample(foreground=None),
            )
            self.assertIn("size_mismatch", {finding.code for finding in findings})

    def test_empty_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_l_image(root / "bottle/test/broken_small/000.png", (4, 4))
            self._save_l_image(
                root / "bottle/ground_truth/broken_small/000_mask.png",
                (4, 4),
            )
            _, findings = check_pixels(
                root,
                _sample(foreground=None),
            )
            self.assertIn("empty_mask", {finding.code for finding in findings})

    def test_non_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_l_image(root / "bottle/test/broken_small/000.png", (4, 4))
            self._save_l_image(
                root / "bottle/ground_truth/broken_small/000_mask.png",
                (4, 4),
                {(0, 0): 128},
            )
            _, findings = check_pixels(
                root,
                _sample(foreground=None),
            )
            found = {finding.code for finding in findings}
            self.assertIn("non_binary_mask", found)
            self.assertNotIn("empty_mask", found)

    def test_corrupt_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "bottle/test/broken_small/000.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not an image")
            info, findings = check_pixels(
                root,
                _sample(mask=None, foreground=None, label=0, label_name="good"),
            )
            self.assertEqual(info["image_size"], None)
            self.assertEqual(findings[0].code, "corrupt_image")

    def test_foreground_overlap_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_l_image(root / "bottle/test/broken_small/000.png", (4, 4))
            self._save_l_image(
                root / "bottle/ground_truth/broken_small/000_mask.png",
                (4, 4),
                {(0, 0): 255, (3, 0): 255},
            )
            self._save_l_image(
                root / "bottle/foreground/test/broken_small/000.png",
                (4, 4),
                {(x, y): 255 for x in (0, 1) for y in range(4)},
            )
            info, findings = check_pixels(root, _sample())
            self.assertEqual(findings, [])
            self.assertAlmostEqual(info["mask_in_foreground_fraction"], 0.5)

    def test_validate_category_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_l_image(root / "bottle/train/good/000.png", (8, 8))
            self._save_l_image(
                root / "bottle/foreground/train/good/000.png",
                (8, 8),
                {(x, y): 255 for x in range(8) for y in range(8)},
            )
            self._save_l_image(root / "bottle/test/good/000.png", (8, 8))
            self._save_l_image(
                root / "bottle/foreground/test/good/000.png",
                (8, 8),
                {(x, y): 255 for x in range(8) for y in range(8)},
            )
            self._save_l_image(
                root / "bottle/test/broken_small/000.png",
                (8, 8),
            )
            self._save_l_image(
                root / "bottle/ground_truth/broken_small/000_mask.png",
                (8, 8),
                {(1, 1): 255, (2, 2): 255},
            )
            self._save_l_image(
                root / "bottle/foreground/test/broken_small/000.png",
                (8, 8),
                {(x, y): 255 for x in range(8) for y in range(8)},
            )

            _write_jsonl(
                root / "bottle" / "train.jsonl",
                [
                    {
                        "filename": "bottle/train/good/000.png",
                        "foreground": "bottle/foreground/train/good/000.png",
                        "clsname": "bottle",
                        "label": 0,
                        "label_name": "good",
                    }
                ],
            )
            _write_jsonl(
                root / "bottle" / "test.jsonl",
                [
                    {
                        "filename": "bottle/test/good/000.png",
                        "foreground": "bottle/foreground/test/good/000.png",
                        "clsname": "bottle",
                        "label": 0,
                        "label_name": "good",
                    },
                    {
                        "filename": "bottle/test/broken_small/000.png",
                        "mask": "bottle/ground_truth/broken_small/000_mask.png",
                        "foreground": "bottle/foreground/test/broken_small/000.png",
                        "clsname": "bottle",
                        "label": 1,
                        "label_name": "broken_small",
                    },
                ],
            )

            result = validate_category(root, "bottle", pixels=True)
            self.assertEqual(result["category"], "bottle")
            self.assertEqual(result["counts"]["train"], 1)
            self.assertEqual(result["counts"]["test_good"], 1)
            self.assertEqual(result["counts"]["large"], 1)
            self.assertEqual(result["counts"]["tiny"], 0)
            self.assertEqual(result["pixels_checked"], 3)
            self.assertEqual(result["n_errors"], 0)
            self.assertEqual(result["image_sizes"], {"8x8": 3})
            self.assertIn("broken_small", result["mask_stats"])
            self.assertAlmostEqual(
                result["mask_stats"]["broken_small"]["mask_area_fraction"]["median"],
                2 / 64,
            )


if __name__ == "__main__":
    unittest.main()
