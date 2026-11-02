"""run_anomalib の MVTec-4K ガードと validate_mvtec4k CLI のテスト。

Pillow を使わないよう、CLI の検証は --skip-pixels とダミーの画像ファイルで
行う。torch / anomalib は import しない。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "mvtec_ad_v1.json"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"モジュールを読み込めません: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN_ANOMALIB = _load_module(
    "run_anomalib_for_mvtec4k_test",
    REPO_ROOT / "scripts" / "run_anomalib.py",
)
VALIDATE_MVTEC4K = _load_module(
    "validate_mvtec4k_for_test",
    REPO_ROOT / "scripts" / "validate_mvtec4k.py",
)


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _prepare_run_anomalib_root(root: Path, *, foreground: bool) -> None:
    for split in ("train", "test"):
        (root / "bottle" / split / "good").mkdir(parents=True, exist_ok=True)
    if foreground:
        (root / "bottle" / "foreground").mkdir(parents=True, exist_ok=True)


def _prepare_valid_mvtec4k(root: Path) -> None:
    _write(root / "bottle" / "train" / "good" / "000.png")
    _write(root / "bottle" / "test" / "good" / "000.png")
    _write(root / "bottle" / "test" / "broken_small" / "000.png")
    _write(
        root
        / "bottle"
        / "ground_truth"
        / "broken_small"
        / "000_mask.png"
    )
    _write_jsonl(
        root / "bottle" / "train.jsonl",
        [
            {
                "filename": "bottle/train/good/000.png",
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
                "clsname": "bottle",
                "label": 0,
                "label_name": "good",
            },
            {
                "filename": "bottle/test/broken_small/000.png",
                "mask": "bottle/ground_truth/broken_small/000_mask.png",
                "clsname": "bottle",
                "label": 1,
                "label_name": "broken_small",
            },
        ],
    )


class RunAnomalibForegroundGuardTest(unittest.TestCase):
    def _validate(self, root: Path) -> Path:
        argv = [
            "--protocol",
            str(PROTOCOL_PATH),
            "--dataset",
            "mvtec_4k",
            "--dataset-root",
            str(root),
            "--category",
            "bottle",
        ]
        args, _, _, _ = RUN_ANOMALIB.parse_and_resolve(argv)
        return RUN_ANOMALIB.validate_args(args, RUN_ANOMALIB.build_parser())

    def test_foreground_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _prepare_run_anomalib_root(root, foreground=True)
            with self.assertRaises(SystemExit):
                self._validate(root)

    def test_without_foreground_dir_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _prepare_run_anomalib_root(root, foreground=False)
            self.assertEqual(self._validate(root), root)


class ValidateMvtec4kCliTest(unittest.TestCase):
    def test_exit_code_zero_for_valid_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _prepare_valid_mvtec4k(root)
            code = VALIDATE_MVTEC4K.main(
                ["--root", str(root), "--skip-pixels"]
            )
            self.assertEqual(code, 0)

    def test_exit_code_two_for_broken_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write(root / "bottle" / "train.jsonl", "not-json\n")
            _write_jsonl(root / "bottle" / "test.jsonl", [])
            code = VALIDATE_MVTEC4K.main(
                ["--root", str(root), "--category", "bottle", "--skip-pixels"]
            )
            self.assertEqual(code, 2)

    def test_exit_code_one_when_no_categories_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = VALIDATE_MVTEC4K.main(["--root", str(root), "--skip-pixels"])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
