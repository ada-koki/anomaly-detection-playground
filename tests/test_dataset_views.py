"""adbench.dataset_views と run_anomalib / make_dataset_view の連携テスト。

torch / anomalib は import しない。マスク生成と検証に Pillow を使う。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Pillow はマスク生成にだけ使う（未導入なら該当テストをスキップ）
    from PIL import Image
except Exception:  # pragma: no cover - Pillow 未導入環境向け
    Image = None  # type: ignore[assignment]

from adbench import dataset_views  # noqa: E402  (sys.path 調整のため後置)
from adbench import protocol as protocol_module  # noqa: E402

VISA_PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "visa_v1.json"
MVTEC_AD2_PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "mvtec_ad2_v1.json"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"モジュールを読み込めません: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN_ANOMALIB = _load_module(
    "run_anomalib_for_dataset_views_test",
    REPO_ROOT / "scripts" / "run_anomalib.py",
)
MAKE_DATASET_VIEW = _load_module(
    "make_dataset_view_for_dataset_views_test",
    REPO_ROOT / "scripts" / "make_dataset_view.py",
)


def _require_pillow() -> None:
    if Image is None:
        raise unittest.SkipTest("Pillow が無いためスキップします")


def _write_text(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_mask(path: Path, values: list[int]) -> None:
    """0/1 の L 画像マスクを書く（二値化の検証用）。"""
    _require_pillow()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (len(values), 1))
    image.putdata(values)
    image.save(path, format="PNG")


VISA_CSV_HEADER = "object,split,label,image,mask\n"


def _make_visa_raw(root: Path) -> None:
    """合成 VisA 生配置（candle と他カテゴリ foil を混在させる）。"""
    rows = [
        "candle,train,normal,candle/Data/Images/Normal/0001.JPG,",
        "candle,train,normal,candle/Data/Images/Normal/0002.JPG,",
        "candle,test,normal,candle/Data/Images/Normal/0003.JPG,",
        "candle,test,anomaly,candle/Data/Images/Anomaly/0004.JPG,"
        "candle/Data/Masks/Anomaly/0004.png",
        "foil,train,normal,foil/Data/Images/Normal/0001.JPG,",
    ]
    _write_text(root / "split_csv" / "1cls.csv", VISA_CSV_HEADER + "\n".join(rows) + "\n")
    for name in ("0001.JPG", "0002.JPG", "0003.JPG"):
        _write_bytes(root / "candle" / "Data" / "Images" / "Normal" / name, b"image")
    _write_bytes(
        root / "candle" / "Data" / "Images" / "Anomaly" / "0004.JPG", b"anomaly"
    )
    _write_mask(
        root / "candle" / "Data" / "Masks" / "Anomaly" / "0004.png", [0, 1, 1, 0]
    )
    _write_bytes(root / "foil" / "Data" / "Images" / "Normal" / "0001.JPG", b"foil")


def _make_mvtec_ad2_raw(root: Path) -> None:
    """合成 MVTec AD 2 生配置（validation / test_private も置く）。"""
    category = root / "cylinder"
    for name in ("000.png", "001.png"):
        _write_bytes(category / "train" / "good" / name, b"train")
    _write_text(category / "train" / "good" / "notes.txt", "ignore")
    _write_bytes(category / "validation" / "good" / "v000.png", b"val")
    _write_bytes(category / "test_public" / "good" / "g000.png", b"good")
    _write_bytes(category / "test_public" / "bad" / "b000.png", b"bad")
    _write_bytes(
        category / "test_public" / "ground_truth" / "bad" / "b000_mask.png", b"mask"
    )
    _write_bytes(category / "test_private" / "bad" / "p000.png", b"private")


def _call_validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Any:
    """validate_args のシグネチャ差異（parser を受けるか）を吸収して呼ぶ。"""
    signature = inspect.signature(RUN_ANOMALIB.validate_args)
    names = set(signature.parameters)
    if {"parser", "args"} <= names:
        return RUN_ANOMALIB.validate_args(parser=parser, args=args)
    if len(signature.parameters) >= 2:
        return RUN_ANOMALIB.validate_args(parser, args)
    return RUN_ANOMALIB.validate_args(args)


class _ViewTestCase(unittest.TestCase):
    """tempfile 上に生配置とビュー出力先を用意する共通ベース。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.raw = self.base / "raw"
        self.out = self.base / "view"

    def tearDown(self) -> None:
        self._tmp.cleanup()


class VisaViewTest(_ViewTestCase):
    def setUp(self) -> None:
        super().setUp()
        _make_visa_raw(self.raw)

    def test_build_visa_view_creates_mvtec_layout(self) -> None:
        _require_pillow()
        info = dataset_views.build_visa_view(self.raw, "candle", self.out)

        self.assertEqual(info["dataset"], "visa")
        self.assertEqual(info["category"], "candle")
        self.assertEqual(info["link_mode"], "hardlink")
        self.assertEqual(info["source"], "split_csv/1cls.csv")
        self.assertEqual(info["excluded"], [])
        self.assertEqual(
            info["counts"],
            {"train_good": 2, "test_good": 1, "test_bad": 1, "masks": 1},
        )

        base = self.out / "visa_pytorch" / "candle"
        self.assertEqual(
            sorted(path.name for path in (base / "train" / "good").iterdir()),
            ["0001.JPG", "0002.JPG"],
        )
        self.assertEqual(
            [path.name for path in (base / "test" / "good").iterdir()],
            ["0003.JPG"],
        )
        self.assertEqual(
            [path.name for path in (base / "test" / "bad").iterdir()],
            ["0004.JPG"],
        )

        # マスクは画像 stem と同じ名前で、0/255 に二値化されている。
        mask_path = base / "ground_truth" / "bad" / "0004.png"
        self.assertTrue(mask_path.is_file())
        with Image.open(mask_path) as mask:
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (4, 1))
            self.assertEqual(list(mask.tobytes()), [0, 255, 255, 0])

        # 他カテゴリの行は取り込まない。
        self.assertFalse((base / "train" / "good" / "foil").exists())
        self.assertEqual(
            sorted(path.name for path in (self.out / "visa_pytorch").iterdir()),
            ["candle"],
        )

    def test_build_visa_view_uses_hardlink(self) -> None:
        dataset_views.build_visa_view(self.raw, "candle", self.out)
        target = (
            self.out / "visa_pytorch" / "candle" / "train" / "good" / "0001.JPG"
        )
        source = self.raw / "candle" / "Data" / "Images" / "Normal" / "0001.JPG"
        self.assertTrue(os.path.samefile(target, source))

    def test_build_visa_view_rerun_skips_unchanged(self) -> None:
        _require_pillow()
        first = dataset_views.build_visa_view(self.raw, "candle", self.out)
        target = (
            self.out / "visa_pytorch" / "candle" / "train" / "good" / "0001.JPG"
        )
        before = target.stat()

        second = dataset_views.build_visa_view(self.raw, "candle", self.out)
        after = target.stat()

        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(before.st_ino, after.st_ino)
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)

    def test_build_visa_view_copy_mode_detects_conflict(self) -> None:
        _require_pillow()
        dataset_views.build_visa_view(
            self.raw, "candle", self.out, link_mode="copy"
        )
        target = (
            self.out / "visa_pytorch" / "candle" / "train" / "good" / "0001.JPG"
        )
        source = self.raw / "candle" / "Data" / "Images" / "Normal" / "0001.JPG"
        self.assertFalse(os.path.samefile(target, source))
        self.assertEqual(target.read_bytes(), b"image")

        source.write_bytes(b"image-changed")
        with self.assertRaises(FileExistsError):
            dataset_views.build_visa_view(
                self.raw, "candle", self.out, link_mode="copy"
            )

        dataset_views.build_visa_view(
            self.raw, "candle", self.out, link_mode="copy", overwrite=True
        )
        self.assertEqual(target.read_bytes(), b"image-changed")

    def test_build_visa_view_mask_conflict(self) -> None:
        _require_pillow()
        dataset_views.build_visa_view(self.raw, "candle", self.out)
        mask_path = (
            self.out / "visa_pytorch" / "candle" / "ground_truth" / "bad" / "0004.png"
        )
        _write_mask(mask_path, [255, 255, 255, 255])

        with self.assertRaises(FileExistsError):
            dataset_views.build_visa_view(self.raw, "candle", self.out)

        dataset_views.build_visa_view(
            self.raw, "candle", self.out, overwrite=True
        )
        with Image.open(mask_path) as mask:
            self.assertEqual(list(mask.tobytes()), [0, 255, 255, 0])

    def test_build_visa_view_rejects_invalid_link_mode(self) -> None:
        with self.assertRaises(ValueError):
            dataset_views.build_visa_view(
                self.raw, "candle", self.out, link_mode="symlink"
            )


class VisaSplitCsvTest(_ViewTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.csv_path = self.raw / "split_csv" / "1cls.csv"

    def test_read_visa_split_reads_only_target_category(self) -> None:
        _make_visa_raw(self.raw)
        rows = dataset_views.read_visa_split(self.raw, "candle")
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [(row.split, row.label) for row in rows],
            [
                ("train", "good"),
                ("train", "good"),
                ("test", "good"),
                ("test", "bad"),
            ],
        )
        self.assertIsNone(rows[0].mask)
        self.assertEqual(rows[3].mask, "candle/Data/Masks/Anomaly/0004.png")

    def test_read_visa_split_rejects_parent_traversal(self) -> None:
        _write_text(
            self.csv_path,
            VISA_CSV_HEADER + "candle,train,normal,../outside.JPG,\n",
        )
        with self.assertRaises(ValueError):
            dataset_views.read_visa_split(self.raw, "candle")

    def test_read_visa_split_rejects_absolute_path(self) -> None:
        _write_text(
            self.csv_path,
            VISA_CSV_HEADER + "candle,train,normal,/abs/0001.JPG,\n",
        )
        with self.assertRaises(ValueError):
            dataset_views.read_visa_split(self.raw, "candle")

    def test_read_visa_split_requires_mask_for_anomaly(self) -> None:
        _write_text(
            self.csv_path,
            VISA_CSV_HEADER + "candle,test,anomaly,candle/Data/Images/Anomaly/0004.JPG,\n",
        )
        with self.assertRaises(ValueError):
            dataset_views.read_visa_split(self.raw, "candle")

    def test_read_visa_split_rejects_unknown_label_and_split(self) -> None:
        _write_text(
            self.csv_path, VISA_CSV_HEADER + "candle,train,unknown,x/0001.JPG,\n"
        )
        with self.assertRaises(ValueError):
            dataset_views.read_visa_split(self.raw, "candle")

        _write_text(
            self.csv_path, VISA_CSV_HEADER + "candle,val,normal,x/0001.JPG,\n"
        )
        with self.assertRaises(ValueError):
            dataset_views.read_visa_split(self.raw, "candle")

    def test_read_visa_split_requires_csv(self) -> None:
        with self.assertRaises(FileNotFoundError):
            dataset_views.read_visa_split(self.raw, "candle")


class MvtecAd2ViewTest(_ViewTestCase):
    def setUp(self) -> None:
        super().setUp()
        _make_mvtec_ad2_raw(self.raw)

    def test_build_mvtec_ad2_view(self) -> None:
        info = dataset_views.build_mvtec_ad2_view(self.raw, "cylinder", self.out)

        self.assertEqual(info["dataset"], "mvtec_ad2")
        self.assertEqual(
            info["counts"],
            {"train_good": 2, "test_good": 1, "test_bad": 1, "masks": 1},
        )
        self.assertEqual(info["official_validation_good"], 1)
        self.assertEqual(
            info["excluded"], ["validation", "test_private", "test_private_mixed"]
        )

        base = self.out / "cylinder"
        self.assertEqual(
            sorted(path.name for path in (base / "train" / "good").iterdir()),
            ["000.png", "001.png"],
        )
        self.assertEqual(
            [path.name for path in (base / "test" / "good").iterdir()],
            ["g000.png"],
        )
        self.assertEqual(
            [path.name for path in (base / "test" / "bad").iterdir()],
            ["b000.png"],
        )
        self.assertTrue((base / "ground_truth" / "bad" / "b000_mask.png").is_file())

        # validation/ と test_private*/ はビューに含めない。
        self.assertFalse((base / "validation").exists())
        self.assertFalse((base / "test_private").exists())

        target = base / "train" / "good" / "000.png"
        source = self.raw / "cylinder" / "train" / "good" / "000.png"
        self.assertTrue(os.path.samefile(target, source))

        mask_target = base / "ground_truth" / "bad" / "b000_mask.png"
        mask_source = (
            self.raw / "cylinder" / "test_public" / "ground_truth" / "bad" / "b000_mask.png"
        )
        self.assertTrue(os.path.samefile(mask_target, mask_source))

    def test_build_mvtec_ad2_view_requires_mask(self) -> None:
        (
            self.raw
            / "cylinder"
            / "test_public"
            / "ground_truth"
            / "bad"
            / "b000_mask.png"
        ).unlink()
        with self.assertRaises(FileNotFoundError):
            dataset_views.build_mvtec_ad2_view(self.raw, "cylinder", self.out)


class ValidateViewTest(_ViewTestCase):
    def test_validate_view_visa_detects_raw_layout(self) -> None:
        _make_visa_raw(self.raw)
        problems = dataset_views.validate_view("visa", self.raw, "candle")
        self.assertTrue(problems)
        self.assertTrue(any("make_dataset_view.py" in item for item in problems))

        _require_pillow()
        dataset_views.build_visa_view(self.raw, "candle", self.out)
        self.assertEqual(dataset_views.validate_view("visa", self.out, "candle"), [])

    def test_validate_view_mvtec_ad2_detects_raw_layout(self) -> None:
        _make_mvtec_ad2_raw(self.raw)
        problems = dataset_views.validate_view("mvtec_ad2", self.raw, "cylinder")
        self.assertTrue(problems)
        self.assertTrue(any("make_dataset_view.py" in item for item in problems))

        dataset_views.build_mvtec_ad2_view(self.raw, "cylinder", self.out)
        self.assertEqual(
            dataset_views.validate_view("mvtec_ad2", self.out, "cylinder"), []
        )

    def test_validate_view_rejects_unknown_dataset(self) -> None:
        with self.assertRaises(ValueError):
            dataset_views.validate_view("mvtec_ad", self.raw, "bottle")


class ViewManifestTest(_ViewTestCase):
    def test_write_view_manifest_has_no_absolute_paths(self) -> None:
        _require_pillow()
        _make_visa_raw(self.raw)
        info = dataset_views.build_visa_view(self.raw, "candle", self.out)
        path = dataset_views.write_view_manifest(self.out, info)

        self.assertEqual(path, self.out / "view_manifest.json")
        text = path.read_text(encoding="utf-8")
        self.assertNotIn(str(self.base), text)
        self.assertNotIn("raw_root", text)

        payload = json.loads(text)
        self.assertEqual(payload["dataset"], "visa")
        self.assertEqual(payload["category"], "candle")
        self.assertEqual(payload["link_mode"], "hardlink")
        self.assertEqual(payload["counts"]["test_bad"], 1)
        self.assertEqual(payload["excluded"], [])
        self.assertTrue(payload["notes"])

    def test_write_view_manifest_records_validation_count(self) -> None:
        _make_mvtec_ad2_raw(self.raw)
        info = dataset_views.build_mvtec_ad2_view(self.raw, "cylinder", self.out)
        path = dataset_views.write_view_manifest(self.out, info)
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)

        self.assertEqual(payload["official_validation_good"], 1)
        self.assertEqual(
            payload["excluded"], ["validation", "test_private", "test_private_mixed"]
        )
        self.assertNotIn(str(self.base), text)


class MakeDatasetViewCliTest(_ViewTestCase):
    def test_cli_writes_visa_view_and_manifest(self) -> None:
        _require_pillow()
        _make_visa_raw(self.raw)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = MAKE_DATASET_VIEW.main(
                [
                    "--dataset",
                    "visa",
                    "--raw-root",
                    str(self.raw),
                    "--category",
                    "candle",
                    "--out",
                    str(self.out),
                ]
            )
        self.assertEqual(code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["dataset"], "visa")
        self.assertEqual(summary["counts"]["test_bad"], 1)
        self.assertTrue((self.out / "view_manifest.json").is_file())

    def test_cli_returns_error_for_missing_raw_root(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = MAKE_DATASET_VIEW.main(
                [
                    "--dataset",
                    "mvtec_ad2",
                    "--raw-root",
                    str(self.raw / "missing"),
                    "--category",
                    "cylinder",
                    "--out",
                    str(self.out),
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("エラー", stderr.getvalue())


class RunAnomalibViewValidationTest(_ViewTestCase):
    def _build_args(
        self, argv: list[str]
    ) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
        # 既存テスト（test_mvtec4k_cli）と同じく、プロトコル適用後の値で検証する。
        args, _, _, _ = RUN_ANOMALIB.parse_and_resolve(argv)
        return RUN_ANOMALIB.build_parser(), args

    def test_validate_args_accepts_visa_view(self) -> None:
        _require_pillow()
        _make_visa_raw(self.raw)
        dataset_views.build_visa_view(self.raw, "candle", self.out)
        parser, args = self._build_args(
            [
                "--protocol",
                str(VISA_PROTOCOL_PATH),
                "--dataset",
                "visa",
                "--dataset-root",
                str(self.out),
                "--category",
                "candle",
            ]
        )
        # SystemExit にならなければよい。
        _call_validate_args(parser, args)

    def test_validate_args_accepts_mvtec_ad2_view(self) -> None:
        _make_mvtec_ad2_raw(self.raw)
        dataset_views.build_mvtec_ad2_view(self.raw, "cylinder", self.out)
        parser, args = self._build_args(
            [
                "--protocol",
                str(MVTEC_AD2_PROTOCOL_PATH),
                "--dataset",
                "mvtec_ad2",
                "--dataset-root",
                str(self.out),
                "--category",
                "cylinder",
            ]
        )
        _call_validate_args(parser, args)

    def test_validate_args_rejects_visa_raw_layout(self) -> None:
        _make_visa_raw(self.raw)
        parser, args = self._build_args(
            [
                "--protocol",
                str(VISA_PROTOCOL_PATH),
                "--dataset",
                "visa",
                "--dataset-root",
                str(self.raw),
                "--category",
                "candle",
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _call_validate_args(parser, args)

    def test_validate_args_rejects_mvtec_ad2_raw_layout(self) -> None:
        _make_mvtec_ad2_raw(self.raw)
        parser, args = self._build_args(
            [
                "--protocol",
                str(MVTEC_AD2_PROTOCOL_PATH),
                "--dataset",
                "mvtec_ad2",
                "--dataset-root",
                str(self.raw),
                "--category",
                "cylinder",
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _call_validate_args(parser, args)


class ProtocolDatasetsTest(unittest.TestCase):
    def test_allowed_datasets_include_new_views(self) -> None:
        allowed = getattr(protocol_module, "_ALLOWED_DATASETS", set())
        self.assertIn("visa", allowed)
        self.assertIn("mvtec_ad2", allowed)

    def test_protocol_files(self) -> None:
        for path, dataset_name in (
            (VISA_PROTOCOL_PATH, "visa"),
            (MVTEC_AD2_PROTOCOL_PATH, "mvtec_ad2"),
        ):
            self.assertTrue(path.is_file(), str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], dataset_name)
            self.assertEqual(payload["val_split_mode"], "from_test")
            self.assertEqual(payload["val_split_ratio"], 0.2)
            self.assertEqual(payload["seed"], 42)

    def test_load_protocol_accepts_new_datasets(self) -> None:
        loader = getattr(protocol_module, "load_protocol", None)
        if loader is None:
            self.skipTest("load_protocol が見つかりません")
        for path in (VISA_PROTOCOL_PATH, MVTEC_AD2_PROTOCOL_PATH):
            loader(path)


if __name__ == "__main__":
    unittest.main()
