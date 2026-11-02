"""MVTec-4K のマニフェスト検証とピクセル整合性チェック。

MVTec-4K は MVTec AD を高解像度化した派生データで、``ground_truth`` と
``foreground`` を独自レイアウトで持つ。本モジュールは Anomalib の MVTecAD
データモジュールへ生配置をそのまま渡さず、まず構造・ラベル・マスク・
参照ファイルを検証するために使う。

重い依存（Pillow）はピクセル検査を行う関数内で遅延 import する。
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

# 配布物に含まれる 7 カテゴリ（確認済みの範囲）。
CATEGORIES: tuple[str, ...] = (
    "bottle",
    "capsule",
    "grid",
    "hazelnut",
    "screw",
    "transistor",
    "wood",
)

# 論文（HiAD, arXiv 2508.12931）の表に記載された枚数。
# これは配布物を実測した値ではなく「参考値」である。実データと差があっても
# 検証を失敗させず、warning として報告する。
PAPER_COUNTS: dict[str, dict[str, int]] = {
    "bottle": {"train": 146, "test_good": 50, "large": 47, "tiny": 68},
    "capsule": {"train": 179, "test_good": 63, "large": 109, "tiny": 88},
    "grid": {"train": 230, "test_good": 55, "large": 57, "tiny": 37},
    "hazelnut": {"train": 351, "test_good": 80, "large": 70, "tiny": 73},
    "screw": {"train": 260, "test_good": 84, "large": 48, "tiny": 55},
    "transistor": {"train": 183, "test_good": 87, "large": 40, "tiny": 34},
    "wood": {"train": 217, "test_good": 48, "large": 55, "tiny": 41},
}

# 画像ファイルとして扱う拡張子。大文字小文字は区別しない。
IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)

_REQUIRED_MANIFEST_KEYS = ("filename", "clsname", "label", "label_name")


@dataclass(frozen=True)
class Mvtec4kSample:
    """MVTec-4K の jsonl 1 行に対応するサンプル。"""

    split: str
    filename: str
    mask: str | None
    foreground: str | None
    clsname: str
    label: int
    label_name: str
    line: int


@dataclass(frozen=True)
class Finding:
    """検証で見つかった問題。"""

    severity: str
    code: str
    message: str
    path: str | None = None


def _posix_name(filename: str) -> str:
    """マニフェスト上のパスからファイル名部分を取り出す。"""
    return PurePosixPath(filename).name


def _posix_stem(filename: str) -> str:
    """マニフェスト上のパスから拡張子を除いたファイル名を取り出す。"""
    return PurePosixPath(filename).stem


def _finding_to_dict(finding: Finding) -> dict[str, Any]:
    return {
        "severity": finding.severity,
        "code": finding.code,
        "message": finding.message,
        "path": finding.path,
    }


def _parse_manifest_file(
    root: Path,
    category: str,
    split: str,
) -> tuple[list[Mvtec4kSample], list[Finding]]:
    """1 つの jsonl を読み、サンプルと検出結果を返す。"""
    path = root / category / f"{split}.jsonl"
    if not path.is_file():
        return [], [
            Finding(
                "error",
                "manifest_missing",
                f"{split}.jsonl が見つかりません: {path}",
                str(path),
            )
        ]

    samples: list[Mvtec4kSample] = []
    findings: list[Finding] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        return [], [
            Finding(
                "error",
                "manifest_unreadable",
                f"{split}.jsonl を開けません: {type(error).__name__}: {error}",
                str(path),
            )
        ]

    with handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                findings.append(
                    Finding(
                        "error",
                        "invalid_json",
                        f"{split}.jsonl の {line_no} 行目が JSON として不正です: {error.msg}",
                        str(path),
                    )
                )
                continue

            if not isinstance(payload, dict):
                findings.append(
                    Finding(
                        "error",
                        "invalid_json_object",
                        f"{split}.jsonl の {line_no} 行目が JSON オブジェクトではありません",
                        str(path),
                    )
                )
                continue

            missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in payload]
            if missing:
                findings.append(
                    Finding(
                        "error",
                        "missing_key",
                        f"{split}.jsonl の {line_no} 行目に必須キーがありません: "
                        + ", ".join(missing),
                        str(path),
                    )
                )
                continue

            filename = payload["filename"]
            clsname = payload["clsname"]
            label = payload["label"]
            label_name = payload["label_name"]
            mask = payload.get("mask")
            foreground = payload.get("foreground")

            field_error = False
            if not isinstance(filename, str) or not filename:
                findings.append(
                    Finding(
                        "error",
                        "invalid_field",
                        f"{split}.jsonl の {line_no} 行目の filename が不正です",
                        str(path),
                    )
                )
                field_error = True
            if not isinstance(clsname, str) or not clsname:
                findings.append(
                    Finding(
                        "error",
                        "invalid_field",
                        f"{split}.jsonl の {line_no} 行目の clsname が不正です",
                        str(path),
                    )
                )
                field_error = True
            if not isinstance(label, int) or isinstance(label, bool):
                findings.append(
                    Finding(
                        "error",
                        "invalid_field",
                        f"{split}.jsonl の {line_no} 行目の label が整数ではありません",
                        str(path),
                    )
                )
                field_error = True
            if not isinstance(label_name, str) or not label_name:
                findings.append(
                    Finding(
                        "error",
                        "invalid_field",
                        f"{split}.jsonl の {line_no} 行目の label_name が不正です",
                        str(path),
                    )
                )
                field_error = True
            if mask is not None and not isinstance(mask, str):
                findings.append(
                    Finding(
                        "error",
                        "invalid_field",
                        f"{split}.jsonl の {line_no} 行目の mask が文字列ではありません",
                        str(path),
                    )
                )
                field_error = True
            if foreground is not None and not isinstance(foreground, str):
                findings.append(
                    Finding(
                        "error",
                        "invalid_field",
                        f"{split}.jsonl の {line_no} 行目の foreground が文字列ではありません",
                        str(path),
                    )
                )
                field_error = True

            if field_error:
                continue

            samples.append(
                Mvtec4kSample(
                    split=split,
                    filename=filename,
                    mask=mask,
                    foreground=foreground,
                    clsname=clsname,
                    label=label,
                    label_name=label_name,
                    line=line_no,
                )
            )
    return samples, findings


def load_manifest(root: Path, category: str) -> tuple[list[Mvtec4kSample], list[Finding]]:
    """``{category}/train.jsonl`` と ``test.jsonl`` を読み込む。

    ファイル欠如・JSON 不正・必須キー欠如は error の Finding にし、
    読める行だけを使って処理を継続する。
    """
    root = Path(root)
    samples: list[Mvtec4kSample] = []
    findings: list[Finding] = []
    for split in ("train", "test"):
        split_samples, split_findings = _parse_manifest_file(root, category, split)
        samples.extend(split_samples)
        findings.extend(split_findings)
    return samples, findings


def validate_entries(samples: Sequence[Mvtec4kSample], category: str) -> list[Finding]:
    """マニフェスト内のラベル・パス・マスク対応を検証する。

    ディスクの存在確認は行わない純粋な検証である。
    """
    findings: list[Finding] = []
    seen_filenames: dict[str, int] = {}

    for sample in samples:
        if sample.clsname != category:
            findings.append(
                Finding(
                    "error",
                    "category_mismatch",
                    f"clsname がカテゴリと一致しません: {sample.clsname} != {category}",
                    sample.filename,
                )
            )

        if sample.split not in ("train", "test"):
            findings.append(
                Finding(
                    "error",
                    "invalid_split",
                    f"split が train/test ではありません: {sample.split}",
                    sample.filename,
                )
            )

        if sample.label not in (0, 1):
            findings.append(
                Finding(
                    "error",
                    "invalid_label",
                    f"label が 0/1 ではありません: {sample.label}",
                    sample.filename,
                )
            )
        elif (sample.label_name == "good") != (sample.label == 0):
            findings.append(
                Finding(
                    "error",
                    "label_name_mismatch",
                    "label_name == 'good' と label == 0 が対応していません",
                    sample.filename,
                )
            )

        if sample.split == "train" and (sample.label_name != "good" or sample.label != 0):
            findings.append(
                Finding(
                    "error",
                    "train_label_not_good",
                    "train は good（label=0, label_name='good'）だけを含む必要があります",
                    sample.filename,
                )
            )

        name = _posix_name(sample.filename)
        if not name:
            findings.append(
                Finding(
                    "error",
                    "invalid_filename",
                    "filename からファイル名を取り出せません",
                    sample.filename,
                )
            )
            continue

        expected_filename = (
            f"{category}/{sample.split}/{sample.label_name}/{name}"
        )
        if sample.filename != expected_filename:
            findings.append(
                Finding(
                    "error",
                    "filename_path_mismatch",
                    f"filename が期待値と一致しません: {expected_filename}",
                    sample.filename,
                )
            )

        if sample.label == 1:
            stem = _posix_stem(sample.filename)
            expected_mask = (
                f"{category}/ground_truth/{sample.label_name}/{stem}_mask.png"
            )
            if sample.mask is None:
                findings.append(
                    Finding(
                        "error",
                        "mask_required",
                        "label=1 のサンプルには mask が必要です",
                        sample.filename,
                    )
                )
            elif sample.mask != expected_mask:
                findings.append(
                    Finding(
                        "error",
                        "mask_path_mismatch",
                        f"mask が期待値と一致しません: {expected_mask}",
                        sample.mask,
                    )
                )
        elif sample.label == 0 and sample.mask is not None:
            findings.append(
                Finding(
                    "error",
                    "unexpected_mask",
                    "label=0 のサンプルに mask が指定されています",
                    sample.mask,
                )
            )

        if sample.foreground is not None:
            expected_foreground = (
                f"{category}/foreground/{sample.split}/{sample.label_name}/{name}"
            )
            if sample.foreground != expected_foreground:
                findings.append(
                    Finding(
                        "error",
                        "foreground_path_mismatch",
                        f"foreground が期待値と一致しません: {expected_foreground}",
                        sample.foreground,
                    )
                )

        if sample.filename in seen_filenames:
            findings.append(
                Finding(
                    "error",
                    "duplicate_filename",
                    "filename が重複しています"
                    f"（初出行: {seen_filenames[sample.filename]} 行目）",
                    sample.filename,
                )
            )
        else:
            seen_filenames[sample.filename] = sample.line

    return findings


def scan_disk(
    root: Path,
    category: str,
    samples: Sequence[Mvtec4kSample],
) -> list[Finding]:
    """参照ファイルの欠如と、マニフェストに無い画像を検出する。"""
    root = Path(root)
    findings: list[Finding] = []
    known_paths: set[str] = set()
    checked_missing: set[str] = set()

    for sample in samples:
        for reference in (sample.filename, sample.mask, sample.foreground):
            if reference is None:
                continue
            known_paths.add(reference)
            if reference in checked_missing:
                continue
            if not (root / reference).is_file():
                checked_missing.add(reference)
                findings.append(
                    Finding(
                        "error",
                        "missing_file",
                        "マニフェストが参照するファイルが見つかりません",
                        reference,
                    )
                )

    for split in ("train", "test", "ground_truth"):
        base = root / category / split
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in known_paths:
                findings.append(
                    Finding(
                        "warning",
                        "unlisted_file",
                        "マニフェストに記載の無い画像ファイルがあります",
                        relative,
                    )
                )
    return findings


def count_by_condition(samples: Sequence[Mvtec4kSample]) -> dict[str, Any]:
    """分割・ラベル別の件数を数える。"""
    by_label_name: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    train = 0
    test_good = 0
    tiny = 0
    large = 0

    for sample in samples:
        by_label_name[sample.split][sample.label_name] += 1
        if sample.split == "train":
            train += 1
        elif sample.split == "test":
            if sample.label_name == "good":
                test_good += 1
            elif sample.label_name == "tiny":
                tiny += 1
            elif sample.label == 1:
                large += 1

    normalized_by_label = {
        split: dict(sorted(counts.items()))
        for split, counts in sorted(by_label_name.items())
    }
    return {
        "train": train,
        "test_good": test_good,
        "tiny": tiny,
        "large": large,
        "by_label_name": normalized_by_label,
    }


def compare_with_paper(category: str, counts: dict[str, Any]) -> list[Finding]:
    """論文表の参考値と件数を比較する。

    配布物と論文表の差はあり得るため error にはせず、warning にとどめる。
    """
    paper = PAPER_COUNTS.get(category)
    if paper is None:
        return [
            Finding(
                "warning",
                "unknown_category",
                f"論文表に無いカテゴリのため件数を比較できません: {category}",
                None,
            )
        ]

    findings: list[Finding] = []
    for key in ("train", "test_good", "large", "tiny"):
        actual = counts.get(key)
        expected = paper[key]
        if actual != expected:
            findings.append(
                Finding(
                    "warning",
                    "paper_count_mismatch",
                    f"{key} の件数が論文表の参考値と異なります: "
                    f"実データ={actual}, 論文表={expected}",
                    category,
                )
            )
    return findings


def _summarize(values: Sequence[float], *, with_max: bool) -> dict[str, Any]:
    if not values:
        summary: dict[str, Any] = {"min": None, "median": None}
        if with_max:
            summary["max"] = None
        summary["n"] = 0
        return summary
    summary = {
        "min": min(values),
        "median": statistics.median(values),
    }
    if with_max:
        summary["max"] = max(values)
    summary["n"] = len(values)
    return summary


def _count_mask_in_foreground(
    mask_binary: Any,
    foreground_path: Path,
    image_module: Any,
) -> int | None:
    """二値マスクのうち foreground>0 の内側にある画素数を返す。"""
    from PIL import ImageChops

    with image_module.open(foreground_path) as foreground:
        foreground.load()
        foreground_size = tuple(int(value) for value in foreground.size)
        if foreground_size != tuple(int(value) for value in mask_binary.size):
            return None
        foreground_l = foreground.convert("L")
        try:
            foreground_binary = foreground_l.point(
                lambda value: 255 if value > 0 else 0
            )
            try:
                product = ImageChops.multiply(mask_binary, foreground_binary)
                try:
                    histogram = product.histogram()
                finally:
                    product.close()
            finally:
                foreground_binary.close()
        finally:
            foreground_l.close()
    return sum(histogram[1:])


def check_pixels(
    root: Path,
    sample: Mvtec4kSample,
    *,
    image_module: Any = None,
) -> tuple[dict[str, Any], list[Finding]]:
    """1 サンプルの画像・マスク・foreground を読み、サイズと画素を記録する。

    4096x4096 画像でメモリを増やしすぎないよう、画像は 1 枚ずつ開いて
    処理後に close する。Pillow は関数内で import する。
    """
    root = Path(root)
    if image_module is None:
        from PIL import Image as pil_image_module

        image_module = pil_image_module

    findings: list[Finding] = []
    info: dict[str, Any] = {
        "filename": sample.filename,
        "split": sample.split,
        "label": sample.label,
        "label_name": sample.label_name,
        "image_size": None,
        "image_mode": None,
        "mask_path": None,
        "mask_size": None,
        "mask_mode": None,
        "mask_area_fraction": None,
        "mask_bbox": None,
        "mask_non_binary_pixels": None,
        "mask_in_foreground_fraction": None,
        "foreground_path": sample.foreground,
        "foreground_size": None,
        "foreground_mode": None,
    }

    try:
        with image_module.open(root / sample.filename) as image:
            image.load()
            image_size = tuple(int(value) for value in image.size)
            image_mode = image.mode
    except Exception as error:
        findings.append(
            Finding(
                "error",
                "corrupt_image",
                f"画像を開けません: {type(error).__name__}: {error}",
                sample.filename,
            )
        )
        return info, findings

    info["image_size"] = list(image_size)
    info["image_mode"] = image_mode

    foreground_size: tuple[int, int] | None = None
    if sample.foreground is not None:
        try:
            with image_module.open(root / sample.foreground) as foreground:
                foreground.load()
                foreground_size = tuple(int(value) for value in foreground.size)
                foreground_mode = foreground.mode
        except Exception as error:
            findings.append(
                Finding(
                    "error",
                    "corrupt_image",
                    f"foreground 画像を開けません: {type(error).__name__}: {error}",
                    sample.foreground,
                )
            )
        else:
            info["foreground_size"] = list(foreground_size)
            info["foreground_mode"] = foreground_mode
            if foreground_size != image_size:
                findings.append(
                    Finding(
                        "error",
                        "size_mismatch",
                        "foreground のサイズが画像と一致しません",
                        sample.foreground,
                    )
                )

    if sample.mask is not None:
        info["mask_path"] = sample.mask
        try:
            with image_module.open(root / sample.mask) as mask_image:
                mask_image.load()
                mask_size = tuple(int(value) for value in mask_image.size)
                mask_mode = mask_image.mode
                info["mask_size"] = list(mask_size)
                info["mask_mode"] = mask_mode
                if mask_size != image_size:
                    findings.append(
                        Finding(
                            "error",
                            "size_mismatch",
                            "マスクのサイズが画像と一致しません",
                            sample.mask,
                        )
                    )

                mask_l = mask_image.convert("L")
                try:
                    histogram = mask_l.histogram()
                    non_binary_pixels = sum(histogram[1:255])
                    mask_pixels = sum(histogram[1:])
                    info["mask_non_binary_pixels"] = non_binary_pixels
                    if non_binary_pixels > 0:
                        findings.append(
                            Finding(
                                "warning",
                                "non_binary_mask",
                                "マスクに 0/255 以外の画素があります",
                                sample.mask,
                            )
                        )

                    bbox = mask_l.getbbox()
                    info["mask_bbox"] = list(bbox) if bbox is not None else None
                    if sample.label == 1 and mask_pixels == 0:
                        findings.append(
                            Finding(
                                "error",
                                "empty_mask",
                                "label=1 ですがマスクの前景画素が 0 です",
                                sample.mask,
                            )
                        )

                    width, height = image_size
                    if width > 0 and height > 0:
                        info["mask_area_fraction"] = mask_pixels / (width * height)

                    if (
                        sample.foreground is not None
                        and foreground_size is not None
                        and foreground_size == mask_size
                        and mask_pixels > 0
                    ):
                        mask_binary = mask_l.point(
                            lambda value: 255 if value > 0 else 0
                        )
                        try:
                            included = _count_mask_in_foreground(
                                mask_binary,
                                root / sample.foreground,
                                image_module,
                            )
                            if included is None:
                                findings.append(
                                    Finding(
                                        "error",
                                        "size_mismatch",
                                        "foreground のサイズがマスクと一致しません",
                                        sample.foreground,
                                    )
                                )
                            else:
                                info["mask_in_foreground_fraction"] = (
                                    included / mask_pixels
                                )
                        except Exception as error:
                            findings.append(
                                Finding(
                                    "error",
                                    "corrupt_image",
                                    "foreground の画素を評価できません: "
                                    f"{type(error).__name__}: {error}",
                                    sample.foreground,
                                )
                            )
                        finally:
                            mask_binary.close()
                finally:
                    mask_l.close()
        except Exception as error:
            findings.append(
                Finding(
                    "error",
                    "corrupt_image",
                    f"マスク画像を開けません: {type(error).__name__}: {error}",
                    sample.mask,
                )
            )

    return info, findings


def classify_resolution_condition(
    source_hw: tuple[int, int] | None,
    input_hw: tuple[int, int] | None,
) -> str:
    """元画像とモデル入力の解像度関係を分類する。"""
    if source_hw is None or input_hw is None:
        return "unknown"
    if len(source_hw) != 2 or len(input_hw) != 2:
        return "unknown"
    if tuple(source_hw) == tuple(input_hw):
        return "native"
    if (
        input_hw[0] <= source_hw[0]
        and input_hw[1] <= source_hw[1]
        and (input_hw[0] < source_hw[0] or input_hw[1] < source_hw[1])
    ):
        return "downscaled"
    if (
        input_hw[0] >= source_hw[0]
        and input_hw[1] >= source_hw[1]
        and (input_hw[0] > source_hw[0] or input_hw[1] > source_hw[1])
    ):
        return "upscaled"
    return "mixed"


def first_test_image(root: Path, category: str) -> Path | None:
    """``{category}/test`` 配下でソートして最初の画像パスを返す。"""
    test_dir = Path(root) / category / "test"
    if not test_dir.is_dir():
        return None
    candidates = sorted(
        (
            path
            for path in test_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix(),
    )
    return candidates[0] if candidates else None


def read_image_size(
    path: Path,
    *,
    image_module: Any = None,
) -> tuple[int, int] | None:
    """画像ヘッダだけを読み、``(高さ, 幅)`` を返す。失敗時は ``None``。"""
    if image_module is None:
        try:
            from PIL import Image as pil_image_module
        except Exception:
            return None
        image_module = pil_image_module
    try:
        with image_module.open(path) as image:
            width, height = image.size
    except Exception:
        return None
    return int(height), int(width)


def source_resolution(
    root: Path,
    category: str,
    *,
    image_module: Any = None,
) -> tuple[int, int] | None:
    """テスト集合の最初の画像から元解像度 ``(高さ, 幅)`` を得る。"""
    path = first_test_image(root, category)
    if path is None:
        return None
    return read_image_size(path, image_module=image_module)


def build_resolution_condition(
    root: Path,
    category: str,
    image_size: int,
    *,
    image_module: Any = None,
) -> dict[str, Any]:
    """report 用の解像度条件を組み立てる。"""
    source = source_resolution(root, category, image_module=image_module)
    input_hw = (int(image_size), int(image_size))
    return {
        "source_resolution": list(source) if source is not None else None,
        "model_input_resolution": [int(image_size), int(image_size)],
        "condition": classify_resolution_condition(source, input_hw),
    }


LINK_MODES = ("hardlink", "copy", "symlink")


def _same_file(target: Path, source_abs: Path, link_mode: str) -> bool:
    """target が既に source を指しているか（同一 inode または同じ symlink 先か）判定する。"""
    try:
        if link_mode == "symlink":
            if not target.is_symlink():
                return False
            current = Path(os.readlink(target))
            if not current.is_absolute():
                current = target.parent / current
            return current.resolve(strict=False) == source_abs
        if target.is_symlink():
            return False
        if link_mode == "hardlink":
            return os.path.samefile(target, source_abs)
        return target.stat().st_size == source_abs.stat().st_size
    except OSError:
        return False


def build_clean_view(
    root: Path,
    category: str,
    samples: Sequence[Mvtec4kSample],
    out_root: Path,
    *,
    overwrite: bool = False,
    link_mode: str = "hardlink",
) -> int:
    """Anomalib MVTecAD 用の clean view を作る。

    Anomalib 2.7 の MVTecAD は ``glob("**/*")`` の下位 3 要素を
    ``(split, label, name)`` とみなすため、生配置の ``foreground/`` を
    誤って train/test の重複行として拾う。そのため foreground を含めない
    ビューを作り、``--dataset-root`` にはこのディレクトリを渡す。

    既定はハードリンク（同一ファイルシステム上で追加容量ほぼゼロ）。
    Anomalib 2.7 は ``is_within_directory`` で symlink を解決し、ルート外を指す
    ファイルを除外するため、``symlink`` で作ったビューは MVTecAD では
    「Found 0 images」になる（実データで確認）。``symlink`` は他ツール用に残す。
    ハードリンクが作れない場合（別ファイルシステム等）は ``copy`` を指定する。

    Returns:
        新規に作成したファイル（リンク）の数。
    """
    if link_mode not in LINK_MODES:
        raise ValueError(f"link_mode は {LINK_MODES} のいずれかを指定してください: {link_mode}")
    root = Path(root)
    out_root = Path(out_root)
    created = 0

    references: list[str] = []
    for sample in samples:
        references.append(sample.filename)
        if sample.mask is not None:
            references.append(sample.mask)

    for reference in references:
        source = root / reference
        if not source.is_file():
            raise FileNotFoundError(str(source))
        source_abs = source.resolve()
        target = out_root / reference
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.is_symlink() or target.exists():
            if target.is_dir() and not target.is_symlink():
                raise FileExistsError(str(target))
            if _same_file(target, source_abs, link_mode):
                continue
            if not overwrite:
                raise FileExistsError(str(target))
            target.unlink()

        if link_mode == "hardlink":
            os.link(source_abs, target)
        elif link_mode == "copy":
            shutil.copy2(source_abs, target)
        else:
            os.symlink(str(source_abs), str(target))
        created += 1
    return created


def has_foreground_dir(root: Path, category: str) -> bool:
    """``{category}/foreground`` が存在するか返す。"""
    return (Path(root) / category / "foreground").is_dir()


def validate_category(
    root: Path,
    category: str,
    *,
    pixels: bool = True,
    max_samples: int | None = None,
    image_module: Any = None,
) -> dict[str, Any]:
    """1 カテゴリを通しで検証し、JSON 化可能な結果を返す。"""
    root = Path(root)
    samples, findings = load_manifest(root, category)
    findings.extend(validate_entries(samples, category))
    findings.extend(scan_disk(root, category, samples))

    counts = count_by_condition(samples)
    paper_counts = PAPER_COUNTS.get(category)
    findings.extend(compare_with_paper(category, counts))

    image_sizes: Counter[str] = Counter()
    mask_area_values: dict[str, list[float]] = defaultdict(list)
    mask_foreground_values: dict[str, list[float]] = defaultdict(list)
    pixels_checked = 0

    if pixels:
        selected: list[Mvtec4kSample] = []
        used: Counter[tuple[str, str]] = Counter()
        for sample in samples:
            key = (sample.split, sample.label_name)
            if max_samples is not None and used[key] >= max_samples:
                continue
            used[key] += 1
            selected.append(sample)

        for sample in selected:
            info, pixel_findings = check_pixels(
                root,
                sample,
                image_module=image_module,
            )
            findings.extend(pixel_findings)
            pixels_checked += 1

            size = info.get("image_size")
            if isinstance(size, list) and len(size) == 2:
                image_sizes[f"{size[0]}x{size[1]}"] += 1

            label_name = str(info.get("label_name"))
            area = info.get("mask_area_fraction")
            if area is not None:
                mask_area_values[label_name].append(float(area))
            foreground_fraction = info.get("mask_in_foreground_fraction")
            if foreground_fraction is not None:
                mask_foreground_values[label_name].append(
                    float(foreground_fraction)
                )

    mask_stats: dict[str, Any] = {}
    for label_name in sorted(set(mask_area_values) | set(mask_foreground_values)):
        mask_stats[label_name] = {
            "mask_area_fraction": _summarize(
                mask_area_values.get(label_name, []),
                with_max=True,
            ),
            "mask_in_foreground_fraction": _summarize(
                mask_foreground_values.get(label_name, []),
                with_max=False,
            ),
        }

    n_errors = sum(1 for finding in findings if finding.severity == "error")
    n_warnings = sum(1 for finding in findings if finding.severity == "warning")

    return {
        "category": category,
        "counts": counts,
        "paper_counts": dict(paper_counts) if paper_counts is not None else None,
        "image_sizes": dict(sorted(image_sizes.items())),
        "mask_stats": mask_stats,
        "findings": [_finding_to_dict(finding) for finding in findings],
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "pixels_checked": pixels_checked,
    }
