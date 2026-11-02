"""VisA / MVTec AD 2 の生配置から Anomalib 用のデータビューを作成する。

Anomalib 2.7 の ``Visa`` / ``MVTecAD`` データモジュールが期待する
ディレクトリ構成へ変換する。標準ライブラリのみを import し、Pillow は
マスク変換の関数内で遅延 import する（CLI の --help とテストを軽く保つ）。
"""

from __future__ import annotations

import csv
import filecmp
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

#: ビュー作成時のリンク方式。symlink は Anomalib の is_within_directory で
#: ルート外を指すファイルとして除外されるため用意しない（MVTec-4K の
#: clean view と同じ理由）。
VIEW_LINK_MODES: tuple[str, ...] = ("hardlink", "copy")
VIEW_DATASETS: tuple[str, ...] = ("visa", "mvtec_ad2")

VISA = "visa"
MVTEC_AD2 = "mvtec_ad2"

#: マスクの前景値。0/255 の L 画像にそろえる。
MASK_ON_VALUE = 255

#: VisA の分割 CSV（raw_root からの相対 posix パス）
VISA_SPLIT_CSV = "split_csv/1cls.csv"
#: Anomalib の Visa データモジュールが読むサブディレクトリ
VISA_VIEW_SUBDIR = "visa_pytorch"
VIEW_MANIFEST_NAME = "view_manifest.json"

#: MVTec AD 2 で使わないディレクトリ。
#: validation は正常のみの公式検証セット、test_private* は正解非公開。
MVTEC_AD2_EXCLUDED: tuple[str, ...] = (
    "validation",
    "test_private",
    "test_private_mixed",
)
#: MVTec AD 2 のビューで使う生配置（raw_root からの相対パス表現）
MVTEC_AD2_SOURCE = (
    "train/good, test_public/good, test_public/bad, test_public/ground_truth/bad"
)

_VISA_MASK_NOTES: tuple[str, ...] = (
    "生配置の split_csv/1cls.csv は全カテゴリを含むため、object 列で絞り込んだ。",
    "マスクは 0 以外を 255 に二値化した L 画像 PNG として書き出した。",
    "Anomalib の Visa データモジュールは visa_pytorch/<category> 配下の MVTec 形式を読む。",
)
_MVTEC_AD2_NOTES: tuple[str, ...] = (
    "validation/ は正常のみの公式検証セットのため、ビューには含めていない。",
    "test_private/ と test_private_mixed/ は正解が非公開のため使っていない。",
    ".png 以外の拡張子は無視した。",
)


@dataclass(frozen=True)
class VisaRow:
    """VisA の split_csv/1cls.csv 1 行分（1 カテゴリ・1 サンプル）。"""

    split: str
    """``train`` または ``test``。"""

    label: str
    """``good``（CSV の normal）または ``bad``（CSV の anomaly）。"""

    image: str
    """生配置からの相対 posix パス。"""

    mask: str | None
    """異常サンプルのマスクの相対 posix パス。正常サンプルは None。"""


def _validate_relative(value: str, *, field: str) -> str:
    """CSV のパス列を検証し、正規化した相対 posix パスを返す。"""
    text = value.strip().replace("\\", "/")
    if not text:
        raise ValueError(f"{field} が空です")
    pure = PurePosixPath(text)
    if pure.is_absolute():
        raise ValueError(f"{field} に絶対パスは指定できません: {value}")
    if ".." in pure.parts:
        raise ValueError(f"{field} に '..' を含むパスは指定できません: {value}")
    return pure.as_posix()


def read_visa_split(raw_root: Path | str, category: str) -> list[VisaRow]:
    """VisA の ``split_csv/1cls.csv`` から category の行だけを読み出す。

    他カテゴリの行は無視する。split は train/test、label は normal/anomaly
    以外を ValueError とし、test anomaly 行の mask が空の場合も ValueError とする。

    Returns:
        ``VisaRow`` のリスト（CSV の出現順）。
    """
    csv_path = Path(raw_root) / Path(VISA_SPLIT_CSV)
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))

    rows: list[VisaRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"object", "split", "label", "image", "mask"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"{csv_path} に必要な列がありません: {sorted(missing)}")

        for index, record in enumerate(reader, start=2):
            if (record.get("object") or "").strip() != category:
                continue

            raw_split = (record.get("split") or "").strip().lower()
            if raw_split not in ("train", "test"):
                raise ValueError(f"{csv_path}:{index} の split が不正です: {raw_split}")

            raw_label = (record.get("label") or "").strip().lower()
            if raw_label == "normal":
                label = "good"
            elif raw_label == "anomaly":
                label = "bad"
            else:
                raise ValueError(f"{csv_path}:{index} の label が不正です: {raw_label}")

            image = _validate_relative(record.get("image") or "", field="image")
            if raw_split == "train" and label == "bad":
                raise ValueError(
                    f"{csv_path}:{index} の train に anomaly 行は扱えません: {image}"
                )

            raw_mask = (record.get("mask") or "").strip()
            mask: str | None = None
            if label == "bad":
                if not raw_mask:
                    raise ValueError(
                        f"{csv_path}:{index} の anomaly 行に mask がありません: {image}"
                    )
                mask = _validate_relative(raw_mask, field="mask")
            elif raw_mask:
                # VisA の正常行の mask は空。値が入っていても使わないが、
                # 不正なパスは早期に弾いておく。
                _validate_relative(raw_mask, field="mask")

            rows.append(
                VisaRow(split=raw_split, label=label, image=image, mask=mask)
            )
    return rows


def _same_file(target: Path, source_abs: Path, link_mode: str) -> bool:
    """target が既に source と同じファイルか判定する。"""
    try:
        if target.is_symlink():
            return False
        if link_mode == "hardlink":
            return os.path.samefile(target, source_abs)
        return filecmp.cmp(str(target), str(source_abs), shallow=False)
    except OSError:
        return False


def _place(source: Path, target: Path, *, link_mode: str, overwrite: bool) -> bool:
    """source を target にハードリンクまたはコピーする。

    Returns:
        新規に作成したら True、既に同一ファイルがあったら False。

    Raises:
        FileExistsError: 内容の異なるファイルが既にあり overwrite が False。
        FileNotFoundError: source が無い。
        ValueError: link_mode が不正。
    """
    if link_mode not in VIEW_LINK_MODES:
        raise ValueError(
            f"link_mode は {VIEW_LINK_MODES} のいずれかを指定してください: {link_mode}"
        )
    source = Path(source)
    target = Path(target)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    source_abs = source.resolve()

    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            raise FileExistsError(str(target))
        if _same_file(target, source_abs, link_mode):
            return False
        if not overwrite:
            raise FileExistsError(
                f"{target} に異なる内容のファイルが既にあります。"
                "--overwrite を指定してください。"
            )
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "hardlink":
        try:
            os.link(source_abs, target)
        except OSError as exc:
            raise OSError(
                "ハードリンクを作成できませんでした（別ファイルシステムなどの可能性）。"
                f"--link-mode copy を指定してください: {target}"
            ) from exc
    else:
        shutil.copy2(source_abs, target)
    return True


def _same_mask(target: Path, expected: Any) -> bool:
    """既存のマスクが expected と同じ画素値か（同じなら書き直さない）。"""
    from PIL import Image  # 遅延 import

    try:
        with Image.open(target) as existing:
            gray = existing.convert("L")
            if gray.size != expected.size:
                return False
            return gray.tobytes() == expected.tobytes()
    except OSError:
        return False


def _write_binary_mask(source: Path, target: Path, *, overwrite: bool) -> bool:
    """マスクを 0/255 の L 画像 PNG として書き出す。

    Returns:
        新規に作成したら True、既存の内容が同一でスキップしたら False。

    Raises:
        FileExistsError: 内容の異なるマスクが既にあり overwrite が False。
        FileNotFoundError: source が無い。
        ValueError: source を画像として読めない。
    """
    from PIL import Image  # 遅延 import（Pillow はマスク変換だけで使う）

    source = Path(source)
    target = Path(target)
    if not source.is_file():
        raise FileNotFoundError(str(source))

    try:
        with Image.open(source) as raw_image:
            gray = raw_image.convert("L")
            binary = gray.point(lambda value: MASK_ON_VALUE if value else 0)
            binary.load()
    except OSError as exc:
        raise ValueError(f"マスク画像を読み込めません: {source}") from exc

    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            raise FileExistsError(str(target))
        if _same_mask(target, binary):
            return False
        if not overwrite:
            raise FileExistsError(
                f"{target} に異なる内容のマスクが既にあります。"
                "--overwrite を指定してください。"
            )
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    binary.save(target, format="PNG")
    return True


def build_visa_view(
    raw_root: Path | str,
    category: str,
    out_root: Path | str,
    *,
    link_mode: str = "hardlink",
    overwrite: bool = False,
) -> dict[str, Any]:
    """VisA の生配置から Anomalib ``Visa`` 用のビューを作成する。

    生成先は ``out_root/visa_pytorch/<category>/`` 配下の MVTec 形式
    （train/good, test/good, test/bad, ground_truth/bad）。画像は
    ``link_mode`` に従ってハードリンクまたはコピーし、マスクは 0/255 に
    二値化した ``<画像stem>.png`` を書き出す。既存ファイルは同一内容なら
    スキップし、相違時は overwrite が無ければ FileExistsError とする。

    Returns:
        dataset / category / link_mode / counts / source / excluded / notes を含む辞書。
        ``counts`` のキーは train_good, test_good, test_bad, masks。
    """
    if link_mode not in VIEW_LINK_MODES:
        raise ValueError(
            f"link_mode は {VIEW_LINK_MODES} のいずれかを指定してください: {link_mode}"
        )
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    rows = read_visa_split(raw_root, category)
    if not rows:
        raise ValueError(
            f"{raw_root / Path(VISA_SPLIT_CSV)} に object={category} の行がありません"
        )

    base = out_root / VISA_VIEW_SUBDIR / category
    for relative in ("train/good", "test/good", "test/bad", "ground_truth/bad"):
        (base / relative).mkdir(parents=True, exist_ok=True)

    counts = {"train_good": 0, "test_good": 0, "test_bad": 0, "masks": 0}
    for row in rows:
        name = Path(row.image).name
        source_image = raw_root / row.image
        if row.label == "good":
            if row.split == "train":
                target = base / "train" / "good" / name
                counts["train_good"] += 1
            else:
                target = base / "test" / "good" / name
                counts["test_good"] += 1
            _place(source_image, target, link_mode=link_mode, overwrite=overwrite)
            continue

        target = base / "test" / "bad" / name
        _place(source_image, target, link_mode=link_mode, overwrite=overwrite)
        counts["test_bad"] += 1

        if row.mask is None:  # read_visa_split で弾いているが念のため
            raise ValueError(f"anomaly 行に mask がありません: {row.image}")
        mask_target = base / "ground_truth" / "bad" / f"{Path(row.image).stem}.png"
        _write_binary_mask(
            raw_root / row.mask, mask_target, overwrite=overwrite
        )
        counts["masks"] += 1

    return {
        "dataset": VISA,
        "category": category,
        "link_mode": link_mode,
        "counts": counts,
        "source": VISA_SPLIT_CSV,
        "excluded": [],
        "notes": list(_VISA_MASK_NOTES),
    }


def _list_pngs(directory: Path) -> list[Path]:
    """directory 直下の .png ファイルを名前順で返す（無ければ FileNotFoundError）。"""
    if not directory.is_dir():
        raise FileNotFoundError(str(directory))
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".png"
    )


def _count_pngs(directory: Path) -> int:
    """directory 直下の .png ファイル数（無ければ 0）。"""
    if not directory.is_dir():
        return 0
    return sum(
        1 for path in directory.iterdir() if path.is_file() and path.suffix == ".png"
    )


def build_mvtec_ad2_view(
    raw_root: Path | str,
    category: str,
    out_root: Path | str,
    *,
    link_mode: str = "hardlink",
    overwrite: bool = False,
) -> dict[str, Any]:
    """MVTec AD 2 の生配置から Anomalib ``MVTecAD`` 用のビューを作成する。

    生成先は ``out_root/<category>/`` 配下の MVTec 形式。
    train/good ← train/good、test/good ← test_public/good、
    test/bad ← test_public/bad、ground_truth/bad/<stem>_mask.png ←
    test_public/ground_truth/bad/<stem>_mask.png。``validation/`` と
    ``test_private*/`` は使わない。``.png`` 以外の拡張子は無視する。

    Returns:
        dataset / category / link_mode / counts / source / excluded /
        official_validation_good / notes を含む辞書。
    """
    if link_mode not in VIEW_LINK_MODES:
        raise ValueError(
            f"link_mode は {VIEW_LINK_MODES} のいずれかを指定してください: {link_mode}"
        )
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    raw_category = raw_root / category
    base = out_root / category

    train_images = _list_pngs(raw_category / "train" / "good")
    test_good_images = _list_pngs(raw_category / "test_public" / "good")
    test_bad_images = _list_pngs(raw_category / "test_public" / "bad")
    mask_dir = raw_category / "test_public" / "ground_truth" / "bad"
    if not mask_dir.is_dir():
        raise FileNotFoundError(str(mask_dir))

    for relative in ("train/good", "test/good", "test/bad", "ground_truth/bad"):
        (base / relative).mkdir(parents=True, exist_ok=True)

    counts = {"train_good": 0, "test_good": 0, "test_bad": 0, "masks": 0}
    for source in train_images:
        _place(
            source,
            base / "train" / "good" / source.name,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        counts["train_good"] += 1
    for source in test_good_images:
        _place(
            source,
            base / "test" / "good" / source.name,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        counts["test_good"] += 1
    for source in test_bad_images:
        _place(
            source,
            base / "test" / "bad" / source.name,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        counts["test_bad"] += 1

        mask_name = f"{source.stem}_mask.png"
        mask_source = mask_dir / mask_name
        if not mask_source.is_file():
            raise FileNotFoundError(str(mask_source))
        _place(
            mask_source,
            base / "ground_truth" / "bad" / mask_name,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        counts["masks"] += 1

    return {
        "dataset": MVTEC_AD2,
        "category": category,
        "link_mode": link_mode,
        "counts": counts,
        "source": MVTEC_AD2_SOURCE,
        "excluded": list(MVTEC_AD2_EXCLUDED),
        "official_validation_good": _count_pngs(
            raw_category / "validation" / "good"
        ),
        "notes": list(_MVTEC_AD2_NOTES),
    }


def write_view_manifest(
    out_root: Path | str, info: Mapping[str, Any]
) -> Path:
    """ビューの情報を ``out_root/view_manifest.json`` に書き出す。

    生データの絶対パスは記録しない（環境依存のパスをログや共有物に残さない）。

    Returns:
        書き出したファイルのパス。
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset": str(info.get("dataset", "")),
        "category": str(info.get("category", "")),
        "link_mode": str(info.get("link_mode", "")),
        "counts": dict(info.get("counts", {})),
        "source": info.get("source", ""),
        "excluded": list(info.get("excluded", [])),
        "notes": list(info.get("notes", [])),
    }
    if "official_validation_good" in info:
        payload["official_validation_good"] = int(info["official_validation_good"])

    path = out_root / VIEW_MANIFEST_NAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _looks_like_raw_visa(root: Path, category: str) -> bool:
    return (root / category / "Data").is_dir() or (
        root / Path(VISA_SPLIT_CSV)
    ).is_file()


def _looks_like_raw_mvtec_ad2(root: Path, category: str) -> bool:
    return (root / category / "test_public").is_dir()


def _validate_visa_view(root: Path, category: str) -> list[str]:
    base = root / VISA_VIEW_SUBDIR / category
    problems = [
        f"{base / split / 'good'} が見つかりません。"
        for split in ("train", "test")
        if not (base / split / "good").is_dir()
    ]
    if problems and _looks_like_raw_visa(root, category):
        problems.append(
            f"{root} は VisA の生配置のようです。"
            "scripts/make_dataset_view.py でビューを作成し、--dataset-root には"
            "その出力ディレクトリを指定してください"
            f"（例: --dataset visa --raw-root {root} --category {category} --out <出力先>）。"
        )
    return problems


def _validate_mvtec_ad2_view(root: Path, category: str) -> list[str]:
    base = root / category
    problems = [
        f"{base / split / 'good'} が見つかりません。"
        for split in ("train", "test")
        if not (base / split / "good").is_dir()
    ]
    if problems and _looks_like_raw_mvtec_ad2(root, category):
        problems.append(
            f"{root} は MVTec AD 2 の生配置のようです（test_public/ がある）。"
            "scripts/make_dataset_view.py でビューを作成し、--dataset-root には"
            "その出力ディレクトリを指定してください"
            f"（例: --dataset mvtec_ad2 --raw-root {root} --category {category} --out <出力先>）。"
        )
    return problems


def validate_view(dataset: str, root: Path | str, category: str) -> list[str]:
    """``--dataset-root`` がビューとして使えるか検証する。

    Anomalib のデータモジュールによる自動ダウンロードを避けるため、
    run_anomalib から呼び出して配置を先に確認する。

    Returns:
        問題点の日本語メッセージ一覧。空リストなら問題なし。
    """
    root = Path(root)
    if dataset == VISA:
        return _validate_visa_view(root, category)
    if dataset == MVTEC_AD2:
        return _validate_mvtec_ad2_view(root, category)
    raise ValueError(f"未対応の dataset です: {dataset}")


def evaluation_scope(dataset: str, root: Path | str, category: str) -> dict[str, Any]:
    """report.json に記録する評価範囲（どの split を使ったか）を返す（Issue #32）。

    ビューの ``view_manifest.json`` があればその ``source`` / ``excluded`` /
    ``counts`` を写す。MVTec AD 2 は公開 test split（``test_public``）のみで
    評価し、``test_private`` / ``test_private_mixed`` は正解非公開のため使わない
    ことを ``public_test_only=True`` で明示する（マニフェストが無くても既定値で記録）。
    """
    root = Path(root)
    manifest_path = root / VIEW_MANIFEST_NAME
    info: dict[str, Any] = {}
    manifest_found = False
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            info = loaded
            manifest_found = True
    scope: dict[str, Any] = {
        "dataset": dataset,
        "category": category,
        "view_manifest_found": manifest_found,
        "view_manifest_category": info.get("category") if manifest_found else None,
        "counts": dict(info.get("counts", {})) if manifest_found else None,
    }
    if dataset == MVTEC_AD2:
        scope["source"] = info.get("source") or MVTEC_AD2_SOURCE
        scope["excluded"] = list(info.get("excluded") or MVTEC_AD2_EXCLUDED)
        scope["public_test_only"] = True
        scope["note"] = (
            "MVTec AD 2 は公開 test split（test_public）のみで評価している。"
            "test_private / test_private_mixed は正解非公開のため使っていない。"
        )
    elif dataset == VISA:
        scope["source"] = info.get("source") or VISA_SPLIT_CSV
        scope["excluded"] = list(info.get("excluded") or [])
        scope["public_test_only"] = None
        scope["note"] = "VisA は公式分割 split_csv/1cls.csv の train / test を使っている。"
    else:
        raise ValueError(f"未対応の dataset です: {dataset}")
    return scope
