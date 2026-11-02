"""MVTec-4K の構造・ラベル・マスク・ピクセル整合性を検証する CLI。

依存なしで ``--help`` と基本動作が使えるよう、Pillow はピクセル検査を
行うときだけ遅延 import する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.mvtec4k import (  # noqa: E402
    CATEGORIES,
    LINK_MODES,
    build_clean_view,
    load_manifest,
    validate_category,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVTec-4K のデータ構造・ラベル・マスクを検証する"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/MVTec-4K"),
        help="MVTec-4K の展開先ルート",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        metavar="CATEGORY",
        help="検証するカテゴリ。省略時は root 配下に存在する既定カテゴリ",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="検証結果を JSON で保存するパス",
    )
    parser.add_argument(
        "--skip-pixels",
        action="store_true",
        help="Pillow を使うピクセル検査を省略する",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="test の各 label_name と train から先頭 N 件ずつだけピクセル検査する",
    )
    parser.add_argument(
        "--make-view",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "エラーが 0 のカテゴリだけ、Anomalib MVTecAD 用の clean view を "
            "DIR に作る（foreground は含めない。既定はハードリンク）"
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=LINK_MODES,
        default="hardlink",
        help=(
            "clean view の作り方。Anomalib 2.7 は root 外を指す symlink を除外するため、"
            "MVTecAD で使うなら hardlink（同一FS）か copy を選ぶ"
        ),
    )
    return parser


def _print_category_result(result: dict) -> None:
    counts = result["counts"]
    print(
        f"[{result['category']}] "
        f"train={counts['train']} test_good={counts['test_good']} "
        f"large={counts['large']} tiny={counts['tiny']} "
        f"errors={result['n_errors']} warnings={result['n_warnings']}"
    )
    if result["image_sizes"]:
        sizes = ", ".join(
            f"{size}:{count}" for size, count in result["image_sizes"].items()
        )
        print(f"  画像サイズ: {sizes}")
    if result["pixels_checked"]:
        print(f"  ピクセル検査: {result['pixels_checked']} 件")

    findings = result["findings"]
    limit = 30
    for finding in findings[:limit]:
        path = finding.get("path")
        suffix = f" ({path})" if path else ""
        print(
            f"  {finding['severity']}: {finding['code']}: "
            f"{finding['message']}{suffix}"
        )
    if len(findings) > limit:
        print(f"  ... ほか {len(findings) - limit} 件")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root)

    categories: list[str]
    if args.category:
        categories = list(args.category)
    else:
        categories = [
            category for category in CATEGORIES if (root / category).is_dir()
        ]

    if not categories:
        print(
            f"検証対象のカテゴリがありません: {root} 配下に "
            f"{', '.join(CATEGORIES)} のいずれも見つかりません。",
            file=sys.stderr,
        )
        return 1

    if args.max_samples is not None and args.max_samples < 1:
        print("--max-samples には 1 以上を指定してください。", file=sys.stderr)
        return 2

    results: list[dict] = []
    view_failed = False
    for category in categories:
        result = validate_category(
            root,
            category,
            pixels=not args.skip_pixels,
            max_samples=args.max_samples,
        )
        results.append(result)
        _print_category_result(result)

        if args.make_view is not None and result["n_errors"] == 0:
            samples, load_findings = load_manifest(root, category)
            if load_findings:
                print(
                    f"[{category}] clean view 作成前にマニフェストを再読込できませんでした。",
                    file=sys.stderr,
                )
                view_failed = True
                continue
            try:
                created = build_clean_view(
                    root,
                    category,
                    samples,
                    Path(args.make_view),
                    link_mode=args.link_mode,
                )
            except Exception as error:
                print(
                    f"[{category}] clean view の作成に失敗しました: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                view_failed = True
            else:
                print(f"[{category}] clean view 作成: {created} 件（{args.link_mode}）")

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(root),
            "categories": results,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"検証結果を保存しました: {out_path}")

    has_errors = any(result["n_errors"] > 0 for result in results) or view_failed
    if has_errors:
        print("検証でエラーが検出されました。", file=sys.stderr)
        return 2
    print("検証は完了しました。エラーはありません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
