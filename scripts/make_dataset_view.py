"""VisA / MVTec AD 2 の生配置から Anomalib 用のデータビューを作成する CLI。

torch / anomalib は使わない。マスク変換に Pillow を使う。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.dataset_views import (  # noqa: E402  (sys.path 調整のため後置)
    MVTEC_AD2,
    VIEW_DATASETS,
    VIEW_LINK_MODES,
    VISA,
    build_mvtec_ad2_view,
    build_visa_view,
    write_view_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """CLI の引数パーサを返す。"""
    parser = argparse.ArgumentParser(
        description="VisA / MVTec AD 2 の生配置から Anomalib 用のビューを作成する",
    )
    parser.add_argument("--dataset", choices=VIEW_DATASETS, required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="生データのルート（VisA は split_csv/ を含むディレクトリ）",
    )
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="ビューの出力先。run_anomalib の --dataset-root にはこのディレクトリを渡す",
    )
    parser.add_argument(
        "--link-mode",
        choices=VIEW_LINK_MODES,
        default="hardlink",
        help="画像の配置方法。別ファイルシステムなら copy を指定する",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存のビューに内容の異なるファイルがある場合に置き換える",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """ビューを作成し、作成結果の要約を標準出力に JSON で出す。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.dataset == VISA:
            info = build_visa_view(
                args.raw_root,
                args.category,
                args.out,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
            )
        elif args.dataset == MVTEC_AD2:
            info = build_mvtec_ad2_view(
                args.raw_root,
                args.category,
                args.out,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
            )
        else:  # argparse の choices で弾かれるが念のため
            parser.error(f"未対応の dataset です: {args.dataset}")
            return 2
    except (OSError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    manifest = write_view_manifest(args.out, info)
    summary: dict[str, Any] = {
        "dataset": info["dataset"],
        "category": info["category"],
        "link_mode": info["link_mode"],
        "counts": info["counts"],
        "manifest": str(manifest),
    }
    if "official_validation_good" in info:
        summary["official_validation_good"] = info["official_validation_good"]
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
