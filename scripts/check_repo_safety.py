#!/usr/bin/env python3
"""追跡ファイルに秘密情報・巨大ファイル・データセットが混入していないか検査する。

標準ライブラリのみで動作する。``git`` の呼び出しは :func:`main` 側だけに閉じ込め、
判定ロジックは純粋関数（:func:`check_size` / :func:`check_extension` /
:func:`check_directory` / :func:`check_secrets` / :func:`check_file`）に切り出して
あるため、``tests/test_check_repo_safety.py`` から git なしで検証できる。

検査項目:

* サイズ上限（既定 1 MiB）。``--max-bytes`` で変更できる。
* 禁止拡張子（モデル重み・アーカイブ）。画像は ``docs/images/`` 配下のみ許可する。
* 禁止ディレクトリ接頭辞（``datasets/`` / ``results/`` / ``artifacts/`` /
  ``checkpoints/``）。
* 秘密情報らしき文字列。UTF-8 として読めるファイルだけを対象にする。

検出しても一致した値そのものは出力しない。ファイル名・行番号・規則名だけを表示する。

使い方::

    python scripts/check_repo_safety.py              # git の追跡ファイル全体
    python scripts/check_repo_safety.py --staged     # インデックスに追加済みのファイル
    python scripts/check_repo_safety.py path/to/file # パスを直接指定

終了コードは、違反なしで 0、違反ありで 1、git の実行に失敗した場合など
検査自体を続けられない場合に 2 を返す。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

#: 既定のサイズ上限（1 MiB）。これ以下の要約 JSON/CSV は通る想定。
DEFAULT_MAX_BYTES: int = 1024 * 1024

#: 秘密情報スキャンのために読み込む最大バイト数。これを超えるファイルは
#: テキスト判定を行わない（サイズ上限で別途検出される想定）。
MAX_SCAN_BYTES: int = 4 * 1024 * 1024

#: 画像ファイルを許可する唯一のディレクトリ接頭辞。
ALLOWED_IMAGE_PREFIX: str = "docs/images/"

#: 違反の種類（出力に含める規則名）。
RULE_SIZE: str = "size"
RULE_EXTENSION: str = "extension"
RULE_DIRECTORY: str = "directory"
RULE_SECRET: str = "secret"

#: ドキュメント用の画像として許可する拡張子。
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)

#: 追跡してはいけない拡張子（モデル重み・アーカイブ）。
#: OpenVINO IR は .xml と .bin の組で配布されるが、.bin の検出で十分なので
#: .xml は含めない（設定ファイル等でも使われ、誤検出しやすいため）。
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset(
    {
        # モデル重み・学習済みモデルの派生物
        ".ckpt",
        ".pt",
        ".pth",
        ".onnx",
        ".bin",
        ".safetensors",
        ".h5",
        ".pkl",
        ".npy",
        ".npz",
        # アーカイブ（".tar.gz" はサフィックス ".gz" として検出される）
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".xz",
        ".7z",
    }
)

#: 追跡してはいけないディレクトリ（リポジトリルート相対の接頭辞）。
FORBIDDEN_DIR_PREFIXES: tuple[str, ...] = (
    "datasets/",
    "results/",
    "artifacts/",
    "checkpoints/",
)

# 秘密情報らしき文字列のパターン。このファイル自身が検査対象になっても誤検出
# しないよう、パターン文字列はリテラルを連結して組み立てる（例: "gh" + "[pors]_"）。
# 値そのものは出力しないので、グループは捕捉しても使わない。
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile("-----BEGIN " + r"[A-Z0-9 ]*" + "PRIVATE KEY-----"),
    ),
    ("aws_access_key_id", re.compile("AKIA" + r"[0-9A-Z]{16}")),
    ("github_token", re.compile("gh" + r"[pors]_" + r"[A-Za-z0-9]{20,}")),
    ("github_fine_grained_pat", re.compile("github" + "_pat_" + r"[A-Za-z0-9_]{20,}")),
    ("openai_api_key", re.compile("sk" + "-" + r"[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile("xox" + r"[baprs]-" + r"[A-Za-z0-9-]{10,}")),
    (
        "assigned_api_key",
        re.compile(
            "api"
            + r"[_-]?key"
            + r"\s*=\s*"
            + r"[\"']"
            + r"([A-Za-z0-9_\-]{16,})"
            + r"[\"']",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class Violation:
    """1 件の違反。``message`` には検出した値そのものを入れない。"""

    path: str
    rule: str
    message: str
    line: int | None = None

    def format(self) -> str:
        """出力用の 1 行文字列を返す（値は含まない）。"""
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{location}: [{self.rule}] {self.message}"


def normalize_path(path: str) -> str:
    """リポジトリルート相対・スラッシュ区切りの表記に正規化する。

    Args:
        path: 検査対象のパス。

    Returns:
        先頭の ``./`` と区切り文字の揺れを除いたパス。
    """
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def check_size(
    path: str, size: int, max_bytes: int = DEFAULT_MAX_BYTES
) -> list[Violation]:
    """ファイルサイズが上限を超えていないか検査する。

    Args:
        path: 検査対象のパス。
        size: ファイルサイズ（バイト）。
        max_bytes: 許容する最大サイズ（バイト）。

    Returns:
        違反のリスト（上限以下なら空）。
    """
    if size > max_bytes:
        return [
            Violation(
                path=normalize_path(path),
                rule=RULE_SIZE,
                message=f"サイズ {size} バイトが上限 {max_bytes} バイトを超えています",
            )
        ]
    return []


def check_extension(path: str) -> list[Violation]:
    """拡張子が禁止されていないか、画像なら許可ディレクトリ配下か検査する。

    Args:
        path: 検査対象のパス。

    Returns:
        違反のリスト（問題なければ空）。
    """
    normalized = normalize_path(path)
    suffix = Path(normalized).suffix.lower()
    if not suffix:
        return []
    if suffix in IMAGE_EXTENSIONS:
        if normalized.startswith(ALLOWED_IMAGE_PREFIX):
            return []
        return [
            Violation(
                path=normalized,
                rule=RULE_EXTENSION,
                message=(
                    f"画像ファイル（{suffix}）は {ALLOWED_IMAGE_PREFIX} 配下のみ"
                    "許可されています"
                ),
            )
        ]
    if suffix in FORBIDDEN_EXTENSIONS:
        return [
            Violation(
                path=normalized,
                rule=RULE_EXTENSION,
                message=f"禁止拡張子です（{suffix}）",
            )
        ]
    return []


def check_directory(path: str) -> list[Violation]:
    """禁止ディレクトリ配下のファイルでないか検査する。

    Args:
        path: 検査対象のパス。

    Returns:
        違反のリスト（問題なければ空）。
    """
    normalized = normalize_path(path)
    for prefix in FORBIDDEN_DIR_PREFIXES:
        if normalized.startswith(prefix) or normalized == prefix.rstrip("/"):
            return [
                Violation(
                    path=normalized,
                    rule=RULE_DIRECTORY,
                    message=f"禁止ディレクトリ {prefix} 配下のファイルです",
                )
            ]
    return []


def check_secrets(path: str, text: str) -> list[Violation]:
    """テキスト中に秘密情報らしき文字列がないか検査する。

    一致した値は戻り値にも出力文字列にも含めない。同じ規則が同じ行で複数回
    一致しても 1 件にまとめる。

    Args:
        path: 検査対象のパス（表示用）。
        text: ファイル内容（UTF-8 として読めた文字列）。

    Returns:
        違反のリスト（問題なければ空）。
    """
    normalized = normalize_path(path)
    violations: list[Violation] = []
    seen: set[tuple[str, int]] = set()
    for name, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            key = (name, line)
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                Violation(
                    path=normalized,
                    rule=RULE_SECRET,
                    message=f"秘密情報らしき文字列を検出しました（規則: {name}）",
                    line=line,
                )
            )
    return violations


def check_file(
    path: str,
    size: int,
    text: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Violation]:
    """1 ファイル分の検査をまとめて実行する。

    Args:
        path: 検査対象のパス。
        size: ファイルサイズ（バイト）。
        text: ファイル内容。``None`` の場合は秘密情報スキャンを行わない
            （バイナリ、または大きすぎて読まなかった場合）。
        max_bytes: 許容する最大サイズ（バイト）。

    Returns:
        違反のリスト（問題なければ空）。
    """
    violations: list[Violation] = []
    violations.extend(check_size(path, size, max_bytes))
    violations.extend(check_extension(path))
    violations.extend(check_directory(path))
    if text is not None:
        violations.extend(check_secrets(path, text))
    return violations


def format_violations(violations: Sequence[Violation]) -> str:
    """違反リストを 1 行 1 件の文字列に整形する。

    Args:
        violations: 整形する違反リスト。

    Returns:
        改行区切りの文字列（空リストなら空文字列）。
    """
    return "\n".join(violation.format() for violation in violations)


def list_tracked_files() -> list[str]:
    """``git ls-files`` で追跡ファイルの一覧を取得する。

    Returns:
        リポジトリルート相対のパスのリスト。

    Raises:
        RuntimeError: git の実行に失敗した場合。
    """
    return _git_nul_paths(["ls-files", "-z"])


def list_staged_files() -> list[str]:
    """``git diff --cached`` でインデックスに追加済みのファイル一覧を取得する。

    削除（``D``）は対象外とし、追加・コピー・変更・リネームのみを返す。

    Returns:
        リポジトリルート相対のパスのリスト。

    Raises:
        RuntimeError: git の実行に失敗した場合。
    """
    return _git_nul_paths(
        ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]
    )


def _git_nul_paths(args: Sequence[str]) -> list[str]:
    """NUL 区切りでパスを返す git サブコマンドを実行する。

    Args:
        args: ``git`` に渡す引数（NUL 区切り出力になるものを指定する）。

    Returns:
        空要素を除いたパスのリスト。

    Raises:
        RuntimeError: git が非 0 で終了した場合。
    """
    completed = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} に失敗しました: {detail}")
    raw = completed.stdout.decode("utf-8", "surrogateescape")
    return [entry for entry in raw.split("\0") if entry]


def _read_text(path: str, size: int) -> str | None:
    """秘密情報スキャン用にファイルをテキストとして読む。

    Args:
        path: リポジトリルート相対のパス。
        size: ファイルサイズ（バイト）。

    Returns:
        UTF-8 として読めた内容。バイナリ、読めない、または大きすぎる場合は ``None``。
    """
    if size > MAX_SCAN_BYTES:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def evaluate_paths(paths: Sequence[str], max_bytes: int) -> list[Violation]:
    """パスの一覧を検査して違反を集める（ファイルの読み込みを伴う）。

    作業ツリーに存在しないパス（``--staged`` で削除が混じった場合など）は
    読み飛ばす。``--staged`` でも内容は作業ツリーから読むため、インデックスと
    作業ツリーが食い違っている場合は作業ツリー側の内容で判定される。

    Args:
        paths: 検査対象のパス。
        max_bytes: 許容する最大サイズ（バイト）。

    Returns:
        検出した違反のリスト。
    """
    violations: list[Violation] = []
    for path in paths:
        try:
            size = os.stat(path).st_size
        except OSError:
            continue
        violations.extend(
            check_file(path, size, text=_read_text(path, size), max_bytes=max_bytes)
        )
    return violations


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサを構築する。

    Returns:
        設定済みの :class:`argparse.ArgumentParser`。
    """
    parser = argparse.ArgumentParser(
        prog="check_repo_safety.py",
        description=(
            "追跡ファイルのサイズ・拡張子・ディレクトリ・秘密情報らしき文字列を検査する。"
            "違反があれば終了コード 1 を返す。"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="検査対象のパス。省略時は git の追跡ファイル全体（--staged なら追加済みのみ）。",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="git のインデックスに追加済みのファイルのみを対象にする（pre-commit 用）。",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        metavar="BYTES",
        help=f"1 ファイルあたりのサイズ上限（バイト、既定 {DEFAULT_MAX_BYTES}）。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """エントリポイント。

    Args:
        argv: コマンドライン引数（``None`` なら :data:`sys.argv`）。

    Returns:
        終了コード。違反なし 0、違反あり 1、検査を続けられない場合 2。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_bytes <= 0:
        parser.error("--max-bytes には 1 以上の整数を指定してください")

    if args.paths:
        paths = list(args.paths)
    else:
        try:
            paths = list_staged_files() if args.staged else list_tracked_files()
        except RuntimeError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 2

    violations = evaluate_paths(paths, args.max_bytes)
    if violations:
        print(format_violations(violations), file=sys.stderr)
        print(
            f"違反 {len(violations)} 件（検査対象 {len(paths)} ファイル）",
            file=sys.stderr,
        )
        return 1
    print(f"違反なし（検査対象 {len(paths)} ファイル）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
