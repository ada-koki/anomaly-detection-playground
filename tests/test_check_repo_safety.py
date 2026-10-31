"""``scripts/check_repo_safety.py`` の純粋関数のテスト（git 不要）。

偽の秘密情報はリテラルで書かず、文字列連結で組み立てる。これによりこのテスト
ファイル自身が ``check_repo_safety.py`` の検査対象になっても誤検出しない。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from scripts import check_repo_safety as crs
except ImportError:  # pragma: no cover - パス構成が特殊な場合の保険
    import importlib.util

    _MODULE_PATH = _REPO_ROOT / "scripts" / "check_repo_safety.py"
    _spec = importlib.util.spec_from_file_location("check_repo_safety", _MODULE_PATH)
    if _spec is None or _spec.loader is None:  # pragma: no cover
        raise
    crs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(crs)


# --- 偽の秘密情報（リテラルで書かない） ---------------------------------


def fake_private_key_block() -> str:
    """秘密鍵ブロック風の行を組み立てる。"""
    return "-----BEGIN " + "RSA " + "PRIVATE KEY-----"


def fake_aws_access_key_id() -> str:
    """AWS アクセスキー ID 風の文字列を組み立てる。"""
    return "AKIA" + "Q" * 16


def fake_github_token() -> str:
    """GitHub トークン風の文字列を組み立てる。"""
    return "gh" + "p_" + "a" * 24


def fake_github_fine_grained_pat() -> str:
    """GitHub の fine-grained PAT 風の文字列を組み立てる。"""
    return "github" + "_pat_" + "b" * 24


def fake_openai_api_key() -> str:
    """OpenAI 風の API キー文字列を組み立てる。"""
    return "s" + "k-" + "c" * 24


def fake_slack_token() -> str:
    """Slack トークン風の文字列を組み立てる。"""
    return "xox" + "b-" + "1" * 12


def assigned_api_key_line() -> str:
    """``api_key = "..."`` 形式の代入行を組み立てる。"""
    return "api_key" + ' = "' + "d" * 24 + '"'


class SizeCheckTest(unittest.TestCase):
    """サイズ上限の検査。"""

    def test_size_below_limit_is_allowed(self) -> None:
        self.assertEqual(crs.check_size("benchmarks/x.json", 100, 1024), [])

    def test_size_equal_to_limit_is_allowed(self) -> None:
        self.assertEqual(crs.check_size("benchmarks/x.json", 1024, 1024), [])

    def test_size_over_limit_is_reported(self) -> None:
        violations = crs.check_size("benchmarks/big.json", 1025, 1024)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule, crs.RULE_SIZE)
        self.assertEqual(violations[0].path, "benchmarks/big.json")
        self.assertIsNone(violations[0].line)

    def test_default_max_bytes_is_used(self) -> None:
        violations = crs.check_size("big.bin", crs.DEFAULT_MAX_BYTES + 1)
        self.assertEqual(len(violations), 1)


class ExtensionCheckTest(unittest.TestCase):
    """禁止拡張子と画像の例外。"""

    def test_model_weight_extensions_are_forbidden(self) -> None:
        for path in (
            "weights/model.ckpt",
            "weights/model.pt",
            "weights/model.pth",
            "weights/model.onnx",
            "weights/model.bin",
            "weights/model.safetensors",
            "weights/model.h5",
            "weights/features.pkl",
            "weights/array.npy",
            "weights/array.npz",
        ):
            with self.subTest(path=path):
                violations = crs.check_extension(path)
                self.assertEqual(len(violations), 1)
                self.assertEqual(violations[0].rule, crs.RULE_EXTENSION)

    def test_archive_extensions_are_forbidden(self) -> None:
        for path in (
            "data/dataset.zip",
            "data/dataset.tar",
            "data/dataset.tar.gz",
            "data/dataset.tgz",
            "data/dataset.tar.xz",
            "data/dataset.7z",
        ):
            with self.subTest(path=path):
                self.assertEqual(len(crs.check_extension(path)), 1)

    def test_extension_check_is_case_insensitive(self) -> None:
        self.assertEqual(len(crs.check_extension("weights/MODEL.PT")), 1)

    def test_image_outside_docs_images_is_forbidden(self) -> None:
        for path in (
            "sample.png",
            "benchmarks/issue7/preview.jpg",
            "docs/other/figure.jpeg",
            "results/x.bmp",
        ):
            with self.subTest(path=path):
                violations = crs.check_extension(path)
                self.assertEqual(len(violations), 1)
                self.assertEqual(violations[0].rule, crs.RULE_EXTENSION)

    def test_image_under_docs_images_is_allowed(self) -> None:
        for path in (
            "docs/images/figure.png",
            "docs/images/sub/figure.jpg",
            "docs/images/scan.tiff",
        ):
            with self.subTest(path=path):
                self.assertEqual(crs.check_extension(path), [])

    def test_model_weight_under_docs_images_is_still_forbidden(self) -> None:
        violations = crs.check_extension("docs/images/model.onnx")
        self.assertEqual(len(violations), 1)

    def test_other_extensions_are_allowed(self) -> None:
        for path in (
            "adbench/core.py",
            "docs/data_and_ci.md",
            "benchmarks/issue7/manifests/run.json",
            "scripts/check_repo_safety.py",
            "Makefile",
        ):
            with self.subTest(path=path):
                self.assertEqual(crs.check_extension(path), [])


class DirectoryCheckTest(unittest.TestCase):
    """禁止ディレクトリ接頭辞の検査。"""

    def test_forbidden_prefixes_are_reported(self) -> None:
        for path in (
            "datasets/mvtec/bottle/000.png",
            "results/run1/metrics.csv",
            "artifacts/export/model.json",
            "checkpoints/epoch_10.ckpt",
        ):
            with self.subTest(path=path):
                violations = crs.check_directory(path)
                self.assertEqual(len(violations), 1)
                self.assertEqual(violations[0].rule, crs.RULE_DIRECTORY)

    def test_dot_slash_prefix_is_normalized(self) -> None:
        violations = crs.check_directory("./datasets/sample.txt")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].path, "datasets/sample.txt")

    def test_normal_paths_are_allowed(self) -> None:
        for path in (
            "adbench/core.py",
            "benchmarks/issue7/manifests/run.json",
            "docs/verification/2025-01-01.md",
            "my_datasets/notes.md",
        ):
            with self.subTest(path=path):
                self.assertEqual(crs.check_directory(path), [])


class SecretCheckTest(unittest.TestCase):
    """秘密情報らしき文字列の検査。"""

    def test_private_key_block_is_detected(self) -> None:
        text = fake_private_key_block() + "\n"
        violations = crs.check_secrets("deploy/key.txt", text)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule, crs.RULE_SECRET)
        self.assertEqual(violations[0].line, 1)

    def test_aws_access_key_id_is_detected(self) -> None:
        violations = crs.check_secrets("notes.txt", "id: " + fake_aws_access_key_id())
        self.assertEqual(len(violations), 1)
        self.assertIn("aws_access_key_id", violations[0].message)

    def test_github_tokens_are_detected(self) -> None:
        for name, token in (
            ("github_token", fake_github_token()),
            ("github_fine_grained_pat", fake_github_fine_grained_pat()),
        ):
            with self.subTest(name=name):
                violations = crs.check_secrets("ci.txt", token)
                self.assertEqual(len(violations), 1)
                self.assertIn(name, violations[0].message)

    def test_openai_api_key_is_detected(self) -> None:
        violations = crs.check_secrets("env.txt", fake_openai_api_key())
        self.assertEqual(len(violations), 1)
        self.assertIn("openai_api_key", violations[0].message)

    def test_slack_token_is_detected(self) -> None:
        violations = crs.check_secrets("env.txt", fake_slack_token())
        self.assertEqual(len(violations), 1)
        self.assertIn("slack_token", violations[0].message)

    def test_assigned_api_key_is_detected(self) -> None:
        violations = crs.check_secrets("config.py", assigned_api_key_line())
        self.assertEqual(len(violations), 1)
        self.assertIn("assigned_api_key", violations[0].message)

    def test_short_or_placeholder_values_are_not_detected(self) -> None:
        text = "\n".join(
            (
                "api_key = " + '"' + "short" + '"',
                "api_key = " + '"' + "<your-key-here>" + '"',
                "see docs/data_and_ci.md for details",
            )
        )
        self.assertEqual(crs.check_secrets("config.py", text), [])

    def test_line_numbers_are_reported(self) -> None:
        token = fake_openai_api_key()
        text = "first line\nsecond line\nvalue: " + token + "\n"
        violations = crs.check_secrets("env.txt", text)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line, 3)

    def test_same_rule_on_same_line_is_reported_once(self) -> None:
        token = fake_openai_api_key()
        violations = crs.check_secrets("env.txt", token + " " + token)
        self.assertEqual(len(violations), 1)

    def test_detected_values_are_not_rendered(self) -> None:
        cases = (
            ("private_key_block", fake_private_key_block()),
            ("aws_access_key_id", fake_aws_access_key_id()),
            ("github_token", fake_github_token()),
            ("github_fine_grained_pat", fake_github_fine_grained_pat()),
            ("openai_api_key", fake_openai_api_key()),
            ("slack_token", fake_slack_token()),
            ("assigned_api_key", assigned_api_key_line()),
        )
        for name, secret in cases:
            with self.subTest(name=name):
                text = "value: " + secret + "\n"
                violations = crs.check_secrets("config/settings.txt", text)
                self.assertTrue(violations, name)
                rendered = crs.format_violations(violations)
                self.assertNotIn(secret, rendered)
                self.assertNotIn("Q" * 16, rendered)
                self.assertNotIn("a" * 24, rendered)
                self.assertNotIn("d" * 24, rendered)
                self.assertIn("config/settings.txt", rendered)

    def test_clean_text_has_no_violation(self) -> None:
        text = "\n".join(
            (
                "# ベンチマークのメモ",
                "python -m unittest discover -s tests -v",
                "model = 'patchcore'",
                "image_size = 256",
                "",
            )
        )
        self.assertEqual(crs.check_secrets("docs/verification/memo.md", text), [])


class CheckFileTest(unittest.TestCase):
    """1 ファイル分のまとめ判定。"""

    def test_clean_file_has_no_violation(self) -> None:
        text = "key,value\naccuracy,0.99\n"
        self.assertEqual(
            crs.check_file("benchmarks/issue7/summary.csv", len(text), text=text),
            [],
        )

    def test_size_and_extension_violations_are_both_reported(self) -> None:
        violations = crs.check_file("models/model.pt", 2048, text=None, max_bytes=1024)
        rules = {violation.rule for violation in violations}
        self.assertEqual(rules, {crs.RULE_SIZE, crs.RULE_EXTENSION})

    def test_binary_file_skips_secret_scan(self) -> None:
        # text=None はバイナリ（または読み込み省略）を表し、秘密情報スキャンしない。
        violations = crs.check_file("benchmarks/manifest.json", 10, text=None)
        self.assertEqual(violations, [])

    def test_secret_violation_is_included_for_text_files(self) -> None:
        text = "token: " + fake_github_token() + "\n"
        violations = crs.check_file("scripts/config.py", len(text), text=text)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule, crs.RULE_SECRET)


class FormatViolationsTest(unittest.TestCase):
    """出力整形。"""

    def test_empty_list_becomes_empty_string(self) -> None:
        self.assertEqual(crs.format_violations([]), "")

    def test_line_number_is_included_when_present(self) -> None:
        violation = crs.Violation(
            path="a/b.txt", rule="secret", message="秘密情報らしき文字列", line=7
        )
        self.assertEqual(
            crs.format_violations([violation]),
            "a/b.txt:7: [secret] 秘密情報らしき文字列",
        )

    def test_line_number_is_omitted_when_absent(self) -> None:
        violation = crs.Violation(
            path="a/b.bin", rule="size", message="サイズ超過", line=None
        )
        self.assertEqual(
            crs.format_violations([violation]),
            "a/b.bin: [size] サイズ超過",
        )


if __name__ == "__main__":
    unittest.main()
