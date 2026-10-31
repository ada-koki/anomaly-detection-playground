"""adbench.dataset_views.evaluation_scope のテスト（Issue #32）。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adbench.dataset_views import (  # noqa: E402
    MVTEC_AD2_EXCLUDED,
    MVTEC_AD2_SOURCE,
    evaluation_scope,
)


def _get(scope: object, key: str):
    """dict / dataclass / NamedTuple いずれでも値を取り出す。"""
    if isinstance(scope, dict):
        return scope[key]
    return getattr(scope, key)


class EvaluationScopeTest(unittest.TestCase):
    def test_mvtec_ad2_with_manifest_copies_source_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = {
                "source": "custom/source",
                "excluded": ["private_a", "private_b"],
                "counts": {"train": 3, "test": 10},
            }
            (root / "view_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            scope = evaluation_scope("mvtec_ad2", root, "vial")
            self.assertEqual(_get(scope, "source"), "custom/source")
            self.assertEqual(_get(scope, "excluded"), ["private_a", "private_b"])
            self.assertTrue(_get(scope, "public_test_only"))
            self.assertTrue(_get(scope, "view_manifest_found"))

    def test_mvtec_ad2_without_manifest_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scope = evaluation_scope("mvtec_ad2", root, "vial")
            self.assertEqual(_get(scope, "source"), MVTEC_AD2_SOURCE)
            self.assertEqual(_get(scope, "excluded"), list(MVTEC_AD2_EXCLUDED))
            self.assertTrue(_get(scope, "public_test_only"))
            self.assertFalse(_get(scope, "view_manifest_found"))
            self.assertIsNone(_get(scope, "counts"))

    def test_visa_public_test_only_none(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scope = evaluation_scope("visa", root, "candle")
            self.assertIsNone(_get(scope, "public_test_only"))
            self.assertEqual(_get(scope, "source"), "split_csv/1cls.csv")

    def test_broken_manifest_marked_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "view_manifest.json").write_text("{not json", encoding="utf-8")
            scope = evaluation_scope("mvtec_ad2", root, "vial")
            self.assertFalse(_get(scope, "view_manifest_found"))

    def test_unsupported_dataset_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                evaluation_scope("unknown_dataset", root, "x")


if __name__ == "__main__":
    unittest.main()
