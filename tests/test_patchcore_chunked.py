"""adbench.patchcore_chunked のユニットテスト。

torch が無い環境では重い依存を要求するテストを skip する。
make_chunked_patchcore のバリデーションは anomalib を import する前に
評価されるが、以降のテストは anomalib（KCenterGreedy）に依存するため
anomalib 不在時も skip する。
"""

from __future__ import annotations

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ANOMALIB = importlib.util.find_spec("anomalib") is not None


def _select_head(chunk, k):
    """先頭 k 行を返すダミーの select_fn。"""
    return chunk[:k].clone()


class ReduceStoreInChunksTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch がインストールされていません")
    def test_two_chunks_and_remainder(self) -> None:
        import torch

        from adbench.patchcore_chunked import reduce_store_in_chunks

        store = [
            torch.arange(10 * 3, dtype=torch.float32).reshape(10, 3),
            torch.arange(7 * 3, dtype=torch.float32).reshape(7, 3) + 1000.0,
            torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3) + 2000.0,
        ]
        original_lengths = [int(t.shape[0]) for t in store]
        self.assertEqual(original_lengths, [10, 7, 5])

        selected, remainder, stats = reduce_store_in_chunks(
            store,
            chunk_rows=8,
            ratio=0.25,
            select_fn=_select_head,
        )

        # 22 行 / 8 行 = 2 チャンク（8, 8）+ 端数 6 行
        self.assertEqual(len(selected), 2)
        for picked in selected:
            # int(8 * 0.25) = 2
            self.assertEqual(int(picked.shape[0]), 2)
            self.assertEqual(int(picked.shape[1]), 3)

        self.assertIsNotNone(remainder)
        assert remainder is not None
        self.assertEqual(int(remainder.shape[0]), 6)

        self.assertEqual(len(stats), 2)
        for row in stats:
            self.assertEqual(row["rows"], 8)
            self.assertEqual(row["selected"], 2)
            self.assertIn("seconds", row)

        # 元 store のリストと各行数は変化しない
        self.assertEqual(len(store), len(original_lengths))
        for tensor, length in zip(store, original_lengths):
            self.assertEqual(int(tensor.shape[0]), length)

    @unittest.skipUnless(HAS_TORCH, "torch がインストールされていません")
    def test_carry_is_prepended(self) -> None:
        import torch

        from adbench.patchcore_chunked import reduce_store_in_chunks

        carry = torch.zeros(4, 3, dtype=torch.float32)
        store = [torch.ones(6, 3, dtype=torch.float32)]

        selected, remainder, stats = reduce_store_in_chunks(
            store,
            chunk_rows=8,
            ratio=0.5,
            select_fn=_select_head,
            carry=carry,
        )

        # carry(4) + store(6) = 10 行 → 1 チャンク（8 行）+ 端数 2 行
        self.assertEqual(len(selected), 1)
        # int(8 * 0.5) = 4 → 先頭 4 行は carry のゼロ
        self.assertEqual(int(selected[0].shape[0]), 4)
        self.assertTrue(bool(torch.all(selected[0] == 0).item()))

        self.assertIsNotNone(remainder)
        assert remainder is not None
        self.assertEqual(int(remainder.shape[0]), 2)

    @unittest.skipUnless(HAS_TORCH, "torch がインストールされていません")
    def test_keep_remainder_false_returns_none(self) -> None:
        import torch

        from adbench.patchcore_chunked import reduce_store_in_chunks

        store = [torch.ones(10, 2, dtype=torch.float32)]

        selected, remainder, _stats = reduce_store_in_chunks(
            store,
            chunk_rows=8,
            ratio=0.5,
            select_fn=_select_head,
            keep_remainder=False,
        )
        self.assertEqual(len(selected), 1)
        self.assertIsNone(remainder)

    @unittest.skipUnless(HAS_TORCH, "torch がインストールされていません")
    def test_invalid_arguments_raise(self) -> None:
        import torch

        from adbench.patchcore_chunked import reduce_store_in_chunks

        store = [torch.ones(10, 2, dtype=torch.float32)]

        with self.assertRaises(ValueError):
            reduce_store_in_chunks(
                store, chunk_rows=0, ratio=0.5, select_fn=_select_head
            )
        with self.assertRaises(ValueError):
            reduce_store_in_chunks(
                store, chunk_rows=8, ratio=0.0, select_fn=_select_head
            )
        with self.assertRaises(ValueError):
            reduce_store_in_chunks(
                store, chunk_rows=8, ratio=1.5, select_fn=_select_head
            )

    @unittest.skipUnless(HAS_TORCH, "torch がインストールされていません")
    def test_select_fn_returning_1d_raises(self) -> None:
        import torch

        from adbench.patchcore_chunked import reduce_store_in_chunks

        store = [torch.ones(10, 3, dtype=torch.float32)]

        def bad_select(chunk, _k):
            return chunk[0]  # 1 次元

        with self.assertRaises(ValueError):
            reduce_store_in_chunks(
                store, chunk_rows=8, ratio=0.5, select_fn=bad_select
            )


class KCenterSelectTest(unittest.TestCase):
    @unittest.skipUnless(
        HAS_TORCH and HAS_ANOMALIB, "torch / anomalib がインストールされていません"
    )
    def test_select_k_rows_from_random_matrix(self) -> None:
        import torch

        from adbench.patchcore_chunked import _kcenter_select

        torch.manual_seed(0)
        matrix = torch.randn(200, 8, dtype=torch.float32)
        picked = _kcenter_select(matrix, 20)

        self.assertEqual(tuple(picked.shape), (20, 8))
        # 各行は元行列のいずれかの行と完全一致する（k-center は部分集合を返す）
        for row in picked:
            matches = bool((matrix == row).all(dim=1).any().item())
            self.assertTrue(matches)

    @unittest.skipUnless(HAS_TORCH, "torch がインストールされていません")
    def test_k_greater_than_rows_raises(self) -> None:
        import torch

        from adbench.patchcore_chunked import _kcenter_select

        matrix = torch.randn(10, 4, dtype=torch.float32)
        with self.assertRaises(ValueError):
            _kcenter_select(matrix, 11)

    @unittest.skipUnless(HAS_TORCH, "torch がインストールされていません")
    def test_k_zero_raises(self) -> None:
        import torch

        from adbench.patchcore_chunked import _kcenter_select

        matrix = torch.randn(10, 4, dtype=torch.float32)
        with self.assertRaises(ValueError):
            _kcenter_select(matrix, 0)


@unittest.skipUnless(
    HAS_ANOMALIB, "anomalib がインストールされていません"
)
class MakeChunkedPatchcoreValidationTest(unittest.TestCase):
    """backbone 重みを構築せず、引数バリデーションだけを検証する。"""

    def test_zero_raises_value_error(self) -> None:
        from adbench.patchcore_chunked import make_chunked_patchcore

        with self.assertRaises(ValueError):
            make_chunked_patchcore(0)

    def test_negative_raises_value_error(self) -> None:
        from adbench.patchcore_chunked import make_chunked_patchcore

        with self.assertRaises(ValueError):
            make_chunked_patchcore(-1)

    def test_string_raises_type_error(self) -> None:
        from adbench.patchcore_chunked import make_chunked_patchcore

        with self.assertRaises(TypeError):
            make_chunked_patchcore("2")  # type: ignore[arg-type]

    def test_bool_raises_type_error(self) -> None:
        from adbench.patchcore_chunked import make_chunked_patchcore

        # bool は int のサブクラスなので明示的に弾く
        with self.assertRaises(TypeError):
            make_chunked_patchcore(True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
