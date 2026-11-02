"""adbench.patchcore_memory のユニットテスト。

torch / anomalib に依存しない純粋関数のみを対象にする。標準ライブラリの
unittest だけを使い、重い依存を要求しない。
"""

from __future__ import annotations

import unittest

from adbench.patchcore_memory import (
    build_mitigation_record,
    bytes_to_mib,
    check_memory_budget,
    chunk_coreset_size,
    estimate_patchcore_feature_memory,
    patches_per_image,
    plan_chunks,
    read_mem_available_mib,
)


class PatchesPerImageTest(unittest.TestCase):
    def test_default_stride_256(self) -> None:
        # (256 // 8) ** 2 = 32 * 32 = 1024
        self.assertEqual(patches_per_image(256), 1024)

    def test_small_sizes(self) -> None:
        self.assertEqual(patches_per_image(64, 8), 64)
        self.assertEqual(patches_per_image(32, 8), 16)

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            patches_per_image(0)
        with self.assertRaises(ValueError):
            patches_per_image(256, 0)
        # image_size < stride でパッチ数 0 になるケース
        with self.assertRaises(ValueError):
            patches_per_image(4, 8)


class BytesToMibTest(unittest.TestCase):
    def test_one_mib(self) -> None:
        self.assertEqual(bytes_to_mib(1024 * 1024), 1.0)

    def test_half_mib(self) -> None:
        self.assertEqual(bytes_to_mib(512 * 1024), 0.5)

    def test_zero(self) -> None:
        self.assertEqual(bytes_to_mib(0), 0.0)


class PlanChunksTest(unittest.TestCase):
    def test_exact_division(self) -> None:
        self.assertEqual(plan_chunks(100, 10), [10] * 10)

    def test_with_remainder(self) -> None:
        self.assertEqual(plan_chunks(25, 10), [10, 10, 5])

    def test_sum_equals_total(self) -> None:
        sizes = plan_chunks(1234, 100)
        self.assertEqual(len(sizes), 13)
        self.assertEqual(sum(sizes), 1234)
        self.assertEqual(sizes[-1], 34)

    def test_zero_total(self) -> None:
        self.assertEqual(plan_chunks(0, 10), [])

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_chunks(-1, 10)
        with self.assertRaises(ValueError):
            plan_chunks(10, 0)


class ChunkCoresetSizeTest(unittest.TestCase):
    def test_basic_ratio(self) -> None:
        self.assertEqual(chunk_coreset_size(100, 0.1), 10)

    def test_minimum_one(self) -> None:
        # int(3 * 0.001) = 0 → 下限 1 に持ち上がる
        self.assertEqual(chunk_coreset_size(3, 0.001), 1)

    def test_upper_bound(self) -> None:
        # ratio=1.0 で n 件
        self.assertEqual(chunk_coreset_size(100, 1.0), 100)

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_coreset_size(0, 0.1)
        with self.assertRaises(ValueError):
            chunk_coreset_size(10, 0.0)
        with self.assertRaises(ValueError):
            chunk_coreset_size(10, 1.5)


class EstimatePatchcoreMemoryTest(unittest.TestCase):
    def test_global_feature_bytes(self) -> None:
        est = estimate_patchcore_feature_memory(
            n_images=900, image_size=256, feature_dim=1536
        )
        # feature_bytes = 900 * 1024 パッチ * 1536 次元 * 4 バイト
        self.assertEqual(est.feature_bytes, 900 * 1024 * 1536 * 4)
        # global は embedding_store + vstack で 2 倍 + 射影で 2*feature_bytes を超える
        self.assertGreater(est.peak_bytes_estimate, 2 * est.feature_bytes)
        self.assertEqual(est.coreset_mode, "global")
        self.assertIsNone(est.chunk_images)

    def test_chunked_smaller_than_global(self) -> None:
        global_est = estimate_patchcore_feature_memory(
            n_images=900, image_size=256, feature_dim=1536
        )
        chunked_est = estimate_patchcore_feature_memory(
            n_images=900,
            image_size=256,
            feature_dim=1536,
            coreset_mode="chunked",
            chunk_images=100,
        )
        self.assertLess(
            chunked_est.peak_bytes_estimate, global_est.peak_bytes_estimate
        )
        self.assertEqual(chunked_est.coreset_mode, "chunked")
        self.assertEqual(chunked_est.chunk_images, 100)

    def test_invalid_n_images_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_patchcore_feature_memory(n_images=0, image_size=256)

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_patchcore_feature_memory(
                n_images=10, image_size=256, coreset_mode="bogus"
            )

    def test_chunked_requires_chunk_images(self) -> None:
        with self.assertRaises(ValueError):
            estimate_patchcore_feature_memory(
                n_images=10,
                image_size=256,
                coreset_mode="chunked",
                chunk_images=None,
            )


class CheckMemoryBudgetTest(unittest.TestCase):
    def test_no_limit_returns_true(self) -> None:
        ok, message = check_memory_budget(10 * 1024 * 1024, None)
        self.assertTrue(ok)
        self.assertIsInstance(message, str)

    def test_within_limit_returns_true(self) -> None:
        ok, _ = check_memory_budget(10 * 1024 * 1024, 100.0)
        self.assertTrue(ok)

    def test_exceeds_limit_returns_false(self) -> None:
        ok, _ = check_memory_budget(200 * 1024 * 1024, 100.0)
        self.assertFalse(ok)

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            check_memory_budget(-1, 100.0)


class ReadMemAvailableMibTest(unittest.TestCase):
    def test_parse_meminfo_text(self) -> None:
        text = "MemTotal: 16000000 kB\nMemAvailable: 8192000 kB\n"
        # 8192000 KiB = 8000 MiB
        self.assertEqual(read_mem_available_mib(text), 8000.0)

    def test_missing_key_returns_none(self) -> None:
        text = "MemTotal: 16000000 kB\nMemFree: 1000 kB\n"
        self.assertIsNone(read_mem_available_mib(text))


class BuildMitigationRecordTest(unittest.TestCase):
    def test_chunked_is_protocol_deviation(self) -> None:
        record = build_mitigation_record(
            "chunked",
            100,
            0.1,
            image_size=256,
            feature_dim=1536,
            train_image_count=900,
        )
        self.assertTrue(record["is_protocol_deviation"])
        self.assertEqual(record["coreset_mode"], "chunked")
        self.assertEqual(record["chunk_images"], 100)
        self.assertIsNotNone(record["approximation"])

    def test_global_is_not_deviation(self) -> None:
        record = build_mitigation_record("global", None, 0.1)
        self.assertFalse(record["is_protocol_deviation"])
        self.assertEqual(record["coreset_mode"], "global")
        self.assertIsNone(record["approximation"])

    def test_global_with_chunk_images_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_mitigation_record("global", 100, 0.1)

    def test_chunked_without_chunk_images_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_mitigation_record("chunked", None, 0.1)


if __name__ == "__main__":
    unittest.main()
