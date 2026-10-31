# Issue #14 検証記録: 欠陥サイズ別メトリクス

## 1. 目的

画像とマスクを持つデータセットで、欠陥サイズ帯ごとの検出率・見逃し率と、正常画像の過検出率を集計できるようにする。本記録では、実装した `adbench/defect_size.py`、`adbench/raw_eval.py` の per-image 収集、`scripts/run_anomalib.py` の `report.json` 出力、`scripts/defect_size_report.py` の再集計を、MVTec AD bottle と MVTec-4K bottle の 6 run で確認した結果をまとめる。

数値は `benchmarks/issue14/defect_size_summary.csv` と `benchmarks/issue14/defect_size_summary.md` の生成物に基づく。表示は小数 4 桁に丸めている。1 条件につき 1 回の実行であり、反復・信頼区間はない。

## 2. 実装と対象

- `adbench/defect_size.py`: 面積率によるサイズ帯分類、元解像度マスクの統計、per-image レコード、サイズ帯 × 異常種別の集計、JSONL 入出力。
- `adbench/raw_eval.py`: `collect_per_image=True` で画像ごとのレコードを収集。サイズはリサイズ前の元解像度マスク、`map_max_in_mask` は `anomaly_map` 解像度へ最近傍リサイズしたマスクで計算。
- `scripts/run_anomalib.py`: `report.json` に `defect_size`（閾値・注記・`per_image_path`・`mask_vanished_after_resize`）を保存し、run ディレクトリに `per_image.jsonl` を出力。
- `scripts/defect_size_report.py`: 複数 run の `per_image.jsonl` を再集計し、CSV / Markdown / PNG を出力。
- テスト: `tests/test_defect_size.py`。全体 281 件 OK、`scripts/check_repo_safety.py` の違反なし。

## 3. 実験条件

### 3.1 共通条件

- CPU 8 スレッド、GPU なし。各条件 1 回実行。
- プロトコル: `mvtec_ad_v1`（`val_split from_test 0.2`, `seed 42`, `batch 1`, `num_workers 0`, `cpu_threads 8`）。
- Anomalib 2.7.0、PyTorch CPU 版。
- 閾値は各 run の検証集合から求めた `PostProcessor` の `F1AdaptiveThreshold`（生スケール）。テスト集合では調整していない。
- データ:
  - MVTec AD bottle: test 67 枚（異常 51、正常 16）。
  - MVTec-4K bottle: test 132 枚（異常 92、正常 40）。異常種別 `tiny` は 54 枚で、うち 52 枚が tiny 帯（面積率 0.0000834〜0.00098）、2 枚が small 帯。tiny 帯の 52 枚はすべて異常種別 `tiny`。
  - MVTec AD bottle には tiny 帯の画像がない（n=0）。
- 全 run で `mask_vanished_after_resize = 0`。今回の作業解像度では、元解像度マスクが `anomaly_map` 解像度で消失した画像はなかった。
- 全 run でマスク無しの異常画像は 0。

### 3.2 実行一覧

| run | データ | model | 空間方式 | 作業解像度 | tile / stride | image AUROC (raw) | train stage [s] | e2e P50 [ms] | e2e P95 [ms] | train peak RSS [MiB] |
|---|---|---|---|---|---|---|---|---|---|---|
| mvtecad_padim_256 | MVTec AD bottle | PaDiM | global_resize | 256x256 | - / - | 0.9963 | 13.0220 | 50.4663 | 61.4060 | 1689.9000 |
| 4k_padim_global_256 | MVTec-4K bottle | PaDiM | global_resize | 256x256 | - / - | 0.9262 | 102.8230 | 498.8833 | 595.3087 | 1691.7300 |
| 4k_padim_global_512 | MVTec-4K bottle | PaDiM | global_resize | 512x512 | - / - | 0.9133 | 140.1700 | 543.9514 | 1178.1499 | 4752.3900 |
| 4k_padim_tile512_t256_s256 | MVTec-4K bottle | PaDiM | tiled | 512x512 | 256 / 256 | 0.9196 | 120.0020 | 669.8523 | 1645.8466 | 4881.9700 |
| 4k_padim_tile512_t256_s128 | MVTec-4K bottle | PaDiM | tiled | 512x512 | 256 / 128 | 0.9202 | 159.6540 | 577.2974 | 635.0381 | 4959.1900 |
| 4k_patchcore_global_256 | MVTec-4K bottle | PatchCore | global_resize | 256x256 | - / - | 0.8746 | 544.3470 | 611.9619 | 1461.3098 | 2579.1900 |

MVTec AD bottle と MVTec-4K bottle はデータ規模が異なるため、train stage 秒を run 間で直接比較しない。512px / タイル条件の train peak RSS は 4752〜4959 MiB で、空きメモリが小さい環境では再実行時に注意が必要である。

### 3.3 閾値

| run | image_threshold | pixel_threshold | 出所 |
|---|---|---|---|
| mvtecad_padim_256 | 45.8716 | 27.6497 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_global_256 | 22.1358 | 38.0946 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_global_512 | 40.0044 | 30.5288 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_tile512_t256_s256 | 37.4010 | 30.5229 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_tile512_t256_s128 | 38.2305 | 31.7233 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_patchcore_global_256 | 23.3793 | 41.4502 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |

## 4. 定義

詳細は `docs/defect_size_metrics.md` を参照する。本記録で使う定義は次のとおり。

- 欠陥サイズ: 元画像解像度のマスク面積率 `マスク画素数 / 元画像画素数`。外接矩形は `[y0, x0, y1, x1]`（`y1` / `x1` は排他的）。
- サイズ帯: `tiny [0, 0.001)`、`small [0.001, 0.01)`、`large [0.01, 1.0]`。下限を含み、最後の帯だけ上限を含む。
- 画像単位の検出: `score >= image_threshold`。
- 領域単位ヒット: `map_max_in_mask >= pixel_threshold`。`map_max_in_mask` は `anomaly_map` をマスクの内側だけで見た最大値。
- 検出率・見逃し率の分母: 異常画像（`label=1`）のみ。正常画像は過検出率の表で別に扱う。
- 領域ヒット率の分母: マスクがあり、かつ `anomaly_map` 解像度へリサイズしてもマスクが残った異常画像。
- n=0 のセルは率を出さない（表では `-`）。実装は `n < min_samples=5` に「サンプル数が少ないため参考値」の注記を付ける。今回の small 帯は n=5 のため注記は付かないが、件数が少ないため参考値として扱う。

## 5. 結果

### 5.1 overall

| run | n | 検出 | 検出率 | 見逃し率 | 領域ヒット率 | 正常画像 FPR | image AUROC (raw) |
|---|---|---|---|---|---|---|---|
| mvtecad_padim_256 | 51 | 47 | 0.9216 | 0.0784 | 0.9804 | 0.0000 | 0.9963 |
| 4k_padim_global_256 | 92 | 80 | 0.8696 | 0.1304 | 0.6413 | 0.2000 | 0.9262 |
| 4k_padim_global_512 | 92 | 76 | 0.8261 | 0.1739 | 0.8696 | 0.1750 | 0.9133 |
| 4k_padim_tile512_t256_s256 | 92 | 80 | 0.8696 | 0.1304 | 0.8696 | 0.2250 | 0.9196 |
| 4k_padim_tile512_t256_s128 | 92 | 79 | 0.8587 | 0.1413 | 0.8587 | 0.1500 | 0.9202 |
| 4k_patchcore_global_256 | 92 | 90 | 0.9783 | 0.0217 | 0.4565 | 0.8500 | 0.8746 |

### 5.2 サイズ帯別（異常画像のみ）

| run | 帯 | n | 検出 | 検出率 | 見逃し率 | 領域ヒット率 |
|---|---|---|---|---|---|---|
| mvtecad_padim_256 | tiny | 0 | 0 | - | - | - |
| mvtecad_padim_256 | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 |
| mvtecad_padim_256 | large | 49 | 45 | 0.9184 | 0.0816 | 0.9796 |
| 4k_padim_global_256 | tiny | 52 | 40 | 0.7692 | 0.2308 | 0.3846 |
| 4k_padim_global_256 | small | 5 | 5 | 1.0000 | 0.0000 | 0.8000 |
| 4k_padim_global_256 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 |
| 4k_padim_global_512 | tiny | 52 | 36 | 0.6923 | 0.3077 | 0.7692 |
| 4k_padim_global_512 | small | 5 | 5 | 1.0000 | 0.0000 | 1.0000 |
| 4k_padim_global_512 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 |
| 4k_padim_tile512_t256_s256 | tiny | 52 | 40 | 0.7692 | 0.2308 | 0.7692 |
| 4k_padim_tile512_t256_s256 | small | 5 | 5 | 1.0000 | 0.0000 | 1.0000 |
| 4k_padim_tile512_t256_s256 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 |
| 4k_padim_tile512_t256_s128 | tiny | 52 | 39 | 0.7500 | 0.2500 | 0.7500 |
| 4k_padim_tile512_t256_s128 | small | 5 | 5 | 1.0000 | 0.0000 | 1.0000 |
| 4k_padim_tile512_t256_s128 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 |
| 4k_patchcore_global_256 | tiny | 52 | 50 | 0.9615 | 0.0385 | 0.0769 |
| 4k_patchcore_global_256 | small | 5 | 5 | 1.0000 | 0.0000 | 0.6000 |
| 4k_patchcore_global_256 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 |

### 5.3 正常画像の過検出

| run | 正常 n | FP | FPR | 正常画素の平均過検出率 |
|---|---|---|---|---|
| mvtecad_padim_256 | 16 | 0 | 0.0000 | 0.0006 |
| 4k_padim_global_256 | 40 | 8 | 0.2000 | 0.0000 |
| 4k_padim_global_512 | 40 | 7 | 0.1750 | 0.0005 |
| 4k_padim_tile512_t256_s256 | 40 | 9 | 0.2250 | 0.0004 |
| 4k_padim_tile512_t256_s128 | 40 | 6 | 0.1500 | 0.0003 |
| 4k_patchcore_global_256 | 40 | 34 | 0.8500 | 0.0000 |

## 6. 観察

- MVTec-4K bottle の tiny 帯（n=52）は、5 run すべてで small / large 帯より画像単位の検出率が低い。small / large は全 run で 1.0000、tiny は 0.6923〜0.9615 である。
- PaDiM の tiny 領域ヒット率は、global 256 の 0.3846 から、global 512 で 0.7692、tiled s256 で 0.7692、tiled s128 で 0.7500 に上がる。一方、tiny の画像単位検出率は 0.7692、0.6923、0.7692、0.7500 で、global 256 を上回らない。領域ヒット率の上昇は、画像単位の検出率の向上には結びついていない。
- PatchCore global 256 の tiny 検出率は 0.9615 と高いが、正常画像 FPR は 0.8500、image AUROC (raw) は 0.8746 である。PaDiM global 256 は tiny 0.7692、FPR 0.2000、AUROC 0.9262 であり、閾値が異なるため検出率だけを性能比較に使えない。
- PatchCore global 256 の tiny 領域ヒット率は 0.0769（52 枚中 4 枚）で全 run 中最低である。画像単位では検出されても、マスク内の `anomaly_map` 最大値が `pixel_threshold` 41.4502 に届かない画像が多い。
- 全 run で `mask_vanished_after_resize = 0` である。tiny の領域ヒット率の低さは、マスクが作業解像度で消失したことでは説明できない。
- MVTec AD bottle は tiny 帯 n=0 である。large 帯は 49 枚中 45 枚（0.9184）を検出し、異常種別では `broken_large` 18/18、`broken_small` 19/19、`contamination` 10/14（0.7143）である。`contamination` の見逃し 4 枚は `contamination/large`（13 枚中 9 枚、0.6923）に含まれる。
- MVTec-4K bottle の small 帯は n=5、MVTec AD bottle は n=2 である。全 run で検出率 1.0000 だが、サンプル数が少なく、サイズ帯間の比較には使えない。
- 正常画像 FPR は 0.0000〜0.8500 と条件で大きく異なる。`image_threshold` も 22.1358〜45.8716 で異なるため、検出率と FPR は閾値とのトレードオフとしてセットで読む必要がある。PatchCore global 256 の `image_threshold` 23.3793 は、同 run のテスト正常画像のスコア分布に対して低く、FPR 0.8500 になっている（検証集合由来の閾値であり、テストでは調整していない）。
- MVTec-4K bottle の 5 run の image AUROC (raw) は、Issue #13 の `benchmarks/issue13/mvtec4k_bottle/runs.csv` と小数 4 桁で一致した（PaDiM 0.9262 / 0.9133 / 0.9196 / 0.9202、PatchCore 0.8746）。同一コード・同一 seed の再実行で再現している。train stage 秒は実行ごとに異なる。

## 7. 未検証事項

- 各条件 1 回のみの実行であり、反復ばらつき・信頼区間・統計的有意性は評価していない。
- MVTec AD bottle と MVTec-4K bottle のみ。他カテゴリ・他データセットでの挙動は未検証。
- 閾値を固定した比較をしていない。条件ごとに `image_threshold` / `pixel_threshold` が異なるため、検出率と FPR のトレードオフを分離できていない。
- ピクセル単位の PRO など、画素レベルの総合指標は未評価。`region_hit_rate` は画像単位の二値判定である。
- tiny 帯の検出率低下が、作業解像度・タイル方式・モデルのどれに由来するかは、今回の条件だけでは分離していない。
- small 帯は n=5（MVTec AD は n=2）と少なく、サイズ帯ごとの率の比較には注意が必要。
- `same_as_test` プロトコルでの楽観バイアスの大きさは未検証。本実験は `from_test` で val / test を分けている。
- EfficientAD は未実行。
- マスク消失が 0 だったのは今回の作業解像度・データ範囲である。他の解像度・データでの頻度は未検証。

## 8. 再現手順と成果物

```bash
python scripts/defect_size_report.py \
    --runs <run_dir1> <run_dir2> ... \
    --labels mvtecad_padim_256 4k_padim_global_256 ... \
    --out-dir benchmarks/issue14
```

各 run ディレクトリには `report.json` と `per_image.jsonl` が必要。`report.json` の `defect_size.thresholds` を閾値として使い、`per_image.jsonl` から再集計する。

成果物:

- `benchmarks/issue14/defect_size_summary.csv`
- `benchmarks/issue14/defect_size_summary.md`
- `benchmarks/issue14/detection_by_size.png`（サイズ帯別検出率の図）

## 9. 関連ドキュメント

- `docs/defect_size_metrics.md`: 指標の定義、分母、注記ルール、メモリ上の注意。
- `docs/tiling.md`: タイル分割の設定と検証。
- `docs/verification/issue13_tiling.md`: タイル分割推論の検証記録。
- `docs/verification/issue12_mvtec4k.md`: MVTec-4K のデータ整合と CPU 実行。
