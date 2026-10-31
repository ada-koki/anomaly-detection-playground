# 欠陥サイズ別レポート

`scripts/defect_size_report.py` の生成物。閾値は各 run の `report.json` にある `defect_size.thresholds`（検証集合由来）を使い、テスト集合では調整していない。

## 定義

- 欠陥サイズ: 元画像解像度のマスク面積率（マスク画素数 / 元画像画素数）。外接矩形は `[y0, x0, y1, x1]`（`y1` / `x1` は排他的）。
- サイズ帯: tiny [0, 0.001) / small [0.001, 0.01) / large [0.01, 1)。下限を含み上限を含まない（最後の帯のみ上限を含む）。
- 閾値: 検証集合から求めた `image_threshold`（画像スコア）と `pixel_threshold`（anomaly_map）。PostProcessor F1AdaptiveThreshold の生スケール値。
- 検出率・見逃し率の分母: 異常画像（label=1）のみ。正常画像は「正常画像の過検出」表で別に扱う。
- 率が `-` の場合は、分母が 0 または閾値が未計算であることを示す。

## 実行条件

| run | model | 空間方式 | 作業解像度 | tile | stride | train stage [s] | e2e P50 [ms] | e2e P95 [ms] | train peak RSS [MiB] | image AUROC (raw) |
|---|---|---|---|---|---|---|---|---|---|---|
| mvtecad_padim_256 | padim | global_resize | 256x256 | - | - | 13.0220 | 50.4663 | 61.4060 | 1689.9000 | 0.9963 |
| 4k_padim_global_256 | padim | global_resize | 256x256 | - | - | 102.8230 | 498.8833 | 595.3087 | 1691.7300 | 0.9262 |
| 4k_padim_global_512 | padim | global_resize | 512x512 | - | - | 140.1700 | 543.9514 | 1178.1499 | 4752.3900 | 0.9133 |
| 4k_padim_tile512_t256_s256 | padim | tiled | 512x512 | 256 | 256 | 120.0020 | 669.8523 | 1645.8466 | 4881.9700 | 0.9196 |
| 4k_padim_tile512_t256_s128 | padim | tiled | 512x512 | 256 | 128 | 159.6540 | 577.2974 | 635.0381 | 4959.1900 | 0.9202 |
| 4k_patchcore_global_256 | patchcore | global_resize | 256x256 | - | - | 544.3470 | 611.9619 | 1461.3098 | 2579.1900 | 0.8746 |

## 閾値

| run | image_threshold | pixel_threshold | 出所 |
|---|---|---|---|
| mvtecad_padim_256 | 45.8716 | 27.6497 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_global_256 | 22.1358 | 38.0946 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_global_512 | 40.0044 | 30.5288 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_tile512_t256_s256 | 37.4010 | 30.5229 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_padim_tile512_t256_s128 | 38.2305 | 31.7233 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |
| 4k_patchcore_global_256 | 23.3793 | 41.4502 | validation（PostProcessor F1AdaptiveThreshold, 生スケール） |

## 正常画像の過検出

| run | 正常 n | FP 件数 | FPR | 正常画素の平均過検出率 | 注記 |
|---|---|---|---|---|---|
| mvtecad_padim_256 | 16 | 0 | 0.0000 | 0.0006 | - |
| 4k_padim_global_256 | 40 | 8 | 0.2000 | 0.0000 | - |
| 4k_padim_global_512 | 40 | 7 | 0.1750 | 0.0005 | - |
| 4k_padim_tile512_t256_s256 | 40 | 9 | 0.2250 | 0.0004 | - |
| 4k_padim_tile512_t256_s128 | 40 | 6 | 0.1500 | 0.0003 | - |
| 4k_patchcore_global_256 | 40 | 34 | 0.8500 | 0.0000 | - |

## サイズ帯別（異常画像のみ）

| run | サイズ帯 | n | 検出 | 検出率 | 見逃し率 | 領域ヒット率 | マスク無し | 注記 |
|---|---|---|---|---|---|---|---|---|
| mvtecad_padim_256 | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| mvtecad_padim_256 | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| mvtecad_padim_256 | large | 49 | 45 | 0.9184 | 0.0816 | 0.9796 | 0 | - |
| 4k_padim_global_256 | tiny | 52 | 40 | 0.7692 | 0.2308 | 0.3846 | 0 | - |
| 4k_padim_global_256 | small | 5 | 5 | 1.0000 | 0.0000 | 0.8000 | 0 | - |
| 4k_padim_global_256 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | tiny | 52 | 36 | 0.6923 | 0.3077 | 0.7692 | 0 | - |
| 4k_padim_global_512 | small | 5 | 5 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | tiny | 52 | 40 | 0.7692 | 0.2308 | 0.7692 | 0 | - |
| 4k_padim_tile512_t256_s256 | small | 5 | 5 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | tiny | 52 | 39 | 0.7500 | 0.2500 | 0.7500 | 0 | - |
| 4k_padim_tile512_t256_s128 | small | 5 | 5 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_patchcore_global_256 | tiny | 52 | 50 | 0.9615 | 0.0385 | 0.0769 | 0 | - |
| 4k_patchcore_global_256 | small | 5 | 5 | 1.0000 | 0.0000 | 0.6000 | 0 | - |
| 4k_patchcore_global_256 | large | 35 | 35 | 1.0000 | 0.0000 | 1.0000 | 0 | - |

## 異常種別別（異常画像のみ）

| run | 異常種別 | n | 検出 | 検出率 | 見逃し率 | 領域ヒット率 | マスク無し | 注記 |
|---|---|---|---|---|---|---|---|---|
| mvtecad_padim_256 | broken_large | 18 | 18 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| mvtecad_padim_256 | broken_small | 19 | 19 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| mvtecad_padim_256 | contamination | 14 | 10 | 0.7143 | 0.2857 | 0.9286 | 0 | - |
| 4k_padim_global_256 | broken_large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_256 | broken_small | 17 | 17 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_256 | contamination | 13 | 13 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_256 | tiny | 54 | 42 | 0.7778 | 0.2222 | 0.3889 | 0 | - |
| 4k_padim_global_512 | broken_large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | broken_small | 17 | 17 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | contamination | 13 | 13 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | tiny | 54 | 38 | 0.7037 | 0.2963 | 0.7778 | 0 | - |
| 4k_padim_tile512_t256_s256 | broken_large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | broken_small | 17 | 17 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | contamination | 13 | 13 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | tiny | 54 | 42 | 0.7778 | 0.2222 | 0.7778 | 0 | - |
| 4k_padim_tile512_t256_s128 | broken_large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | broken_small | 17 | 17 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | contamination | 13 | 13 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | tiny | 54 | 41 | 0.7593 | 0.2407 | 0.7593 | 0 | - |
| 4k_patchcore_global_256 | broken_large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_patchcore_global_256 | broken_small | 17 | 17 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_patchcore_global_256 | contamination | 13 | 13 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_patchcore_global_256 | tiny | 54 | 52 | 0.9630 | 0.0370 | 0.0741 | 0 | - |

## 異常種別×サイズ帯（異常画像のみ）

| run | 異常種別 | サイズ帯 | n | 検出 | 検出率 | 見逃し率 | 領域ヒット率 | マスク無し | 注記 |
|---|---|---|---|---|---|---|---|---|---|
| mvtecad_padim_256 | broken_large | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| mvtecad_padim_256 | broken_large | small | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| mvtecad_padim_256 | broken_large | large | 18 | 18 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| mvtecad_padim_256 | broken_small | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| mvtecad_padim_256 | broken_small | small | 1 | 1 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| mvtecad_padim_256 | broken_small | large | 18 | 18 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| mvtecad_padim_256 | contamination | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| mvtecad_padim_256 | contamination | small | 1 | 1 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| mvtecad_padim_256 | contamination | large | 13 | 9 | 0.6923 | 0.3077 | 0.9231 | 0 | - |
| 4k_padim_global_256 | broken_large | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_256 | broken_large | small | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_256 | broken_large | large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_256 | broken_small | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_256 | broken_small | small | 1 | 1 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_global_256 | broken_small | large | 16 | 16 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_256 | contamination | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_256 | contamination | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_global_256 | contamination | large | 11 | 11 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_256 | tiny | tiny | 52 | 40 | 0.7692 | 0.2308 | 0.3846 | 0 | - |
| 4k_padim_global_256 | tiny | small | 2 | 2 | 1.0000 | 0.0000 | 0.5000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_global_256 | tiny | large | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_512 | broken_large | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_512 | broken_large | small | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_512 | broken_large | large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | broken_small | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_512 | broken_small | small | 1 | 1 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_global_512 | broken_small | large | 16 | 16 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | contamination | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_global_512 | contamination | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_global_512 | contamination | large | 11 | 11 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_global_512 | tiny | tiny | 52 | 36 | 0.6923 | 0.3077 | 0.7692 | 0 | - |
| 4k_padim_global_512 | tiny | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_global_512 | tiny | large | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s256 | broken_large | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s256 | broken_large | small | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s256 | broken_large | large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | broken_small | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s256 | broken_small | small | 1 | 1 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_tile512_t256_s256 | broken_small | large | 16 | 16 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | contamination | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s256 | contamination | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_tile512_t256_s256 | contamination | large | 11 | 11 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s256 | tiny | tiny | 52 | 40 | 0.7692 | 0.2308 | 0.7692 | 0 | - |
| 4k_padim_tile512_t256_s256 | tiny | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_tile512_t256_s256 | tiny | large | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s128 | broken_large | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s128 | broken_large | small | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s128 | broken_large | large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | broken_small | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s128 | broken_small | small | 1 | 1 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_tile512_t256_s128 | broken_small | large | 16 | 16 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | contamination | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_padim_tile512_t256_s128 | contamination | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_tile512_t256_s128 | contamination | large | 11 | 11 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_padim_tile512_t256_s128 | tiny | tiny | 52 | 39 | 0.7500 | 0.2500 | 0.7500 | 0 | - |
| 4k_padim_tile512_t256_s128 | tiny | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_padim_tile512_t256_s128 | tiny | large | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_patchcore_global_256 | broken_large | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_patchcore_global_256 | broken_large | small | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_patchcore_global_256 | broken_large | large | 8 | 8 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_patchcore_global_256 | broken_small | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_patchcore_global_256 | broken_small | small | 1 | 1 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_patchcore_global_256 | broken_small | large | 16 | 16 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_patchcore_global_256 | contamination | tiny | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |
| 4k_patchcore_global_256 | contamination | small | 2 | 2 | 1.0000 | 0.0000 | 1.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_patchcore_global_256 | contamination | large | 11 | 11 | 1.0000 | 0.0000 | 1.0000 | 0 | - |
| 4k_patchcore_global_256 | tiny | tiny | 52 | 50 | 0.9615 | 0.0385 | 0.0769 | 0 | - |
| 4k_patchcore_global_256 | tiny | small | 2 | 2 | 1.0000 | 0.0000 | 0.0000 | 0 | サンプル数が少ないため参考値。 |
| 4k_patchcore_global_256 | tiny | large | 0 | 0 | - | - | - | 0 | n=0 のため率は算出できません。 |

## 推論時間・メモリとのトレードオフ

| run | e2e P50 [ms] | e2e P95 [ms] | train peak RSS [MiB] | overall 検出率 | tiny | small | large |
|---|---|---|---|---|---|---|---|
| mvtecad_padim_256 | 50.4663 | 61.4060 | 1689.9000 | 0.9216 | - | 1.0000 | 0.9184 |
| 4k_padim_global_256 | 498.8833 | 595.3087 | 1691.7300 | 0.8696 | 0.7692 | 1.0000 | 1.0000 |
| 4k_padim_global_512 | 543.9514 | 1178.1499 | 4752.3900 | 0.8261 | 0.6923 | 1.0000 | 1.0000 |
| 4k_padim_tile512_t256_s256 | 669.8523 | 1645.8466 | 4881.9700 | 0.8696 | 0.7692 | 1.0000 | 1.0000 |
| 4k_padim_tile512_t256_s128 | 577.2974 | 635.0381 | 4959.1900 | 0.8587 | 0.7500 | 1.0000 | 1.0000 |
| 4k_patchcore_global_256 | 611.9619 | 1461.3098 | 2579.1900 | 0.9783 | 0.9615 | 1.0000 | 1.0000 |

## 注記

- mvtecad_padim_256: 画像単位の検出は score >= image_threshold で判定しています。 領域単位ヒットは map_max_in_mask >= pixel_threshold で判定しています。 サイズは元解像度マスクの面積率で分類し、anomaly_map 側の解像度には依存しません。 anomaly_map は作業解像度のため、微小欠陥は最近傍リサイズで消失し得ます。 閾値は検証集合から求め、テスト集合では調整していません。
- 4k_padim_global_256: 画像単位の検出は score >= image_threshold で判定しています。 領域単位ヒットは map_max_in_mask >= pixel_threshold で判定しています。 サイズは元解像度マスクの面積率で分類し、anomaly_map 側の解像度には依存しません。 anomaly_map は作業解像度のため、微小欠陥は最近傍リサイズで消失し得ます。 閾値は検証集合から求め、テスト集合では調整していません。
- 4k_padim_global_512: 画像単位の検出は score >= image_threshold で判定しています。 領域単位ヒットは map_max_in_mask >= pixel_threshold で判定しています。 サイズは元解像度マスクの面積率で分類し、anomaly_map 側の解像度には依存しません。 anomaly_map は作業解像度のため、微小欠陥は最近傍リサイズで消失し得ます。 閾値は検証集合から求め、テスト集合では調整していません。
- 4k_padim_tile512_t256_s256: 画像単位の検出は score >= image_threshold で判定しています。 領域単位ヒットは map_max_in_mask >= pixel_threshold で判定しています。 サイズは元解像度マスクの面積率で分類し、anomaly_map 側の解像度には依存しません。 anomaly_map は作業解像度のため、微小欠陥は最近傍リサイズで消失し得ます。 閾値は検証集合から求め、テスト集合では調整していません。
- 4k_padim_tile512_t256_s128: 画像単位の検出は score >= image_threshold で判定しています。 領域単位ヒットは map_max_in_mask >= pixel_threshold で判定しています。 サイズは元解像度マスクの面積率で分類し、anomaly_map 側の解像度には依存しません。 anomaly_map は作業解像度のため、微小欠陥は最近傍リサイズで消失し得ます。 閾値は検証集合から求め、テスト集合では調整していません。
- 4k_patchcore_global_256: 画像単位の検出は score >= image_threshold で判定しています。 領域単位ヒットは map_max_in_mask >= pixel_threshold で判定しています。 サイズは元解像度マスクの面積率で分類し、anomaly_map 側の解像度には依存しません。 anomaly_map は作業解像度のため、微小欠陥は最近傍リサイズで消失し得ます。 閾値は検証集合から求め、テスト集合では調整していません。
- 推論時間・メモリの数値は各 run の report.json に記録された値をそのまま転記している。
- 検出率の差が統計的に有意かどうかは評価していない（未検証）。
