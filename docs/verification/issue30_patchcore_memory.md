# Issue #30 検証記録: PatchCore 学習メモリ対策

## 目的

PatchCore の学習時ピークメモリを抑える手段（`--patchcore-coreset-mode {global,chunked}` / `--patchcore-chunk-images N` / `--max-train-feature-mib M`）と、学習前メモリ見積り・上限ガードの実機動作を確認する。あわせて、16GB・スワップなしのホストで VisA candle の PatchCore が完走可能かを評価する。

## 環境

- CPU 8 vCPU、RAM 16GB、スワップなし、GPU なし。
- `--device cpu --cpu-threads 8`、Anomalib 2.7。
- backbone: wide_resnet50_2（layer2 + layer3、256px）、`coreset_sampling_ratio` 0.1。
- プロトコル: `mvtec_ad_v1` / `visa_v1` / `mvtec_ad2_v1`（val_split from_test 0.2、seed 42、batch 1、num_workers 0）。
- 再現: 作業用スクリプト `tools/i30_run.sh` 相当（リポジトリ外）で、`scripts/run_anomalib.py` に上記オプションを付与して実行。結果はリポジトリ外の出力ディレクトリに保存し、集計 CSV のみ `benchmarks/issue30/patchcore_memory_summary.csv` としてコミットする。

## 条件

- `global`（既定）と `chunked`（チャンク枚数 20 / 100）を、MVTec AD（bottle）、VisA（candle）、MVTec AD 2（vial）で実行。
- 一部は `--max-train-feature-mib` を指定し、上限ガードの動作を確認。

## 結果

出典: `benchmarks/issue30/patchcore_memory_summary.csv`。

### 精度・shape

| condition | status | dataset | coreset_mode | chunk_images | max_train_feature_mib | train_images | memory_bank_rows | image_auroc_engine | pixel_auroc_engine | image_auroc_raw | pixel_auroc_raw | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ad2_vial_patchcore_chunked100 | ok | mvtec_ad2 | chunked | 100 | — | 291 | 29798 | 0.863946 | 0.943753 | 0.863946 | 0.943404 | 20261031T190316Z-1814b883 |
| mvtecad_bottle_patchcore_chunked20 | ok | mvtec_ad | chunked | 20 | — | 209 | 21401 | 1.0 | 0.985721 | 1.0 | 0.985538 | 20261031T182505Z-72e46020 |
| mvtecad_bottle_patchcore_global | ok | mvtec_ad | global | — | — | 209 | 21401 | 1.0 | 0.985694 | 1.0 | 0.985522 | 20261031T181935Z-19645561 |
| visa_candle_patchcore_chunked100 | ok | visa | chunked | 100 | 4000.0 | 900 | 92160 | 0.967187 | 0.991786 | 0.967187 | 0.991567 | 20261031T182701Z-24d16eda |
| visa_candle_patchcore_global_guard | failed | visa | global | — | 6000.0 | 900 | — | — | — | — | — | 20261031T181858Z-a773908b |

### メモリ・レイテンシ

| condition | status | train_stage_s | train_peak_rss_mib | process_peak_rss_mib | mem_available_mib_at_start | estimate_peak_mib | estimate_within_budget | model_p50_ms | e2e_p50_ms | e2e_p95_ms | threads | other_process_interference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ad2_vial_patchcore_chunked100 | ok | 586.405 | 2241.39 | 2241.39 | 8346.91 | 1485.54 | True | 258.1 | 282.1 | 1222.8 | 8 | no |
| mvtecad_bottle_patchcore_chunked20 | ok | 39.22 | 1135.98 | 1212.32 | 8579.63 | 384.54 | True | 194.3 | 215.8 | 1231.1 | 8 | no |
| mvtecad_bottle_patchcore_global | ok | 276.773 | 3209.56 | 3209.66 | 8396.55 | 2755.37 | True | 158.1 | 183.0 | 250.1 | 8 | no |
| visa_candle_patchcore_chunked100 | ok | 1748.553 | 2572.51 | 2572.51 | 6803.9 | 1850.94 | True | 649.4 | 665.0 | 853.6 | 8 | no |
| visa_candle_patchcore_global_guard | failed | — | — | 1289.26 | 7998.65 | 11991.8 | False | — | — | — | 8 | no |

> `other_process_interference` は「同一ホスト上の別ジョブ（別プロジェクトのプロセス）の有無」のみを記録したものである。レイテンシは本 Issue の対象外。

### 参考: Issue #16 の既存実測（`benchmarks/issue16/datasets_summary.csv` 由来）

- VisA candle PatchCore `global`: 2026-10-31 22:40:41 JST に OOM Kill（anon-rss 9.06 GiB、学習中）。
- MVTec AD 2 vial PatchCore `global`: accuracy run（別ジョブ干渉あり）で img AUROC 0.857993 / pixel AUROC 0.94292、train 1895.38 s、train peak RSS 4194.06 MiB。干渉なし再計測で train 1646.26 s、peak RSS 4185.46 MiB。

## 考察

- **見積りは概算であり、実測ピークを下回る傾向**。完走した全 run で `estimate_peak_mib` < 実測 `train_peak_rss_mib`（ad2 vial: 1485.54 → 2241.39 / bottle chunked20: 384.54 → 1135.98 / bottle global: 2755.37 → 3209.56 / candle chunked100: 1850.94 → 2572.51）。見積りは特徴量テンソルのみを対象とし、アロケータ断片化・DataLoader・距離配列・sklearn 作業領域を含まないため。上限を設定する際は物理メモリに対し余裕を持たせるべき。
- **上限ガードの動作**: candle `global` を `--max-train-feature-mib 6000` で実行したところ、見積り概算 11991.8 MiB が上限 6000 MiB を超えたため学習を開始せずに停止（`status=failed`、プロセスピーク 1289.26 MiB、`estimate_within_budget=False`）。開始時空きメモリ 7998.65 MiB に対しても見積りは超過しており、学習に進めば OOM リスクが高かったと考えられる。
- **chunked は近似**: `chunked` はチャンクごとに coreset を選ぶため、選択点集合が `global` と異なる。したがって `chunked` のbottle・candle・vial の精度は `global` と同一条件の比較ではない。bottle では `global`（img 1.0 / pix 0.985694）と `chunked20`（img 1.0 / pix 0.985721）が近い値だが、これは選択点集合が近かったことを示すだけで、一般に一致を保証するものではない。
- **vial（参考比較）**: 本 Issue の vial 実行は `chunked100` のみ（img 0.863946 / pix 0.943753、学習ピーク RSS 2241.39 MiB）。Issue #16 の vial `global`（同一プロトコル・別日の別 run）は img 0.857993 / pix 0.94292、学習ピーク RSS 4194.06 MiB（干渉なし再計測 4185.46 MiB）。chunked は近似のため精度は同一条件の比較ではなく、1 run ずつの値の差を優劣とは解釈しない。
- **学習時間（参考）**: bottle は global 276.773 s に対し chunked20 39.22 s。vial は chunked100 586.405 s に対し Issue #16 の global 干渉なし再計測 1646.26 s。KCenterGreedy の計算量が対象行数に依存するためと考えられるが、要因の切り分けは未検証。
- **candle の既定条件は未完走**: 既定 `global` は Issue #16 の OOM Kill、本 Issue の guard 実行は学習開始前に停止しており、既定条件での完走結果は得られていない。
- **レイテンシは対象外**: `chunked` run は e2e p95 が大きい（ad2 vial 1222.8、bottle chunked20 1231.1、candle chunked100 853.6 ms に対し、bottle global は 250.1 ms）。この差の理由は**未検証**であり、本 Issue の対象外とする。Issue #31 で干渉検出を導入予定。

## 受け入れ条件の判定

受け入れ条件: 「16GB スワップなしのホストで VisA candle の PatchCore（wide_resnet50_2, layer2+3, 256px, coreset 0.1）が完走する」。

- `chunked100`（`--max-train-feature-mib 4000`）では**完走**（`status=ok`、学習ピーク RSS 2572.51 MiB）。
- 既定条件（`global`）では完走**未達**。guard 実行は見積り概算 11991.8 MiB が上限 6000 MiB を超えて学習開始前に停止（`status=failed`、プロセスピーク 1289.26 MiB）。

**判定**: 受け入れ条件は `chunked`（プロトコル逸脱の近似）では満たすが、既定条件（`global`）では満たさない。既定条件での完走は未達であり、`chunked` は近似として variant 列 `chunked_coreset_100img` で既定条件と分けて記録する。

## 未実施・未検証事項

未実施:

- `float16` メモリバンク。
- 学習画像の間引き。
- チャンク枚数のスイープ（本記録は 20 / 100 の 2 点のみ）。
- `global` と `chunked` の選択点集合の差の定量化。
- 複数 seed での再現性確認。

未検証:

- `chunked` の精度差が一般にどの程度か（本記録は 3 カテゴリ各 1 run のみ）。
- 見積り式の誤差の一般性（本記録では実測 > 見積りの傾向のみ確認）。
- `chunked` run で e2e p95 が大きい理由（本 Issue の対象外、Issue #31 で干渉検出を導入予定）。
