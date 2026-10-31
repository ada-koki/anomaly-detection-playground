# カテゴリ予算レポート

> **注意**: 本レポートの予算は暫定値です（`is_provisional=true`）。

予算名: `category_budget_v1`

## 予算の定義

| name | max_train_stage_seconds | max_peak_rss_mib | max_download_mib | is_provisional | notes |
| --- | --- | --- | --- | --- | --- |
| category_budget_v1 | 1800 | 6000 | 2048 | True | 8 vCPU / 16GB RAM（スワップなし）の計測ホスト向けの暫定予算（Issue #32）。基準: VisA candle PatchCore global は OOM Kill、MVTec AD 2 vial PatchCore global の学習段階 1895.38 s（干渉なし再計測 1646.26 s）、ピーク 4194.06 MiB。download はデータセット取得に必要なアーカイブ量（VisA.tar は全カテゴリ共通で 1840.44 MiB）。統計的に決めた値ではない。 |

## run 一覧

| condition | dataset | category | model | variant | status | image_auroc | pixel_auroc | train_stage_seconds | test_stage_seconds | peak_rss_mib | e2e_p95_ms | latency_valid | train_images | test_images | run_id | within_budget | reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visa_candle_padim | visa | candle | padim |  | ok | 0.878828 | 0.979952 | 128.24 | 32.79 | 3414.98 | 372.95 | — | 900 | 160 | 20261031T132811Z-01d9c417 | True |  |
| visa_candle_patchcore_global | visa | candle | patchcore |  | oom_killed | — | — | — | — | — | — | — | — | — | — | False | status=oom_killed |
| visa_candle_patchcore_global_retry1 | visa | candle | patchcore |  | oom_killed | — | — | — | — | — | — | — | — | — | — | False | status=oom_killed |
| visa_candle_patchcore_global_mitigation | visa | candle | patchcore |  | oom_killed | — | — | — | — | — | — | — | — | — | — | False | status=oom_killed |
| visa_candle_patchcore_global_guard | visa | candle | patchcore |  | memory_guard_stopped | — | — | — | — | — | — | — | — | — | — | False | status=memory_guard_stopped |
| visa_candle_patchcore_chunked100 | visa | candle | patchcore | chunked_coreset_100img | ok | 0.967187 | 0.991786 | 1748.55 | 129.28 | 2572.51 | 853.56 | — | 900 | 160 | 20261031T182701Z-24d16eda | True |  |
| visa_cashew_padim | visa | cashew | padim |  | ok | 0.806875 | 0.977509 | 28.82 | 5.63 | 2300.02 | 42.61 | True | 450 | 120 | 20261031T194701Z-df7a0728 | True |  |
| visa_chewinggum_padim | visa | chewinggum | padim |  | ok | 0.986875 | 0.983823 | 13.96 | 5.41 | 2085.89 | 38.26 | True | 453 | 120 | 20261031T194752Z-227b7ca8 | True |  |
| visa_pcb1_padim | visa | pcb1 | padim |  | ok | 0.786875 | 0.986378 | 27.33 | 7.39 | 3451.46 | 51.81 | True | 904 | 160 | 20261031T194827Z-44171ee7 | True |  |
| visa_cashew_patchcore_chunked100 | visa | cashew | patchcore | chunked_coreset_100img | ok | 0.976250 | 0.991120 | 788.87 | 66.35 | 2331.62 | 297.69 | True | 450 | 120 | 20261031T194924Z-35a826f1 | True |  |
| visa_chewinggum_patchcore_chunked100 | visa | chewinggum | patchcore | chunked_coreset_100img | ok | 0.990000 | 0.989822 | 763.82 | 44.80 | 2335.78 | 1247.89 | True | 453 | 120 | 20261031T200500Z-1b741e98 | True |  |
| visa_pcb1_patchcore_chunked100 | visa | pcb1 | patchcore | chunked_coreset_100img | ok | 0.917969 | 0.996223 | 1512.53 | 125.76 | 2617.83 | 1715.01 | True | 904 | 160 | 20261031T202014Z-e0208b2f | True |  |
| ad2_vial_padim | mvtec_ad2 | vial | padim |  | ok | 0.631378 | 0.846444 | 32.98 | 8.80 | 2008.66 | 64.05 | — | 291 | 112 | 20261031T134248Z-27de2a6f | True |  |
| ad2_vial_patchcore_global | mvtec_ad2 | vial | patchcore |  | ok | 0.857993 | 0.942920 | 1895.38 | 35.48 | 4224.95 | 583.71 | — | 291 | 112 | 20261031T134352Z-de865b08 | False | train_stage_seconds=1895.38 > 1800 |
| ad2_vial_patchcore_chunked100 | mvtec_ad2 | vial | patchcore | chunked_coreset_100img | ok | 0.863946 | 0.943753 | 586.40 | 34.37 | 2241.39 | 1222.84 | — | 291 | 112 | 20261031T190316Z-1814b883 | True |  |

## データセット要約

| dataset | model | variant | n_runs | n_ok | n_failed | n_within_budget | categories | image_auroc_mean | image_auroc_std | image_auroc_min | image_auroc_max | pixel_auroc_mean | pixel_auroc_std | pixel_auroc_min | pixel_auroc_max | train_stage_seconds_total | train_stage_seconds_max | peak_rss_mib_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visa | padim |  | 4 | 4 | 0 | 4 | candle,cashew,chewinggum,pcb1 | 0.864863 | 0.090419 | 0.786875 | 0.986875 | 0.981915 | 0.003951 | 0.977509 | 0.986378 | 198.37 | 128.24 | 3451.46 |
| visa | patchcore |  | 4 | 0 | 4 | 0 |  | — | — | — | — | — | — | — | — | — | — | — |
| visa | patchcore | chunked_coreset_100img | 4 | 4 | 0 | 4 | candle,cashew,chewinggum,pcb1 | 0.962852 | 0.031357 | 0.917969 | 0.990000 | 0.992238 | 0.002779 | 0.989822 | 0.996223 | 4813.77 | 1748.55 | 2617.83 |
| mvtec_ad2 | padim |  | 1 | 1 | 0 | 1 | vial | 0.631378 | — | 0.631378 | 0.631378 | 0.846444 | — | 0.846444 | 0.846444 | 32.98 | 32.98 | 2008.66 |
| mvtec_ad2 | patchcore |  | 1 | 1 | 0 | 0 | vial | 0.857993 | — | 0.857993 | 0.857993 | 0.942920 | — | 0.942920 | 0.942920 | 1895.38 | 1895.38 | 4224.95 |
| mvtec_ad2 | patchcore | chunked_coreset_100img | 1 | 1 | 0 | 1 | vial | 0.863946 | — | 0.863946 | 0.863946 | 0.943753 | — | 0.943753 | 0.943753 | 586.40 | 586.40 | 2241.39 |
