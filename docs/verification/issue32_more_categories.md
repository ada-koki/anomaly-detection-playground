# Issue #32 検証記録: VisA / MVTec AD 2 のカテゴリを予算内で拡張する

## 目的

- VisA と MVTec AD 2 で対象カテゴリを増やし、CPU のみの環境で「1 カテゴリあたり予算」内に
  収まるかを実測で確認する。
- 予算判定とデータセット単位の集計を行う `scripts/summarize_categories.py` の動作を確認する。
- MVTec AD 2 の評価範囲（公開 test split のみ）を report.json の `evaluation_scope` に記録する。

## 環境

- Linux x86_64、8 vCPU、RAM 16 GB、GPU なし、スワップ無し。新規 6 run 開始時の MemAvailable は 7728–9246 MiB（実行ログの記録値）。
- Python 3.13.5、Anomalib 2.7.0、PyTorch 2.15.0+cpu、Lightning 2.6.6、timm 1.0.30（report.json の versions）。
- 計測規約: 共通プロトコル `configs/protocols/visa_v1.json`、
  `--device cpu --cpu-threads 8 --image-size 256`。
- PatchCore は `--patchcore-coreset-mode chunked --patchcore-chunk-images 100`、
  `--max-train-feature-mib 4000`。
- 全 run で、開始前に「同一ホストの別ジョブ」の有無を確認した（全 run で無し）。

## 選定と予算

### カテゴリ選定

- 観測可能な指標（学習画像枚数、抽出後サイズ）で選定した。
- VisA.tar（全 12 カテゴリ共通、1,929,840,640 bytes = 1840.44 MiB、S3 から取得）を一度取得し、
  その中から cashew・chewinggum・pcb1 を抽出し、tar は削除した。
  - cashew: 抽出 177M、train 450 / test good 50 / bad 100
  - chewinggum: 抽出 82M、train 453 / test good 50 / bad 100
  - pcb1: 抽出 286M、train 904 / test good 100 / bad 100
- train 約 450 枚の cashew / chewinggum と、candle（900 枚）と同規模の pcb1 を選んだ。
  candle の chunked100 が学習 1748.553 s で予算内だったため、900 枚級の pcb1 も
  予算内に収まる見込みで選んだ（見込みであり、結果は下記の実測）。
- ビュー作成は `scripts/make_dataset_view.py --dataset visa`（ハードリンク）。

### 予算（暫定）

- `train_stage_seconds ≤ 1800`、`peak_rss_mib ≤ 6000`、`download_mib ≤ 2048`。
- これらは暫定値で、統計的根拠は無い。既存実測（candle PatchCore global の OOM Kill、
  vial PatchCore global の学習 1895.38 s）を基準にした目安である。

## 手順

- 新規 6 run を逐次実行（1 run ずつ）。
  実行順: padim cashew → padim chewinggum → padim pcb1 →
  patchcore(chunked100) cashew → patchcore(chunked100) chewinggum → patchcore(chunked100) pcb1。
- 各 run 後にチェックポイントを削除。
- Issue #31 の干渉判定（暫定閾値 mean 1.5 / p95 3.0）を有効にした。
- 既存 run（candle: Issue #16 PaDiM、Issue #16 PatchCore global の OOM Kill ×3、
  Issue #30 global のメモリガード停止、Issue #30 chunked100。vial: Issue #16 PaDiM、
  Issue #16 PatchCore global、Issue #30 chunked100）も runs.jsonl に含め、同じ予算で判定した。

注意: Issue #16 の candle PaDiM / vial PaDiM / vial PatchCore global の学習時間は、
同一ホストの別ジョブが動いていた状態での値である（vial PatchCore global の
干渉なし再計測は 1646.26 s）。Issue #16 の run は latency_valid フラグ導入前のため
latency_valid は空である。

## 結果

### run 表（benchmarks/issue32/category_runs.csv）

AUROC は engine 値、ピークはプロセス生涯ピーク RSS。
download_mib は VisA 行に 1840.44（共通アーカイブ）、vial 行は未記録で判定対象外（unavailable）。

| condition | status | variant | img AUROC | pix AUROC | train_s | test_s | peak MiB | e2e p95 ms | latency_valid | train/test imgs | within_budget | reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visa_candle_padim | ok | - | 0.878828 | 0.979952 | 128.244 | 32.786 | 3414.98 | 372.9 | - | 900/160 | True | |
| visa_candle_patchcore_global | oom_killed | - | - | - | - | - | - | - | - | - | False | status=oom_killed |
| visa_candle_patchcore_global_retry1 | oom_killed | - | - | - | - | - | - | - | - | - | False | status=oom_killed |
| visa_candle_patchcore_global_mitigation | oom_killed | - | - | - | - | - | - | - | - | - | False | status=oom_killed |
| visa_candle_patchcore_global_guard | memory_guard_stopped | - | - | - | - | - | - | - | - | - | False | status=memory_guard_stopped（特徴量概算 11991.8 MiB > 上限 6000 で学習前に停止） |
| visa_candle_patchcore_chunked100 | ok | chunked_coreset_100img | 0.967187 | 0.991786 | 1748.553 | 129.284 | 2572.51 | 853.6 | - | 900/160 | True | |
| visa_cashew_padim | ok | - | 0.806875 | 0.977509 | 28.822 | 5.632 | 2300.02 | 42.6 | True | 450/120 | True | |
| visa_chewinggum_padim | ok | - | 0.986875 | 0.983823 | 13.965 | 5.413 | 2085.89 | 38.3 | True | 453/120 | True | |
| visa_pcb1_padim | ok | - | 0.786875 | 0.986378 | 27.334 | 7.389 | 3451.46 | 51.8 | True | 904/160 | True | |
| visa_cashew_patchcore_chunked100 | ok | chunked_coreset_100img | 0.97625 | 0.99112 | 788.868 | 66.354 | 2331.62 | 297.7 | True | 450/120 | True | |
| visa_chewinggum_patchcore_chunked100 | ok | chunked_coreset_100img | 0.99 | 0.989822 | 763.815 | 44.797 | 2335.78 | 1247.9 | True | 453/120 | True | |
| visa_pcb1_patchcore_chunked100 | ok | chunked_coreset_100img | 0.917969 | 0.996223 | 1512.531 | 125.756 | 2617.83 | 1715.0 | True | 904/160 | True | |
| ad2_vial_padim | ok | - | 0.631378 | 0.846444 | 32.979 | 8.796 | 2008.66 | 64.1 | - | 291/112 | True | |
| ad2_vial_patchcore_global | ok | - | 0.857993 | 0.94292 | 1895.38 | 35.475 | 4224.95 | 583.7 | - | 291/112 | False | train_stage_seconds=1895.38 > 1800 |
| ad2_vial_patchcore_chunked100 | ok | chunked_coreset_100img | 0.863946 | 0.943753 | 586.405 | 34.365 | 2241.39 | 1222.8 | - | 291/112 | True | |

test 枚数は val 分割後の test ロール数。

### データセット要約表（benchmarks/issue32/dataset_summary.csv）

| dataset | model | variant | n_runs | n_ok | n_failed | n_within | categories | img AUROC mean/std/min/max | pix AUROC mean/std/min/max | train 合計/最大 s | peak 最大 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visa | padim | - | 4 | 4 | 0 | 4 | candle,cashew,chewinggum,pcb1 | 0.864863/0.090419/0.786875/0.986875 | 0.981915/0.003951/0.977509/0.986378 | 198.365/128.244 | 3451.46 |
| visa | patchcore | (global) | 4 | 0 | 4 | 0 | - | - | - | - | - |
| visa | patchcore | chunked_coreset_100img | 4 | 4 | 0 | 4 | candle,cashew,chewinggum,pcb1 | 0.962852/0.031357/0.917969/0.99 | 0.992238/0.002779/0.989822/0.996223 | 4813.767/1748.553 | 2617.83 |
| mvtec_ad2 | padim | - | 1 | 1 | 0 | 1 | vial | 0.631378 | 0.846444 | 32.979 | 2008.66 |
| mvtec_ad2 | patchcore | (global) | 1 | 1 | 0 | 0 | vial | 0.857993 | 0.94292 | 1895.38 | 4224.95 |
| mvtec_ad2 | patchcore | chunked_coreset_100img | 1 | 1 | 0 | 1 | vial | 0.863946 | 0.943753 | 586.405 | 2241.39 |

std は標本標準偏差（n=4）。カテゴリ数 4 / 1 のためばらつきの推定としては小標本。
n=1 の行は std を出さない。

### レイテンシと干渉（新規 6 run, benchmarks/issue32/issue32_runs_latency.csv）

| condition | model p50/p95 ms | e2e p50/p95 ms | other_busy_cores mean/p95 | raw img/pix AUROC |
| --- | --- | --- | --- | --- |
| cashew_padim | 23.9/29.0 | 34.3/42.6 | 0.142/0.279 | 0.806875/0.977574 |
| chewinggum_padim | 23.1/28.9 | 31.0/38.3 | 0.203/0.378 | 0.986875/0.983688 |
| pcb1_padim | 29.4/39.2 | 39.8/51.8 | 0.144/0.376 | 0.787031/0.986207 |
| cashew_patchcore_chunked100 | 241.7/281.9 | 255.5/297.7 | 0.142/0.318 | 0.97625/0.991039 |
| chewinggum_patchcore_chunked100 | 236.9/1159.7 | 250.4/1247.9 | 0.584/2.223 | 0.99/0.989677 |
| pcb1_patchcore_chunked100 | 634.2/1593.7 | 649.0/1715.0 | 0.878/1.534 | 0.917969/0.996097 |

全 run latency_valid=True（試行 1 回）。ただし chewinggum / pcb1 PatchCore は other_busy_cores p95 が
それぞれ 2.223 / 1.534 と他の run（0.28–0.38）より高く、e2e p95 レイテンシが p50 の約 5.0 倍 / 2.6 倍（1247.9/250.4 ms、1715.0/649.0 ms）。
暫定閾値内だが干渉の影響を否定できない（未検証）。
PatchCore の memory bank 行数: cashew 46080、chewinggum 46387、pcb1 92569、candle 92160（chunk_count 5/5/10/9）。

## 合計所要時間

- 新規 6 run の壁時計: 2026-11-01 04:47:01–05:51:17 JST（ログ時刻、64 分 16 秒 = 3856 s）。
- 新規 6 run の train_stage 合計: 3135.335 s。

## 考察

- VisA 3 カテゴリ×2 モデル = 6 run はすべて ok で dataset_summary.csv を生成できた。
- PaDiM は 4 カテゴリすべて予算内。ピーク RSS は 2085–3451 MiB、学習は 14–128 s。
- PatchCore chunked100 は 4 カテゴリすべて予算内だが学習が重い（764–1749 s）。
  candle（900 枚）は学習 1748.553 s で、暫定上限 1800 s に近い。
- PatchCore global（VisA）は candle で 3 回 OOM Kill、メモリガード停止（特徴量概算 11991.8 MiB）で
  完走せず、予算外判定となった。900 枚級では global 構成は現実的でない。
- MVTec AD 2 は vial のみ評価。global は学習 1895.38 s で予算超過、chunked100 は 586.405 s で予算内。
- 精度面では PatchCore chunked100 は PaDiM より img AUROC が高い傾向
  （VisA mean 0.962852 vs 0.864863）。ただし chunked は近似であり global と同一条件ではない。

## 受け入れ条件の判定

| 条件 | 結果 |
| --- | --- |
| 追加カテゴリの report.json が揃い summary 生成（VisA 3×2=6 run 全て ok、dataset_summary.csv 生成） | 達成 |
| 各 run のピークメモリ・所要時間の実測と予算内外判別（within_budget 列） | 達成 |
| 全 run の合計所要時間の記録 | 達成（壁時計 3856 s、train_stage 合計 3135.335 s） |
| MVTec AD 2 の評価範囲の明記（docs と evaluation_scope） | 達成 |
| 失敗 run を記録から消さない（status 付きで残す） | 達成（candle global の OOM Kill 3 件・ガード停止 1 件を残置） |
| check_repo_safety の通過 | 達成（`python scripts/check_repo_safety.py`、違反なし） |

## 未検証 / 未実施

- MVTec AD 2 の追加カテゴリ: 未実施。配布元からのダウンロードが約 30 KB/s（#16 時の観測）で、
  1 カテゴリ数百 MB 規模でも数時間かかり、今回の時間予算外と判断した。vial（#16/#30 の既存 run）のみ。
- VisA 残り 8 カテゴリ: 未実施。
- chunked と global の精度差: 同カテゴリで global が完走しないため未検証。
- 複数 seed・繰り返し: 未実施。
- 予算値の他ホストでの妥当性: 未検証。
- chewinggum / pcb1 PatchCore のレイテンシ裾: 未検証。
- ダウンロード時間: 今回の VisA.tar 取得時間は記録していない。
