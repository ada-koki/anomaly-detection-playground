# Issue #31 検証記録: レイテンシ計測の干渉検出（latency_valid フラグ）

## 目的

レイテンシ計測中の他プロセス干渉を検出し、`latency_valid` フラグとして記録できることを確認する。意図的に CPU 負荷を
かけた条件では `latency_valid=false`、負荷なし条件では `true` になり、再計測によって負荷終了後の試行が採用されて
`true` になることを確認する。

## 環境

- Linux x86_64 / 8 vCPU / GPU なし / スワップなし。
- 同一ホストの別ジョブは無し（検証期間中）。
- データ: MVTec AD bottle。モデル: PaDiM。入力 256px。
- 実行: `--device cpu --cpu-threads 8 --skip-raw-eval`、プロトコル `mvtec_ad_v1`。
- 実施時刻: 2026-11-01 04:15–04:45 JST（run_id のタイムスタンプは UTC 表記）。
- 干渉再現用に `scripts/cpu_load.py`（指定プロセス数でビジーループ）を使用。各 run 開始 2 秒前に起動し `--seconds 600`、run 終了時に停止。

## 手順

次の 5 条件を順に実行した。

1. `clean_a`: 負荷なし。
2. `load4`: `cpu_load.py --procs 4`、`--latency-max-retries 0`。
3. `load8`: `--procs 8`、`--latency-max-retries 0`。
4. `clean_b`: 負荷なし。
5. `load8_retry2`: `--procs 8`、`--latency-max-retries 2`。

## 結果

`benchmarks/issue31/latency_interference_summary.csv`（試行ごと 1 行。レイテンシ列は採用試行の値）。干渉指標は 3 桁、
レイテンシは 1 桁に丸めて表示している。

| condition | procs | retries | attempt | adopted | attempt_valid | run_valid | other_busy_mean | other_busy_p95 | self_busy_mean | psi_mean | n_windows | model_p50_ms | model_p95_ms | e2e_p50_ms | e2e_p95_ms | image_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_a | 0 | 1 | 0 | True | True | True | 0.192 | 0.516 | 4.287 | 0.066 | 6 | 20.4 | 29.3 | 39.0 | 52.5 | 0.996324 |
| clean_b | 0 | 1 | 0 | True | True | True | 0.103 | 0.296 | 4.227 | 0.063 | 6 | 25.1 | 33.2 | 44.6 | 52.5 | 0.996324 |
| load4 | 4 | 0 | 0 | True | False | False | 4.032 | 4.295 | 3.852 | 0.569 | 146 | 927.5 | 1082.9 | 1077.6 | 1234.2 | 0.996324 |
| load8 | 8 | 0 | 0 | True | False | False | 4.733 | 5.403 | 3.102 | 0.766 | 88 | 31.5 | 1651.1 | 51.7 | 1930.5 | 0.996324 |
| load8_retry2 | 8 | 2 | 0 | False | False | True | 4.646 | 5.139 | 3.278 | 0.736 | 116 | （非採用・非保存） | （非採用・非保存） | （非採用・非保存） | （非採用・非保存） | 0.996324 |
| load8_retry2 | 8 | 2 | 1 | True | True | True | 0.138 | 0.277 | 4.126 | 0.057 | 6 | 25.4 | 32.6 | 43.6 | 53.1 | 0.996324 |

補足:

- **load8**: 試行中に `cpu_load.py` の 600 秒が切れたため、p50 は低く p95 だけが大きい（画像ごとの時系列は確認していない）。8 プロセスの負荷に対し other_busy_cores 平均が 4.733 に留まるのは、8 vCPU を計測プロセスと負荷が取り合ったためと考えられる（未検証）。
- load4 / load8 では学習時間（`train_stage_s` 311.562 / 405.756）も大きく伸びたが、**学習時間は干渉判定の対象外**。
- load8_retry2 は 1 つの run に 2 試行がある。attempt 0 は負荷中で false、attempt 1 で true を採用（`adopted_attempt=1`）。
  **非採用試行（attempt 0）のレイテンシ値は report に保存していない**（保存しているのは干渉指標のみ）。表で attempt 0 の
  レイテンシ列を「非採用・非保存」としているのはこのため。attempt 1 の値が採用値。
- `image_auroc` は全条件で 0.996324（データ・モデル・入力が同一のため。干渉はレイテンシに現れるが AUROC には影響していない）。

### 集計結果

`scripts/aggregate_results.py` で 5 run を集計した結果:

- `n_latency_invalid=2`（load4, load8）、`n_latency_unjudged=0`。
- レイテンシ要約は valid の 3 run（clean_a, clean_b, load8_retry2）のみで計算。
- `e2e_p95_ms_mean = 52.70`、`model_p95_ms_mean = 31.69`。

（要約値は丸め前の per-run 値から計算される。上の表は 1 桁丸めのため、表の値から直接平均を再現すると端数が一致しない場合がある。）

## 考察

- 負荷なし条件（clean_a / clean_b）はいずれも `other_busy_cores` 平均 0.10–0.19 / p95 0.30–0.52 で、閾値
  （平均 1.5 / p95 3.0）を大きく下回り `latency_valid=true` となった。
- 意図的負荷条件（load4 / load8）は `other_busy_cores` 平均 4.0–4.7 / p95 4.3–5.4 で閾値を超え、`latency_valid=false`。
  実際のレイテンシも clean 条件より大きく悪化しており、フラグが干渉を正しく反映している。
- `load8_retry2` は attempt 0 が false、attempt 1 が true となり、`measure_with_interference_check` が
  **最初の false でない試行（attempt 1）を採用** した。再計測の仕組みが機能している。
  - ただし attempt 1 が clean になったのは、`cpu_load.py` の **負荷時間 600 秒が attempt 0 の終了直後に切れた** ためである。
    再計測の時刻を制御した実験ではなく、「負荷が終わったタイミングで再計測したら clean だった」という結果である。
- PSI（`psi_cpu_some_ratio_mean`）は干渉なしでも 0.057–0.066 と非ゼロであり、負荷時は 0.569–0.766 まで上がった。
  ただし PSI は自プロセス負荷と他プロセス負荷を区別できないため、既定（`max_psi_cpu_some_ratio_mean=None`）では判定に
  使わず記録のみとした。
- 集計は `latency_valid=false` の 2 run を除外し、有効な 3 run で要約した。これにより干渉で膨らんだ値を要約から外せている。

## 受け入れ条件の判定

Issue #31 の受け入れ条件「レイテンシ計測時の他プロセス干渉を検出し、`latency_valid` フラグを記録する」に対して:

| 確認項目 | 結果 | 判定 |
| --- | --- | --- |
| 意図的負荷 2 条件（load4, load8）で `latency_valid=false` | 両条件とも false | 満たす |
| 負荷なし 2 条件（clean_a, clean_b）で `latency_valid=true` | 両条件とも true | 満たす |
| 再計測で負荷終了後の試行を採用して `true`（load8_retry2） | attempt 1 を採用し true | 満たす |

3 項目すべてを満たす。レイテンシ要約は `latency_valid=false` の run を除外して計算できている。

## 未検証事項

- VM の **ホスト側（ハイパーバイザ）の steal** が観測指標にどう現れるかは未検証。
- **GPU 実行時** の干渉判定。未検証。
- **メモリ帯域・I/O 干渉** のように CPU 使用率に現れない干渉。未検証。
- 閾値（平均 1.5 / p95 3.0）の **他ホストでの妥当性**。単一ホストでの評価のみ。未検証。
- `sample_interval_s=0.5` 秒より **短い瞬間的な干渉** の取りこぼし。未検証。
- `psutil` の子プロセス CPU 集計が **短命プロセスを取りこぼす** 可能性。未検証。
- 閾値は暫定値（`is_provisional=true`）であり、統計的根拠に基づく決定ではない。
