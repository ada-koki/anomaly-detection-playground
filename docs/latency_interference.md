# レイテンシ計測の干渉検出（latency_valid フラグ）

本ドキュメントは Issue #31 で追加した「レイテンシ計測中の他プロセス干渉の検出」機能を利用者向けに説明する。実測結果と受け入れ条件の判定は
[`docs/verification/issue31_latency_interference.md`](verification/issue31_latency_interference.md) を参照。

## 背景と目的

CPU 優先ベンチマークでは、レイテンシ計測中に同一ホストの別プロセスが CPU を使うと、計測値が大きくぶれる。従来は
この干渉を検出する手段がなく、干渉で膨らんだレイテンシ値をそのまま集計に混ぜてしまっていた。本機能はレイテンシ
計測区間中の「自分以外の CPU 使用量」を測定し、閾値を超えた場合はその run のレイテンシ計測を **無効（latency_valid=false）**
として記録する。集計では無効 run をレイテンシ要約から除外できる。

## 判定方法

`adbench/interference.py` が次を行う。

- **サンプリング**: `/proc/stat`（ホスト全体の busy jiffies）と `psutil`（自プロセス＋子孫プロセスの CPU 秒）を、
  既定 `sample_interval_s=0.5` 秒間隔で **別スレッド** からサンプリングする。
- **窓ごとの指標**: 各サンプリング窓について
  `other_busy_cores = host_busy_cores − self_busy_cores` を計算する。
  `host_busy_cores` は窓内の busy jiffies 増分から、`self_busy_cores` は窓内の自プロセス CPU 秒増分から算出する。
- **summary**: 全区間の窓について `other_busy_cores` と `self_busy_cores` の mean / p95（nearest rank 法）/ max、
  全区間の生値、窓数 `n_windows` を記録する。
- **補助情報**: `/proc/pressure/cpu` の `some total` 比（`psi_cpu_some_ratio_mean`）と loadavg も記録する。

判定は `judge_latency_validity` が行う。指標が取得できない場合（`/proc` や psutil が使えない等）は **None（未判定）** を返し、
「有効」とは区別する。

## 閾値と protocol JSON の任意キー `latency_interference`

判定に使う閾値は `InterferenceThresholds` で定義する。既定値は次のとおり。

| キー | 既定値 | 意味 |
| --- | --- | --- |
| `max_other_busy_cores_mean` | 1.5 | `other_busy_cores` の平均の上限。超過で無効。 |
| `max_other_busy_cores_p95` | 3.0 | `other_busy_cores` の p95 の上限。超過で無効。 |
| `max_psi_cpu_some_ratio_mean` | `None` | PSI を使う場合の上限。既定は **判定に使わない**。 |
| `sample_interval_s` | 0.5 | サンプリング間隔（秒）。 |
| `is_provisional` | true | 暫定値であることのフラグ。 |

### 閾値は暫定値である

上記の閾値は **統計的に導出したものではない（暫定値、`is_provisional=true`）**。根拠として観測できているのは次の範囲にとどまる。

- 干渉なし時の実測 `other_busy_cores` は平均 0.10–0.19 / p95 0.30–0.52（本 Issue の clean 条件、および再計測で採用された試行）。
- Issue #30 の期間中に別 run で観測された `other_busy_cores` は平均 0.66–0.89。

これらは「干渉なしは概ね 1 コア未満、意図的負荷時は 4 コア前後」という粗い分離を示すだけであり、閾値 1.5（平均）/ 3.0（p95）は
この分離の間に置いた運用上の値である。他ホストでの妥当性は未検証（後述「限界」）。

### PSI を既定で判定に使わない理由

`/proc/pressure/cpu` の `some` は **自プロセスの負荷と他プロセスの負荷を区別できない**。そのため PSI は記録のみ行い、
既定（`max_psi_cpu_some_ratio_mean=None`）では判定に使わない。参考として、干渉なし条件でも `psi_cpu_some_ratio_mean` は
0.057–0.066 と記録された（ゼロではない）。他プロセス由来の干渉だけを見たい本機能の目的には PSI 単独では不十分である。

### protocol JSON での指定

`latency_interference` は **任意のトップレベルキー** で、未指定時は上記の暫定既定値を使う。既存のプロトコル JSON は変更していない。

```json
{
  "latency_interference": {
    "max_other_busy_cores_mean": 1.5,
    "max_other_busy_cores_p95": 3.0,
    "max_psi_cpu_some_ratio_mean": null,
    "sample_interval_s": 0.5,
    "is_provisional": true
  }
}
```

## 再計測（`--latency-max-retries`）

`scripts/run_anomalib.py` の `--latency-max-retries`（既定 1、0 以上）で、`latency_valid=false` だった場合の再計測回数を指定する。

- `measure_with_interference_check` が `latency_valid=false` のとき最大 `max_retries` 回まで再計測する。
- **最初の false でない試行** を採用する。全試行が false の場合は **最後の試行** を採用する。
- `--latency-max-retries 0` にすると再計測しない（1 回だけ計測）。

## report.json のフィールド

`<output-dir>/<run_id>/report.json` に次を追加する。

- `latency_interference`:
  - `thresholds`: 使用した `InterferenceThresholds`。
  - `attempts`: 各試行の配列。各要素は `latency_valid`（true/false/null）、`reasons`（判定理由）、`summary`（干渉指標）を持つ。
  - `adopted_attempt`: 採用した試行番号。
  - `latency_valid`: 採用試行の判定結果。
  - `reasons`: 採用試行の判定理由。
- `latency_valid`: 採用試行の判定結果（利便性のためのトップレベル複製）。

`--skip-latency` 指定時は `latency_interference` と `latency_valid` はともに `null`。採用試行が `latency_valid=false` の場合は
標準エラーに警告を出す。

**注意**: 非採用試行の **レイテンシ値は report に保存しない**（干渉指標のみ保存する）。レイテンシ値は採用試行のものだけを保持する。

## 集計での扱い

`adbench/aggregate.py` / `scripts/aggregate_results.py` は次を行う。

- `runs.csv` に `latency_valid` 列を追加。
- `summary.csv` のレイテンシ要約は **`latency_valid=false` の run を除外** して計算する。
- `n_latency_invalid`（無効 run 数）と `n_latency_unjudged`（未判定 run 数）列を追加。
- `report.md` に「無効 run を除外した」旨の注記を出力。
- `latency_valid` キーを持たない **旧 report は「未判定」として扱う**（除外はしない。集計対象に残る）。

## 限界（未検証事項を含む）

- **VM のホスト側（ハイパーバイザ）の steal** が本機能の指標（`/proc/stat` と psutil）にどう現れるかは未検証。
- **GPU 実行時** の干渉判定は対象外。未検証。
- **メモリ帯域・I/O 干渉** のように CPU 使用率に現れない干渉は検出できない。未検証。
- 閾値の **他ホストでの妥当性** は未検証（本 Issue は単一ホストで評価）。
- `sample_interval_s=0.5` 秒より **短い瞬間的な干渉** は取りこぼす可能性がある。未検証。
- `psutil` による子プロセス CPU 集計が **短命プロセスを取りこぼす** 可能性がある。未検証。
- 干渉検出は「無効 run の除外」を可能にするものであり、無効 run のレイテンシ値を補正・復元するものではない。
