# PaDiM / PatchCore の共通条件比較

このドキュメントは、同一カテゴリ・同一入力条件で PaDiM と PatchCore を比較し、
精度・時間・メモリの結果を CSV / Markdown に集計する手順を説明します。
特定の実測値は記載しません。数値は必ず実行後の `report.json` から集計します。

## 前提

- CPU 優先の環境で逐次実行します。並列実行は計測干渉を避けるため行いません。
- `scripts/run_anomalib.py` は `--output-dir D` に対して
  `D/<run_id>/report.json` を書き、失敗時も report を書いて exit 1 になります。
- 学習途中のチェックポイント `*.ckpt` は `D/<run_id>/` 以下に保存されます。
- repeat は同一 seed で実行します。seed はプロトコル JSON 側で固定し、
  繰り返しはモデル条件の違いではなく計測ばらつきの評価に使います。
- 欠損指標は CSV では空文字、Markdown では `—` として扱います。
  0 や NaN で補いません。

## 1. dry-run で実行内容を確認する

```bash
python scripts/run_matrix.py \
  --config configs/experiments/mvtec_ad_cpu_baseline.json \
  --dry-run
```

`--dry-run` は実行せず、run_key と `run_anomalib.py` に渡す argv を表示します。
一部だけ確認・実行する場合は `--only` を使います。

```bash
python scripts/run_matrix.py \
  --config configs/experiments/mvtec_ad_cpu_baseline.json \
  --dry-run \
  --only padim__bottle
```

## 2. 実験を実行する

```bash
python scripts/run_matrix.py \
  --config configs/experiments/mvtec_ad_cpu_baseline.json \
  --output-dir results/matrix/mvtec_ad_cpu_baseline
```

主なオプション:

- `--output-dir`: 出力先。省略時は `results/matrix/<name>`。
- `--only <部分一致>`: run_key の部分一致で対象を絞る。複数指定可。
- `--resume`: `runs.jsonl` に記録済みの run_key をスキップする。
- `--timeout-s`: 1 実行あたりのタイムアウト秒。既定は 3600。

出力例（ディレクトリ構造）:

```text
results/matrix/<name>/
  matrix_meta.json
  matrix.log
  runs.jsonl
  logs/
    <run_key>.log
  runs/
    <run_key>/
      <run_id>/
        report.json
        *.ckpt        # keep_checkpoints=false なら実行後に削除
```

`matrix_meta.json` には設定全文、設定ファイルの sha256、開始時刻 UTC、
取得できた場合の git commit、Python 実行ファイルを保存します。
`runs.jsonl` には 1 実行 1 行で run_key、条件、returncode、壁時計時間、
report の相対パス、status、error を追記します。

`keep_checkpoints` が `false` の場合、実行後に `runs/<run_key>` 以下の
`*.ckpt` を削除し、削除したファイル数とバイト数を `matrix.log` に記録します。

### 長時間実行

端末切断に備えて `nohup` で回し、中断・失敗後は `--resume` で再開できます。

```bash
nohup python scripts/run_matrix.py \
  --config configs/experiments/mvtec_ad_cpu_baseline.json \
  --output-dir results/matrix/mvtec_ad_cpu_baseline \
  --timeout-s 3600 \
  > results/matrix/mvtec_ad_cpu_baseline.nohup.log 2>&1 &
```

再開:

```bash
python scripts/run_matrix.py \
  --config configs/experiments/mvtec_ad_cpu_baseline.json \
  --output-dir results/matrix/mvtec_ad_cpu_baseline \
  --resume
```

OOM 等で子プロセスが SIGKILL された場合、report が無いことがあります。
その場合 status は `failed`、error は `signal SIGKILL` とログ末尾から作った要約になります。
タイムアウトの場合は status が `timeout` になります。

## 3. 集計する

```bash
python scripts/aggregate_results.py \
  --matrix-dir results/matrix/mvtec_ad_cpu_baseline \
  --out-dir results/matrix/mvtec_ad_cpu_baseline
```

出力:

- `runs.csv`: 1 実行 1 行の明細。
- `summary.csv`: (model, category, image_size, cpu_threads) ごとの集計。
- `report.md`: 実行条件、カテゴリ別比較表、失敗一覧、指標の注記。

行の並びは入力順ではなくキーで安定ソートします。入力の並べ替えで
指標の意味や集計値が変わらないようにしています。

## 4. runs.csv の列

| 列 | 意味 |
|---|---|
| `run_key` | 実行識別子。`<model>__<category>__s<size>__t<threads>__r<repeat>` |
| `model` | `padim` または `patchcore` |
| `category` | データセットのカテゴリ |
| `image_size` | 入力画像の一辺。report の `image_size` ではなく実行条件 |
| `cpu_threads` | 要求 CPU スレッド数 |
| `repeat` | 同一条件内の繰り返し番号（1 始まり） |
| `status` | `ok` / `failed` / `timeout` |
| `image_AUROC` | engine.test の Image AUROC |
| `pixel_AUROC` | engine.test の Pixel AUROC |
| `image_F1Score` | engine.test の Image F1 |
| `pixel_F1Score` | engine.test の Pixel F1 |
| `image_AUROC_raw` | 正規化前の生スコアによる Image AUROC |
| `pixel_AUROC_raw` | 正規化前の生スコアによる Pixel AUROC |
| `train_stage_seconds` | `engine.fit` 工程全体の秒 |
| `test_stage_seconds` | `engine.test` 工程全体の秒 |
| `e2e_p50_ms` | 画像 1 枚あたり end-to-end の P50 ミリ秒 |
| `e2e_p95_ms` | 画像 1 枚あたり end-to-end の P95 ミリ秒 |
| `model_p50_ms` | モデル推論部分の P50 ミリ秒 |
| `model_p95_ms` | モデル推論部分の P95 ミリ秒 |
| `train_peak_rss_mib` | 学習工程のサンプリング最大 RSS（MiB） |
| `process_peak_rss_mib` | プロセス寿命中のサンプリング最大 RSS（MiB） |
| `protocol_name` | プロトコル名 |
| `manifest_sha256` | データ分割 manifest の sha256 |
| `error` | 失敗時のエラー要約。成功時は空 |
| `notes` | NaN/inf を欠損にした場合などの注記 |

CSV の欠損は空文字です。NaN/inf は欠損として扱い、
`notes` に `nonfinite:<列名>` を記録します。

## 5. summary.csv の列

グループは `(model, category, image_size, cpu_threads)` です。

- `n_runs`: そのグループの実行数。
- `n_ok`: status が `ok` の実行数。
- `n_failed`: `ok` 以外の実行数。`failed` と `timeout` を含みます。
- `manifest_consistent`: ok 実行間で manifest_sha256 が一致する場合 true。
- 各数値列の `_mean`, `_std`, `_min`, `_max`。
  - 平均・標準偏差・最小・最大は `ok` の実行だけから計算します。
  - 標準偏差は標本標準偏差で、n_ok が 2 未満のときは欠損です。
  - n_ok が 0 のグループも行を残し、数値は欠損です。

## 6. 指標の意味

- engine 指標は val 由来の閾値・正規化を経た後の metrics
  （`engine.test` の `test_metrics[0]`）です。
- raw 指標は post_processor による正規化を通さない生スコアの
  順位ベース指標（`raw_score_metrics`）です。
- 学習時間は `engine.fit` 工程全体です。
- レイテンシはバッチ 1 で計測し、ウォームアップ画像を集計から除外します。
- RSS は psutil によるサンプリング最大値です。短いピークを取り逃し得ます。
- repeat は同一 seed で実行しており、計測ばらつきの評価です。

## 7. 失敗の扱い

- report が存在しない実行は status `failed` として `runs.csv` に残します。
- report に `error_type` / `error_message` があればそれを `error` に使います。
- report が無い場合は returncode とログ末尾 20 行から要約を作ります。
- シグナル終了は `signal SIGKILL` のように表記します。
- タイムアウトは status `timeout` として記録します。
- 失敗した実行の指標は平均に入れません。存在しない指標を架空の数値で補いません。
