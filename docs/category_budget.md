# カテゴリ予算（VisA / MVTec AD 2）— category_budget_v1

本ドキュメントは Issue #32 で導入した「1 カテゴリあたり予算」の考え方と、
設定ファイル `configs/budgets/category_budget_v1.json`、
集計 CLI `scripts/summarize_categories.py` の使い方を説明する。
CPU のみ（8 vCPU）・RAM 16 GB・スワップ無しという制約下で、
データセットのカテゴリを追加しても現実的な時間とメモリで完走できるかを
判定するための「目安」である。厳密な合否基準ではない。

## 1. 予算の定義

1 カテゴリ・1 run あたり次の 3 項目を上限として扱う。

- `train_stage_seconds`: 学習ステージに要した秒数
- `peak_rss_mib`: プロセスの生涯ピーク RSS（MiB）
- `download_mib`: その run のために取得したデータ量（MiB）

いずれも値が小さいほど良い。run が 3 つすべてで上限以下なら `within_budget=True`、
1 つでも超えれば `False`、判定に必要な値がすべて欠けていれば `None` とする。
`status` が `ok` でない run（OOM Kill、メモリガード停止、report 欠落など）は常に `False` とする。
個別指標が欠けている場合は `unavailable` に列挙し、残りの指標で判定する。

## 2. configs/budgets/category_budget_v1.json の各キー

| キー | 型 | 既定 v1 の値 | 意味 |
| --- | --- | --- | --- |
| `name` | str | `"category_budget_v1"` | 予算セットの識別子 |
| `max_train_stage_seconds` | number / null | `1800` | 学習ステージ秒数の上限（null なら判定しない） |
| `max_peak_rss_mib` | number / null | `6000` | ピーク RSS の上限（MiB） |
| `max_download_mib` | number / null | `2048` | データ取得量の上限（MiB） |
| `is_provisional` | bool | `true` | 暫定値であることの明示（既定 true） |
| `notes` | str | 任意 | 値の由来・注意書き（自由記述） |

未知のキー、負値、数値欄への bool はエラー（`ValueError`）になる。

これらは「暫定値」である。統計的な根拠は無く、次の既存実測を基準にした目安に過ぎない。

- Issue #16 の VisA candle PatchCore global は OOM Kill で完走しなかった（Kill のためピーク RSS は report に残っていない）。Issue #30 のメモリガードでは特徴量テンソルの概算が 11991.8 MiB（概算であり実測ではない）。
- MVTec AD 2 vial PatchCore global は学習ステージが 1895.38 s（同一ホストの別ジョブあり。干渉なし再計測は 1646.26 s）、学習ピーク 4194.06 MiB。

予算値はハードウェアやデータセットが変われば再設定が必要である（他ホストでの妥当性は未検証）。

## 3. カテゴリ選定の基準

「予算内に収まりそうか」を観測可能な指標で見積もってから選ぶ。主な指標は次。

- 学習画像枚数（ビュー作成後の `train_good` 枚数）
- 抽出後サイズ（ディスク上、MiB）

例（Issue #32）:

- VisA.tar（全 12 カテゴリ共通、1,929,840,640 bytes = 1840.44 MiB、S3 から取得）を一度取得し、
  その中から cashew・chewinggum・pcb1 を抽出、tar は削除した。
- train 枚数が約 450 の cashew / chewinggum と、candle（900 枚）と同規模の pcb1 を選んだ。
- candle の chunked100 が学習 1748.553 s で予算内だったため、900 枚級の pcb1 も
  予算内に収まる「見込み」で選んでいる（見込みであり、実測は検証記録を参照）。

## 4. scripts/summarize_categories.py

### 入力: runs.jsonl

1 行 1 レコードの JSON Lines。各行は次のキーを持つ。

- `condition` (str): run の識別名（例 `visa_cashew_padim`）
- `dataset` (str): `visa` / `mvtec_ad2` など
- `category` (str): カテゴリ名
- `model` (str): `padim` / `patchcore` など
- `report_path` (str): run の `report.json` へのパス（存在しない場合もある）
- `failure_status` (str, 任意): report が無い／失敗した run の理由（例 `oom_killed`）
- `download_mib` (number, 任意): その run のデータ取得量

`report_path` のファイルが無い行は `failure_status`（無ければ `missing_report`）で
失敗行として残す。`report.json` の `status` が `ok` 以外の行も失敗行として残す。

### 出力: 3 ファイル（`--out-dir` 配下）

- `category_runs.csv`: run ごとの 1 行（condition, status, variant, img/pix AUROC,
  train/test 秒, peak RSS, e2e p95, latency_valid, 枚数, within_budget, reasons）。
- `dataset_summary.csv`: `dataset × model × variant` 単位の集計
  （n_runs, n_ok, n_failed, n_within_budget, AUROC の mean/std/min/max,
  学習時間の合計・最大, ピーク RSS 最大）。
- `category_report.md`: 上記を人間向けに表形式でまとめた Markdown。

`std` は標本標準偏差である。カテゴリ数が少ないときのばらつき推定は小標本になる点に注意。

## 5. report.json の evaluation_scope

`scripts/run_anomalib.py` は `visa` / `mvtec_ad2` のとき、report.json に
`evaluation_scope` を記録する。主なキー:

- `dataset`: データセット名
- `source`: 入力ソース（ビュー manifest があればそこから写す）
- `excluded`: 評価から除外した split 名の一覧
- `public_test_only`: MVTec AD 2 のとき `true`、VisA のとき `null`
- `view_manifest_found`: ビュー manifest を読めたか
- `counts`: 画像枚数などのサマリ（manifest が無ければ `null`）

この項目は Issue #32 で追加された。実行順の後半（chewinggum / pcb1 の PatchCore run）から
記録されるようになり、それ以前に実行した run の report.json には存在しない。

## 6. MVTec AD 2 は公開 test split のみで評価する

MVTec AD 2 は test が `test_public` / `test_private` / `test_private_mixed` に分かれる。
`test_private` / `test_private_mixed` は正解（ラベル・マスク）が非公開のため、
本ベンチマークでは使わない。`evaluation_scope.public_test_only = true` と
`excluded` に除外 split を明示することで、評価範囲を report.json から追跡できる。

## 7. 限界

- 予算値（1800 s / 6000 MiB / 2048 MiB）は暫定の目安で、統計的根拠は無い。
- 他ホスト・他データセットでの妥当性は未検証。
- レイテンシは干渉の影響を受けうる。暫定の干渉判定を併用しているが、裾（p95 など）の安定性は未検証。
- PatchCore の chunked 構成（チャンク 100 枚ごとの KCenterGreedy、`sampling_ratio` 0.1）は
  プロトコル逸脱の近似であり、global と同一条件の比較ではない。
  `variant` 列 `chunked_coreset_100img` で分離して扱う。
- ダウンロード時間は判定対象外（未記録）の場合がある。
