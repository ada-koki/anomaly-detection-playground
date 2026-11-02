# 欠陥サイズ別メトリクス（Issue #14）

画像とマスクがあるデータセットで、欠陥サイズ帯ごとの検出率・見逃し率と、
正常画像の過検出率を集計するための定義と使い方をまとめる。数値の実測値はここには
書かない（ベンチマーク結果は run ごとの `report.json` と生成レポート側に置く）。

## 1. 欠陥サイズの定義

- **サイズ = 元画像座標のマスク面積率**。`area_ratio = マスク画素数 / 元画像画素数`。
- **外接矩形**は `[y0, x0, y1, x1]` で表す。`y1` / `x1` は排他的（Python のスライスと同じ）。
- 面積と外接矩形は **リサイズ前の元解像度マスク** で計算する。`anomaly_map` や作業解像度
  （`--image-size`）のサイズには依存しない。
- 連結成分数は 8 連結（斜めも同一成分）で数える。scipy が無い環境では `None` になる。
  ラベリングはメモリ節約のため外接矩形の内側だけで行う。
- マスクファイルが存在するが全画素 0 の場合は面積 0 として扱い、`classify_size(0.0)` は最小の帯（`tiny`）になる。全画素 0 の
  マスクは「欠陥面積が 0 の異常画像」として `overall` と異常種別別には数えられる。

## 2. サイズ帯

既定の帯は次の 3 つ。下限を含み、上限を含まない。最後の帯だけ上限（1.0）を含む。

| 帯 | 面積率の範囲 |
|---|---|
| `tiny` | `[0, 1e-3)` |
| `small` | `[1e-3, 1e-2)` |
| `large` | `[1e-2, 1.0]` |

帯の定義は `adbench/defect_size.py` の `SIZE_BANDS_DEFAULT` にあり、`summarize()` の
`bands` 引数で差し替えられる。レポート出力には必ず帯の定義を含める。

## 3. 閾値

- `image_threshold`（画像スコア）と `pixel_threshold`（`anomaly_map`）は
  **検証集合から求めた値**を使う。Anomalib 2.7.0 の `PostProcessor` が
  `F1AdaptiveThreshold` で推定する生スケールの値であり、`raw_eval` が扱う
  生の `pred_score` / `anomaly_map` と直接比較できる。
- **テスト集合で閾値を調整しない**。`same_as_test`（val と test が同一）の
  プロトコルでは閾値がテストデータ由来になるため、検出率・過検出率は
  楽観的にずれる。この場合はレポートの閾値の出所を確認すること。
- 閾値が NaN / 未計算（`val_split_mode=none` など）のときは、該当する率を `null`
  にし、注記へ「閾値未計算」を残す。0 で埋めない。

## 4. 分母の定義

| 指標 | 分母 | 分子 |
|---|---|---|
| `detection_rate` | 異常画像（`label=1`）の枚数 `n` | `score >= image_threshold` の枚数 |
| `miss_rate` | 同上 | `n - detected` |
| `region_hit_rate` | マスクがあり、かつ `anomaly_map` 解像度へリサイズしてもマスクが残った異常画像の枚数 | `map_max_in_mask >= pixel_threshold` の枚数 |
| `false_positive_rate` | 正常画像（`label=0`）の枚数 | `score >= image_threshold` の枚数 |
| `mean_normal_fp_pixel_ratio` | `normal_fp_pixel_ratio` を計算できた正常画像の枚数 | 各画像の `anomaly_map >= pixel_threshold` の画素割合の単純平均 |

補足:

- `detection_rate` / `miss_rate` は **画像単位**。異常種別・サイズ帯をまたいだ
  画素レベルの寄与は見ない。
- 正常画像は `detection_rate` の分母に入れない。正常画像は過検出率の表で別に扱う。
- `mean_normal_fp_pixel_ratio` の分母は画素ではなく画像（画像ごとの割合の平均）。
  画素数の重み付けはしていない。
- スコアが NaN の画像は未検出として数え、注記に件数を残す。

## 5. 領域単位ヒットの意味と解像度の影響

`map_max_in_mask` は「`anomaly_map` をマスクの内側だけで見た最大値」である。
`pixel_threshold` 以上なら、その画像は少なくとも 1 画素は閾値を超えている。

重要な注意:

- `anomaly_map` は **作業解像度**（`--image-size`、タイル方式でも同じ）で出力される。
  元解像度のマスクを最近傍でその解像度へ縮小すると、**微小な欠陥は 1 画素も
  残らないことがある**。その場合 `map_max_in_mask = None` となり、領域単位指標の
  分母から除外される（`n_mask_vanished` / `mask_px_at_map_resolution = 0`）。
- したがって `region_hit_rate` は「作業解像度で表現できた欠陥」に対する率であり、
  画像単位の `detection_rate` とは分母が異なる。両者を直接比較しない。
- `map_max_outside_mask` はマスク外の最大値。マスクが全面を覆う場合は `None`。
  正常画像は全画素がマスク外なので、実質的に画像全体の最大値になる。
- 縮小方式（`global_resize`）とタイル方式（`tiled`）の比較では、この
  「作業解像度でマスクが消えるかどうか」がサイズ帯別の差として現れ得る。
  どの程度消えるかはデータセットと作業解像度に依存し、本リポジトリでは未検証。

## 6. 注記ルール

各セルには `note` 文字列を付ける。主な内容:

- `n < min_samples`（既定 5）のセル: 「サンプル数が少ないため参考値」。
- `n = 0` のセル: 上記に加えて「n=0 のため率は算出できません」。率は `null`。
- マスクが無い異常画像があるセル: 「マスクが無い異常画像が N 件あり、領域単位指標の
  分母から除外しています」。
- 最近傍リサイズでマスクが消えた画像があるセル: 「作業解像度への最近傍リサイズで
  マスクが消失した異常画像が N 件あり、…」。
- スコアが NaN の画像があるセル: 件数を注記。
- 閾値が未計算のセル: 「閾値未計算（image_threshold / pixel_threshold）」。

## 7. 集計単位

`summarize()` は次の単位でセルを作る。すべて異常画像のみを対象とする
（正常画像は `normal` キーに別途まとめる）。

- `overall`: 異常画像全体。
- `by_size_band`: サイズ帯ごと（既定の帯は常に全帯を出力し、0 件の帯も残す）。
- `by_defect_type`: 異常種別ごと。異常種別は `image_path` の親ディレクトリ名
  （例: `.../test/broken_small/000.png` → `broken_small`、正常は `good`）。
- `by_defect_type_and_size_band`: 異常種別 × サイズ帯。

サイズ帯に分類できない（マスクが無い）異常画像は `overall` と異常種別別には
含まれるが、サイズ帯別には含まれない。件数は
`counts.n_abnormal_without_size_band` で確認できる。

## 8. 使い方

`scripts/run_anomalib.py` は生スコア評価（`--skip-raw-eval` でない場合）で自動的に
画像ごとのレコードを run ディレクトリの `per_image.jsonl` に保存し、集計を
`report.json` の `defect_size` に記録する（閾値は `PostProcessor` の検証集合由来の値）。
複数 run の比較は `scripts/defect_size_report.py --runs DIR... --out-dir OUT` で行う。
以下は個別に呼び出す場合の例。

### 8.1 生スコア評価で画像ごとのレコードを集める

```python
from adbench.raw_eval import evaluate_raw_score_metrics

result = evaluate_raw_score_metrics(
    model,
    samples,
    device="cpu",
    collect_per_image=True,
    image_threshold=float(post_processor.image_threshold),
    pixel_threshold=float(post_processor.pixel_threshold),
)
records = result["per_image_records"]  # list[PerImageRecord]
```

`record["per_image_records"]` は JSON 化前のオブジェクトなので、保存は呼び出し側で
`adbench.defect_size.records_to_jsonl_lines()` を使う。既存の返り値キー
（AUROC など）は変更していない。

### 8.2 集計する

```python
from adbench.defect_size import summarize

summary = summarize(records, image_threshold, pixel_threshold, min_samples=5)
```

### 8.3 CLI でレポートを生成する

```bash
python scripts/defect_size_report.py \
    --runs runs/run_global_resize runs/run_tiled_256 \
    --labels global_resize tiled_256 \
    --out-dir reports/defect_size
```

各 run ディレクトリには `report.json` と `per_image.jsonl` が必要。
`report.json` の `defect_size.thresholds` を閾値として使い、
`per_image.jsonl` から再集計する。

出力:

- `defect_size_summary.csv`: 集計セルを 1 行 1 セルで並べたもの。
  列は `run_label, spatial_strategy, working_resolution, tile_size, stride, model,
  group, key, n, detected, detection_rate, miss_rate, region_hit_rate, normal_fpr,
  image_AUROC_raw, train_stage_seconds, e2e_p50_ms, e2e_p95_ms, train_peak_rss_mib, note`。
  `normal_fpr` は run 単位の正常画像 FPR（全行に同じ値が入る）。
- `defect_size_summary.md`: 日本語の表。定義・分母・帯・閾値・注記を含む。
- `detection_by_size.png`: matplotlib がある場合のみ。横軸がサイズ帯、棒が run ごとの
  画像単位検出率。凡例に e2e P50 と train peak RSS を併記する。matplotlib が無い場合は
  注記だけ残してスキップする（CSV / MD は生成される）。

## 9. 実装上の注意（メモリ）

- マスク統計は 1 画像ずつ計算し、元解像度の配列を保持し続けない。
- 連結成分のラベリングは外接矩形の内側だけで行う。全面 4096×4096 のような
  大きなマスクでも、欠陥が小さいうちはラベル配列が小さく済む。
- scipy が無い環境では連結成分数を `None` にし、他の統計（面積・外接矩形）だけ返す。
- 画素 AUROC 用の連結配列は従来どおり全画像分を保持する（`raw_eval` の既存挙動）。

## 10. 未検証事項

- 作業解像度（`--image-size`）と縮小方式ごとに、マスクが消失する頻度がどの程度かは
  本リポジトリのデータでは未検証。
- 8 連結を採用した妥当性（欠陥が分離して見えるか）はデータセット依存で未検証。
- matplotlib の日本語フォント有無は環境依存のため未検証。PNG の軸ラベルは英語にしてある。
- サイズ帯ごとの検出率差の統計的有意性は評価していない（信頼区間も出していない）。
- `same_as_test` プロトコルでの楽観バイアスの大きさは未検証。`from_test` で
  val / test を分けた run と併記して確認する想定。
