# Issue #7 検証記録: 標準プロトコル・データ分割 manifest・再現性

すべて実機計測値です（推定値は含みません）。環境は `docs/verification/issue6_latency.md` と同じです（Intel Xeon 8 vCPU、RAM 15.6 GiB、GPU なし、torch 2.15.0+cpu、anomalib 2.7.0）。

## 標準プロトコル

| プロトコル | val | test | 閾値の決め方 | 用途 |
|---|---|---|---|---|
| `configs/protocols/mvtec_ad_v1.json`（既定） | 公式 test から seed=42・label_aware で 20% | 残り 80% | val（test と重複なし） | 手法間比較の標準 |
| `configs/protocols/mvtec_ad_full_test.json` | test と同一（`same_as_test`） | 公式 test 全数 | test で調整（F1 等は楽観的） | 論文値との比較（AUROC のみ） |

## manifest の再現性（bottle）

`--manifest-only` を各プロトコルで別プロセスとして2回ずつ実行し、`manifest_sha256` が一致することを確認しました。

| プロトコル | train | val | test | val のラベル内訳 | manifest_sha256（先頭12桁） |
|---|---:|---:|---:|---|---|
| mvtec_ad_v1 | 209 | 16 | 67 | good 4, broken_large 2, broken_small 3, contamination 7 | 872f367684e4 |
| mvtec_ad_full_test | 209 | 83 | 83 | test と同一（重複 83 枚を許容として記録） | 14058eb7ddc6 |

- 層化は Anomalib の `label_index`（正常/異常）単位で、異常種別単位ではありません（上表のとおり種別ごとの比率は揃いません）。
- manifest 全体（相対パス一覧）: `benchmarks/issue7/manifests/`

## 実験結果（bottle, 256px, バッチサイズ1, 8スレッド, 各1回）

| model | protocol | engine Image AUROC | raw Image AUROC | engine Pixel AUROC | raw Pixel AUROC | 学習 [s] | 生涯最大RSS [MiB] |
|---|---|---:|---:|---:|---:|---:|---:|
| PaDiM | mvtec_ad_v1 | 0.9963 | 0.9975 | 0.9768 | 0.9749 | 10.73 | 1714.2 |
| PatchCore | mvtec_ad_v1 | 1.0000 | 1.0000 | 0.9857 | 0.9839 | 276.82 | 3209.5 |
| PaDiM | mvtec_ad_full_test | 0.9976 | 0.9984 | 0.9777 | 0.9757 | 14.34 | 1870.6 |
| PatchCore | mvtec_ad_full_test | 1.0000 | 1.0000 | 0.9856 | 0.9838 | 257.90 | 3203.7 |

- 全項目: `benchmarks/issue7/summary.csv` と同ディレクトリの JSON
- 同じ mvtec_ad_v1 条件の PaDiM bottle は、Issue #5・#6 の実行（別プロセス・別スレッド数）と engine の Image AUROC（0.99632）と Pixel AUROC（0.97676）が小数第5位まで一致しました（Pixel は float32 の丸め差で 1e-7 程度の違いあり）。
- PaDiM の full_test の raw Image AUROC 0.998413 は、検証用に別スクリプトで `engine.predict` の生スコアから sklearn で計算した値と一致しました。

## 検証中に見つかった問題と対処

1. **val なし（`val_split_mode: none`）では engine.test の AUROC が信頼できない**: PaDiM bottle で Image AUROC が 0.675、別実行で 0.625 となりました。同じモデルの生スコアからは 0.9984 でした。PostProcessor の正規化パラメータ・閾値が NaN のままになるためと見られます。→ `none` は CLI・プロトコル検証で拒否し、full_test は `same_as_test` に変更しました。
2. **torchmetrics の `binary_auroc` は [0, 1] 外の値にシグモイドを掛ける**: 生スコア（約 18〜280）では飽和して raw AUROC が 0.5 になりました。→ 順位ベースの scikit-learn `roc_auc_score` に変更し、[0, 1] 外のスコアでの回帰テストを追加しました。

## 未検証・制限

- engine と raw の Pixel AUROC の差（約 0.002）の原因は特定していません（マスクのリサイズ方法の違いなどが考えられますが未確認です）。
- 他カテゴリでの manifest・raw 評価は未実行です。
- seed 固定での学習結果の決定性は PaDiM で一致を確認しました。PatchCore の coreset 選択が実行間で一致するかは、AUROC 1.0 で差が見えないため未確認です。
