# Issue #13 検証記録: タイル分割推論（Tiling）

- 対象 Issue: #13「重なりのあるタイル分割と元画像座標への統合」
- 実装: `adbench/tiling.py`（新規）, `scripts/run_anomalib.py`（拡張）, `tests/test_tiling.py`, `tests/test_tiling_cli.py`, `docs/tiling.md`
- 実測データ: `benchmarks/issue13/mvtec4k_bottle/runs.csv`（本記録の数値はすべてこの CSV の列値。丸めは表示のみ）
- 記述方針: 実測値と見積もりを区別する。見積もりは「見積もり」、未確認事項は「未検証」「コード未確認」と明記する。推測を実測として書かない。

## 1. 目的

全体縮小（global resize）では 4K 級画像の微小欠陥が作業解像度上で 1 画素未満に潰れ、異常マップ上で消失しうる。Issue #13 では次を行う。

1. 作業解像度内で重なりを持つタイル分割推論を追加する。
2. タイル推論結果を元画像（作業解像度）座標へ再投影する。
3. 重複領域の統合ルールを定義し、隙間・座標ズレがないことをテストで確認する。
4. 全体縮小とタイル分割の精度・速度・メモリを比較する。

本記録は、上記の実装内容と CPU 実行での実測結果、および CPU で処理可能な条件とメモリ上限をまとめる。Issue の受け入れ条件との対応は、座標一致 → 3.3、全体縮小とタイルの結果を区別して保存 → 3.2（report の `spatial_strategy`）、CPU で処理可能な条件とメモリ上限 → 6 章。

## 2. 実験条件

- データセット: MVTec-4K bottle（Anomalib 用ビュー。元画像は 4K 級）。test 132 枚。
- プロトコル: `mvtec_ad_v1`（val_split from_test 0.2, seed 42）。
- モデル: PaDiM / PatchCore（EfficientAD はタイル非対応のため対象外。3.2）。
- 実行環境: CPU 8 スレッド、GPU なし、スワップなし、測定時の空きメモリ約 7 GiB。
- バッチ: train / eval とも batch size 1。
- 反復: 各条件 1 回のみ（信頼区間なし）。
- 前処理: `--image-size N` で元画像を N×N に縮小し、その解像度を「作業解像度」としてタイル化する。タイル分割時もモデル入力は N×N である。
- `global_256`（PaDiM / PatchCore）は Issue #12 の前処理修正後の実行（`results/issue12/...`）を参照する。同一コード系統のため比較に用いる。
- AUROC は `image_auroc_engine` / `pixel_auroc_engine` を使用。`*_raw` との差は全条件で 1e-4 未満（image AUROC は一致、pixel は最大 7.3e-5）。
- レイテンシ: `model_p50_ms` / `model_p95_ms` はモデル推論時間。`e2e_p50_ms` / `e2e_p95_ms` は 1 枚あたりの end-to-end 時間（decode / preprocess / model / postprocess を含む。工程別の P50/P95 は各 report.json の `per_image_inference_latency` に記録。4K 画像の decode が支配的）。

## 3. 実装の要点

### 3.1 タイル計画 `adbench/tiling.py`（標準ライブラリのみ）

- `TilingConfig(tile_size, stride, mode, remove_border_count)` でタイル構成を表す。
- 主な API（シグネチャは `docs/tiling.md` と実装を参照）: `tile_starts`, `padded_length`, `requires_padding`, `plan_tiles`, `axis_coverage`, `coverage_summary`, `validate_for_model`, `describe_tiling`, `describe_global`。
- `plan_tiles` は作業解像度と config からタイル開始座標の列を返す。並びは Anomalib `Tiler.tile` の出力順（batch → 行 → 列）に一致させる。
- `requires_padding` / `validate_for_model` は次を検査し、違反時は例外とする（CLI では終了コード 2）。
  - パディング不要であること: 各辺で `(image - tile) % stride == 0`。
  - `tile <= 作業解像度`。
  - `stride <= tile`。
- `coverage_summary` / `describe_tiling` / `describe_global` は report と CLI 出力用の人間向け記述（軸ごとの被覆率・重なり・全体縮小の説明）。
- 純粋ロジックのみで torch / anomalib に依存しない（CLI の `--help` と基本テストは重い依存なしで動作）。

### 3.2 `scripts/run_anomalib.py`

- 追加 CLI: `--tile-size`, `--tile-stride`（省略時は `--tile-size` と同値）, `--tile-mode {padding,interpolation}`。
- タイル指定時は `Tiler(tile_size, stride, mode, remove_border_count)` を **モデル生成直後、およびチェックポイント読込時** に `model.model.tiler` へ設定する（Anomalib の `TilerConfigurationCallback.setup` と同等の状態を作る）。
- 対応モデルは PaDiM / PatchCore のみ。EfficientAD は torch model に `tiler` 属性が無いため、タイル指定時は終了コード 2 で終了する。
- 終了コード 2 となる構成: パディングが必要な構成、`tile > 作業解像度`、`stride > tile`。パディングを許可しない理由は 3.3 の 4 行目（Anomalib `Tiler.untile` の切り出しが入力画素サイズで行われるため、特徴マップでは切り出しが効かず座標が保存されない）による。
- `remove_border_count` は CLI では 0 固定（0 以外は未検証）。
- report に `spatial_strategy` を保存する。内容は `global_resize` / `tiled`、`grid`、`tiles_per_image`、`overlap_ratio`、`coverage`、`merge_rule`。これにより全体縮小とタイルの結果を区別して保存できる（Issue 受け入れ条件）。
- タイル有効性の確認: `tile512_t256_s256`（image AUROC 0.9196）と `global_512`（0.9133）で結果が異なるため、Tiler が実際に適用されていると判断した。同一結果なら Tiler が無効である疑いがあった。

### 3.3 座標一致の確認方法（テスト）

| 観点 | テスト | 内容 |
| --- | --- | --- |
| Tiler 実体での tile → untile 一致 | `tests/test_tiling.py` | 座標を値に埋め込んだテンソルを Anomalib `Tiler` で tile → untile し、重複領域のカウント平均後も元座標が復元されることを確認 |
| タイル順序と座標の一致 | `tests/test_tiling.py` | `plan_tiles` が返す座標列が `Tiler.tile` の出力順（batch → 行 → 列）と一致することを確認 |
| 1/4 解像度特徴の整合 | `tests/test_tiling.py` | パディング不要構成で、1/4 特徴を untile した結果が全体 avg_pool と一致することを確認 |
| パディング構成の回帰 | `tests/test_tiling.py` | パディングが必要な構成では特徴形状が padded/4 になり、切り出しが効かない（= 座標が保存されない）ことを固定する回帰テスト |
| 1 画素欠陥の座標保存 | `tests/test_tiling.py` | 画像中に 1 画素の欠陥を置き、untile 後の異常マップの最大位置が元座標と一致することを確認 |
| CLI の受理・拒否 | `tests/test_tiling_cli.py`（5 件） | 3.2 に示す終了コード 2 の条件と、タイル指定時の受理 |

テスト件数: `tests/test_tiling.py` 23 件、`tests/test_tiling_cli.py` 5 件（計 28 件）。リポジトリ全体 254 件 OK、`check_repo_safety` 違反なし。

## 4. 結果

各条件 1 回のみの実行。`global_256_ref_issue12` は Issue #12 の実行を参照。丸めは表示上のもので、原値は CSV を参照。

### 4.1 PaDiM（4 条件）

| 条件 | 作業解像度 | タイル | stride | タイル数 | image AUROC | pixel AUROC | train 工程 s | model P50/P95 ms | e2e P50/P95 ms | train peak RSS MiB | test peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `global_256_ref_issue12` | 256×256 | — | — | 1 | 0.9262 | 0.9922 | 102.98 | 40.4 / 134.5 | 542.9 / 717.3 | 1647.32 | 1155.54 |
| `global_512` | 512×512 | — | — | 1 | 0.9133 | 0.9936 | 124.96 | 122.5 / 215.5 | 515.5 / 622.2 | 4318.97 | 2750.78 |
| `tile512_t256_s256` | 512×512 | 256 | 256 | 4 | 0.9196 | 0.9943 | 134.71 | 119.2 / 184.2 | 520.0 / 606.8 | 4385.64 | 2522.73 |
| `tile512_t256_s128` | 512×512 | 256 | 128 | 9 | 0.9202 | 0.9942 | 142.53 | 209.5 / 782.9 | 634.5 / 1269.0 | 4961.04 | 2661.12 |

### 4.2 PatchCore（2 条件）

| 条件 | 作業解像度 | タイル | stride | タイル数 | image AUROC | pixel AUROC | train 工程 s | model P50/P95 ms | e2e P50/P95 ms | train peak RSS MiB | test peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `global_256_ref_issue12` | 256×256 | — | — | 1 | 0.8746 | 0.9948 | 514.21（5.3 参照、比較しない） | 94.4 / 167.3 | 501.6 / 583.3 | 2411.07 | 1377.18 |
| `tile256_t128_s128` | 256×256 | 128 | 128 | 4 | 0.8595 | 0.9945 | 196.59（5.3 参照、比較しない） | 166.4 / 1010.3 | 674.6 / 1596.4 | 2457.39 | 1209.51 |

### 4.3 メモリ（実測、ステージ別）

最右列は CSV に記録された 4 工程（train / test / raw 評価 / レイテンシ計測）のサンプリング RSS の最大値。

| モデル / 条件 | train | test | raw 評価 | レイテンシ計測 | 4 工程の最大 |
| --- | --- | --- | --- | --- | --- |
| PaDiM `global_256_ref_issue12` | 1647.32 | 1155.54 | 1537.24 | 1164.28 | 1647.32 |
| PaDiM `global_512` | 4318.97 | 2750.78 | 3627.82 | 2040.50 | 4318.97 |
| PaDiM `tile512_t256_s256` | 4385.64 | 2522.73 | 3469.59 | 1766.43 | 4385.64 |
| PaDiM `tile512_t256_s128` | 4961.04 | 2661.12 | 3731.96 | 2146.02 | 4961.04 |
| PatchCore `global_256_ref_issue12` | 2411.07 | 1377.18 | 1565.93 | 1294.27 | 2411.07 |
| PatchCore `tile256_t128_s128` | 2457.39 | 1209.51 | 1563.96 | 1196.27 | 2457.39 |

（単位 MiB）

## 5. 観察

数値の差はすべて各条件 1 回の実行に基づく。test 132 枚・1 回の測定では、image AUROC の差が 0.01 未満の場合は誤差範囲である可能性があり、優劣は断定しない。

### 5.1 PaDiM

- 512 化で image AUROC は 256 より低下する（0.9262 → 0.9133、差 −0.0129）。
- タイル化は 512 global より image AUROC が僅かに高い（`tile512_t256_s256` 0.9196、`tile512_t256_s128` 0.9202。差 +0.0063 / +0.0069）が、256 global（0.9262）は超えない（差 −0.0067 / −0.0060）。差はいずれも 0.01 未満で、上記のとおり誤差範囲の可能性がある。
- pixel AUROC は 256（異常マップ 256×256）と 512（512×512）で異常マップ解像度が異なるため直接比較できない。512 内の 3 条件（global 0.9936 / s256 0.9943 / s128 0.9942）は同じ解像度で比較可能であり、差は 1e-3 未満。
- train peak RSS は、256 の 1647.32 MiB に対し 512 の 3 条件が 4318.97 / 4385.64 / 4961.04 MiB で、約 2.6〜3.0 倍。タイル化自体（作業解像度が同じ 512 内の比較）はメモリをほとんど増やさない（`global_512` 4318.97 MiB に対し `tile512_t256_s256` 4385.64 MiB）。test peak RSS は 512 内で 2522.73〜2750.78 MiB。
- 重なり 50%（9 タイル、`tile512_t256_s128`）は model P50 が 209.5 ms で、重なりなし 4 タイル（`tile512_t256_s256`、119.2 ms）の約 1.8 倍（1.76 倍）。e2e P50 は 520.0 → 634.5 ms（+114.5 ms、1.22 倍）で、推論以外の固定費のため伸びは小さい。
- train 工程秒はタイル数にともない 124.96 → 134.71 → 142.53 s と緩やかに増える。

### 5.2 PatchCore

- `tile256_t128_s128` は image AUROC が 0.8746 → 0.8595（差 −0.0151）と低下し、同値スコアが 6 件 → 21 件に増加した。原因は未特定（タイル数の増加に伴う文脈（受容野・近傍特徴）の変化などが候補だが未検証）。
- pixel AUROC は 0.9948 → 0.9945（差 −3.2e-4）。異常マップサイズは両条件とも 256×256。
- train peak RSS は 2411.07 → 2457.39 MiB（+1.9%）でほぼ同じ。test peak RSS は 1377.18 → 1209.51 MiB。e2e P50 は 501.6 → 674.6 ms（+173.0 ms、1.34 倍）。

### 5.3 比較しない項目・外れ値

- PatchCore の train 工程秒は比較しない。Issue #12 の同一系統の実行で 227 s と 514 s という原因不明のばらつきがあり、本記録の 514.21 s と 196.59 s の差をタイル化の効果として解釈できない。
- P95 の外れ: PaDiM `tile512_t256_s128` の model P95 782.9 ms（P50 209.5 ms）、PatchCore `tile256_t128_s128` の model P95 1010.3 ms（P50 166.4 ms）。いずれも 1 回の測定で、原因は未特定（スレッド競合・ページキャッシュ等の可能性はあるが未検証）。

## 6. CPU で処理可能な条件とメモリ上限

測定時の空きメモリは約 7 GiB（スワップなし、GPU なし）。

### 6.1 実測で完走した範囲

- PaDiM: 作業解像度 512 / tile 256 / stride 128（9 タイル、重なり 50%）まで完走。train peak RSS 4961.04 MiB（約 4.85 GiB）、model P50 209.5 ms / P95 782.9 ms。
- PatchCore: 作業解像度 256 / tile 128 / stride 128（4 タイル）まで完走。train peak RSS 2457.39 MiB（約 2.40 GiB）。
- 上記はいずれも train / eval batch size 1、8 スレッドでの実測。これを超える構成は本記録では未検証。

### 6.2 見積もり（未実行）

以下は **見積もり** であり実測ではない。

- PaDiM 作業解像度 1024（1/4 特徴 = 256×256）: 学習時のメモリバンク ≈ 146 × 100 × 256² × 4 B ≈ 3.83 GB（≈ 3.56 GiB）。連結時に一時的に 2 倍 ≈ 7.66 GB（≈ 7.13 GiB）。
  - 146 は学習画像枚数（ctx の式による）。
  - 100 は特徴次元数で、Anomalib 2.7.0 の resnet18 既定値（`padim/torch_model.py` の対応表で確認）。
  - 参考: 同じ式で作業解像度 512 は ≈ 0.96 GB だが、実測の train peak RSS は 4318.97 MiB だった。式はメモリバンク本体のみで、共分散・逆行列・backbone 活性・プロセス基礎分を含まないため下限に近い。
  - したがって空きメモリ約 7 GiB の環境では不足する可能性が高い（未実行・未検証）。
- PatchCore 作業解像度 512: 未実行。作業解像度を 2 倍にすると特徴マップの面積は 4 倍（解像度比の 2 乗）になるが、ctx に PatchCore 用の見積もり式が無いため数値は出さない（未検証）。

## 7. 再現手順

- テスト: `python -m unittest discover -s tests`（254 件 OK）。
- 実行例（タイル指定。既存の必須引数は省略している。正確な引数は `python scripts/run_anomalib.py --help` を参照）:

```bash
python scripts/run_anomalib.py ... --image-size 512 --tile-size 256 --tile-stride 256
python scripts/run_anomalib.py ... --image-size 512 --tile-size 256 --tile-stride 128
```

- 結果: `benchmarks/issue13/mvtec4k_bottle/runs.csv`。各行の `source` 列に report.json のパス、`run_id` に実行 ID を記録している。

## 8. 未検証事項

- 複数回の反復と信頼区間（全条件 1 回のみ）。
- `interpolation` モード（実装済み・CLI で選択可能だが、本検証では未使用）。
- `remove_border_count` が 0 以外の構成（CLI では 0 固定）。
- batch size > 1 でのピーク RSS とレイテンシ。
- bottle 以外のカテゴリ・他データセットでの挙動。
- 欠陥サイズ別の効果（#14 で扱う）。
- P95 外れ（PaDiM s128、PatchCore tile）の原因。
- PaDiM 1024 / PatchCore 512 の実測。
- パディング不要構成以外での Anomalib `Tiler.untile` の座標挙動（本実装では許可していない）。
