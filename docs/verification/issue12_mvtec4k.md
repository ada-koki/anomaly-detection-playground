# Issue #12 検証記録: MVTec-4K（bottle）のデータ整合と 256px 縮小入力での CPU 実行

この記録は実測値のみを掲載します。各実行は 1 回ずつで、ばらつきは評価していません。推定値には「推定」と明記します。

## 条件

| 項目 | 値 |
| --- | --- |
| データ | MVTec-4K `bottle`（HF `XimiaoZhang/MVTec-4K`、revision `671e4f408db5bb42db5c2e99476e05e55490bc38`） |
| 取得量 / 時間 | bottle と直下のメタデータで約 4.2GB、約 19 分（2026-10-31 19:03〜19:22 JST、この環境の回線での 1 回の値） |
| 環境 | Linux、8 論理コアの共有 VM、GPU なし、RAM 15GiB |
| ソフトウェア | Python 3.13.5、torch 2.15.0+cpu、torchvision 0.29.2+cpu、Anomalib 2.7.0、Lightning 2.6.6 |
| 分割 | val_split `from_test`（0.2、seed 42）。train 146 / val 33 / test 132 |
| モデル入力 | 4096×4096 → 256×256 に縮小（`resolution_condition.condition = downscaled`）、8 スレッド |

データ・重みはリポジトリに含めていません（MVTec AD 由来の CC BY-NC-SA 4.0 として扱う。[docs/mvtec4k.md](../mvtec4k.md)）。

## 1. データ整合検証（`scripts/validate_mvtec4k.py`）

```bash
python scripts/validate_mvtec4k.py --root ../datasets/MVTec-4K --category bottle \
  --out ../results/issue12/validate_bottle.json --make-view ../datasets/MVTec-4K-view
```

- 件数: train 146、test good 50、large 47（broken_large 10 / broken_small 22 / contamination 15）、tiny 68。論文表（HiAD, arXiv:2508.12931）の bottle の値と一致しました。
- 画像サイズ: 311 枚すべて 4096×4096。エラー 0、警告 0。
- 全画素検査あり（`--skip-pixels` なし）で 139.3 秒、最大 RSS 211 MiB。
- clean view: ハードリンク 426 個（追加ディスク使用は約 60KB）。

マスク面積率（画像全体に対する欠陥画素の割合）と、マスクのうち foreground 内にある割合:

| 欠陥種別 | 面積率 最小 | 面積率 中央値 | 面積率 最大 | foreground 内割合 最小 |
| --- | ---: | ---: | ---: | ---: |
| broken_large | 0.0409 | 0.1125 | 0.1468 | 1.0 |
| broken_small | 0.00885 | 0.0244 | 0.0806 | 1.0 |
| contamination | 0.00576 | 0.0725 | 0.1653 | 0.929（中央値 1.0） |
| tiny | 8.34e-5 | 4.63e-4 | 1.43e-3 | 1.0 |

tiny の中央値 4.63e-4 は、256×256 に縮小すると約 30 画素に相当します（面積率からの**推定**で、縮小後マスクの実測ではありません）。

## 2. Anomalib のデータ読み込み確認（`make_mvtec_ad_dataset`）

| 配置 | 結果 |
| --- | --- |
| 生配置（`foreground/` あり） | `ValueError: Must have equal len keys and value when setting with an iterable`（重複行） |
| シンボリックリンクの view | `Found 0 images`（`is_within_directory()` がリンク先を解決し root 外として除外） |
| ハードリンクの view | 311 行（test 165 / train 146）、重複 0、マスク 115、foreground 混入なし |

このため `--make-view` の既定はハードリンク（`--link-mode hardlink`）です。また `scripts/run_anomalib.py --dataset mvtec_4k` に `foreground/` を含む生配置を渡すと、学習前に終了コード 2 で止まることを実機で確認しました。

## 3. 前処理の不一致の発見と修正

最初の実行で、engine の Image AUROC と生スコア評価（`raw_score_metrics`）が大きく食い違いました（PaDiM 0.926 / 0.744、PatchCore 0.875 / 0.759）。

調査の結果、生スコア評価とレイテンシ計測が `PreProcessor.forward()` を呼んでいたことが原因でした。Anomalib 2.7.0 の `forward()` は ONNX 互換用の `export_transform`（`Resize(..., antialias=False)`）を使い、`engine.test` は `on_test_batch_start` で `transform`（`antialias=True`）を使います。4096px→256px（16 分の 1）の縮小では、antialias の有無で入力がほぼ別物になります。

修正（`adbench/preprocess.py`）:

- 生スコア評価は常に `transform`（`eval_transform`）を使う。
- レイテンシ計測は既定 `eval_transform`、`--latency-preprocess export_transform` で従来条件も選べる。
- 使った変換を `raw_score_metrics.preprocess` / `per_image_inference_latency.preprocess` に記録する。

修正前後の比較（元データ: [`benchmarks/issue12/mvtec4k_bottle_256/runs.csv`](../../benchmarks/issue12/mvtec4k_bottle_256/runs.csv)）:

| 実行 | データ | モデル | Image AUROC engine / raw | Pixel AUROC engine / raw |
| --- | --- | --- | --- | --- |
| 修正前 | MVTec-4K | PaDiM | 0.9262 / 0.7444 | 0.9922 / 0.9877 |
| 修正前 | MVTec-4K | PatchCore | 0.8746 / 0.7588 | 0.9948 / 0.9935 |
| 修正後 | MVTec-4K | PaDiM | 0.9262 / 0.9262 | 0.9922 / 0.9922 |
| 修正後 | MVTec-4K | PatchCore | 0.8746 / 0.8746 | 0.9948 / 0.9947 |
| 修正前（Issue #9 の実行） | MVTec AD（900px） | PaDiM | 0.9963 / 0.9975 | 0.9768 / 0.9749 |
| 修正後 | MVTec AD（900px） | PaDiM | 0.9963 / 0.9963 | 0.9768 / 0.9769 |

修正後は image AUROC が engine と raw で一致しました。pixel AUROC の小さな差（1e-4 程度）は、マスクの扱い（raw は anomaly_map サイズへ最近傍リサイズ）の違いによるものと考えていますが、詳細は未確認です。

**影響範囲:** Issue #12 より前の `raw_score_metrics` とレイテンシ（`preprocess_ms` などを含む）は antialias 無効の条件です。MVTec AD（900px→256px）では差は小さい（上表）ものの、厳密には engine 指標と別条件です。過去結果の再計測はしていません。

## 4. CPU 実行結果（修正後、256px、8 スレッド、各 1 回）

```bash
python scripts/run_anomalib.py --dataset mvtec_4k --dataset-root ../datasets/MVTec-4K-view \
  --category bottle --model padim --device cpu --cpu-threads 8 --image-size 256 \
  --output-dir ../results/issue12/fix_padim_256
```

| モデル | train 工程 s | test 工程 s | Image AUROC | Pixel AUROC | decode P50 ms | preprocess P50 ms | model P50 / P95 ms | e2e P50 / P95 ms | 学習時ピーク RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| PaDiM | 102.98 | 86.41 | 0.9262 | 0.9922 | 482.0 | 16.8 | 40.4 / 134.5 | 542.9 / 717.3 | 1647.3 |
| PatchCore | 514.21 | 82.98 | 0.8746 | 0.9948 | 376.2 | 13.1 | 94.4 / 167.3 | 501.6 / 583.3 | 2411.1 |

レイテンシは test 132 枚のうちウォームアップ 5 枚を除く 127 枚の値です。

観察:

- 1 枚あたりの時間は 4K PNG のデコード（`read_image`）が大半を占め、モデル本体は e2e P50 の 7〜19% でした。MVTec AD（900px）の PaDiM では decode P50 18.1ms、e2e P50 51.9ms です。
- antialias 有効の縮小は 4096px 入力で P50 13〜17ms かかりました（修正前の antialias 無効では約 1ms）。
- 同じ条件の PatchCore の train 工程は、修正前の実行で 226.56 秒、修正後で 514.21 秒でした。学習経路は修正の影響を受けないため、原因は実行環境のばらつきと考えていますが、未確認です。PaDiM の修正後 e2e も修正前より遅く、decode 自体が 401.7→482.0ms に増えているため、同様にばらつきを含みます。

## 未検証事項

- bottle 以外の 6 カテゴリ（capsule, grid, hazelnut, screw, transistor, wood）の取得・整合検証・実行。論文表と HF 配布物の件数一致も bottle でしか確認していません。
- 4096px のネイティブ入力（PaDiM/PatchCore はメモリ不足の可能性）。タイル分割は Issue #13 で扱います。
- 繰り返し実行によるばらつき（特に PatchCore 学習時間の 2 倍差）。
- EfficientAD（Issue #10）の engine / raw の差が前処理の違いで説明できるかどうか（再計測していません）。
- 欠陥サイズ別（tiny / large）の指標。Issue #14 で扱います。
- StableSR（S-Lab License 1.0）の条件が生成画像にどう適用されるか。
