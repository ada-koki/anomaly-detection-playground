# Issue #9 検証記録: PaDiM / PatchCore 共通条件比較

実行日: 2026-10-31（JST）。数値はすべて実測（各 run の `report.json` を `scripts/aggregate_results.py` で集計）。推定値は含みません。

## 実行環境・条件

- CPU: x86_64 論理 8 コア、RAM 16 GB、GPU なし（共有の Linux VM）
- Python 3.13.5 / torch 2.15.0+cpu / torchvision 0.29.2+cpu / anomalib 2.7.0 / lightning 2.6.6 / timm 1.0.30
- プロトコル: `configs/protocols/mvtec_ad_v1.json`（from_test 0.2、seed 固定）
- 設定: `configs/experiments/mvtec_ad_cpu_baseline.json`（PaDiM/PatchCore × bottle, grid, toothbrush, metal_nut × 256px × 8 スレッド × 2 反復 = 16 run）、`configs/experiments/mvtec_ad_resolution_sweep.json`（PaDiM bottle × 192/256/320px × 2 反復 = 6 run）
- 実行: `scripts/run_matrix.py --resume --timeout-s 3600`（1 run = 1 サブプロセスで逐次実行、各 run 後に checkpoint を削除）
- 結果: 22/22 run が status=ok。失敗・タイムアウトはありません。

集計結果（runs.csv / summary.csv / report.md）:

- `benchmarks/issue9/mvtec_ad_cpu_baseline/`
- `benchmarks/issue9/mvtec_ad_resolution_sweep/`

## 結果の要約（256px・8 スレッド、2 反復の平均 ± 標準偏差）

| カテゴリ | モデル | Image AUROC (engine / raw) | Pixel AUROC (engine / raw) | 学習時間 s | e2e P50 ms | e2e P95 ms | ピーク RSS MiB |
|---|---|---|---|---:|---:|---:|---:|
| bottle | PaDiM | 0.9963 / 0.9975 | 0.9768 / 0.9749 | 17.34 ± 6.90 | 44.52 ± 0.34 | 384.40 ± 409.62 | 1791.0 |
| bottle | PatchCore | 1.0000 / 1.0000 | 0.9857 / 0.9839 | 298.64 ± 75.44 | 177.66 ± 38.38 | 684.65 ± 599.21 | 3205.0 |
| grid | PaDiM | 0.7973 / 0.7737 | 0.9068 / 0.8806 | 14.10 ± 2.20 | 41.20 ± 3.21 | 76.95 ± 45.47 | 1714.4 |
| grid | PatchCore | 0.9885 / 0.9808 | 0.9820 / 0.9793 | 1038.00 ± 575.95 | 139.33 ± 17.56 | 663.74 ± 651.14 | 3915.6 |
| toothbrush | PaDiM | 0.7958 / 0.7833 | 0.9857 / 0.9853 | 5.89 ± 0.46 | 57.69 ± 0.35 | 86.58 ± 26.81 | 1420.4 |
| toothbrush | PatchCore | 0.9292 / 0.9250 | 0.9899 / 0.9898 | 17.97 ± 5.52 | 127.66 ± 5.01 | 328.01 ± 214.75 | 1418.1 |
| metal_nut | PaDiM | 0.9533 / 0.9644 | 0.9404 / 0.9282 | 13.25 ± 0.90 | 44.15 ± 0.30 | 93.90 ± 2.48 | 1769.9 |
| metal_nut | PatchCore | 0.9963 / 0.9911 | 0.9864 / 0.9855 | 640.96 ± 402.31 | 129.07 ± 6.39 | 221.04 ± 93.12 | 3337.7 |

解像度スイープ（PaDiM / bottle / 8 スレッド）:

| 入力 | Image AUROC (engine / raw) | Pixel AUROC (engine / raw) | 学習時間 s | e2e P50 ms | e2e P95 ms | ピーク RSS MiB |
|---|---|---|---:|---:|---:|---:|
| 192 | 0.9951 / 0.9926 | 0.9722 / 0.9706 | 8.91 ± 0.40 | 37.63 ± 0.88 | 54.84 ± 1.05 | 1303.1 |
| 256 | 0.9963 / 0.9975 | 0.9768 / 0.9749 | 13.56 ± 0.72 | 61.02 ± 13.35 | 109.66 ± 61.75 | 1755.5 |
| 320 | 0.9890 / 0.9902 | 0.9783 / 0.9771 | 36.56 ± 15.90 | 76.89 ± 22.03 | 179.39 ± 145.67 | 2417.7 |

## 観察

- 精度指標（AUROC）は 2 反復で完全に一致しました（標準偏差 0）。データ分割・seed の固定（#7）が効いています。
- 時間指標は反復間のばらつきが大きく、特に PatchCore の学習時間（grid: 697 s と 1492 s、metal_nut: 441 s と 998 s）と、e2e P95 が大きくぶれています。P50 は比較的安定しています。
- PatchCore はこの CPU 環境で学習時間が PaDiM の数十倍、e2e P50 が約 2〜4 倍、ピーク RSS が約 1.8〜2.3 倍（toothbrush は同程度）でした。精度は 4 カテゴリすべてで PatchCore が上回りました。
- PaDiM の 256px と 320px の比較では、Pixel AUROC はわずかに上がる一方 Image AUROC は下がり、学習時間・レイテンシ・RSS は増えました。

## 計測上の注意（未統制の要因）

- 実行環境は他のプロセスと共有される VM で、CPU 割り当てを専有していません。マトリクス実行中にはデータセットのダウンロード（PaDiM bottle r1 と重複）、短い単体テスト、API 呼び出しなど軽い作業が並行していました。時間指標、特に P95 と PatchCore の学習時間のばらつきには、これらの影響が含まれている可能性があります。原因は切り分けていません。
- 反復 2 回のみのため、標準偏差は参考値です。時間指標の比較には P50 と min を優先してください。
- 256px の PaDiM bottle は baseline と resolution_sweep で別々に実行しているため、時間指標の値が異なります（精度は一致）。
- 同一条件の再計測（CPU を専有できる環境、反復回数の増加）は未実施です。
