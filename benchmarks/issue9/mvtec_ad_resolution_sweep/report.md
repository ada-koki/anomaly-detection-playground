# PaDiM / PatchCore 比較レポート

このレポートの数値は各 run の report.json からのみ取得しています。欠損は `—` で表し、0 などの仮の値で補いません。

## 1. 実行条件

- 設定名: mvtec_ad_resolution_sweep
- プロトコル: configs/protocols/mvtec_ad_v1.json
- プロトコル名: mvtec_ad_v1
- 環境:
  - python: 3.13.5
  - torch: 2.15.0+cpu
  - torchvision: 0.29.2+cpu
  - anomalib: 2.7.0
  - lightning: 2.6.6
  - timm: 1.0.30
  - numpy: 2.5.4
  - os: Linux-6.12.94+-x86_64-with-glibc2.41
  - machine: x86_64
  - cpu_logical_cores: 8

## 2. カテゴリ別比較

### bottle

| 条件 | モデル | n_ok/n_runs | Image AUROC engine | Image AUROC raw | Pixel AUROC engine | Pixel AUROC raw | 学習時間 s | e2e P50 ms | e2e P95 ms | ピーク RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s192 / t8 | padim | 2/2 | 0.9951 ± 0.0000 | 0.9926 ± 0.0000 | 0.9722 ± 0.0000 | 0.9706 ± 0.0000 | 8.91 ± 0.40 | 37.63 ± 0.88 | 54.84 ± 1.05 | 1303.1 ± 11.0 |
| s256 / t8 | padim | 2/2 | 0.9963 ± 0.0000 | 0.9975 ± 0.0000 | 0.9768 ± 0.0000 | 0.9749 ± 0.0000 | 13.56 ± 0.72 | 61.02 ± 13.35 | 109.66 ± 61.75 | 1755.5 ± 1.4 |
| s320 / t8 | padim | 2/2 | 0.9890 ± 0.0000 | 0.9902 ± 0.0000 | 0.9783 ± 0.0000 | 0.9771 ± 0.0000 | 36.56 ± 15.90 | 76.89 ± 22.03 | 179.39 ± 145.67 | 2417.7 ± 97.8 |

## 3. 失敗・タイムアウト一覧

失敗・タイムアウトはありません。

## 4. 指標の意味

- engine 指標は val 由来の閾値・正規化を経た後の metrics（test_metrics）です。
- raw 指標は post_processor による正規化を通さない生スコアの順位ベース指標です。
- 学習時間（train_stage_seconds）は engine.fit 工程全体の所要時間です。
- レイテンシはバッチ 1 で計測し、ウォームアップ画像を集計から除外しています。
- RSS は psutil によるサンプリング最大値であり、短いピークを取り逃し得ます。
- repeat は同一 seed で実行しており、モデル条件の違いではなく計測ばらつきの評価です。
