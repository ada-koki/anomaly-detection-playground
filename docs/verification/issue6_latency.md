# Issue #6 検証記録: 画像1枚あたり推論レイテンシ計測

すべて実機計測値です（推定値は含みません）。

## 計測環境

- CPU: Intel Xeon 8 vCPU（1ソケット・8コア・SMTなし）、RAM 15.6 GiB、GPU なし
- OS: Debian 13 (Linux 6.12 x86_64)、Python 3.13.5
- torch 2.15.0+cpu, torchvision 0.29.2+cpu, anomalib 2.7.0, lightning 2.6.6
- 注意: 計測中、同じマシン上でデータセットのダウンロード・展開が並行して動いていました。完全に静かな環境の値ではありません。

## 条件

- データ: MVTec AD bottle、入力 256x256、バッチサイズ1、`num_workers=0`
- 分割: 公式 test から 20% を検証用に抽出（seed 固定、Anomalib `from_test`）。残り 62 枚で計測
- ウォームアップ 5 枚（集計から除外）、`--latency-repeats` 回の繰り返し（PaDiM 3回 = 186 サンプル、PatchCore 2回 = 124 サンプル）
- 工程: decode（`read_image`）/ preprocess（`pre_processor`）/ model（`model.model`）/ postprocess（`post_processor`）/ end_to_end（合計）
- `torch.inference_mode()`、`torch.set_num_threads(N)`

実行コマンド例:

```bash
python scripts/run_anomalib.py --dataset-root datasets/MVTecAD --category bottle \
  --model padim --cpu-threads 4 --latency-repeats 3 --output-dir results/i6
```

## 結果

| model | threads | e2e P50 [ms] | e2e P95 [ms] | model P50 [ms] | model P95 [ms] | decode P50 [ms] | latency RSS [MiB] | Image AUROC | Pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PaDiM | 1 | 86.36 | 100.27 | 66.03 | 78.33 | 18.31 | 802.3 | 0.9963 | 0.9768 |
| PaDiM | 2 | 61.53 | 66.00 | 42.86 | 46.86 | 17.25 | 995.2 | 0.9963 | 0.9768 |
| PaDiM | 4 | 51.01 | 55.20 | 32.97 | 36.73 | 17.08 | 803.7 | 0.9963 | 0.9768 |
| PaDiM | 8 | 46.04 | 52.77 | 27.70 | 32.86 | 17.18 | 801.9 | 0.9963 | 0.9768 |
| PatchCore | 8 | 151.02 | 176.81 | 130.91 | 156.79 | 18.16 | 1049.6 | 1.0000 | 0.9858 |

- 学習時間・学習時 RSS など全項目: `benchmarks/issue6/summary.csv` と同ディレクトリの JSON
- PatchCore 学習（coreset 選択含む）: 286.45 s、学習時 RSS 3150.1 MiB、プロセス生涯最大 RSS 3203.2 MiB

## 所見（計測値から言えること）

- PaDiM の model 工程はスレッド数を増やすと短くなりました（1→8 スレッドで P50 66.0→27.7 ms）。4→8 の改善幅は小さいです。
- decode（PNG 読み込み）は約 17〜18 ms でスレッド数にほぼ依存せず、8 スレッドでは end_to_end の約 4 割を占めます。
- 同じ seed・同じ分割では、スレッド数を変えても AUROC は同じ値でした。
- 繰り返し 3 回でも P50 と P95 の差は小さく（PaDiM 8 スレッドで約 7 ms）、外れ値は限定的でした。

## 未検証・制限

- 学習（fit）とテスト（test）の段階時間は各条件1回のみの計測です。並行処理の影響もあるため、スレッド間比較には使わないでください。
- RSS は psutil による 0.05 秒間隔のサンプリング値です。短いピークは取り逃す可能性があります。
- 他カテゴリ、解像度を変えた比較、PatchCore の 8 スレッド以外は未計測です（Issue #9 の比較基盤で扱う予定）。
