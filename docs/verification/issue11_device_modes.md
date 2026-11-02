# Issue #11 検証記録: 学習/推論デバイスの記録と二段実行（CPU 実機）

この記録は実測値のみを掲載します。検証環境に GPU が無いため、GPU 経路は実機で確認していません。

## 条件

| 項目 | 値 |
| --- | --- |
| データ | MVTec AD `bottle`（protocol `mvtec_ad_v1`、test 67 枚） |
| モデル | PaDiM（Anomalib 2.7.0, torch 2.15.0+cpu） |
| 解像度 / スレッド | 256×256 / 8（8 論理コアの共有 VM、GPU なし） |

```bash
# 1 段目: 学習してチェックポイントと来歴サイドカーを保存
python scripts/run_anomalib.py --model padim --dataset-root ../datasets/MVTecAD \
  --category bottle --cpu-threads 8 --device cpu \
  --save-checkpoint ../results/issue11/ckpt --output-dir ../results/issue11/stage1

# 2 段目: 保存したチェックポイントで推論のみ
python scripts/run_anomalib.py --model padim --dataset-root ../datasets/MVTecAD \
  --category bottle --cpu-threads 8 --device cpu \
  --checkpoint ../results/issue11/ckpt/model.ckpt --output-dir ../results/issue11/stage2
```

GPU 環境では 1 段目を `--device cuda` にすると `gpu_train_cpu_infer` として記録される設計です（[docs/device_modes.md](../device_modes.md)）。

## 結果

元データ: [`benchmarks/issue11/padim_two_stage_cpu/runs.csv`](../../benchmarks/issue11/padim_two_stage_cpu/runs.csv)

| stage | execution_mode | training_mode | 観測デバイス（学習 / 推論） | train s | Image AUROC (engine / raw) | Pixel AUROC (engine / raw) | e2e P50 / P95 ms | peak RSS MiB |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: |
| stage1 | cpu_only | cpu_train_cpu_infer | cpu / cpu | 15.09 | 0.9963 / 0.9975 | 0.9768 / 0.9749 | 48.25 / 57.92 | 1634 |
| stage2 | cpu_only | external_checkpoint_cpu_infer | –（来歴: cpu）/ cpu | – | 0.9963 / 0.9975 | 0.9768 / 0.9749 | 46.33 / 55.66 | 1102 |

- 1 段目は `model.ckpt`（約 169 MiB、リポジトリ外）と `model.ckpt.provenance.json` を保存しました。サイドカーには学習デバイス（要求 / 観測とも cpu）、モデル、カテゴリ、protocol の SHA-256、torch / Anomalib のバージョン、チェックポイントの SHA-256 が入ります。
- 2 段目はサイドカーを読み、チェックポイントの SHA-256 一致（`sha256_matches: true`）を確認しました。来歴の学習デバイス cpu と推論デバイス cpu から `execution_mode` は `cpu_only` になりました。
- 2 段の engine / raw AUROC は全桁一致しました。チェックポイント経由の推論で精度が変わらないことを確認しました。
- 推論のみの 2 段目はプロセス最大 RSS が約 530 MiB 小さくなりました（学習工程が無いため）。レイテンシ差（P50 で約 2 ms）は共有 VM の負荷変動の範囲と考え、差の有無は結論づけません。

## 異常系

- GPU の無い環境で `--device cuda` を指定すると、`CUDAを利用できません。--device cpu を指定してください。` でエラー終了（終了コード 1、report は `status: failed` で保存）しました。黙って CPU で続行しないことを確認しました。
- 観測デバイスと要求の不一致を検出するロジック（`adbench.devices.detect_device_mismatch`）は単体テストで確認しています。不一致時に終了コード 3 で終える CLI 経路（`--allow-device-mismatch` で警告のみ）は、実機でも CLI テストでも再現していません。

## 未検証

- GPU 学習（`gpu_train_cpu_infer`）、GPU 推論（`gpu_reference`、`cpu_train_gpu_infer`）の実機動作。
- 来歴サイドカーの無い外部チェックポイント（`external_checkpoint_unknown_train_device`）の実データでの動作（単体テストのみ）。
- EfficientAD / PatchCore での `--save-checkpoint`。EfficientAD の `--checkpoint` 推論は Issue #10 で確認済みです。
