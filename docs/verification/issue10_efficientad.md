# Issue #10 検証記録: EfficientAD の CPU 学習・CPU 推論（短縮条件）

この記録は実測値のみを掲載します。推定値には「推定」と明記します。

## 条件

| 項目 | 値 |
| --- | --- |
| データ | MVTec AD `bottle`（train 209 / val 16 / test 67、val は公式 test から seed 42・label_aware で抽出） |
| モデル | EfficientAD `small`（Anomalib 2.7.0, torch 2.15.0+cpu） |
| 学習 | `--max-epochs 1`（209 ステップ, train_batch_size 1）。論文既定は 70,000 ステップ |
| 解像度 | 256×256（タイル分割なし） |
| スレッド | 8（8 論理コアの共有 VM, RAM 15 GiB） |
| 外部依存 | ImageNette (`imagenette2`, 13,394 枚)、Anomalib 配布の教師重み `pretrained_teacher_small.pth` |
| 反復 | 学習 2 回 + e1_r1 のチェックポイントを使った CPU 推論 1 回 |

再現コマンド（データ・重みは各自で取得。コミットしません）:

```bash
export XDG_CACHE_HOME="$PWD/../cache"   # 教師重みの保存先を作業ディレクトリ配下に
python scripts/run_anomalib.py --model efficient_ad \
  --dataset-root ../datasets/MVTecAD --category bottle \
  --imagenet-dir ../datasets/imagenette2 --max-epochs 1 --cpu-threads 8 \
  --output-dir ../results/issue10/e1_r1

# 学習済みチェックポイントで CPU 推論のみ
python scripts/run_anomalib.py --model efficient_ad \
  --dataset-root ../datasets/MVTecAD --category bottle \
  --imagenet-dir ../datasets/imagenette2 --cpu-threads 8 \
  --checkpoint <e1_r1>/EfficientAd/MVTecAD/bottle/v0/weights/lightning/model.ckpt \
  --output-dir ../results/issue10/ckpt_infer
```

## 結果

元データ: [`benchmarks/issue10/efficientad_cpu_short/runs.csv`](../../benchmarks/issue10/efficientad_cpu_short/runs.csv)

| run | training_mode | train s | s/step | validate s | Image AUROC (engine / raw) | Pixel AUROC (engine / raw) | e2e P50 / P95 ms | model P50 / P95 ms | peak RSS MiB |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| e1_r1 | cpu_train_cpu_infer | 206.88 | 0.990 | 5.22 | 0.9596 / 0.9694 | 0.7610 / 0.7641 | 176.36 / 226.85 | 157.11 / 200.61 | 1087 |
| e1_r2 | cpu_train_cpu_infer | 183.84 | 0.880 | 4.55 | 0.9596 / 0.9694 | 0.7610 / 0.7641 | 174.98 / 229.56 | 155.85 / 210.75 | 1072 |
| ckpt_infer（e1_r1 の ckpt） | external_checkpoint_cpu_infer | – | – | – | 0.9596 / 0.9694 | 0.7610 / 0.7641 | 215.18 / 267.16 | 194.85 / 244.10 | 964 |

- AUROC は 2 回の学習と ckpt 推論で全桁一致しました（seed 固定）。
- `--checkpoint` による CPU 推論は正常終了し、学習時と同じ指標を再現しました。ckpt 内の統計（quantile 等）を使い、validate は呼びません。
- `s/step` は `train_stage_seconds / trained_steps` の計算値です。`engine.fit` には教師特徴の平均・分散計算（学習画像 209 枚の順伝播）が含まれるため、純粋な 1 ステップの時間ではありません。
- 学習時間は 2 回で約 12% 異なりました。共有 VM のため他プロセスの負荷は統制していません。ckpt 推論のレイテンシが学習ランより約 2 割大きい点も、同じ要因の可能性があり、原因は未検証です。
- engine の Image AUROC（正規化後スコア）と raw（post_processor を通さない生スコアを sklearn で計算）が 0.9596 と 0.9694 で異なります。原因は未検証です（正規化後のクリップで同点が生じる可能性があります）。
  - 追記（Issue #12）: 当時の生スコア評価は `PreProcessor.forward()`（antialias 無効の `export_transform`）を使っており、`engine.test`（antialias 有効の `transform`）と前処理が異なっていました。この差が原因の一部である可能性がありますが、EfficientAD での再計測は未実施です（`docs/verification/issue12_mvtec4k.md`）。

## 解釈上の注意

- 1 エポック（209 ステップ）は論文既定の 70,000 ステップの約 0.3% であり、この AUROC は EfficientAD の本来の性能を示しません。PaDiM / PatchCore（Issue #9）との精度比較には使わないでください。
- **推定**: 本条件の s/step（0.88〜0.99 s）を 70,000 ステップにそのまま掛けると約 17〜19 時間です。教師統計計算の初期コストや負荷変動を含む粗い外挿で、実測ではありません。
- 推論レイテンシ（e2e P50 約 175 ms）は学習ステップ数に依存しないため、CPU 推論コストの目安としては有効です。同条件の Issue #9 の PaDiM / PatchCore とは測定日時・負荷が異なります。

## 未検証

- 70,000 ステップ学習、`medium` モデル、bottle 以外のカテゴリ、スレッド数・解像度の変化。
- GPU 学習（実機での確認なし）。
