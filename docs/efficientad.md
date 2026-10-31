# EfficientAD の実行手順と前提条件

このドキュメントは Anomalib 2.7.0 の EfficientAD を CPU 優先ベンチマークで扱うための
前提条件・取得手順・実行例をまとめたものです。実測値（学習時間・推論時間・精度など）は
ここには記載せず、別途の検証記録に保存します。

## API と前提条件

- Anomalib 2.7.0 の `EfficientAd` を `--model efficient_ad` で利用します。
- 学習時は `train_batch_size=1` が必須です。Anomalib の `EfficientAd` は
  `on_train_start` で `train_batch_size != 1` の場合に `ValueError` を送出します。
  本リポジトリの CLI でも `train_batch_size != 1` を拒否します。
- `pre_processor` に `Normalize` を含めないでください。EfficientAD は教師モデル側で
  正規化を扱うため、`Normalize` があると学習開始時に `ValueError` になります。
- 学習には ImageNette（`imagenette2`）を `--imagenet-dir` に配置する必要があります。
  本 CLI は自動ダウンロードを計測に含めないため、実際の学習前に配置を確認し、
  見つからない場合は `parser.error` で終了します。
  ただし `--prefetch-weights`、`--manifest-only`、`--checkpoint` では
  学習用の ImageNette 配置チェックを省略します。
- 教師重みは Anomalib の配布物です。初回は `--prefetch-weights` で取得できます。
  取得先は Anomalib のキャッシュディレクトリ配下です。
- EfficientAD の validation では、validation dataloader の **good 画像のみ** から
  map 正規化用の quantile（`qa_st`, `qb_st`, `qa_ae`, `qb_ae`）を計算します。
  既定プロトコル `mvtec_ad_v1` の `from_test` では、`bottle` の validation は
  16 枚で、そのうち good は 4 枚です。validation が一度も走らないと quantile が
  未設定のまま test されるため、本 CLI は EfficientAD では fit 後に
  `engine.validate` を明示的に 1 回実行します。
- `--max-steps` を指定した場合も、fit 後の `engine.validate` により
  quantile と PostProcessor 統計を確保します。`validate_stage_seconds` を
  `train_stage_seconds` とは別に記録します。
- 論文既定の学習は 70,000 ステップ（GPU 前提）です。本リポジトリの
  `--max-epochs` / `--max-steps` による短縮学習は論文条件とは **別条件** です。
  精度を同等として比較しないでください。

## 外部データとライセンスの確認事項

### ImageNette (imagenette2)

- 配布元 URL: `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz`
- ImageNette は ImageNet のサブセットです。ImageNet の利用条件
  （研究用途・非商用条件などを含む）に従う必要があります。
- 利用前に配布元と ImageNet のライセンス・利用規約を確認してください。

### 教師重み

- Anomalib の GitHub Releases で配布される
  `efficientad_pretrained_weights.zip` を利用します。
- 教師重みは ImageNet 由来の蒸留を含みます。配布元で利用条件・ライセンスを
  確認してください。
- 本リポジトリは重みファイルをコミットしません。取得物は `datasets/` や
  キャッシュ領域など、Git 管理外に置いてください。

## ImageNette の取得手順

リポジトリルートで実行する例です。ダウンロード用の tgz は展開後に削除します。

```bash
mkdir -p datasets
curl -L -o /tmp/imagenette2.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xzf /tmp/imagenette2.tgz -C datasets
rm /tmp/imagenette2.tgz
```

展開後、`datasets/imagenette2/` が `--imagenet-dir` の既定値になります。

## 教師重みの取得

教師重みだけを取得する場合は `--prefetch-weights` を使います。
`EfficientAd` を生成して `prepare_pretrained_model()` を呼びます。

```bash
python scripts/run_anomalib.py \
  --model efficient_ad \
  --prefetch-weights
```

## CPU 学習の実行例

以下は条件を短縮して CPU 学習・CPU 推論の可否を確認する例です。
`--max-steps` の値は目的に合わせて調整してください。ここに示す値は
ベンチマーク結果ではなく、実行例のための設定値です。

```bash
python scripts/run_anomalib.py \
  --model efficient_ad \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --imagenet-dir datasets/imagenette2 \
  --max-epochs 1 \
  --max-steps 100
```

短縮学習の結果は論文既定の 70,000 ステップ条件とは別条件です。
`max_epochs`、`max_steps`、`trained_steps`、`training_mode` を確認し、
条件が異なる実行を同一の比較表で同等扱いしないでください。

## 外部学習済みチェックポイントによる CPU 推論

`--checkpoint` を指定すると、学習をスキップしてチェックポイントを CPU に読み込み、
test・レイテンシ計測・生スコア評価のみを行います。

```bash
python scripts/run_anomalib.py \
  --model efficient_ad \
  --checkpoint path/to/model.ckpt \
  --dataset-root datasets/MVTecAD \
  --category bottle
```

- `training_mode` は `external_checkpoint_cpu_infer` になります。
- `train_stage_seconds` は `null` になります。
- `checkpoint_sha256` を記録します。
- `--checkpoint` 時は追加の `validate` を呼びません。チェックポイントに保存された
  PostProcessor 統計を使用します。したがって、チェックポイント作成時の
  検証集合・前処理条件が異なる場合、閾値依存指標を単純比較できません。

## 比較時の注意

- `training_mode` が異なる行（`cpu_train_cpu_infer` と
  `external_checkpoint_cpu_infer`）は同等比較しないでください。
- `trained_steps` が異なる行は同等比較しないでください。
- EfficientAD の短縮学習は論文既定の 70,000 ステップ条件とは別条件です。
- EfficientAD は ImageNette と教師重みという外部データ依存を持ちます。
  PaDiM/PatchCore と同じ入力データ分割・同じ推論計測を使っていても、
  学習条件は同一ではありません。
- `model_specific` の `train_seconds_per_step` は
  `train_stage_seconds / trained_steps` から計算した値です。
  ステップ数やモデルサイズが異なる実行と同等比較しないでください。
