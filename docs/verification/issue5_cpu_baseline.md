# Issue #5 CPUベースライン検証記録（PaDiM / PatchCore）

## 1. 目的

Anomalib 2.7.0 をCPUのみで実行し、PaDiM と PatchCore の学習・テストが完走することを確認する。
数値はすべて実測値であり、推測値は記載しない。

## 2. 実行環境

| 項目 | 値 |
| --- | --- |
| 実行日時（UTC） | 2026-10-31 06:51〜07:11 |
| OS | Debian GNU/Linux 13 (trixie), Linux 6.12 x86_64（仮想マシン） |
| CPU | Intel(R) Xeon(R) Processor（型番非公開の仮想CPU）, 1ソケット8コア, SMTなし |
| CPU論理コア数 | 8（`torch.get_num_threads()` = 8） |
| メモリ | 総量 15.6GiB、他プロセスと共有のため実行時の空きは約2〜6GiB、スワップなし |
| GPU | なし（CPUのみ） |
| Python | 3.13.5（venv） |
| 実行場所 | 一時的なクラウドVM（Grokの実行環境）。他の計算機での速度を保証しない |

補足:

- `report.json` の `host`、`cpu_logical_cores`、`versions` に実行時情報が保存される。
- Windows は未検証。本記録は Linux CPU 環境を対象とする。

## 3. 依存バージョン

| パッケージ | バージョン |
| --- | --- |
| Python | 3.13.5 |
| torch | 2.15.0+cpu（`--index-url https://download.pytorch.org/whl/cpu`） |
| torchvision | 0.29.2+cpu |
| anomalib | 2.7.0 |
| lightning | 2.6.6 |
| timm | 1.0.30 |
| numpy | 2.5.4 |

補足:

- 上記は `report.json` の `versions` にも記録される。
- 未インストールのパッケージがある場合、`report.json` では `null` になる。

## 4. データセット配置手順

MVTec AD のデータセットはリポジトリにコミットしない。配布元の利用条件に従って取得する。
計測対象にはデータセットのダウンロード・展開時間を含めないため、実行前に配置を完了させる。

### 4.1 カテゴリ単位で取得する

検証では単一カテゴリ（例: `bottle`）のみを取得する。
配布元からカテゴリ単位のアーカイブを取得し、展開後にアーカイブを削除する。

```bash
# 例: 取得したカテゴリ単位のアーカイブを datasets/MVTecAD 配下へ展開する
mkdir -p datasets/MVTecAD
tar -xf <取得したカテゴリのアーカイブ> -C datasets/MVTecAD
rm <取得したカテゴリのアーカイブ>
```

アーカイブ内のトップディレクトリが `datasets/MVTecAD/<category>` になるように配置する。
配布アーカイブの形式・展開後の階層は配布元の手順に従う。

### 4.2 期待する配置例

```text
datasets/MVTecAD/
└── bottle/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   ├── broken_large/
    │   ├── broken_small/
    │   └── contamination/
    └── ground_truth/
        ├── broken_large/
        ├── broken_small/
        └── contamination/
```

MVTec 4K を使う場合は `datasets/MVTec-4K/<category>` に同様に配置する。

## 5. 重み取得手順

backbone 重みの初回ダウンロードを計測から除外するため、事前に `--prefetch-weights` を実行する。
`--prefetch-weights` はモデルを生成して backbone 重みを取得するだけで終了し、学習・テストは行わない。

```bash
python scripts/run_anomalib.py --model padim --prefetch-weights
python scripts/run_anomalib.py --model patchcore --prefetch-weights
```

`--backbone` を変更する場合も、計測前に同様に実行する。

```bash
python scripts/run_anomalib.py --model padim --backbone <backbone名> --prefetch-weights
```

補足:

- 初回はネットワーク接続が必要。
- backbone 重みは timm 経由で Hugging Face Hub から取得される。
- キャッシュ先は環境変数などに依存するため、必要に応じて実行環境のキャッシュ設定を確認する。
- `--prefetch-weights` はデータセット配置チェックを行わない。

## 6. 再現コマンド

### 6.1 PaDiM

```bash
python scripts/run_anomalib.py \
  --dataset mvtec_ad \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --model padim \
  --device cpu \
  --image-size 256 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --num-workers 0 \
  --cpu-threads 8 \
  --output-dir results/issue5
```

### 6.2 PatchCore

```bash
python scripts/run_anomalib.py \
  --dataset mvtec_ad \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --model patchcore \
  --coreset-sampling-ratio 0.1 \
  --device cpu \
  --image-size 256 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --num-workers 0 \
  --cpu-threads 8 \
  --output-dir results/issue5
```

結果は `results/issue5/<run_id>/report.json` に保存される。

## 7. 実測結果

以下はすべて上記環境での実測値（各1回の実行）。`--cpu-threads` は未指定（PyTorch既定 = 8スレッド）、画像サイズ256、バッチサイズ1、`num_workers=0`、seed 42、`val_split_mode=FROM_TEST`（テスト集合の20%を検証に使用し、残り80%で評価）。
PatchCore の backbone は既定の `wide_resnet50_2`、コアセット比率 0.1。PaDiM は既定の `resnet18`。

| カテゴリ | モデル | 学習工程 [s] | テスト工程 [s] | 学習時ピークRSS [MiB]（サンプリング） | テスト時ピークRSS [MiB]（サンプリング） | Image AUROC | Pixel AUROC | report.json |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bottle | PaDiM | 14.400 | 4.592 | 1831.5 | 980.4 | 0.9963 | 0.9768 | [`benchmarks/issue5/bottle_padim.json`](../../benchmarks/issue5/bottle_padim.json) |
| bottle | PatchCore | 308.478 | 25.308 | 3124.6 | 1472.5 | 1.0000 | 0.9857 | [`benchmarks/issue5/bottle_patchcore.json`](../../benchmarks/issue5/bottle_patchcore.json) |
| grid | PaDiM | 14.458 | 4.502 | 1994.6 | 877.9 | 0.7973 | 0.9068 | [`benchmarks/issue5/grid_padim.json`](../../benchmarks/issue5/grid_padim.json) |

注意:

- 「学習工程」「テスト工程」は `engine.fit` / `engine.test` 全体の経過時間（データ読み込み・評価指標計算・チェックポイント保存を含む）で、画像1枚の推論時間ではない。
- PatchCore の学習時間の大半はコアセットの貪欲選択（`Selecting Coreset Indices`、約21,400点）であった。
- 同一VM上で他プロセスが動作しており、時間は環境負荷の影響を受け得る。繰り返し計測はしていない（#6 以降で対応）。
- 評価は検証用に分けた20%を除くテスト集合で行っており、論文の全テスト集合での値とは直接比較できない（#7 で扱う）。
- チェックポイントは `results/<run_id>/<Model>/MVTecAD/<category>/` に保存され、PaDiM bottle で約169MB、PatchCore bottle で約221MBのディスクを使用した。

実行ログ:

- 実行ログはデータセットのパス等を含むためコミットしていない。`report.json` に実行条件と結果を保存している。
- 例外時の `report.json` の例（存在しない backbone を指定）: [`benchmarks/issue5/failed_example_invalid_backbone.json`](../../benchmarks/issue5/failed_example_invalid_backbone.json)

## 8. 正常終了・異常終了時の挙動

- 正常終了時: `report.json` の `status` は `"ok"`、終了コードは `0`。
- 例外発生時: `report.json` の `status` は `"failed"`、`error_type` と `error_message` を保存し、終了コードは非0。
- `--prefetch-weights` 実行時も `report.json` を出力し、成功時は `status` が `"ok"` になる。

## 9. 既知の制限

- 本記録は Linux CPU 環境を対象とする。Windows は未検証。
- GPU 実行は未検証。
- Anomalib 2.7.0 を対象とする。他のバージョンは未検証。
- Anomalib Engine は内部で `ModelCheckpoint` を追加するため、`enable_checkpointing=False` を指定すると `MisconfigurationException` になる。本スクリプトでは `enable_checkpointing` を指定しない。チェックポイントが出力ディレクトリ配下に保存され得る。
- 画像全体を `--image-size` にリサイズして処理する。タイル分割やスライディングウィンドウは行わない。
- 画像1枚当たりの推論時間は計測しない。
- RSS は約 0.1 秒間隔のサンプリング値であり、瞬間的なピークを取りこぼす可能性がある。
- PatchCore の `--coreset-sampling-ratio` はメモリ削減のためのパラメータだが、実際の削減量はデータや backbone に依存する。本記録では未検証。
- 初回の重みダウンロードとデータセット配置は計測対象外。`--prefetch-weights` と手動配置により事前に完了させる。
- MVTec AD のデータセットおよび学習済み重みはリポジトリにコミットしない。
- `--backbone` にモデル既定以外を指定した場合の精度・メモリは未検証。
