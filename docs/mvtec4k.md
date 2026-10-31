# MVTec-4K の検証と利用上の注意

本ドキュメントは、MVTec-4K をこのリポジトリの CPU 優先ベンチマークで扱う前に
確認すべきデータ構造・ラベル・マスク・ライセンス条件をまとめたものである。
本リポジトリは MVTec-4K の画像・派生画像・重みを再配布せず、コミットしない。

## データの由来（確認済みの範囲）

- MVTec-4K は MVTec AD の画像を高解像度化した派生データである。
- 配布物には 7 カテゴリが含まれる: bottle, capsule, grid, hazelnut, screw,
  transistor, wood。
- 画像は 4096x4096 の高解像度画像として配布されている（bottle の全 311 枚で
  実データ確認済み。他カテゴリは未確認。`docs/verification/issue12_mvtec4k.md` 参照）。
- 論文によれば、画像は MVTec AD を StableSR で 4096x4096 へ超解像したもの
  （実際の 4K センサーで撮影した画像ではない）。
- tiny は元データの ground truth 欠陥領域を切り出して縮小し、ポアソン合成で
  正常画像へ貼り込んだ合成欠陥として説明されている。
- 出典は HF のデータカード（XimiaoZhang/MVTec-4K）と HiAD 論文
  （arXiv 2508.12931）である。論文表の枚数は本リポジトリでは「参考値」として
  扱い、実データとの差は warning として報告する。

### 論文表の参考枚数

以下は論文表に記載された参考値であり、このリポジトリで実測した値ではない。

| category | train | test_good | large | tiny |
|---|---:|---:|---:|---:|
| bottle | 146 | 50 | 47 | 68 |
| capsule | 179 | 63 | 109 | 88 |
| grid | 230 | 55 | 57 | 37 |
| hazelnut | 351 | 80 | 70 | 73 |
| screw | 260 | 84 | 48 | 55 |
| transistor | 183 | 87 | 40 | 34 |
| wood | 217 | 48 | 55 | 41 |

`large` は test のうち good と tiny 以外の異常画像を指す。test.jsonl の行数は
`test_good + large + tiny` に対応する想定である。

## レイアウト

```text
{category}/
  train/good/NNN.png
  test/{defect_name|good|tiny}/NNN.png
  ground_truth/{defect_name|tiny}/NNN_mask.png
  foreground/{train,test}/{label_name}/NNN.png
  train.jsonl
  test.jsonl
train_uni.jsonl
test_uni.jsonl
```

`train_uni.jsonl` と `test_uni.jsonl` は全カテゴリを連結したマニフェストである。
本リポジトリの検証はカテゴリ別の `train.jsonl` / `test.jsonl` を使う。

### jsonl の形式

train の good:

```json
{"filename": "bottle/train/good/000.png", "foreground": "bottle/foreground/train/good/000.png", "clsname": "bottle", "label": 0, "label_name": "good"}
```

test の異常:

```json
{"filename": "bottle/test/broken_small/000.png", "mask": "bottle/ground_truth/broken_small/000_mask.png", "foreground": "bottle/foreground/test/broken_small/000.png", "clsname": "bottle", "label": 1, "label_name": "broken_small"}
```

test の good:

```json
{"filename": "bottle/test/good/000.png", "foreground": "bottle/foreground/test/good/000.png", "clsname": "bottle", "label": 0, "label_name": "good"}
```

tiny:

```json
{"filename": "bottle/test/tiny/000.png", "mask": "bottle/ground_truth/tiny/000_mask.png", "foreground": "bottle/foreground/test/tiny/000.png", "clsname": "bottle", "label": 1, "label_name": "tiny"}
```

## Anomalib MVTecAD に生配置を渡してはいけない理由

Anomalib 2.7 の MVTecAD データモジュールは `root/category` 配下を
`glob("**/*")` で走査し、各ファイルパスの下位 3 要素を
`(split, label, name)` とみなして画像パスを再構成する。

MVTec-4K の生配置には `foreground/train/good/000.png` や
`foreground/test/<defect>/000.png` が含まれる。このため生配置をそのまま渡すと、
foreground 配下の画像が train/test の重複行として解釈される。異常行数とマスク数が
合わない、重複学習になる、といった問題が起きる。

### clean view の作成

`scripts/validate_mvtec4k.py --make-view` は、エラーが 0 のカテゴリについて
`train/`、`test/`、`ground_truth/` の必要ファイルだけをハードリンク（既定、`--link-mode`）で並べた
ビューを作る。foreground は含めない。同一ファイルシステム上のハードリンクなので
追加容量はほぼゼロ（bottle で約 60KB、ディレクトリ分のみ）。

symlink ではなくハードリンクにしているのは、Anomalib 2.7 の
`make_mvtec_ad_dataset` が `is_within_directory()` でパスを解決し、root 外を指す
symlink を除外するためである。symlink のビューでは `Found 0 images` になる
（実データで確認）。別ファイルシステムでハードリンクが作れない場合は
`--link-mode copy` を使う。

```bash
python scripts/validate_mvtec4k.py \
  --root datasets/MVTec-4K \
  --make-view /path/to/mvtec4k-view
```

このビューを `run_anomalib.py` の `--dataset-root` に渡す。

```bash
python scripts/run_anomalib.py \
  --dataset mvtec_4k \
  --dataset-root /path/to/mvtec4k-view \
  --category bottle
```

`run_anomalib.py` は `--dataset mvtec_4k` かつ
`{dataset_root}/{category}/foreground` が存在する場合、生配置とみなして
`parser.error` で停止する。`--prefetch-weights` だけの実行では、既存どおり
データ配置の確認をスキップする。

## 検証スクリプト

```bash
python scripts/validate_mvtec4k.py --root datasets/MVTec-4K
```

主なオプション:

- `--root`: MVTec-4K の展開先。既定は `datasets/MVTec-4K`。
- `--category`: 検証するカテゴリ。複数指定可。省略時は root 配下に存在する
  既定 7 カテゴリ。
- `--out`: 検証結果を JSON で保存する。
- `--skip-pixels`: Pillow を使うピクセル検査を省略する。
- `--max-samples`: test の各 label_name と train から先頭 N 件ずつだけ
  ピクセル検査する。メモリと時間を抑えたいときに使う。
- `--make-view DIR`: エラー 0 のカテゴリだけ clean view を作る。
- `--link-mode {hardlink,copy,symlink}`: clean view の作り方（既定 hardlink。symlink は Anomalib では使えない）。

終了コード:

- 0: エラーなし。
- 1: 検証対象カテゴリが一つも無い。
- 2: error の Finding が見つかった、または clean view 作成に失敗した。

### 検出項目

マニフェスト読み込み:

- `manifest_missing`: `{category}/train.jsonl` または `test.jsonl` が無い。
- `manifest_unreadable`: jsonl を開けない。
- `invalid_json`: jsonl の行が JSON として不正。
- `invalid_json_object`: 行が JSON オブジェクトではない。
- `missing_key`: 必須キー（filename, clsname, label, label_name）が無い。
- `invalid_field`: 型や値が不正。

ラベル・パス整合性:

- `category_mismatch`: clsname がカテゴリ名と一致しない。
- `invalid_split`: split が train/test 以外。
- `invalid_label`: label が 0/1 以外。
- `label_name_mismatch`: label_name == "good" と label == 0 が対応しない。
- `train_label_not_good`: train に good 以外が含まれる。
- `filename_path_mismatch`: filename が期待レイアウトと一致しない。
- `mask_required`: label=1 なのに mask が無い。
- `mask_path_mismatch`: mask が `ground_truth/{label_name}/{stem}_mask.png`
  と一致しない。
- `unexpected_mask`: label=0 なのに mask がある。
- `foreground_path_mismatch`: foreground が期待レイアウトと一致しない。
- `duplicate_filename`: filename が重複している。

ディスク:

- `missing_file`: マニフェストが参照する画像・マスク・foreground が無い。
- `unlisted_file`: train/test/ground_truth 配下にマニフェスト未記載の画像がある。

ピクセル:

- `corrupt_image`: 画像・マスク・foreground を開けない。
- `size_mismatch`: マスクまたは foreground のサイズが画像と一致しない。
- `non_binary_mask`: マスクに 0/255 以外の画素がある。
- `empty_mask`: label=1 なのにマスクの前景画素が 0。

判定はしないが記録する統計: マスク面積比 `mask_area_fraction`、bbox、
マスク画素のうち foreground>0 の内側にある割合 `mask_in_foreground_fraction`
（欠陥が物体上にあるかの座標整合の目安）。

論文表との差:

- `paper_count_mismatch`: 件数が論文表の参考値と異なる。error ではなく warning。
- `unknown_category`: 論文表に無いカテゴリ。warning。

## 解像度条件の扱い

MVTec-4K のネイティブ画像は 4096x4096 を想定している。一方、CPU 優先の
ベンチマークではモデル入力を 256px などへ縮小する実行が現実的である。
縮小入力とネイティブ 4K 入力は別条件であり、同じ表で単純比較してはいけない。

`run_anomalib.py` は report に `resolution_condition` を追加する。

```json
{
  "source_resolution": [4096, 4096],
  "model_input_resolution": [256, 256],
  "condition": "downscaled"
}
```

`condition` は次の分類を取る。

- `native`: 元画像サイズとモデル入力サイズが一致。
- `downscaled`: 縦横とも入力が元以下で、少なくとも一方が小さい。
- `upscaled`: 縦横とも入力が元以上で、少なくとも一方が大きい。
- `mixed`: 拡大と縮小が混在。
- `unknown`: PIL 不在、画像不在、ヘッダ読めない等で元サイズ不明。

`source_resolution` は test 配下をソートして最初の画像ヘッダから読む。
実画像の読み込みではなくヘッダのみを使う。読み取れない場合は `null` になる。
この値は最初の test 画像 1 枚に基づく代表値であり、カテゴリ内の全画像の
サイズを保証するものではない。全数確認は検証スクリプトの `image_sizes` を使う。

## ライセンスと制限事項

以下は本リポジトリの開発時点で確認・整理した内容であり、法的助言ではない。
最終的な利用可否は配布元の一次情報と利用者自身の判断で確認すること。

- HF のデータカード（`license: apache-2.0`）と HiAD の GitHub リポジトリは
  Apache-2.0 と表記している。
- しかし MVTec-4K の画像は MVTec AD（CC BY-NC-SA 4.0：表示・非商用・継承）を
  元にした派生物である。NC・SA 条件の付いた素材の派生物を、派生元より緩い
  Apache-2.0 で再許諾できるとは考えにくい。本リポジトリは、MVTec AD の条件
  （非商用・継承）が及ぶものとして扱う（解釈であり、権利者の見解は未確認）。
- 超解像に使われた StableSR のコードは S-Lab License 1.0（非商用目的の再配布・
  利用を許可、商用は権利者に要連絡）。生成画像への適用範囲は明記されていない
  （未確認）。
- したがって本リポジトリでは、MVTec-4K およびその派生画像・重みの商用利用は
  不可として扱う。本リポジトリはデータ・派生画像・重みを再配布せず、
  コミットしない。
- データを取得・利用する際は、配布元のデータカード、MVTec AD のライセンス、
  StableSR のライセンスをそれぞれ確認すること。
- 実測結果は `docs/verification/issue12_mvtec4k.md` に記録する。
