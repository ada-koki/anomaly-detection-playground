# 実行モードとチェックポイント来歴（Issue #11）

CPU 優先ベンチマークで「CPU だけで学習・推論した結果」と「GPU で学習し CPU で
推論した結果」を取り違えないための仕組みを説明します。あわせて、意図しない
デバイスフォールバック（CUDA を要求したのに CPU で走ってしまう等）の検知方法を
まとめます。

- 実装（純粋関数）: `adbench/devices.py`、`adbench/training_mode.py`
- 集計: `adbench/aggregate.py`
- CLI: `scripts/run_anomalib.py`

## 1. 前提: 1 プロセス内でデバイスを切り替えない

GPU 学習＋CPU 推論は、**1 プロセス内でデバイスを切り替えるのではなく、
2 段階の実行に分けて**実現します。

1. GPU 機で学習し、チェックポイントと来歴サイドカーを保存する
2. チェックポイントとサイドカーを CPU 機へコピーする
3. CPU 機でチェックポイントを読み込み、推論だけを行う

これは、GPU と CPU のモデルを同時に保持しないことでメモリ使用量を抑えるためです。
本リポジトリの検証環境は CPU のみ・メモリが限られるため、この方式を採用しています。

## 2. 実行モード一覧

実行モードは `adbench/devices.py` の `EXECUTION_MODES` で定義されています。

| execution_mode | 学習 | 推論 | 意味 |
|---|---|---|---|
| `cpu_only` | cpu | cpu | 学習も推論も CPU。CPU 優先ベンチマークの基本モード。 |
| `gpu_train_cpu_infer` | cuda | cpu | 来歴で GPU 学習が証明されたチェックポイントを CPU で推論した結果。 |
| `gpu_reference` | cuda | cuda | GPU 上で学習・推論を完結させた参考値。CPU 実用性の比較には使わない。 |
| `external_checkpoint_unknown_train_device` | 不明 | 任意 | `--checkpoint` を使ったが、来歴が無い・学習デバイスが不明・sha256 不一致などで学習デバイスを信頼できない場合。 |
| `cpu_train_gpu_infer` | cpu | cuda | CPU 学習＋GPU 推論。特殊な検証用。通常は使いません。 |

分類は `classify_execution_mode(train_device, infer_device)` が行います。

- `train_device` が `None`（学習していない、または来歴が不明・不信）の場合は
  `external_checkpoint_unknown_train_device`。
- `"cuda:0"` のようなインデックス付き指定は `normalize_device` で `"cuda"` に
  正規化してから判定します。
- 上記以外の組合せ（例: `mps` を含む組合せ）は `ValueError` になります。

## 3. 2 段階実行の手順

引数名は `scripts/run_anomalib.py` の実装に合わせています。`[通常の実行オプション]`
の部分は、普段使っているデータ・設定・出力先などの引数に置き換えてください。

```bash
# 1) GPU 機: 学習してチェックポイントと来歴サイドカーを保存する
python scripts/run_anomalib.py --device cuda --save-checkpoint ~/adbench_ckpt/run1 \
    [通常の実行オプション]

# 2) 生成物を CPU 機へコピーする（リポジトリには入れない）
#    ~/adbench_ckpt/run1/model.ckpt
#    ~/adbench_ckpt/run1/model.ckpt.provenance.json

# 3) CPU 機: チェックポイントを読み込んで推論する
python scripts/run_anomalib.py --device cpu \
    --checkpoint ~/adbench_ckpt/run1/model.ckpt \
    [通常の実行オプション]
```

注意点:

- `--save-checkpoint DIR` は保存先ディレクトリを指定します。保存名は
  `DIR/model.ckpt` です。学習後の best/last チェックポイントがあればそれを、
  無ければ `save_checkpoint` で保存します。
- `--save-checkpoint` と `--checkpoint` の同時指定はエラーです。
- 保存先は**リポジトリ外**を推奨します。重み（チェックポイント）はコミット
  しないでください。
- 学習側の report.json には `saved_checkpoint`（`path` / `sha256` /
  `provenance_path`）が記録されます。
- 推論側の report.json には、`devices`（`train` / `infer` / `warnings`）、
  `execution_mode`、`--checkpoint` 使用時の `checkpoint_provenance`
  （サイドカーが読めた場合。パスはファイル名のみ）が記録されます。
- `training_mode` は後方互換のため残しています。来歴で `train_device=cuda` が
  証明され、推論が cpu の場合は `gpu_train_cpu_infer` になります。

## 4. 来歴（provenance）による学習デバイスの証明

チェックポイントの横に、学習デバイスなどを記録した JSON サイドカーを置きます。

- ファイル名: `model.ckpt.provenance.json`
  （`provenance_path_for(ckpt)` が返す名前）
- スキーマ版: `schema_version: 1`
- 主なキー: `checkpoint_sha256`、`train_device`、`observed_train_device`、
  `cuda_device_name`、`torch_version`、`anomalib_version`、`model`、
  `category`、`protocol_name`、`protocol_sha256`、`run_id`

推論側はサイドカーを読み、次の 2 点を確認します。

1. **sha256 の一致**: サイドカーの `checkpoint_sha256` が実際のチェックポイント
   ファイルの sha256 と一致するか（`provenance_matches`）。
2. **学習デバイス**: `train_device` が `cuda` か。

sha256 が一致しない、サイドカーが無い、`train_device` が読めない、といった場合は
学習デバイスを信頼せず `train_device = None` として扱い、実行モードは
`external_checkpoint_unknown_train_device` になります。警告は report の
`devices.warnings` に入ります。

sha256 は手元でも確認できます。

```bash
sha256sum ~/adbench_ckpt/run1/model.ckpt
```

## 5. フォールバック検知と終了コード 3

要求デバイスと実際に観測したデバイスが違う場合（例: `--device cuda` を要求したが
モデルのパラメータが CPU 上にあった）は、`detect_device_mismatch` が日本語の
警告文を返します。観測は学習直後と推論直前に行い、パラメータを持たないモデルでは
`unknown` になります。

- 不一致が検出された場合、**report を保存した上で終了コード 3** で終了します。
- `--allow-device-mismatch` を指定した場合は警告のみとし、終了コード 3 にしません。
- 「CUDA を要求したが CUDA が使えない」場合は従来どおりエラーです
  （フォールバックして実行はしません）。

| 終了コード | 意味 |
|---|---|
| 0 | 正常終了（`--allow-device-mismatch` で不一致を警告に留めた場合を含む） |
| 3 | 要求デバイスと観測デバイスの不一致を検出した |

## 6. 比較時の注意

`adbench/aggregate.py` は実行モードを区別して集計します。

- runs テーブルに `execution_mode` 列が入ります。
- summary のグループキーは
  `(model, category, image_size, cpu_threads, execution_mode)` です。
- `report.md` のカテゴリ別比較は実行モードごとに別表になり、
  見出しは `#### 実行モード: cpu_only` のようになります。全行が `cpu_only`
  でも見出しは出ます。
- レポートには「実行モードが異なる行は同一表で比較しない。`cpu_only` 以外の
  結果は CPU のみの実用性の数値と混同しない」という注記が入ります。

**実行モードの異なる行を同じ表で比較しないでください。** 特に `gpu_reference` や
`gpu_train_cpu_infer` の数値は、CPU のみで完結した結果（`cpu_only`）の
スループット・レイテンシの目安として使えません。

Issue #10 以前の report（`execution_mode` が無い旧形式）は、`training_mode` と
`device` から実行モードを推定します（`execution_mode_of_report`）。推定値であり
証明ではないため、可能なら当時の環境情報とあわせて確認してください。

## 7. 検証状況

- 本リポジトリの検証環境は GPU の無い Linux x86_64（CPU のみ）です。
  CPU 経路（`cpu_only`、`external_checkpoint_unknown_train_device`）のみ実測して
  います。
- **GPU 経路（`gpu_train_cpu_infer`、`gpu_reference`、`cpu_train_gpu_infer`）は
  本環境では未実機検証です。** 分類・来歴・不一致検知のロジックは torch /
  anomalib に依存しない純粋関数として単体テストで確認していますが、実際の GPU
  機での学習・チェックポイント保存・CPU 機への受け渡しは未検証です。
- `--save-checkpoint` の保存挙動（best チェックポイントの有無による分岐）も
  GPU 実行を伴うため未検証です。
