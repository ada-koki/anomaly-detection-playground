# PatchCore 学習メモリ対策

## 背景

PatchCore は正常画像のパッチ特徴量をメモリバンクとして保持し、推論時に最近傍距離で異常度を計算する手法である。学習（`fit`）時には全学習画像のパッチ特徴量を一括でメモリ上に展開してから KCenterGreedy で coreset を選ぶため、学習ピークメモリは「学習画像枚数 × 1 枚あたりのパッチ数 × 特徴次元」に比例して増大する。

wide_resnet50_2（layer2 + layer3、特徴次元 1536）・256px のような設定では、学習画像が多いカテゴリで学習時ピークメモリが十数 GiB 規模になり、16GB・スワップなしのホストでは OOM Kill に至ることがある（VisA candle の既定条件 PatchCore が学習中に OOM Kill した例を Issue #16 に記録済み）。

本 Issue では、PatchCore の学習時メモリを抑えるための次の3オプションと、学習前のメモリ見積りおよび上限ガードを追加した。

- `--patchcore-coreset-mode {global,chunked}`
- `--patchcore-chunk-images N`
- `--max-train-feature-mib M`

`chunked` は coreset 選択をチャンク分割する**近似**であり、標準プロトコルから逸脱する。詳細は後述する。

## オプション

### `--patchcore-coreset-mode {global,chunked}`

- `global`（既定）: 従来どおり、全学習画像のパッチ特徴量を一括で展開し、1 回の KCenterGreedy で coreset を選ぶ。標準プロトコルの条件。
- `chunked`: 学習をチャンク単位で進め、チャンクごとに KCenterGreedy（`sampling_ratio` 0.1）で縮約し、最後に残りを縮約して結合する。メモリバンクは `float32` で保持する。ピークメモリは「1 チャンク分の特徴量 + それまでに蓄積したメモリバンク」に抑えられる。
  - `chunked` ではチャンク枚数の指定が必須。

> `global` と `chunked` は coreset の**選択点集合が異なる**。したがって両者の精度は同一条件の比較ではなく、`chunked` は近似として扱う（後述）。

### `--patchcore-chunk-images N`

`chunked` のときに 1 チャンクで処理する学習画像枚数。小さいほどピークメモリは下がるが、縮約回数が増え、選択点集合は `global` からさらに離れる傾向がある。本 Issue では 20 / 100 を用いた（枚数のスイープは未実施。後述）。

### `--max-train-feature-mib M`

学習前に見積った特徴量メモリの概算ピーク（MiB）がこの上限を超える場合、または上限が指定されていて見積りができない場合（タイル分割・既定外 backbone など）は、学習を開始せずに失敗レポートを残す（非ゼロ終了）。物理メモリより十分小さい値を指定することで、OOM Kill を未然に防ぐ安全弁として使う。

## 見積り式の考え方と限界

見積りは「学習画像枚数 × 1 枚あたりのパッチ数 × 特徴次元 × 4 byte」を基本単位とする。これは `adbench/patchcore_memory.py` の実装に基づく**概算**であり、次のように組み立てる。

- `global`（概算）: 全画像分の特徴量（embedding 全量）と、vstack 後の全量が同時に存在すると仮定して 2 倍し、さらに KCenterGreedy の射影特徴（339 次元）分を加算する。
- `chunked`（概算）: 1 チャンク分の特徴量と、蓄積済みメモリバンクの和を取る。

いずれもバイト→MiB 変換のうえで概算ピークとする。

**限界（重要）**: 見積りは特徴量テンソルのみを対象としており、アロケータの断片化・DataLoader・距離配列・sklearn の作業領域などを含まない。このため実測ピーク RSS は見積りを**上回る傾向**がある。本 Issue の完走した全 run で実測ピーク > 見積り であった（実測は `docs/verification/issue30_patchcore_memory.md` を参照）。上限 `--max-train-feature-mib` は、実測が見積りを超えても安全側に倒れるよう、物理メモリより十分小さい値に余裕をもって設定することを推奨する。

## chunked はプロトコル逸脱（variant 列）

`chunked` は coreset 選択を分割する近似であり、選択点集合が `global` と一致しない。そのため `chunked` の精度は標準プロトコル（`global`）の結果と**同一条件では比較できない**。

集計（`scripts/aggregate_results.py` → `adbench/aggregate.py`）では `variant` 列を追加し、グループキーに含めている。`run_anomalib.py` の `chunked` 実行は `chunked_coreset_{N}img`（`N` は `--patchcore-chunk-images`）として記録され、既定条件（`variant` 空 = `global`）と分けて集計される。`report.md` では `patchcore [variant]` の形式で表示される。`global` と `chunked` を同じ行に混ぜて平均などを取らないこと。

## report.json のフィールド

- `training_memory_estimate`: 学習**前**に算出した特徴量メモリの概算（ピーク MiB 相当）と、上限に対する可否（`within_budget`、超過時は `budget_message`）を記録する。上限超過や見積り不能で学習を開始しなかった場合は、この情報を残したまま失敗レポートとなる。
- `training_memory_mitigation`: `fit` **後**に、実際に適用された緩和策（`coreset_mode`、`chunked` のときの `chunk_images` など）を記録する。

> 上記は意味レベルの説明である。キー構成の厳密な定義は実装（`adbench/patchcore_memory.py`、`scripts/run_anomalib.py`）を正とする。

## 使い方

`chunked` で学習メモリを抑える例（値は例。実際の必須引数は `scripts/run_anomalib.py --help` を参照）:

```
python scripts/run_anomalib.py \
  --model patchcore \
  --device cpu --cpu-threads 8 \
  --patchcore-coreset-mode chunked \
  --patchcore-chunk-images 100 \
  --max-train-feature-mib 4000
```

`global` のまま上限ガードだけを使い、OOM を避けたい例:

```
python scripts/run_anomalib.py \
  --model patchcore \
  --device cpu --cpu-threads 8 \
  --patchcore-coreset-mode global \
  --max-train-feature-mib 6000
```

この場合、見積り概算が 6000 MiB を超えれば学習を開始せずに失敗レポートを残す。

## 注意・未検証

- これら3オプションは **PatchCore 専用**である。
- `chunked` の精度が `global` に対して一般にどの程度変わるかは未検証（本 Issue は 3 カテゴリ各 1 run のみ）。
- 見積り式の誤差の一般性は未検証。見積りは概算であり、実測を保証しない。
- 未実施: `float16` メモリバンク、学習画像の間引き、チャンク枚数のスイープ、`global` と `chunked` の選択点集合の差の定量化、複数 seed。
- レイテンシは本 Issue の対象外。`chunked` run で e2e p95 が大きい理由は未検証（Issue #31 で干渉検出を導入予定）。
