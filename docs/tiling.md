# タイル分割の設定と検証

## 目的

高解像度画像をそのまま 1 枚に縮小すると、微小な欠陥が作業解像度上で消えることがある。
本リポジトリでは、作業解像度の画像を重なりのあるタイルに分割し、Anomalib Tiler で
タイルごとの特徴（埋め込み）を抽出した後に元の作業解像度へ合成する方式を扱う。

この文書は Issue #13 のうち、タイル設定・座標計算・設定検証・レポート項目の
仕様をまとめる。`run_anomalib.py` への組み込みは別途行う。

## 全体縮小とタイルの違い

- 全体縮小（`global_resize`）: 元画像を `--image-size N` の正方形へ Resize し、
  1 枚の画像として推論する。処理は単純でメモリも小さいが、縮小率が大きいと
  微小欠陥が失われやすい。
- タイル（`tiled`）: 同じ作業解像度 `N x N` を作業画像とし、その上で
  `tile_size` と `stride` に従ってタイル化する。タイルは重なりを持てる。
  Anomalib Tiler はタイルごとの特徴を `untile` で合成し、その合成後の
  特徴マップ上で位置ごとに採点する。

注意: 本リポジトリの `run_anomalib.py` は `--image-size N` で
`Resize(N, N)` してからタイル化する想定である。したがってタイル分割だけでは
`N` より高い解像度の情報は復元されない。微小欠陥を扱うには `N` を十分大きく
設定した上でタイル化する。`N` を大きくするとメモリ使用量が増えるため、
後述のメモリ概算式と実測で上限を決める。

## 設定値と座標

`TilingConfig` の設定値:

- `tile_size`: タイルの一辺。1 以上。
- `stride`: タイル開始位置の間隔。`1 <= stride <= tile_size`。
- `mode`: `padding` または `interpolation`。
- `remove_border_count`: タイル境界から除去する画素数。0 以上かつ
  `2 * remove_border_count < tile_size`。

重なり率:

```text
overlap_ratio = 1 - stride / tile_size
```

タイル開始位置は Anomalib と同じく、パディング後の各辺 `length` に対して次で求める。

```text
range(0, length - tile_size + 1, stride)
```

パディング後の長さは Anomalib の `compute_new_image_size` の 1 辺分と同じ式である。

```text
delta = length - tile_size
delta % stride == 0 なら length
それ以外は ceil(delta / stride) * stride + tile_size
```

タイル座標はパディング後の作業解像度上で `(y0, x0, y1, x1)` とし、
行方向を外側、列方向を内側にした行優先順で並べる。

## タイル数と重なり率の例

| 作業解像度 | tile_size | stride | overlap_ratio | 1 辺のタイル数 | タイル数 | 可否 |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 256 | 256 | 0.0 | 2 x 2 | 4 | 可 |
| 512 | 256 | 128 | 0.5 | 3 x 3 | 9 | 可 |
| 1024 | 256 | 256 | 0.0 | 4 x 4 | 16 | 可 |
| 600 | 256 | 256 | 0.0 | 3 x 3（パディング後 768） | 9 | 不可 |

計算根拠:

- 512 / 256 / 256: `range(0, 257, 256)` は `0, 256` の 2 個。2 x 2 = 4。
- 512 / 256 / 128: `range(0, 257, 128)` は `0, 128, 256` の 3 個。3 x 3 = 9。
- 1024 / 256 / 256: `range(0, 769, 256)` は `0, 256, 512, 768` の 4 個。4 x 4 = 16。
- 600 / 256 / 256: 600 はパディング後 768 になり、`range(0, 513, 256)` は
  `0, 256, 512` の 3 個。ただしパディングが必要なため本リポジトリでは拒否する。

## Anomalib Tiler の合成ルール

Anomalib 2.7.0 の Tiler では、`tile(image)` の出力は
`(B * nh * nw, C, tile, tile)` で、並びは batch 優先、その中が行優先、列優先である。
`untile(tiles)` はタイルを加算し、被覆数で割って平均する。重複領域は
カウント平均になる。NaN は 0 に置き換えられる。

PaDiM / PatchCore の torch model では、大まかに次の順で処理される。

1. `tiler.tile(input)` で入力をタイル化する。
2. 各タイルを backbone に通し、埋め込み（特徴マップ）を得る。
3. `tiler.untile(embedding)` で特徴マップを合成する。
4. 合成後の特徴マップ上で位置ごとに採点する。

つまり異常マップをタイルごとに作って平均するのではなく、特徴を合成してから
位置ごとに採点する。PaDiM は合成後の特徴マップ全体に対して、位置ごとの
平均・共分散などの統計を学習する。

EfficientAD の torch model には `tiler` 属性が無いため、本リポジトリでは
タイル非対応として扱う。

## パディングが必要な構成を拒否する理由

Anomalib Tiler は、各辺で `(image_size - tile_size) % stride != 0` の場合に
右端・下端をゼロパディングしてタイル数を増やす。しかし `untile` の最後の
切り出しは「入力画素サイズ」で行われる。特徴マップは 1/4 解像度などの
小さい空間サイズを持つため、この切り出しがそのままでは効かず、パディングで
拡張した領域を含んだ特徴マップが `output_size` へ補間される。その結果、
異常マップの座標がずれる。

このため本リポジトリでは、各辺で次の条件を満たす構成のみ許可する。

```text
(image_size - tile_size) % stride == 0
```

`validate_for_model(image_size, config)` は、タイルサイズが作業解像度を超える場合や
パディングが必要な場合にエラー文を返す。パディングが必要な場合は、有効な
`stride` の候補や作業解像度の候補もメッセージに含める。

`mode` は `padding` / `interpolation` を設定できるが、本リポジトリでは
パディングが必要な構成自体を拒否する。

## メモリの見積もり

PaDiM の学習時メモリバンクの概算式を次に示す。

```text
学習時メモリバンク（概算）
  ≈ 学習枚数
    × 埋め込み次元
    × (作業解像度 / 4)^2
    × 4 bytes
```

- `4 bytes` は float32 の埋め込みを保持する場合の概算である。
- PaDiM の既定の `n_features` の具体的な値はこの文書では未確認として扱う。
  モデル設定から確認すること。
- タイル化しても、`untile` 後の特徴マップは作業解像度 `N` に依存する。
  そのため学習時メモリバンクは「タイル数だけ増える」わけではない。
  ただしタイル化中の一時テンソルや backbone の活性は別途メモリを使う。
- 実際のピーク RSS はモデル、DataLoader、推論バッチサイズ、スレッド数などに
  依存するため、この式だけで上限を確定しない。

## 使い方（CLI）

`scripts/run_anomalib.py` のオプションは次のとおり（PaDiM/PatchCore のみ。EfficientAD は tiler 属性が無いため拒否する）。

```text
--image-size N            作業解像度（正方形）
--tile-size T             タイルの一辺
--tile-stride S           タイル開始位置の間隔（省略時は T、重なりなし）
--tile-mode {padding,interpolation}  既定 padding
```

`remove_border_count` は `TilingConfig` では指定できるが、CLI では 0 固定とする
（Anomalib の untile は特徴マップ上で境界除去を行うため、画素単位の意味が
モデルの特徴ストライドに依存し、検証していないため）。

例（PaDiM、作業解像度 512、タイル 256、50% 重なり）:

```bash
python scripts/run_anomalib.py --model padim --image-size 512 \
  --tile-size 256 --tile-stride 128 ...
```

タイル設定はモデル生成直後に `model.model.tiler` へ直接設定する
（`TilerConfigurationCallback.setup` と同等。チェックポイント読み込み時も有効）。

タイル指定が無い場合は全体縮小として扱う。タイル指定がある場合は、
引数検証時に `validate_for_model(N, TilingConfig(...))` を呼び、
エラーがあれば終了コード 2 で止める。

レポートには `spatial_strategy` を保存し、全体縮小とタイルを区別する。

- 全体縮小: `describe_global(N)` を使い、`spatial_strategy` は
  `global_resize`、`tiles_per_image` は 1。
- タイル: `describe_tiling(N, config)` を使い、`spatial_strategy` は
  `tiled`。主なキーは `working_resolution`, `tile_size`, `stride`,
  `overlap_ratio`, `mode`, `remove_border_count`, `grid`,
  `tiles_per_image`, `padded_resolution`, `requires_padding`,
  `coverage`, `merge_rule`, `notes`。

`coverage` は元画像範囲内の最小被覆数・最大被覆数・未被覆画素数を表す。
パディング不要な有効構成では、未被覆画素数は 0 であることをテストで確認する。

## 未検証事項

この文書には実測の精度・速度・ピーク RSS を記載しない。Issue #13 の実測は
`docs/verification/issue13_tiling.md` に記録予定である。特に次は未検証。

- CPU 実行時のタイル分割あり・なしの精度、処理時間、ピーク RSS。
- `remove_border_count` を変更した場合の座標・精度への影響。
- `mode=interpolation` を実際に使った場合の合成結果と座標。
- 推論バッチサイズとピーク RSS の関係。
