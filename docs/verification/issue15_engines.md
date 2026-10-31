# Issue #15 検証記録: CPU 推論エンジン比較（PyTorch / ONNX Runtime / OpenVINO）

本記録の数値はすべて `benchmarks/issue15/engines_summary.csv`（および同内容の
`engines_summary.md`）に由来する。比率は CSV の値から計算したものである。
設計と CLI 仕様は `docs/engines.md` を参照。

## 1. 目的

- 学習済み Anomalib モデル（`model.model`）を PyTorch CPU 以外の推論エンジンでも
  実行できるかを確認する。
- PyTorch CPU を基準に、推論結果の整合性（`pred_score` / `anomaly_map` の差、AUROC、
  画像レベル判定一致率）と、P50/P95 レイテンシ・ピーク RSS・ロード時間を比較する。
- 書き出し方法（dynamo / legacy）、opset、モデルサイズ、再現コマンドを残す。

## 2. 実験条件

### 2.1 環境

| 項目 | 値 |
| --- | --- |
| Python | 3.13.5 |
| torch | 2.15.0+cpu |
| anomalib | 2.7.0 |
| onnx | 1.23.2 |
| onnxscript | 0.7.2 |
| onnxruntime | 1.31.1（CPUExecutionProvider） |
| openvino | 2026.4.1（CPU、PERFORMANCE_HINT=LATENCY、f32） |
| CPU | 論理 8 コア、GPU なし |
| ホスト | 共有環境。本計測と無関係な他プロセスが常駐。各 run 開始時の load average（1 分）は 1.49〜7.46 |

### 2.2 データ・モデル

- MVTec AD の `bottle`、`image_size=256`。
- プロトコル `mvtec_ad_v1`（`val_split from_test 0.2`、`seed 42`）で学習した
  PaDiM / PatchCore の checkpoint（`scripts/run_anomalib.py --save-checkpoint`）を使用。
- test 67 枚。ラベルは同じ run の `manifest.json` の `label_index`、画素 AUROC のマスクは
  元解像度から anomaly_map 解像度へ最近傍リサイズしたものを使用。
- AUROC は PostProcessor を通さない生スコア・生マップで計算（CSV の `*_raw` 列）。

### 2.3 計測条件

- `threads=8`: warmup 5 枚、repeats 3 → 67 枚 × 3 = 201 サンプル。
- `threads=1`: warmup 5 枚、repeats 1 → 67 サンプル。
- 計測範囲は **model 推論のみ**（前処理済み入力テンソル → 出力 numpy 変換まで）。
  decode・前処理（eval_transform）・PostProcessor は含まない。
- エンジンごとに別プロセス（ワーカー）で計測。入力は親が 1 回だけ前処理して保存した
  `inputs.npy` を `mmap` で読み、全エンジンでバイト一致の入力を使う。
- run_id とラベルの対応:

| label | model | threads | run_id |
| --- | --- | --- | --- |
| padim_t8 | padim | 8 | 20261031_213625 |
| patchcore_t8 | patchcore | 8 | 20261031_213700 |
| padim_t1 | padim | 1 | 20261031_213859 |
| patchcore_t1 | patchcore | 1 | 20261031_213931 |

## 3. antialias と前処理の扱い

Anomalib 2.7 の `to_onnx` は Lightning モジュール全体を書き出すため、`PreProcessor` の
`export_transform`（`Resize` の antialias 無効）がグラフに含まれる。4K→256 のような
大きな縮小では antialias の有無で前処理結果が大きく変わる（Issue #12）ため、
`engine.test` の精度指標と別条件になってしまう。

そこで本実装では:

- 書き出し対象を `model.model` のみとし、PreProcessor / PostProcessor を含めない。
- 前処理は全エンジン共通で Python 側の `eval_transform`
  （`Resize(256, bilinear, antialias=True)` → `Normalize(ImageNet)`、`engine.test` と同じ）で行い、
  **前処理済みテンソル（1×3×H×W float32）**を各エンジンに渡す。
- 比較対象は生の `pred_score` / `anomaly_map`。判定一致率のみ、run の
  `report.json` の `defect_size.thresholds.image` を使って計算する。

書き出した ONNX グラフ内の `Resize` は、PaDiM が 3 個（特徴マップの
nearest/asymmetric 2 個、異常マップの linear/half_pixel 1 個）、PatchCore が 2 個
（nearest/asymmetric 1 個、linear/half_pixel 1 個）。いずれも `antialias=0` だが、
すべて**拡大**（アップサンプリング）であり、PyTorch の `F.interpolate`（antialias なし）と
同じ処理である。縮小側の antialias はグラフ外の前処理で担保している。

結果として、ONNX Runtime / OpenVINO の出力は torch と高い整合性を示した
（判定一致率 1.0、AUROC は 3 エンジンで同一、スコア差は最大 2.99e-4。§6）。

## 4. 書き出し結果

| モデル | export_method | opset（CSV の値） | 外部データ | ONNX サイズ (B) | OpenVINO IR サイズ (B) | export_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| PaDiM | dynamo | 18 | `model.onnx.data` | 176,711,168 | 176,724,489 | 1.849 |
| PatchCore | dynamo | 18 | `model.onnx.data` | 231,153,221 | 231,197,127 | 3.613 |

- 全 12 行で `export_method=dynamo`、`opset=18`。torch 2.15.0+cpu の dynamo exporter で、
  opset 17 を要求すると 18 で書き出した後のダウンコンバートに失敗した
  （`No Adapter To Version $17 for Pad`）ため、既定を 18 とした。
- `engine_report.json` には要求値 `opset_requested` と、ファイルの `opset_import` から
  読んだ実値 `opset` を分けて記録している。本計測は要求 18・実値 18（CSV の `opset` 列）。
- 重みは外部データ `model.onnx.data` に保存される。CSV の `model_size_bytes` は
  `model.onnx` と `model.onnx.data` の合計。
- OpenVINO IR は `save_model(..., compress_to_fp16=False)`（FP32）で保存。
  IR サイズは ONNX 合計より PaDiM で 13,321 B、PatchCore で 43,906 B 大きい（差分は CSV から計算）。
- CSV の `export_seconds` は onnxruntime 行と openvino 行が同値（PaDiM 1.849、PatchCore 3.613）。
  ONNX 書き出しを共有しているためで、**IR 変換の所要秒は本 CSV の列からは分離できない**。
- 演算子ヒストグラム（`op_histogram`）は `engine_report.json` に残る（非コミット）。

## 5. 結果表

### 5.1 threads=8（warmup 5、repeats 3、201 サンプル）

| モデル | エンジン | P50 (ms) | P95 (ms) | ロード (s) | ピーク RSS (MiB, VmHWM) | 画像 AUROC (raw) | 画素 AUROC (raw) | 判定一致率 | map 最大絶対差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PaDiM | torch | 26.985 | 47.916 | 3.839 | 920.73 | 0.996324 | 0.976871 | 1.0 | — |
| PaDiM | onnxruntime | 13.542 | 19.535 | 0.084 | 379.16 | 0.996324 | 0.976871 | 1.0 | 0.000259 |
| PaDiM | openvino | 14.312 | 31.197 | 0.185 | 565.72 | 0.996324 | 0.976871 | 1.0 | 0.000824 |
| PatchCore | torch | 186.759 | 906.286 | 5.515 | 1287.31 | 1.0 | 0.985522 | 1.0 | — |
| PatchCore | onnxruntime | 82.649 | 100.484 | 0.879 | 806.1 | 1.0 | 0.985522 | 1.0 | 0.000809 |
| PatchCore | openvino | 81.275 | 112.881 | 2.161 | 1213.07 | 1.0 | 0.985522 | 1.0 | 0.001853 |

### 5.2 threads=1（warmup 5、repeats 1、67 サンプル）

| モデル | エンジン | P50 (ms) | P95 (ms) | ロード (s) | ピーク RSS (MiB, VmHWM) | 画像 AUROC (raw) | 画素 AUROC (raw) | 判定一致率 | map 最大絶対差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PaDiM | torch | 59.838 | 63.388 | 3.630 | 920.61 | 0.996324 | 0.976871 | 1.0 | — |
| PaDiM | onnxruntime | 43.312 | 51.221 | 0.044 | 378.24 | 0.996324 | 0.976871 | 1.0 | 0.000259 |
| PaDiM | openvino | 52.022 | 58.044 | 0.250 | 563.75 | 0.996324 | 0.976871 | 1.0 | 0.000824 |
| PatchCore | torch | 691.930 | 876.424 | 4.824 | 1287.11 | 1.0 | 0.985522 | 1.0 | — |
| PatchCore | onnxruntime | 558.090 | 611.630 | 0.937 | 805.79 | 1.0 | 0.985522 | 1.0 | 0.000674 |
| PatchCore | openvino | 545.028 | 601.215 | 2.626 | 1207.94 | 1.0 | 0.985522 | 1.0 | 0.001794 |

### 5.3 差分の内訳（基準: torch）

| モデル | threads | エンジン | score 最大絶対差 | map 最大絶対差 | map 相対差 p99.9 |
| --- | --- | --- | --- | --- | --- |
| PaDiM | 8 | onnxruntime | 0.000104 | 0.000259 | 6e-05 |
| PaDiM | 8 | openvino | 0.000275 | 0.000824 | 0.000112 |
| PatchCore | 8 | onnxruntime | 0.000177 | 0.000809 | 3.7e-05 |
| PatchCore | 8 | openvino | 0.000299 | 0.001853 | 8.5e-05 |
| PaDiM | 1 | onnxruntime | 0.000104 | 0.000259 | 6e-05 |
| PaDiM | 1 | openvino | 0.000275 | 0.000824 | 0.000112 |
| PatchCore | 1 | onnxruntime | 0.000177 | 0.000674 | 3.7e-05 |
| PatchCore | 1 | openvino | 0.000299 | 0.001794 | 8.5e-05 |

## 6. 観察（CSV から確認できる事実）

- **整合性**: 全 12 行で判定一致率 1.0。画像 AUROC（raw）は PaDiM 0.996324 /
  PatchCore 1.0、画素 AUROC（raw）は PaDiM 0.976871 / PatchCore 0.985522 で、
  同一モデル内では 3 エンジン・2 スレッド条件すべて同値。差分は
  `score_max_abs_diff` 1.04e-4〜2.99e-4、`map_max_abs_diff` 2.59e-4〜1.853e-3、
  `map_p999_rel_diff` 3.7e-5〜1.12e-4。
- **threads=8 の P50**: ONNX Runtime / OpenVINO は torch より小さい。
  PaDiM は 13.542 / 14.312 ms で torch 26.985 ms の 0.50 倍 / 0.53 倍。
  PatchCore は 82.649 / 81.275 ms で torch 186.759 ms の 0.44 倍 / 0.44 倍。
- **threads=1 の P50**: 差が縮む。PaDiM は 43.312 / 52.022 ms で torch 59.838 ms の
  0.72 倍 / 0.87 倍。PatchCore は 558.090 / 545.028 ms で torch 691.930 ms の
  0.81 倍 / 0.79 倍。
- **スレッド 1→8 の P50 短縮（t1 / t8）**: PaDiM は torch 2.22 倍、onnxruntime 3.20 倍、
  openvino 3.63 倍。PatchCore は torch 3.70 倍、onnxruntime 6.75 倍、openvino 6.71 倍。
- **ピーク RSS（VmHWM）**: ONNX Runtime が最小。PaDiM は 379.16 MiB で
  torch 920.73 MiB の 0.41 倍、OpenVINO は 565.72 MiB で 0.61 倍。
  PatchCore は 806.1 MiB で torch 1287.31 MiB の 0.63 倍、OpenVINO は 1213.07 MiB で 0.94 倍。
  torch のピークは threads=8 と 1 でほぼ同一（PaDiM 920.73 / 920.61、PatchCore 1287.31 / 1287.11）。
- **ロード秒**: onnxruntime < openvino < torch。PaDiM は 0.084 / 0.185 / 3.839 s、
  PatchCore は 0.879 / 2.161 / 5.515 s。OpenVINO は ONNX Runtime の 2.20 倍（PaDiM）、
  2.46 倍（PatchCore）。
- **裾の重さ**: PatchCore torch threads=8 は P50 186.759 ms に対し P95 906.286 ms
  （P50 の 4.85 倍）、標準偏差 244.374 ms。同じ条件の onnxruntime は P95 100.484 ms、
  標準偏差 10.136 ms、openvino は P95 112.881 ms、標準偏差 14.136 ms。
- **スレッド数による差分の変化**: PaDiM の差分は threads=8 と 1 で同一
  （onnxruntime 0.000104 / 0.000259、openvino 0.000275 / 0.000824）。
  PatchCore は `map_max_abs_diff` が onnxruntime 0.000809 → 0.000674、
  openvino 0.001853 → 0.001794 と threads で異なる（`score_max_abs_diff` は同一）。

## 7. 計測上の注意

- **スレッド設定の意味はエンジンごとに異なる**。`torch.set_num_threads`、
  `intra_op_num_threads`、`INFERENCE_NUM_THREADS` は同じ数値でも内部の並列化戦略が違う。
  本記録の「threads=8 の比較」を同一条件の比較とみなさないこと。
- **共有ホストの揺らぎ**。計測ホストには本計測と無関係なプロセスが常駐し、run 開始時の
  load average（1 分）は 1.49〜7.46 だった。PatchCore torch threads=8 の P95（906.286 ms）が
  P50（186.759 ms）から大きく外れるのはこの揺らぎの影響が疑われるが、**原因は未確定**。
- **ピーク RSS の定義**。`process_peak_rss_mib` はワーカープロセスの
  `/proc/self/status` の VmHWM で、モデル／セッションのロードを含むプロセス全体のピーク。
  推論ループ中だけの値ではない。`latency_peak_rss_mib_sampled` は 50 ms 間隔の
  サンプリングで推論ループ中のみを対象とするため、短いピークは取り逃す
  （例: PaDiM torch threads=8 は sampled 822.84 MiB に対し VmHWM 920.73 MiB）。
- **VmHWM に切り替えた経緯**。初回計測では `resource.getrusage(RUSAGE_SELF).ru_maxrss` を
  使っており、3 エンジンとも `process_peak_rss_mib` が 1225.91 MiB の同一値になった。
  原因は Linux の `ru_maxrss` が fork 時の親プロセスの値を引き継ぐことで、親
  （compare_engines 本体）の RSS が下限として混入していた。VmHWM に切り替え、
  getrusage の値は `getrusage_maxrss_mib_inherits_parent` として参考に残している
  （比較には使わない）。
- **レイテンシの再計測**。初回の repeats 1 の計測は、計測中に別の検証用プロセスを
  並行実行してしまったためレイテンシを不採用とし、並行プロセス無しで再計測した値のみを
  本表に載せている。精度（差分・AUROC・判定一致）は初回と再計測で同じ値だった。
- **計測範囲**。model 推論のみ（入力テンソル → 出力 numpy）。decode・前処理・
  PostProcessor は含まない。前処理済み入力の構築時間は `engine_report.json` の
  `inputs` に記録されるが、本 CSV には含まれない。
- **`load_seconds` の対象はエンジンごとに異なる**（torch は checkpoint のロード、
  ONNX Runtime はセッション生成、OpenVINO は `read_model` と `compile_model`）。
  値の大小は同一処理の比較ではない。
- **表示**。CSV の値をそのまま小数 3 桁（相対差は CSV の表記のまま）で示している。
- **成果物は非コミット**。`inputs.npy` / `model.onnx` / `model.onnx.data` / `model.xml` /
  `model.bin` / `outputs_*.npz` / `engine_report.json` は `--output-dir` 配下
  （リポジトリ外）に置いている。コミットしたのは
  `benchmarks/issue15/engines_summary.csv` と `.md` のみ。

## 8. 再現コマンド

```bash
# 1) 学習と checkpoint 保存（PaDiM の例。PatchCore は --model patchcore に差し替え）
python scripts/run_anomalib.py \
  --protocol configs/protocols/mvtec_ad_v1.json \
  --dataset-root <MVTecAD> \
  --category bottle \
  --model padim \
  --image-size 256 \
  --cpu-threads 8 \
  --save-checkpoint <ckpt_dir>/padim \
  --output-dir <results>/train_padim

# 2) エンジン比較（threads=8、warmup 5、repeats 3 → 201 サンプル）
python scripts/compare_engines.py \
  --checkpoint <ckpt_dir>/padim/model.ckpt \
  --run-dir <run_dir> \
  --output-dir <out> \
  --engines torch,onnxruntime,openvino \
  --threads 8 --warmup 5 --repeats 3

# 3) threads=1（warmup 5、repeats 1 → 67 サンプル）
python scripts/compare_engines.py \
  --checkpoint <ckpt_dir>/padim/model.ckpt \
  --run-dir <run_dir> \
  --output-dir <out> \
  --engines torch,onnxruntime,openvino \
  --threads 1 --warmup 5 --repeats 1

# 4) 集計（run_id は実測時のもの。再実行時は読み替える）
python scripts/engines_summary.py \
  --reports <out>/20261031_213625 <out>/20261031_213700 \
            <out>/20261031_213859 <out>/20261031_213931 \
  --labels padim_t8 patchcore_t8 padim_t1 patchcore_t1 \
  --out-dir benchmarks/issue15
```

エンジン比較を使う時だけ、次をインストールする（requirements の変更は不要）:

```bash
pip install --no-cache-dir onnx onnxscript onnxruntime openvino
```

## 9. 未検証事項

- EfficientAD など他モデル。`model.model` が `pred_score` / `anomaly_map` 以外の出力を
  持つ場合の挙動、`load_from_checkpoint(..., map_location="cpu")` で imagenet 関連の
  前提無しに推論できるかは未検証。
- `bottle` 以外のカテゴリ、および MVTec-4K の 4K 入力（縮小率が大きく antialias の影響が
  出やすい条件）での比較。
- INT8 量子化・FP16（本計測は ONNX はそのまま、OpenVINO は
  `compress_to_fp16=False` の FP32 固定）。
- 動的バッチ・動的入力形状（本実装は (1, 3, H, W) 固定、動的軸なし）。
- OpenVINO の `PERFORMANCE_HINT=THROUGHPUT` など他のヒント設定、ストリーム数の変更。
- 反復回数を増やした信頼区間・外れ値の扱い（PatchCore torch threads=8 の P95 の要因を含む）。
- 専用（非共有）ホストでの再計測。本記録は共有ホストの値である。
- legacy exporter（`torch.onnx.export(..., dynamo=False)`）経路。全 run で dynamo が
  成功したため、実機では実行されていない。
- opset 17 での書き出し（dynamo 経由のダウンコンバートは `Pad` の 18→17 アダプタが無く失敗することを確認済み。legacy exporter で opset 17 を直接書き出す経路は未試行）。
- `pred_score` が 0 次元テンソルになるモデルでの書き出し可否。
- OpenVINO の出力ポート名取得（`get_any_name()`）が失敗した場合の出力順フォールバック経路。
- サンプリング間隔 50 ms でピーク RSS を十分に捉えられるか。
