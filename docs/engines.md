# ONNX Runtime・OpenVINO による CPU 推論比較（Issue #15）

このドキュメントは、CPU 推論エンジン（PyTorch CPU / ONNX Runtime / OpenVINO）の比較を
同一入力・同一スレッド数・同一環境で行うための設計と CLI 仕様の暫定版である。
数値結果はここに書かず、実測後に別文書（`docs/verification/` 配下）に記録する。

## 目的

- 学習済みの Anomalib モデルを、PyTorch CPU 以外の推論エンジンでも実行できるか確認する。
- 書き出し可能なモデル・非対応演算・非対応エンジンの範囲を事前確認する（`--check-only`）。
- PyTorch CPU を基準として、推論結果の整合性（pred_score / anomaly_map の差、AUROC、判定一致率）と、
  P50/P95 レイテンシ・ピーク RSS を比較する。
- 変換モデルのサイズ・変換手順・再現コマンドを残す。

## 設計の要点

### 1) antialias 問題と前処理の共通化（最重要）

Anomalib 2.7 の `to_onnx` は Lightning モジュール全体を書き出すため、
`PreProcessor` の `export_transform`（= `get_exportable_transform`。`Resize` の
antialias を無効化）がグラフに含まれる。4K→256 のような大きな縮小では
antialias の有無で前処理結果が大きく変わり（Issue #12）、`engine.test` の精度指標と
別条件になる。

そこで本実装では:

- 書き出し対象は `model.model`（Anomalib の torch モデル本体）のみとし、PreProcessor / PostProcessor は含めない。
- 前処理は全エンジン共通で Python 側の `apply_pre_processor(pre_processor, x, "eval_transform")`
  （antialias 有効、`engine.test` と同じ変換）で行い、**前処理済みテンソル（1×3×H×W float32）**を
  各エンジンに渡す。
- PostProcessor（検証集合由来の正規化・閾値）は比較対象に含めない。生の `pred_score` /
  `anomaly_map` を比較する。画像レベルの判定一致率のみ、別途 `defect_size.thresholds.image`
  があれば計算する。

### 2) ONNX 書き出し

- `model.model` を eval で呼び、InferenceBatch（NamedTuple）から `pred_score` と
  `anomaly_map` のテンソルだけを返す `nn.Module` ラッパ（`make_outputs_only`）を生成する。
- 入力形状は固定 (1, 3, H, W)。動的軸は使わない。
- まず `torch.onnx.export(..., dynamo=True, opset_version=18)` を試し、失敗したら
  `dynamo=False`（legacy）を 1 回だけ試す。どちらで成功したか・失敗理由（例外型と
  メッセージ先頭）・出力名と形状・書き出し秒数・ファイルサイズを記録する。
- `onnx.load` でグラフを読み、`graph.node` の `op_type` を集計したヒストグラムを残す。
  非対応演算の有無を人間が確認できるようにするため。

### 3) エンジン

| エンジン | 実行方法 | スレッド | 備考 |
| --- | --- | --- | --- |
| `torch` | `model.model(x)`、`torch.inference_mode()`、`torch.set_num_threads(T)` | PyTorch のスレッド数 | 基準 |
| `onnxruntime` | `CPUExecutionProvider`、`SessionOptions.intra_op_num_threads=T`、`inter_op_num_threads=1`、`ORT_ENABLE_ALL`、`ORT_SEQUENTIAL` | intra_op | 未導入なら skipped |
| `openvino` | `ov.Core().read_model(onnx)` → `compile_model(m, "CPU", {"INFERENCE_NUM_THREADS": T, "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"})` | 推論スレッド数 | IR (.xml/.bin) も `save_model(..., compress_to_fp16=False)` で保存 |

`--threads` は全エンジン共通で与えるが、PyTorch の `set_num_threads`、ONNX Runtime の
`intra_op_num_threads`、OpenVINO の `INFERENCE_NUM_THREADS` は意味が異なる。
同じ数値でも内部の分割・スケジューリングは揃わないため、**同一条件での比較とみなさない**
こと（後述の「注意点」）。

### 4) プロセス分離

ピーク RSS を公平に測るため、エンジンごとの推論は別プロセス（`subprocess` で同スクリプトの
`--worker CONFIG.json` を起動）で実行する。親プロセスは書き出し（export）と集計のみを行う。

- ワーカー内で行うこと: モデル/セッション読み込み → 前処理済み入力の読み込み（mmap）→
  ウォームアップ → 計測 → 出力保存。
- ワーカー内で `adbench.memory.sampled_rss`（推論ループ中のサンプリング推定）と
  `adbench.memory.peak_rss_vmhwm_mib`（`/proc/self/status` の VmHWM。ワーカー自身のロード時を含むピーク）を使う。
  `resource.getrusage(RUSAGE_SELF).ru_maxrss` は Linux では fork 時の親プロセスの値を引き継ぐため、
  親（書き出し済みの compare_engines 本体）の RSS が下限として混入する。参考値として
  `getrusage_maxrss_mib_inherits_parent` に残すが、比較には使わない。
- opset: torch 2.15 の dynamo exporter は opset 18 未満に対応せず、17 を要求しても 18 で書き出した後の
  ダウンコンバートに失敗する（`Pad` の 18→17 アダプタが無い）。そのため既定を 18 とし、
  `opset_requested` と、ファイルの `opset_import` から読んだ実際の値 `opset` を分けて記録する。
- dynamo は重みを外部データ `model.onnx.data` に保存する。`file_size_bytes` は `model.onnx` と
  `model.onnx.data` の合計。

### 5) 共通入力の固定

- 親プロセスで test 画像を一度だけ前処理し、`inputs.npy`（N×3×H×W float32）として
  作業ディレクトリ（`--output-dir/<run_id>/` 配下、リポジトリ外）に保存する。
- ワーカーは `np.load(..., mmap_mode="r")` で読み、全エンジンがバイト一致の入力を使う。
- 前処理の時間は `inputs.npy` 構築時に 1 回だけ記録する。エンジン比較の対象は
  「model 推論のみ」（入力テンソル → 出力 numpy）の時間とし、decode / preprocess は含めない。
  torch の出力は `.numpy()` 変換まで含めて他エンジンと揃える。

### 6) 整合性の検証

各ワーカーは全 test 画像（`--max-images` があればその枚数）の `pred_score`（N）と
`anomaly_map`（N×H×W、float32）を `outputs_<engine>.npz` に保存する。親は torch を基準に、
エンジンごとに次を計算する:

- `pred_score` の max/mean 絶対差、最大相対差（`|a-b| / max(|b|, eps)`）。
- `anomaly_map` の max/mean 絶対差、最大相対差、相対差の 99.9 パーセンタイル。
- 画像 AUROC（`adbench.raw_eval.binary_auroc_from_values`、ラベルは manifest の `label_index`）。
- 画素 AUROC（`adbench.raw_eval.pixel_auroc_from_arrays`、マスクは元解像度から
  `resize_mask_nearest` で anomaly_map 解像度へ最近傍リサイズ）。
- 画像レベル判定一致率（`adbench.engines.decision_agreement`、閾値は親 run の
  `report.json` の `defect_size.thresholds.image`、無ければ `None`）。
- 同値スコア数（`count_tied_scores`）。

### 7) 入力となる学習済みモデル

- `scripts/run_anomalib.py --save-checkpoint DIR` が出力する `DIR/model.ckpt` を
  `--checkpoint` で受け取る。
- test 画像一覧・マスク・ラベルは同じ run の `manifest.json`（schema_version 1:
  `dataset_root`、`splits.test[]` に `image_path`（dataset_root 相対）、`label`、
  `label_index`、`mask_path`）から読む。`--run-dir` に run の出力ディレクトリ
  （`report.json` と `manifest.json` がある）を指定する。
- モデル種別と image_size は `report.json` の `model` と `image_size` を使う。
- ロードは `Padim/Patchcore/EfficientAd.load_from_checkpoint(path, map_location="cpu")`。
  EfficientAD は推論のみなら imagenet データ不要（未検証、後述）。

### 8) 計測条件

- `--threads`（既定 8）、`--warmup`（既定 5 枚）、`--repeats`（既定 1）、
  `--max-images`（既定 None = 全 test）。
- ウォームアップは集計から除外する。P50/P95 は `adbench.stats.summarize_latencies`。

## 依存の追加

エンジン比較を使う時だけ、次をインストールする（requirements の変更は不要）:

```bash
pip install --no-cache-dir onnx onnxscript onnxruntime openvino
```

- `torch` と `anomalib` は既存の環境を使う（CPU 版で良い）。
- `onnx` は演算子ヒストグラムに、`onnxscript` は `torch.onnx.export(dynamo=True)` に必要。
- 未導入のエンジンは `status=skipped` と理由を記録して続行する（torch は必須）。

## CLI の使い方

### 対応可否の事前確認（export のみ）

```bash
python scripts/compare_engines.py \
  --checkpoint /path/to/run/model.ckpt \
  --run-dir /path/to/run \
  --output-dir /path/to/engine-bench \
  --check-only
```

`--output-dir/<run_id>/engine_report.json` に `export` 欄（エンジンごとの status /
方法 dynamo|legacy / 所要秒 / サイズ / 失敗理由 / 出力名と形状 / op_histogram）が残る。
`model.onnx`、`model.xml/.bin`、`inputs.npy` は同じディレクトリに置かれ、コミットしない。

### 本計測（例: 全エンジン、スレッド 8、ウォームアップ 5 枚）

```bash
python scripts/compare_engines.py \
  --checkpoint /path/to/run/model.ckpt \
  --run-dir /path/to/run \
  --output-dir /path/to/engine-bench \
  --engines torch,onnxruntime,openvino \
  --threads 8 \
  --warmup 5 \
  --repeats 1 \
  --max-images 100
```

`--threads` を 1 に固定して 1 枚ずつの最悪ケースを見る、あるいは 8 でスループットを
見るなど、目的に応じて切り替える。結果ディレクトリはリポジトリ外に置くこと。

## 出力 JSON（`engine_report.json`）の主な項目

- `schema_version`: 本レポートの schema（現在 1）。
- `run_id`: `--output-dir` 配下のサブディレクトリ名（タイムスタンプ）。
- `source_run_dir`: 元になった `run_anomalib.py` の出力ディレクトリ。
- `checkpoint_sha256`: checkpoint の SHA-256。
- `model`: `report.json` から引き継いだモデル種別。
- `image_size`: 前処理後の想定解像度（`report.json` の `image_size` から）。
- `input_shape`: `inputs.npy` の形状（N×3×H×W）。
- `threads` / `warmup` / `repeats` / `max_images` / `opset`: 実行パラメータ。
- `engines_requested`: 指定されたエンジン一覧（`parse_engines` の結果）。
- `host` / `versions`: `adbench.report.collect_environment()` と各種ライブラリのバージョン。
- `inputs`: `inputs.npy` の構築情報（形状・前処理モード・所要秒）。
- `export`: エンジンごとの書き出し情報。
  - `status`: `ok` / `failed` / `skipped`。
  - `method`: `dynamo` / `legacy`（成功時）。
  - `seconds`: 書き出し秒数。
  - `file_size_bytes`: ONNX のサイズ。`onnx_size_bytes` も併記。
  - `ir_xml_path` / `ir_bin_path` / `ir_size_bytes`: OpenVINO IR のサイズ（openvino のみ）。
  - `output_names` / `output_shapes`: 出力名と形状。
  - `op_histogram`: `op_type → 件数`（件数降順）。
  - `error` / `attempts`: 失敗理由と試行ごとの記録。
- `engines`: エンジンごとの計測結果。
  - `status` / `load_seconds` / `model_ms`（count/mean/std/min/p50/p95/max）。
  - `latency_peak_rss_mib_sampled` / `process_peak_rss_mib` / `sampling_interval_s`。
  - `num_images` / `warmup_executed` / `repeats` / `latency_samples` / `output_shape`。
  - `returncode` / `wall_seconds`（親が worker プロセスから取得）。
- `reference`: torch の基準値（image/pixel AUROC、image_threshold、num_images）。
- `agreement`: エンジンごとの整合性。
  - `status` / `diffs`（pred_score / anomaly_map の差）。
  - `decision_agreement`: 画像レベル判定一致率（閾値が無い場合 null）。
  - `image_AUROC_raw` / `pixel_AUROC_raw`: エンジン側の AUROC（torch 側は `reference`）。
  - `tied_scores`: 同値スコア数。
- `notes`: 日本語の注意書き。

コミットするのはベンチマーク用の CSV/Markdown だけで、`inputs.npy` / `model.onnx` /
`model.xml` / `model.bin` / `outputs_*.npz` / `engine_report.json` は
`--output-dir` 配下に置き、リポジトリには入れない。

## 注意点

- **スレッド設定の意味がエンジンで異なる**。`torch.set_num_threads`、
  `intra_op_num_threads`、`INFERENCE_NUM_THREADS` は同じ数値でも内部の並列化
  戦略が異なる。数値の大小のみで「同じ設定」と解釈しないこと。
- **ピーク RSS はサンプリング推定**。`sampled_rss` は 50ms 間隔のサンプリングで、
  それより短いピークは取り逃す。`process_peak_rss_mib`（VmHWM）はワーカー全体（モデル読み込みを含む）のピークで、
  エンジン計測中だけの値ではない。
- **PostProcessor（検証集合由来の正規化・閾値）は比較外**。`pred_score` / `anomaly_map`
  は生スコア・生マップで比較する。AUROC はその生の値で計算する。
- **`--check-only` は export と op ヒストグラムのみ**で、エンジン計測はしない。
- **モデルの切り替え**: Padim / Patchcore / EfficientAd を想定しているが、
  EfficientAd は追加パラメータを持つため `load_from_checkpoint` の挙動を環境で確認する
  （未検証）。
- **メモリ**: `inputs.npy` は N×3×H×W×4 バイト。100 枚・256px なら約 78 MB。
  PatchCore はメモリバンクを持つため、`model.model` を読み込んだ状態で export すると
  一時的にメモリを多く使う。export は親で 1 回だけ行い、終わったら `del` する。
- **ONNX の非対応演算**: `--check-only` の `op_histogram` を見て、ONNX 化した演算子の
  種類を把握する。未対応演算があると onnxruntime/openvino のセッション生成で
  `status=failed` になる。失敗時は `error_type` / `error_message` が残る。
- **成果物の非コミット**: `inputs.npy` / `model.onnx` / `model.xml` / `model.bin` /
  `outputs_*.npz` / `engine_report.json` はすべて `--output-dir` 配下（リポジトリ外）
  に置くこと。

## 未検証事項

- `torch.onnx.export(..., dynamo=True)` と `output_names` / tuple 出力の組み合わせが、
  実行環境の torch / onnxscript のバージョンで期待どおり動くか。
- `torch.onnx.export(..., dynamo=False)`（legacy）が、実行環境の torch で利用可能か。
  両方失敗した場合に「非対応」と理由を残す経路は実装済みだが、実機で確認していない。
- Padim / Patchcore / EfficientAd 個別の `model.model` が InEferenceBatch として
  `pred_score` と `anomaly_map` を必ず返すか。EfficientAd は追加出力を持つ可能性がある
  （None 出力は除外する実装だが、実機での確認が必要）。
- EfficientAD を `load_from_checkpoint(..., map_location="cpu")` で推論専用に読み込んだ際、
  imagenet 関連の前提が無くても動くか。
- `pred_score` が 0 次元テンソル（shape `[]`）になるモデルで、ONNX 書き出しが成功するか。
- OpenVINO の `compiled.outputs` の出力ポートから `get_any_name()` で
  `pred_score` / `anomaly_map` を引けるか。引けない場合は出力順フォールバックを使う。
- スレッド数とレイテンシの関係が、エンジンごとに線形・非線形のどちらになるか。
- サンプリング間隔（50ms）でピーク RSS を十分に捉えられるか。
- `--max-images` で画像枚数を絞ったとき、AUROC の分散がどれだけ大きくなるか。
