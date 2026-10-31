# 異常検知ベンチマークの評価方針（暫定版 v0.2）

## 目的

「異常が検出できるか」と「CPUで実用的な処理時間か」を分けて評価します。**CPUのみ（学習・推論ともCPU）** と **CPU推論（学習のみGPU利用可）** は別の結果として管理します。初版CLIが実行できるのは、前者と、学習・テストの両方をGPUにした参考比較です。

## アルゴリズム

| 手法 | 技術的特徴 | 現状 |
| --- | --- | --- |
| PaDiM | 位置ごとの特徴量分布とマハラノビス距離 | CLIあり |
| PatchCore | 特徴量メモリバンクと近傍探索 | CLIあり |
| EfficientAD | 教師・生徒モデルによる蒸留、ImageNette・教師重みの前提条件あり | 未実装 |
| FastFlow / Dinomalyなど | 追加比較候補 | 未実装 |

## データセットと比較条件

- **MVTec AD**：初期比較の基準。カテゴリごとの精度・速度・ばらつきを記録する。
- **MVTec-4K**：HiADの高解像度派生データ。tiny（微小欠陥）などの欠陥タイプ別の評価を計画。元画像は4096pxでも、モデル入力サイズが256pxなら**縮小入力条件**に分類する。
- **MVTec AD 2 / VisA**：難条件・別分布での追加検証。正式な評価条件と公開/非公開テストの区別を守る。
- **利用許可済みの実撮影画像**：欠陥の元画像上でのサイズ（px）、照明・位置ズレ条件を記録。自作または明示的に利用許可を得たデータのみ使用し、第三者の非公開データはGit管理しない。

1. モデル・データ分割・乱数シードを固定する。初版は `ValSplitMode.FROM_TEST` でテストデータの20%を検証に回し、残りをテストする。**テスト集合で閾値調整しない。**
2. 学習用正常画像枚数、バックボーン/重み、バッチサイズ、CPUスレッド数、データ読み込みワーカー数、前処理と入力解像度を記録する。
3. モデル・データセットの初回ダウンロードを計測に含めない。ウォームアップを統一する。
4. 全体縮小とタイル分割を別条件として比較する。タイル方式にはサイズ・オーバーラップ・合成方式を記録する。
5. 速度は原則3回以上計測し、中央値とばらつきを記録する。

## 標準プロトコル

標準プロトコルは `configs/protocols/` のJSON設定ファイルで定義します。実行時は `--protocol <path>` で読み込み、CLIで明示指定した項目だけを上書きします。上書きした項目はレポートの `protocol.overrides` に残します。

- `configs/protocols/mvtec_ad_v1.json`：既定の標準プロトコル。
- `configs/protocols/mvtec_ad_full_test.json`：公式test全数で評価する比較用プロトコル。`val_split_mode` は `same_as_test`。

設定にはプロトコル名・版、データセット、検証集合の作り方と割合、乱数シード、入力解像度、学習・評価バッチサイズ、データ読み込みワーカー数、CPUスレッド数、モデル別パラメータ（backbone、layers、PatchCoreのcoresetサンプリング比率・近傍数など）を保存します。設定ファイル自体のSHA-256もレポートに記録します。

### 役割

- **train**：データセットの公式train/good全数を使います。
- **val**：標準プロトコルでは公式testから、設定されたシードと `val_split_ratio` に従って `label_aware` に抽出します。valは閾値調整専用であり、AUROCなどの閾値非依存指標の評価には使いません。
- **test**：公式testのうち、valに抽出されなかった残りを使います。

`mvtec_ad_full_test.json` では `val_split_mode: "same_as_test"` を指定します。AnomalibのPostProcessorは検証集合から正規化パラメータ（`image_min` / `image_max`）と閾値を求めるため、検証集合を空にはしません。`same_as_test` ではvalがテスト集合の複製になるので、公式test全数が評価対象になります。このプロトコルは論文値などとの比較用であり、閾値依存指標（F1など）は**テスト集合で調整済み（楽観的）** であるため、他プロトコルの値と比較しません。

### `val_split_mode: none` を拒否する理由

Anomalib 2.7.0 のPostProcessorは、検証集合から画像スコアの最小値・最大値と閾値を推定し、生スコアを `((pred - threshold) / (max - min) + 0.5).clamp(0, 1)` で正規化します。`val_split_mode: none`（Engineでは `limit_val_batches=0`）では検証集合が空になり、これらの値がNaNのまま正規化されません。

実機検証（Anomalib 2.7.0、PaDiM、MVTec AD bottle、256px）では、この条件の `engine.test` の image AUROC が実行ごとに変動し、同じ学習済みモデルについて生スコアから計算した image AUROC と大きく食い違う現象が確認されました（Issues #7）。原因は検証集合に依存する正規化・閾値推定にあると考えられます。数値は実行環境依存のため本ドキュメントには記載しません。

そのため、CLI（`--val-split-mode none`）とプロトコル検証の両方で `none` をエラーとして拒否し、エラーメッセージに理由を表示します。`same_as_test` または `from_test` を指定してください。

### val/test の完全一致のみ許容する重複検出

`same_as_test` ではvalとtestが同じ画像集合になります。manifestの重複検出は、valとtestの画像集合が完全一致し、各画像がどちらの役割にも1回だけ現れる場合に限り、この重複を許容します。許容した場合はレポートに次を記録します。

- `data_split.overlap_detected: true`
- `data_split.overlap_allowed: true`
- `data_split.overlap_image_count`：重複した画像数
- `data_split.overlap_reason`：許容した理由（`same_as_test` によるvalの複製であること、閾値依存指標が楽観的になること）

それ以外の重複（同一役割内の重複、trainとtestの重複、valとtestの部分的な重複など）は従来どおり実行エラーです。

### 件数と同一条件の追跡

実行後の `report.json` には次を記録します。

- `protocol`：プロトコル名・版、設定全体、設定ファイルのSHA-256、CLI上書き項目。
- `data_split.roles`：train/val/testごとの件数とlabel別件数。
- `data_split.overlap_detected` / `data_split.overlap_allowed` / `data_split.overlap_image_count`：役割間・同一役割内の画像パス重複の有無。既定では重複があれば実行は失敗します。`val_split_mode: same_as_test` でvalとtestが完全一致する場合のみ `overlap_allowed: true` として許容し、`overlap_reason` に理由を記録します。
- `data_split.manifest_sha256`：役割・相対パス・label・label_index・mask相対パスを正規化してソートした一覧のSHA-256。
- `data_split.threshold_selection`：閾値調整に使う集合と注意事項。`same_as_test` では `optimistic: true` と `threshold_dependent_metrics: "tuned_on_test_optimistic"` を記録します。
- `randomness`：seed、`seed_everything` と `torch.manual_seed` の実行有無、`torch.are_deterministic_algorithms_enabled()`、`PYTHONHASHSEED` の有無。

`results/<run_id>/manifest.json` には役割ごとの画像一覧全体を保存します。レポート本文には件数・重複有無・SHA-256などの要約だけを載せます。

`--manifest-only` を指定すると、学習せずにDataModuleのsetupとマニフェスト作成・保存だけを行います。同じプロトコル・シードで再実行したときは、`protocol.sha256` と `manifest_sha256` を比較して対象画像と設定が一致したか確認できます。

### 閾値の扱い

- 標準プロトコルでは、閾値はvalだけで調整し、testでは調整しません。
- AUROCなど閾値非依存の指標はtestで評価します。
- `mvtec_ad_full_test.json`（`same_as_test`）の結果は、閾値依存指標がテスト集合で調整済み（楽観的）であるため、他プロトコルと比較する用途に使いません。
- `val_split_mode: none` は、PostProcessorの正規化パラメータがNaNになり `engine.test` のメトリクスが信頼できないため使用しません。
- 異常画像を閾値調整に使う場合でも、その画像を同じ実行のtest評価に含めません。標準プロトコルではvalとtestが排他になるよう重複検出で確認します。

### 層化の制限

Anomalibの `ValSplitMode.FROM_TEST` は `label_aware=True` で分割しますが、これは `label_index`（正常・異常）単位の層化です。`broken_large` や `broken_small` などの異常種別ディレクトリ単位で層化されるわけではありません。異常種別ごとの件数比較や、特定の異常種別だけを狙った評価では、manifestのlabel別件数を確認し、必要なら別プロトコルを定義します。

## 指標

| 分野 | 指標 | 注意点 |
| --- | --- | --- |
| 精度 | Image AUROC, Pixel AUROC, AUPRO（利用可能な場合） | 閾値非依存のベースライン。`raw_score_metrics`（生スコア）と `test_metrics`（engine.test）の両方を記録し、差がある場合はrawを正とする |
| 現場精度 | FNR（欠陥の見逃し率）、FPR（正常の過検出率） | 閾値は検証データだけで確定 |
| 微小欠陥 | 欠陥サイズ/種類別の検出率 | 元画像座標のサイズを保持 |
| 学習 | 経過時間、ピークRSS | CPU-onlyで学習成立するか |
| 推論 | P50/P95 ms/枚、平均、件数、段別内訳 | decode / preprocess / model / postprocess / end_to_end を分離。ウォームアップは集計から除外 |
| メモリ | プロセスツリーRSS、生涯最大RSS、モデル容量 | RSSはサンプリングによる推定。短いピークは取り逃す。生涯最大RSSは工程別ではない |
| 実行環境 | 学習・推論ともCPU / 推論のみCPU / 学習・推論ともGPU | PyTorch / ONNX Runtime / OpenVINOごとに実施 |

本リポジトリでは、**高解像度画像における微小欠陥の検出精度とCPU推論性能を比較する**ことを目的とします。性能の合否基準はデータセット・実行環境・用途に応じて設定し、実測値と未検証の目標を区別します。AUROCだけでは実用性を判断しません。

### 生スコア評価（`raw_score_metrics`）

`engine.test` のメトリクスは、PostProcessorが検証集合から求めた正規化パラメータと閾値を経由します。同じ学習済みモデルでも正規化の影響で値が変わり得るため、`scripts/run_anomalib.py` はtest後に生スコア評価を実行し、`report.json` の `raw_score_metrics` に保存します（`adbench/raw_eval.py`）。

- test集合の全画像を1枚ずつ（`read_image` → `pre_processor` → `model.model`、`torch.inference_mode()`）処理し、PostProcessorを通さない生の `pred_score` と `anomaly_map` を得ます。
- `image_AUROC_raw`：gt_label（`label_index`）に対するimage AUROC。
- `pixel_AUROC_raw`：mask（`anomaly_map` の空間サイズへ最近傍リサイズ、正常画像はゼロマスク）に対するpixel AUROC。メモリ節約のため、全画像分のuint8マスクとfloat32マップを連結して1回だけ計算します。
- `image_count` / `pixel_count` / `positive_pixel_count`：処理件数。想定（256px×100枚相当）を超えた場合はエラーにせず、件数を `notes` に記録します。
- `tied_image_scores`：同値のimageスコア件数（丸め・飽和の検出用）。
- `notes`：計算条件と注意事項。

`--skip-raw-eval` で省略できます。省略・失敗時はその旨を `raw_score_metrics` に記録します。

**AUROCの主指標は `raw_score_metrics` と `engine.test`（`test_metrics`）の両方を記録し、両者に差がある場合は `raw_score_metrics` を正とします。** 正規化は検証集合のスコア範囲に依存し、範囲外のスコアが0/1に飽和すると同値が生じてAUROCが変化し得るためです。

## 画像1枚あたりの推論レイテンシ計測

`scripts/run_anomalib.py` は `engine.fit` / `engine.test` の工程全体とは別に、テスト画像を使った画像1枚単位の推論レイテンシを計測します。

- モデルは `eval()` にし、`torch.inference_mode()` で実行する。
- バッチサイズは1とする。
- ウォームアップ画像を指定回数だけ先に実行し、その時間は集計から除外する。実行したウォームアップ件数は結果に記録する。
- 計測対象画像は `--latency-max-images` で制限できる。`--latency-repeats` により同じ画像集合を複数回繰り返し、すべての計測サンプルを集計に含める。
- `--skip-latency` を指定した場合は画像単位の計測を行わない。

各段の定義は次のとおりです。

| キー | 範囲 |
| --- | --- |
| `decode_ms` | 画像ファイルの読み込みとテンソル化（`read_image`） |
| `preprocess_ms` | バッチ次元の付与と `pre_processor` の実行 |
| `model_ms` | `model.model` の実行 |
| `postprocess_ms` | `post_processor` の実行 |
| `end_to_end_ms` | `decode_ms` から `postprocess_ms` までの合計 |

各段について件数・平均・標準偏差・最小値・P50・P95・最大値を `per_image_inference_latency` に保存します。計測時の `torch.get_num_threads()`、バッチサイズ、入力解像度、計測中のサンプリングRSSピークも記録します。

## メモリ計測の方法と限界

- 学習・テスト・レイテンシ計測中のRSSは、`psutil` で自プロセスと子プロセスのRSS合計を一定間隔でサンプリングした推定値です。**サンプリング間隔より短いピークは取り逃す可能性があります。**
- `process_lifetime_peak_rss_mib` は `resource.getrusage(RUSAGE_SELF).ru_maxrss` に基づく自プロセスの生涯最大RSSです。学習・テスト・推論などの工程別の値ではなく、プロセス全体の最大値です。Windowsなど取得できない環境では `null` になります。
- `num_workers=0` の場合、通常はデータ読み込み用の子プロセスは発生しません。`num_workers` を増やすと子プロセスが生じ、RSSの解釈が変わります。
- サンプリング値は短時間の計測ではばらつくため、繰り返し計測や余裕を見た解釈が必要です。
- 生スコア評価は test 画像を再度1枚ずつ推論するため、`raw_eval_peak_rss_mib_sampled` を別途記録します。

## 初期実装の限界

- `test_stage_seconds` は `engine.test` 全体（テストデータ読み込み・評価指標計算を含む）の時間です。**画像1枚あたりの推論時間ではありません。** 画像1枚の値は `per_image_inference_latency` を参照してください。
- 画像1枚あたりの計測は、1プロセス・バッチサイズ1・指定スレッド数での値です。前処理や後処理の実装、入力解像度、モデルによって結果は変わります。
- MVTec-4K は生配置に `foreground/` があり、Anomalib の `MVTecAD` にそのまま渡せません。`scripts/validate_mvtec4k.py --make-view` で作る clean view を使います（`docs/mvtec4k.md`）。整合検証は bottle のみ実施済みで、他の 6 カテゴリは未検証です。
- PaDiM/PatchCoreで入力4096pxを指定するとメモリ不足の可能性があります。**4Kのネイティブ入力対応は未保証です。**
- CPU での `fit` / `test` は PaDiM（bottle, grid）と PatchCore（bottle）で実機完走を確認済みです（256px, `docs/verification/` 参照）。その他のカテゴリ・4K条件での CPU 実行は未検証です。
- `raw_score_metrics` は PostProcessor を通さない検証用の値です。実機確認は PaDiM・MVTec AD bottle・256px の条件に限られ、他のカテゴリ・モデル・解像度では未検証です。
- `val_split_mode: none` の拒否と `same_as_test` の重複許容は、コード上のテストで確認していますが、`PatchCore` など他モデルでの `same_as_test` 実行は未検証です。

## 参考資料

- [Anomalib 2.7](https://github.com/open-edge-platform/anomalib)
- [MVTec ADのデータ読み込み機能](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/datamodules/image/mvtecad.html)
- [HiAD / MVTec-4K](https://github.com/cnulab/HiAD)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
