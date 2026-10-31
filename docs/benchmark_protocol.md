# 異常検知ベンチマークの評価方針（暫定版 v0.1）

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

## 指標

| 分野 | 指標 | 注意点 |
| --- | --- | --- |
| 精度 | Image AUROC, Pixel AUROC, AUPRO（利用可能な場合） | 閾値非依存のベースライン |
| 現場精度 | FNR（欠陥の見逃し率）、FPR（正常の過検出率） | 閾値は検証データだけで確定 |
| 微小欠陥 | 欠陥サイズ/種類別の検出率 | 元画像座標のサイズを保持 |
| 学習 | 経過時間、ピークRSS | CPU-onlyで学習成立するか |
| 推論 | P50/P95 ms/枚、平均、件数、段別内訳 | decode / preprocess / model / postprocess / end_to_end を分離。ウォームアップは集計から除外 |
| メモリ | プロセスツリーRSS、生涯最大RSS、モデル容量 | RSSはサンプリングによる推定。短いピークは取り逃す。生涯最大RSSは工程別ではない |
| 実行環境 | 学習・推論ともCPU / 推論のみCPU / 学習・推論ともGPU | PyTorch / ONNX Runtime / OpenVINOごとに実施 |

本リポジトリでは、**高解像度画像における微小欠陥の検出精度とCPU推論性能を比較する**ことを目的とします。性能の合否基準はデータセット・実行環境・用途に応じて設定し、実測値と未検証の目標を区別します。AUROCだけでは実用性を判断しません。

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

## 初期実装の限界

- `test_stage_seconds` は `engine.test` 全体（テストデータ読み込み・評価指標計算を含む）の時間です。**画像1枚あたりの推論時間ではありません。** 画像1枚の値は `per_image_inference_latency` を参照してください。
- 画像1枚あたりの計測は、1プロセス・バッチサイズ1・指定スレッド数での値です。前処理や後処理の実装、入力解像度、モデルによって結果は変わります。
- MVTec-4KはMVTec ADに似たディレクトリ構成なので同じ `MVTecAD` データモジュールを試す段階です。ラベルと全カテゴリの整合性は未検証です。
- PaDiM/PatchCoreで入力4096pxを指定するとメモリ不足の可能性があります。**4Kのネイティブ入力対応は未保証です。**
- CPU での `fit` / `test` は PaDiM（bottle, grid）と PatchCore（bottle）で実機完走を確認済みです（256px, `docs/verification/` 参照）。その他のカテゴリ・4K条件での CPU 実行は未検証です。

## 参考資料

- [Anomalib 2.7](https://github.com/open-edge-platform/anomalib)
- [MVTec ADのデータ読み込み機能](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/datamodules/image/mvtecad.html)
- [HiAD / MVTec-4K](https://github.com/cnulab/HiAD)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
