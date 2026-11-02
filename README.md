# 異常検知アルゴリズムの実験・評価基盤

`anomaly-detection-playground` は、産業用の**画像異常検知（Visual Anomaly Detection）**を、精度・学習時間・推論速度・メモリ効率の観点で比較する実験リポジトリです。特に**CPUだけで学習・推論できるか**と、**高解像度画像上の微小欠陥**を重視します。

> [!IMPORTANT]
> **初期ベースラインです。** Anomalib 2.7.0 の PaDiM / PatchCore を呼び出し、CPUの学習・テスト工程の時間・ピークRSSと精度指標をJSONに保存します。Linux（8 vCPU、GPUなし）上で MVTec AD の bottle / grid カテゴリを用いた CPU のみの学習・テスト完走を確認済みです（[検証記録](docs/verification/issue5_cpu_baseline.md)）。Windows は未検証です。EfficientAD、画像1枚ごとの推論P50/P95、タイル分割、ONNX Runtime / OpenVINO比較は今後追加します。

## 評価データセット

| データセット | 目的 | 状態 |
| --- | --- | --- |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | 標準的な手法比較 | ベースライン実行対象 |
| [MVTec-4K](https://huggingface.co/datasets/XimiaoZhang/MVTec-4K) | 高解像度、微小異常、縮小耐性 | ベースライン実行対象 |
| [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) | 難条件での頑健性 | 今後追加 |
| [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/README.html) | 異なる画像分布での汎化性能 | 今後追加 |
| 利用許可済みの実撮影データ | 実用条件での精度と性能 | 今後追加（データは非公開） |

MVTec-4K は [HiAD](https://github.com/cnulab/HiAD) による高解像度派生データセットで、実際の4Kセンサー画像だけからなるデータセットではありません。**4Kの画像を256pxに縮小して得た結果は、4K入力での異常検知性能ではありません。** 処理前とモデル入力時の解像度、および縮小/タイル方式を必ず明記します。

## クイックスタート：CPUのみ

Python 3.11～3.13を想定しています（Windows/Linuxを想定、実機での依存関係の動作検証は未実施）。PyTorch・事前学習済み重み・データセットの初回取得には通信が必要です。

```bash
python -m venv .venv
# Windows PowerShellの場合: .venv\Scripts\Activate.ps1
# Linux/macOSの場合: source .venv/bin/activate
# CPU版PyTorchを先に入れる（CUDA版の巨大なホイールを避ける）
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -e .
# 計測から初回ダウンロードを除外するため、backbone重みを事前取得する
python scripts/run_anomalib.py --model padim --prefetch-weights
python scripts/run_anomalib.py --model patchcore --prefetch-weights
```

MVTec AD は[公式ダウンロードページ](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)でカテゴリ単位のアーカイブ（例：`bottle.tar.xz`、約150MB）を取得し、`datasets/MVTecAD/` 配下で展開してアーカイブを削除する方法でも配置できます（全カテゴリ一括のアーカイブは約4.9GB）。詳細は[検証記録](docs/verification/issue5_cpu_baseline.md)を参照してください。

MVTec ADは先にダウンロードします。Anomalibのデータセット取得機能を使う例：

```bash
python -c "from anomalib.data import MVTecAD; MVTecAD(root='datasets/MVTecAD', category='bottle').prepare_data()"
python scripts/run_anomalib.py --dataset mvtec_ad --category bottle --model padim --device cpu --image-size 256
python scripts/run_anomalib.py --dataset mvtec_ad --category bottle --model patchcore --device cpu --image-size 256
```

MVTec-4Kは[配布ページ](https://huggingface.co/datasets/XimiaoZhang/MVTec-4K/tree/main)から入手し、`datasets/MVTec-4K/<category>/train/good/`、`test/good/` 等を含む構成で配置してください。全体で約45.7GBあります。

```bash
# 4K画像を256pxに縮小した比較（「高解像度推論」ではない）
python scripts/run_anomalib.py --dataset mvtec_4k --category bottle --model padim --device cpu --image-size 256
# 入力解像度を増やした試験（CPUメモリ不足に注意）
python scripts/run_anomalib.py --dataset mvtec_4k --category bottle --model patchcore --device cpu --image-size 512
```

`--device cpu` を明示すると学習・評価の両方でCPUを使います。参考として `--device cuda` でも実行できます。`--cpu-threads 8`、`--num-workers 0`、`--train-batch-size 1`、`--eval-batch-size 1` 等で条件を固定できます。

## 標準プロトコル（再現性）

実験条件は `configs/protocols/*.json` で固定します。既定は `mvtec_ad_v1`（公式 test から seed=42 で 20% を閾値調整用 val に分け、残りで評価）です。論文値との比較には `--protocol configs/protocols/mvtec_ad_full_test.json`（test 全数、閾値は test で調整されるため F1 等は比較に使わない）を使います。CLI で明示した引数だけがプロトコルを上書きし、上書き内容は `report.json` の `protocol.overrides` に残ります。

```bash
# 学習せずにデータ分割だけを確認（results/<run-id>/manifest.json を保存）
python scripts/run_anomalib.py --category bottle --model padim --manifest-only
```

`report.json` の `data_split.manifest_sha256` が同じなら、train/val/test の画像構成は同一です。AUROC は Anomalib の `engine.test` の値に加え、正規化を通さない生スコアから計算した `raw_score_metrics` も記録します（`--skip-raw-eval` で省略）。

## 現状の出力

`results/<run-id>/report.json` に次の値を記録します。

- **精度**：Anomalib の `engine.test` が返したメトリクス（提供された指標のみ）
- **学習時間・テスト時間**：`train_stage_seconds` / `test_stage_seconds`。データ読み込み・検証を含む**工程全体**の時間であり、画像1枚の推論レイテンシではありません
- **画像1枚あたり推論レイテンシ**：`per_image_inference_latency`。バッチサイズ1で decode / preprocess / model / postprocess / end_to_end の工程別に P50・P95 等を記録します（ウォームアップは除外）。`--latency-warmup`、`--latency-repeats`、`--latency-max-images`、`--skip-latency` で調整できます
- **ホストメモリ**：`memory`。学習・テスト・レイテンシ計測の各工程で、親プロセスと子プロセスのRSSを約50ms間隔でサンプリングした最大値（短いピークは取り逃す可能性あり）と、プロセス生涯最大RSS（`ru_maxrss`）
- **実験条件**：OS、Anomalib/PyTorchバージョン、入力サイズ、モデル、データ分割、seed、バッチサイズ、CPUスレッド数、引数
- **処理方式**：現時点は `whole_image_resized_not_tiled`（全体を入力サイズに縮小、タイル処理なし）

## 開発ロードマップ

進捗は [ロードマップ親Issue #4](https://github.com/ada-koki/anomaly-detection-playground/issues/4) を起点に管理します。以下は実装順の目安であり、固定日程ではありません。

| Phase | 目的 | 実装Issue |
| --- | --- | --- |
| **1. ベースライン確立** | CPU実動作確認・画像単位の計測・再現性・CI | [#5](https://github.com/ada-koki/anomaly-detection-playground/issues/5) / [#6](https://github.com/ada-koki/anomaly-detection-playground/issues/6) / [#7](https://github.com/ada-koki/anomaly-detection-playground/issues/7) / [#8](https://github.com/ada-koki/anomaly-detection-playground/issues/8) |
| **2. アルゴリズム比較** | 共通条件の比較・EfficientAD・CPU推論モード | [#9](https://github.com/ada-koki/anomaly-detection-playground/issues/9) / [#10](https://github.com/ada-koki/anomaly-detection-playground/issues/10) / [#11](https://github.com/ada-koki/anomaly-detection-playground/issues/11) |
| **3. 高解像度・微小欠陥** | MVTec-4Kの検証・タイル分割・欠陥サイズ別評価 | [#12](https://github.com/ada-koki/anomaly-detection-playground/issues/12) / [#13](https://github.com/ada-koki/anomaly-detection-playground/issues/13) / [#14](https://github.com/ada-koki/anomaly-detection-playground/issues/14) |
| **4. 高速化・評価拡張** | ONNX Runtime / OpenVINO・MVTec AD 2 / VisA | [#15](https://github.com/ada-koki/anomaly-detection-playground/issues/15) / [#16](https://github.com/ada-koki/anomaly-detection-playground/issues/16) |

**最初に取り組むIssueは [#5](https://github.com/ada-koki/anomaly-detection-playground/issues/5)** です。実データでのAnomalibの学習・評価が完走した後に、他手法や高解像度の比較へ進みます。

評価指標・注意点は [評価プロトコル](docs/benchmark_protocol.md) を参照してください。`python -m unittest discover -s tests -v` で依存関係なしの基本テストが実行できます。

## CI とデータの取り扱い

[![CI](https://github.com/ada-koki/anomaly-detection-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/ada-koki/anomaly-detection-playground/actions/workflows/ci.yml)

GitHub Actions（`.github/workflows/ci.yml`）は push と pull_request で、Python 3.11 / 3.13 の matrix により、構文検査・`unittest`・`scripts/run_anomalib.py --help`・`scripts/check_repo_safety.py` を実行します。外部データセットや学習済み重みは取得せず、依存パッケージもインストールしません。MVTec AD など実データを使う学習・評価は CI では実行せず、手元で手動実行して結果の要約のみを `benchmarks/` と `docs/verification/` に記録します。データ・重みの権利とライセンス確認手順、コミット前チェック（`python scripts/check_repo_safety.py --staged`）は [docs/data_and_ci.md](docs/data_and_ci.md) を参照してください。

## 公開方針

本リポジトリで公開するのは、公開可能な技術情報と個人で作成・公開する権利を持つコードです。勤務先・顧客・第三者の非公開ソースコード、画像、モデル、業務要件、計測ログは含めません。データやモデルの公開・利用前には、利用許諾と権利帰属を確認します。

## ライセンス

このリポジトリのコードは [MIT](LICENSE) です。画像データおよび事前学習済み重みのライセンスは別です。MVTec AD は非商用の CC BY-NC-SA 4.0。MVTec-4K 配布ページには Apache-2.0 表記がありますが、派生元の権利・利用条件も確認してください。データと重みはGit管理しません。

参考：[Anomalib](https://github.com/open-edge-platform/anomalib) / [Anomalib公式ドキュメント](https://anomalib.readthedocs.io/en/latest/) / [HiAD](https://github.com/cnulab/HiAD)
