# Benchmark Protocol (draft v0.1)

## 目的

「異常が検出できるか」と「CPUで実用的な処理時間か」を分けて評価します。**CPU-only（学習・推論ともCPU）** と **CPU-inference（学習だけGPU可）** は別の結果として管理します。初版CLIが実行できるのは、前者と、学習・テストの両方をGPUにした参考比較です。

## アルゴリズム

| 手法 | 技術的特徴 | 現状 |
| --- | --- | --- |
| PaDiM | 位置ごとの特徴量分布とマハラノビス距離 | CLIあり |
| PatchCore | 特徴量メモリバンクと近傍探索 | CLIあり |
| EfficientAD | teacher/studentによる蒸留、ImageNette・教師重みの前提条件あり | 未実装 |
| FastFlow / Dinomalyなど | 追加比較候補 | 未実装 |

## データセットと比較条件

- **MVTec AD**：初期比較の基準。カテゴリごとの精度・速度・ばらつきを記録する。
- **MVTec-4K**：HiADの高解像度派生データ。tinyなどの欠陥タイプ別の評価を計画。元画像は4096pxでも、モデル入力サイズが256pxなら**縮小入力条件**に分類する。
- **MVTec AD 2 / VisA**：難条件・別分布での追加検証。正式な評価条件と公開/非公開テストの区別を守る。
- **5MP/12MP実画像**：欠陥の元画像上でのサイズ（px）、照明・位置ズレ条件を記録。顧客画像は公開しない。

1. モデル・データ分割・seedを固定する。初版は `ValSplitMode.FROM_TEST` でテストデータの20%を検証に回し、残りをテストする。**テスト集合で閾値調整しない。**
2. 学習用正常画像枚数、バックボーン/重み、バッチサイズ、CPUスレッド数、worker数、前処理と入力解像度を記録する。
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
| 推論 | P50/P95 ms/枚 | モデルのみと前後処理込みを分離（**未実装**） |
| メモリ | プロセスツリーRSS、モデル容量 | RSSはサンプリングによる推定 |
| 実行環境 | CPU only / CPU inference / GPU | PyTorch/ONNX Runtime/OpenVINO別に実施 |

目標は暫定的に「5MPで1秒/枚以下、12MPで5秒/枚以下、10px程度の欠陥を見逃さない処理を検討」とします。**達成した数値ではありません。** AUROCだけでは現場での可否を判断しません。

## 初期実装の限界

- `scripts/run_anomalib.py` は Anomalibの `fit`/`test` にかかった**工程全体の時間**だけを測ります。画像1枚のP95や前処理を除いた推論速度はまだありません。
- RSSの測定間隔は約100msで、短いピークは検出できないことがあります。
- MVTec-4KはMVTec ADに似たディレクトリ構成なので同じ `MVTecAD` データモジュールを試す段階です。ラベルと全カテゴリの整合性は未検証です。
- PaDiM/PatchCoreで入力4096pxを指定するとメモリ不足の可能性があります。**4Kのネイティブ入力対応は未保証です。**
- CPU学習とCPU推論の両方が動作したという検証結果はまだありません。実機確認後に表を追加します。

## 参考

- [Anomalib 2.7](https://github.com/open-edge-platform/anomalib)
- [MVTecAD DataModule](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/datamodules/image/mvtecad.html)
- [HiAD / MVTec-4K](https://github.com/cnulab/HiAD)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
