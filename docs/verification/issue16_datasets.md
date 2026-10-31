# Issue #16 検証記録: VisA / MVTec AD 2 のデータセット対応（ビュー経由・256px・CPU）

- 対象 Issue: #16（VisA / MVTec AD 2 のデータセット対応）
- 結果 CSV: `benchmarks/issue16/datasets_summary.csv`（`tools` 側の集計スクリプトが各 run の `report.json` から生成）
- 本記録に載せる数値は上記 CSV の実測値のみ。CSV に無い値は「未取得」、確認していない事項は「未検証」と明記する。
- データ・マスク・重みはリポジトリに含めない。

## 1. 目的

既存のベンチマークは MVTec AD（`mvtec_ad_v1`）と MVTec-4K を対象にしてきた。Issue #16 では、公開データセットである **VisA** と **MVTec AD 2** を、MVTec AD と同じ評価プロトコル（`from_test` 0.2 / seed 42 / 256px / batch 1 / num_workers 0）で CPU 実行できるようにすることを目的とする（計測は `--cpu-threads 8`）。

両データセットは公式のディレクトリ配置・マスク値・分割定義が MVTec AD と異なるため、生配置をそのまま Anomalib に渡すと (a) レイアウト不一致で失敗する、(b) 意図しない自動ダウンロードが走る、(c) 1 カテゴリのみを展開した環境で分割処理が失敗する、といった問題が起きる。本 Issue では「Anomalib が読める MVTec 形式のビュー」を別途作り、そのビューを入力として既存の `scripts/run_anomalib.py` を流用する方針を採った。

## 2. データセットの公式形式・ライセンス・分割

| 項目 | VisA（candle） | MVTec AD 2（vial） |
| --- | --- | --- |
| ライセンス | CC BY 4.0（amazon-science/spot-diff の README に記載） | CC BY-NC-SA 4.0（配布物の `license.txt`、Copyright 2025 MVTec Software GmbH） |
| 引用 | 同上 README | Heckler-Kram et al., arXiv:2503.21622 |
| 生配置 | `<raw>/<cat>/Data/...`、`<raw>/split_csv/1cls.csv` | `<raw>/<cat>/{train,validation,test_public,test_private,test_private_mixed}` |
| 画像 | `.JPG`、1284x1168 | `.png`、グレースケール（L）、1400x1900 |
| マスク | 値 0/1 の L 画像 | 値 0/255、ファイル名 `<stem>_mask.png` |
| 分割定義 | `1cls`（`split_csv/1cls.csv`） | `train` / `validation` / `test_public` / `test_private` / `test_private_mixed` |
| 枚数（candle / vial） | train 正常 900、test 正常 100・異常 100（マスク 100） | train/good 291、validation/good 41、test_public good 35 / bad 105（マスク 105）、test_private 276、test_private_mixed 276 |
| 本検証で使用した枚数 | train 900、val 40（正常 20 / 異常 20）、test 160（正常 80 / 異常 80） | train/good 291、val 28（正常 7 / 異常 21）、test 112（正常 28 / 異常 84） |

- 入力は 256x256 への**全体縮小**（タイル分割なし、縦横比は保持しない）。
- MVTec AD 2 はグレースケール画像を 3ch に複製し、ImageNet 正規化を適用する（Anomalib の読み込み挙動に従う）。
- 閾値は検証集合（val）で調整する（report.json の `data_split.threshold_selection`）。AUROC は閾値に依存しない。

## 3. 公開・非公開 test の扱い

MVTec AD 2 は test が 3 種に分かれている。

- `test_public`: 正解（マスク）が公開されている。**本検証ではこれを使用**し、`from_test` 0.2 / seed 42 / label_aware で閾値調整用の val を抽出した。
- `test_private` / `test_private_mixed`: 正解が非公開のため**使用しない**（評価不能）。
- `validation`: 公式には正常画像のみ（vial は 41 枚）。本リポジトリは他データセットと条件を揃えるため**公式 validation は使わず**、`test_public` から異常を含む val を抽出する方式にした。この判断の妥当性は 9 章の未検証事項に含めた。

VisA には非公開 test は無く、`1cls.csv` の train/test 定義をそのまま使う。test の正常・異常から `from_test` 0.2 で val を抽出する。

## 4. MVTec AD（既存）との違い

| 観点 | MVTec AD | MVTec AD 2 | VisA |
| --- | --- | --- | --- |
| 画像 | `.png`（bottle はカラー） | グレースケール（L）`.png`、1400x1900 | `.JPG`、1284x1168（candle） |
| マスク値 | 0/255 | 0/255 | **0/1**（MVTec 形式の 0/255 に変換が必要） |
| マスク名 | `<stem>_mask.png` | `<stem>_mask.png` | 画像と同じ stem の `.png`（ビューでも `<stem>.png`） |
| 分割 | train / test / ground_truth | train / validation / test_public / test_private / test_private_mixed | train / test（1cls） |
| 照明条件 | なし | test_public に 7 種（overexposed / regular / shift_1〜4 / underexposed）をファイル名に含む | なし |

- AD2 の照明条件別の内訳は**未集計・未評価**。ファイル名に条件名が含まれることのみ確認した。
- 既存の参照点として、同じコードで MVTec AD bottle の PaDiM を実行した（CSV の `mvtecad_bottle_padim_256`）。

## 5. ビューの設計と理由

実装: `adbench/dataset_views.py`（標準ライブラリのみ。Pillow はマスク書き出し時に遅延 import）、`scripts/make_dataset_view.py`。

### 5.1 なぜビューが必要か（Anomalib 2.7.0 のソースで確認した事実）

- `Visa.apply_cls1_split` は `split_csv/1cls.csv` の**全カテゴリ**を処理するため、1 カテゴリだけを展開した環境では失敗する。
- `Visa.prepare_data` は `<root>/visa_pytorch/<cat>` も `<root>/<cat>` も存在しない場合に**自動ダウンロード**を実行する。
- `MVTecAD` は `.png` / `.PNG` のみを読み、ルート外を指すシンボリックリンクを除外する。
- `MVTecAD2` データモジュールは公式 validation（正常のみ）を使う設計であり、本リポジトリの規約（from_test 0.2 で閾値調整用 val を抽出）と一致しない。

### 5.2 VisA

- 自前で `<view>/visa_pytorch/<cat>/` を作り、`anomalib.data.Visa` に読ませる。これにより `apply_cls1_split` の全カテゴリ処理と自動ダウンロードの両方を回避する。
- 実行後、ビュー配下に新規ファイル（ダウンロード物）が生成されていないことを確認した。
- マスクは値 0/1 の L 画像のため、0 以外を 255 にした PNG として書き出す（MVTec 形式の想定 0/255 に合わせる）。画像は `.JPG` のまま（`VisaDataset` は `.png` / `.jpg` / `.JPG` を読む）。

### 5.3 MVTec AD 2

- `MVTecAD` データモジュールで読めるよう MVTec 形式のディレクトリ構成に写像する。`.png` のみを読みルート外シンボリックリンクを除外する仕様のため、リンクは**ハードリンク**にした。
- マスク `<stem>_mask.png` は `MVTecAD` の対応付け（ソート順＋stem 包含）で正しく対応し、実行が完走した。
- val は `test_public` から `from_test` 0.2（seed 42, label_aware）で抽出する。

### 5.4 共通

- 既定はハードリンク（追加ディスク消費はほぼ 0）。`--link-mode copy` も選択できる。
- `view_manifest.json` に件数・出典・除外ディレクトリを記録する。**絶対パスは記録しない**（環境依存情報を残さないため）。
- `validate_view` は生配置のまま渡された場合を検出し、ビュー作成コマンドを案内して停止する。これにより Anomalib の自動ダウンロードを防ぐ。
- `adbench/protocol.py` の許可データセットに `mvtec_ad2` / `visa` を追加し、`configs/protocols/visa_v1.json` と `configs/protocols/mvtec_ad2_v1.json` を追加した（`mvtec_ad_v1` と同条件: `from_test` 0.2 / seed 42 / 256px / PaDiM resnet18 / PatchCore wide_resnet50_2 coreset 0.1）。
- `scripts/run_anomalib.py --dataset {mvtec_ad,mvtec_4k,mvtec_ad2,visa}`。`visa` は `anomalib.data.Visa`、`mvtec_ad2` はビュー経由で `anomalib.data.MVTecAD` を使う。両者は実行前に `validate_view` で配置を検証し、問題があれば停止する。
- テスト: `tests/test_dataset_views.py` 28 件（Pillow が無い CI 環境では 12 件 skip を確認）。全体 333 件 OK。

## 6. 計測条件

- CPU 8 論理コア、RAM 16 GB（スワップなし）、GPU なし。
- Python 3.13.5、torch 2.15.0+cpu、torchvision 0.29.2+cpu、anomalib 2.7.0、lightning 2.6.6、timm 1.0.30、numpy 2.5.4。
- プロトコル共通条件: batch 1、num_workers 0、実行時 `--cpu-threads 8`、画像 256x256（CSV の `threads` 列は全行 8、`image_size` は全行 256x256）。
- 1 カテゴリずつ逐次実行（VisA candle、MVTec AD 2 vial）。
- **干渉**: 同一ホストで本計測と無関係の別プロセス（nice 19、2 スレッド）が断続的に動作していた。`phase=accuracy` の run はその終了を待たずに実行し、実行中に別プロセスが存在したかを 2 秒間隔で監視して `other_process_interference` に記録した。AUROC は計算内容が CPU 負荷に依存しないため採用するが、**学習・テスト秒とレイテンシは干渉ありの値として参考扱い**とする。

## 7. 結果

表示は CSV の値そのまま（AUROC 6 桁、秒 2 桁、ms 1 桁、MiB 2 桁）。

### 表 1: 精度

| 条件 | phase | 干渉 | test 枚数 | 画像 AUROC (engine) | 画像 AUROC (raw) | 画素 AUROC (engine) | 画素 AUROC (raw) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| visa_candle_padim_256 | accuracy | yes | 160 | 0.878828 | 0.87875 | 0.979952 | 0.979997 |
| visa_candle_patchcore_256 | accuracy | yes | — | OOM Kill（exit 137）のため未取得 | — | — | — |
| ad2_vial_padim_256 | accuracy | yes | 112 | 0.631378 | 0.629252 | 0.846444 | 0.846682 |
| ad2_vial_patchcore_256 | accuracy | yes | 112 | 0.857993 | 0.857993 | 0.94292 | 0.942598 |
| mvtecad_bottle_padim_256 | accuracy | yes | 67 | 0.996324 | 0.996324 | 0.97676 | 0.976871 |

### 表 2: 時間・レイテンシ・メモリ（すべて干渉あり・参考値）

| 条件 | 学習 s | テスト s | model P50 ms | model P95 ms | e2e P50 ms | e2e P95 ms | 学習ピーク RSS MiB | テストピーク RSS MiB | レイテンシピーク RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visa_candle_padim_256 | 128.24 | 32.79 | 54.1 | 307.9 | 81.5 | 372.9 | 3377.44 | 1205.52 | 1148.75 |
| visa_candle_patchcore_256 | — | — | — | — | — | — | — | — | — |
| ad2_vial_padim_256 | 32.98 | 8.80 | 30.2 | 40.2 | 55.6 | 64.1 | 2008.66 | 1137.75 | 872.18 |
| ad2_vial_patchcore_256 | 1895.38 | 35.48 | 243.9 | 517.1 | 281.0 | 583.7 | 4194.06 | 1852.69 | 1855.59 |
| mvtecad_bottle_padim_256 | 16.40 | 6.68 | 80.3 | 825.9 | 123.2 | 937.4 | 1562.87 | 948.86 | 797.42 |

- AD2 vial PatchCore の `train_stage_s` 1895.38 s には coreset 選択（29,797 点）の時間を含む。
- VisA candle PatchCore の行は全フィールド空（OOM Kill）。

### 表 3: run_id（追跡用）

| 条件 | run_id |
| --- | --- |
| visa_candle_padim_256 | 20261031T132811Z-01d9c417 |
| ad2_vial_padim_256 | 20261031T134248Z-27de2a6f |
| ad2_vial_patchcore_256 | 20261031T134352Z-de865b08 |
| mvtecad_bottle_padim_256 | 20261031T141724Z-a62b77c8 |

## 8. 観察（CSV から確認できる事実のみ）

- **AD2 vial は 2 モデルとも画像 AUROC が相対的に低い**。AD2 vial の画像 AUROC は PaDiM 0.631378、PatchCore 0.857993。同一データセット・同一カテゴリ・同一画像サイズでの差は画像 AUROC で +0.226615、画素 AUROC で +0.096476（0.846444 → 0.94292）。
- 参照の MVTec AD bottle PaDiM は画像 0.996324 / 画素 0.97676。AD2 vial PaDiM の画像 AUROC はこれより 0.364946 低い（0.996324 − 0.631378）。ただしデータセット・カテゴリが異なるため直接比較ではない。
- 画像 AUROC の最大は `mvtecad_bottle_padim_256` の 0.996324、最小は `ad2_vial_padim_256` の 0.631378。画素 AUROC の最大は `visa_candle_padim_256` の 0.979952。
- **engine と raw の差は小さい**。画像 AUROC の差は最大 0.002126（AD2 vial PaDiM: 0.631378 vs 0.629252）、他は 0.000078 以下または 0。画素 AUROC の差は最大 0.000322（AD2 vial PatchCore: 0.94292 vs 0.942598）。
- **学習段階の秒数**は PatchCore（AD2 vial）が 1895.38 s で最大。PaDiM は 16.40〜128.24 s。学習ピーク RSS の最大も AD2 vial PatchCore の 4194.06 MiB。
- **レイテンシの裾**は条件によって大きい。`mvtecad_bottle_padim_256` は model P50 80.3 ms に対し P95 825.9 ms（P95/P50 ≒ 10.3 倍）。一方 `ad2_vial_padim_256` は 30.2 → 40.2 ms（≒ 1.3 倍）。ただし全行が干渉ありのため、モデル自体の分布の裾なのか干渉由来なのかは本 CSV では区別できない（**未検証**）。
- e2e と model の差（P95）は 23.9〜111.5 ms（`ad2_vial_padim_256` 64.1 − 40.2 = 23.9、`mvtecad_bottle_padim_256` 937.4 − 825.9 = 111.5）。前後処理と画像読み込みを含む差だが、内訳は本 CSV では分離できない。

## 9. 計測上の注意

1. **干渉の扱い**: 本 CSV の全行が `other_process_interference=yes`。規約上、時間・レイテンシの採用値は `phase=latency_remeasure_tryN` かつ干渉なしの行に限るが、**本 CSV には該当行が無い**。したがって「干渉なしのレイテンシは未取得」である。AUROC のみ採用値として扱う。
2. **VisA candle PatchCore の OOM Kill**: 学習中に exit 137 で強制終了（カーネルログで global OOM、anon-rss 約 9.1 GiB = 9,501,928 kB 時点）。Anomalib 2.7 の `PatchcoreModel` は全パッチ特徴を list に溜め、`torch.vstack` で結合してから coreset 選択する。**仮説（未検証）**: 900 枚 × 1024 パッチ × 1536 次元 × 4 バイト ≈ 5.7 GB の特徴が結合時に二重に存在し、16 GB ホストでは収まらない。VisA candle PatchCore の精度は未取得。
3. **縮小と縦横比**: 両データセットとも 256x256 への全体縮小で、縦横比は保持しない（タイル分割なし）。AD2 は 1400x1900 → 256x256 のため、細い欠陥は縮小で消失し得る。この影響は本記録では評価していない（未検証）。
4. **1 カテゴリずつの実行**: 本記録の結果は VisA candle と AD2 vial の各 1 カテゴリのみ。他カテゴリの傾向は未検証。
5. **AD2 の val 抽出**: 公式 validation ではなく `test_public` からの `from_test` 0.2 抽出を使っている。この違いが閾値と F1 に与える影響は本記録では評価していない（AUROC は閾値非依存のため影響しない）。
6. **非公開 test**: `test_private` / `test_private_mixed` は正解非公開のため使用していない。
7. **データ・重みの非混入**: データ・マスク・重みはリポジトリに含めない。ビューの `view_manifest.json` には絶対パスを記録しない。

## 10. 再現コマンド

```
python scripts/make_dataset_view.py --dataset visa --raw-root <VisA_raw> --category candle --out <VisA_view>
python scripts/make_dataset_view.py --dataset mvtec_ad2 --raw-root <MVTecAD2_raw> --category vial --out <MVTecAD2_view>
python scripts/run_anomalib.py --protocol configs/protocols/visa_v1.json --dataset visa --dataset-root <VisA_view> --category candle --model padim --device cpu --cpu-threads 8 --image-size 256 --output-dir <results>/visa_candle_padim_256
python scripts/run_anomalib.py --protocol configs/protocols/mvtec_ad2_v1.json --dataset mvtec_ad2 --dataset-root <MVTecAD2_view> --category vial --model patchcore --device cpu --cpu-threads 8 --image-size 256 --output-dir <results>/ad2_vial_patchcore_256
```

- VisA の生配置: `<VisA_raw>/candle/Data/...` と `<VisA_raw>/split_csv/1cls.csv`。
- MVTec AD 2 の生配置: `<MVTecAD2_raw>/vial/{train,validation,test_public,...}`。
- 結果ディレクトリ（`<results>`）はリポジトリ外に置く。コミットするのは `benchmarks/` 配下の CSV / MD のみ。
- MVTec AD bottle（参照行）はドキュメント記載の `mvtec_ad_v1` 手順で同じ `run_anomalib.py` から実行した。

## 11. 未検証事項

- VisA の candle 以外のカテゴリ、MVTec AD 2 の vial 以外のカテゴリの精度・時間・メモリ。
- MVTec AD 2 の照明条件（overexposed / regular / shift_1〜4 / underexposed）別の精度。内訳は未集計・未評価。
- MVTec AD 2 の公式 validation（vial は正常 41 枚）を閾値調整に使う条件との比較。
- `test_private` / `test_private_mixed` を用いる評価（正解非公開のため本リポジトリでは対象外）。
- VisA candle PatchCore の精度、および OOM の原因仮説の検証とメモリ対策（coreset 比率・特徴の逐次蓄積など）。
- EfficientAD の VisA / MVTec AD 2 での実行。
- タイル分割・高解像度入力（256px 全体縮小以外）での両データセットの評価。
- Windows 環境での動作。
- ONNX / OpenVINO 変換後の推論での両データセットの評価。
