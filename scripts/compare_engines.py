#!/usr/bin/env python3
"""ONNX Runtime・OpenVINO による CPU 推論比較 CLI（Issue #15）。

親プロセスは inputs.npy の生成・ONNX export・集計のみを行い、エンジンごとの
推論計測は別プロセス（``--worker``）で実行する。``--check-only`` で export のみ
実施し、ONNX グラフの演算子ヒストグラムを記録して終了できる。

``--help`` は torch 無しで動く。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adbench import engines as engines_mod  # noqa: E402
from adbench.memory import DEFAULT_SAMPLING_INTERVAL_S  # noqa: E402
from adbench.report import collect_environment, json_safe  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_engines",
        description="ONNX Runtime・OpenVINO による CPU 推論比較（Issue #15）",
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="run_anomalib.py --save-checkpoint の model.ckpt へのパス")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="report.json と manifest.json がある run の出力ディレクトリ")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="結果と書き出しファイルの出力先（リポジトリ外）")
    parser.add_argument("--engines", type=str, default="torch,onnxruntime,openvino",
                        help="比較対象エンジン（カンマ区切り、torch は必須）")
    parser.add_argument("--threads", type=int, default=8,
                        help="各エンジンに与えるスレッド数（既定 8）")
    parser.add_argument("--warmup", type=int, default=5,
                        help="ウォームアップ枚数（集計から除外、既定 5）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="計測の繰り返し回数（既定 1）")
    parser.add_argument("--max-images", type=int, default=None,
                        help="計測する最大画像枚数（既定 None = 全 test）")
    parser.add_argument("--opset", type=int, default=18, help="要求する ONNX opset（既定 18。torch 2.15 の dynamo exporter は 18 未満を自動で 18 に引き上げる）")
    parser.add_argument("--check-only", action="store_true",
                        help="export のみ実施し、ONNX 演算子ヒストグラムを記録して終了する")
    parser.add_argument("--worker", type=Path, default=None,
                        help="ワーカーモード: 設定 JSON を読み、1エンジンを実行する")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 汎用ヘルパー
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    names = ("torch", "onnx", "onnxscript", "onnxruntime", "openvino")
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"report.json がありません: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _image_size_from_report(report: dict[str, Any], default: int = 256) -> int:
    size = report.get("image_size")
    if isinstance(size, (list, tuple)) and size:
        try:
            return int(size[0])
        except (TypeError, ValueError):
            return default
    return default


def _threshold_from_report(report: dict[str, Any]) -> float | None:
    defect = report.get("defect_size") or {}
    thresholds = defect.get("thresholds") if isinstance(defect, dict) else None
    value = thresholds.get("image") if isinstance(thresholds, dict) else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 親プロセス: inputs.npy 構築 + ONNX export
# ---------------------------------------------------------------------------


def _prepare_parent(
    args: argparse.Namespace,
    ckpt_path: Path,
    model_name: str,
    samples: list[dict[str, Any]],
    work_dir: Path,
    engines: list[str],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from anomalib.data.utils import read_image

    from adbench.engines import (
        export_onnx,
        file_size_bytes,
        load_anomalib_model,
        onnx_op_histogram,
    )
    from adbench.preprocess import apply_pre_processor, describe_preprocess

    info: dict[str, Any] = {"inputs": None, "export": {}}
    inputs_path = work_dir / "inputs.npy"
    onnx_path = work_dir / "model.onnx"

    t_inputs = time.perf_counter()
    model = load_anomalib_model(model_name, ckpt_path)
    model.eval()
    pre = getattr(model, "pre_processor", None)

    arrays: list[Any] = []
    with torch.inference_mode():
        for s in samples:
            img = read_image(s["image_path"], as_tensor=True).unsqueeze(0)
            x = apply_pre_processor(pre, img, "eval_transform")
            if isinstance(x, dict):
                x = next(iter(x.values()))
            x = x.detach().float().cpu()
            if x.ndim == 4:
                # (1, 3, H, W) → (3, H, W)。inputs.npy は N×3×H×W で保存する。
                x = x[0]
            arrays.append(x.numpy().astype(np.float32))
    stacked = np.stack(arrays, axis=0)
    del arrays
    np.save(str(inputs_path), stacked)
    info["inputs"] = {
        "status": "ok",
        "shape": list(stacked.shape),
        "preprocess": describe_preprocess(pre, "eval_transform"),
        "seconds": round(time.perf_counter() - t_inputs, 3),
        "num_images": len(samples),
    }

    need_onnx = any(e in {"onnxruntime", "openvino"} for e in engines)
    if not need_onnx:
        return info

    example = torch.from_numpy(np.array(stacked[0], dtype=np.float32, copy=True)).unsqueeze(0)
    export_result = export_onnx(model.model, example, onnx_path, opset=args.opset)
    del example

    if export_result["status"] == "ok":
        try:
            export_result["op_histogram"] = onnx_op_histogram(onnx_path)
        except Exception as exc:
            export_result["op_histogram_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        export_result["onnx_size_bytes"] = file_size_bytes([onnx_path, onnx_path.with_name(onnx_path.name + ".data")])
        export_result["onnx_path"] = str(onnx_path)

    export_map: dict[str, Any] = {}
    for e in engines:
        if e == "torch":
            export_map[e] = {"status": "skipped", "reason": "torch は export 不要"}
            continue
        entry: dict[str, Any] = dict(export_result)
        if e == "openvino" and export_result["status"] == "ok":
            xml_path = work_dir / "model.xml"
            bin_path = work_dir / "model.bin"
            try:
                import openvino as ov

                core = ov.Core()
                ov_model = core.read_model(str(onnx_path))
                ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)
                entry["ir_xml_path"] = str(xml_path)
                entry["ir_bin_path"] = str(bin_path)
                entry["ir_size_bytes"] = file_size_bytes([xml_path, bin_path])
            except Exception as exc:
                entry["ir_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        export_map[e] = entry
    info["export"] = export_map

    del model, stacked
    return info


# ---------------------------------------------------------------------------
# ワーカー起動
# ---------------------------------------------------------------------------


def _run_engine_via_worker(
    *,
    engine: str,
    ckpt_path: Path,
    onnx_path: Path,
    model_name: str,
    inputs_path: Path,
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, Any]:
    if engine in {"onnxruntime", "openvino"}:
        model_path = str(onnx_path)
    else:
        model_path = str(ckpt_path)

    config = {
        "engine": engine,
        "model_path_or_ckpt": model_path,
        "model_name": model_name,
        "inputs_path": str(inputs_path),
        "threads": int(args.threads),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "max_images": args.max_images,
        "out_path": str(work_dir / f"outputs_{engine}.npz"),
        "result_path": str(work_dir / f"worker_{engine}.json"),
    }
    config_path = work_dir / f"config_{engine}.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_seconds = round(time.perf_counter() - t0, 3)

    result_path = Path(config["result_path"])
    if not result_path.is_file():
        return {
            "engine": engine,
            "status": "failed",
            "error_type": "worker_result_missing",
            "error_message": (proc.stderr or "")[-2000:],
            "returncode": proc.returncode,
            "wall_seconds": wall_seconds,
        }
    try:
        summary = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "engine": engine,
            "status": "failed",
            "error_type": "worker_result_parse_error",
            "error_message": f"{type(exc).__name__}: {exc}",
            "returncode": proc.returncode,
            "wall_seconds": wall_seconds,
        }
    summary["returncode"] = proc.returncode
    summary["wall_seconds"] = wall_seconds
    if proc.returncode != 0 and summary.get("status") == "ok":
        summary["status"] = "failed"
        summary["error_message"] = (proc.stderr or "")[-2000:]
    return summary


def _run_worker(config_path: Path) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    engine = str(config.pop("engine"))
    result_path = Path(config.pop("result_path"))
    try:
        summary = engines_mod.run_engine_worker(engine, **config)
    except Exception as exc:
        summary = {
            "engine": engine,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc()[:2000],
        }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if summary.get("status") == "ok" else 1


# ---------------------------------------------------------------------------
# 集計: 出力の読み込みと AUROC
# ---------------------------------------------------------------------------


def _load_outputs(npz_path: Path) -> dict[str, Any]:
    import numpy as np

    data = np.load(str(npz_path))
    scores = np.asarray(data["pred_score"], dtype=np.float32).reshape(-1)
    maps = np.asarray(data["anomaly_map"], dtype=np.float32)
    return {"scores": scores, "maps": maps}


def _ref_metrics(ref_data: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image as PILImage

    from adbench.raw_eval import (
        binary_auroc_from_values,
        pixel_auroc_from_arrays,
        resize_mask_nearest,
    )

    ref_scores = np.asarray(ref_data["scores"]).reshape(-1)
    ref_maps = np.asarray(ref_data["maps"], dtype=np.float32)
    if ref_maps.ndim == 4:
        if ref_maps.shape[1] == 1:
            ref_maps = ref_maps[:, 0]
        else:
            raise ValueError(f"anomaly_map のチャネル数が不正: {ref_maps.shape}")
    if ref_maps.ndim != 3:
        raise ValueError(f"anomaly_map の形状が不正: {ref_maps.shape}")

    n, h, w = ref_maps.shape
    if ref_scores.shape[0] != n:
        raise ValueError(
            f"pred_score 件数と anomaly_map 件数が一致しません: {ref_scores.shape[0]} vs {n}"
        )
    if n > len(samples):
        raise ValueError(f"samples {len(samples)} 件 < 出力 {n} 件")

    labels = [int(s.get("label_index", 0)) for s in samples[:n]]

    mask_parts = []
    for s in samples[:n]:
        mp = s.get("mask_path")
        if mp:
            with PILImage.open(mp) as im:
                arr = np.asarray(im.convert("L"))
                mask_parts.append(resize_mask_nearest(arr, h, w).reshape(-1))
        else:
            mask_parts.append(np.zeros((h * w,), dtype=np.uint8))
    masks_flat = (
        np.concatenate(mask_parts) if mask_parts else np.zeros((0,), dtype=np.uint8)
    )

    image_auroc = binary_auroc_from_values(ref_scores.tolist(), labels)
    pixel_auroc = pixel_auroc_from_arrays(ref_maps.reshape(-1), masks_flat)

    return {
        "scores": ref_scores,
        "maps": ref_maps,
        "labels": labels,
        "masks_flat": masks_flat,
        "image_auroc": None if image_auroc is None else round(float(image_auroc), 6),
        "pixel_auroc": None if pixel_auroc is None else round(float(pixel_auroc), 6),
    }


def _compute_agreement(
    *,
    ref_metrics: dict[str, Any],
    target_npz: Path,
    image_threshold: float | None,
) -> dict[str, Any]:
    import numpy as np

    from adbench.engines import compare_outputs, decision_agreement
    from adbench.raw_eval import (
        binary_auroc_from_values,
        count_tied_scores,
        pixel_auroc_from_arrays,
    )

    target = _load_outputs(target_npz)
    ref_scores = ref_metrics["scores"]
    ref_maps = ref_metrics["maps"]
    labels = ref_metrics["labels"]
    masks_flat = ref_metrics["masks_flat"]

    tgt_scores = target["scores"]
    tgt_maps = target["maps"]
    if ref_maps.ndim == 4 and tgt_maps.ndim == 3 and ref_maps.shape[1] == 1:
        ref_maps = ref_maps[:, 0]
    if tgt_maps.ndim == 4 and tgt_maps.shape[1] == 1:
        tgt_maps = tgt_maps[:, 0]

    if ref_scores.shape != tgt_scores.shape:
        return {
            "status": "failed",
            "reason": f"pred_score 形状不一致: {ref_scores.shape} vs {tgt_scores.shape}",
        }
    if ref_maps.shape != tgt_maps.shape:
        return {
            "status": "failed",
            "reason": f"anomaly_map 形状不一致: {ref_maps.shape} vs {tgt_maps.shape}",
        }

    diffs = compare_outputs(ref_scores, ref_maps, tgt_scores, tgt_maps)
    agreement = decision_agreement(ref_scores, tgt_scores, image_threshold)
    target_image = binary_auroc_from_values(tgt_scores.tolist(), labels)
    target_pixel = pixel_auroc_from_arrays(tgt_maps.reshape(-1), masks_flat)
    tied = count_tied_scores(tgt_scores.tolist())

    return {
        "status": "ok",
        "diffs": diffs,
        "decision_agreement": agreement,
        "image_AUROC_raw": None if target_image is None else round(float(target_image), 6),
        "pixel_AUROC_raw": None if target_pixel is None else round(float(target_pixel), 6),
        "tied_scores": tied,
        "image_threshold": image_threshold,
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def run_main(args: argparse.Namespace) -> int:
    if args.worker is not None:
        return _run_worker(args.worker)

    if args.checkpoint is None or args.run_dir is None or args.output_dir is None:
        print("error: --checkpoint, --run-dir, --output-dir は必須です", file=sys.stderr)
        return 2
    if args.threads < 1:
        print("error: --threads は1以上を指定してください", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("error: --warmup は0以上を指定してください", file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("error: --repeats は1以上を指定してください", file=sys.stderr)
        return 2
    if args.max_images is not None and args.max_images < 0:
        print("error: --max-images は0以上を指定してください", file=sys.stderr)
        return 2

    try:
        selected_engines = engines_mod.parse_engines(args.engines)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    work_dir = output_dir / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).resolve()
    try:
        source_report = _load_run_metadata(run_dir)
        manifest_info = engines_mod.load_manifest_test_samples(run_dir / "manifest.json")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    model_name = str(source_report.get("model") or "padim")
    image_size = _image_size_from_report(source_report)
    image_threshold = _threshold_from_report(source_report)
    samples = manifest_info["samples"]

    engine_report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "source_run_dir": str(run_dir),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "model": model_name,
        "image_size": [image_size, image_size],
        "input_shape": None,
        "threads": int(args.threads),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "max_images": args.max_images,
        "opset": int(args.opset),
        "engines_requested": selected_engines,
        "host": collect_environment(),
        "versions": _versions(),
        "inputs": None,
        "export": {},
        "engines": {},
        "reference": None,
        "agreement": {},
        "notes": [
            "前処理は全エンジン共通で eval_transform（antialias 有効、engine.test と同じ）を使い、"
            "1×3×H×W float32 の input に固定して各エンジンに渡しています。",
            "PostProcessor は比較対象に含めていません（生の pred_score / anomaly_map を比較）。",
            "各エンジンは別プロセスで実行し、ピーク RSS は sampled_rss（推論ループ中のサンプリング推定）と "
            "ワーカー自身の VmHWM（/proc/self/status、ロード時を含むピーク）を併記しています。"
            "getrusage の ru_maxrss は Linux では fork 時の親プロセスの値を引き継ぐため、"
            "getrusage_maxrss_mib_inherits_parent として参考値にとどめています。",
            "torch を基準に、pred_score と anomaly_map の差分・画像/画素 AUROC・"
            "画像レベル閾値での判定一致率を計算しています。",
            "エンジン比較の対象は model 推論時間（入力テンソル → 出力 numpy）で、"
            "decode/preprocess は含みません。torch も .numpy() 変換まで含めています。",
        ],
    }

    try:
        prep = _prepare_parent(
            args,
            Path(args.checkpoint),
            model_name,
            samples,
            work_dir,
            selected_engines,
        )
    except Exception as exc:
        engine_report["notes"].append(
            f"親プロセスでの準備に失敗しました: {type(exc).__name__}: {str(exc)[:300]}"
        )
        _write_json(work_dir / "engine_report.json", engine_report)
        return 1

    engine_report["inputs"] = prep["inputs"]
    engine_report["export"] = prep["export"]
    engine_report["input_shape"] = (prep["inputs"] or {}).get("shape")

    if (prep["inputs"] or {}).get("status") != "ok":
        engine_report["notes"].append(
            "inputs.npy の構築に失敗したため、エンジン計測は行いません。"
        )
        _write_json(work_dir / "engine_report.json", engine_report)
        return 1

    if args.check_only:
        for e in selected_engines:
            if e not in engine_report["export"]:
                engine_report["export"][e] = {
                    "status": "skipped",
                    "reason": "ONNX 変換対象外のエンジン",
                }
        engine_report["notes"].append(
            "--check-only が指定されたため、ワーカーは起動していません。"
        )
        _write_json(work_dir / "engine_report.json", engine_report)
        return 0

    inputs_path = work_dir / "inputs.npy"
    onnx_path = work_dir / "model.onnx"
    onnx_ready = onnx_path.is_file()

    engine_summaries: dict[str, Any] = {}
    for engine in selected_engines:
        if engine in {"onnxruntime", "openvino"} and not onnx_ready:
            engine_summaries[engine] = {
                "engine": engine,
                "status": "skipped",
                "reason": "ONNX export が成功していないため実行できません",
            }
            continue
        summary = _run_engine_via_worker(
            engine=engine,
            ckpt_path=Path(args.checkpoint),
            onnx_path=onnx_path,
            model_name=model_name,
            inputs_path=inputs_path,
            args=args,
            work_dir=work_dir,
        )
        engine_summaries[engine] = summary
    engine_report["engines"] = engine_summaries

    ref = engine_summaries.get("torch") or {}
    if ref.get("status") != "ok":
        engine_report["notes"].append(
            "torch（基準）が成功していないため agreement は計算しません。"
        )
        _write_json(work_dir / "engine_report.json", engine_report)
        return 1

    ref_npz = work_dir / "outputs_torch.npz"
    if not ref_npz.is_file():
        engine_report["notes"].append("outputs_torch.npz が見つかりません。")
        _write_json(work_dir / "engine_report.json", engine_report)
        return 1

    try:
        ref_data = _load_outputs(ref_npz)
        ref_metrics = _ref_metrics(ref_data, samples)
    except Exception as exc:
        engine_report["notes"].append(
            f"torch 出力の読み込みに失敗しました: {type(exc).__name__}: {str(exc)[:300]}"
        )
        _write_json(work_dir / "engine_report.json", engine_report)
        return 1

    engine_report["reference"] = {
        "engine": "torch",
        "image_AUROC_raw": ref_metrics["image_auroc"],
        "pixel_AUROC_raw": ref_metrics["pixel_auroc"],
        "image_threshold": image_threshold,
        "num_images": len(ref_metrics["labels"]),
        "sampling_interval_s": DEFAULT_SAMPLING_INTERVAL_S,
    }

    agreement: dict[str, Any] = {}
    for engine in selected_engines:
        if engine == "torch":
            continue
        summary = engine_summaries.get(engine) or {}
        if summary.get("status") != "ok":
            agreement[engine] = {
                "status": "skipped",
                "reason": "エンジンが成功していません",
            }
            continue
        tgt_npz = work_dir / f"outputs_{engine}.npz"
        if not tgt_npz.is_file():
            agreement[engine] = {
                "status": "failed",
                "reason": "outputs npz が見つかりません",
            }
            continue
        try:
            agreement[engine] = _compute_agreement(
                ref_metrics=ref_metrics,
                target_npz=tgt_npz,
                image_threshold=image_threshold,
            )
        except Exception as exc:
            agreement[engine] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
    engine_report["agreement"] = agreement

    _write_json(work_dir / "engine_report.json", engine_report)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
