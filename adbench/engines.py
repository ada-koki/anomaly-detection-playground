"""ONNX Runtime・OpenVINO による CPU 推論比較のためのエンジン補助。

重い依存（torch, onnx, onnxruntime, openvino, numpy）はすべて関数内で遅延 import
するため、本モジュールの import 自体は軽い。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

ENGINE_NAMES = ("torch", "onnxruntime", "openvino")
_OUTPUT_FIELDS = ("pred_score", "anomaly_map")


# ---------------------------------------------------------------------------
# 設定パース
# ---------------------------------------------------------------------------


def parse_engines(text: str | None) -> list[str]:
    """エンジン指定文字列をパースする。

    Args:
        text: カンマ区切りのエンジン名。None または空文字なら既定値全種を返す。

    Returns:
        選択されたエンジン名のリスト（順序は入力順、重複なし）。

    Raises:
        ValueError: 未知のエンジン名・重複・torch が含まれない場合。
    """
    raw_text = "" if text is None else str(text)
    parts = [part.strip().lower() for part in raw_text.split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return list(ENGINE_NAMES)

    seen: set[str] = set()
    selected: list[str] = []
    for name in parts:
        if name not in ENGINE_NAMES:
            raise ValueError(
                f"未知のエンジンです: {name!r}（{ENGINE_NAMES} のいずれか）"
            )
        if name in seen:
            raise ValueError(f"エンジン名が重複しています: {name!r}")
        seen.add(name)
        selected.append(name)

    if "torch" not in selected:
        raise ValueError("engines には必ず torch を含めてください（比較の基準）")
    return selected


# ---------------------------------------------------------------------------
# 出力差分・判定一致率
# ---------------------------------------------------------------------------


def _diff_stats(a: Any, b: Any, eps: float) -> dict[str, float]:
    import numpy as np

    diff = np.abs(a - b)
    rel = diff / np.maximum(np.abs(b), eps)
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "p999_rel": float(np.percentile(rel, 99.9)) if rel.size else 0.0,
    }


def compare_outputs(
    ref_scores: Any,
    ref_maps: Any,
    scores: Any,
    maps: Any,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """torch の出力を基準に、別エンジンの出力との差分を計算する。

    相対差は ``|a - b| / max(|b|, eps)`` で計算する（b は基準値）。

    Args:
        ref_scores: 基準 pred_score（flatten 可能な配列）。
        ref_maps: 基準 anomaly_map（flatten 可能な配列）。
        scores: 比較対象 pred_score。
        maps: 比較対象 anomaly_map。
        eps: 相対差の分母の下限。

    Returns:
        差分指標の dict。キーは
        ``score_max_abs_diff`` / ``score_mean_abs_diff`` / ``score_max_rel_diff`` /
        ``map_max_abs_diff`` / ``map_mean_abs_diff`` / ``map_max_rel_diff`` /
        ``map_p999_rel_diff``。

    Raises:
        ValueError: pred_score または anomaly_map の形状が一致しない場合。
    """
    import numpy as np

    rs = np.asarray(ref_scores, dtype=np.float64).reshape(-1)
    ss = np.asarray(scores, dtype=np.float64).reshape(-1)
    if rs.shape != ss.shape:
        raise ValueError(f"pred_score の形状が一致しません: {rs.shape} vs {ss.shape}")
    rm = np.asarray(ref_maps, dtype=np.float64)
    sm = np.asarray(maps, dtype=np.float64)
    if rm.shape != sm.shape:
        raise ValueError(f"anomaly_map の形状が一致しません: {rm.shape} vs {sm.shape}")

    s = _diff_stats(ss, rs, eps)
    m = _diff_stats(sm, rm, eps)
    return {
        "score_max_abs_diff": round(s["max_abs"], 6),
        "score_mean_abs_diff": round(s["mean_abs"], 6),
        "score_max_rel_diff": round(s["max_rel"], 6),
        "map_max_abs_diff": round(m["max_abs"], 6),
        "map_mean_abs_diff": round(m["mean_abs"], 6),
        "map_max_rel_diff": round(m["max_rel"], 6),
        "map_p999_rel_diff": round(m["p999_rel"], 6),
    }


def decision_agreement(
    ref_scores: Any,
    scores: Any,
    threshold: float | None,
) -> float | None:
    """画像レベル閾値で 0/1 判定したときの一致率を返す。

    Args:
        ref_scores: 基準 pred_score。
        scores: 比較対象 pred_score。
        threshold: 画像スコアの閾値。None / 非有限の場合は ``None`` を返す。

    Returns:
        一致率（0.0〜1.0）。計算できない場合は ``None``。

    Raises:
        ValueError: pred_score の形状が一致しない場合。
    """
    import math

    import numpy as np

    if threshold is None:
        return None
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(thr):
        return None

    rs = np.asarray(ref_scores, dtype=np.float64).reshape(-1)
    ss = np.asarray(scores, dtype=np.float64).reshape(-1)
    if rs.shape != ss.shape:
        raise ValueError(f"pred_score の形状が一致しません: {rs.shape} vs {ss.shape}")
    if rs.size == 0:
        return None
    ref_dec = rs >= thr
    dec = ss >= thr
    return float((ref_dec == dec).mean())


# ---------------------------------------------------------------------------
# ONNX 演算子ヒストグラム・ファイルサイズ
# ---------------------------------------------------------------------------


def onnx_op_histogram(onnx_path: str | Path) -> dict[str, int]:
    """ONNX グラフの演算子種類のヒストグラムを返す。

    Args:
        onnx_path: ONNX ファイルのパス。

    Returns:
        ``{op_type: count}`` を件数降順（同数は op_type 昇順）に並べた辞書。

    Raises:
        ImportError: onnx が import できない場合。
    """
    import onnx

    model = onnx.load(str(onnx_path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        key = str(node.op_type)
        counts[key] = counts.get(key, 0) + 1
    return {k: counts[k] for k in sorted(counts, key=lambda k: (-counts[k], k))}


def file_size_bytes(paths: Iterable[str | Path]) -> int:
    """指定パスのうち存在するファイルの合計サイズを返す（バイト）。"""
    total = 0
    for p in paths:
        path = Path(str(p))
        if path.is_file():
            total += path.stat().st_size
    return total


# ---------------------------------------------------------------------------
# ONNX 書き出し
# ---------------------------------------------------------------------------


def _output_value(output: Any, name: str) -> Any:
    if isinstance(output, dict):
        return output.get(name)
    return getattr(output, name, None)


def _present_output_names(output: Any) -> list[str]:
    """InferenceBatch（NamedTuple/dict）に含まれる非None出力の名前を順序付きで返す。"""
    names: list[str] = []
    for name in _OUTPUT_FIELDS:
        if _output_value(output, name) is not None:
            names.append(name)
    return names


def _extract_outputs(output: Any) -> tuple[Any, ...]:
    """InferenceBatch から非None出力を ``_OUTPUT_FIELDS`` の順にタプルで返す。"""
    import torch

    values: list[Any] = []
    for name in _OUTPUT_FIELDS:
        value = _output_value(output, name)
        if value is None:
            continue
        values.append(torch.as_tensor(value))
    if not values:
        raise RuntimeError(
            "model.model の出力に pred_score / anomaly_map がありません"
        )
    return tuple(values)


def make_outputs_only(model: Any) -> Any:
    """``model.model`` を eval で呼び、``pred_score`` と ``anomaly_map`` のテンソルだけを返す nn.Module を生成する。

    torch は関数内で遅延 import する。
    """
    import torch.nn as nn

    class OutputsOnly(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any) -> tuple[Any, ...]:  # type: ignore[override]
            return _extract_outputs(self.inner(x))

    return OutputsOnly(model)


def export_onnx(
    torch_model: Any,
    example: Any,
    path: str | Path,
    *,
    opset: int = 17,
) -> dict[str, Any]:
    """``torch_model``（= ``model.model``）を固定入力で ONNX に書き出す。

    まず ``dynamo=True`` を試し、失敗したら ``dynamo=False``（legacy）を 1 回だけ試す。
    どちらで成功したか・失敗理由・出力名と形状・所要秒・ファイルサイズを返す。

    Args:
        torch_model: Anomalib の ``model.model``（torch.nn.Module）。
        example: 入力例（shape (1, 3, H, W), float32）。
        path: 拡張子 .onnx の保存先。
        opset: ONNX opset バージョン。

    Returns:
        dict: ``status``（ok/failed）、``method``（dynamo/legacy/None）、
        ``opset``、``seconds``、``file_size_bytes``、``output_names``、
        ``output_shapes``、``error``、``attempts``（試行ごとの記録）。
    """
    import torch

    path = Path(str(path))
    wrapper = make_outputs_only(torch_model)
    wrapper.eval()

    result: dict[str, Any] = {
        "status": "failed",
        "method": None,
        "opset": int(opset),
        "seconds": None,
        "file_size_bytes": None,
        "output_names": None,
        "output_shapes": None,
        "error": None,
        "attempts": [],
    }

    try:
        with torch.inference_mode():
            raw = wrapper.inner(example)
    except Exception as exc:
        result["error"] = f"probe {type(exc).__name__}: {str(exc)[:200]}"
        return result

    output_names = _present_output_names(raw)
    output_shapes: list[list[int] | None] = []
    for name in output_names:
        value = _output_value(raw, name)
        try:
            output_shapes.append([int(d) for d in value.shape])  # type: ignore[attr-defined]
        except Exception:
            output_shapes.append(None)
    del raw

    result["output_names"] = output_names
    result["output_shapes"] = output_shapes

    for method, use_dynamo in (("dynamo", True), ("legacy", False)):
        attempt: dict[str, Any] = {
            "method": method,
            "status": "failed",
            "error": None,
            "seconds": None,
        }
        t0 = time.perf_counter()
        try:
            torch.onnx.export(
                wrapper,
                (example,),
                str(path),
                dynamo=use_dynamo,
                opset_version=int(opset),
                input_names=["input"],
                output_names=output_names,
            )
        except Exception as exc:
            attempt["seconds"] = round(time.perf_counter() - t0, 3)
            attempt["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            result["attempts"].append(attempt)
            continue

        attempt["status"] = "ok"
        attempt["seconds"] = round(time.perf_counter() - t0, 3)
        result["attempts"].append(attempt)
        result["status"] = "ok"
        result["method"] = method
        result["seconds"] = attempt["seconds"]
        result["file_size_bytes"] = path.stat().st_size if path.exists() else None
        return result

    result["error"] = "dynamo と legacy の両方で失敗しました"
    return result


# ---------------------------------------------------------------------------
# Anomalib モデル読み込み
# ---------------------------------------------------------------------------


def _anomalib_class(model_name: str) -> Any:
    from anomalib.models import EfficientAd, Padim, Patchcore

    key = str(model_name).strip().lower()
    mapping = {
        "padim": Padim,
        "patchcore": Patchcore,
        "efficientad": EfficientAd,
    }
    if key not in mapping:
        raise ValueError(
            f"未対応のモデルです: {model_name!r}（padim / patchcore / efficientad）"
        )
    return mapping[key]


def load_anomalib_model(model_name: str, checkpoint_path: str | Path) -> Any:
    """checkpoint から Anomalib モデルを CPU へ読み込む。

    torch / anomalib は関数内で遅延 import する。
    """
    cls = _anomalib_class(model_name)
    return cls.load_from_checkpoint(str(checkpoint_path), map_location="cpu")


# ---------------------------------------------------------------------------
# ワーカー: 1エンジン推論
# ---------------------------------------------------------------------------


def _build_engine_runner(
    engine: str,
    *,
    model_path_or_ckpt: str,
    model_name: str,
    threads: int,
    inputs: Any,
) -> Any:
    """エンジンごとの「1画像推論」関数を返す。戻り値は (score: float, anomaly_map: np.ndarray)。"""
    if engine == "torch":
        import numpy as np
        import torch

        model = load_anomalib_model(model_name, model_path_or_ckpt)
        model.eval()
        torch.set_num_threads(int(threads))
        inner = model.model

        def run_torch(idx: int) -> tuple[float, Any]:
            arr = np.array(inputs[idx], dtype=np.float32, copy=True)
            x = torch.from_numpy(arr).unsqueeze(0)
            with torch.inference_mode():
                out = inner(x)
            score_v = _output_value(out, "pred_score")
            map_v = _output_value(out, "anomaly_map")
            if score_v is None or map_v is None:
                raise RuntimeError(
                    "model.model の出力に pred_score / anomaly_map がありません"
                )
            score = float(
                torch.as_tensor(score_v).detach().float().reshape(-1)[0].item()
            )
            m = torch.as_tensor(map_v).detach().float().cpu()
            while m.dim() > 2:
                m = m[0]
            return score, m.numpy().astype(np.float32)

        return run_torch

    if engine == "onnxruntime":
        import numpy as np
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(model_path_or_ckpt),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        def run_ort(idx: int) -> tuple[float, Any]:
            arr = np.array(inputs[idx], dtype=np.float32, copy=True)
            x = np.ascontiguousarray(arr[np.newaxis, ...])
            outs = session.run(output_names, {input_name: x})
            by_name = dict(zip(output_names, outs))
            score_arr = by_name.get("pred_score")
            map_arr = by_name.get("anomaly_map")
            if score_arr is None or map_arr is None:
                raise RuntimeError(
                    f"ONNX の出力名に pred_score / anomaly_map が見つかりません: {output_names}"
                )
            score = float(np.asarray(score_arr).reshape(-1)[0])
            m = np.asarray(map_arr, dtype=np.float32)
            while m.ndim > 2:
                m = m[0]
            return score, m.astype(np.float32)

        return run_ort

    if engine == "openvino":
        import numpy as np
        import openvino as ov

        core = ov.Core()
        ov_model = core.read_model(str(model_path_or_ckpt))
        compiled = core.compile_model(
            ov_model,
            "CPU",
            {
                "INFERENCE_NUM_THREADS": int(threads),
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "f32",
            },
        )
        out_ports = list(compiled.outputs)

        def _port_name(port: Any) -> str | None:
            try:
                return str(port.get_any_name())
            except Exception:
                return None

        port_names = [_port_name(p) for p in out_ports]

        def run_ov(idx: int) -> tuple[float, Any]:
            arr = np.array(inputs[idx], dtype=np.float32, copy=True)
            x = np.ascontiguousarray(arr[np.newaxis, ...])
            outs = compiled([x])
            arrays = [np.asarray(outs[p]) for p in out_ports]
            if all(name is not None for name in port_names):
                by_name = {name: arr for name, arr in zip(port_names, arrays)}
            else:
                by_name = {}
                for i, name in enumerate(_OUTPUT_FIELDS):
                    if i < len(arrays):
                        by_name[name] = arrays[i]
            score_arr = by_name.get("pred_score")
            map_arr = by_name.get("anomaly_map")
            if score_arr is None or map_arr is None:
                raise RuntimeError(
                    f"OpenVINO の出力名に pred_score / anomaly_map が見つかりません: {port_names}"
                )
            score = float(np.asarray(score_arr).reshape(-1)[0])
            m = np.asarray(map_arr, dtype=np.float32)
            while m.ndim > 2:
                m = m[0]
            return score, m.astype(np.float32)

        return run_ov

    raise ValueError(f"未知のエンジンです: {engine!r}")


def run_engine_worker(
    engine: str,
    *,
    model_path_or_ckpt: str,
    model_name: str,
    inputs_path: str,
    threads: int,
    warmup: int,
    repeats: int,
    max_images: int | None,
    out_path: str,
) -> dict[str, Any]:
    """1エンジン分の推論計測を別プロセスで実行するワーカー。

    入力（``inputs.npy``, N×3×H×W float32）を mmap で読み、1枚ずつ推論して
    ``pred_score``（N）と ``anomaly_map``（N×H×W, float32）を ``out_path``（.npz）に保存する。

    Args:
        engine: エンジン名（``ENGINE_NAMES`` のいずれか）。
        model_path_or_ckpt: torch なら model.ckpt、onnxruntime/openvino なら ONNX パス。
        model_name: モデル種別（torch のみ使用）。
        inputs_path: 共通前処理済み入力の .npy パス。
        threads: スレッド数（エンジンごとに意味が異なる）。
        warmup: ウォームアップ枚数。
        repeats: 計測の繰り返し回数（1 以上）。
        max_images: 計測する最大画像枚数（None または 0 は全枚数）。
        out_path: 出力 npz の保存先（リポジトリ外を想定）。

    Returns:
        dict: 計測結果（``status``、``load_seconds``、``model_ms``、RSS、出力形状など）。

    Raises:
        ValueError: パラメータ不正・入力形状不正の場合。
    """
    if engine not in ENGINE_NAMES:
        raise ValueError(f"未知のエンジンです: {engine!r}")
    if repeats < 1:
        raise ValueError("repeats には1以上を指定してください")
    if warmup < 0:
        raise ValueError("warmup には0以上を指定してください")

    import numpy as np

    from .memory import (
        DEFAULT_SAMPLING_INTERVAL_S,
        peak_rss_lifetime_mib,
        sampled_rss,
    )
    from .stats import summarize_latencies

    inputs = np.load(str(inputs_path), mmap_mode="r")
    if inputs.ndim != 4 or int(inputs.shape[1]) != 3:
        raise ValueError(
            f"inputs.npy の形状が不正です: {tuple(inputs.shape)}（N,3,H,W を期待）"
        )
    n_total = int(inputs.shape[0])
    if max_images is not None and int(max_images) > 0:
        n = min(int(max_images), n_total)
    else:
        n = n_total
    if n <= 0:
        raise ValueError("計測対象の画像がありません")

    selected_indices = list(range(n))
    warmup_count = min(int(warmup), n) if warmup > 0 else 0
    warmup_indices = selected_indices[:warmup_count]

    out_path_obj = Path(str(out_path))
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    run_one = _build_engine_runner(
        engine,
        model_path_or_ckpt=str(model_path_or_ckpt),
        model_name=str(model_name),
        threads=int(threads),
        inputs=inputs,
    )
    load_seconds = time.perf_counter() - load_start

    warmup_executed = 0
    for idx in warmup_indices:
        run_one(int(idx))
        warmup_executed += 1

    model_times: list[float] = []
    collected_scores: list[float] | None = None
    collected_maps: list[Any] | None = None

    with sampled_rss(DEFAULT_SAMPLING_INTERVAL_S) as rss:
        for rep in range(repeats):
            for k, idx in enumerate(selected_indices):
                t0 = time.perf_counter()
                score, amap = run_one(int(idx))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                model_times.append(elapsed_ms)
                if rep == 0:
                    if collected_scores is None:
                        collected_scores = [0.0] * len(selected_indices)
                        collected_maps = [None] * len(selected_indices)
                    collected_scores[k] = float(score)
                    collected_maps[k] = np.asarray(amap, dtype=np.float32)

    if collected_scores is not None:
        scores_arr = np.asarray(collected_scores, dtype=np.float32)
    else:
        scores_arr = np.zeros((0,), dtype=np.float32)
    if collected_maps and all(m is not None for m in collected_maps):
        maps_arr = np.stack(collected_maps, axis=0).astype(np.float32)  # type: ignore[arg-type]
    else:
        maps_arr = np.zeros((0, 0, 0), dtype=np.float32)
    np.savez(str(out_path_obj), pred_score=scores_arr, anomaly_map=maps_arr)

    return {
        "engine": engine,
        "status": "ok",
        "load_seconds": round(load_seconds, 3),
        "model_ms": summarize_latencies(model_times),
        "latency_peak_rss_mib_sampled": rss["peak_rss_mib"],
        "process_peak_rss_mib": peak_rss_lifetime_mib(),
        "sampling_interval_s": DEFAULT_SAMPLING_INTERVAL_S,
        "num_images": n,
        "warmup_executed": warmup_executed,
        "repeats": int(repeats),
        "latency_samples": len(model_times),
        "output_shape": {
            "pred_score": list(scores_arr.shape),
            "anomaly_map": list(maps_arr.shape),
        },
        "inputs_path": str(inputs_path),
        "out_path": str(out_path_obj),
    }


# ---------------------------------------------------------------------------
# manifest から test サンプル抽出
# ---------------------------------------------------------------------------


def load_manifest_test_samples(manifest_path: str | Path) -> dict[str, Any]:
    """manifest.json から test サンプル（画像・マスク・ラベル）を読み出す。

    manifest は run_anomalib.py --save-checkpoint が書く schema_version 1 を想定する。
    パスは ``dataset_root`` を基準に解決する。

    Args:
        manifest_path: manifest.json のパス。

    Returns:
        ``{"dataset_root": str, "samples": [{"image_path", "mask_path", "label", "label_index"}, ...]}``。

    Raises:
        FileNotFoundError: ファイルが無い場合。
        ValueError: schema が不正、または splits.test が空の場合。
    """
    path = Path(str(manifest_path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest.json の最上位が辞書ではありません")

    dataset_root = data.get("dataset_root")
    if not dataset_root:
        raise ValueError("manifest.json に dataset_root がありません")
    root = Path(str(dataset_root))

    splits = data.get("splits") or {}
    test = splits.get("test") if isinstance(splits, dict) else None
    if not isinstance(test, list) or not test:
        raise ValueError("manifest.json の splits.test が空です")

    samples: list[dict[str, Any]] = []
    for entry in test:
        if not isinstance(entry, dict):
            raise ValueError("splits.test の要素が辞書ではありません")
        rel_image = entry.get("image_path")
        if not rel_image:
            raise ValueError("splits.test の要素に image_path がありません")
        image_path = root / str(rel_image)
        mask_rel = entry.get("mask_path")
        mask_path = None if not mask_rel else str(root / str(mask_rel))
        label = str(entry.get("label") or "")
        try:
            label_index = int(entry.get("label_index", 0))
        except (TypeError, ValueError):
            label_index = 0 if label == "good" else 1
        samples.append(
            {
                "image_path": str(image_path),
                "mask_path": mask_path,
                "label": label,
                "label_index": label_index,
            }
        )

    return {"dataset_root": str(root), "samples": samples}
