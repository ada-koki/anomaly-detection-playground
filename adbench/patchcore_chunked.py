"""PatchCore の coreset 選択をチャンク単位で行うためのヘルパー。

学習中に保持する embedding を 1 チャンク分（``chunk_images`` 枚相当）に抑えるため、
``anomalib.models.Patchcore`` のサブクラスを生成する ``make_chunked_patchcore`` を
提供する。torch / anomalib は関数内で遅延 import するため、本モジュールの import
自体は重い依存を要求しない。

Anomalib 2.7 では MemoryBankMixin が validation 開始時または学習 epoch 終了時に
``fit()`` を 1 回だけ呼ぶ（``_is_fitted`` で制御）。本実装は ``fit()`` を上書きし、
既定の全体 vstack + 全体 KCenterGreedy の代わりにチャンク選択結果を結合する。
"""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

from adbench.patchcore_memory import chunk_coreset_size

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _import_kcenter_greedy() -> Any:
    """``KCenterGreedy`` クラスを遅延 import する（配置差異を吸収）。"""
    candidates = (
        "anomalib.models.components.sampling.k_center_greedy",
        "anomalib.models.components.sampling",
        "anomalib.models.components",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, "KCenterGreedy", None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "KCenterGreedy を import できませんでした。Anomalib のバージョンと"
        "モジュール配置を確認してください。"
    )


def _kcenter_select(chunk: "torch.Tensor", k: int) -> "torch.Tensor":
    """チャンクから k 行を k-center-greedy で選び、コピーを返す。

    ``KCenterGreedy`` はコンストラクタの ``sampling_ratio`` から
    ``int(行数 * ratio)`` で件数を決めるため、``ratio = k / 行数`` を渡した上で
    float 誤差により件数がずれた場合だけ ``coreset_size`` を補正する。
    """
    if k < 1:
        raise ValueError("k には1以上を指定してください")
    rows = int(chunk.shape[0])
    if k > rows:
        raise ValueError(f"k ({k}) が行数 ({rows}) を超えています")
    kcenter_cls = _import_kcenter_greedy()
    sampler = kcenter_cls(embedding=chunk, sampling_ratio=k / rows)
    if int(getattr(sampler, "coreset_size", k)) != k:
        sampler.coreset_size = k
    idxs = sampler.select_coreset_idxs()
    # 選択結果は新しいテンソルとして保持する（元チャンクの storage を掴まない）。
    return chunk[idxs].clone()


def _take_rows(
    tensors: Sequence["torch.Tensor"],
    n_rows: int,
) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
    """先頭から ``n_rows`` 行を取り出し、``(取得した行, 残りの行)`` を返す。

    テンソル境界をまたぐ場合は行単位で分割する。
    """
    taken: list["torch.Tensor"] = []
    remaining: list["torch.Tensor"] = []
    needed = n_rows
    index = 0
    while index < len(tensors) and needed > 0:
        tensor = tensors[index]
        rows = int(tensor.shape[0])
        if rows <= needed:
            taken.append(tensor)
            needed -= rows
        else:
            taken.append(tensor[:needed])
            remaining.append(tensor[needed:])
            needed = 0
        index += 1
    remaining.extend(tensors[index:])
    return taken, remaining


def reduce_store_in_chunks(
    store: Sequence["torch.Tensor"],
    chunk_rows: int,
    ratio: float,
    select_fn: Callable[["torch.Tensor", int], "torch.Tensor"],
    carry: "torch.Tensor | None" = None,
    *,
    keep_remainder: bool = True,
) -> tuple[list["torch.Tensor"], "torch.Tensor | None", list[dict[str, Any]]]:
    """``store`` を ``chunk_rows`` 行ずつに区切って coreset を選ぶ。

    ``store`` 自体は変更しない。``carry`` は前回の端数（先頭に連結する）。
    端数は ``keep_remainder`` が True のとき次回へ繰り越す。

    Args:
        store: 特徴量テンソル（2 次元）のリスト。1 要素は複数行を持つ。
        chunk_rows: 1 チャンクの行数。
        ratio: coreset のサンプリング比率。
        select_fn: ``(チャンク, 選ぶ行数) -> 選んだ行（2 次元）`` を返す関数。
        carry: 前回の端数テンソル。
        keep_remainder: False なら端数を返さない（破棄する）。

    Returns:
        ``(選択された行のリスト, 端数テンソルまたは None, チャンク統計のリスト)``。
        統計の各要素は ``{"rows": int, "selected": int, "seconds": float}``。

    Raises:
        ValueError: 引数や ``select_fn`` の戻り値が不正な場合。
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunk_rows には1以上の int を指定してください")
    if not 0 < ratio <= 1:
        raise ValueError("ratio には0より大きく1以下の値を指定してください")

    import torch

    rows_source: list["torch.Tensor"] = []
    if carry is not None and int(carry.shape[0]) > 0:
        rows_source.append(carry)
    for tensor in store:
        if int(tensor.shape[0]) > 0:
            rows_source.append(tensor)

    selected: list["torch.Tensor"] = []
    stats: list[dict[str, Any]] = []
    while True:
        available = sum(int(tensor.shape[0]) for tensor in rows_source)
        if available < chunk_rows:
            break
        head, rows_source = _take_rows(rows_source, chunk_rows)
        chunk = head[0] if len(head) == 1 else torch.vstack(head)
        k = chunk_coreset_size(int(chunk.shape[0]), ratio)
        started = time.perf_counter()
        picked = select_fn(chunk, k)
        seconds = time.perf_counter() - started
        if not hasattr(picked, "shape") or len(picked.shape) != 2:
            raise ValueError(
                "select_fn は 2 次元テンソルを返してください。"
                f"実際: {getattr(picked, 'shape', None)}"
            )
        if int(picked.shape[1]) != int(chunk.shape[1]):
            raise ValueError(
                "select_fn の戻り値の列数が入力チャンクと一致しません。"
                f"入力: {tuple(chunk.shape)}, 出力: {tuple(picked.shape)}"
            )
        if int(picked.shape[0]) < 1:
            raise ValueError("select_fn が1行も選んでいません")
        selected.append(picked)
        stats.append(
            {
                "rows": int(chunk.shape[0]),
                "selected": int(picked.shape[0]),
                "seconds": round(seconds, 6),
            }
        )

    if not keep_remainder or not rows_source:
        remainder: "torch.Tensor | None" = None
    elif len(rows_source) == 1:
        remainder = rows_source[0]
    else:
        remainder = torch.vstack(rows_source)
    return selected, remainder, stats


def make_chunked_patchcore(chunk_images: int, **patchcore_kwargs: Any) -> Any:
    """チャンク単位で coreset を選ぶ PatchCore インスタンスを返す。

    学習中は ``self.model.embedding_store`` の総行数が
    ``chunk_images * 1 枚あたりのパッチ数`` 以上になった時点で
    ``torch.vstack`` → ``KCenterGreedy`` → 選択行を保持 → store を clear する。
    端数は行単位で次チャンクへ繰り越す。

    Args:
        chunk_images: 1 チャンクあたり画像枚数（1 以上）。
        **patchcore_kwargs: ``anomalib.models.Patchcore`` に渡す引数。

    Returns:
        ``Patchcore`` のサブクラスのインスタンス。``chunk_stats`` 属性に
        チャンクごとの ``{"rows", "selected", "seconds", "final"}`` を保持する。

    Raises:
        TypeError: ``chunk_images`` が int でない場合。
        ValueError: ``chunk_images`` が 1 未満の場合。
    """
    if isinstance(chunk_images, bool) or not isinstance(chunk_images, int):
        raise TypeError("chunk_images には int を指定してください")
    if chunk_images < 1:
        raise ValueError("chunk_images には1以上を指定してください")

    from anomalib.models import Patchcore

    class ChunkedPatchcore(Patchcore):
        """embedding_store を ``chunk_images`` 枚分に抑える PatchCore。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.chunk_images = chunk_images
            self.chunk_stats: list[dict[str, Any]] = []
            self._selected_chunks: list[Any] = []
            self._rows_per_image: int | None = None
            self._finalized = False

        # --- 学習中の縮約 -------------------------------------------------
        def training_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
            result = super().training_step(batch, *args, **kwargs)
            self._reduce_after_step(batch)
            return result

        def fit(self) -> None:
            """MemoryBankMixin から（validation 開始時または epoch 終了時に）1 回呼ばれる。

            既定の Patchcore.fit（全特徴の vstack → 全体 KCenterGreedy）の代わりに、
            残りの端数を処理して選択済みチャンクから memory_bank を確定する。
            """
            self.finalize_memory_bank()

        # --- 内部処理 -----------------------------------------------------
        def _embedding_store(self) -> list[Any]:
            store = getattr(self.model, "embedding_store", None)
            if not isinstance(store, list):
                return []
            return store

        def _infer_rows_per_image(self, batch: Any, last: Any) -> int | None:
            if self._rows_per_image is not None:
                return self._rows_per_image
            images = getattr(batch, "image", None)
            shape = getattr(images, "shape", None)
            rows = int(getattr(last, "shape", (0,))[0])
            if shape is None or int(shape[0]) < 1 or rows < 1:
                return None
            batch_size = int(shape[0])
            if rows % batch_size != 0:
                return None
            self._rows_per_image = rows // batch_size
            return self._rows_per_image

        def _reduce_after_step(self, batch: Any) -> None:
            if self._finalized:
                return
            store = self._embedding_store()
            if not store:
                return
            rows_per_image = self._infer_rows_per_image(batch, store[-1])
            if rows_per_image is None:
                return
            target_rows = self.chunk_images * rows_per_image
            if target_rows < 1:
                return
            rows = sum(int(tensor.shape[0]) for tensor in store)
            if rows < target_rows:
                return
            self._reduce_rows(target_rows)

        def _reduce_rows(self, chunk_rows: int) -> None:
            store = self._embedding_store()
            ratio = float(getattr(self, "coreset_sampling_ratio", 0.1))
            selected, remainder, stats = reduce_store_in_chunks(
                store,
                chunk_rows=chunk_rows,
                ratio=ratio,
                select_fn=_kcenter_select,
                carry=None,
                keep_remainder=True,
            )
            if not selected:
                return
            for stat in stats:
                stat["final"] = False
            self._selected_chunks.extend(selected)
            self.chunk_stats.extend(stats)
            store.clear()
            if remainder is not None:
                store.append(remainder)

        # --- 確定 ---------------------------------------------------------
        def finalize_memory_bank(self) -> None:
            """残りの端数を処理し、``memory_bank`` を float32 で確定する。"""
            if self._finalized:
                return
            import torch

            inner = self.model
            store = self._embedding_store()
            rows = sum(int(tensor.shape[0]) for tensor in store)
            if rows > 0:
                chunk = store[0] if len(store) == 1 else torch.vstack(store)
                ratio = float(getattr(self, "coreset_sampling_ratio", 0.1))
                k = chunk_coreset_size(int(chunk.shape[0]), ratio)
                started = time.perf_counter()
                picked = _kcenter_select(chunk, k)
                seconds = time.perf_counter() - started
                self._selected_chunks.append(picked)
                self.chunk_stats.append(
                    {
                        "rows": int(chunk.shape[0]),
                        "selected": int(picked.shape[0]),
                        "seconds": round(seconds, 6),
                        "final": True,
                    }
                )
                store.clear()
                del chunk

            if not self._selected_chunks:
                raise RuntimeError(
                    "PatchCore のチャンク縮約で coreset を1件も選べませんでした。"
                    "学習データが空か、embedding_store に特徴量が蓄積されていません。"
                )

            # PatchcoreModel.forward は入力を memory_bank.dtype にキャストするため、
            # 常に float32 のまま保持する（float64 にするとメモリが倍になる）。
            inner.memory_bank = torch.vstack(self._selected_chunks).to(torch.float32)
            inner.embedding_store = []
            self._selected_chunks = []
            self._finalized = True

        def memory_bank_info(self) -> dict[str, Any]:
            """学習後の ``memory_bank`` の行数・dtype・バイト数を返す。"""
            bank = getattr(self.model, "memory_bank", None)
            shape = getattr(bank, "shape", None)
            if bank is None or shape is None or len(shape) < 2:
                return {"rows": None, "dtype": None, "bytes": None}
            return {
                "rows": int(shape[0]),
                "dtype": str(getattr(bank, "dtype", "")),
                "bytes": int(bank.numel() * bank.element_size()),
            }

    return ChunkedPatchcore(**patchcore_kwargs)
