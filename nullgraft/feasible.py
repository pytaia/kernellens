"""Алгоритм B1: диаметры допустимых множеств по патчам (Алгоритмы 1 и 2 статьи, Thm. 3.9).

Подробности: docs/04_algorithm_B1.md.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial import cKDTree

from nullgraft._io import atomic_write_text, fingerprint, slices_digest
from nullgraft.metrics import LocalizedMetric
from nullgraft.operators import RadonOperator

logger = logging.getLogger(__name__)

Placement = Literal["original", "center"]
_PCA_FIT_SAMPLES = 20_000
_PAIRWISE_BUDGET_BYTES = 128 * 2**20


@dataclass(frozen=True)
class SinogramPooling:
    """``Q_pool``: суммы по непересекающимся блокам синограммы с весом ``1/√b`` (ортонормированные строки)."""

    angle_block: int = 4
    det_block: int = 4

    def __call__(self, sino: NDArray) -> NDArray[np.float32]:
        s = np.asarray(sino, np.float64)
        n_a, n_d = s.shape[-2:]
        ba, bd = self.angle_block, self.det_block
        a_use, d_use = (n_a // ba) * ba, (n_d // bd) * bd
        s = s[..., :a_use, :d_use].reshape(*s.shape[:-2], a_use // ba, ba, d_use // bd, bd)
        pooled = s.sum(axis=(-3, -1)) / np.sqrt(ba * bd)
        return pooled.reshape(*pooled.shape[:-2], -1).astype(np.float32)


@dataclass(frozen=True)
class PCAProjection:
    """``Q_pca``: проекция на главные направления без центрирования (линейна, ``‖Qv‖ ≤ ‖v‖``)."""

    components: NDArray[np.float32]

    @classmethod
    def fit(cls, data: NDArray, n_components: int) -> PCAProjection:
        data = np.asarray(data, np.float64)
        n_components = min(n_components, *data.shape)
        _, _, vt = np.linalg.svd(data - data.mean(axis=0), full_matrices=False)
        return cls(vt[:n_components].astype(np.float32))

    def __call__(self, v: NDArray) -> NDArray[np.float32]:
        return (np.asarray(v, np.float32) @ self.components.T).astype(np.float32)


@dataclass
class PatchArchive:
    """Патчи ``p_n``, метаданные ``(срез, строка, столбец)`` и признаки ``Q_pool A(E_n p_n)``."""

    patches: NDArray[np.float32]
    meta: NDArray[np.int32]
    pooled: NDArray[np.float32]
    patch_size: int
    placement: Placement
    pooling: SinogramPooling

    def __len__(self) -> int:
        return int(self.patches.shape[0])


def patch_positions(shape: tuple[int, int], patch_size: int, stride: int) -> list[tuple[int, int]]:
    half = patch_size // 2
    rows = range(half, shape[0] - (patch_size - half) + 1, stride)
    cols = range(half, shape[1] - (patch_size - half) + 1, stride)
    return [(r, c) for r in rows for c in cols]


def embed_patches(
    patches: NDArray, centers: NDArray, shape: tuple[int, int], placement: Placement
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Вставка патчей с общим центром в нулевые изображения; возвращает (изображения, маску носителя)."""
    s = patches.shape[-1]
    half = s // 2
    r, c = (shape[0] // 2, shape[1] // 2) if placement == "center" else (int(centers[0]), int(centers[1]))
    out = np.zeros((patches.shape[0], *shape), np.float32)
    out[:, r - half : r - half + s, c - half : c - half + s] = patches
    support = np.zeros(shape, bool)
    support[r - half : r - half + s, c - half : c - half + s] = True
    return out, support


def _measure_group(
    op: RadonOperator, patches: NDArray, center: tuple[int, int], placement: Placement
) -> NDArray[np.float32]:
    imgs, support = embed_patches(patches, np.asarray(center), op.image_shape, placement)
    return op.forward(imgs, support=support)


def build_patch_archive(
    slices: Sequence[NDArray],
    op: RadonOperator,
    patch_size: int = 32,
    stride: int = 16,
    placement: Placement = "original",
    pooling: SinogramPooling | None = None,
    min_std: float = 0.0,
    out_dir: str | Path | None = None,
    checkpoint_every: int = 16,
) -> PatchArchive:
    """Нарезка срезов на патчи и признаки их измерений; с ``out_dir`` — memmap и возобновляемые чекпоинты."""
    pooling = pooling or SinogramPooling()
    positions = patch_positions(op.image_shape, patch_size, stride)
    if not positions:
        raise ValueError("patch_size larger than the image")
    n_max = len(slices) * len(positions)
    feat_dim = pooling(np.zeros(op.sino_shape, np.float32)).size
    progress: dict = {"done_slices": 0, "count": 0}
    if out_dir is not None:
        build_fp = fingerprint({
            "op": op.fingerprint(), "patch_size": patch_size, "stride": stride, "placement": placement,
            "pooling": [pooling.angle_block, pooling.det_block], "min_std": min_std, "n_max": n_max,
            "slices": slices_digest(slices),
        })
        # своя подпапка на отпечаток: на Windows нельзя перезаписать файлы, открытые как memmap
        out = Path(out_dir) / f"fp_{build_fp[:16]}"
        out.mkdir(parents=True, exist_ok=True)
        prog_file = out / "progress.json"
        mode = "w+"
        if prog_file.exists():
            saved = json.loads(prog_file.read_text())
            if saved.get("fingerprint") == build_fp:
                progress, mode = saved, "r+"
                logger.info("resuming patch archive at slice %d", progress["done_slices"])
        progress["fingerprint"] = build_fp
        open_mm = np.lib.format.open_memmap
        patches = open_mm(out / "patches.npy", mode=mode, dtype=np.float32, shape=(n_max, patch_size, patch_size))
        meta = open_mm(out / "meta.npy", mode=mode, dtype=np.int32, shape=(n_max, 3))
        pooled = open_mm(out / "pooled.npy", mode=mode, dtype=np.float32, shape=(n_max, feat_dim))
    else:
        patches = np.zeros((n_max, patch_size, patch_size), np.float32)
        meta = np.zeros((n_max, 3), np.int32)
        pooled = np.zeros((n_max, feat_dim), np.float32)

    half = patch_size // 2
    count = int(progress["count"])
    chunk_start = count
    for si in range(int(progress["done_slices"]), len(slices)):
        img = np.asarray(slices[si], np.float32)
        if img.shape != op.image_shape:
            raise ValueError(f"slice {si} has shape {img.shape}, operator expects {op.image_shape}")
        for r, c in positions:
            p = img[r - half : r - half + patch_size, c - half : c - half + patch_size]
            if float(p.std()) < min_std:
                continue
            patches[count] = p
            meta[count] = (si, r, c)
            count += 1
        if (si + 1) % checkpoint_every == 0 or si + 1 == len(slices):
            _measure_into(op, patches, meta, pooled, np.arange(chunk_start, count), placement, pooling)
            chunk_start = count
            if out_dir is not None:
                for arr in (patches, meta, pooled):
                    arr.flush()
                atomic_write_text(out / "progress.json", json.dumps(
                    {"done_slices": si + 1, "count": count, "fingerprint": progress["fingerprint"]}))
    logger.info("patch archive: %d patches from %d slices", count, len(slices))
    return PatchArchive(patches[:count], meta[:count], pooled[:count], patch_size, placement, pooling)


def _measure_into(
    op: RadonOperator,
    patches: NDArray,
    meta: NDArray,
    pooled: NDArray,
    indices: NDArray[np.int64],
    placement: Placement,
    pooling: SinogramPooling,
) -> None:
    if indices.size == 0:
        return
    if placement == "center":
        groups = [indices]
    else:
        keys = meta[indices, 1].astype(np.int64) * 100_000 + meta[indices, 2]
        groups = [indices[keys == k] for k in np.unique(keys)]
    for grp in groups:
        center = (int(meta[grp[0], 1]), int(meta[grp[0], 2]))
        pooled[grp] = pooling(_measure_group(op, np.asarray(patches[grp]), center, placement))


def patch_measurement(op: RadonOperator, archive: PatchArchive, indices: NDArray[np.int64]) -> NDArray[np.float32]:
    """Точные ``A(E_n p_n)`` для ``indices`` (пересчёт на лету, группами по позиции)."""
    indices = np.asarray(indices, np.int64)
    out = np.zeros((indices.size, *op.sino_shape), np.float32)
    if archive.placement == "center":
        groups = [np.arange(indices.size)]
    else:
        keys = archive.meta[indices, 1].astype(np.int64) * 100_000 + archive.meta[indices, 2]
        groups = [np.flatnonzero(keys == k) for k in np.unique(keys)]
    for g in groups:
        sel = indices[g]
        center = (int(archive.meta[sel[0], 1]), int(archive.meta[sel[0], 2]))
        out[g] = _measure_group(op, np.asarray(archive.patches[sel]), center, archive.placement)
    return out


@dataclass(frozen=True)
class FeasibleInfo:
    """Результат Алгоритма 1: разреженная ``FA`` (N × K) и статистика кандидатов."""

    fa: sparse.csr_matrix
    n_candidates: int
    approximate_search: bool

    @property
    def com(self) -> sparse.csr_matrix:
        """``Com = FA·FAᵀ``: ненулевой элемент — пара в общем допустимом множестве."""
        fa = self.fa.astype(np.int32)
        return (fa @ fa.T).tocsr()


class FeasibleSetEstimator:
    """Алгоритмы 1 и 2 на архиве патчей."""

    def __init__(
        self,
        op: RadonOperator,
        archive: PatchArchive,
        n_components: int | None = 32,
        index: Literal["kdtree", "faiss"] = "kdtree",
        faiss_nprobe: int = 16,
        seed: int = 0,
    ) -> None:
        self.op = op
        self.archive = archive
        self.seed = seed
        self.index_kind = index
        self.faiss_nprobe = faiss_nprobe
        pooled = np.asarray(archive.pooled)
        if n_components is not None and n_components < pooled.shape[1]:
            rng = np.random.default_rng(seed)
            sample = pooled if len(pooled) <= _PCA_FIT_SAMPLES else pooled[rng.choice(len(pooled), _PCA_FIT_SAMPLES,
                                                                                      replace=False)]
            self.pca: PCAProjection | None = PCAProjection.fit(sample, n_components)
        else:
            self.pca = None
        self._features = self._reduce(pooled)
        self._tree = cKDTree(self._features) if index == "kdtree" else None
        self._faiss = self._build_faiss() if index == "faiss" else None

    def _reduce(self, pooled: NDArray) -> NDArray[np.float32]:
        return self.pca(pooled) if self.pca is not None else np.asarray(pooled, np.float32)

    def features_of(self, ys: NDArray) -> NDArray[np.float32]:
        return self._reduce(self.archive.pooling(ys))

    def _build_faiss(self):  # pragma: no cover
        import faiss

        d = self._features.shape[1]
        nlist = max(1, int(np.sqrt(len(self._features))))
        quant = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFFlat(quant, d, nlist)
        idx.train(self._features)
        idx.add(self._features)
        idx.nprobe = self.faiss_nprobe
        return idx

    def _candidates(self, feats: NDArray[np.float32], eps: float) -> list[NDArray[np.int64]]:
        if self._tree is not None:
            return [np.asarray(c, np.int64) for c in self._tree.query_ball_point(feats, r=eps, workers=-1)]
        lims, _, labels = self._faiss.range_search(feats, float(eps) ** 2)  # pragma: no cover
        return [labels[lims[i] : lims[i + 1]].astype(np.int64) for i in range(len(feats))]  # pragma: no cover

    def feasible_matrix(self, ys: NDArray, eps: float, batch_size: int = 256) -> FeasibleInfo:
        """Алгоритм 1: ``FA[n, k] = 1`` ⇔ ``‖A(E_n p_n) − y_k‖ < ε`` (точно, после фильтра кандидатов)."""
        ys = np.asarray(ys, np.float32)
        if ys.ndim == 2:
            ys = ys[None]
        k_count = ys.shape[0]
        cand_rows: list[NDArray] = []
        cand_cols: list[NDArray] = []
        for start in range(0, k_count, batch_size):
            chunk = ys[start : start + batch_size]
            for j, cands in enumerate(self._candidates(self.features_of(chunk), eps)):
                cand_rows.append(cands)
                cand_cols.append(np.full(cands.size, start + j, np.int64))
        rows = np.concatenate(cand_rows) if cand_rows else np.array([], np.int64)
        cols = np.concatenate(cand_cols) if cand_cols else np.array([], np.int64)
        keep = np.zeros(rows.size, bool)
        order = np.argsort(rows, kind="stable")
        uniq, first = np.unique(rows[order], return_index=True)
        bounds = np.append(first, rows.size)
        for chunk_start in range(0, uniq.size, batch_size):
            pids = uniq[chunk_start : chunk_start + batch_size]
            meas = patch_measurement(self.op, self.archive, pids)
            for i, pid in enumerate(pids):
                sel = order[bounds[chunk_start + i] : bounds[chunk_start + i + 1]]
                diff = meas[i][None] - ys[cols[sel]]
                keep[sel] = np.linalg.norm(diff.reshape(sel.size, -1), axis=1) < eps
        fa = sparse.csr_matrix(
            (np.ones(int(keep.sum()), bool), (rows[keep], cols[keep])), shape=(len(self.archive), k_count)
        )
        logger.info("Algorithm 1: %d candidates, %d feasible pairs", rows.size, int(keep.sum()))
        return FeasibleInfo(fa, int(rows.size), approximate_search=self.index_kind == "faiss")


@dataclass(frozen=True)
class DiameterResult:
    """``diam(F^N_{y_k})`` — оценка снизу ``diam(F_{y_k})`` — и пара, на которой он достигается."""

    k: int
    size: int
    diameter: float
    pair: tuple[int, int] | None
    approximate: bool

    @property
    def decoder_error_lower_bound(self) -> float:
        """Любой декодер ошибается хотя бы на ``diam / 2`` на некотором элементе ``F_{y_k}``."""
        return self.diameter / 2.0


def _pairwise_max(points: NDArray, metric: LocalizedMetric, chunk: int | None = None) -> tuple[float, tuple[int, int]]:
    best, pair = 0.0, (0, 0)
    if chunk is None:
        per_row = 3 * 8 * len(points) * int(np.prod(points.shape[1:]))
        chunk = max(1, _PAIRWISE_BUDGET_BYTES // max(per_row, 1))
    for i in range(0, len(points), chunk):
        block = points[i : i + chunk]
        d = np.asarray(metric(block[:, None], points[None]))
        j = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[j] > best:
            best, pair = float(d[j]), (i + int(j[0]), int(j[1]))
    return best, pair


def _farthest_point_diameter(
    points: NDArray, metric: LocalizedMetric, n_restarts: int, rng: np.random.Generator
) -> tuple[float, tuple[int, int]]:
    """Повторные проходы «дальней точки»; результат не превышает точного диаметра."""
    best, pair = 0.0, (0, 0)
    for _ in range(n_restarts):
        a = int(rng.integers(len(points)))
        for _ in range(3):
            d = np.asarray(metric(points, points[a][None]))
            b = int(np.argmax(d))
            if d[b] > best:
                best, pair = float(d[b]), (a, b)
            a = b
    return best, pair


def diameters(
    archive: PatchArchive,
    info: FeasibleInfo,
    metric: LocalizedMetric | None = None,
    exact_limit: int = 1000,
    n_restarts: int = 8,
    subset: NDArray[np.int64] | None = None,
    seed: int = 0,
) -> list[DiameterResult]:
    """Алгоритм 2: диаметр каждого эмпирического допустимого множества; больше ``exact_limit`` — приближённо."""
    metric = metric or LocalizedMetric()
    rng = np.random.default_rng(seed)
    fa = info.fa.tocsc()
    allowed = None
    if subset is not None:
        allowed = np.zeros(fa.shape[0], bool)
        allowed[subset] = True
    results = []
    for k in range(fa.shape[1]):
        members = fa.indices[fa.indptr[k] : fa.indptr[k + 1]]
        if allowed is not None:
            members = members[allowed[members]]
        members = np.sort(members)
        if members.size < 2:
            results.append(DiameterResult(k, int(members.size), 0.0, None, approximate=False))
            continue
        pts = np.asarray(archive.patches[members], np.float32)
        approx = members.size > exact_limit
        if approx:
            dval, (i, j) = _farthest_point_diameter(pts, metric, n_restarts, rng)
        else:
            dval, (i, j) = _pairwise_max(pts, metric)
        results.append(DiameterResult(k, int(members.size), dval, (int(members[i]), int(members[j])), approx))
    return results


def convergence_curve(
    archive: PatchArchive,
    info: FeasibleInfo,
    fractions: Sequence[float] = tuple(np.round(np.arange(0.1, 1.01, 0.1), 2)),
    metric: LocalizedMetric | None = None,
    exact_limit: int = 1000,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Кривая сходимости: диаметры на вложенных подвыборках архива; оценка растёт и сходится снизу."""
    order = np.random.default_rng(seed).permutation(len(archive))
    fr = np.asarray(fractions, np.float64)
    curve = np.zeros((fr.size, info.fa.shape[1]))
    for i, f in enumerate(fr):
        sub = order[: max(1, int(round(f * len(archive))))]
        curve[i] = [r.diameter for r in diameters(archive, info, metric, exact_limit, subset=sub, seed=seed)]
    return fr, curve
