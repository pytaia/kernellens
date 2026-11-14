"""Этапы 6–9 демо: сравнение декодеров, похожие артефакты (B2), предел задачи (B1), карта риска."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nullgraft.api import default_eps
from nullgraft.assess import AlwaysRealistic, AssessmentConfig, DetailAssessor
from nullgraft.feasible import (
    FeasibleSetEstimator,
    SinogramPooling,
    build_patch_archive,
    convergence_curve,
    diameters,
    embed_patches,
)
from nullgraft.graft import Grafter
from nullgraft.metrics import LocalizedMetric, ssim
from nullgraft.similar import ArtifactIndex
from nullgraft_demo.compute import ReconResult, get_operator, make_decoder, reconstruct
from nullgraft_demo.datasource import ArraySource
from nullgraft_demo.detail import CheckResult, detail_source, projector_for, run_check
from nullgraft_demo.state import DetailParams, ReconParams
from nullgraft_demo.tools.mask import disk_region
from nullgraft_demo.workers import CancelToken, ProgressFn, ResultCache, cache_key

logger = logging.getLogger(__name__)


def archive_indices(n_slices: int, current: int, gap: int = 2, limit: int = 16) -> list[int]:
    """Срезы архива: подальше от текущего (соседние почти совпадают); при нехватке — любые другие."""
    idx = [i for i in range(n_slices) if abs(i - current) > gap]
    if len(idx) < 2:
        idx = [i for i in range(n_slices) if i != current]
    step = max(1, len(idx) // limit)
    return idx[::step][:limit]


def crop(img: NDArray, center: tuple[float, float], size: int) -> NDArray:
    half = size // 2
    r, c = int(round(center[0])), int(round(center[1]))
    padded = np.pad(np.asarray(img, np.float32), half, mode="edge")
    return padded[r : r + size, c : c + size]


@dataclass(frozen=True)
class CompareRow:
    decoder: str
    recon: ReconResult
    check: CheckResult | None
    error: str
    metrics: dict[str, float]


def compare_decoders(
    x: NDArray, params: ReconParams, decoders: Sequence[str], mask: NDArray[np.bool_], detail: DetailParams,
    make_source: Callable[[NDArray], NDArray], cache: ResultCache | None, mu_water: float, realism: Any = None,
    *, token: CancelToken, progress: ProgressFn,
) -> list[CompareRow]:
    """Одна синограмма и один шум (общий seed), одна область и деталь — несколько декодеров."""
    rows = []
    metric = LocalizedMetric(mask)
    for i, name in enumerate(decoders):
        p = params.with_(decoder=name)
        base = i / len(decoders)

        def sub(f: float, m: str, base: float = base, name: str = name) -> None:
            progress(base + f / len(decoders), f"{name}: {m}")

        recon = reconstruct(x, p, cache, token=token, progress=sub)
        z = recon.z
        metrics = {
            "‖z − x‖ в R": float(metric(z, x)),
            "ошибка HU в R": float(np.mean(np.abs(1000.0 * (z - x) / mu_water)[mask])),
            "SSIM": ssim(z, x),
            "‖Az − y‖/ε": recon.consistency / recon.eps,
        }
        try:
            check = run_check(recon, make_source(z), mask, p, detail, realism=realism, token=token, progress=sub)
            err = ""
        except ValueError as exc:
            check, err = None, str(exc)
        rows.append(CompareRow(name, recon, check, err, metrics))
    progress(1.0, "сравнение готово")
    return rows


@dataclass
class SimilarItem:
    slice_index: int
    row: int
    col: int
    distance: float
    ncc: float
    is_structure: bool
    recon: NDArray
    residual: NDArray
    truth: NDArray


@dataclass
class SimilarResult:
    items: list[SimilarItem]
    fraction_real: float
    fraction_background: float
    n_index: int
    seconds: float


def similar_artifacts(
    source: ArraySource, current: int, z: NDArray, mask: NDArray[np.bool_], params: ReconParams,
    cache: ResultCache | None, index_dir: Path, patch_size: int = 32, k: int = 8,
    *, token: CancelToken, progress: ProgressFn,
) -> SimilarResult:
    """Индекс артефактов того же декодера по архиву (кэшируется по отпечатку), top-K для центра R."""
    t0 = time.perf_counter()
    idx = archive_indices(source.n_slices, current)
    op = get_operator(mask.shape, params.n_angles, params.beam)
    decoder = make_decoder(params, op)
    progress(0.05, f"индекс артефактов по {len(idx)} срезам")
    index = ArtifactIndex.build([source.get(i) for i in idx], op, decoder, params.sigma,
                                eps=default_eps(op, params.sigma, params.eps_factor), patch_size=patch_size,
                                stride=max(4, patch_size // 2), out_dir=index_dir, seed=params.seed)
    token.check()
    rows, cols = np.nonzero(mask)
    centre = (int(round(rows.mean())), int(round(cols.mean())))
    matches = index.query_from_image(z, [centre], k=k)[0]
    items = []
    for j, m in enumerate(matches):
        token.check()
        progress(0.6 + 0.4 * j / max(len(matches), 1), "загрузка совпадений")
        gi = idx[m.slice_index]
        x_m = source.get(gi)
        r = reconstruct(x_m, params, cache, token=token, progress=lambda f, s: None)
        pos = (m.row, m.col)
        items.append(SimilarItem(gi, m.row, m.col, m.descriptor_distance, m.ncc, m.is_structure,
                                 crop(r.z, pos, patch_size), crop(r.z - x_m, pos, patch_size), crop(x_m, pos, patch_size)))
    summary = ArtifactIndex.summarize([matches])
    progress(1.0, "похожие артефакты найдены")
    return SimilarResult(items, summary.fraction_real_structure, summary.fraction_background, len(index),
                         time.perf_counter() - t0)


@dataclass
class LimitResult:
    angles: list[int]
    sizes: list[int]
    diameters: list[float]
    current: int
    pair: tuple[NDArray, NDArray] | None
    pair_meta: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    fractions: NDArray
    curve: NDArray
    n_archive: int
    seconds: float
    from_cache: bool = False
    notes: list[str] = field(default_factory=list)


def feasible_limit(
    source: ArraySource, current: int, mask: NDArray[np.bool_], params: ReconParams, angles: Sequence[int],
    cache: ResultCache | None, patch_size: int = 24,
    *, token: CancelToken, progress: ProgressFn,
) -> LimitResult:
    """diam(F_y^N) и diam/2 для патча вокруг R в зависимости от N_θ; пара x, x'; кривая сходимости.

    Патч-адаптация (docs/04_algorithm_B1.md): патчи архива и запрос вставляются в центр изображения
    (``placement="center"``), ε — как для шума σ на измерении патча.
    """
    t0 = time.perf_counter()
    rows, cols = np.nonzero(mask)
    centre = (float(rows.mean()), float(cols.mean()))
    idx = archive_indices(source.n_slices, current, gap=0, limit=64)
    key = cache_key("b1-v1", source.name, source.n_slices, current, [round(c) for c in centre], patch_size,
                    list(angles), params.sigma, params.seed, params.beam, idx)
    if cache is not None and (hit := cache.get(key)) is not None:
        progress(1.0, "предел задачи из кэша")
        meta = hit["pair_meta"].astype(int)
        pair = (hit["pair_a"], hit["pair_b"]) if hit["pair_a"].size else None
        return LimitResult(hit["angles"].astype(int).tolist(), hit["sizes"].astype(int).tolist(),
                           hit["diameters"].tolist(), int(hit["current"]), pair,
                           (tuple(meta[0]), tuple(meta[1])) if pair is not None else None,
                           hit["fractions"], hit["curve"], int(hit["n_archive"]), time.perf_counter() - t0, True)
    slices = [source.get(i) for i in idx]
    query_patch = crop(source.get(current), centre, patch_size)
    sizes, diams = [], []
    pair = pair_meta = None
    fr, curve = np.array([]), np.zeros((0, 1))
    n_archive = 0
    for j, n in enumerate(angles):
        token.check()
        progress(0.05 + 0.9 * j / len(angles), f"B1: N_θ = {n}")
        op = get_operator(mask.shape, int(n), params.beam)
        archive = build_patch_archive(slices, op, patch_size, max(4, patch_size // 2), placement="center",
                                      pooling=SinogramPooling(2, 2))
        n_archive = len(archive)
        img, support = embed_patches(query_patch[None], np.array([0, 0]), mask.shape, "center")
        rng = np.random.default_rng(params.seed)
        y = op.forward(img, support=support) + params.sigma * rng.standard_normal((1, *op.sino_shape))
        est = FeasibleSetEstimator(op, archive, n_components=32, seed=params.seed)
        info = est.feasible_matrix(y.astype(np.float32), default_eps(op, params.sigma, params.eps_factor))
        res = diameters(archive, info, LocalizedMetric())[0]
        sizes.append(res.size)
        diams.append(res.diameter)
        if int(n) == params.n_angles:
            fr, curve = convergence_curve(archive, info)
            if res.pair is not None:
                a, b = res.pair
                pair = (np.asarray(archive.patches[a]), np.asarray(archive.patches[b]))
                ma, mb = archive.meta[a], archive.meta[b]
                pair_meta = ((idx[int(ma[0])], int(ma[1]), int(ma[2])), (idx[int(mb[0])], int(mb[1]), int(mb[2])))
    if cache is not None:
        cache.put(key, {
            "angles": np.asarray(angles), "sizes": np.asarray(sizes), "diameters": np.asarray(diams),
            "current": np.asarray(params.n_angles), "fractions": fr, "curve": curve, "n_archive": np.asarray(n_archive),
            "pair_a": pair[0] if pair else np.zeros(0), "pair_b": pair[1] if pair else np.zeros(0),
            "pair_meta": np.asarray(pair_meta if pair_meta else ((0, 0, 0), (0, 0, 0))),
        })
    progress(1.0, "предел задачи готов")
    return LimitResult(list(map(int, angles)), sizes, diams, params.n_angles, pair, pair_meta, fr, curve, n_archive,
                       time.perf_counter() - t0)


@dataclass
class RiskMap:
    centres: list[tuple[int, int]]
    risk: NDArray[np.float32]
    image: NDArray[np.float32]
    radius: int
    seconds: float


def risk_map(
    recon: ReconResult, params: ReconParams, step: int = 16, radius: int = 8, n_noise: int = 4,
    body_threshold: float = 0.05,
    *, token: CancelToken, progress: ProgressFn,
) -> RiskMap:
    """Грубая оценка: в каждом окне — доля реализаций шума, где φ(y'') ближе к z, чем к x'' (деталь «удалена»).

    Быстрый режим: M = ``n_noise``, проекция внутри окна, без проверки реалистичности.
    """
    t0 = time.perf_counter()
    shape = recon.z.shape
    op = get_operator(shape, params.n_angles, params.beam)
    decoder = make_decoder(params, op)
    cfg = AssessmentConfig(eps=recon.eps, n_noise=n_noise, noise_sigma=params.sigma, seed=params.seed)
    grafter = Grafter(op, projector_for(params, shape), window_width=2.0, projection="local")
    assessor = DetailAssessor(op, decoder, cfg, realism=AlwaysRealistic(), grafter=grafter)
    centres = [(r, c) for r in range(radius, shape[0] - radius, step) for c in range(radius, shape[1] - radius, step)]
    level = body_threshold * float(np.max(np.abs(recon.z)))
    centres = [rc for rc in centres if abs(recon.z[rc]) > level]
    image = np.full(shape, np.nan, np.float32)
    risk = np.zeros(len(centres), np.float32)
    for i, (r, c) in enumerate(centres):
        token.check()
        progress(i / max(len(centres), 1), f"карта риска: окно {i + 1} из {len(centres)}")
        mask = disk_region(shape, r, c, radius)
        res = assessor.assess(recon.y, detail_source(DetailParams(mode="remove"), recon.z, mask), mask, z=recon.z)
        if res.d_to_z:
            risk[i] = float(np.mean(np.asarray(res.d_to_z) < np.asarray(res.d_to_graft)))
        half = step // 2
        image[max(r - half, 0) : r + half, max(c - half, 0) : c + half] = risk[i]
    progress(1.0, "карта риска готова")
    return RiskMap(centres, risk, image, radius, time.perf_counter() - t0)
