"""Алгоритм A: проверка подозрительной детали для конкретного декодера (Thm. 3.2, Sec. 4.2).

Подробности: docs/03_algorithm_A.md.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.transform import resize

from nullgraft.decoders import Decoder
from nullgraft.graft import Grafter, GraftResult, region_center
from nullgraft.metrics import LocalizedMetric
from nullgraft.operators import RadonOperator

logger = logging.getLogger(__name__)

_NOISE_RADIUS_FRACTION = 0.99
_LSQR_ITER_LIMIT = 7
_KERNEL_RATIO_WARN = 1e-2


class Outcome(str, Enum):
    INCONSISTENT = "INCONSISTENT"
    BLEND_INVALID = "BLEND_INVALID"
    LIKELY_HALLUCINATION_RISK = "LIKELY_HALLUCINATION_RISK"
    DETAIL_VISIBLE = "DETAIL_VISIBLE"
    UNINFORMATIVE = "UNINFORMATIVE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class RealismVerdict:
    realistic: bool
    score: float
    threshold: float


class RealismChecker(Protocol):
    """Решает, правдоподобен ли ``x''`` (``x'' ∈ M1``); можно подставить свой классификатор."""

    def check(self, x: NDArray, region: NDArray[np.bool_]) -> RealismVerdict: ...


class AlwaysRealistic:
    """Принимает всё; только для синтетических тестов."""

    def check(self, x: NDArray, region: NDArray[np.bool_]) -> RealismVerdict:
        return RealismVerdict(True, 0.0, float("inf"))


def _extract_patch(image: NDArray, center: tuple[int, int], size: int) -> NDArray[np.float32]:
    half = size // 2
    padded = np.pad(np.asarray(image, np.float32), half, mode="edge")
    r, c = center[0] + half, center[1] + half
    return padded[r - half : r - half + size, c - half : c - half + size]


def patch_features(patch: NDArray, cells: int = 8) -> NDArray[np.float32]:
    """Блочные средние яркости и модуля градиента на сетке ``cells × cells``."""
    patch = np.asarray(patch, np.float32)
    gy, gx = np.gradient(patch)
    grad = np.hypot(gy, gx)
    a = resize(patch, (cells, cells), anti_aliasing=True, preserve_range=True)
    b = resize(grad, (cells, cells), anti_aliasing=True, preserve_range=True)
    return np.concatenate([a.ravel(), b.ravel()]).astype(np.float32)


class ArchivePatchRealism:
    """Реалистичность по умолчанию: расстояние до ближайшего патча архива в пространстве признаков."""

    def __init__(self, patch_size: int = 32, cells: int = 8, quantile: float = 0.99, margin: float = 1.0) -> None:
        self.patch_size = patch_size
        self.cells = cells
        self.quantile = quantile
        self.margin = margin
        self._tree: cKDTree | None = None
        self.threshold = float("nan")

    def fit(self, images: Iterable[NDArray], stride: int = 16, max_patches: int = 200_000,
            seed: int = 0, k_search: int = 16) -> ArchivePatchRealism:
        """Индексирует патчи архива; порог — квантиль расстояний до соседей из других срезов."""
        feats, owner = [], []
        half = self.patch_size // 2
        for n, img in enumerate(images):
            padded = np.pad(np.asarray(img, np.float32), half, mode="edge")
            h, w = img.shape
            for r in range(half, h - half + 1, stride):
                for c in range(half, w - half + 1, stride):
                    feats.append(patch_features(padded[r : r + self.patch_size, c : c + self.patch_size], self.cells))
                    owner.append(n)
        if len(feats) < 2 or len(set(owner)) < 2:
            raise ValueError("realism model needs at least two archive images")
        data, owner_arr = np.stack(feats), np.asarray(owner)
        if len(data) > max_patches:
            keep = np.random.default_rng(seed).choice(len(data), max_patches, replace=False)
            data, owner_arr = data[keep], owner_arr[keep]
        self._tree = cKDTree(data)
        k = min(k_search, len(data))
        dists, idx = self._tree.query(data, k=k)
        other = owner_arr[idx] != owner_arr[:, None]
        # первый сосед из другого среза; если среди k такого нет — k-е расстояние
        first = np.where(other.any(axis=1), np.argmax(other, axis=1), k - 1)
        self.threshold = float(np.quantile(dists[np.arange(len(data)), first], self.quantile)) * self.margin
        return self

    def check(self, x: NDArray, region: NDArray[np.bool_]) -> RealismVerdict:
        if self._tree is None:
            raise RuntimeError("ArchivePatchRealism.fit() must be called first")
        feat = patch_features(_extract_patch(x, region_center(region), self.patch_size), self.cells)
        dist = float(self._tree.query(feat, k=1)[0])
        return RealismVerdict(dist <= self.threshold, dist, self.threshold)


@dataclass(frozen=True)
class AssessmentConfig:
    """Параметры Алгоритма A (смысл каждого — в docs/03_algorithm_A.md)."""

    eps: float
    n_noise: int = 20
    noise_sigma: float = 0.0
    p: float = 1.0
    q: float = 1.0
    visible_fraction: float = 0.1
    follow_fraction: float = 0.5
    null_threshold: float = 0.0
    run_decoder_if_unrealistic: bool = False
    seed: int = 0


@dataclass(frozen=True)
class Assessment:
    """Исход проверки, интервал ``[η_min, η_max]`` и диагностика."""

    outcome: Outcome
    consistency_residual: float
    eps: float
    eta_min: float = float("nan")
    eta_max: float = float("nan")
    graft_size: float = float("nan")
    detail_size: float = float("nan")
    projected_size: float = float("nan")
    kernel_ratio: float = float("nan")
    delta: float = float("nan")
    delta_limit: float = float("nan")
    realism: RealismVerdict | None = None
    d_to_z: tuple[float, ...] = ()
    d_to_graft: tuple[float, ...] = ()
    consistent_output_fraction: float = float("nan")
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def interval_nonempty(self) -> bool:
        return bool(self.eta_min <= self.eta_max and self.eta_min <= self.graft_size / 2)

    def summary(self) -> str:
        return (
            f"{self.outcome.value}: eta in [{self.eta_min:.4g}, {self.eta_max:.4g}], "
            f"||z-x''||={self.graft_size:.4g}, ||x_det||={self.detail_size:.4g}, "
            f"Delta={self.delta:.3g} (limit {self.delta_limit:.3g}), consistency={self.consistency_residual:.3g}"
            " - a bound on what is possible, not a diagnosis"
        )


def select_point(samples: NDArray) -> NDArray[np.float32]:
    """Медоид семплов декодера — элемент ``z ∈ φ(y)``."""
    if samples.shape[0] == 1:
        return samples[0]
    mean = samples.mean(axis=0)
    return samples[int(np.argmin(np.linalg.norm((samples - mean).reshape(len(samples), -1), axis=1)))]


def sample_noise(shape: tuple[int, ...], n: int, sigma: float, eps: float, rng: np.random.Generator) -> NDArray:
    """``n`` гауссовых реализаций шума, радиально ужатых до ``‖e‖ < eps``."""
    noise = sigma * rng.standard_normal((n, *shape))
    norms = np.linalg.norm(noise.reshape(n, -1), axis=1)
    limit = _NOISE_RADIUS_FRACTION * eps
    factor = np.where(norms > limit, limit / np.maximum(norms, 1e-300), 1.0)
    if np.any(factor < 1.0):
        logger.debug("shrunk %d/%d noise draws into the eps-ball", int(np.sum(factor < 1.0)), n)
    return (noise * factor.reshape(n, *([1] * len(shape)))).astype(np.float32)


class DetailAssessor:
    """Запуск Алгоритма A для заданных оператора и декодера."""

    def __init__(
        self,
        op: RadonOperator,
        decoder: Decoder,
        config: AssessmentConfig,
        realism: RealismChecker | None = None,
        grafter: Grafter | None = None,
    ) -> None:
        self.op = op
        self.decoder = decoder
        self.config = config
        if config.noise_sigma <= 0:
            logger.warning("noise_sigma <= 0: all %d noise draws are zero, eta_min/eta_max then describe a single "
                           "noiseless decode only", config.n_noise)
        if realism is None:
            logger.warning("no RealismChecker given: x'' is assumed realistic (check it separately)")
        self.realism = realism or AlwaysRealistic()
        self.grafter = grafter or Grafter(op)

    def consistency(self, z: NDArray, y: NDArray) -> float:
        return float(np.linalg.norm(self.op.forward(z) - np.asarray(y, np.float32)))

    def assess(self, y: NDArray, source: NDArray, region: NDArray[np.bool_], z: NDArray | None = None) -> Assessment:
        """Оценить содержимое ``z`` в ``region`` против альтернативы из ``source``."""
        cfg = self.config
        region = np.asarray(region, dtype=bool)
        if z is None:
            z = select_point(self.decoder.reconstruct(y))
        resid = self.consistency(z, y)
        base = {"consistency_residual": resid, "eps": cfg.eps}
        if resid >= cfg.eps:
            return Assessment(Outcome.INCONSISTENT, notes=("||A z - y|| >= eps: analysis not applicable",), **base)

        g = self.grafter.graft(z, source, region, cfg.eps)
        metric = LocalizedMetric(region, cfg.p, cfg.q)
        graft_size = float(metric.norm(g.blended))
        detail_size = float(metric.norm(g.x_det))
        projected_size = float(metric.norm(g.x_proj))
        base |= {
            "graft_size": graft_size, "detail_size": detail_size, "projected_size": projected_size,
            "kernel_ratio": g.kernel_ratio, "delta": g.delta, "delta_limit": g.delta_limit,
        }
        notes: list[str] = []
        # остановка по iter_lim — норма; проблема только при большом остаточном отношении
        if g.projection.istop == _LSQR_ITER_LIMIT and g.kernel_ratio > _KERNEL_RATIO_WARN:
            notes.append(f"LSQR not converged (istop={g.projection.istop}, ||A x_proj||/||A x_det||="
                         f"{g.kernel_ratio:.2e}); check Delta and consider a larger iter_lim")
        if not g.valid:
            return Assessment(Outcome.BLEND_INVALID, notes=tuple(notes + ["Delta > delta_factor * eps"]), **base)
        if detail_size == 0.0:
            return Assessment(Outcome.INCONCLUSIVE, notes=tuple(notes + ["null detail"]), **base)
        if projected_size <= cfg.visible_fraction * detail_size:
            return Assessment(Outcome.DETAIL_VISIBLE, notes=tuple(
                notes + ["projection removed most of the detail: x'' stays close to z"]), **base)
        if graft_size <= cfg.visible_fraction * detail_size:
            notes.append("blending window removed most of the projected detail: region R too small for the window")
            return Assessment(Outcome.INCONCLUSIVE, notes=tuple(notes), **base)

        verdict = self.realism.check(g.x_graft, region)
        base["realism"] = verdict
        if not verdict.realistic and not cfg.run_decoder_if_unrealistic:
            return Assessment(Outcome.UNINFORMATIVE, notes=tuple(notes + ["x'' is not realistic"]), **base)

        d_z, d_g, consistent = self._decoder_distances(g, z, metric)
        eta_min, eta_max = float(np.max(d_z)), float(np.min(d_g))
        base |= {
            "eta_min": eta_min, "eta_max": eta_max, "d_to_z": tuple(map(float, d_z)),
            "d_to_graft": tuple(map(float, d_g)), "consistent_output_fraction": consistent,
        }
        if consistent < 1.0:
            notes.append(f"only {consistent:.0%} of decoder outputs on y'' are eps-consistent")
        if not verdict.realistic:
            return Assessment(Outcome.UNINFORMATIVE, notes=tuple(notes + ["x'' is not realistic"]), **base)
        if eta_min <= eta_max and eta_min <= graft_size / 2 and graft_size > 2 * cfg.null_threshold:
            return Assessment(Outcome.LIKELY_HALLUCINATION_RISK, notes=tuple(notes), **base)
        if float(np.max(d_g)) <= cfg.follow_fraction * graft_size:
            return Assessment(Outcome.DETAIL_VISIBLE, notes=tuple(notes + ["decoder reproduces x''"]), **base)
        return Assessment(Outcome.INCONCLUSIVE, notes=tuple(notes), **base)

    def _decoder_distances(
        self, g: GraftResult, z: NDArray, metric: LocalizedMetric
    ) -> tuple[NDArray, NDArray, float]:
        """Один батчевый вызов декодера на ``y''_m = A x'' + e_m``."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        y_clean = self.op.forward(g.x_graft)
        # ‖e‖ < ε − Δ оставляет z допустимым для каждого y''_m
        radius = max(cfg.eps - g.delta, 0.0)
        ys = y_clean[None] + sample_noise(y_clean.shape, cfg.n_noise, cfg.noise_sigma, radius, rng)
        outs = self.decoder.reconstruct_batch(ys)
        d_z = np.max(metric(outs, np.asarray(z)[None, None]), axis=1)
        d_g = np.max(metric(outs, g.x_graft[None, None]), axis=1)
        m, l = outs.shape[:2]
        flat = outs.reshape(m * l, *outs.shape[2:])
        resid = np.linalg.norm((self.op.forward(flat) - np.repeat(ys, l, axis=0)).reshape(m * l, -1), axis=1)
        return d_z, d_g, float(np.mean(resid < cfg.eps))


@dataclass(frozen=True)
class CalibrationResult:
    """Порог шума пайплайна, доля ложных срабатываний и записи прогонов."""

    threshold: float
    false_positive_rate: float
    outcomes: dict[str, int]
    records: tuple[tuple[int, float, str, float, float], ...]


def calibrate_null_detail(
    x_slices: Sequence[NDArray],
    op: RadonOperator,
    decoder: Decoder,
    config: AssessmentConfig,
    region: NDArray[np.bool_],
    scales: Sequence[float] = (0.0, 1e-3, 1e-2),
    quantile: float = 0.99,
    data_noise_sigma: float | None = None,
    seed: int = 0,
    n_jobs: int = 1,
) -> CalibrationResult:
    """Прогон Алгоритма A с нулевыми и очень малыми деталями на ``N`` срезах."""
    cal_cfg = AssessmentConfig(**{**config.__dict__, "null_threshold": 0.0, "run_decoder_if_unrealistic": True,
                                  "visible_fraction": -1.0})
    sigma = config.noise_sigma if data_noise_sigma is None else data_noise_sigma

    def run_slice(n: int, x: NDArray) -> list[tuple[int, float, str, float, float]]:
        rng = np.random.default_rng([seed, n])
        assessor = DetailAssessor(op, decoder, cal_cfg, realism=AlwaysRealistic())
        y = op.forward(x) + sample_noise(op.sino_shape, 1, sigma, config.eps, rng)[0]
        z = select_point(decoder.reconstruct(y))
        pattern = gaussian_filter(rng.standard_normal(op.image_shape), 2.0)
        pattern = (pattern / max(np.abs(pattern).max(), 1e-12)).astype(np.float32)
        dyn = float(np.ptp(x)) or 1.0
        rows = []
        for s in scales:
            res = assessor.assess(y, z + s * dyn * pattern * region, region, z=z)
            if res.outcome in (Outcome.INCONSISTENT, Outcome.BLEND_INVALID) or np.isnan(res.eta_min):
                eta = _decoder_floor(assessor, z, region, cal_cfg)
            else:
                eta = res.eta_min
            rows.append((n, float(s), res.outcome.value, eta, float(np.nan_to_num(res.graft_size))))
        return rows

    results = Parallel(n_jobs=n_jobs)(delayed(run_slice)(n, np.asarray(x, np.float32)) for n, x in enumerate(x_slices))
    records = tuple(r for rows in results for r in rows)
    if not records:
        raise ValueError("no slices given")
    etas = np.array([r[3] for r in records])
    outcomes: dict[str, int] = {}
    for r in records:
        outcomes[r[2]] = outcomes.get(r[2], 0) + 1
    fpr = outcomes.get(Outcome.LIKELY_HALLUCINATION_RISK.value, 0) / len(records)
    threshold = float(np.quantile(etas, quantile))
    logger.info("null-detail calibration: threshold=%.4g, false-positive rate=%.3f", threshold, fpr)
    return CalibrationResult(threshold, fpr, outcomes, records)


def _decoder_floor(assessor: DetailAssessor, z: NDArray, region: NDArray[np.bool_], cfg: AssessmentConfig) -> float:
    """``max_m d(φ(A z + e_m), z)`` — флуктуация декодера вокруг ``z`` без вклейки."""
    rng = np.random.default_rng(cfg.seed)
    y0 = assessor.op.forward(z)
    ys = y0[None] + sample_noise(y0.shape, cfg.n_noise, cfg.noise_sigma, cfg.eps, rng)
    outs = assessor.decoder.reconstruct_batch(ys)
    metric = LocalizedMetric(region, cfg.p, cfg.q)
    return float(np.max(metric(outs, np.asarray(z)[None, None])))
