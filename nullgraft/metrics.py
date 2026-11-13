"""Метрики: локализованная норма ‖·‖_{p,q,R}, RMSE, SSIM, ошибка в HU, Dice.

Подробности: docs/07_metrics.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

DistanceFn = Callable[[NDArray, NDArray], NDArray | float]
Regions = NDArray[np.bool_] | Sequence[NDArray[np.bool_]] | None


def _region(mask: NDArray[np.bool_] | None, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(shape[-2:], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(shape[-2:]):
        raise ValueError(f"mask shape {mask.shape} does not match image shape {shape[-2:]}")
    if not mask.any():
        raise ValueError("region of interest R is empty")
    return mask


def _regions(regions: Regions, shape: tuple[int, ...]) -> list[NDArray[np.bool_]]:
    if regions is None or (isinstance(regions, np.ndarray) and regions.ndim == 2):
        return [_region(regions, shape)]
    return [_region(r, shape) for r in regions]


def _lq_over_region(x: NDArray[np.float64], mask: NDArray[np.bool_], q: float, normalize: bool) -> NDArray:
    # для области «всё изображение» не копируем через булеву маску (узкое место по профилю)
    vals = np.abs(x).reshape(*x.shape[:-2], -1) if mask.all() else np.abs(x[..., mask])
    if not np.isfinite(q):
        return vals.max(axis=-1)
    out = np.sum(vals**q, axis=-1) ** (1.0 / q)
    return out / mask.sum() ** (1.0 / q) if normalize else out


def localized_norm(
    x: NDArray,
    region: Regions = None,
    p: float = 1.0,
    q: float = 1.0,
    normalize: bool = True,
) -> NDArray[np.float64] | float:
    """``‖x‖_{p,q,R}`` для ``x`` формы ``(..., H, W)``; ``region`` — маска, список масок или ``None``."""
    if p < 1 or q < 1:
        raise ValueError("p, q must be >= 1 for a norm")
    x = np.asarray(x, dtype=np.float64)
    per_region = np.stack([_lq_over_region(x, m, q, normalize) for m in _regions(region, x.shape)], axis=-1)
    out = per_region.max(axis=-1) if not np.isfinite(p) else np.sum(per_region**p, axis=-1) ** (1.0 / p)
    return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class LocalizedMetric:
    """Расстояние ``d(a, b) = ‖a − b‖_{p,q,R}``."""

    region: Regions = None
    p: float = 1.0
    q: float = 1.0
    normalize: bool = True

    def __call__(self, a: NDArray, b: NDArray) -> NDArray[np.float64] | float:
        return localized_norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), self.region, self.p, self.q,
                              self.normalize)

    def norm(self, x: NDArray) -> NDArray[np.float64] | float:
        return localized_norm(x, self.region, self.p, self.q, self.normalize)


def hausdorff_to_point(samples: NDArray, x: NDArray, metric: DistanceFn) -> float:
    """``d(x, φ(y)) = max_u d(u, x)`` по семплам ``(L, H, W)`` стохастического декодера."""
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples[None]
    return float(np.max(metric(samples, np.asarray(x)[None])))


def rmse(a: NDArray, b: NDArray, region: NDArray[np.bool_] | None = None) -> float:
    mask = _region(region, np.shape(a))
    diff = (np.asarray(a, np.float64) - np.asarray(b, np.float64))[..., mask]
    return float(np.sqrt(np.mean(diff**2)))


def ssim(a: NDArray, b: NDArray, data_range: float | None = None) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min())) or 1.0
    return float(structural_similarity(a, b, data_range=data_range))


def to_hu(mu: NDArray, mu_water: float, mu_air: float = 0.0) -> NDArray[np.float64]:
    """Ослабление в единицы Хаунсфилда: ``HU = 1000 (μ − μ_w) / (μ_w − μ_air)``."""
    return 1000.0 * (np.asarray(mu, np.float64) - mu_water) / (mu_water - mu_air)


def hu_error(a: NDArray, b: NDArray, mu_water: float, region: NDArray[np.bool_] | None = None) -> float:
    """Средняя абсолютная ошибка в HU по ``R``."""
    mask = _region(region, np.shape(a))
    return float(np.mean(np.abs(to_hu(a, mu_water) - to_hu(b, mu_water))[..., mask]))


def dice(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    """Коэффициент Дайса ``2|a∩b| / (|a| + |b|)``; 1.0 для двух пустых масок."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)
