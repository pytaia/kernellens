"""Проекция на ядро прямого оператора: P_N(x) = x − A⁺(A x), A⁺ через LSQR.

Подробности: docs/02_nullspace.md.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import lsqr

from nullgraft.operators import RadonOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LsqrConfig:
    """Критерии остановки LSQR."""

    atol: float = 1e-7
    btol: float = 1e-7
    iter_lim: int = 500


@dataclass(frozen=True)
class ProjectionResult:
    """Результат ``P_N(x)``: проекция, удалённая видимая часть ``A⁺Ax`` и диагностика LSQR."""

    x_proj: NDArray[np.float32]
    pinv_component: NDArray[np.float32]
    residual_ratio: float
    iterations: int
    istop: int
    from_cache: bool


def kernel_residual(op: RadonOperator, x: NDArray, reference: NDArray | None = None) -> float:
    """Видимость ``x`` в данных: ``||Ax|| / ||A ref||`` или ``||Ax|| / (||A|| ||x||)``."""
    ax = float(np.linalg.norm(op.forward(x)))
    if reference is not None:
        denom = float(np.linalg.norm(op.forward(reference)))
    else:
        denom = op.norm() * float(np.linalg.norm(x))
    return 0.0 if denom == 0.0 else ax / denom


class NullspaceProjector:
    """``A⁺`` и ``P_N`` через LSQR с LRU-кэшем по (деталь, маска, геометрия, допуски)."""

    def __init__(self, op: RadonOperator, config: LsqrConfig | None = None, cache_size: int = 64) -> None:
        self.op = op
        self.config = config or LsqrConfig()
        self.cache_size = cache_size
        self._linop = op.as_linear_operator()
        self._cache: OrderedDict[str, ProjectionResult] = OrderedDict()
        if not op.exact_adjoint:
            logger.warning("operator backend %s has an inexact adjoint; P_N is approximate", op.backend_name)

    def pinv(self, b: NDArray, x0: NDArray | None = None) -> tuple[NDArray[np.float64], int, int]:
        """``A⁺ b``; возвращает ``(w, istop, iterations)``."""
        cfg = self.config
        x0_flat = None if x0 is None else np.asarray(x0, np.float64).ravel()
        result = lsqr(
            self._linop,
            np.asarray(b, np.float64).ravel(),
            atol=cfg.atol,
            btol=cfg.btol,
            iter_lim=cfg.iter_lim,
            x0=x0_flat,
        )
        w, istop, itn = result[0], int(result[1]), int(result[2])
        return w.reshape(self.op.image_shape), istop, itn

    def _key(self, x: NDArray[np.float32], support: NDArray[np.bool_] | None) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(x).tobytes())
        h.update(b"none" if support is None else np.packbits(support).tobytes())
        h.update(self.op.fingerprint().encode())
        h.update(repr(self.config).encode())
        return h.hexdigest()

    def project(
        self, x: NDArray, support: NDArray[np.bool_] | None = None, x0: NDArray | None = None
    ) -> ProjectionResult:
        """``P_N(x)``; ``support`` — носитель детали (по умолчанию ``x != 0``), ``x0`` — warm start."""
        x32 = np.asarray(x, dtype=np.float32)
        if x32.shape != self.op.image_shape:
            raise ValueError(f"expected image of shape {self.op.image_shape}, got {x32.shape}")
        if support is None:
            support = x32 != 0
        support = np.asarray(support, dtype=bool)
        key = self._key(x32, support)
        if key in self._cache:
            self._cache.move_to_end(key)
            cached = self._cache[key]
            return ProjectionResult(
                cached.x_proj.copy(), cached.pinv_component.copy(), cached.residual_ratio,
                cached.iterations, cached.istop, from_cache=True,
            )

        ax = self.op.forward(x32, support=support)
        ax_norm = float(np.linalg.norm(ax))
        if ax_norm == 0.0:
            result = ProjectionResult(x32.copy(), np.zeros_like(x32), 0.0, 0, -1, from_cache=False)
        else:
            w, istop, itn = self.pinv(ax, x0=x0)
            x_proj = (x32.astype(np.float64) - w).astype(np.float32)
            ratio = float(np.linalg.norm(self.op.forward(x_proj))) / ax_norm
            logger.info("P_N: ||A x_proj||/||A x|| = %.3e after %d LSQR iterations (istop=%d)", ratio, itn, istop)
            if ratio > 1e-2:
                logger.warning("P_N residual ratio %.2e is large: increase lsqr iter_lim or check the operator", ratio)
            result = ProjectionResult(x_proj, w.astype(np.float32), ratio, itn, istop, from_cache=False)
        self._store(key, result)
        # копии, чтобы изменения у вызывающего не портили кэш
        return ProjectionResult(result.x_proj.copy(), result.pinv_component.copy(), result.residual_ratio,
                                result.iterations, result.istop, from_cache=False)

    def project_local(self, x: NDArray, support: NDArray[np.bool_]) -> ProjectionResult:
        """Проекция на ``ker(A) ∩ {supp ⊂ R}``: ``x − A_R⁺ A_R x`` по столбцам ``R`` (результат локален)."""
        x32 = np.asarray(x, dtype=np.float32)
        support = np.asarray(support, dtype=bool)
        key = "local:" + self._key(x32, support)
        if key in self._cache:
            self._cache.move_to_end(key)
            c = self._cache[key]
            return ProjectionResult(c.x_proj.copy(), c.pinv_component.copy(), c.residual_ratio, c.iterations,
                                    c.istop, from_cache=True)
        vals = x32[support].astype(np.float64)
        lin = self.op.as_linear_operator(support=support)
        b = lin.matvec(vals)
        b_norm = float(np.linalg.norm(b))
        if b_norm == 0.0:
            result = ProjectionResult(x32.copy(), np.zeros_like(x32), 0.0, 0, -1, from_cache=False)
        else:
            cfg = self.config
            out = lsqr(lin, b, atol=cfg.atol, btol=cfg.btol, iter_lim=cfg.iter_lim)
            w = np.zeros(x32.shape)
            w[support] = out[0]
            x_proj = (x32 - w).astype(np.float32)
            ratio = float(np.linalg.norm(self.op.forward(x_proj, support=support))) / b_norm
            logger.info("P_N,R: ||A x_proj||/||A x|| = %.3e after %d LSQR iterations", ratio, int(out[2]))
            result = ProjectionResult(x_proj, w.astype(np.float32), ratio, int(out[2]), int(out[1]), from_cache=False)
        self._store(key, result)
        return ProjectionResult(result.x_proj.copy(), result.pinv_component.copy(), result.residual_ratio,
                                result.iterations, result.istop, from_cache=False)

    def _store(self, key: str, result: ProjectionResult) -> None:
        if self.cache_size <= 0:
            return
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._cache.clear()


def local_kernel_element(
    op: RadonOperator,
    region: NDArray[np.bool_],
    rng: np.random.Generator,
    config: LsqrConfig | None = None,
    smooth: float = 0.0,
) -> tuple[NDArray[np.float32], float]:
    """Случайный элемент ``ker(A)`` с носителем в ``region``; возвращает ``(k, ||Ak||/||Ar||)``."""
    cfg = config or LsqrConfig()
    region = np.asarray(region, dtype=bool)
    r = rng.standard_normal(op.image_shape)
    if smooth > 0:
        r = gaussian_filter(r, smooth)
    r_vals = r[region]
    lin = op.as_linear_operator(support=region)
    b = lin.matvec(r_vals)
    w = lsqr(lin, b, atol=cfg.atol, btol=cfg.btol, iter_lim=cfg.iter_lim)[0]
    k = np.zeros(op.image_shape)
    k[region] = r_vals - w
    ratio = float(np.linalg.norm(op.forward(k))) / max(float(np.linalg.norm(b)), 1e-300)
    return k.astype(np.float32), ratio
