"""Алгоритм A, шаги 2–5: выбор детали, проекция на ker(A), блендинг, проверка остатка Δ.

Подробности: docs/03_algorithm_A.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from nullgraft.nullspace import NullspaceProjector, ProjectionResult
from nullgraft.operators import RadonOperator

logger = logging.getLogger(__name__)

WindowKind = Literal["cosine", "gaussian"]


def blend_window(region: NDArray[np.bool_], width: float, kind: WindowKind = "cosine") -> NDArray[np.float32]:
    """Гладкая маска ``α(r) ∈ [0, 1]``: ноль вне ``R``, рост от границы ``R`` до 1 на глубине ``width``."""
    region = np.asarray(region, dtype=bool)
    if not region.any():
        raise ValueError("region R is empty")
    if width <= 0:
        return region.astype(np.float32)
    depth = np.clip(distance_transform_edt(region) - 0.5, 0.0, None)
    if kind == "cosine":
        alpha = 0.5 * (1.0 - np.cos(np.pi * np.clip(depth / width, 0.0, 1.0)))
    elif kind == "gaussian":
        alpha = 1.0 - np.exp(-(depth**2) / (2.0 * width**2))
    else:
        raise ValueError(f"unknown window {kind!r}")
    return (alpha * region).astype(np.float32)


def place_patch(patch: NDArray, shape: tuple[int, int], center: tuple[int, int]) -> NDArray[np.float32]:
    """Вставить ``patch`` в нулевое изображение с центром в ``center`` (с обрезкой по краям)."""
    out = np.zeros(shape, dtype=np.float32)
    ph, pw = patch.shape
    top, left = center[0] - ph // 2, center[1] - pw // 2
    r0, c0 = max(top, 0), max(left, 0)
    r1, c1 = min(top + ph, shape[0]), min(left + pw, shape[1])
    if r1 <= r0 or c1 <= c0:
        raise ValueError("patch does not overlap the image")
    out[r0:r1, c0:c1] = patch[r0 - top : r1 - top, c0 - left : c1 - left]
    return out


def region_center(region: NDArray[np.bool_]) -> tuple[int, int]:
    rows, cols = np.nonzero(region)
    return int(round(rows.mean())), int(round(cols.mean()))


def extract_detail(z: NDArray, source: NDArray, region: NDArray[np.bool_]) -> NDArray[np.float32]:
    """``x_det = (x_source − z)·1_R``; ``source`` — целый срез или патч, центрируемый на ``R``."""
    z = np.asarray(z, dtype=np.float32)
    region = np.asarray(region, dtype=bool)
    source = np.asarray(source, dtype=np.float32)
    if source.shape != z.shape:
        placed = place_patch(source, z.shape, region_center(region))
        covered = place_patch(np.ones_like(source), z.shape, region_center(region)) > 0
        source = np.where(covered, placed, z)
    return ((source - z) * region).astype(np.float32)


@dataclass(frozen=True)
class GraftResult:
    """``x'' = z + α·P_N(x_det)`` и диагностика (``kernel_ratio``, ``Δ``, предел ``Δ``)."""

    x_det: NDArray[np.float32]
    x_proj: NDArray[np.float32]
    alpha: NDArray[np.float32]
    blended: NDArray[np.float32]
    x_graft: NDArray[np.float32]
    kernel_ratio: float
    delta: float
    delta_limit: float
    projection: ProjectionResult

    @property
    def valid(self) -> bool:
        return self.delta <= self.delta_limit


class Grafter:
    """Строит ``x''`` по ``z``, источнику детали и области ``R``."""

    def __init__(
        self,
        op: RadonOperator,
        projector: NullspaceProjector | None = None,
        window: WindowKind = "cosine",
        window_width: float = 3.0,
        delta_factor: float = 0.5,
    ) -> None:
        self.op = op
        self.projector = projector or NullspaceProjector(op)
        self.window = window
        self.window_width = window_width
        self.delta_factor = delta_factor

    def graft(self, z: NDArray, source: NDArray, region: NDArray[np.bool_], eps: float) -> GraftResult:
        region = np.asarray(region, dtype=bool)
        z32 = np.asarray(z, dtype=np.float32)
        x_det = extract_detail(z32, source, region)
        proj = self.projector.project(x_det, support=region)
        logger.info("graft: ||A x_proj||/||A x_det|| = %.3e", proj.residual_ratio)
        alpha = blend_window(region, self.window_width, self.window)
        blended = (alpha * proj.x_proj).astype(np.float32)
        support = np.abs(blended) > 0
        delta = float(np.linalg.norm(self.op.forward(blended, support=support))) if support.any() else 0.0
        limit = self.delta_factor * float(eps)
        if delta > limit:
            logger.warning("blend residual Delta=%.3e exceeds %.2f*eps=%.3e: test invalid", delta,
                           self.delta_factor, limit)
        return GraftResult(
            x_det=x_det, x_proj=proj.x_proj, alpha=alpha, blended=blended, x_graft=(z32 + blended).astype(np.float32),
            kernel_ratio=proj.residual_ratio, delta=delta, delta_limit=limit, projection=proj,
        )
