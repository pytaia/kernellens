"""Тонкие обёртки над nullgraft для внешних приложений (например, nullgraft_demo)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nullgraft.decoders import CGLS, FBP, SART, TVADMM, Decoder
from nullgraft.operators import RadonOperator
from nullgraft.similar import _signature

DECODER_NAMES = ("cgls", "fbp-ramp", "fbp-hann", "fbp-shepp-logan", "sart", "tv-admm")


def decoder_signature(decoder: Decoder) -> object:
    """Описание декодера (класс и скалярные настройки) для ключей кэша."""
    return _signature(decoder)


def decoder_from_name(name: str, op: RadonOperator, n_iter: int | None = None) -> Decoder:
    """Декодер по короткому имени из ``DECODER_NAMES``."""
    if name.startswith("fbp-"):
        return FBP(op, window=name.removeprefix("fbp-"))  # type: ignore[arg-type]
    if name == "cgls":
        return CGLS(op, n_iter=n_iter or 250)
    if name == "sart":
        return SART(op, n_iter=n_iter or 200, nonneg=False)
    if name == "tv-admm":
        return TVADMM(op, n_iter=n_iter or 50)
    raise ValueError(f"unknown decoder {name!r}; expected one of {DECODER_NAMES}")


def detector_residual(op: RadonOperator, z: NDArray, y: NDArray) -> NDArray[np.float32]:
    """Карта невязки ``|A z − y|`` в пространстве детектора."""
    return np.abs(op.forward(z) - np.asarray(y, np.float32)).astype(np.float32)


def default_eps(op: RadonOperator, sigma: float, factor: float = 1.5) -> float:
    """``ε = factor · σ · √n_rays``."""
    return float(factor * sigma * math.sqrt(op.shape[0]))


def local_resolution(op: RadonOperator, center: tuple[float, float]) -> float:
    """Грубая оценка разрешения оператора (в пикселях) в точке ``center = (row, col)``.

    Максимум из шага детектора в изоцентре и шага угловой выборки на расстоянии точки от
    центра вращения (``ρ·Δθ``); см. README демо, «Допущения».
    """
    g = op.geometry
    h, w = g.shape
    rho = max(1.0, math.hypot(center[0] - (h - 1) / 2, center[1] - (w - 1) / 2))
    iso_spacing = g.spacing if g.beam == "parallel" else g.spacing * g.d_so / g.d_sd
    return float(max(iso_spacing, rho * g.angle_step))
