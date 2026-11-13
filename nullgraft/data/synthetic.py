"""Синтетические данные для проверки пайплайна без скачанного датасета."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def shepp_logan(size: int = 256) -> NDArray[np.float32]:
    """Фантом Шеппа–Логана из ``skimage.data`` размером ``size × size``."""
    return resize(shepp_logan_phantom(), (size, size), anti_aliasing=True).astype(np.float32)


def _ellipse(shape: tuple[int, int], cy: float, cx: float, ay: float, ax: float, angle: float) -> NDArray[np.bool_]:
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    yy -= cy
    xx -= cx
    c, s = np.cos(angle), np.sin(angle)
    u = (xx * c + yy * s) / ax
    v = (-xx * s + yy * c) / ay
    return u**2 + v**2 <= 1.0


def _body(size: int) -> NDArray[np.float32]:
    return 0.2 * _ellipse((size, size), size / 2, size / 2, 0.42 * size, 0.38 * size, 0.0).astype(np.float32)


def random_ellipse_phantom(size: int, rng: np.random.Generator, n_ellipses: int = 8) -> NDArray[np.float32]:
    """Фантом «тело»: внешний эллипс со случайными внутренними эллипсами."""
    shape = (size, size)
    img = _body(size)
    for _ in range(n_ellipses):
        r = rng.uniform(0.02, 0.15) * size
        cy, cx = size / 2 + rng.uniform(-0.25, 0.25, 2) * size
        img += rng.uniform(-0.1, 0.3) * _ellipse(shape, cy, cx, r, r * rng.uniform(0.5, 1.5), rng.uniform(0, np.pi))
    return np.clip(img, 0.0, None).astype(np.float32)


def synthetic_volume(n_slices: int, size: int = 128, seed: int = 0, drift: float = 0.5) -> NDArray[np.float32]:
    """Стопка срезов с медленно дрейфующими эллипсами: соседние срезы почти совпадают, как в реальной КТ."""
    rng = np.random.default_rng(seed)
    n_e = 10
    radius = rng.uniform(0.02, 0.15, n_e) * size
    centre = rng.uniform(-0.25, 0.25, (n_e, 2)) * size + size / 2
    aspect = rng.uniform(0.5, 1.5, n_e)
    angle = rng.uniform(0, np.pi, n_e)
    value = rng.uniform(-0.1, 0.3, n_e)
    vel = rng.normal(0, drift, (n_e, 2))
    shape = (size, size)
    body = _body(size)
    vol = np.zeros((n_slices, size, size), np.float32)
    for k in range(n_slices):
        img = body.copy()
        for e in range(n_e):
            cy, cx = centre[e] + k * vel[e]
            img += value[e] * _ellipse(shape, cy, cx, radius[e], radius[e] * aspect[e], angle[e])
        vol[k] = np.clip(img, 0.0, None)
    return vol


def iter_synthetic(n_slices: int, size: int = 128, seed: int = 0) -> Iterator[NDArray[np.float32]]:
    """Ленивый итератор независимых случайных фантомов."""
    rng = np.random.default_rng(seed)
    for _ in range(n_slices):
        yield random_ellipse_phantom(size, rng)
