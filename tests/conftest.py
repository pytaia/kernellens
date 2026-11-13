from __future__ import annotations

import numpy as np
import pytest

from nullgraft.operators import RadonOperator, make_operator


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def sparse_view_op() -> RadonOperator:
    """48×48, 24 проекции: 24·70 лучей < 2304 пикселей, поэтому ker(A) нетривиально."""
    return make_operator((48, 48), n_angles=24, beam="parallel", backend="pixel")


@pytest.fixture(scope="session")
def fan_op() -> RadonOperator:
    return make_operator((40, 40), n_angles=20, beam="fan", backend="pixel")


def disk(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2).astype(np.float32)
