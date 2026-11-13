"""Мягкие тесты на регресс производительности; пороги — ``benchmark.max_seconds`` конфига или ``NULLGRAFT_PERF_CONFIG``."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from nullgraft.config import load_config
from nullgraft.operators import Geometry, RadonOperator

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def limits() -> dict[str, float]:
    cfg = load_config(os.environ.get("NULLGRAFT_PERF_CONFIG"))
    return {k: float(v) for k, v in cfg["benchmark"]["max_seconds"].items()}


def test_projection_speed(limits: dict[str, float]) -> None:
    op = RadonOperator(Geometry((128, 128), 60), backend="pixel")
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(10):
        op.adjoint(op.forward(x))
    assert (time.perf_counter() - t0) / 10 < limits["forward"]


def test_detail_check_speed(limits: dict[str, float]) -> None:
    from nullgraft.assess import AlwaysRealistic, AssessmentConfig, DetailAssessor
    from nullgraft.decoders import FBP
    from nullgraft.experiments.common import disk_mask

    op = RadonOperator(Geometry((64, 64), 30), backend="pixel")
    rng = np.random.default_rng(0)
    x = rng.random((64, 64)).astype(np.float32)
    y = op.forward(x)
    dec = FBP(op)
    z = dec.reconstruct(y)[0]
    eps = 2 * float(np.linalg.norm(op.forward(z) - y)) + 1.0
    assessor = DetailAssessor(op, dec, AssessmentConfig(eps=eps, n_noise=20, noise_sigma=1e-3, visible_fraction=-1),
                              realism=AlwaysRealistic())
    t0 = time.perf_counter()
    assessor.assess(y, rng.random((64, 64)).astype(np.float32), disk_mask((64, 64), (32, 32), 8), z=z)
    assert time.perf_counter() - t0 < limits["detail_check"]
