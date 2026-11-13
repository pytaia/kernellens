from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.transform import radon

from nullgraft.operators import Geometry, PixelDrivenBackend, RadonOperator, astra_available, make_operator
from tests.conftest import disk


def _dot_test(op: RadonOperator, rng: np.random.Generator) -> float:
    x = rng.standard_normal(op.image_shape)
    s = rng.standard_normal(op.sino_shape)
    lhs = float(np.vdot(op.forward(x).astype(np.float64), s))
    rhs = float(np.vdot(x, op.adjoint(s).astype(np.float64)))
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs))


@pytest.mark.parametrize("beam", ["parallel", "fan"])
def test_pixel_backend_adjoint_is_exact_transpose(beam: str, rng: np.random.Generator) -> None:
    op = make_operator((32, 36), n_angles=17, beam=beam, backend="pixel")
    assert _dot_test(op, rng) < 1e-5


@pytest.mark.parametrize("beam", ["parallel", "fan"])
def test_matrix_cache_matches_matrix_free(beam: str, rng: np.random.Generator) -> None:
    geom = Geometry((30, 30), n_angles=12, beam=beam)
    cached = PixelDrivenBackend(geom, matrix_budget_mb=64)
    free = PixelDrivenBackend(geom, matrix_budget_mb=0)
    assert cached.has_matrix and not free.has_matrix
    x = rng.standard_normal((3, 900))
    s = rng.standard_normal((3, 12 * cached.n_det))
    np.testing.assert_allclose(cached.forward(x, None), free.forward(x, None), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cached.adjoint(s), free.adjoint(s), rtol=1e-5, atol=1e-5)


def test_batched_equals_loop(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    xs = rng.standard_normal((4, *sparse_view_op.image_shape)).astype(np.float32)
    batched = sparse_view_op.forward(xs)
    looped = np.stack([sparse_view_op.forward(x) for x in xs])
    np.testing.assert_allclose(batched, looped, rtol=1e-6, atol=1e-6)
    back = sparse_view_op.adjoint(batched)
    assert back.shape == xs.shape and back.dtype == np.float32


def test_support_restricted_forward_is_exact(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    support = disk(sparse_view_op.image_shape, (20, 25), 5).astype(bool)
    x = rng.standard_normal(sparse_view_op.image_shape).astype(np.float32) * support
    np.testing.assert_allclose(sparse_view_op.forward(x, support=support), sparse_view_op.forward(x), atol=1e-5)
    mask = sparse_view_op.ray_mask(support)
    assert 0 < mask.mean() < 0.5
    assert np.all(sparse_view_op.forward(x)[~mask] == 0)


def test_parallel_projection_matches_line_integrals() -> None:
    """Проекция центрального диска радиуса r под любым углом равна 2√(r² − t²)."""
    n, r = 96, 30.0
    op = make_operator((n, n), n_angles=8, backend="pixel")
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    phantom = (yy**2 + xx**2 <= r**2).astype(np.float32)
    sino = op.forward(phantom)
    t = (np.arange(op.n_det) - (op.n_det - 1) / 2) * op.geometry.spacing
    analytic = 2 * np.sqrt(np.clip(r**2 - t**2, 0, None))
    central = np.abs(t) < r - 3
    rel = np.abs(sino[:, central] - analytic[central]) / analytic[central]
    assert np.median(rel) < 0.05
    np.testing.assert_allclose(sino.sum(axis=1) * op.geometry.spacing, phantom.sum(), rtol=1e-4)


def test_fan_beam_preserves_mass_on_isocentre_scale(fan_op: RadonOperator) -> None:
    x = disk(fan_op.image_shape, (20, 20), 6)
    sino = fan_op.forward(x)
    g = fan_op.geometry
    iso_spacing = g.spacing * g.d_so / g.d_sd
    np.testing.assert_allclose(sino.sum(axis=1) * iso_spacing, x.sum(), rtol=0.05)


def test_skimage_backend_agrees_with_pixel_backend() -> None:
    n = 64
    x = disk((n, n), (30, 34), 12)
    sk = make_operator((n, n), n_angles=10, backend="skimage")
    sino_sk = sk.forward(x)
    ref = radon(x, theta=np.rad2deg(sk.geometry.angles), circle=False).T
    np.testing.assert_allclose(sino_sk, ref, rtol=1e-5, atol=1e-5)
    px = make_operator((n, n), n_angles=10, backend="pixel")
    # сравниваем массу по проекции: ориентация детектора у бэкендов может различаться
    np.testing.assert_allclose(sino_sk.sum(axis=1), px.forward(x).sum(axis=1), rtol=0.02)


def test_linear_operator_view(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    lin = sparse_view_op.as_linear_operator()
    x = rng.standard_normal(sparse_view_op.image_shape)
    np.testing.assert_allclose(lin @ x.ravel(), sparse_view_op.forward(x).ravel(), rtol=1e-4, atol=1e-4)
    assert lin.shape == sparse_view_op.shape
    xs = rng.standard_normal((sparse_view_op.shape[1], 3))
    np.testing.assert_allclose(lin.matmat(xs)[:, 1], lin @ xs[:, 1], rtol=1e-10)


def test_norm_estimate_bounds(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    nrm = sparse_view_op.norm()
    for _ in range(3):
        x = rng.standard_normal(sparse_view_op.image_shape)
        assert np.linalg.norm(sparse_view_op.forward(x)) <= nrm * np.linalg.norm(x) * 1.01


def test_geometry_validation_and_fingerprint() -> None:
    with pytest.raises(ValueError):
        Geometry((32, 32), n_angles=0)
    with pytest.raises(ValueError):
        Geometry((32, 32), n_angles=10, beam="fan", source_origin=10)
    g1 = Geometry((32, 32), 10)
    assert g1.fingerprint() == Geometry((32, 32), 10).fingerprint()
    assert g1.fingerprint() != Geometry((32, 32), 11).fingerprint()
    assert math.isclose(g1.angles[-1], math.pi * 9 / 10)


def test_shape_errors(sparse_view_op: RadonOperator) -> None:
    with pytest.raises(ValueError):
        sparse_view_op.forward(np.zeros((10, 10)))
    with pytest.raises(ValueError):
        sparse_view_op.adjoint(np.zeros(sparse_view_op.image_shape))


@pytest.mark.skipif(not astra_available(), reason="ASTRA not installed")
def test_astra_backend_adjoint(rng: np.random.Generator) -> None:
    op = make_operator((32, 32), n_angles=16, backend="astra")
    assert _dot_test(op, rng) < (1e-4 if op.exact_adjoint else 5e-2)
