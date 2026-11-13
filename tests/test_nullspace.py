from __future__ import annotations

import numpy as np

from nullgraft.nullspace import LsqrConfig, NullspaceProjector, kernel_residual
from nullgraft.operators import RadonOperator, make_operator
from tests.conftest import disk


def test_projection_of_random_image_is_invisible(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    """‖A P_N(x)‖ на порядки меньше ‖A x‖."""
    proj = NullspaceProjector(sparse_view_op)
    x = rng.standard_normal(sparse_view_op.image_shape).astype(np.float32)
    res = proj.project(x)
    assert res.residual_ratio < 1e-3
    direct = np.linalg.norm(sparse_view_op.forward(res.x_proj)) / np.linalg.norm(sparse_view_op.forward(x))
    assert np.isclose(direct, res.residual_ratio, rtol=1e-3)
    assert np.linalg.norm(res.x_proj) > 0.1 * np.linalg.norm(x)


def test_lsqr_matches_exact_svd_projector(rng: np.random.Generator) -> None:
    """Сверка с медленным точным проектором I − VVᵀ (плотное SVD) по направлениям с σ > 1e-2·σ_max."""
    op = make_operator((24, 24), n_angles=12, backend="pixel")
    dense = op.backend._matrix.toarray().astype(np.float64)
    _, sig, vt = np.linalg.svd(dense, full_matrices=False)
    rank = int((sig > 1e-9 * sig[0]).sum())
    v_range = vt[:rank]
    x = rng.standard_normal(op.image_shape).astype(np.float32)
    exact = x.ravel() - v_range.T @ (v_range @ x.ravel())
    approx = NullspaceProjector(op).project(x)
    diff = approx.x_proj.ravel().astype(np.float64) - exact
    v_visible = vt[sig > 1e-2 * sig[0]]
    assert np.linalg.norm(v_visible @ diff) / np.linalg.norm(x) < 1e-3
    assert approx.residual_ratio < 1e-3


def test_projection_reduces_visibility_and_is_orthogonal(
    sparse_view_op: RadonOperator, rng: np.random.Generator
) -> None:
    proj = NullspaceProjector(sparse_view_op)
    x = rng.standard_normal(sparse_view_op.image_shape).astype(np.float32)
    p1 = proj.project(x)
    p2 = proj.project(p1.x_proj)
    assert np.linalg.norm(sparse_view_op.forward(p2.x_proj)) <= np.linalg.norm(sparse_view_op.forward(p1.x_proj))
    # допуск — из-за неразрешённых LSQR компонент с малыми σ (см. test_lsqr_matches_exact_svd_projector)
    cos = np.vdot(p1.x_proj, p1.pinv_component) / (np.linalg.norm(p1.x_proj) * np.linalg.norm(p1.pinv_component))
    assert abs(cos) < 5e-2


def test_localized_detail_with_support(sparse_view_op: RadonOperator) -> None:
    support = disk(sparse_view_op.image_shape, (18, 28), 6).astype(bool)
    detail = 2.0 * disk(sparse_view_op.image_shape, (18, 28), 3)
    res = NullspaceProjector(sparse_view_op).project(detail, support=support)
    assert res.residual_ratio < 1e-3
    assert kernel_residual(sparse_view_op, res.x_proj, reference=detail) < 1e-3


def test_cache_hit_and_determinism(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    proj = NullspaceProjector(sparse_view_op, cache_size=4)
    x = rng.standard_normal(sparse_view_op.image_shape).astype(np.float32)
    first = proj.project(x)
    second = proj.project(x)
    assert not first.from_cache and second.from_cache
    np.testing.assert_array_equal(first.x_proj, second.x_proj)
    fresh = NullspaceProjector(sparse_view_op).project(x)
    np.testing.assert_array_equal(first.x_proj, fresh.x_proj)


def test_cache_eviction(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    proj = NullspaceProjector(sparse_view_op, config=LsqrConfig(iter_lim=20), cache_size=2)
    xs = [rng.standard_normal(sparse_view_op.image_shape).astype(np.float32) for _ in range(3)]
    for x in xs:
        proj.project(x)
    assert not proj.project(xs[0]).from_cache
    assert proj.project(xs[2]).from_cache


def test_zero_detail_is_trivial(sparse_view_op: RadonOperator) -> None:
    res = NullspaceProjector(sparse_view_op).project(np.zeros(sparse_view_op.image_shape))
    assert res.residual_ratio == 0.0 and res.iterations == 0
    assert not res.x_proj.any()


def test_warm_start_keeps_kernel_property(sparse_view_op: RadonOperator, rng: np.random.Generator) -> None:
    proj = NullspaceProjector(sparse_view_op, cache_size=0)
    x = rng.standard_normal(sparse_view_op.image_shape).astype(np.float32)
    cold = proj.project(x)
    x2 = (x + 0.01 * rng.standard_normal(x.shape)).astype(np.float32)
    warm = proj.project(x2, x0=cold.pinv_component)
    assert warm.residual_ratio < 1e-3
    assert warm.iterations <= cold.iterations


def test_fan_beam_projection(fan_op: RadonOperator, rng: np.random.Generator) -> None:
    x = rng.standard_normal(fan_op.image_shape).astype(np.float32)
    assert NullspaceProjector(fan_op).project(x).residual_ratio < 1e-3
