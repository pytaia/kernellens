from __future__ import annotations

import numpy as np
import pytest

from nullgraft.graft import Grafter, blend_window, extract_detail, place_patch
from nullgraft.nullspace import local_kernel_element
from nullgraft.operators import make_operator
from tests.conftest import disk


@pytest.mark.parametrize("kind", ["cosine", "gaussian"])
def test_blend_window_is_smooth_and_vanishes_at_boundary(kind: str) -> None:
    region = disk((40, 40), (20, 20), 10).astype(bool)
    alpha = blend_window(region, width=4, kind=kind)
    assert np.all(alpha[~region] == 0)
    assert 0.9 < alpha.max() <= 1.0
    boundary = region & ~disk((40, 40), (20, 20), 9).astype(bool)
    assert alpha[boundary].max() < 0.2
    assert alpha[20, 20] > alpha[20, 12] > alpha[20, 11]


def test_place_patch_and_extract_detail() -> None:
    z = np.ones((20, 20), np.float32)
    patch = np.full((5, 5), 3.0, np.float32)
    placed = place_patch(patch, (20, 20), (10, 10))
    assert placed.sum() == pytest.approx(75.0)
    cropped = place_patch(patch, (20, 20), (0, 0))
    assert cropped.sum() == pytest.approx(27.0)
    region = disk((20, 20), (10, 10), 2).astype(bool)
    det = extract_detail(z, patch, region)
    assert np.all(det[~region] == 0)
    assert np.allclose(det[region], 2.0)
    wide = disk((20, 20), (10, 10), 4).astype(bool)
    det_wide = extract_detail(z, patch, wide)
    assert np.allclose(det_wide[wide & (placed == 0)], 0.0)
    det_full = extract_detail(z, np.full((20, 20), 4.0, np.float32), region)
    assert np.allclose(det_full[region], 3.0)


def test_pure_kernel_detail_leaves_measurement_unchanged(rng: np.random.Generator) -> None:
    op = make_operator((48, 48), n_angles=8, backend="pixel")
    inner = disk((48, 48), (24, 24), 7).astype(bool)
    region = disk((48, 48), (24, 24), 12).astype(bool)
    k, ratio = local_kernel_element(op, inner, rng, smooth=1.0)
    assert ratio < 1e-4 and np.abs(k).max() > 0
    z = np.zeros((48, 48), np.float32)
    g = Grafter(op, window_width=3).graft(z, z + k, region, eps=1.0)
    np.testing.assert_allclose(g.blended, g.x_proj * g.alpha, atol=1e-7)
    assert np.linalg.norm(g.blended - k) / np.linalg.norm(k) < 1e-2
    assert np.linalg.norm(op.forward(g.x_graft) - op.forward(z)) / np.linalg.norm(op.forward(k + 1.0)) < 1e-4
    # kernel_ratio здесь — отношение двух почти нулей и смысла не имеет
    assert g.valid and g.delta < 1e-3 * g.delta_limit / 0.5


def test_local_projection_stays_in_region_and_kernel(rng: np.random.Generator) -> None:
    op = make_operator((48, 48), n_angles=8, backend="pixel")
    region = disk((48, 48), (24, 24), 12).astype(bool)
    z = np.zeros((48, 48), np.float32)
    detail = (rng.standard_normal((48, 48)) * region).astype(np.float32)
    g = Grafter(op, window_width=0, projection="local").graft(z, z + detail, region, eps=1.0)
    assert not g.x_proj[~region].any()  # локальна
    assert g.delta < 1e-3 * np.linalg.norm(op.forward(detail))  # в ядре
    assert np.abs(g.x_proj).sum() > 0.1 * np.abs(detail).sum()  # при 8 проекциях часть детали невидима
    glob = Grafter(op, window_width=0).graft(z, z + detail, region, eps=1.0)
    assert np.abs(glob.x_proj[~region]).max() > 1e-3  # глобальная проекция размазана за пределы R


def test_large_detail_in_sparse_view_invalidates_blend() -> None:
    op = make_operator((48, 48), n_angles=8, backend="pixel")
    region = disk((48, 48), (24, 24), 12).astype(bool)
    z = np.zeros((48, 48), np.float32)
    g = Grafter(op).graft(z, disk((48, 48), (24, 24), 5), region, eps=0.1)
    assert not g.valid and g.delta > g.delta_limit
