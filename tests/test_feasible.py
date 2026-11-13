from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import lsqr

from nullgraft.feasible import (
    FeasibleSetEstimator,
    PCAProjection,
    SinogramPooling,
    build_patch_archive,
    convergence_curve,
    diameters,
    embed_patches,
    patch_measurement,
)
from nullgraft.metrics import LocalizedMetric
from nullgraft.operators import RadonOperator, make_operator

SHAPE = (32, 32)
PATCH = 16


@pytest.fixture(scope="module")
def op() -> RadonOperator:
    return make_operator(SHAPE, n_angles=6, backend="pixel")


def _patch_kernel_element(op: RadonOperator, center: tuple[int, int], seed: int) -> np.ndarray:
    """Элемент ker(A E) для патчей, вставленных в ``center``."""
    _, support = embed_patches(np.zeros((1, PATCH, PATCH)), np.asarray(center), SHAPE, "original")
    lin = op.as_linear_operator(support=support)
    r = gaussian_filter(np.random.default_rng(seed).standard_normal(SHAPE), 1.0)[support]
    k = r - lsqr(lin, lin.matvec(r), atol=1e-10, btol=1e-10, iter_lim=2000)[0]
    return k.reshape(PATCH, PATCH).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic(op: RadonOperator):
    """Срезы с известной парой: срез 1 = срез 0 + элемент ядра в одном патче."""
    rng = np.random.default_rng(0)
    slices = [gaussian_filter(rng.standard_normal(SHAPE), 2.0).astype(np.float32) for _ in range(5)]
    center = (16, 16)
    k = _patch_kernel_element(op, center, seed=1)
    k *= 0.5 / np.abs(k).max()
    twin = slices[0].copy()
    twin[center[0] - 8 : center[0] + 8, center[1] - 8 : center[1] + 8] += k
    slices.insert(1, twin)
    archive = build_patch_archive(slices, op, patch_size=PATCH, stride=8, pooling=SinogramPooling(2, 2))
    return slices, archive, k, center


def _brute_force_fa(op: RadonOperator, archive, ys: np.ndarray, eps: float) -> np.ndarray:
    meas = patch_measurement(op, archive, np.arange(len(archive)))
    d = np.linalg.norm((meas[:, None] - ys[None]).reshape(len(archive), len(ys), -1), axis=2)
    return d < eps


def _brute_force_diameter(points: np.ndarray, metric: LocalizedMetric) -> float:
    return max((float(metric(a, b)) for a in points for b in points), default=0.0)


def _queries(archive, op: RadonOperator, slice_idx: int, center: tuple[int, int]) -> np.ndarray:
    n = int(np.flatnonzero((archive.meta[:, 0] == slice_idx) & (archive.meta[:, 1] == center[0])
                           & (archive.meta[:, 2] == center[1]))[0])
    others = np.arange(0, len(archive), 7)
    return patch_measurement(op, archive, np.concatenate([[n], others]))


@pytest.mark.parametrize("eps", [1e-3, 0.5, 2.0])
def test_algorithm1_filter_never_misses_feasible_pairs(op: RadonOperator, synthetic, eps: float) -> None:
    _, archive, _, center = synthetic
    ys = _queries(archive, op, 0, center)
    for n_components in (None, 8):
        info = FeasibleSetEstimator(op, archive, n_components=n_components).feasible_matrix(ys, eps)
        np.testing.assert_array_equal(info.fa.toarray(), _brute_force_fa(op, archive, ys, eps))


def test_known_kernel_pair_gives_correct_diameter(op: RadonOperator, synthetic) -> None:
    _, archive, k, center = synthetic
    ys = _queries(archive, op, 0, center)
    eps = 1e-3 * float(np.linalg.norm(ys[0]))
    info = FeasibleSetEstimator(op, archive).feasible_matrix(ys[:1], eps)
    metric = LocalizedMetric()
    res = diameters(archive, info, metric)[0]
    members = info.fa[:, 0].nonzero()[0]
    assert res.size == len(members) >= 2
    assert set(archive.meta[members, 0]) >= {0, 1}
    expected = float(metric.norm(k))
    assert res.diameter == pytest.approx(expected, rel=1e-4)
    assert res.diameter == pytest.approx(_brute_force_diameter(archive.patches[members], metric), rel=1e-6)
    assert res.decoder_error_lower_bound == pytest.approx(res.diameter / 2)
    assert res.pair is not None and set(res.pair) <= set(members)


def test_estimates_never_exceed_true_sample_diameter(op: RadonOperator, synthetic) -> None:
    _, archive, _, center = synthetic
    ys = _queries(archive, op, 0, center)
    info = FeasibleSetEstimator(op, archive).feasible_matrix(ys, eps=3.0)
    metric = LocalizedMetric(p=2, q=2)
    exact = diameters(archive, info, metric)
    approx = diameters(archive, info, metric, exact_limit=1, n_restarts=2)
    for e, a in zip(exact, approx, strict=True):
        members = info.fa[:, e.k].nonzero()[0]
        true = _brute_force_diameter(np.asarray(archive.patches[members]), metric)
        assert e.diameter == pytest.approx(true, rel=1e-6)
        assert a.diameter <= e.diameter + 1e-9
        assert a.approximate == (e.size > 1)


def test_convergence_curve_is_monotone_from_below(op: RadonOperator, synthetic) -> None:
    _, archive, _, center = synthetic
    ys = _queries(archive, op, 0, center)
    info = FeasibleSetEstimator(op, archive).feasible_matrix(ys, eps=3.0)
    fr, curve = convergence_curve(archive, info)
    assert fr[0] == pytest.approx(0.1) and fr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(curve, axis=0) >= -1e-12)
    np.testing.assert_allclose(curve[-1], [r.diameter for r in diameters(archive, info)])


def test_com_is_sparse_symmetric(op: RadonOperator, synthetic) -> None:
    _, archive, _, center = synthetic
    ys = _queries(archive, op, 0, center)
    info = FeasibleSetEstimator(op, archive).feasible_matrix(ys, eps=3.0)
    com = info.com
    assert com.shape == (len(archive), len(archive))
    assert (com != com.T).nnz == 0
    np.testing.assert_array_equal(com.diagonal(), np.asarray(info.fa.sum(axis=1)).ravel())


def test_feature_maps_are_contractions(rng: np.random.Generator) -> None:
    pool = SinogramPooling(3, 2)
    v = rng.standard_normal((7, 20, 31))
    assert np.all(np.linalg.norm(pool(v), axis=1) <= np.linalg.norm(v.reshape(7, -1), axis=1) + 1e-6)
    pca = PCAProjection.fit(rng.standard_normal((50, 12)), 4)
    w = rng.standard_normal((9, 12)).astype(np.float32)
    assert np.all(np.linalg.norm(pca(w), axis=1) <= np.linalg.norm(w, axis=1) + 1e-5)


def test_center_placement_is_translation_agnostic(op: RadonOperator) -> None:
    rng = np.random.default_rng(3)
    img = gaussian_filter(rng.standard_normal(SHAPE), 2.0).astype(np.float32)
    shifted = np.roll(img, 8, axis=1)
    archive = build_patch_archive([img, shifted], op, patch_size=PATCH, stride=8, placement="center")
    a = int(np.flatnonzero((archive.meta[:, 0] == 0) & (archive.meta[:, 2] == 8))[0])
    b = int(np.flatnonzero((archive.meta[:, 0] == 1) & (archive.meta[:, 1] == archive.meta[a, 1])
                           & (archive.meta[:, 2] == 16))[0])
    m = patch_measurement(op, archive, np.array([a, b]))
    np.testing.assert_allclose(m[0], m[1], atol=1e-5)


class _Crashing(list):
    def __init__(self, items, crash_at: int) -> None:
        super().__init__(items)
        self.crash_at = crash_at

    def __getitem__(self, i):  # type: ignore[override]
        if i == self.crash_at:
            raise RuntimeError("simulated interruption")
        return super().__getitem__(i)


def test_archive_build_checkpoints_and_resumes(op: RadonOperator, synthetic, tmp_path) -> None:
    slices = synthetic[0]
    ref = build_patch_archive(slices, op, patch_size=PATCH, stride=8)
    with pytest.raises(RuntimeError):
        build_patch_archive(_Crashing(slices, 3), op, patch_size=PATCH, stride=8, out_dir=tmp_path,
                            checkpoint_every=1)
    resumed = build_patch_archive(slices, op, patch_size=PATCH, stride=8, out_dir=tmp_path)
    np.testing.assert_array_equal(np.asarray(resumed.patches), ref.patches)
    np.testing.assert_allclose(np.asarray(resumed.pooled), ref.pooled, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(resumed.meta), ref.meta)
