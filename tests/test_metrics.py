from __future__ import annotations

import numpy as np
import pytest

from nullgraft.metrics import (
    LocalizedMetric,
    dice,
    hausdorff_to_point,
    hu_error,
    localized_norm,
    rmse,
    ssim,
    to_hu,
)


def test_localized_l1_is_mean_abs_over_region(rng: np.random.Generator) -> None:
    x = rng.standard_normal((16, 16))
    region = np.zeros((16, 16), bool)
    region[4:8, 2:10] = True
    assert np.isclose(localized_norm(x, region), np.abs(x[region]).mean())
    y = x.copy()
    y[~region] = 1e6
    assert np.isclose(localized_norm(y, region), localized_norm(x, region))


def test_localized_l2_is_rms_and_global_default(rng: np.random.Generator) -> None:
    x = rng.standard_normal((10, 12))
    assert np.isclose(localized_norm(x, None, p=2, q=2), np.sqrt(np.mean(x**2)))
    assert np.isclose(localized_norm(x, None, p=2, q=2, normalize=False), np.linalg.norm(x))


def test_collection_of_regions_combines_with_lp(rng: np.random.Generator) -> None:
    x = rng.standard_normal((12, 12))
    r1 = np.zeros((12, 12), bool)
    r1[:4, :4] = True
    r2 = np.zeros((12, 12), bool)
    r2[6:, 6:] = True
    n1 = localized_norm(x, r1, q=2)
    n2 = localized_norm(x, r2, q=2)
    assert np.isclose(localized_norm(x, [r1, r2], p=2, q=2), np.hypot(n1, n2))
    assert np.isclose(localized_norm(x, [r1, r2], p=1, q=2), n1 + n2)


def test_metric_is_a_norm_on_region(rng: np.random.Generator) -> None:
    region = np.zeros((12, 12), bool)
    region[3:9, 3:9] = True
    metric = LocalizedMetric(region, p=1, q=2)
    a, b, c = (rng.standard_normal((12, 12)) for _ in range(3))
    assert metric(a, c) <= metric(a, b) + metric(b, c) + 1e-12
    assert np.isclose(metric.norm(3 * a), 3 * metric.norm(a))
    batch = np.stack([a, b])
    np.testing.assert_allclose(metric(batch, c[None]), [metric(a, c), metric(b, c)])


def test_hausdorff_to_point_is_max_over_samples() -> None:
    x = np.zeros((4, 4))
    samples = np.stack([np.full((4, 4), v) for v in (0.1, -0.5, 0.3)])
    assert np.isclose(hausdorff_to_point(samples, x, LocalizedMetric()), 0.5)


def test_misc_metrics() -> None:
    a = np.zeros((8, 8))
    b = np.ones((8, 8))
    assert np.isclose(rmse(a, b), 1.0)
    assert ssim(b, b) == pytest.approx(1.0)
    assert np.isclose(to_hu(np.array(0.02), mu_water=0.02), 0.0)
    assert np.isclose(to_hu(np.array(0.0), mu_water=0.02), -1000.0)
    assert np.isclose(hu_error(np.full((4, 4), 0.022), np.full((4, 4), 0.02), mu_water=0.02), 100.0)
    m1 = np.zeros((5, 5), bool)
    m1[:2] = True
    assert dice(m1, m1) == 1.0 and dice(m1, ~m1) == 0.0 and dice(np.zeros(3, bool), np.zeros(3, bool)) == 1.0


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        localized_norm(np.ones((4, 4)), np.zeros((4, 4), bool))
    with pytest.raises(ValueError):
        localized_norm(np.ones((4, 4)), p=0.5)
    with pytest.raises(ValueError):
        localized_norm(np.ones((4, 4)), np.ones((3, 3), bool))
