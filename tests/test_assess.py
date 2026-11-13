from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from nullgraft.assess import (
    AlwaysRealistic,
    ArchivePatchRealism,
    AssessmentConfig,
    DetailAssessor,
    Outcome,
    RealismVerdict,
    calibrate_null_detail,
    sample_noise,
    select_point,
)
from nullgraft.decoders import SART, PerturbedDecoder
from nullgraft.nullspace import local_kernel_element
from nullgraft.operators import RadonOperator, make_operator
from tests.conftest import disk

SHAPE = (48, 48)
SIGMA = 1e-3


@dataclass
class Scenario:
    op: RadonOperator
    decoder: SART
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    eps: float
    region: np.ndarray


def _scenario(n_angles: int, n_iter: int = 300) -> Scenario:
    op = make_operator(SHAPE, n_angles=n_angles, backend="pixel")
    x = resize(shepp_logan_phantom(), SHAPE, anti_aliasing=True).astype(np.float32)
    rng = np.random.default_rng(0)
    noise = (SIGMA * rng.standard_normal(op.sino_shape)).astype(np.float32)
    y = op.forward(x) + noise
    dec = SART(op, n_iter=n_iter, nonneg=False)
    z = dec.reconstruct(y)[0]
    eps = max(2 * float(np.linalg.norm(noise)), 1.2 * float(np.linalg.norm(op.forward(z) - y)))
    return Scenario(op, dec, x, y, z, eps, disk(SHAPE, (24, 24), 12).astype(bool))


@pytest.fixture(scope="module")
def sparse() -> Scenario:
    return _scenario(8)


@pytest.fixture(scope="module")
def dense() -> Scenario:
    return _scenario(120, n_iter=100)


def _assessor(s: Scenario, **kw: object) -> DetailAssessor:
    cfg = AssessmentConfig(eps=s.eps, n_noise=10, noise_sigma=SIGMA, **kw)  # type: ignore[arg-type]
    return DetailAssessor(s.op, s.decoder, cfg, realism=AlwaysRealistic())


def _kernel_source(s: Scenario, amplitude: float = 0.3) -> np.ndarray:
    k, _ = local_kernel_element(s.op, disk(SHAPE, (24, 24), 7).astype(bool), np.random.default_rng(5), smooth=1.0)
    return s.z + amplitude * k / np.abs(k).max()


def test_pure_kernel_detail_gives_hallucination_risk(sparse: Scenario) -> None:
    res = _assessor(sparse).assess(sparse.y, _kernel_source(sparse), sparse.region, z=sparse.z)
    assert res.outcome is Outcome.LIKELY_HALLUCINATION_RISK
    assert res.interval_nonempty
    assert res.eta_min <= res.eta_max and res.eta_min <= res.graft_size / 2
    assert res.delta <= res.delta_limit
    assert len(res.d_to_z) == 10
    assert "not a diagnosis" in res.summary()


def test_large_contrast_detail_outside_kernel_is_visible(dense: Scenario) -> None:
    source = dense.z + disk(SHAPE, (24, 24), 5)
    res = _assessor(dense).assess(dense.y, source, dense.region, z=dense.z)
    assert res.outcome is Outcome.DETAIL_VISIBLE
    assert res.graft_size < 0.1 * res.detail_size


def test_null_detail_does_not_trigger(sparse: Scenario) -> None:
    res = _assessor(sparse).assess(sparse.y, sparse.z, sparse.region, z=sparse.z)
    assert res.outcome is not Outcome.LIKELY_HALLUCINATION_RISK


def test_calibration_returns_threshold_without_false_positives(sparse: Scenario) -> None:
    cfg = AssessmentConfig(eps=sparse.eps, n_noise=6, noise_sigma=SIGMA)
    slices = [sparse.x, np.roll(sparse.x, 3, axis=1)]
    cal = calibrate_null_detail(slices, sparse.op, sparse.decoder, cfg, sparse.region, scales=(0.0, 1e-4))
    assert np.isfinite(cal.threshold) and cal.threshold >= 0
    assert len(cal.records) == 4
    null_runs = [r for r in cal.records if r[1] == 0.0]
    assert all(r[2] != Outcome.LIKELY_HALLUCINATION_RISK.value for r in null_runs)
    tiny = sparse.z + 1e-4 * (_kernel_source(sparse) - sparse.z)
    res = _assessor(sparse, null_threshold=cal.threshold).assess(sparse.y, tiny, sparse.region, z=sparse.z)
    assert res.outcome is not Outcome.LIKELY_HALLUCINATION_RISK


def test_inconsistent_reconstruction_stops_analysis(sparse: Scenario) -> None:
    res = _assessor(sparse).assess(sparse.y, _kernel_source(sparse), sparse.region, z=sparse.z + 0.5)
    assert res.outcome is Outcome.INCONSISTENT and np.isnan(res.eta_min)


def test_blend_invalid_for_leaky_detail(sparse: Scenario) -> None:
    res = _assessor(sparse).assess(sparse.y, sparse.z + disk(SHAPE, (24, 24), 5), sparse.region, z=sparse.z)
    assert res.outcome is Outcome.BLEND_INVALID


class _Reject:
    def check(self, x: np.ndarray, region: np.ndarray) -> RealismVerdict:
        return RealismVerdict(False, 1.0, 0.5)


def test_unrealistic_graft_is_uninformative(sparse: Scenario) -> None:
    cfg = AssessmentConfig(eps=sparse.eps, n_noise=4, noise_sigma=SIGMA)
    res = DetailAssessor(sparse.op, sparse.decoder, cfg, realism=_Reject()).assess(
        sparse.y, _kernel_source(sparse), sparse.region, z=sparse.z
    )
    assert res.outcome is Outcome.UNINFORMATIVE and res.realism is not None


def test_stochastic_decoder_uses_hausdorff_and_is_deterministic(sparse: Scenario) -> None:
    dec = PerturbedDecoder(sparse.decoder, n_samples=4, sigma=1e-4, seed=1)
    cfg = AssessmentConfig(eps=sparse.eps * 2, n_noise=5, noise_sigma=SIGMA, seed=7)
    runs = [DetailAssessor(sparse.op, dec, cfg, realism=AlwaysRealistic()).assess(
        sparse.y, _kernel_source(sparse), sparse.region) for _ in range(2)]
    assert runs[0].outcome is runs[1].outcome
    assert runs[0].d_to_z == runs[1].d_to_z and runs[0].eta_min == runs[1].eta_min


def test_archive_patch_realism() -> None:
    x = resize(shepp_logan_phantom(), (64, 64), anti_aliasing=True).astype(np.float32)
    archive = [x, np.fliplr(x), np.roll(x, 4, axis=0)]
    checker = ArchivePatchRealism(patch_size=16, cells=4, quantile=1.0, margin=1.5).fit(archive, stride=4)
    region = disk((64, 64), (32, 32), 6).astype(bool)
    assert checker.check(x, region).realistic
    noise = np.random.default_rng(0).uniform(-3, 3, (64, 64)).astype(np.float32)
    assert not checker.check(noise, region).realistic
    with pytest.raises(RuntimeError):
        ArchivePatchRealism().check(x, region)


def test_noise_draws_stay_in_eps_ball(rng: np.random.Generator) -> None:
    e = sample_noise((10, 20), 50, sigma=1.0, eps=2.0, rng=rng)
    assert np.all(np.linalg.norm(e.reshape(50, -1), axis=1) < 2.0)
    samples = np.stack([np.zeros((3, 3)), np.ones((3, 3)), 2 * np.ones((3, 3))]).astype(np.float32)
    np.testing.assert_array_equal(select_point(samples), samples[1])
