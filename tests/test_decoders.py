from __future__ import annotations

import numpy as np
import pytest
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from nullgraft.decoders import FBP, SART, TVADMM, PerturbedDecoder, build_decoder
from nullgraft.metrics import LocalizedMetric, rmse
from nullgraft.operators import make_operator
from tests.conftest import disk


@pytest.fixture(scope="module")
def phantom() -> np.ndarray:
    return resize(shepp_logan_phantom(), (64, 64), anti_aliasing=True).astype(np.float32)


@pytest.mark.parametrize("beam", ["parallel", "fan"])
def test_fbp_recovers_disk_amplitude(beam: str) -> None:
    op = make_operator((64, 64), n_angles=180 if beam == "parallel" else 360, beam=beam, backend="pixel")
    x = disk((64, 64), (32, 32), 18)
    z = FBP(op, window="ramp").reconstruct(op.forward(x))[0]
    interior = disk((64, 64), (32, 32), 12).astype(bool)
    assert abs(z[interior].mean() - 1.0) < 0.05
    outside = ~disk((64, 64), (32, 32), 22).astype(bool)
    assert abs(z[outside].mean()) < 0.05


@pytest.mark.parametrize("window", ["ramp", "hann", "shepp-logan", "cosine"])
def test_fbp_windows_reconstruct_phantom(window: str, phantom: np.ndarray) -> None:
    op = make_operator((64, 64), n_angles=120, backend="pixel")
    z = FBP(op, window=window).reconstruct(op.forward(phantom))[0]
    assert rmse(z, phantom) < 0.12


def test_iterative_decoders_beat_fbp_in_sparse_view(phantom: np.ndarray) -> None:
    op = make_operator((64, 64), n_angles=30, backend="pixel")
    y = op.forward(phantom)
    e_fbp = rmse(FBP(op).reconstruct(y)[0], phantom)
    e_sart = rmse(SART(op, n_iter=100).reconstruct(y)[0], phantom)
    e_tv = rmse(TVADMM(op, n_iter=40).reconstruct(y)[0], phantom)
    assert e_sart < e_fbp
    assert e_tv < e_fbp


def test_cgls_matches_lsqr_and_is_consistent(phantom: np.ndarray) -> None:
    from scipy.sparse.linalg import lsqr

    from nullgraft.decoders import CGLS

    op = make_operator((64, 64), n_angles=30, backend="pixel")
    y = op.forward(phantom) + (1e-3 * np.random.default_rng(0).standard_normal(op.sino_shape)).astype(np.float32)
    lin = op.as_linear_operator()
    early = CGLS(op, n_iter=5).reconstruct(y)[0]
    ref5 = lsqr(lin, y.ravel().astype(np.float64), iter_lim=5, atol=0, btol=0)[0]
    np.testing.assert_allclose(early.ravel(), ref5, rtol=1e-3, atol=1e-5)
    # дальше округления разводят методы по компонентам с малыми σ: сравниваем невязки в данных
    z = CGLS(op, n_iter=40).reconstruct(y)[0]
    ref = lsqr(lin, y.ravel().astype(np.float64), iter_lim=40, atol=0, btol=0)[0].reshape(64, 64)
    res_cgls = np.linalg.norm(op.forward(z) - y)
    res_lsqr = np.linalg.norm(op.forward(ref) - y)
    assert res_cgls == pytest.approx(res_lsqr, rel=0.05)
    batch = CGLS(op, n_iter=40).reconstruct_batch(np.stack([y, 2 * y]))
    np.testing.assert_allclose(batch[1, 0], 2 * batch[0, 0], rtol=1e-4, atol=1e-5)  # линейность
    from nullgraft.api import default_eps

    z_full = CGLS(op, n_iter=250).reconstruct(y)[0]
    assert np.linalg.norm(op.forward(z_full) - y) < default_eps(op, 1e-3)  # согласован в смысле Def. 2.1


def test_batch_reconstruction_equals_single(phantom: np.ndarray, rng: np.random.Generator) -> None:
    op = make_operator((64, 64), n_angles=40, backend="pixel")
    y = op.forward(phantom)
    ys = y[None] + 0.01 * rng.standard_normal((3, *y.shape)).astype(np.float32)
    for dec in (FBP(op), SART(op, n_iter=10), TVADMM(op, n_iter=5)):
        batch = dec.reconstruct_batch(ys)
        assert batch.shape == (3, 1, 64, 64) and batch.dtype == np.float32
        np.testing.assert_allclose(batch[1, 0], dec.reconstruct(ys[1])[0], rtol=1e-4, atol=1e-5)


def test_perturbed_decoder_is_stochastic_but_reproducible(phantom: np.ndarray) -> None:
    op = make_operator((64, 64), n_angles=40, backend="pixel")
    dec = PerturbedDecoder(FBP(op), n_samples=5, sigma=0.05, seed=3)
    y = op.forward(phantom)
    s1 = dec.reconstruct(y)
    s2 = dec.reconstruct(y)
    assert s1.shape == (5, 64, 64) and dec.stochastic
    np.testing.assert_array_equal(s1, s2)
    assert np.std(s1, axis=0).mean() > 0
    other = dec.reconstruct(y * 1.001)
    assert not np.array_equal(other, s1)
    assert LocalizedMetric()(s1[0], s1[1]) > 0


def test_build_decoder_from_config() -> None:
    op = make_operator((32, 32), n_angles=20, backend="pixel")
    assert isinstance(build_decoder({"type": "fbp", "window": "hann"}, op), FBP)
    assert isinstance(build_decoder({"type": "tv-admm", "n_iter": 3}, op), TVADMM)
    pert = build_decoder({"type": "perturbed", "base": {"type": "sart", "n_iter": 2}, "n_samples": 3}, op)
    assert isinstance(pert, PerturbedDecoder) and pert.n_samples == 3
    with pytest.raises(ValueError):
        build_decoder({"type": "unet"}, op)


def test_input_validation() -> None:
    op = make_operator((32, 32), n_angles=20, backend="pixel")
    with pytest.raises(ValueError):
        FBP(op).reconstruct_batch(np.zeros((20, op.n_det)))
    with pytest.raises(ValueError):
        FBP(op, window="bogus")  # type: ignore[arg-type]
