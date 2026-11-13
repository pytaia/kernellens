from __future__ import annotations

import numpy as np
import pytest
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from nullgraft.decoders import FBP
from nullgraft.operators import RadonOperator, make_operator
from nullgraft.similar import (
    ArtifactIndex,
    DescriptorConfig,
    artifact_estimate,
    describe,
    ncc,
    truth_contrast,
)

SHAPE = (64, 64)


@pytest.fixture(scope="module")
def op() -> RadonOperator:
    return make_operator(SHAPE, n_angles=16, backend="pixel")


@pytest.fixture(scope="module")
def slices() -> list[np.ndarray]:
    base = resize(shepp_logan_phantom(), SHAPE, anti_aliasing=True).astype(np.float32)
    return [base, np.fliplr(base).copy(), np.roll(base, 6, axis=0), np.zeros(SHAPE, np.float32) + 0.1 * base]


@pytest.fixture(scope="module")
def index(op: RadonOperator, slices: list[np.ndarray]) -> ArtifactIndex:
    return ArtifactIndex.build(slices, op, FBP(op), noise_sigma=1e-3, patch_size=16, stride=8, batch_slices=3)


def test_descriptor_properties(rng: np.random.Generator) -> None:
    cfg = DescriptorConfig()
    p = rng.standard_normal((16, 16)).astype(np.float32)
    d1, d2 = describe(np.stack([p, 5 * p]), cfg)
    n_patch = cfg.cells**2
    np.testing.assert_allclose(d1[:n_patch], d2[:n_patch], atol=1e-5)
    np.testing.assert_allclose(d1[n_patch : n_patch + cfg.n_radial], d2[n_patch : n_patch + cfg.n_radial], atol=1e-5)
    assert d1[n_patch + cfg.n_radial] != d2[n_patch + cfg.n_radial]
    xx = np.tile(np.sin(np.arange(16) * 0.8), (16, 1)).astype(np.float32)
    vertical, horizontal = describe(np.stack([xx, xx.T]), cfg)
    stats_v, stats_h = vertical[-3:], horizontal[-3:]
    assert stats_v[2] > 0.9 and stats_h[2] > 0.9
    assert stats_v[0] * stats_h[0] < 0


def test_ncc_and_contrast(rng: np.random.Generator) -> None:
    a = rng.standard_normal((3, 8, 8))
    np.testing.assert_allclose(ncc(a, a), 1.0)
    np.testing.assert_allclose(ncc(a, -a), -1.0)
    flat = np.zeros((1, 10, 10), np.float32)
    edge = np.zeros((1, 10, 10), np.float32)
    edge[0, :, 5:] = 1.0
    assert truth_contrast(flat)[0] == 0 and truth_contrast(edge)[0] == pytest.approx(1.0)
    assert artifact_estimate(np.ones((10, 10)), 2.0).max() < 1e-5


def test_query_finds_itself_first(index: ArtifactIndex, op: RadonOperator, slices: list[np.ndarray]) -> None:
    assert len(index) == 4 * 49
    n = 60
    matches = index.query(np.asarray(index.artifacts[[n, n + 1]]), k=5)
    assert len(matches) == 2 and len(matches[0]) == 5
    top = matches[0][0]
    assert top.index == n and top.descriptor_distance == pytest.approx(0.0, abs=1e-5)
    assert top.ncc == pytest.approx(1.0, abs=1e-5)
    assert all(m.descriptor_distance >= top.descriptor_distance for m in matches[0])


def test_query_from_reconstruction_and_summary(index: ArtifactIndex, op: RadonOperator,
                                               slices: list[np.ndarray]) -> None:
    rng = np.random.default_rng(9)
    y = op.forward(slices[0]) + (1e-3 * rng.standard_normal(op.sino_shape)).astype(np.float32)
    z = FBP(op).reconstruct(y)[0]
    matches = index.query_from_image(z, [(32, 32), (8, 8)], k=4)
    summary = ArtifactIndex.summarize(matches)
    assert summary.n_matches == 8
    assert summary.fraction_real_structure + summary.fraction_background == pytest.approx(1.0)
    assert "similar artifacts" in str(summary)
    centre = [m for m in index.query(np.asarray(index.artifacts[[24]]), k=1)[0]]
    assert centre[0].is_structure
    bg = int(np.flatnonzero((index.meta[:, 0] == 3) & (index.meta[:, 1] == 8) & (index.meta[:, 2] == 8))[0])
    assert not bool(index.contrast[bg] > index.structure_threshold)


def test_save_load_roundtrip_and_resume(op: RadonOperator, slices: list[np.ndarray], index: ArtifactIndex,
                                        tmp_path) -> None:
    class Crashing(list):
        def __getitem__(self, i):  # type: ignore[override]
            if isinstance(i, slice) and (i.start or 0) >= 2:
                raise RuntimeError("simulated interruption")
            return super().__getitem__(i)

    with pytest.raises(RuntimeError):
        ArtifactIndex.build(Crashing(slices), op, FBP(op), noise_sigma=1e-3, patch_size=16, stride=8, batch_slices=2,
                            out_dir=tmp_path)
    built = ArtifactIndex.build(slices, op, FBP(op), noise_sigma=1e-3, patch_size=16, stride=8, batch_slices=2,
                                out_dir=tmp_path)
    loaded = ArtifactIndex.load(tmp_path)
    assert len(loaded) == len(built) == len(index)
    q = np.asarray(index.artifacts[[5, 77]])
    a = [[m.index for m in ms] for ms in built.query(q, k=3)]
    b = [[m.index for m in ms] for ms in loaded.query(q, k=3)]
    assert a == b


def test_true_residual_mode(op: RadonOperator, slices: list[np.ndarray]) -> None:
    idx = ArtifactIndex.build(slices[:2], op, FBP(op), noise_sigma=1e-3, patch_size=16, stride=16,
                              residual_mode="true")
    res = idx.query(np.asarray(idx.artifacts[[3]]), k=2)
    assert res[0][0].index == 3
    with pytest.raises(ValueError):
        idx.query_from_image(np.zeros(SHAPE), [(32, 32)])


def test_build_is_deterministic(op: RadonOperator, slices: list[np.ndarray], index: ArtifactIndex) -> None:
    again = ArtifactIndex.build(slices, op, FBP(op), noise_sigma=1e-3, patch_size=16, stride=8, batch_slices=3)
    np.testing.assert_array_equal(again.descriptors, index.descriptors)
