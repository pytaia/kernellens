from __future__ import annotations

import json

import numpy as np
import pytest

from nullgraft.data.splits import block_split, region_split
from nullgraft.data.synthetic import iter_synthetic, random_ellipse_phantom, shepp_logan, synthetic_volume
from nullgraft.data.vhp import cryo_to_ct_grid, cryo_to_gray, hu_to_mu, scan_ct_series


def test_block_split_has_no_adjacent_leakage() -> None:
    split = block_split(200, block_size=20, test_fraction=0.3, gap=3, seed=1)
    assert split.train.size and split.test.size
    dist = np.min(np.abs(split.train[:, None] - split.test[None]), axis=0)
    assert dist.min() > 3
    blocks_test = set(split.test // 20)
    assert blocks_test.isdisjoint(set(split.train // 20))
    assert block_split(200, 20, 0.3, 3, seed=1).test.tolist() == split.test.tolist()


def test_region_split() -> None:
    s = region_split(["head", "head", "thorax", "abdomen"], ["thorax"])
    assert s.test.tolist() == [2] and s.train.tolist() == [0, 1, 3]
    with pytest.raises(ValueError):
        region_split(["head"], ["pelvis"])


def test_synthetic_generators_are_deterministic() -> None:
    assert shepp_logan(32).shape == (32, 32)
    v1, v2 = synthetic_volume(5, 32, seed=2), synthetic_volume(5, 32, seed=2)
    np.testing.assert_array_equal(v1, v2)
    near = np.abs(v1[0] - v1[1]).mean()
    far = np.abs(v1[0] - v1[4]).mean()
    assert near < far
    assert len(list(iter_synthetic(3, 16))) == 3
    assert random_ellipse_phantom(16, np.random.default_rng(0)).min() >= 0


def test_image_ct_folder_reports_metadata(tmp_path) -> None:
    for i in range(3):
        np.save(tmp_path / f"s{i:03d}.npy", np.full((8, 8), i, np.float32))
    with pytest.raises(ValueError):
        scan_ct_series(tmp_path)
    (tmp_path / "ct_meta.json").write_text(json.dumps({"pixel_spacing_mm": [0.9, 0.9], "slice_spacing_mm": 1.0}))
    rep = scan_ct_series(tmp_path).report()
    assert rep["n_slices"] == 3 and rep["pixel_spacing_mm"] == [(0.9, 0.9)]
    assert rep["slice_spacing_mm_median"] == pytest.approx(1.0)


def test_cryo_conversion_and_resampling() -> None:
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[10:30, 20:40] = 255
    gray = cryo_to_gray(rgb)
    assert gray.max() == pytest.approx(1.0)
    out = cryo_to_ct_grid(gray, cryo_pixel_mm=0.5, ct_pixel_mm=(1.0, 1.0), ct_shape=(32, 32))
    assert out.shape == (32, 32)
    assert out.sum() == pytest.approx(gray.sum() / 4, rel=0.05)
    assert hu_to_mu(np.array([-1000.0, 0.0]))[0] == 0.0
