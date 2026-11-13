"""Регрессионные тесты на проблемы, найденные при ревью."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from nullgraft.config import load_config, load_slices
from nullgraft.data.vhp import cryo_to_gray, scan_ct_series
from nullgraft.decoders import FBP
from nullgraft.experiments.common import JsonlLog
from nullgraft.feasible import build_patch_archive
from nullgraft.nullspace import NullspaceProjector
from nullgraft.operators import Geometry, RadonOperator, make_operator
from nullgraft.similar import ArtifactIndex, central_fraction
from tests.conftest import disk


@pytest.fixture(scope="module")
def op() -> RadonOperator:
    return make_operator((32, 32), n_angles=8, backend="pixel")


def _slices(seed: int, n: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [gaussian_filter(rng.standard_normal((32, 32)), 2.0).astype(np.float32) for _ in range(n)]


def test_patch_archive_rebuilds_when_data_changes(op: RadonOperator, tmp_path) -> None:
    a = build_patch_archive(_slices(0), op, patch_size=16, stride=8, out_dir=tmp_path)
    b = build_patch_archive(_slices(1), op, patch_size=16, stride=8, out_dir=tmp_path)
    assert not np.array_equal(np.asarray(a.patches), np.asarray(b.patches))
    ref = build_patch_archive(_slices(1), op, patch_size=16, stride=8)
    np.testing.assert_array_equal(np.asarray(b.patches), ref.patches)
    c = build_patch_archive(_slices(1), op, patch_size=8, stride=8, out_dir=tmp_path)
    assert np.asarray(c.patches).shape[1:] == (8, 8)


def test_artifact_index_rebuilds_on_other_decoder_and_reloads_without_pickle(op: RadonOperator, tmp_path) -> None:
    sl = _slices(2)
    first = ArtifactIndex.build(sl, op, FBP(op, window="ramp"), 1e-3, patch_size=16, stride=8, out_dir=tmp_path)
    assert not (tmp_path / "index.pkl").exists()
    same = ArtifactIndex.build(sl, op, FBP(op, window="ramp"), 1e-3, patch_size=16, stride=8, out_dir=tmp_path)
    assert same.fingerprint == first.fingerprint
    np.testing.assert_array_equal(np.asarray(same.descriptors), np.asarray(first.descriptors))
    other = ArtifactIndex.build(sl, op, FBP(op, window="hann"), 1e-3, patch_size=16, stride=8, out_dir=tmp_path)
    assert other.fingerprint != first.fingerprint
    assert not np.array_equal(np.asarray(other.descriptors), np.asarray(first.descriptors))
    reloaded = ArtifactIndex.load(tmp_path)
    np.testing.assert_array_equal(np.asarray(reloaded.descriptors), np.asarray(other.descriptors))


def test_segmentation_structure_uses_labelled_fraction() -> None:
    full = np.ones((1, 16, 16))
    sparse = np.zeros((1, 16, 16))
    sparse[0, 8, 8] = 1
    assert central_fraction(full)[0] == 1.0
    assert central_fraction(sparse)[0] < 0.1


def test_projection_result_is_not_aliased_with_cache(op: RadonOperator, rng: np.random.Generator) -> None:
    proj = NullspaceProjector(op)
    x = rng.standard_normal(op.image_shape).astype(np.float32)
    first = proj.project(x)
    first.x_proj[:] = 0.0
    assert np.abs(proj.project(x).x_proj).max() > 0


def test_jsonl_skips_truncated_last_line(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"slice": 1}\n{"slice": 2}\n{"sli')
    assert JsonlLog(path, key="slice").done == {1, 2}


def test_missing_data_path_is_reported() -> None:
    cfg = load_config(None)
    cfg["data"] = {"kind": "npy"}
    with pytest.raises(FileNotFoundError, match="--data"):
        load_slices(cfg, None)


def test_parallel_fbp_with_full_turn_has_correct_scale() -> None:
    x = disk((64, 64), (32, 32), 18)
    half = RadonOperator(Geometry((64, 64), 180), backend="pixel")
    full = RadonOperator(Geometry((64, 64), 360, angular_range=2 * math.pi), backend="pixel")
    inner = disk((64, 64), (32, 32), 12).astype(bool)
    for op_ in (half, full):
        z = FBP(op_).reconstruct(op_.forward(x))[0]
        assert abs(z[inner].mean() - 1.0) < 0.05


def test_natural_sort_of_image_slices(tmp_path) -> None:
    for i in (1, 2, 10):
        np.save(tmp_path / f"slice{i}.npy", np.full((4, 4), i, np.float32))
    (tmp_path / "ct_meta.json").write_text(json.dumps({"pixel_spacing_mm": [1, 1], "slice_spacing_mm": 1}))
    names = [s.path.name for s in scan_ct_series(tmp_path).slices]
    assert names == ["slice1.npy", "slice2.npy", "slice10.npy"]


def test_dark_uint8_cryo_is_not_rescaled() -> None:
    dark = np.ones((4, 4, 3), np.uint8)
    assert cryo_to_gray(dark).max() == pytest.approx(1 / 255, rel=1e-3)
