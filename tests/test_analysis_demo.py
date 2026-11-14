from __future__ import annotations

import numpy as np
import pytest

from nullgraft_demo.analysis import archive_indices, compare_decoders, feasible_limit, risk_map, similar_artifacts
from nullgraft_demo.compute import reconstruct
from nullgraft_demo.datasource import synthetic_source
from nullgraft_demo.detail import detail_source
from nullgraft_demo.state import DetailParams, ReconParams
from nullgraft_demo.tools import disk_region
from nullgraft_demo.workers import CancelToken, ResultCache

PARAMS = ReconParams(n_angles=15, decoder="cgls", decoder_iters=100, n_noise=4)
NOOP = {"token": CancelToken(), "progress": lambda f, m: None}


@pytest.fixture(scope="module")
def scene():
    src = synthetic_source(size=48, n_slices=8)
    x = src.get(1)
    mask = disk_region(x.shape, 24, 24, 7)
    return src, x, mask


def test_archive_indices_skip_neighbours() -> None:
    assert 4 not in archive_indices(10, 4) and 6 not in archive_indices(10, 4) and 8 in archive_indices(10, 4)
    assert archive_indices(3, 1) == [0, 2]


def test_compare_uses_one_sinogram(scene, tmp_path) -> None:
    src, x, mask = scene
    rows = compare_decoders(x, PARAMS, ["cgls", "fbp-hann"], mask, DetailParams(),
                            lambda z: detail_source(DetailParams(), z, mask), ResultCache(tmp_path), 0.2, **NOOP)
    assert [r.decoder for r in rows] == ["cgls", "fbp-hann"]
    np.testing.assert_array_equal(rows[0].recon.y, rows[1].recon.y)
    assert all(np.isfinite(v) for r in rows for v in r.metrics.values())
    assert rows[0].check is not None


def test_similar_artifacts(scene, tmp_path) -> None:
    src, x, mask = scene
    z = reconstruct(x, PARAMS, None, **NOOP).z
    res = similar_artifacts(src, 1, z, mask, PARAMS, ResultCache(tmp_path / "c"), tmp_path / "b2", patch_size=16, k=4,
                            **NOOP)
    assert len(res.items) == 4 and res.n_index > 0
    assert res.fraction_real + res.fraction_background == pytest.approx(1.0)
    assert res.items[0].truth.shape == (16, 16) and res.items[0].slice_index != 1


def test_feasible_limit_is_cached_and_monotone_curve(scene, tmp_path) -> None:
    src, x, mask = scene
    cache = ResultCache(tmp_path)
    res = feasible_limit(src, 1, mask, PARAMS, [8, 15], cache, patch_size=16, **NOOP)
    assert res.angles == [8, 15] and len(res.diameters) == 2 and res.n_archive > 0
    assert np.all(np.diff(res.curve, axis=0) >= -1e-9)
    again = feasible_limit(src, 1, mask, PARAMS, [8, 15], cache, patch_size=16, **NOOP)
    assert again.from_cache and again.diameters == pytest.approx(res.diameters)


def test_risk_map_is_rough_but_bounded(scene) -> None:
    src, x, mask = scene
    recon = reconstruct(x, PARAMS, None, **NOOP)
    rm = risk_map(recon, PARAMS, step=16, radius=6, n_noise=2, **NOOP)
    assert rm.centres and np.all((rm.risk >= 0) & (rm.risk <= 1))
    assert np.isfinite(rm.image).any()


def test_docx_report_and_png(tmp_path, scene) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from docx import Document
    from matplotlib.figure import Figure

    from nullgraft_demo.report import save_figures, write_docx

    fig = Figure()
    fig.add_subplot().imshow(scene[1])
    path = write_docx(tmp_path / "r.docx", PARAMS, DetailParams(), "Синтетика", 1, "R: 10 пикс.", None, {"сетка": fig})
    text = "\n".join(p.text for p in Document(str(path)).paragraphs)
    assert "границу возможного" in text and "доказанн" not in text
    assert save_figures({"a": fig}, tmp_path / "png")[0].exists()
