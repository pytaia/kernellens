from __future__ import annotations

import numpy as np
import pytest

from nullgraft.assess import Outcome
from nullgraft_demo.compute import reconstruct
from nullgraft_demo.datasource import synthetic_source
from nullgraft_demo.detail import STAGES, detail_source, preview_detail, run_check
from nullgraft_demo.state import DetailParams, ReconParams
from nullgraft_demo.tools import disk_region
from nullgraft_demo.workers import Cancelled, CancelToken

PARAMS = ReconParams(n_angles=20, decoder="cgls", decoder_iters=120, n_noise=6, sigma=1e-3)


@pytest.fixture(scope="module")
def scene():
    src = synthetic_source(size=64, n_slices=8)
    x = src.get(0)
    recon = reconstruct(x, PARAMS, None, token=CancelToken(), progress=lambda f, m: None)
    mask = disk_region(x.shape, 32, 26, 6)
    return src, x, recon, mask


def test_reconstruction_is_consistent(scene) -> None:
    _, _, recon, _ = scene
    assert recon.consistent, (recon.consistency, recon.eps)


def test_detail_sources(scene) -> None:
    src, x, recon, mask = scene
    from scipy.ndimage import binary_dilation

    removed = detail_source(DetailParams(mode="remove"), recon.z, mask)
    outside = ~binary_dilation(mask, iterations=1)
    assert removed.shape == x.shape and np.allclose(removed[outside], recon.z[outside], atol=1e-6)
    other = src.get(3)
    shifted = detail_source(DetailParams(mode="add", source="slice"), recon.z, mask, other, offset=(2, -3))
    assert shifted[10, 10] == pytest.approx(other[12, 7])
    patch = detail_source(DetailParams(mode="add", source="file"), recon.z, mask, file_image=np.ones((50, 70)))
    assert patch.shape == (13, 13)


def test_preview_and_stages(scene) -> None:
    src, x, recon, mask = scene
    source = detail_source(DetailParams(mode="add", source="slice"), recon.z, mask, src.get(4))
    prev = preview_detail(recon.z, source, mask, PARAMS, token=CancelToken(), progress=lambda f, m: None)
    np.testing.assert_allclose(prev.x_det, prev.pinv + prev.x_proj, atol=1e-4)
    assert prev.kernel_ratio < 1e-2

    events: list[tuple[str, str]] = []
    fractions: list[float] = []
    res = run_check(recon, source, mask, PARAMS, DetailParams(mode="add"), batch_size=2,
                    on_stage=lambda s, st, d: events.append((s, st)), token=CancelToken(),
                    progress=lambda f, m: fractions.append(f))
    outcome = res.assessment.outcome
    assert isinstance(outcome, Outcome)
    assert events[0] == ("consistency", "running") and ("consistency", "done") in events
    decoder_steps = [e for e in events if e[0] == "decoder"]
    if outcome not in (Outcome.BLEND_INVALID, Outcome.DETAIL_VISIBLE, Outcome.INCONCLUSIVE):
        assert len([e for e in decoder_steps if e[1] in ("running", "done")]) >= 3  # батчи по 2 из 6
        assert {"out_mean", "out_std", "x_graft"} <= set(res.assessment.extras)
    assert events[-1][0] == "classify" and fractions == sorted(fractions)
    assert res.first_stage_seconds <= res.seconds
    assert [s for s in STAGES if any(e[0] == s for e in events)][0] == "consistency"


def test_check_can_be_cancelled(scene) -> None:
    src, x, recon, mask = scene
    token = CancelToken()
    source = detail_source(DetailParams(mode="remove"), recon.z, mask)

    def cancel_on_projection(stage: str, status: str, data: dict) -> None:
        if stage == "projection" and status == "done":
            token.cancel()

    with pytest.raises(Cancelled):
        run_check(recon, source, mask, PARAMS, DetailParams(), on_stage=cancel_on_projection, token=token,
                  progress=lambda f, m: None)
