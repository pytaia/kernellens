from __future__ import annotations

import json

import numpy as np
import pytest

from nullgraft_demo.config import load_config, recon_params
from nullgraft_demo.state import DetailParams, ReconParams, SessionState


def test_session_roundtrip(tmp_path, rng: np.random.Generator) -> None:
    mask = rng.random((17, 23)) > 0.6
    state = SessionState(
        source_kind="synthetic", slice_index=3,
        params=ReconParams(n_angles=30, decoder="fbp-hann", sigma=2e-3),
        detail=DetailParams(source="file", file_path="x.png", window_width=4.5),
        window_level=(0.3, 0.8), mask=mask,
        arrays={"z": rng.standard_normal((17, 23)).astype(np.float32)},
        results={"outcome": "DETAIL_VISIBLE", "eta": (0.1, float("nan")), "n": np.int64(5)},
    )
    json_path, npz_path = state.save(tmp_path / "s1.session")
    assert json_path.exists() and npz_path.exists()
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["results"]["eta"] == [0.1, None]
    back = SessionState.load(tmp_path / "s1.session")
    assert back.params == state.params and back.detail == state.detail
    assert back.slice_index == 3 and back.window_level == (0.3, 0.8)
    np.testing.assert_array_equal(back.mask, mask)
    np.testing.assert_array_equal(back.arrays["z"], state.arrays["z"])
    assert back.results["outcome"] == "DETAIL_VISIBLE"


def test_session_without_mask_and_version_check(tmp_path) -> None:
    SessionState().save(tmp_path / "empty")
    assert SessionState.load(tmp_path / "empty").mask is None
    meta_path = tmp_path / "empty.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["version"] = 999
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        SessionState.load(tmp_path / "empty")


def test_unknown_fields_are_ignored_on_load(tmp_path) -> None:
    SessionState().save(tmp_path / "s")
    meta_path = tmp_path / "s.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["params"]["obsolete_option"] = 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert SessionState.load(tmp_path / "s").params == ReconParams()


def test_params_are_immutable_and_config_defaults() -> None:
    p = ReconParams()
    q = p.with_(n_angles=120)
    assert p.n_angles == 60 and q.n_angles == 120
    with pytest.raises(AttributeError):
        p.n_angles = 1  # type: ignore[misc]
    cfg = load_config()
    assert recon_params(cfg).decoder == cfg["recon"]["decoder"]
