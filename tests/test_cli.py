"""Сквозные smoke-тесты пайплайна экспериментов на маленькой синтетике."""

from __future__ import annotations

import json

import pytest
import yaml

from nullgraft.cli import main

TINY = {
    "operator": {"shape": [48, 48], "n_angles": 12, "backend": "pixel"},
    "decoder": {"type": "sart", "n_iter": 50, "nonneg": False},
    "noise": {"sigma": 1.0e-3, "eps_factor": 3.0},
    "data": {"kind": "synthetic", "n_slices": 12, "size": 48, "block_size": 3, "test_fraction": 0.34, "gap": 0},
    "assess": {"n_noise": 4, "region_radius": 6, "n_calibration": 2, "source_offset": 5},
    "feasible": {"patch_size": 16, "stride": 8, "pool": [2, 2], "n_queries": 5, "n_components": 8},
    "similar": {"patch_size": 16, "stride": 8, "k": 3, "n_queries": 2},
}


@pytest.fixture()
def config(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY))
    return path


@pytest.mark.slow
@pytest.mark.parametrize("cmd", ["assess", "feasible", "similar"])
def test_cli_runs_end_to_end(cmd: str, config, tmp_path, capsys) -> None:
    out = tmp_path / "run"
    assert main(["--log-level", "WARNING", cmd, "--config", str(config), "--out", str(out)]) == 0
    summary = json.loads((out / f"{cmd}_summary.json").read_text())
    assert summary
    if cmd == "assess":
        assert sum(summary["outcomes"].values()) == summary["n_test"]
        lines = (out / "assess.jsonl").read_text().splitlines()
        main(["--log-level", "WARNING", cmd, "--config", str(config), "--out", str(out)])
        assert (out / "assess.jsonl").read_text().splitlines() == lines
    if cmd == "feasible":
        assert (out / "convergence.csv").exists() and (out / "diameters.csv").exists()


def test_cli_requires_data_path_for_real_datasets(tmp_path) -> None:
    cfg = tmp_path / "vhp.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "data": {**TINY["data"], "kind": "vhp_registered"}}))
    with pytest.raises(FileNotFoundError):
        main(["assess", "--config", str(cfg), "--out", str(tmp_path / "o")])
