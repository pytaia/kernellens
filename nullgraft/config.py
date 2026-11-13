"""YAML-конфиги экспериментов → оператор, декодер, данные; путь к данным только из ``--data`` или ``data.path``."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from nullgraft.decoders import Decoder, build_decoder
from nullgraft.operators import Geometry, RadonOperator

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "n_jobs": 1,
    "operator": {"shape": [128, 128], "n_angles": 60, "beam": "parallel", "backend": "auto", "matrix_budget_mb": 512},
    "noise": {"sigma": 1e-3, "eps_factor": 1.5},
    "lsqr": {"atol": 1e-7, "btol": 1e-7, "iter_lim": 500},
    "decoder": {"type": "sart", "n_iter": 200, "nonneg": False},
    "data": {"kind": "synthetic", "n_slices": 40, "size": 128, "block_size": 10, "test_fraction": 0.25, "gap": 1},
    "assess": {"n_noise": 20, "p": 1, "q": 1, "region_radius": 12, "window": "cosine", "window_width": 3,
               "delta_factor": 0.5, "source_offset": 5, "calibrate": True, "n_calibration": 4},
    "feasible": {"patch_size": 32, "stride": 16, "placement": "original", "pool": [4, 4], "n_components": 32,
                 "eps_relative": 0.05, "n_queries": 20, "exact_limit": 1000},
    "similar": {"patch_size": 32, "stride": 16, "k": 10, "n_queries": 10},
    "benchmark": {"sizes": [1000, 10000, 100000], "max_seconds": {"forward": 5.0, "detail_check": 120.0}},
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    return deep_merge(DEFAULTS, user)


def build_operator(cfg: Mapping[str, Any]) -> RadonOperator:
    o = dict(cfg["operator"])
    backend = o.pop("backend", "auto")
    budget = float(o.pop("matrix_budget_mb", 512))
    o["shape"] = tuple(int(v) for v in o["shape"])
    return RadonOperator(Geometry(**o), backend=backend, matrix_budget_mb=budget)


def build_decoder_from_config(cfg: Mapping[str, Any], op: RadonOperator) -> Decoder:
    return build_decoder(cfg["decoder"], op)


def eps_from_noise(cfg: Mapping[str, Any], op: RadonOperator) -> float:
    """``ε = eps_factor · σ · √n_rays``, так как ``E‖e‖² = σ² n_rays``."""
    return float(cfg["noise"]["eps_factor"]) * float(cfg["noise"]["sigma"]) * math.sqrt(op.shape[0])


def load_slices(cfg: Mapping[str, Any], data_path: str | Path | None) -> NDArray[np.float32]:
    """Эталонные срезы на сетке оператора: ``synthetic``, ``npy`` ``(N, H, W)`` или ``vhp_registered``."""
    d = cfg["data"]
    kind = d["kind"]
    shape = tuple(cfg["operator"]["shape"])
    if kind == "synthetic":
        from nullgraft.data.synthetic import synthetic_volume

        vol = synthetic_volume(int(d["n_slices"]), int(d.get("size", shape[0])), seed=int(cfg["seed"]))
    else:
        raw = data_path or d.get("path")
        if not raw:  # проверка до Path(): Path("") — это "." и он всегда существует
            raise FileNotFoundError("dataset path missing: pass --data PATH (data is never downloaded)")
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"dataset path does not exist: {path}")
        if kind == "npy":
            vol = np.load(path, mmap_mode="r")
        elif kind == "vhp_registered":
            from nullgraft.data.vhp import load_registered_pairs

            _, vol, _ = load_registered_pairs(path)
        else:
            raise ValueError(f"unknown data.kind {kind!r}")
    if tuple(vol.shape[1:]) != shape:
        raise ValueError(f"slices have shape {vol.shape[1:]}, operator expects {shape}")
    return vol
