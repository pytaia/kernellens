"""Эксперимент по Алгоритму A на тестовых срезах; результаты в ``assess.jsonl`` (с возобновлением)."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from nullgraft.assess import AssessmentConfig, DetailAssessor, calibrate_null_detail, sample_noise, select_point
from nullgraft.config import build_decoder_from_config, build_operator, eps_from_noise, load_slices
from nullgraft.data.splits import block_split
from nullgraft.experiments.common import JsonlLog, disk_mask, most_suspicious_center, write_json
from nullgraft.graft import Grafter
from nullgraft.nullspace import LsqrConfig, NullspaceProjector

logger = logging.getLogger(__name__)


def run(cfg: dict[str, Any], out_dir: str | Path, data_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    op = build_operator(cfg)
    decoder = build_decoder_from_config(cfg, op)
    vol = load_slices(cfg, data_path)
    d, a = cfg["data"], cfg["assess"]
    split = block_split(len(vol), int(d["block_size"]), float(d["test_fraction"]), int(d["gap"]), int(cfg["seed"]))
    eps = eps_from_noise(cfg, op)
    sigma = float(cfg["noise"]["sigma"])
    acfg = AssessmentConfig(eps=eps, n_noise=int(a["n_noise"]), noise_sigma=sigma, p=float(a["p"]), q=float(a["q"]),
                            seed=int(cfg["seed"]))
    radius = int(a["region_radius"])
    center = (op.image_shape[0] // 2, op.image_shape[1] // 2)
    if a.get("calibrate", True):
        train = [np.asarray(vol[i]) for i in split.train[: int(a["n_calibration"])]]
        cal = calibrate_null_detail(train, op, decoder, acfg, disk_mask(op.image_shape, center, radius),
                                    n_jobs=int(cfg["n_jobs"]), seed=int(cfg["seed"]))
        write_json(out / "calibration.json", {"threshold": cal.threshold, "false_positive_rate": cal.false_positive_rate,
                                              "outcomes": cal.outcomes})
        acfg = AssessmentConfig(**{**acfg.__dict__, "null_threshold": cal.threshold})
    lsqr_cfg = LsqrConfig(**cfg["lsqr"])
    grafter = Grafter(op, NullspaceProjector(op, lsqr_cfg), window=a["window"], window_width=float(a["window_width"]),
                      delta_factor=float(a["delta_factor"]))
    assessor = DetailAssessor(op, decoder, acfg, grafter=grafter)
    log = JsonlLog(out / "assess.jsonl", key="slice")
    for i in split.test:
        if int(i) in log.done:
            continue
        rng = np.random.default_rng([int(cfg["seed"]), int(i)])
        x = np.asarray(vol[i], np.float32)
        y = op.forward(x) + sample_noise(op.sino_shape, 1, sigma, eps, rng)[0]
        z = select_point(decoder.reconstruct(y))
        c = most_suspicious_center(z, radius)
        region = disk_mask(op.image_shape, c, radius)
        src = np.asarray(vol[(int(i) + int(a["source_offset"])) % len(vol)], np.float32)
        res = assessor.assess(y, src, region, z=z)
        logger.info("slice %d: %s", i, res.summary())
        log.append({"slice": int(i), "center": c, "outcome": res.outcome.value, "eta_min": res.eta_min,
                    "eta_max": res.eta_max, "graft_size": res.graft_size, "detail_size": res.detail_size,
                    "delta": res.delta, "delta_limit": res.delta_limit, "consistency": res.consistency_residual,
                    "eps": eps, "notes": list(res.notes)})
    counts = Counter(r["outcome"] for r in log.records())
    summary = {"eps": eps, "n_test": int(len(split.test)), "outcomes": dict(counts),
               "note": "outcomes are bounds on what is possible, not diagnoses"}
    write_json(out / "assess_summary.json", summary)
    return summary
