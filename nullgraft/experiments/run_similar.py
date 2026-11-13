"""Эксперимент по B2: поиск похожих артефактов для подозрительных участков тестовых срезов."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from nullgraft.assess import sample_noise, select_point
from nullgraft.config import build_decoder_from_config, build_operator, eps_from_noise, load_slices
from nullgraft.data.splits import block_split
from nullgraft.experiments.common import most_suspicious_center, write_csv, write_json
from nullgraft.similar import ArtifactIndex

logger = logging.getLogger(__name__)


def run(cfg: dict[str, Any], out_dir: str | Path, data_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    op = build_operator(cfg)
    decoder = build_decoder_from_config(cfg, op)
    vol = load_slices(cfg, data_path)
    d, s = cfg["data"], cfg["similar"]
    split = block_split(len(vol), int(d["block_size"]), float(d["test_fraction"]), int(d["gap"]), int(cfg["seed"]))
    sigma = float(cfg["noise"]["sigma"])
    eps = eps_from_noise(cfg, op)
    index =ArtifactIndex.build([vol[i] for i in split.train], op, decoder, sigma, eps=eps,
                                patch_size=int(s["patch_size"]), stride=int(s["stride"]),
                                out_dir=out / "artifact_index", seed=int(cfg["seed"]))
    rows, all_matches = [], []
    for i in split.test[: int(s["n_queries"])]:
        rng = np.random.default_rng([int(cfg["seed"]) + 1, int(i)])
        x = np.asarray(vol[i], np.float32)
        z = select_point(decoder.reconstruct(op.forward(x) + sample_noise(op.sino_shape, 1, sigma, eps, rng)[0]))
        c = most_suspicious_center(z, int(s["patch_size"]) // 2)
        matches = index.query_from_image(z, [c], k=int(s["k"]))[0]
        all_matches.append(matches)
        for rank, m in enumerate(matches):
            rows.append({"query_slice": int(i), "query_center": c, "rank": rank, **m.__dict__})
    summary = ArtifactIndex.summarize(all_matches)
    logger.info("%s", summary)
    write_csv(out / "similar_matches.csv", rows)
    result = {"n_index": len(index), **summary.__dict__}
    write_json(out / "similar_summary.json", result)
    return result
