"""Эксперимент по B1: диаметры допустимых множеств тестовых патчей по архиву train-срезов."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from nullgraft.config import build_operator, load_slices
from nullgraft.data.splits import block_split
from nullgraft.experiments.common import maybe_plot, write_csv, write_json
from nullgraft.feasible import (
    FeasibleSetEstimator,
    SinogramPooling,
    build_patch_archive,
    convergence_curve,
    diameters,
    patch_measurement,
)
from nullgraft.metrics import LocalizedMetric

logger = logging.getLogger(__name__)


def run(cfg: dict[str, Any], out_dir: str | Path, data_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    op = build_operator(cfg)
    vol = load_slices(cfg, data_path)
    d, f = cfg["data"], cfg["feasible"]
    split = block_split(len(vol), int(d["block_size"]), float(d["test_fraction"]), int(d["gap"]), int(cfg["seed"]))
    pooling = SinogramPooling(*[int(v) for v in f["pool"]])
    common = {"patch_size": int(f["patch_size"]), "stride": int(f["stride"]), "placement": f["placement"],
              "pooling": pooling}
    train = [vol[i] for i in split.train]
    archive = build_patch_archive(train, op, out_dir=out / "archive", **common)
    queries = build_patch_archive([vol[i] for i in split.test], op, **common)
    rng = np.random.default_rng(int(cfg["seed"]))
    pick = rng.choice(len(queries), min(int(f["n_queries"]), len(queries)), replace=False)
    ys = patch_measurement(op, queries, pick)
    ys = ys + (float(cfg["noise"]["sigma"]) * rng.standard_normal(ys.shape)).astype(np.float32)
    eps = float(f["eps_relative"]) * float(np.median(np.linalg.norm(ys.reshape(len(ys), -1), axis=1)))
    est = FeasibleSetEstimator(op, archive, n_components=f["n_components"], seed=int(cfg["seed"]))
    info = est.feasible_matrix(ys, eps)
    metric = LocalizedMetric(p=float(cfg["assess"]["p"]), q=float(cfg["assess"]["q"]))
    res = diameters(archive, info, metric, exact_limit=int(f["exact_limit"]), seed=int(cfg["seed"]))
    rows = []
    for r, qi in zip(res, pick, strict=True):
        pair = [archive.meta[p].tolist() for p in r.pair] if r.pair else None
        rows.append({"query": int(qi), "query_meta": queries.meta[qi].tolist(), "size": r.size,
                     "diameter_lower_bound": r.diameter, "decoder_error_lower_bound": r.decoder_error_lower_bound,
                     "approximate": r.approximate, "pair_meta": pair})
    fr, curve = convergence_curve(archive, info, metric=metric, exact_limit=int(f["exact_limit"]))
    write_csv(out / "diameters.csv", rows)
    write_csv(out / "convergence.csv", [{"fraction": float(fv), **{f"q{j}": float(v) for j, v in enumerate(row)}}
                                        for fv, row in zip(fr, curve, strict=True)])
    maybe_plot("convergence", out / "convergence.png", fractions=fr, curve=curve)
    informative = [r.diameter for r in res if r.size >= 2]
    summary = {"eps": eps, "n_archive": len(archive), "n_queries": len(ys), "n_candidates": info.n_candidates,
               "n_feasible_pairs": int(info.fa.nnz), "n_queries_with_pairs": len(informative),
               "median_diameter": float(np.median(informative)) if informative else None,
               "note": "diameters are lower bounds (Thm. 3.9); any decoder errs by >= diam/2 on F_y"}
    write_json(out / "feasible_summary.json", summary)
    return summary
