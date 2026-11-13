"""Бенчмарки: время и пиковая память (tracemalloc) для A/Aᵀ, Алгоритма A, B1 и B2; вывод в JSON и CSV."""

from __future__ import annotations

import logging
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from nullgraft.assess import AlwaysRealistic, AssessmentConfig, DetailAssessor, select_point
from nullgraft.data.synthetic import synthetic_volume
from nullgraft.decoders import FBP, SART
from nullgraft.experiments.common import disk_mask, write_csv, write_json
from nullgraft.feasible import (
    FeasibleSetEstimator,
    SinogramPooling,
    build_patch_archive,
    diameters,
    patch_measurement,
    patch_positions,
)
from nullgraft.operators import Geometry, RadonOperator
from nullgraft.similar import ArtifactIndex

logger = logging.getLogger(__name__)


TRACE_MEMORY = True


def measure(name: str, fn: Callable[[], Any], repeat: int = 1, **info: Any) -> tuple[dict[str, Any], Any]:
    """Лучшее время по ``repeat`` прогонам без трассировки и пиковая память по отдельному прогону."""
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    peak = float("nan")
    if TRACE_MEMORY:
        tracemalloc.start()
        fn()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    row = {"name": name, "seconds": best, "peak_mb": peak / 2**20, **info}
    logger.info("%-22s %8.3f s  %8.1f MB  %s", name, best, peak / 2**20, info or "")
    return row, result


def _bench_operator(shape: int, n_angles: int, rows: list[dict[str, Any]]) -> RadonOperator:
    geom = Geometry((shape, shape), n_angles)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((shape, shape)).astype(np.float32)
    op_free = RadonOperator(geom, backend="pixel", matrix_budget_mb=0)
    r, _ = measure("matrix_build", lambda: RadonOperator(geom, backend="pixel"), shape=shape, n_angles=n_angles)
    rows.append(r)
    op = RadonOperator(geom, backend="pixel")
    s = op.forward(x)
    for label, o in (("matrix_free", op_free), ("cached_matrix", op)):
        rows.append(measure("forward", lambda o=o: o.forward(x), repeat=3, variant=label, shape=shape)[0])
        rows.append(measure("adjoint", lambda o=o: o.adjoint(s), repeat=3, variant=label, shape=shape)[0])
    xb = np.repeat(x[None], 20, axis=0)
    rows.append(measure("forward_batch20", lambda: op.forward(xb), variant="cached_matrix", shape=shape)[0])
    return op


def _bench_detail(op: RadonOperator, rows: list[dict[str, Any]], n_noise: int) -> None:
    n = op.image_shape[0]
    x = synthetic_volume(1, n, seed=1)[0]
    sigma = 1e-3
    noise = (sigma * np.random.default_rng(1).standard_normal(op.sino_shape)).astype(np.float32)
    y = op.forward(x) + noise
    dec = SART(op, n_iter=100, nonneg=False)
    z = select_point(dec.reconstruct(y))
    eps = max(2 * float(np.linalg.norm(noise)), 1.2 * float(np.linalg.norm(op.forward(z) - y)))
    region = disk_mask(op.image_shape, (n // 2, n // 2), n // 10)
    source = synthetic_volume(1, n, seed=7)[0]
    assessor = DetailAssessor(op, dec, AssessmentConfig(eps=eps, n_noise=n_noise, noise_sigma=sigma,
                                                        visible_fraction=-1.0), realism=AlwaysRealistic())
    row, res = measure("detail_check", lambda: assessor.assess(y, source, region, z=z), n_noise=n_noise, shape=n)
    row["outcome"] = res.outcome.value
    rows.append(row)


def _slices_for(n_patches: int, size: int, patch: int, stride: int) -> int:
    return int(np.ceil(n_patches / len(patch_positions((size, size), patch, stride))))


def _bench_b1(op: RadonOperator, sizes: list[int], rows: list[dict[str, Any]], patch: int, stride: int,
              n_queries: int) -> None:
    size = op.image_shape[0]
    pool = SinogramPooling(4, 4)
    for n in sizes:
        n_sl = _slices_for(n, size, patch, stride)
        vol = synthetic_volume(n_sl, size, seed=3)
        row, archive = measure("b1_build", lambda vol=vol: build_patch_archive(vol, op, patch, stride, pooling=pool),
                               n_patches=n)
        rows.append(row)
        ys = patch_measurement(op, archive, np.arange(0, len(archive), max(1, len(archive) // n_queries)))
        eps = 0.05 * float(np.median(np.linalg.norm(ys.reshape(len(ys), -1), axis=1)))
        est = FeasibleSetEstimator(op, archive)
        row, info = measure("b1_feasible", lambda est=est, ys=ys, eps=eps: est.feasible_matrix(ys, eps),
                            n_patches=len(archive), n_queries=len(ys))
        row["n_candidates"], row["n_feasible"] = info.n_candidates, int(info.fa.nnz)
        rows.append(row)
        rows.append(measure("b1_diameters", lambda archive=archive, info=info: diameters(archive, info),
                            n_patches=len(archive))[0])
        if n == min(sizes):
            def brute(archive=archive, ys=ys, eps=eps) -> int:
                hits = 0
                for start in range(0, len(archive), 256):
                    m = patch_measurement(op, archive, np.arange(start, min(start + 256, len(archive))))
                    d = np.linalg.norm((m[:, None] - ys[None]).reshape(len(m), len(ys), -1), axis=2)
                    hits += int((d < eps).sum())
                return hits

            row, hits = measure("b1_bruteforce", brute, n_patches=len(archive), n_queries=len(ys))
            row["n_feasible"] = hits
            rows.append(row)


def _bench_b2(op: RadonOperator, rows: list[dict[str, Any]], n_slices: int, patch: int, stride: int) -> None:
    size = op.image_shape[0]
    vol = synthetic_volume(n_slices, size, seed=5)
    dec = FBP(op)
    row, index = measure("b2_index", lambda: ArtifactIndex.build(vol, op, dec, 1e-3, patch_size=patch, stride=stride),
                         n_slices=n_slices)
    row["n_patches"] = len(index)
    rows.append(row)
    q = np.asarray(index.artifacts[: min(256, len(index))])
    rows.append(measure("b2_query", lambda: index.query(q, k=10), n_queries=len(q))[0])


def run(cfg: dict[str, Any], out_dir: str | Path, quick: bool = False) -> list[dict[str, Any]]:
    b = cfg["benchmark"]
    shape = int(b.get("shape", 128))
    n_angles = int(b.get("n_angles", 60))
    sizes = [int(v) for v in (b["sizes"] if not quick else [1000])]
    patch, stride = int(b.get("patch_size", 16)), int(b.get("stride", 8))
    rows: list[dict[str, Any]] = []
    op = _bench_operator(shape, n_angles, rows)
    _bench_detail(op, rows, n_noise=int(b.get("n_noise", 20)))
    _bench_b1(op, sizes, rows, patch, stride, n_queries=int(b.get("n_queries", 100)))
    _bench_b2(op, rows, n_slices=int(b.get("b2_slices", 20 if not quick else 4)), patch=patch * 2, stride=stride * 2)
    out = Path(out_dir)
    write_json(out / "benchmark.json", rows)
    write_csv(out / "benchmark.csv", [{k: r.get(k) for k in _columns(rows)} for r in rows])
    return rows


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    return cols
