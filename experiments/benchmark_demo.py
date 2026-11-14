"""Бенчмарк демо: время от «Запуск» до первого результата и до финала проверки детали (без GUI).

python experiments/benchmark_demo.py --sizes 128 256 512 --out bench_demo.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np

from nullgraft.data.synthetic import shepp_logan
from nullgraft_demo.compute import get_operator, reconstruct
from nullgraft_demo.detail import detail_source, run_check
from nullgraft_demo.state import DetailParams, ReconParams
from nullgraft_demo.tools import disk_region
from nullgraft_demo.workers import CancelToken


def bench(size: int, params: ReconParams, detail: DetailParams) -> dict:
    x = shepp_logan(size)
    t0 = time.perf_counter()
    get_operator(x.shape, params.n_angles, params.beam)
    t_op = time.perf_counter() - t0
    t0 = time.perf_counter()
    recon = reconstruct(x, params, None, token=CancelToken(), progress=lambda f, m: None)
    t_recon = time.perf_counter() - t0
    mask = disk_region(x.shape, size * 0.58, size * 0.5, max(4, size // 16))
    source = detail_source(detail, recon.z, mask)
    res = run_check(recon, source, mask, params, detail, token=CancelToken(), progress=lambda f, m: None)
    return {
        "size": size, "n_angles": params.n_angles, "decoder": params.decoder, "decoder_iters": params.decoder_iters,
        "M": params.n_noise, "projection": detail.projection, "operator_build_s": t_op, "reconstruction_s": t_recon,
        "check_first_result_s": res.first_stage_seconds, "check_total_s": res.seconds,
        "outcome": res.assessment.outcome.value, "consistent": recon.consistent,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--angles", type=int, default=60)
    ap.add_argument("--decoder", default="cgls")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("bench_demo.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    params = ReconParams(n_angles=args.angles, decoder=args.decoder, decoder_iters=args.iters, n_noise=args.M)
    detail = DetailParams(mode="remove", projection="local")
    rows = []
    for size in args.sizes:
        row = bench(size, params, detail)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))  # noqa: T201
    args.out.write_text(json.dumps({"machine": platform.processor() or platform.machine(),
                                    "numpy": np.__version__, "results": rows}, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
