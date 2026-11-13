"""Командная строка: ``nullgraft {assess,feasible,similar,benchmark,report} ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nullgraft.config import load_config
from nullgraft.experiments.common import setup_logging


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nullgraft", description="Provable hallucination assessment for CT")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("assess", "Algorithm A: decoder-dependent detail test"),
                        ("feasible", "Algorithm B1: feasible-set diameters on patches"),
                        ("similar", "Algorithm B2: similar-artifact retrieval"),
                        ("benchmark", "time / memory benchmarks")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", type=Path, help="YAML config (defaults if omitted)")
        p.add_argument("--out", type=Path, required=True, help="output folder")
        if name != "benchmark":
            p.add_argument("--data", type=Path, help="dataset path (never downloaded)")
        else:
            p.add_argument("--quick", action="store_true", help="only N=1e3 for B1")
    rep = sub.add_parser("report", help="print CT series metadata found in a DICOM/PNG folder")
    rep.add_argument("folder", type=Path)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging(args.log_level)
    if args.cmd == "report":
        from nullgraft.data.vhp import scan_ct_series

        print(json.dumps(scan_ct_series(args.folder).report(), indent=1, default=str))
        return 0
    cfg = load_config(args.config)
    if args.cmd == "benchmark":
        from nullgraft.experiments.benchmark import run

        run(cfg, args.out, quick=args.quick)
        return 0
    module = {"assess": "run_assess", "feasible": "run_feasible", "similar": "run_similar"}[args.cmd]
    runner = __import__(f"nullgraft.experiments.{module}", fromlist=["run"]).run
    result = runner(cfg, args.out, data_path=args.data)
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
