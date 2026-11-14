"""Конфиг демо: значения по умолчанию из ``default.yaml`` поверх пользовательского YAML."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from nullgraft.config import deep_merge
from nullgraft_demo.state import ReconParams


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default = yaml.safe_load(resources.files("nullgraft_demo").joinpath("default.yaml").read_text(encoding="utf-8"))
    if path is None:
        return default
    with open(path, encoding="utf-8") as fh:
        return deep_merge(default, yaml.safe_load(fh) or {})


def cache_dir(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("cache_dir") or ""
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "nullgraft_demo"


def recon_params(cfg: dict[str, Any]) -> ReconParams:
    return ReconParams(**cfg["recon"])
