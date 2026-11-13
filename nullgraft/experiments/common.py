"""Общие функции экспериментов: файлы результатов, возобновляемый JSONL-лог, выбор области."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, uniform_filter

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(name)s %(levelname)s %(message)s")


class JsonlLog:
    """JSON-lines файл с дозаписью; множество ``done`` позволяет продолжить после остановки."""

    def __init__(self, path: str | Path, key: str) -> None:
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[Any] = set()
        if self.path.exists():
            for rec in self.records():
                self.done.add(rec[key])
            logger.info("resuming %s: %d records already done", self.path.name, len(self.done))

    def append(self, record: Mapping[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(record)) + "\n")
        self.done.add(record[self.key])

    def records(self) -> list[dict[str, Any]]:
        """Все целые записи; обрезанная последняя строка (сбой при записи) пропускается."""
        if not self.path.exists():
            return []
        out = []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                logger.warning("%s: ignoring truncated last record", self.path.name)
        return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(obj), indent=1), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(_jsonable(r) for r in rows)


def disk_mask(shape: tuple[int, int], center: tuple[int, int], radius: float) -> NDArray[np.bool_]:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def most_suspicious_center(z: NDArray, radius: int, sigma: float = 2.0) -> tuple[int, int]:
    """Центр окна с наибольшей high-pass энергией ``z`` — автоматический выбор области интереса."""
    hp = (z - gaussian_filter(z, sigma)) ** 2
    energy = uniform_filter(hp, size=2 * radius + 1)
    m = radius + 1
    energy[:m], energy[-m:], energy[:, :m], energy[:, -m:] = 0, 0, 0, 0
    r, c = np.unravel_index(int(np.argmax(energy)), energy.shape)
    return int(r), int(c)


def maybe_plot(fn_name: str, out: Path, **kw: Any) -> None:
    """График через matplotlib, если он установлен."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed: skipping plot %s", fn_name)
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if fn_name == "convergence":
        for col in kw["curve"].T[: kw.get("max_lines", 10)]:
            ax.plot(kw["fractions"] * 100, col, marker="o", ms=3)
        ax.set_xlabel("patches used, %")
        ax.set_ylabel("diam(F^N_y) (lower bound)")
        ax.set_title("Feasible-set diameter vs N (converges from below)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
