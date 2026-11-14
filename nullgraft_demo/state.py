"""Состояние сессии демо: параметры, маска, детали, результаты; сохранение в JSON + .npz."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SESSION_VERSION = 1


@dataclass(frozen=True)
class ReconParams:
    """Параметры измерения и реконструкции."""

    n_angles: int = 60
    beam: str = "parallel"
    sigma: float = 1e-3
    eps: float = 0.0
    eps_factor: float = 1.5
    decoder: str = "cgls"
    decoder_iters: int = 250
    n_noise: int = 20
    seed: int = 0
    weights_path: str = ""

    def with_(self, **changes: Any) -> ReconParams:
        return replace(self, **changes)


@dataclass(frozen=True)
class DetailParams:
    """Источник детали и параметры блендинга для Алгоритма A."""

    source: str = "slice"
    source_slice: int = -1
    mode: str = "remove"
    file_path: str = ""
    window: str = "cosine"
    window_width: float = 3.0
    projection: str = "local"


@dataclass
class SessionState:
    """Всё, что нужно, чтобы восстановить сеанс: источник, срез, параметры, маска, результаты."""

    source_kind: str = "synthetic"
    data_path: str = ""
    slice_index: int = 0
    params: ReconParams = field(default_factory=ReconParams)
    detail: DetailParams = field(default_factory=DetailParams)
    window_level: tuple[float, float] = (0.5, 1.0)
    mask: NDArray[np.bool_] | None = None
    arrays: dict[str, NDArray] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Пишет ``<path>.json`` (параметры, результаты) и ``<path>.npz`` (маска, массивы)."""
        base = Path(path).with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": SESSION_VERSION,
            "source_kind": self.source_kind,
            "data_path": self.data_path,
            "slice_index": self.slice_index,
            "params": asdict(self.params),
            "detail": asdict(self.detail),
            "window_level": list(self.window_level),
            "results": _jsonable(self.results),
            "arrays": sorted(self.arrays),
        }
        json_path, npz_path = base.with_suffix(".json"), base.with_suffix(".npz")
        json_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        payload = {f"arr_{k}": np.asarray(v) for k, v in self.arrays.items()}
        if self.mask is not None:
            payload["mask"] = np.packbits(self.mask)
            payload["mask_shape"] = np.asarray(self.mask.shape)
        np.savez_compressed(npz_path, **payload)
        return json_path, npz_path

    @classmethod
    def load(cls, path: str | Path) -> SessionState:
        base = Path(path).with_suffix("")
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("version") != SESSION_VERSION:
            raise ValueError(f"unsupported session version {meta.get('version')}")
        state = cls(
            source_kind=meta["source_kind"],
            data_path=meta["data_path"],
            slice_index=int(meta["slice_index"]),
            params=_from_dict(ReconParams, meta["params"]),
            detail=_from_dict(DetailParams, meta["detail"]),
            window_level=tuple(meta["window_level"]),  # type: ignore[arg-type]
            results=meta.get("results", {}),
        )
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as data:
            for name in meta.get("arrays", []):
                state.arrays[name] = data[f"arr_{name}"]
            if "mask" in data:
                shape = tuple(int(v) for v in data["mask_shape"])
                bits = np.unpackbits(data["mask"], count=shape[0] * shape[1])
                state.mask = bits.reshape(shape).astype(bool)
        return state


def _from_dict(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in known})


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
