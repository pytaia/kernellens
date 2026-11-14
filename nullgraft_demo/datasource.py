"""Источники срезов: синтетика, папка Visible Human (выход register_vhp.py), отдельный файл."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from nullgraft.data.synthetic import shepp_logan, synthetic_volume

logger = logging.getLogger(__name__)


class DataSource(Protocol):
    name: str
    kind: str
    path: str
    pixel_mm: float
    has_truth: bool
    units: str

    @property
    def n_slices(self) -> int: ...

    def get(self, index: int) -> NDArray[np.float32]: ...


@dataclass
class ArraySource:
    """Срезы из массива ``(N, H, W)`` (в памяти или memmap)."""

    name: str
    kind: str
    path: str
    volume: NDArray
    pixel_mm: float = 1.0
    has_truth: bool = True
    units: str = "arb"

    @property
    def n_slices(self) -> int:
        return int(self.volume.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.volume.shape[1]), int(self.volume.shape[2])

    def get(self, index: int) -> NDArray[np.float32]:
        if not 0 <= index < self.n_slices:
            raise IndexError(f"slice {index} out of range 0..{self.n_slices - 1}")
        return np.asarray(self.volume[index], np.float32)


def synthetic_source(size: int = 128, n_slices: int = 24, pixel_mm: float = 1.0, seed: int = 0) -> ArraySource:
    """Срез 0 — фантом Шеппа–Логана, остальные — «объём» с дрейфующими эллипсами."""
    vol = np.concatenate([shepp_logan(size)[None], synthetic_volume(max(n_slices - 1, 1), size, seed=seed)])
    return ArraySource("Синтетика", "synthetic", "", vol, pixel_mm=pixel_mm)


def vhp_source(folder: str | Path) -> ArraySource:
    """Пары из ``scripts/register_vhp.py``: эталон x — зарегистрированный криосрез."""
    folder = Path(folder)
    cryo = np.load(folder / "cryo.npy", mmap_mode="r")
    pixel_mm = 1.0
    log_path = folder / "registration_log.json"
    if log_path.exists():
        report = json.loads(log_path.read_text(encoding="utf-8")).get("ct_report", {})
        spacings = report.get("pixel_spacing_mm") or []
        if spacings:
            pixel_mm = float(spacings[0][0])
    return ArraySource("Visible Human", "vhp", str(folder), cryo, pixel_mm=pixel_mm)


def file_source(path: str | Path, pixel_mm: float = 1.0) -> ArraySource:
    """Один срез или стопка из PNG/TIFF/JPEG, ``.npy`` или DICOM; эталона нет."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
    elif suffix in (".dcm", "") :
        import pydicom

        ds = pydicom.dcmread(path)
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) + float(
            getattr(ds, "RescaleIntercept", 0.0))
        arr = np.clip(0.0192 * (1 + hu / 1000.0), 0, None)
        spacing = getattr(ds, "PixelSpacing", None)
        pixel_mm = float(spacing[0]) if spacing is not None else pixel_mm
    else:
        from skimage.color import rgb2gray
        from skimage.io import imread

        img = imread(path)
        arr = rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(np.float32) / max(float(img.max()), 1.0)
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    arr = _square(arr)
    return ArraySource(path.name, "file", str(path), arr, pixel_mm=pixel_mm, has_truth=False)


def _square(vol: NDArray[np.float32]) -> NDArray[np.float32]:
    """Дополнение нулями до квадрата: оператор работает на квадратной сетке."""
    n, h, w = vol.shape
    if h == w:
        return vol
    s = max(h, w)
    out = np.zeros((n, s, s), np.float32)
    out[:, (s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = vol
    return out


def open_source(data_path: str | Path | None, cfg: dict) -> ArraySource:
    """Синтетика без пути; папка с ``cryo.npy`` — Visible Human; файл — отдельный срез."""
    if not data_path:
        syn = cfg.get("synthetic", {})
        return synthetic_source(int(syn.get("size", 128)), int(syn.get("n_slices", 24)), float(cfg.get("pixel_mm", 1)))
    path = Path(data_path)
    if path.is_dir():
        if (path / "cryo.npy").exists():
            return vhp_source(path)
        raise FileNotFoundError(f"{path}: expected cryo.npy from scripts/register_vhp.py")
    if path.is_file():
        return file_source(path, float(cfg.get("pixel_mm", 1)))
    raise FileNotFoundError(f"data path not found: {path}")
