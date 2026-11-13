"""Загрузчики NLM Visible Human Project; данные скачивает пользователь, геометрия читается из файлов.

Формат папок и порядок работы: docs/08_data.md.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_IMAGE_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class CtSlice:
    path: Path
    position_mm: float
    instance: int
    pixel_spacing_mm: tuple[float, float]
    shape: tuple[int, int]
    rescale: tuple[float, float]


@dataclass(frozen=True)
class CtSeries:
    slices: tuple[CtSlice, ...]

    @property
    def slice_spacing_mm(self) -> float:
        z = np.array([s.position_mm for s in self.slices])
        return float(np.median(np.abs(np.diff(z)))) if z.size > 1 else float("nan")

    def report(self) -> dict[str, object]:
        """Метаданные, реально найденные в файлах (проверить до использования данных)."""
        z = np.array([s.position_mm for s in self.slices])
        spacings = sorted({s.pixel_spacing_mm for s in self.slices})
        gaps = np.abs(np.diff(z)) if z.size > 1 else np.array([])
        inst = np.array([s.instance for s in self.slices])
        return {
            "n_slices": len(self.slices),
            "shapes": sorted({s.shape for s in self.slices}),
            "pixel_spacing_mm": spacings,
            "slice_spacing_mm_median": self.slice_spacing_mm,
            "slice_spacing_mm_range": (float(gaps.min()), float(gaps.max())) if gaps.size else None,
            "instance_numbers_monotone": bool(np.all(np.diff(inst) > 0)) if inst.size > 1 else True,
            "z_range_mm": (float(z.min()), float(z.max())) if z.size else None,
        }


def _natural_key(path: Path) -> list[object]:
    """Естественная сортировка: ``slice2`` раньше ``slice10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def scan_ct_series(folder: str | Path) -> CtSeries:
    """Индексирует папку КТ; срезы сортируются по z (DICOM) или по имени файла (изображения)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"CT folder not found: {folder}")
    files = sorted((p for p in folder.iterdir() if p.is_file()), key=_natural_key)
    dicoms = [p for p in files if p.suffix.lower() == ".dcm" or (not p.suffix and _is_dicom(p))]
    if dicoms:
        return _scan_dicom(dicoms)
    images = [p for p in files if p.suffix.lower() in _IMAGE_EXT + (".npy",)]
    if not images:
        raise ValueError(f"no DICOM or image slices in {folder}")
    meta_file = folder / "ct_meta.json"
    if not meta_file.exists():
        raise ValueError("image-based CT folders need ct_meta.json with pixel_spacing_mm and slice_spacing_mm")
    meta = json.loads(meta_file.read_text())
    dy, dx = meta["pixel_spacing_mm"]
    dz = float(meta["slice_spacing_mm"])
    slope, intercept = meta.get("rescale", [1.0, 0.0])
    out = []
    for i, p in enumerate(images):
        shape = _read_image(p).shape[:2]
        out.append(CtSlice(p, i * dz, i, (float(dy), float(dx)), (int(shape[0]), int(shape[1])),
                           (float(slope), float(intercept))))
    return CtSeries(tuple(out))


def _scan_dicom(paths: Sequence[Path]) -> CtSeries:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading DICOM requires pydicom (pip install nullgraft[dicom])") from exc
    out = []
    for p in paths:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        pos = getattr(ds, "ImagePositionPatient", None)
        z = float(pos[2]) if pos is not None else float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0)))
        spacing = tuple(float(v) for v in getattr(ds, "PixelSpacing", (np.nan, np.nan)))
        out.append(CtSlice(p, z, int(getattr(ds, "InstanceNumber", 0)), spacing,  # type: ignore[arg-type]
                           (int(ds.Rows), int(ds.Columns)),
                           (float(getattr(ds, "RescaleSlope", 1.0)), float(getattr(ds, "RescaleIntercept", 0.0)))))
    out.sort(key=lambda s: s.position_mm)
    return CtSeries(tuple(out))


def _read_image(path: Path) -> NDArray:
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r")
    from skimage.io import imread

    return imread(path)


def read_ct_slice(s: CtSlice) -> NDArray[np.float32]:
    """Срез в HU (с применённым DICOM rescale)."""
    if s.path.suffix.lower() in _IMAGE_EXT + (".npy",):
        raw = np.asarray(_read_image(s.path), np.float32)
    else:
        import pydicom

        raw = pydicom.dcmread(s.path).pixel_array.astype(np.float32)
    slope, intercept = s.rescale
    return raw * slope + intercept


def iter_ct(series: CtSeries, indices: Sequence[int] | None = None) -> Iterator[NDArray[np.float32]]:
    """Ленивый итератор по срезам КТ (в памяти один срез)."""
    for i in indices if indices is not None else range(len(series.slices)):
        yield read_ct_slice(series.slices[i])


def hu_to_mu(hu: NDArray, mu_water: float = 0.0192) -> NDArray[np.float32]:
    """HU в линейное ослабление (1/мм, ~70 кэВ): ``μ = μ_w (1 + HU/1000)``, обрезка снизу нулём."""
    return np.clip(mu_water * (1.0 + np.asarray(hu, np.float32) / 1000.0), 0.0, None).astype(np.float32)


def cryo_to_gray(rgb: NDArray) -> NDArray[np.float32]:
    """Яркость в [0, 1] цветной фотографии криосреза."""
    arr = np.asarray(rgb)
    # масштаб по диапазону целого типа, а не по содержимому: тёмный фрагмент должен остаться тёмным
    scale = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0
    arr = arr.astype(np.float32) / scale
    if arr.ndim == 2:
        return arr
    rgb = arr[..., :3]
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def cryo_to_ct_grid(
    gray: NDArray, cryo_pixel_mm: float, ct_pixel_mm: tuple[float, float], ct_shape: tuple[int, int]
) -> NDArray[np.float32]:
    """Пересэмплирование криосреза на сетку КТ с антиалиасингом (совпадение мм/пиксель), центрирование."""
    from skimage.transform import rescale

    factors = (cryo_pixel_mm / ct_pixel_mm[0], cryo_pixel_mm / ct_pixel_mm[1])
    small = rescale(np.asarray(gray, np.float32), factors, anti_aliasing=True, preserve_range=True)
    out = np.zeros(ct_shape, np.float32)
    h, w = small.shape
    oy, ox = (ct_shape[0] - h) // 2, (ct_shape[1] - w) // 2
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    hh, ww = min(h - sy0, ct_shape[0] - dy0), min(w - sx0, ct_shape[1] - dx0)
    out[dy0 : dy0 + hh, dx0 : dx0 + ww] = small[sy0 : sy0 + hh, sx0 : sx0 + ww]
    return out


def list_cryo(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"cryosection folder not found: {folder}")
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXT), key=_natural_key)


def load_registered_pairs(folder: str | Path) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    """Выход ``scripts/register_vhp.py``: ``ct.npy``, ``cryo.npy`` (memmap) и ``registration_log.json``."""
    folder = Path(folder)
    ct = np.load(folder / "ct.npy", mmap_mode="r")
    cryo = np.load(folder / "cryo.npy", mmap_mode="r")
    log = json.loads((folder / "registration_log.json").read_text())
    return ct, cryo, log
