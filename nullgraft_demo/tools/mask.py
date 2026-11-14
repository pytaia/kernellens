"""Маска области R: растеризация фигур, булевы операции, перемещение, undo/redo. Без GUI."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from skimage.draw import polygon2mask

Mode = Literal["add", "subtract", "replace"]
Shape = tuple[int, int]


def rect_region(shape: Shape, x0: float, x1: float, y0: float, y1: float) -> NDArray[np.bool_]:
    """Пиксели, центры которых лежат в прямоугольнике (x — столбец, y — строка)."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return (cols >= xa) & (cols <= xb) & (rows >= ya) & (rows <= yb)


def ellipse_region(shape: Shape, x0: float, x1: float, y0: float, y1: float) -> NDArray[np.bool_]:
    """Эллипс, вписанный в прямоугольник ``[x0, x1] × [y0, y1]``."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = max(abs(x1 - x0) / 2, 0.5), max(abs(y1 - y0) / 2, 0.5)
    return ((cols - cx) / ax) ** 2 + ((rows - cy) / ay) ** 2 <= 1.0


def disk_region(shape: Shape, row: float, col: float, radius: float) -> NDArray[np.bool_]:
    return ellipse_region(shape, col - radius, col + radius, row - radius, row + radius)


def polygon_region(shape: Shape, verts_xy: Sequence[tuple[float, float]]) -> NDArray[np.bool_]:
    """Многоугольник (лассо) по вершинам ``(x, y)``."""
    if len(verts_xy) < 3:
        return np.zeros(shape, bool)
    rc = np.array([(y, x) for x, y in verts_xy], dtype=np.float64)
    return polygon2mask(shape, rc)


def shift_mask(mask: NDArray[np.bool_], dy: int, dx: int) -> NDArray[np.bool_]:
    """Сдвиг маски с заполнением нулями (то, что ушло за край, теряется)."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    src_r = slice(max(0, -dy), min(h, h - dy))
    src_c = slice(max(0, -dx), min(w, w - dx))
    dst_r = slice(max(0, dy), min(h, h + dy))
    dst_c = slice(max(0, dx), min(w, w + dx))
    if src_r.start < src_r.stop and src_c.start < src_c.stop:
        out[dst_r, dst_c] = mask[src_r, src_c]
    return out


class MaskEditor:
    """Булева маска R с историей правок."""

    def __init__(self, shape: Shape, history: int = 50) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.history = history
        self._mask = np.zeros(self.shape, bool)
        self._undo: list[bytes] = []
        self._redo: list[bytes] = []
        self._move_base: NDArray[np.bool_] | None = None
        self._listeners: list[Callable[[NDArray[np.bool_]], None]] = []

    @property
    def mask(self) -> NDArray[np.bool_]:
        view = self._mask.view()
        view.flags.writeable = False
        return view

    @property
    def empty(self) -> bool:
        return not self._mask.any()

    @property
    def area_px(self) -> int:
        return int(self._mask.sum())

    def area_mm2(self, pixel_mm: float) -> float:
        return self.area_px * pixel_mm**2

    def equivalent_diameter_px(self) -> float:
        return 2.0 * math.sqrt(self.area_px / math.pi)

    def centroid(self) -> tuple[float, float] | None:
        if self.empty:
            return None
        rows, cols = np.nonzero(self._mask)
        return float(rows.mean()), float(cols.mean())

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def subscribe(self, fn: Callable[[NDArray[np.bool_]], None]) -> None:
        self._listeners.append(fn)

    def _pack(self, m: NDArray[np.bool_]) -> bytes:
        return np.packbits(m).tobytes()

    def _unpack(self, b: bytes) -> NDArray[np.bool_]:
        bits = np.unpackbits(np.frombuffer(b, np.uint8), count=self.shape[0] * self.shape[1])
        return bits.reshape(self.shape).astype(bool)

    def _set(self, new: NDArray[np.bool_], record: bool = True) -> None:
        if np.array_equal(new, self._mask):
            return
        if record:
            self._undo.append(self._pack(self._mask))
            del self._undo[: -self.history]
            self._redo.clear()
        self._mask = new
        for fn in self._listeners:
            fn(self.mask)

    def apply(self, region: NDArray[np.bool_], mode: Mode = "add") -> None:
        region = np.asarray(region, bool)
        if region.shape != self.shape:
            raise ValueError(f"region shape {region.shape} != mask shape {self.shape}")
        if mode == "add":
            new = self._mask | region
        elif mode == "subtract":
            new = self._mask & ~region
        elif mode == "replace":
            new = region.copy()
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self._set(new)

    def set_mask(self, mask: NDArray[np.bool_]) -> None:
        self.apply(mask, "replace")

    def clear(self) -> None:
        self._set(np.zeros(self.shape, bool))

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._pack(self._mask))
        self._set(self._unpack(self._undo.pop()), record=False)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._pack(self._mask))
        self._set(self._unpack(self._redo.pop()), record=False)
        return True

    def begin_move(self) -> None:
        self._move_base = self._mask.copy()

    def preview_move(self, dy: int, dx: int) -> None:
        """Промежуточный сдвиг во время перетаскивания: в историю не попадает."""
        if self._move_base is None:
            return
        self._set(shift_mask(self._move_base, dy, dx), record=False)

    def end_move(self) -> None:
        """Фиксирует перемещение одной записью в истории."""
        if self._move_base is None:
            return
        base, self._move_base = self._move_base, None
        if not np.array_equal(base, self._mask):
            self._undo.append(self._pack(base))
            del self._undo[: -self.history]
            self._redo.clear()
