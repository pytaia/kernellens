"""Сетка изображений: эталон, реконструкция, остаток (общие оси) и синограмма, невязка (общие оси)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.image import AxesImage
from matplotlib.patches import Circle
from numpy.typing import NDArray

MASK_RGBA = (1.0, 0.85, 0.0)
MASK_ALPHA = 0.25


class ImageGrid:
    """Пять панелей; обновление только изменившихся артистов (``set_data``, ``set_clim``)."""

    def __init__(self, fig: Figure, spec: SubplotSpec | None = None) -> None:
        self.fig = fig
        gs = spec.subgridspec(2, 3) if spec is not None else fig.add_gridspec(2, 3)
        self.ax_x = fig.add_subplot(gs[0, 0])
        self.ax_z = fig.add_subplot(gs[0, 1], sharex=self.ax_x, sharey=self.ax_x)
        self.ax_r = fig.add_subplot(gs[0, 2], sharex=self.ax_x, sharey=self.ax_x)
        self.ax_y = fig.add_subplot(gs[1, 0])
        self.ax_d = fig.add_subplot(gs[1, 1], sharex=self.ax_y, sharey=self.ax_y)
        self.ax_info = fig.add_subplot(gs[1, 2])
        self.ax_info.axis("off")
        titles = {
            self.ax_x: "Эталон x", self.ax_z: "Реконструкция z = φ(y)", self.ax_r: "Остаток z − x",
            self.ax_y: "Синограмма y", self.ax_d: "Невязка |A z − y|",
        }
        for ax, title in titles.items():
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
        self._im: dict[Axes, AxesImage] = {}
        self._overlay: dict[Axes, AxesImage] = {}
        self._contours: list = []
        self._cand_patches: list[Circle] = []
        self._placeholder: dict[Axes, object] = {}
        self._info = self.ax_info.text(0.0, 1.0, "", va="top", ha="left", fontsize=8, family="monospace",
                                       transform=self.ax_info.transAxes)
        self.x: NDArray | None = None
        self.z: NDArray | None = None
        self.window = (0.5, 1.0)

    @property
    def image_axes(self) -> list[Axes]:
        return [self.ax_x, self.ax_z, self.ax_r]

    def _show(self, ax: Axes, data: NDArray, cmap: str = "gray", aspect: str = "equal",
              clim: tuple[float, float] | None = None) -> None:
        self._clear_placeholder(ax)
        if ax in self._im and self._im[ax].get_array().shape == data.shape:
            self._im[ax].set_data(data)
        else:
            if ax in self._im:
                self._im[ax].remove()
            self._im[ax] = ax.imshow(data, cmap=cmap, aspect=aspect, interpolation="nearest")
        if clim is not None:
            self._im[ax].set_clim(*clim)

    def _clear_placeholder(self, ax: Axes) -> None:
        if ax in self._placeholder:
            self._placeholder.pop(ax).remove()

    def _set_placeholder(self, ax: Axes, text: str) -> None:
        if ax in self._im:
            self._im.pop(ax).remove()
        self._clear_placeholder(ax)
        self._placeholder[ax] = ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, color="0.4",
                                        transform=ax.transAxes)

    def set_slice(self, x: NDArray | None, shape: tuple[int, int]) -> None:
        """Новый срез: эталон (или заглушка), сброс реконструкции."""
        self.x = None if x is None else np.asarray(x, np.float32)
        self.z = None
        if self.x is None:
            self._set_placeholder(self.ax_x, "эталона нет")
            self.ax_x.set_xlim(-0.5, shape[1] - 0.5)
            self.ax_x.set_ylim(shape[0] - 0.5, -0.5)
        else:
            self._show(self.ax_x, self.x, clim=self._clim())
        for ax in (self.ax_z, self.ax_r, self.ax_y, self.ax_d):
            self._set_placeholder(ax, "нажмите «Реконструировать»")

    def set_recon(self, y: NDArray, z: NDArray, det_residual: NDArray) -> None:
        self.z = np.asarray(z, np.float32)
        self._show(self.ax_z, self.z, clim=self._clim())
        if self.x is not None:
            r = self.z - self.x
            lim = float(np.max(np.abs(r))) or 1.0
            self._show(self.ax_r, r, cmap="RdBu_r", clim=(-lim, lim))
        else:
            self._set_placeholder(self.ax_r, "нужен эталон")
        self._show(self.ax_y, np.asarray(y), cmap="gray", aspect="auto")
        self._show(self.ax_d, np.asarray(det_residual), cmap="magma", aspect="auto")

    def _clim(self) -> tuple[float, float]:
        level, width = self.window
        return level - width / 2, level + width / 2

    def set_window(self, level: float, width: float) -> None:
        self.window = (float(level), max(float(width), 1e-6))
        for ax in (self.ax_x, self.ax_z):
            if ax in self._im:
                self._im[ax].set_clim(*self._clim())

    def set_mask(self, mask: NDArray[np.bool_] | None) -> None:
        """Полупрозрачная заливка и контур маски R на панелях изображения."""
        for c in self._contours:
            c.remove()
        self._contours = []
        for ax in self.image_axes:
            if mask is None or not mask.any():
                if ax in self._overlay:
                    self._overlay[ax].set_visible(False)
                continue
            rgba = np.zeros((*mask.shape, 4), np.float32)
            rgba[mask, :3] = MASK_RGBA
            rgba[mask, 3] = MASK_ALPHA
            if ax in self._overlay and self._overlay[ax].get_array().shape[:2] == mask.shape:
                self._overlay[ax].set_data(rgba)
                self._overlay[ax].set_visible(True)
            else:
                if ax in self._overlay:
                    self._overlay[ax].remove()
                self._overlay[ax] = ax.imshow(rgba, interpolation="nearest", zorder=3)
            self._contours.append(ax.contour(mask.astype(np.float32), levels=[0.5], colors=[MASK_RGBA],
                                             linewidths=1.0, zorder=4))

    def set_candidates(self, cands: Sequence[tuple[int, int, int, float]]) -> None:
        for p in self._cand_patches:
            p.remove()
        self._cand_patches = []
        for i, (r, c, rad, _) in enumerate(cands):
            for ax in self.image_axes:
                circ = Circle((c, r), rad, fill=False, ls="--", lw=1.2, ec="cyan", zorder=5, picker=True)
                circ.candidate_index = i  # type: ignore[attr-defined]
                ax.add_patch(circ)
                self._cand_patches.append(circ)

    def set_overlay(self, image: NDArray | None, cmap: str = "inferno", alpha: float = 0.45) -> None:
        """Полупрозрачное наложение (карта риска) на панель реконструкции; ``None`` — убрать."""
        old = getattr(self, "_risk", None)
        if old is not None:
            old.remove()
            self._risk = None
        if image is not None:
            self._risk = self.ax_z.imshow(np.ma.masked_invalid(image), cmap=cmap, alpha=alpha, vmin=0, vmax=1,
                                          interpolation="nearest", zorder=2)

    def set_info(self, text: str) -> None:
        self._info.set_text(text)

    def pixel_info(self, ax: Axes | None, xdata: float | None, ydata: float | None) -> str | None:
        """Координаты и значения пикселя под курсором (для строки состояния)."""
        if ax is None or xdata is None or ydata is None:
            return None
        col, row = int(round(xdata)), int(round(ydata))
        if ax in (self.ax_y, self.ax_d):
            im = self._im.get(ax)
            if im is None:
                return None
            arr = im.get_array()
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                name = "y" if ax is self.ax_y else "|Az−y|"
                return f"угол #{row}, бин {col}: {name} = {float(arr[row, col]):.4g}"
            return None
        ref = self.x if self.x is not None else self.z
        if ref is None or not (0 <= row < ref.shape[0] and 0 <= col < ref.shape[1]):
            return None
        parts = [f"строка {row}, столбец {col}"]
        if self.x is not None:
            parts.append(f"x = {self.x[row, col]:.4g}")
        if self.z is not None:
            parts.append(f"z = {self.z[row, col]:.4g}")
        return ", ".join(parts)
