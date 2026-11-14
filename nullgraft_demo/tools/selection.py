"""Выбор области мышью на панелях matplotlib: прямоугольник, эллипс, лассо, перемещение."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import EllipseSelector, LassoSelector, RectangleSelector

from nullgraft_demo.tools.mask import MaskEditor, Mode, ellipse_region, polygon_region, rect_region

Tool = Literal["rect", "ellipse", "lasso", "move", "none"]
TOOLS: tuple[Tool, ...] = ("rect", "ellipse", "lasso", "move", "none")
_PROPS = {"edgecolor": "yellow", "facecolor": "none", "linestyle": "--"}


class SelectionController:
    """Связывает селекторы matplotlib с ``MaskEditor``; Shift при выделении — режим «вычесть»."""

    def __init__(self, axes: Sequence[Axes], editor: MaskEditor, request_draw: Callable[[], None]) -> None:
        self.axes = list(axes)
        self.editor = editor
        self.request_draw = request_draw
        self.tool: Tool = "rect"
        self.mode: Mode = "add"
        self._move_start: tuple[float, float] | None = None
        self._selectors: dict[Tool, list] = {"rect": [], "ellipse": [], "lasso": []}
        for ax in self.axes:
            self._selectors["rect"].append(RectangleSelector(
                ax, self._on_rect, useblit=True, button=[1], minspanx=0.5, minspany=0.5,
                spancoords="data", interactive=False, props=_PROPS))
            self._selectors["ellipse"].append(EllipseSelector(
                ax, self._on_ellipse, useblit=True, button=[1], minspanx=0.5, minspany=0.5,
                spancoords="data", interactive=False, props=_PROPS))
            self._selectors["lasso"].append(LassoSelector(
                ax, self._on_lasso, useblit=True, button=[1], props={"color": "yellow", "linestyle": "--"}))
        canvas = self.axes[0].figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        self.set_tool("rect")

    def set_editor(self, editor: MaskEditor) -> None:
        self.editor = editor

    def set_tool(self, tool: Tool) -> None:
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        self.tool = tool
        for name, sels in self._selectors.items():
            for s in sels:
                s.set_active(name == tool)

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode

    def _mode_for(self, key: str | None) -> Mode:
        return "subtract" if key and "shift" in key else self.mode

    def select_rect(self, x0: float, x1: float, y0: float, y1: float, mode: Mode | None = None) -> None:
        self.editor.apply(rect_region(self.editor.shape, x0, x1, y0, y1), mode or self.mode)
        self.request_draw()

    def select_ellipse(self, x0: float, x1: float, y0: float, y1: float, mode: Mode | None = None) -> None:
        self.editor.apply(ellipse_region(self.editor.shape, x0, x1, y0, y1), mode or self.mode)
        self.request_draw()

    def select_polygon(self, verts: Sequence[tuple[float, float]], mode: Mode | None = None) -> None:
        self.editor.apply(polygon_region(self.editor.shape, verts), mode or self.mode)
        self.request_draw()

    def _on_rect(self, eclick: MouseEvent, erelease: MouseEvent) -> None:
        self.select_rect(eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata, self._mode_for(eclick.key))

    def _on_ellipse(self, eclick: MouseEvent, erelease: MouseEvent) -> None:
        self.select_ellipse(eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata, self._mode_for(eclick.key))

    def _on_lasso(self, verts: Sequence[tuple[float, float]]) -> None:
        self.select_polygon(verts)

    def _on_press(self, event: MouseEvent) -> None:
        if self.tool != "move" or event.button != 1 or event.inaxes not in self.axes or event.xdata is None:
            return
        r, c = int(round(event.ydata)), int(round(event.xdata))
        m = self.editor.mask
        if 0 <= r < m.shape[0] and 0 <= c < m.shape[1] and m[r, c]:
            self._move_start = (event.ydata, event.xdata)
            self.editor.begin_move()

    def _on_motion(self, event: MouseEvent) -> None:
        if self._move_start is None or event.xdata is None or event.inaxes not in self.axes:
            return
        dy = int(round(event.ydata - self._move_start[0]))
        dx = int(round(event.xdata - self._move_start[1]))
        self.editor.preview_move(dy, dx)
        self.request_draw()

    def _on_release(self, event: MouseEvent) -> None:
        if self._move_start is not None:
            self._move_start = None
            self.editor.end_move()
            self.request_draw()
