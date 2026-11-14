"""Запасной интерфейс без Qt: matplotlib.widgets, фоновые задачи в потоке, результаты через таймер."""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Any

import numpy as np

from nullgraft.api import DECODER_NAMES
from nullgraft_demo.compute import ReconResult, reconstruct, suggest_candidates
from nullgraft_demo.datasource import ArraySource
from nullgraft_demo.state import ReconParams
from nullgraft_demo.tools import MaskEditor, disk_region
from nullgraft_demo.tools.selection import SelectionController
from nullgraft_demo.views import DrawThrottle, ImageGrid
from nullgraft_demo.workers import Job, JobRunner, ResultCache

logger = logging.getLogger(__name__)

N_ANGLES = ("8", "15", "30", "60", "120", "360")
TOOLS = ("rect", "ellipse", "lasso", "move")
TOOL_NAMES = {"rect": "прямоуг.", "ellipse": "эллипс", "lasso": "лассо", "move": "сдвиг"}


class MplApp:
    """Окно на чистом matplotlib: виджеты слева, та же сетка ``ImageGrid`` справа."""

    def __init__(self, fig: Any, cfg: dict[str, Any], source: ArraySource, cache: ResultCache,
                 runner: JobRunner | None, sync: bool = False) -> None:
        from matplotlib.widgets import Button, RadioButtons, Slider

        self.fig, self.cfg, self.source, self.cache, self.runner = fig, cfg, source, cache, runner
        self.sync = sync or runner is None
        self.params = ReconParams(**cfg["recon"])
        self.recon: ReconResult | None = None
        self.job: Job | None = None
        self.candidates: list[tuple[int, int, int, float]] = []
        self._results: queue.Queue = queue.Queue()

        outer = fig.add_gridspec(1, 2, width_ratios=[1, 3.6], left=0.02, right=0.99, top=0.95, bottom=0.05)
        self.grid = ImageGrid(fig, outer[0, 1])
        self.throttle = DrawThrottle(fig.canvas, int(cfg.get("redraw_ms", 30)))
        self.x = source.get(0)
        self.editor = MaskEditor(self.x.shape, int(cfg.get("history", 50)))
        self.editor.subscribe(lambda m: (self.grid.set_mask(m), self._update_status(), self.throttle.request()))
        self.selection = SelectionController(self.grid.image_axes, self.editor, self.throttle.request)

        def axes(top: float, height: float) -> Any:
            return fig.add_axes((0.07, top, 0.13, height))

        self.slice = Slider(axes(0.92, 0.025), "срез", 0, max(source.n_slices - 1, 1), valinit=0, valstep=1)
        self.slice.on_changed(lambda v: self.load_slice(int(v)))
        self.angles = RadioButtons(axes(0.80, 0.10), N_ANGLES, active=N_ANGLES.index(str(self.params.n_angles))
                                   if str(self.params.n_angles) in N_ANGLES else 1)
        self.decoder = RadioButtons(axes(0.62, 0.16), DECODER_NAMES,
                                    active=DECODER_NAMES.index(self.params.decoder)
                                    if self.params.decoder in DECODER_NAMES else 0)
        self.btn_recon = Button(axes(0.57, 0.04), "Реконструировать")
        self.btn_recon.on_clicked(lambda _e: self.start_reconstruction())
        self.tool = RadioButtons(axes(0.42, 0.13), [TOOL_NAMES[t] for t in TOOLS], active=0)
        self.tool.on_clicked(lambda label: self.selection.set_tool(
            next(t for t, n in TOOL_NAMES.items() if n == label)))
        self.mode = RadioButtons(axes(0.34, 0.07), ("добавить", "вычесть"), active=0)
        self.mode.on_clicked(lambda label: self.selection.set_mode("add" if label == "добавить" else "subtract"))
        self._buttons = []
        for i, (text, fn) in enumerate((("Отменить", self.editor.undo), ("Повторить", self.editor.redo),
                                        ("Очистить", self.editor.clear))):
            b = Button(fig.add_axes((0.07 + i * 0.044, 0.29, 0.042, 0.035)), text)
            b.on_clicked(lambda _e, f=fn: (f(), self.throttle.request()))
            self._buttons.append(b)
        self.btn_cand = Button(axes(0.24, 0.035), "Кандидаты")
        self.btn_cand.on_clicked(lambda _e: self.suggest())
        self.level = Slider(axes(0.18, 0.025), "уровень", -1.0, 2.0, valinit=0.5)
        self.width = Slider(axes(0.14, 0.025), "ширина", 0.01, 3.0, valinit=1.0)
        for s in (self.level, self.width):
            s.on_changed(lambda _v: (self.grid.set_window(self.level.val, self.width.val), self.throttle.request()))
        self.status = fig.text(0.02, 0.02, "", fontsize=8)
        fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        fig.canvas.mpl_connect("pick_event", self._on_pick)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        if not self.sync:
            self._poll = fig.canvas.new_timer(interval=100)
            self._poll.add_callback(self._drain)
            self._poll.start()
        self.load_slice(0)

    def load_slice(self, index: int) -> None:
        self.x = self.source.get(index)
        if self.editor.shape != self.x.shape:
            self.editor = MaskEditor(self.x.shape, int(self.cfg.get("history", 50)))
            self.selection.set_editor(self.editor)
        self.recon = None
        self.grid.set_candidates([])
        self.grid.set_slice(self.x, self.x.shape)
        lo, hi = (float(v) for v in np.percentile(self.x, [0.5, 99.5]))
        self.level.set_val((lo + hi) / 2)
        self.width.set_val(max(hi - lo, 1e-3))
        self.start_reconstruction()

    def current_params(self) -> ReconParams:
        return self.params.with_(n_angles=int(self.angles.value_selected), decoder=self.decoder.value_selected)

    def start_reconstruction(self) -> None:
        self.params = self.current_params()
        if self.sync:
            from nullgraft_demo.workers import CancelToken

            self._apply(reconstruct(self.x, self.params, self.cache, token=CancelToken(), progress=self._progress))
            return
        if self.job is not None and not self.job.done():
            self.job.cancel()
        self.job = self.runner.submit("recon", reconstruct, self.x, self.params, self.cache,
                                      on_progress=self._progress, on_done=self._results.put)

    def _progress(self, fraction: float, message: str) -> None:
        self._results.put(("progress", fraction, message))

    def _drain(self) -> None:
        """Таймер в GUI-потоке забирает результаты фоновых задач."""
        while not self._results.empty():
            item = self._results.get_nowait()
            if isinstance(item, ReconResult):
                self._apply(item)
            elif isinstance(item, tuple) and item[0] == "progress":
                self.status.set_text(f"{item[2]} ({item[1]:.0%})")
                self.throttle.request()

    def _apply(self, r: ReconResult) -> None:
        self.recon = r
        self.grid.set_recon(r.y, r.z, r.det_residual)
        self.grid.set_info(f"‖A z − y‖ = {r.consistency:.4g}\nε = {r.eps:.4g}\n"
                           f"время {r.seconds:.2f} с{' (кэш)' if r.from_cache else ''}")
        self.throttle.request()

    def suggest(self) -> None:
        if self.recon is None:
            return
        c = self.cfg.get("candidates", {})
        self.candidates = suggest_candidates(self.recon.z, self.x, int(c.get("k", 5)), int(c.get("radius", 6)))
        self.grid.set_candidates(self.candidates)
        self.throttle.request()

    def choose_candidate(self, index: int) -> None:
        r, c, rad, _ = self.candidates[index]
        self.editor.apply(disk_region(self.editor.shape, r, c, rad), "replace")

    def _on_pick(self, event: Any) -> None:
        idx = getattr(event.artist, "candidate_index", None)
        if idx is not None:
            self.choose_candidate(int(idx))

    def _on_key(self, event: Any) -> None:
        keys = {"ctrl+z": self.editor.undo, "ctrl+y": self.editor.redo, "delete": self.editor.clear}
        if event.key in keys:
            keys[event.key]()
            self.throttle.request()

    def _on_hover(self, event: Any) -> None:
        info = self.grid.pixel_info(event.inaxes, event.xdata, event.ydata)
        if info:
            self.status.set_text(info)
            self.throttle.request()

    def _update_status(self) -> None:
        if not self.editor.empty:
            self.status.set_text(f"R: {self.editor.area_px} пикс. = "
                                 f"{self.editor.area_mm2(self.source.pixel_mm):.1f} мм²")


def run(cfg: dict[str, Any], source: ArraySource, cache: ResultCache, screenshot: str | None = None) -> int:
    import matplotlib

    if screenshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key in matplotlib.rcParams:
        if key.startswith("keymap."):
            matplotlib.rcParams[key] = []
    fig = plt.figure(figsize=(15, 8.5))
    runner = None if screenshot else JobRunner(max_workers=int(cfg["workers"]["threads"]))
    app = MplApp(fig, cfg, source, cache, runner, sync=bool(screenshot))
    if screenshot:
        app.suggest()
        if app.candidates:
            app.choose_candidate(0)
        app.throttle.flush()
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(screenshot, dpi=100)
        logger.info("снимок окна: %s", screenshot)
        return 0
    plt.show()
    if runner is not None:
        runner.shutdown()
    return 0
