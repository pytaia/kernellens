"""Ограничение частоты перерисовки: запросы за интервал сливаются в один ``draw_idle``."""

from __future__ import annotations

from matplotlib.backend_bases import FigureCanvasBase


class DrawThrottle:
    def __init__(self, canvas: FigureCanvasBase, interval_ms: int = 30) -> None:
        self.canvas = canvas
        self._pending = False
        self.draws = 0
        self._timer = canvas.new_timer(interval=interval_ms)
        self._timer.single_shot = True
        self._timer.add_callback(self._fire)

    def request(self) -> None:
        if not self._pending:
            self._pending = True
            self._timer.start()

    def flush(self) -> None:
        """Немедленная перерисовка (для тестов и снимков экрана)."""
        self._timer.stop()
        self._fire()

    def _fire(self) -> None:
        self._pending = False
        self.draws += 1
        self.canvas.draw_idle()
