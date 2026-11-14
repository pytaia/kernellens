"""Главное окно на PyQt6 + matplotlib: параметры слева, сетка в центре, результаты справа, лог внизу."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backend_bases import MouseEvent, PickEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nullgraft.api import DECODER_NAMES, local_resolution
from nullgraft.metrics import to_hu
from nullgraft.operators import Geometry
from nullgraft_demo.analysis import (
    LimitResult,
    RiskMap,
    SimilarResult,
    compare_decoders,
    feasible_limit,
    risk_map,
    similar_artifacts,
)
from nullgraft_demo.compute import ReconResult, reconstruct, suggest_candidates
from nullgraft_demo.report import save_figures, write_docx
from nullgraft_demo.views.analysis import draw_compare, draw_limit, draw_similar
from nullgraft_demo.datasource import ArraySource, open_source
from nullgraft_demo.detail import (
    STAGE_TITLES,
    STAGES,
    CheckResult,
    DetailPreview,
    detail_source,
    fit_realism,
    preview_detail,
    run_check,
)
from nullgraft_demo.state import DetailParams, ReconParams, SessionState
from nullgraft_demo.tools import MaskEditor, disk_region
from nullgraft_demo.tools.selection import SelectionController, Tool
from nullgraft_demo.views import DrawThrottle, ImageGrid
from nullgraft_demo.views.results import DISCLAIMER, badge, draw_preview, draw_result
from nullgraft_demo.workers import Job, JobRunner, ResultCache

STATUS_MARK = {"pending": "○", "running": "▶", "done": "✔", "failed": "✖", "skipped": "–"}

logger = logging.getLogger(__name__)

DECODER_LABELS = {
    "cgls": "CGLS (наим. квадраты)", "fbp-ramp": "FBP (ramp)", "fbp-hann": "FBP (Hann)", "fbp-shepp-logan": "FBP (Shepp–Logan)",
    "sart": "SART", "tv-admm": "TV-ADMM", "neural": "Нейросеть (файл весов)",
}
N_ANGLES = (8, 15, 30, 60, 120, 360)
TOOL_LABELS: dict[Tool, tuple[str, str]] = {
    "rect": ("Прямоугольник", "R"), "ellipse": ("Эллипс", "E"), "lasso": ("Лассо", "L"), "move": ("Сдвиг", "M"),
}
WL_STEPS = 1000


def disable_mpl_keymaps() -> None:
    """Горячие клавиши задаёт окно; стандартные клавиши matplotlib (l, s, r, …) им мешают."""
    for key in mpl.rcParams:
        if key.startswith("keymap."):
            mpl.rcParams[key] = []


class Bridge(QObject):
    """Сигналы из рабочих потоков в GUI-поток (очередь Qt)."""

    progress = pyqtSignal(str, float, str)
    done = pyqtSignal(str, str, object)
    failed = pyqtSignal(str, str, str)
    cancelled = pyqtSignal(str, str)
    stage = pyqtSignal(str, str, str, object)
    log = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: Bridge) -> None:
        super().__init__(logging.INFO)
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.log.emit(self.format(record))
        except RuntimeError:
            pass


class MainWindow(QMainWindow):
    def __init__(self, cfg: dict[str, Any], source: ArraySource, cache: ResultCache, runner: JobRunner) -> None:
        super().__init__()
        self.cfg = cfg
        self.source = source
        self.cache = cache
        self.runner = runner
        self.params = ReconParams(**cfg["recon"])
        self.job: Job | None = None
        self.recon: ReconResult | None = None
        self.x: np.ndarray | None = None
        self.candidates: list[tuple[int, int, int, float]] = []
        self.chosen_tool: Tool = "rect"
        self._wl_range = (0.0, 1.0)
        self.setWindowTitle("nullgraft — проверка деталей реконструкции")

        self.bridge = Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.done.connect(self._on_done)
        self.bridge.failed.connect(self._on_failed)
        self.bridge.cancelled.connect(self._on_cancelled)
        self.bridge.stage.connect(self._on_stage)
        self.check: CheckResult | None = None
        self.preview: DetailPreview | None = None
        self.file_detail: np.ndarray | None = None
        self._picking_offset = False
        self._realism: dict[tuple[str, int], Any] = {}
        self.log_handler = QtLogHandler(self.bridge)
        logging.getLogger().addHandler(self.log_handler)

        disable_mpl_keymaps()
        self.fig = Figure(figsize=(10, 7), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.grid = ImageGrid(self.fig)
        self.throttle = DrawThrottle(self.canvas, int(cfg.get("redraw_ms", 30)))
        self.editor = self._new_editor(source.get(0).shape)
        self.selection = SelectionController(self.grid.image_axes, self.editor, self.throttle.request)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        self.setCentralWidget(central)
        self._build_left()
        self._build_right()
        self._build_bottom()
        self._build_shortcuts()
        self._build_menu()
        for name in ("pan", "zoom"):
            action = self.toolbar._actions.get(name)  # noqa: SLF001
            if action is not None:
                action.toggled.connect(self._sync_toolbar_mode)
        self.statusBar().showMessage("Готово")
        self.resize(1500, 900)
        self._load_slice(0)

    # ------------------------------------------------------------------ панели
    def _build_left(self) -> None:
        panel = QWidget()
        v = QVBoxLayout(panel)

        data = QGroupBox("Данные")
        f = QFormLayout(data)
        self.source_label = QLabel(self._source_text())
        self.source_label.setWordWrap(True)
        open_btn = QPushButton("Открыть папку / файл…")
        open_btn.clicked.connect(self._open_source)
        syn_btn = QPushButton("Синтетика")
        syn_btn.clicked.connect(lambda: self._set_source(open_source(None, self.cfg)))
        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(syn_btn)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self._load_slice)
        self.slice_label = QLabel()
        f.addRow(self.source_label)
        f.addRow(row)
        f.addRow("Срез", self.slice_slider)
        f.addRow("", self.slice_label)
        v.addWidget(data)

        rec = QGroupBox("Измерение и реконструкция")
        f = QFormLayout(rec)
        self.angles_box = QComboBox()
        for n in N_ANGLES:
            self.angles_box.addItem(str(n), n)
        self.angles_box.setCurrentIndex(max(0, N_ANGLES.index(self.params.n_angles)
                                            if self.params.n_angles in N_ANGLES else 1))
        self.beam_box = QComboBox()
        self.beam_box.addItem("параллельный", "parallel")
        self.beam_box.addItem("веерный", "fan")
        self.beam_box.setCurrentIndex(0 if self.params.beam == "parallel" else 1)
        self.sigma_box = _float_box(self.params.sigma, 0.0, 1.0, 5, 1e-4)
        self.eps_box = _float_box(self.params.eps, 0.0, 1e6, 4, 0.01)
        self.eps_box.setSpecialValueText("авто")
        self.decoder_box = QComboBox()
        for name in (*DECODER_NAMES, "neural"):
            self.decoder_box.addItem(DECODER_LABELS[name], name)
        self.decoder_box.setCurrentIndex(max(0, self.decoder_box.findData(self.params.decoder)))
        self.iters_box = _int_box(self.params.decoder_iters, 1, 5000)
        self.noise_box = _int_box(self.params.n_noise, 1, 500)
        self.seed_box = _int_box(self.params.seed, 0, 10**6)
        self.auto_recon = QCheckBox("пересчитывать при смене среза")
        self.auto_recon.setChecked(True)
        self.recon_btn = QPushButton("Реконструировать  (Ctrl+R)")
        self.recon_btn.clicked.connect(self.start_reconstruction)
        f.addRow("Проекций N_θ", self.angles_box)
        f.addRow("Геометрия", self.beam_box)
        f.addRow("σ шума", self.sigma_box)
        f.addRow("ε", self.eps_box)
        f.addRow("Декодер", self.decoder_box)
        f.addRow("Итераций", self.iters_box)
        f.addRow("Реализаций шума M", self.noise_box)
        f.addRow("seed", self.seed_box)
        f.addRow(self.auto_recon)
        f.addRow(self.recon_btn)
        v.addWidget(rec)

        wl = QGroupBox("Окно / уровень")
        f = QFormLayout(wl)
        self.level_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        for s in (self.level_slider, self.width_slider):
            s.setRange(1, WL_STEPS)
            s.valueChanged.connect(self._on_window_changed)
        f.addRow("Уровень", self.level_slider)
        f.addRow("Ширина", self.width_slider)
        v.addWidget(wl)

        sel = QGroupBox("Область R")
        g = QVBoxLayout(sel)
        tools_row = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_buttons: dict[Tool, QToolButton] = {}
        for tool, (label, key) in TOOL_LABELS.items():
            b = QToolButton()
            b.setText(f"{label} ({key})")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, t=tool: self.set_tool(t))
            self.tool_group.addButton(b)
            tools_row.addWidget(b)
            self.tool_buttons[tool] = b
        self.tool_buttons["rect"].setChecked(True)
        g.addLayout(tools_row)
        mode_row = QHBoxLayout()
        self.add_radio = QRadioButton("Добавить (A)")
        self.sub_radio = QRadioButton("Вычесть (S, или Shift)")
        self.add_radio.setChecked(True)
        self.add_radio.toggled.connect(lambda on: self.selection.set_mode("add" if on else "subtract"))
        mode_row.addWidget(self.add_radio)
        mode_row.addWidget(self.sub_radio)
        g.addLayout(mode_row)
        edit_row = QHBoxLayout()
        for text, slot in (("Отменить", self.undo), ("Повторить", self.redo), ("Очистить", self.clear_mask)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            edit_row.addWidget(b)
        g.addLayout(edit_row)
        cand_btn = QPushButton("Предложить кандидатов")
        cand_btn.clicked.connect(self.suggest)
        g.addWidget(cand_btn)
        v.addWidget(sel)

        det = QGroupBox("Деталь и проверка (A)")
        f = QFormLayout(det)
        self.mode_box = QComboBox()
        self.mode_box.addItem("удалить деталь", "remove")
        self.mode_box.addItem("добавить деталь", "add")
        self.detail_src_box = QComboBox()
        self.detail_src_box.addItem("срез архива", "slice")
        self.detail_src_box.addItem("файл", "file")
        self.projection_box = QComboBox()
        self.projection_box.addItem("на ядро внутри R", "local")
        self.projection_box.addItem("на всё ядро (глобальная)", "global")
        self.src_slice_box = _int_box(0, 0, max(self.source.n_slices - 1, 0))
        self.off_y_box = _int_box(0, -2048, 2048)
        self.off_x_box = _int_box(0, -2048, 2048)
        off_row = QHBoxLayout()
        off_row.addWidget(self.off_y_box)
        off_row.addWidget(self.off_x_box)
        self.pick_btn = QPushButton("Указать мышью на срезе-источнике")
        self.pick_btn.setCheckable(True)
        self.pick_btn.toggled.connect(self._toggle_pick_offset)
        file_btn = QPushButton("Файл детали…")
        file_btn.clicked.connect(self._open_detail_file)
        self.width_slider_blend = QSlider(Qt.Orientation.Horizontal)
        self.width_slider_blend.setRange(0, 20)
        self.width_slider_blend.setValue(6)
        self.width_label = QLabel()
        self.width_slider_blend.valueChanged.connect(
            lambda v: self.width_label.setText(f"{v / 2:.1f} пикс."))
        self.width_label.setText(f"{self.width_slider_blend.value() / 2:.1f} пикс.")
        self.realism_check = QCheckBox("проверять реалистичность x'' по архиву")
        self.realism_check.setChecked(bool(self.cfg.get("realism_check", True)))
        prev_btn = QPushButton("Превью детали")
        prev_btn.clicked.connect(self.start_preview)
        self.check_btn = QPushButton("Запустить проверку  (Ctrl+Enter)")
        self.check_btn.clicked.connect(self.start_check)
        f.addRow("Режим", self.mode_box)
        f.addRow("Источник", self.detail_src_box)
        f.addRow("Проекция", self.projection_box)
        f.addRow("Срез-источник", self.src_slice_box)
        f.addRow("Сдвиг (строка, столбец)", off_row)
        f.addRow(self.pick_btn)
        f.addRow(file_btn)
        f.addRow("Ширина перехода α", self.width_slider_blend)
        f.addRow("", self.width_label)
        f.addRow(self.realism_check)
        f.addRow(prev_btn)
        f.addRow(self.check_btn)
        v.addWidget(det)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        dock = QDockWidget("Параметры", self)
        dock.setObjectName("params")
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_right(self) -> None:
        self.tabs = QTabWidget()
        area = QWidget()
        v = QVBoxLayout(area)
        self.area_label = QLabel("Выделите область на любом изображении.")
        self.area_label.setWordWrap(True)
        self.area_label.setTextFormat(Qt.TextFormat.RichText)
        self.cand_list = QListWidget()
        self.cand_list.itemClicked.connect(lambda item: self.choose_candidate(self.cand_list.row(item)))
        v.addWidget(self.area_label)
        v.addWidget(QLabel("Кандидаты (клик — выбрать):"))
        v.addWidget(self.cand_list, 1)
        self.tabs.addTab(area, "Область")

        prev = QWidget()
        pv = QVBoxLayout(prev)
        self.preview_fig = Figure(figsize=(4, 2.2), layout="constrained")
        self.preview_canvas = FigureCanvasQTAgg(self.preview_fig)
        self.preview_label = QLabel("Нажмите «Превью детали».")
        self.preview_label.setWordWrap(True)
        pv.addWidget(self.preview_canvas, 1)
        pv.addWidget(self.preview_label)
        self.tabs.addTab(prev, "Деталь")

        chk = QWidget()
        cv = QVBoxLayout(chk)
        self.stage_list = QListWidget()
        self.stage_list.setMaximumHeight(118)
        self._reset_stages()
        self.badge_label = QLabel("—")
        self.badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_label.setStyleSheet("background:#dee2e6; color:#212529; padding:6px; font-weight:bold;")
        self.verdict_label = QLabel("Выберите область и нажмите «Запустить проверку».")
        self.verdict_label.setWordWrap(True)
        self.numbers_label = QLabel("")
        self.numbers_label.setWordWrap(True)
        self.result_fig = Figure(figsize=(4, 3.2), layout="constrained")
        self.result_canvas = FigureCanvasQTAgg(self.result_fig)
        disclaimer = QLabel(DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color:#495057; font-size:9pt; border-top:1px solid #ced4da; padding-top:4px;")
        cv.addWidget(self.stage_list)
        cv.addWidget(self.badge_label)
        cv.addWidget(self.verdict_label)
        cv.addWidget(self.numbers_label)
        cv.addWidget(self.result_canvas, 1)
        cv.addWidget(disclaimer)
        self.tabs.addTab(chk, "Проверка (A)")
        self._build_analysis_tabs()
        dock = QDockWidget("Результаты", self)
        dock.setObjectName("results")
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(320)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _figure_tab(self, title: str, controls: list[QWidget], height: float = 3.5) -> tuple[Figure, FigureCanvasQTAgg,
                                                                                            QLabel]:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        for c in controls:
            row.addWidget(c)
        row.addStretch(1)
        v.addLayout(row)
        fig = Figure(figsize=(4, height), layout="constrained")
        canvas = FigureCanvasQTAgg(fig)
        label = QLabel("")
        label.setWordWrap(True)
        v.addWidget(canvas, 1)
        v.addWidget(label)
        self.tabs.addTab(w, title)
        return fig, canvas, label

    def _build_analysis_tabs(self) -> None:
        self.compare_boxes: dict[str, QCheckBox] = {}
        cmp_controls: list[QWidget] = []
        for name in DECODER_NAMES[:4]:
            cb = QCheckBox(DECODER_LABELS[name])
            cb.setChecked(name in ("cgls", "fbp-hann"))
            self.compare_boxes[name] = cb
            cmp_controls.append(cb)
        run_cmp = QPushButton("Сравнить")
        run_cmp.clicked.connect(self.start_compare)
        self.compare_fig, self.compare_canvas, self.compare_label = self._figure_tab(
            "Сравнение", [*cmp_controls, run_cmp], 2.6)
        self.compare_table = QTableWidget()
        self.compare_table.setMaximumHeight(150)
        self.tabs.widget(self.tabs.count() - 1).layout().insertWidget(2, self.compare_table)

        self.k_box = _int_box(int(self.cfg.get("b2", {}).get("k", 6)), 1, 20)
        run_b2 = QPushButton("Найти похожие артефакты")
        run_b2.clicked.connect(self.start_similar)
        self.similar_fig, self.similar_canvas, self.similar_label = self._figure_tab(
            "Похожие (B2)", [QLabel("K"), self.k_box, run_b2], 5)
        self._similar_axes: dict = {}
        self._similar_items: list = []
        self.similar_canvas.mpl_connect("button_press_event", self._on_similar_click)

        run_b1 = QPushButton("Вычислить предел")
        run_b1.clicked.connect(self.start_limit)
        self.limit_fig, self.limit_canvas, self.limit_label = self._figure_tab("Предел (B1)", [run_b1], 4)

        self.step_box = _int_box(int(self.cfg.get("risk", {}).get("step", 16)), 4, 128)
        run_risk = QPushButton("Построить карту риска")
        run_risk.clicked.connect(self.start_risk)
        clear_risk = QPushButton("Убрать с изображения")
        clear_risk.clicked.connect(lambda: (self.grid.set_overlay(None), self.throttle.request()))
        self.risk_fig, self.risk_canvas, self.risk_label = self._figure_tab(
            "Карта риска", [QLabel("шаг"), self.step_box, run_risk, clear_risk], 3)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Файл")
        for text, key, slot in (("Сохранить сессию…", "Ctrl+S", self.save_session),
                                ("Открыть сессию…", "Ctrl+O", self.load_session),
                                ("Отчёт .docx…", "Ctrl+D", self.export_report),
                                ("Сохранить все панели в PNG…", "Ctrl+P", self.export_png)):
            act = menu.addAction(text)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(slot)

    def _build_bottom(self) -> None:
        w = QWidget()
        h = QHBoxLayout(w)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.bridge.log.connect(self.log_view.appendPlainText)
        side = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.stage_label = QLabel("нет задач")
        self.cancel_btn = QPushButton("Отменить задачу (Esc)")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_job)
        side.addWidget(self.stage_label)
        side.addWidget(self.progress)
        side.addWidget(self.cancel_btn)
        side.addStretch(1)
        h.addWidget(self.log_view, 1)
        h.addLayout(side)
        dock = QDockWidget("Лог и прогресс", self)
        dock.setObjectName("log")
        dock.setWidget(w)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [170], Qt.Orientation.Vertical)

    def _build_shortcuts(self) -> None:
        keys = {
            "Ctrl+Z": self.undo, "Ctrl+Y": self.redo, "Ctrl+Shift+Z": self.redo, "Delete": self.clear_mask,
            "Ctrl+R": self.start_reconstruction, "Escape": self.cancel_job,
            "Ctrl+Return": self.start_check, "Ctrl+Enter": self.start_check,
            "A": lambda: self.add_radio.setChecked(True), "S": lambda: self.sub_radio.setChecked(True),
            "PgUp": lambda: self.slice_slider.setValue(self.slice_slider.value() - 1),
            "PgDown": lambda: self.slice_slider.setValue(self.slice_slider.value() + 1),
        }
        for tool, (_, key) in TOOL_LABELS.items():
            keys[key] = lambda t=tool: self.set_tool(t)
        self._shortcuts = []
        for seq, slot in keys.items():
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    # ------------------------------------------------------------------ данные
    def _source_text(self) -> str:
        s = self.source
        return f"<b>{s.name}</b>: {s.n_slices} срезов, {s.shape[0]}×{s.shape[1]}, {s.pixel_mm:g} мм/пикс."

    def _open_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Срез или cryo.npy", "", "Изображения и массивы (*.png *.tif *.tiff *.jpg *.npy *.dcm);;Все файлы (*)")
        if not path:
            return
        p = Path(path)
        target = p.parent if p.name == "cryo.npy" else p
        try:
            self._set_source(open_source(target, self.cfg))
        except (OSError, ValueError, ImportError) as exc:
            logger.error("не удалось открыть %s: %s", target, exc)

    def _set_source(self, source: ArraySource) -> None:
        self.source = source
        self.source_label.setText(self._source_text())
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, source.n_slices - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self._load_slice(0)

    def _new_editor(self, shape: tuple[int, int]) -> MaskEditor:
        editor = MaskEditor(shape, history=int(self.cfg.get("history", 50)))
        editor.subscribe(self._on_mask_changed)
        return editor

    def _load_slice(self, index: int) -> None:
        self.cancel_job()
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, self.source.n_slices - 1)
        self.slice_slider.setValue(index)
        self.slice_slider.blockSignals(False)
        self.slice_label.setText(f"{index + 1} из {self.source.n_slices}")
        self.x = self.source.get(index)
        if self.editor.shape != self.x.shape:
            self.editor = self._new_editor(self.x.shape)
            self.selection.set_editor(self.editor)
        self.recon = None
        self.candidates = []
        self.cand_list.clear() if hasattr(self, "cand_list") else None
        self.grid.set_candidates([])
        self.grid.set_slice(self.x, self.x.shape)
        self.grid.set_mask(self.editor.mask)
        self._reset_window()
        self._update_area_info()
        self.throttle.request()
        if self.auto_recon.isChecked():
            self.start_reconstruction()

    def _reset_window(self) -> None:
        lo, hi = (float(v) for v in np.percentile(self.x, [0.5, 99.5]))
        if hi <= lo:
            hi = lo + 1.0
        pad = 0.25 * (hi - lo)
        self._wl_range = (lo - pad, hi + pad)
        for s, v in ((self.level_slider, (lo + hi) / 2), (self.width_slider, hi - lo)):
            s.blockSignals(True)
            s.setValue(self._to_slider(v, s is self.width_slider))
            s.blockSignals(False)
        self._on_window_changed()

    def _to_slider(self, value: float, is_width: bool) -> int:
        lo, hi = self._wl_range
        frac = value / (hi - lo) if is_width else (value - lo) / (hi - lo)
        return int(np.clip(round(frac * WL_STEPS), 1, WL_STEPS))

    def _on_window_changed(self) -> None:
        lo, hi = self._wl_range
        level = lo + self.level_slider.value() / WL_STEPS * (hi - lo)
        width = self.width_slider.value() / WL_STEPS * (hi - lo)
        self.grid.set_window(level, width)
        self.throttle.request()

    # ------------------------------------------------------------- реконструкция
    def current_params(self) -> ReconParams:
        return ReconParams(
            n_angles=int(self.angles_box.currentData()), beam=self.beam_box.currentData(),
            sigma=self.sigma_box.value(), eps=self.eps_box.value(), eps_factor=self.params.eps_factor,
            decoder=self.decoder_box.currentData(), decoder_iters=self.iters_box.value(),
            n_noise=self.noise_box.value(), seed=self.seed_box.value(), weights_path=self.params.weights_path,
        )

    def start_reconstruction(self) -> None:
        if self.x is None:
            return
        self.params = self.current_params()
        holder: dict[str, str] = {}

        def job(x, params, cache, *, token, progress):
            # быстрое FBP-превью, затем подмена результатом выбранного декодера (кэш)
            if params.decoder not in ("fbp-ramp", "fbp-hann", "fbp-shepp-logan"):
                prev = reconstruct(x, params.with_(decoder="fbp-hann"), cache, token=token, progress=lambda f, m: None)
                self.bridge.stage.emit(holder.get("id", ""), "recon-preview", "done", prev)
            return reconstruct(x, params, cache, token=token, progress=progress)

        self._start_job("recon", job, self.x, self.params, self.cache)
        holder["id"] = self.job.id

    def _start_job(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        self.cancel_job()
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(True)
        self.stage_label.setText(f"{name}: запуск")
        self._t0 = time.perf_counter()
        holder: dict[str, str] = {}
        self.job = self.runner.submit(
            name, fn, *args,
            on_progress=lambda f, m: self.bridge.progress.emit(holder.get("id", ""), f, m),
            on_done=lambda r: self.bridge.done.emit(name, holder.get("id", ""), r),
            on_error=lambda e: self.bridge.failed.emit(name, holder.get("id", ""), f"{type(e).__name__}: {e}"),
            on_cancel=lambda: self.bridge.cancelled.emit(name, holder.get("id", "")),
            **kwargs,
        )
        holder["id"] = self.job.id
        self._job_holder = holder

    def _is_current(self, job_id: str) -> bool:
        return self.job is not None and job_id in ("", self.job.id)

    def _on_progress(self, job_id: str, fraction: float, message: str) -> None:
        if self._is_current(job_id):
            self.progress.setValue(int(fraction * 1000))
            self.stage_label.setText(message)

    def _on_done(self, name: str, job_id: str, result: object) -> None:
        if not self._is_current(job_id):
            return
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(1000)
        if name == "recon" and isinstance(result, ReconResult):
            self._show_recon(result)
        elif name == "preview" and isinstance(result, DetailPreview):
            self._show_preview(result)
        elif name == "check" and isinstance(result, CheckResult):
            self._show_check(result)
        elif name == "compare":
            self._show_compare(result)
        elif name == "similar" and isinstance(result, SimilarResult):
            self._show_similar(result)
        elif name == "limit" and isinstance(result, LimitResult):
            self._show_limit(result)
        elif name == "risk" and isinstance(result, RiskMap):
            self._show_risk(result)

    def _on_failed(self, name: str, job_id: str, message: str) -> None:
        if self._is_current(job_id):
            self.cancel_btn.setEnabled(False)
            self.stage_label.setText(f"{name}: ошибка")
            logger.error("%s: %s", name, message)

    def _on_cancelled(self, name: str, job_id: str) -> None:
        self.stage_label.setText(f"{name}: отменено")

    def cancel_job(self) -> None:
        if self.job is not None and not self.job.done():
            self.job.cancel()
        self.cancel_btn.setEnabled(False) if hasattr(self, "cancel_btn") else None

    def _show_recon(self, r: ReconResult) -> None:
        self.recon = r
        self.grid.set_recon(r.y, r.z, r.det_residual)
        state = "согласована" if r.consistent else "НЕ согласована"
        self.grid.set_info(
            f"‖A z − y‖ = {r.consistency:.4g}\nε        = {r.eps:.4g}\nреконструкция {state}\n"
            f"время: {r.seconds:.2f} с{' (кэш)' if r.from_cache else ''}\n"
            f"декодер: {DECODER_LABELS.get(self.params.decoder, self.params.decoder)}\n"
            f"N_θ = {self.params.n_angles}, σ = {self.params.sigma:g}")
        logger.info("реконструкция за %.2f с%s; ‖Az−y‖ = %.4g, ε = %.4g", r.seconds,
                    " (из кэша)" if r.from_cache else "", r.consistency, r.eps)
        self.throttle.request()

    # ------------------------------------------------------------------ деталь и A
    def current_detail(self) -> DetailParams:
        return DetailParams(
            source=self.detail_src_box.currentData(), source_slice=self.src_slice_box.value(),
            mode=self.mode_box.currentData(), window_width=self.width_slider_blend.value() / 2,
            projection=self.projection_box.currentData(),
        )

    def _detail_source_image(self) -> np.ndarray | None:
        if self.recon is None or self.editor.empty:
            logger.warning("нужны реконструкция и выбранная область R")
            return None
        d = self.current_detail()
        other = None
        if d.mode == "add" and d.source == "slice":
            other = self.source.get(min(d.source_slice, self.source.n_slices - 1))
        try:
            return detail_source(d, self.recon.z, np.array(self.editor.mask), other,
                                 (self.off_y_box.value(), self.off_x_box.value()), self.file_detail)
        except ValueError as exc:
            logger.error("источник детали: %s", exc)
            return None

    def start_preview(self) -> None:
        source = self._detail_source_image()
        if source is not None:
            self._start_job("preview", preview_detail, self.recon.z, source, np.array(self.editor.mask), self.params,
                            self.current_detail().projection)

    def start_check(self) -> None:
        source = self._detail_source_image()
        if source is None:
            return
        self._reset_stages()
        self.badge_label.setText("выполняется…")
        self.badge_label.setStyleSheet("background:#dee2e6; color:#212529; padding:6px; font-weight:bold;")
        self.verdict_label.setText("")
        holder_ref: dict[str, Any] = {}
        realism_slices = self._realism_slices() if self.realism_check.isChecked() else None
        realism_key = (self.source.name, self.slice_slider.value())
        cache = self._realism

        def job(recon, source_img, mask, params, detail, *, token, progress, on_stage):
            realism = None
            if realism_slices is not None:
                if realism_key not in cache:
                    progress(0.01, "модель реалистичности по архиву")
                    cache[realism_key] = fit_realism(realism_slices, patch_size=min(32, mask.shape[0] // 4))
                realism = cache[realism_key]
            return run_check(recon, source_img, mask, params, detail, realism=realism, on_stage=on_stage,
                             token=token, progress=progress)

        self._start_job("check", job, self.recon, source, np.array(self.editor.mask), self.params,
                        self.current_detail(),
                        on_stage=lambda s, st, d: self.bridge.stage.emit(holder_ref.get("id", ""), s, st, d))
        holder_ref["id"] = self.job.id

    def _realism_slices(self) -> list[np.ndarray] | None:
        """Срезы для модели реалистичности: подальше от текущего (соседние почти совпадают)."""
        cur = self.slice_slider.value()
        idx = [i for i in range(self.source.n_slices) if abs(i - cur) > 2][:12]
        if len(idx) < 2:
            idx = [i for i in range(self.source.n_slices) if i != cur][:12]
        if len(idx) < 2:
            logger.warning("в архиве меньше двух других срезов: проверка реалистичности x'' пропущена")
            return None
        return [self.source.get(i) for i in idx]

    def _reset_stages(self) -> None:
        self.stage_list.clear()
        for s in STAGES:
            self.stage_list.addItem(f"{STATUS_MARK['pending']} {STAGE_TITLES[s]}")

    def _on_stage(self, job_id: str, stage: str, status: str, data: object) -> None:
        if stage == "recon-preview" and self._is_current(job_id) and isinstance(data, ReconResult):
            self.grid.set_recon(data.y, data.z, data.det_residual)
            self.grid.set_info("превью FBP (Hann);\nидёт выбранный декодер…")
            self.throttle.request()
            return
        if not self._is_current(job_id) or stage not in STAGES:
            return
        item = self.stage_list.item(STAGES.index(stage))
        extra = ""
        if stage == "decoder" and isinstance(data, dict) and "done" in data:
            extra = f" ({data['done']}/{data['total']})"
        if stage == "consistency" and isinstance(data, dict) and "residual" in data:
            extra = f" ({data['residual']:.3g} vs {data['eps']:.3g})"
        if stage == "blend" and isinstance(data, dict) and "delta" in data:
            extra = f" (Δ = {data['delta']:.3g}, предел {data['delta_limit']:.3g})"
        item.setText(f"{STATUS_MARK.get(status, '?')} {STAGE_TITLES[stage]}{extra}")
        if status == "failed":
            for later in STAGES[STAGES.index(stage) + 1:]:
                self.stage_list.item(STAGES.index(later)).setText(f"{STATUS_MARK['skipped']} {STAGE_TITLES[later]}")

    def _show_preview(self, p: DetailPreview) -> None:
        self.preview = p
        draw_preview(self.preview_fig, p.x_det, p.pinv, p.x_proj, np.array(self.editor.mask), p.kernel_ratio)
        self.preview_canvas.draw_idle()
        self.preview_label.setText(f"‖A x_det^proj‖ / ‖A x_det‖ = {p.kernel_ratio:.2e} (должно быть ≪ 1); "
                                   f"{p.seconds:.2f} с")
        self.tabs.setCurrentIndex(1)

    def _show_check(self, c: CheckResult) -> None:
        self.check = c
        res = c.assessment
        color, text = badge(res.outcome.value)
        self.badge_label.setText(res.outcome.value)
        self.badge_label.setStyleSheet(f"background:{color}; color:white; padding:6px; font-weight:bold;")
        self.verdict_label.setText(text)
        nums = [f"‖z − x''‖ = {res.graft_size:.4g}", f"Δ = {res.delta:.3g} (предел {res.delta_limit:.3g})"]
        if np.isfinite(res.eta_min):
            nums.insert(0, f"η ∈ [{res.eta_min:.4g}, {res.eta_max:.4g}]")
        if res.realism is not None and np.isfinite(res.realism.threshold):
            nums.append(f"реалистичность: {'да' if res.realism.realistic else 'нет'} "
                        f"({res.realism.score:.3g} при пороге {res.realism.threshold:.3g})")
        elif res.realism is not None:
            nums.append("реалистичность x'' не проверялась")
        nums.append(f"время: первый результат {c.first_stage_seconds:.2f} с, всего {c.seconds:.2f} с")
        self.numbers_label.setText("; ".join(nums) + ("<br><i>" + "; ".join(res.notes) + "</i>" if res.notes else ""))
        draw_result(self.result_fig, res, c.z, c.mask)
        self.result_canvas.draw_idle()
        self.tabs.setCurrentIndex(2)

    def _toggle_pick_offset(self, on: bool) -> None:
        """Показать срез-источник на панели эталона; клик задаёт сдвиг относительно центра R."""
        self._picking_offset = on
        if on:
            self.selection.set_tool("none")
            src = self.source.get(min(self.src_slice_box.value(), self.source.n_slices - 1))
            self.grid._im[self.grid.ax_x].set_data(src)  # noqa: SLF001
            self.statusBar().showMessage("Кликните на панели «Эталон x» по участку-источнику детали")
            if not hasattr(self, "_pick_cid"):
                self._pick_cid = self.canvas.mpl_connect("button_press_event", self._on_pick_offset)
        else:
            self.selection.set_tool(self.chosen_tool)
            self.grid._im[self.grid.ax_x].set_data(self.x)  # noqa: SLF001
        self.throttle.request()

    def _on_pick_offset(self, event: MouseEvent) -> None:
        if not self._picking_offset or event.inaxes is not self.grid.ax_x or event.xdata is None:
            return
        centre = self.editor.centroid()
        if centre is None:
            logger.warning("сначала выберите область R")
            return
        self.off_y_box.setValue(int(round(event.ydata - centre[0])))
        self.off_x_box.setValue(int(round(event.xdata - centre[1])))
        self.mode_box.setCurrentIndex(1)
        self.detail_src_box.setCurrentIndex(0)
        self.pick_btn.setChecked(False)
        logger.info("источник детали: срез %d, сдвиг (%d, %d)", self.src_slice_box.value(),
                    self.off_y_box.value(), self.off_x_box.value())

    def _open_detail_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Изображение детали", "", "Изображения (*.png *.tif *.jpg *.npy)")
        if not path:
            return
        try:
            img = open_source(path, self.cfg).get(0)
        except (OSError, ValueError) as exc:
            logger.error("файл детали: %s", exc)
            return
        self.file_detail = img
        self.detail_src_box.setCurrentIndex(1)
        self.mode_box.setCurrentIndex(1)
        logger.info("деталь из файла %s, %d×%d", Path(path).name, *img.shape)

    # ------------------------------------------------------------ этапы 6–9
    def _need_recon_and_mask(self) -> bool:
        if self.recon is None or self.editor.empty:
            logger.warning("нужны реконструкция и выбранная область R")
            return False
        return True

    def start_compare(self) -> None:
        if not self._need_recon_and_mask():
            return
        names = [n for n, cb in self.compare_boxes.items() if cb.isChecked()]
        if not 2 <= len(names) <= 4:
            logger.warning("выберите от 2 до 4 декодеров")
            return
        mask = np.array(self.editor.mask)
        detail = self.current_detail()
        other = self.source.get(min(detail.source_slice, self.source.n_slices - 1))
        offset = (self.off_y_box.value(), self.off_x_box.value())
        file_img = self.file_detail

        def make_source(z: np.ndarray) -> np.ndarray:
            return detail_source(detail, z, mask, other, offset, file_img)

        self._start_job("compare", compare_decoders, self.x, self.params, names, mask, detail, make_source,
                        self.cache, float(self.cfg.get("arb_water", 0.2)))

    def _show_compare(self, rows: list) -> None:
        self.compare_rows = rows
        draw_compare(self.compare_fig, rows, np.array(self.editor.mask))
        self.compare_canvas.draw_idle()
        keys = list(rows[0].metrics) if rows else []
        self.compare_table.setRowCount(len(rows))
        self.compare_table.setColumnCount(len(keys) + 2)
        self.compare_table.setHorizontalHeaderLabels(["декодер", "исход", *keys])
        for i, r in enumerate(rows):
            outcome = r.check.assessment.outcome.value if r.check else r.error
            cells = [DECODER_LABELS.get(r.decoder, r.decoder), outcome, *[f"{r.metrics[k]:.4g}" for k in keys]]
            for j, text in enumerate(cells):
                self.compare_table.setItem(i, j, QTableWidgetItem(text))
        self.compare_table.resizeColumnsToContents()
        self.compare_label.setText("Одна синограмма и одна реализация шума (общий seed), одна область и деталь. "
                                   "Метрики — против эталона x: доступны только на синтетике и VHP.")
        self.tabs.setCurrentIndex(3)

    def start_similar(self) -> None:
        if not self._need_recon_and_mask():
            return
        from nullgraft_demo.config import cache_dir

        size = min(32, self.editor.shape[0] // 4)
        self._start_job("similar", similar_artifacts, self.source, self.slice_slider.value(), self.recon.z,
                        np.array(self.editor.mask), self.params, self.cache, cache_dir(self.cfg) / "b2",
                        size, self.k_box.value())

    def _show_similar(self, res: SimilarResult) -> None:
        self._similar_items = res.items
        self._similar_axes = draw_similar(self.similar_fig, res.items)
        self.similar_canvas.draw_idle()
        self.similar_label.setText(
            f"Похожие артефакты того же декодера: {res.fraction_real:.0%} — на реальной структуре эталона, "
            f"{res.fraction_background:.0%} — на пустом месте (индекс: {res.n_index} патчей, {res.seconds:.1f} с). "
            "Это частота в архиве, а не вероятность диагноза. Клик по патчу открывает его срез.")
        self.tabs.setCurrentIndex(4)

    def _on_similar_click(self, event: MouseEvent) -> None:
        idx = self._similar_axes.get(event.inaxes)
        if idx is None:
            return
        it = self._similar_items[idx]
        self.slice_slider.setValue(it.slice_index)
        self.editor.apply(disk_region(self.editor.shape, it.row, it.col, 6), "replace")
        logger.info("открыт срез %d, позиция (%d, %d)", it.slice_index + 1, it.row, it.col)

    def start_limit(self) -> None:
        if self.editor.empty:
            logger.warning("выберите область R")
            return
        angles = [n for n in N_ANGLES if n <= 120]
        size = min(24, self.editor.shape[0] // 4)
        self._start_job("limit", feasible_limit, self.source, self.slice_slider.value(), np.array(self.editor.mask),
                        self.current_params(), angles, self.cache, size)

    def _show_limit(self, res: LimitResult) -> None:
        draw_limit(self.limit_fig, res)
        self.limit_canvas.draw_idle()
        cur = res.angles.index(res.current) if res.current in res.angles else None
        txt = (f"Архив: {res.n_archive} патчей. Диаметры — оценки снизу; любой декодер ошибается хотя бы на "
               f"diam/2 на некотором элементе допустимого множества. {res.seconds:.1f} с"
               f"{' (кэш)' if res.from_cache else ''}.")
        if cur is not None:
            txt += f" Для N_θ = {res.current}: |F| = {res.sizes[cur]}, diam = {res.diameters[cur]:.4g}."
        self.limit_label.setText(txt)
        self.tabs.setCurrentIndex(5)

    def start_risk(self) -> None:
        if self.recon is None:
            logger.warning("сначала выполните реконструкцию")
            return
        self._start_job("risk", risk_map, self.recon, self.params, self.step_box.value(),
                        max(4, self.step_box.value() // 2))

    def _show_risk(self, rm: RiskMap) -> None:
        self.grid.set_overlay(rm.image)
        self.throttle.request()
        self.risk_fig.clear()
        ax = self.risk_fig.add_subplot()
        im = ax.imshow(np.ma.masked_invalid(rm.image), cmap="inferno", vmin=0, vmax=1)
        ax.set_title("доля реализаций шума, где деталь не воспроизводится (грубо)", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        self.risk_fig.colorbar(im, ax=ax, shrink=0.8)
        self.risk_canvas.draw_idle()
        self.risk_label.setText(f"Грубая оценка: окна радиуса {rm.radius} с шагом {self.step_box.value()}, "
                                f"M = 4, без проверки реалистичности; {len(rm.centres)} окон за {rm.seconds:.1f} с.")
        self.tabs.setCurrentIndex(6)

    # ------------------------------------------------------------------ экспорт
    def session_state(self) -> SessionState:
        arrays = {}
        if self.recon is not None:
            arrays = {"z": self.recon.z, "y": self.recon.y}
        results: dict[str, Any] = {}
        if self.check is not None:
            a = self.check.assessment
            results = {"outcome": a.outcome.value, "eta_min": a.eta_min, "eta_max": a.eta_max,
                       "graft_size": a.graft_size, "delta": a.delta, "notes": list(a.notes)}
        return SessionState(source_kind=self.source.kind, data_path=self.source.path,
                            slice_index=self.slice_slider.value(), params=self.current_params(),
                            detail=self.current_detail(), window_level=self.grid.window,
                            mask=None if self.editor.empty else np.array(self.editor.mask),
                            arrays=arrays, results=results)

    def save_session(self, path: str | None = None) -> None:
        path = path or QFileDialog.getSaveFileName(self, "Сохранить сессию", "", "Сессия (*.json)")[0]
        if path:
            json_path, _ = self.session_state().save(path)
            logger.info("сессия сохранена: %s (+ .npz)", json_path)

    def load_session(self, path: str | None = None) -> None:
        path = path or QFileDialog.getOpenFileName(self, "Открыть сессию", "", "Сессия (*.json)")[0]
        if not path:
            return
        st = SessionState.load(path)
        if st.source_kind != self.source.kind or st.data_path != self.source.path:
            self._set_source(open_source(st.data_path or None, self.cfg))
        p = st.params
        self.angles_box.setCurrentIndex(max(0, self.angles_box.findData(p.n_angles)))
        self.beam_box.setCurrentIndex(max(0, self.beam_box.findData(p.beam)))
        self.decoder_box.setCurrentIndex(max(0, self.decoder_box.findData(p.decoder)))
        self.sigma_box.setValue(p.sigma)
        self.eps_box.setValue(p.eps)
        self.iters_box.setValue(p.decoder_iters)
        self.noise_box.setValue(p.n_noise)
        self.seed_box.setValue(p.seed)
        d = st.detail
        self.mode_box.setCurrentIndex(max(0, self.mode_box.findData(d.mode)))
        self.detail_src_box.setCurrentIndex(max(0, self.detail_src_box.findData(d.source)))
        self.projection_box.setCurrentIndex(max(0, self.projection_box.findData(d.projection)))
        self.width_slider_blend.setValue(int(round(d.window_width * 2)))
        self.slice_slider.setValue(st.slice_index)
        self._load_slice(st.slice_index)
        if st.mask is not None:
            self.editor.set_mask(st.mask)
        if st.results:
            self.verdict_label.setText(f"Из сессии: {st.results.get('outcome')}, η ∈ [{st.results.get('eta_min')}, "
                                       f"{st.results.get('eta_max')}]. Для графиков запустите проверку заново.")
        logger.info("сессия загружена: %s", path)

    def figures(self) -> dict[str, Figure]:
        return {"сетка": self.fig, "проверка_A": self.result_fig, "деталь": self.preview_fig,
                "сравнение": self.compare_fig, "похожие_B2": self.similar_fig, "предел_B1": self.limit_fig,
                "карта_риска": self.risk_fig}

    def export_png(self, folder: str | None = None) -> None:
        folder = folder or QFileDialog.getExistingDirectory(self, "Папка для PNG")
        if folder:
            paths = save_figures(self.figures(), folder)
            logger.info("сохранено панелей: %d в %s", len(paths), folder)

    def export_report(self, path: str | None = None) -> None:
        path = path or QFileDialog.getSaveFileName(self, "Отчёт", "", "Word (*.docx)")[0]
        if not path:
            return
        figs = {k: v for k, v in self.figures().items() if k in ("сетка", "проверка_A", "деталь")}
        write_docx(path, self.current_params(), self.current_detail(), self.source.name, self.slice_slider.value(),
                   self.area_label.text().replace("<br>", "; ").replace("<b>", "").replace("</b>", ""),
                   self.check.assessment if self.check else None, figs)

    # ------------------------------------------------------------------ область
    def set_tool(self, tool: Tool) -> None:
        self.chosen_tool = tool
        self.tool_buttons[tool].setChecked(True)
        if not self._toolbar_busy():
            self.selection.set_tool(tool)

    def _toolbar_busy(self) -> bool:
        return bool(getattr(self.toolbar, "mode", ""))

    def _sync_toolbar_mode(self, _checked: bool = False) -> None:
        self.selection.set_tool("none" if self._toolbar_busy() else self.chosen_tool)

    def undo(self) -> None:
        self.editor.undo()
        self.throttle.request()

    def redo(self) -> None:
        self.editor.redo()
        self.throttle.request()

    def clear_mask(self) -> None:
        self.editor.clear()
        self.throttle.request()

    def _on_mask_changed(self, mask: np.ndarray) -> None:
        self.grid.set_mask(mask)
        self._update_area_info()
        self.throttle.request()

    def _update_area_info(self) -> None:
        if not hasattr(self, "area_label"):
            return
        ed = self.editor
        if ed.empty:
            self.area_label.setText("Область не выбрана. Инструменты: R, E, L, M; Shift — вычесть.")
            return
        mm = self.source.pixel_mm
        d_px = ed.equivalent_diameter_px()
        centre = ed.centroid()
        p = self.current_params()
        geom = Geometry(self.editor.shape, p.n_angles, beam=p.beam)  # type: ignore[arg-type]
        res = local_resolution(SimpleNamespace(geometry=geom), centre)  # type: ignore[arg-type]
        text = (f"Площадь R: <b>{ed.area_px}</b> пикс. = <b>{ed.area_mm2(mm):.1f}</b> мм²<br>"
                f"Эквивалентный диаметр: {d_px:.1f} пикс. ({d_px * mm:.1f} мм)<br>"
                f"Центр: строка {centre[0]:.0f}, столбец {centre[1]:.0f}<br>"
                f"Разрешение оператора здесь: ~{res:.1f} пикс.")
        if d_px < res:
            text += ("<br><span style='color:#b35900'>⚠ Область мельче разрешения оператора: "
                     "деталь такого размера данные почти не различают.</span>")
        self.area_label.setText(text)

    def suggest(self) -> None:
        if self.recon is None:
            logger.warning("сначала выполните реконструкцию")
            return
        c = self.cfg.get("candidates", {})
        self.candidates = suggest_candidates(self.recon.z, self.x, int(c.get("k", 5)), int(c.get("radius", 6)))
        self.grid.set_candidates(self.candidates)
        self.cand_list.clear()
        for i, (r, col, rad, score) in enumerate(self.candidates):
            self.cand_list.addItem(f"#{i + 1}: строка {r}, столбец {col}, радиус {rad}, оценка {score:.3g}")
        logger.info("предложено кандидатов: %d (по пикам остатка |z − x|)", len(self.candidates))
        self.throttle.request()

    def choose_candidate(self, index: int) -> None:
        if not 0 <= index < len(self.candidates):
            return
        r, c, rad, _ = self.candidates[index]
        self.editor.apply(disk_region(self.editor.shape, r, c, rad), "replace")
        self.cand_list.setCurrentRow(index)

    def _on_pick(self, event: PickEvent) -> None:
        idx = getattr(event.artist, "candidate_index", None)
        if idx is not None:
            self.choose_candidate(int(idx))

    def _on_hover(self, event: MouseEvent) -> None:
        info = self.grid.pixel_info(event.inaxes, event.xdata, event.ydata)
        if info is None:
            return
        if event.inaxes in self.grid.image_axes and self.recon is not None:
            value = float(self.recon.z[int(round(event.ydata)), int(round(event.xdata))])
            info += f", z ≈ {self._hu_text(value)}"
        self.statusBar().showMessage(info)

    def _hu_text(self, value: float) -> str:
        if self.source.units == "mu":
            return f"{float(to_hu(np.array(value), float(self.cfg.get('mu_water', 0.0192)))):.0f} HU"
        water = float(self.cfg.get("arb_water", 0.2))
        return f"{1000.0 * (value - water) / water:.0f} усл. HU"

    # -------------------------------------------------------------- служебное
    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Обрабатывать события, пока не завершится текущая задача (для тестов и снимков)."""
        app = QApplication.instance()
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            app.processEvents()
            if self.job is None or self.job.done():
                for _ in range(5):
                    app.processEvents()
                return True
            time.sleep(0.01)
        return False

    def save_screenshot(self, path: str | Path) -> Path:
        self.throttle.flush()
        self.canvas.draw()
        QApplication.instance().processEvents()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.grab().save(str(path))
        return path

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self.cancel_job()
        logging.getLogger().removeHandler(self.log_handler)
        super().closeEvent(event)


def _float_box(value: float, lo: float, hi: float, decimals: int, step: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _int_box(value: int, lo: int, hi: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    return box


def ensure_offscreen_fonts() -> None:
    """У offscreen-платформы Qt нет своей базы шрифтов: указываем системный каталог (иначе вместо букв квадраты)."""
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen" or "QT_QPA_FONTDIR" in os.environ:
        return
    candidates = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/dejavu"),
                  Path("/usr/share/fonts"), Path("/System/Library/Fonts")]
    for path in candidates:
        if path.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(path)
            return


def run(cfg: dict[str, Any], source: ArraySource, cache: ResultCache, screenshot: str | None = None) -> int:
    ensure_offscreen_fonts()
    app = QApplication.instance() or QApplication([])
    runner = JobRunner(max_workers=int(cfg["workers"]["threads"]), process_workers=int(cfg["workers"]["processes"]))
    win = MainWindow(cfg, source, cache, runner)
    win.show()
    if screenshot:
        demo_script(win)
        logger.info("снимок окна: %s", win.save_screenshot(screenshot))
        win.close()
        runner.shutdown()
        return 0
    code = app.exec()
    runner.shutdown()
    return code


def demo_script(win: MainWindow) -> None:
    """Сценарий для снимков экрана и smoke-теста: реконструкция, кандидаты, выбор первого."""
    win.wait_idle(120)
    win.suggest()
    if win.candidates:
        win.choose_candidate(0)
    win.start_check()
    win.wait_idle(300)
