"""Запуск окна в offscreen-режиме: синтетика, реконструкция, выделение, кандидаты, снимок."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nullgraft.assess import Outcome  # noqa: E402
from nullgraft_demo.app_qt import MainWindow, ensure_offscreen_fonts  # noqa: E402
from nullgraft_demo.config import load_config  # noqa: E402
from nullgraft_demo.datasource import synthetic_source  # noqa: E402
from nullgraft_demo.tools import rect_region  # noqa: E402
from nullgraft_demo.workers import JobRunner, ResultCache  # noqa: E402


@pytest.fixture(scope="module")
def app():
    ensure_offscreen_fonts()
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(app, tmp_path):
    cfg = load_config()
    cfg["recon"].update({"n_angles": 30, "decoder": "cgls", "decoder_iters": 120, "n_noise": 4})
    runner = JobRunner(max_workers=1)
    win = MainWindow(cfg, synthetic_source(size=64, n_slices=3), ResultCache(tmp_path / "cache"), runner)
    win.show()
    yield win
    win.close()
    runner.shutdown(wait=True)


def test_window_reconstructs_selects_and_screenshots(window: MainWindow, tmp_path) -> None:
    assert window.wait_idle(120)
    assert window.recon is not None and window.recon.z.shape == (64, 64)
    assert "‖A z − y‖" in window.grid._info.get_text()  # noqa: SLF001

    window.selection.select_rect(20, 30, 25, 35)
    assert window.editor.area_px == 121
    assert "121" in window.area_label.text()
    window.selection.select_ellipse(22, 28, 27, 33, mode="subtract")
    assert window.editor.area_px < 121
    window.undo()
    assert window.editor.area_px == 121
    window.redo()
    window.clear_mask()
    assert window.editor.empty

    window.suggest()
    assert window.candidates and window.cand_list.count() == len(window.candidates)
    window.choose_candidate(0)
    assert not window.editor.empty

    window.start_preview()
    assert window.wait_idle(120) and window.preview is not None
    assert window.preview.kernel_ratio < 1e-2

    window.start_check()
    assert window.wait_idle(300) and window.check is not None
    outcome = window.check.assessment.outcome.value
    assert window.badge_label.text() == outcome
    assert outcome in {o.value for o in Outcome}
    texts = [window.stage_list.item(i).text() for i in range(window.stage_list.count())]
    assert texts[0].startswith("✔") and texts[-1].startswith("✔")

    shot = window.save_screenshot(tmp_path / "smoke.png")
    assert shot.exists() and shot.stat().st_size > 10_000

    window.save_session(str(tmp_path / "s.json"))
    mask_before = np.array(window.editor.mask)
    window.clear_mask()
    window.load_session(str(tmp_path / "s.json"))
    assert window.wait_idle(120)
    np.testing.assert_array_equal(window.editor.mask, mask_before)
    window.export_report(str(tmp_path / "r.docx"))
    window.export_png(str(tmp_path / "png"))
    assert (tmp_path / "r.docx").exists() and len(list((tmp_path / "png").glob("*.png"))) == 7


def test_analysis_tabs(window: MainWindow) -> None:
    assert window.wait_idle(120)
    window.editor.set_mask(np.zeros(window.editor.shape, bool) | (np.indices(window.editor.shape)[0] > 100))
    window.choose_candidate(0) if window.candidates else window.editor.apply(
        rect_region(window.editor.shape, 26, 38, 26, 38), "replace")
    window.start_compare()
    assert window.wait_idle(300) and window.compare_table.rowCount() == 2
    window.start_limit()
    assert window.wait_idle(300) and "Архив" in window.limit_label.text()
    window.start_similar()
    assert window.wait_idle(300) and "%" in window.similar_label.text()
    window.step_box.setValue(32)
    window.start_risk()
    assert window.wait_idle(300) and "окон" in window.risk_label.text()


def test_slice_change_uses_cache_and_keeps_mask(window: MainWindow) -> None:
    assert window.wait_idle(120)
    window.editor.apply(rect_region(window.editor.shape, 5, 10, 5, 10))
    window.slice_slider.setValue(1)
    assert window.wait_idle(120) and window.recon is not None
    window.slice_slider.setValue(0)
    assert window.wait_idle(120) and window.recon.from_cache
    assert window.editor.area_px == 36


def test_cancel_and_window_level(window: MainWindow) -> None:
    window.start_reconstruction()
    window.cancel_job()
    assert window.wait_idle(60)
    before = window.grid.window
    window.level_slider.setValue(window.level_slider.value() // 2)
    assert window.grid.window[0] < before[0]
    np.testing.assert_allclose(window.grid._im[window.grid.ax_x].get_clim(),  # noqa: SLF001
                               (window.grid.window[0] - window.grid.window[1] / 2,
                                window.grid.window[0] + window.grid.window[1] / 2), rtol=1e-6)


def test_mpl_fallback_builds_and_selects(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nullgraft_demo.app_mpl import MplApp

    fig = plt.figure(figsize=(12, 7))
    app = MplApp(fig, load_config(), synthetic_source(size=48, n_slices=2), ResultCache(tmp_path), None, sync=True)
    assert app.recon is not None
    app.selection.select_polygon([(10, 10), (30, 10), (30, 30), (10, 30)])
    assert app.editor.area_px > 300
    app.suggest()
    app.choose_candidate(0)
    fig.savefig(tmp_path / "mpl.png")
    plt.close(fig)
