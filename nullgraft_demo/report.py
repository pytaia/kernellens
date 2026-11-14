"""Экспорт: отчёт .docx (python-docx), сохранение панелей в PNG."""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from nullgraft_demo.views.results import DISCLAIMER, badge

logger = logging.getLogger(__name__)


def save_figures(figures: Mapping[str, Figure], folder: str | Path, dpi: int = 150) -> list[Path]:
    """Каждая фигура — в ``<folder>/<имя>.png``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, fig in figures.items():
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=dpi)
        out.append(path)
    return out


def _png(fig: Figure, dpi: int = 130) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return buf


def write_docx(
    path: str | Path,
    params: Any,
    detail: Any,
    source_name: str,
    slice_index: int,
    area_text: str,
    assessment: Any | None,
    figures: Mapping[str, Figure],
    extra: Mapping[str, str] | None = None,
) -> Path:
    """Отчёт: параметры, область, исход с расшифровкой, η_min/η_max, изображения и блок оговорок."""
    from docx import Document
    from docx.shared import Cm, RGBColor

    doc = Document()
    doc.add_heading("nullgraft: проверка детали реконструкции", level=1)
    doc.add_paragraph(f"Дата: {datetime.now():%Y-%m-%d %H:%M}. Данные: {source_name}, срез {slice_index + 1}.")

    doc.add_heading("Параметры", level=2)
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for key, value in {**asdict(params), **{f"деталь.{k}": v for k, v in asdict(detail).items()}}.items():
        row = table.add_row().cells
        row[0].text, row[1].text = str(key), str(value)

    doc.add_heading("Область R", level=2)
    doc.add_paragraph(area_text or "не выбрана")

    doc.add_heading("Результат Алгоритма A", level=2)
    if assessment is None:
        doc.add_paragraph("Проверка не запускалась.")
    else:
        color, text = badge(assessment.outcome.value)
        p = doc.add_paragraph()
        run = p.add_run(assessment.outcome.value)
        run.bold = True
        run.font.color.rgb = RGBColor.from_string(color.lstrip("#").upper())
        doc.add_paragraph(text)
        nums = doc.add_table(rows=0, cols=2)
        nums.style = "Light Grid Accent 1"
        for k, v in (("η_min", assessment.eta_min), ("η_max", assessment.eta_max),
                     ("‖z − x''‖", assessment.graft_size), ("Δ", assessment.delta), ("предел Δ", assessment.delta_limit),
                     ("‖A z − y‖", assessment.consistency_residual), ("ε", assessment.eps)):
            r = nums.add_row().cells
            r[0].text, r[1].text = k, "—" if not np.isfinite(v) else f"{v:.4g}"
        for note in assessment.notes:
            doc.add_paragraph(note, style="List Bullet")
    for k, v in (extra or {}).items():
        doc.add_heading(k, level=2)
        doc.add_paragraph(v)

    doc.add_heading("Изображения", level=2)
    for name, fig in figures.items():
        doc.add_paragraph(name)
        doc.add_picture(_png(fig), width=Cm(16))

    doc.add_heading("Оговорки", level=2)
    doc.add_paragraph(DISCLAIMER)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    logger.info("отчёт сохранён: %s", path)
    return path
