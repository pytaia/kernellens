"""Отрисовка этапов 6–9: сравнение декодеров, похожие артефакты (B2), предел задачи (B1)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nullgraft_demo.views.results import _zoom_box, badge, draw_interval


def draw_compare(fig: Figure, rows: Sequence, mask: NDArray[np.bool_]) -> None:
    """Столбец на декодер: z в R (заголовок цвета бейджа) и интервал η."""
    fig.clear()
    if not rows:
        return
    axes = fig.subplots(2, len(rows), squeeze=False)
    box = _zoom_box(mask, margin=10)
    for j, row in enumerate(rows):
        ax = axes[0, j]
        ax.imshow(row.recon.z[box], cmap="gray", interpolation="nearest")
        ax.contour(mask[box].astype(float), levels=[0.5], colors=["yellow"], linewidths=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        outcome = row.check.assessment.outcome.value if row.check else "ошибка"
        color, _ = badge(outcome)
        ax.set_title(f"{row.decoder}\n{outcome}", fontsize=7, color=color, fontweight="bold")
        if row.check:
            draw_interval(axes[1, j], row.check.assessment)
        else:
            axes[1, j].text(0.5, 0.5, row.error, ha="center", va="center", fontsize=7, wrap=True,
                            transform=axes[1, j].transAxes)
            axes[1, j].axis("off")


def draw_similar(fig: Figure, items: Sequence) -> dict[Axes, int]:
    """Строка на совпадение: реконструкция, остаток, эталон; возвращает оси → индекс для клика."""
    fig.clear()
    lookup: dict[Axes, int] = {}
    if not items:
        return lookup
    axes = fig.subplots(len(items), 3, squeeze=False)
    for i, it in enumerate(items):
        for j, (img, cmap) in enumerate(((it.recon, "gray"), (it.residual, "RdBu_r"), (it.truth, "gray"))):
            ax = axes[i, j]
            if cmap == "RdBu_r":
                lim = float(np.max(np.abs(img))) or 1.0
                ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
            else:
                ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            lookup[ax] = i
        kind = "структура" if it.is_structure else "фон"
        axes[i, 0].set_ylabel(f"срез {it.slice_index + 1}\n({it.row},{it.col})", fontsize=6)
        axes[i, 1].set_title(f"d = {it.distance:.3g}, NCC = {it.ncc:.2f}", fontsize=6)
        axes[i, 2].set_title(kind, fontsize=6, color="#2b8a3e" if it.is_structure else "#868e96")
    axes[0, 0].set_title("реконструкция", fontsize=7)
    axes[0, 2].set_title(f"эталон: {axes[0, 2].get_title()}", fontsize=6)
    return lookup


def draw_limit(fig: Figure, res) -> None:
    """diam(F^N) и diam/2 по N_θ (текущая точка выделена), пара x, x' и кривая сходимости."""
    fig.clear()
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :])
    ang = np.asarray(res.angles)
    d = np.asarray(res.diameters)
    ax.plot(ang, d, "o-", label="diam(F_y^N) — оценка снизу")
    ax.plot(ang, d / 2, "s--", label="diam/2 — нижняя граница ошибки любого декодера")
    if res.current in res.angles:
        k = res.angles.index(res.current)
        ax.plot([res.current], [d[k]], "o", ms=12, mfc="none", mec="#d9480f", mew=2, label="текущие настройки")
    for a, dv, n in zip(ang, d, res.sizes, strict=True):
        ax.annotate(f"|F|={n}", (a, dv), (0, 6), textcoords="offset points", ha="center", fontsize=6)
    ax.set_xscale("log")
    ax.set_xticks(ang)
    ax.set_xticklabels([str(a) for a in ang], fontsize=7)
    ax.set_xlabel("число проекций N_θ", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)
    for j, title in enumerate(("x", "x'")):
        axp = fig.add_subplot(gs[1, j])
        axp.set_xticks([])
        axp.set_yticks([])
        if res.pair is None:
            axp.text(0.5, 0.5, "F_y^N: < 2 элементов", ha="center", va="center", fontsize=7, transform=axp.transAxes)
            continue
        axp.imshow(res.pair[j], cmap="gray", interpolation="nearest")
        meta = res.pair_meta[j]
        axp.set_title(f"{title}: срез {meta[0] + 1}, ({meta[1]},{meta[2]})", fontsize=6)
    axc = fig.add_subplot(gs[1, 2])
    if res.curve.size:
        axc.plot(res.fractions * 100, res.curve[:, 0], "o-", ms=3)
    axc.set_title("сходимость (оценка сходится снизу)", fontsize=6)
    axc.set_xlabel("% патчей архива", fontsize=6)
    axc.tick_params(labelsize=6)
