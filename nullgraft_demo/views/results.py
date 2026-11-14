"""Отрисовка результата Алгоритма A: изображения, интервал η, гистограмма расстояний; бейджи исходов."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nullgraft.assess import Assessment, Outcome

BADGES: dict[str, tuple[str, str]] = {
    Outcome.LIKELY_HALLUCINATION_RISK.value: (
        "#d9480f", "Данные не различают z и правдоподобную альтернативу x'', а декодер выдаёт z в обоих "
                   "случаях: деталь в R данными не подтверждена."),
    Outcome.DETAIL_VISIBLE.value: (
        "#2b8a3e", "Подменить деталь незаметно для данных нельзя: содержимое R определяется измерениями "
                   "(для этого источника детали)."),
    Outcome.UNINFORMATIVE.value: (
        "#868e96", "Альтернатива x'' не прошла проверку реалистичности: тест ничего не утверждает."),
    Outcome.INCONSISTENT.value: (
        "#7048e8", "Реконструкция не объясняет данные (‖A z − y‖ ≥ ε): теорема неприменима, проверка остановлена."),
    Outcome.BLEND_INVALID.value: (
        "#e67700", "Вклейка заметна в данных (Δ > ε/2): тест некорректен. Увеличьте область или измените "
                   "ширину перехода."),
    Outcome.INCONCLUSIVE.value: ("#495057", "Однозначного вывода нет (см. пояснения в логе)."),
}

DISCLAIMER = (
    "Метод даёт границу возможного, а не диагноз. Вывод зависит от выбранной нормы, области R, источника "
    "детали и проверки реалистичности x''. Предполагаются линейный оператор A и аддитивный ограниченный шум.")


def badge(outcome: str) -> tuple[str, str]:
    return BADGES.get(outcome, ("#495057", ""))


def _zoom_box(mask: NDArray[np.bool_], margin: int = 6) -> tuple[slice, slice]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return slice(None), slice(None)
    h, w = mask.shape
    return (slice(max(rows.min() - margin, 0), min(rows.max() + margin + 1, h)),
            slice(max(cols.min() - margin, 0), min(cols.max() + margin + 1, w)))


def draw_result(fig: Figure, res: Assessment, z: NDArray, mask: NDArray[np.bool_]) -> None:
    """2×3: x'' | среднее φ(y'') | СКО по шумам; φ(y'') − z в R | интервал η | гистограмма расстояний."""
    fig.clear()
    axes = fig.subplots(2, 3)
    ex = res.extras
    box = _zoom_box(mask)
    panels = [
        (axes[0, 0], ex.get("x_graft"), "x'' (z с деталью)", "gray"),
        (axes[0, 1], ex.get("out_mean"), "среднее φ(y'')", "gray"),
        (axes[0, 2], ex.get("out_std"), "СКО по шумам", "magma"),
    ]
    for ax, img, title, cmap in panels:
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is None:
            ax.text(0.5, 0.5, "нет данных", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue
        ax.imshow(np.asarray(img)[box], cmap=cmap, interpolation="nearest")
        ax.contour(mask[box].astype(float), levels=[0.5], colors=["yellow"], linewidths=0.8)
    ax = axes[1, 0]
    ax.set_title("φ(y'') − z в R", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if "out_mean" in ex:
        diff = (np.asarray(ex["out_mean"]) - np.asarray(z)) * mask
        lim = float(np.max(np.abs(diff[box]))) or 1.0
        ax.imshow(diff[box], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    draw_interval(axes[1, 1], res)
    draw_histogram(axes[1, 2], res)


def draw_interval(ax, res: Assessment) -> None:
    """Интервал [η_min, η_max] на числовой оси с отметкой ‖x_det‖/2 (по статье: ‖z − x''‖/2)."""
    ax.set_title("интервал η", fontsize=8)
    ax.set_yticks([])
    if not np.isfinite(res.eta_min):
        ax.text(0.5, 0.5, "не вычислялся", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return
    half = res.graft_size / 2
    ok = res.interval_nonempty
    color = "#d9480f" if ok else "#868e96"
    lo, hi = res.eta_min, res.eta_max
    ax.hlines(0, min(lo, hi), max(lo, hi), colors=color, lw=6, alpha=0.7)
    ax.plot([lo], [0], "o", color="black", ms=5)
    ax.plot([hi], [0], "s", color="black", ms=5)
    ax.axvline(half, color="#1971c2", ls="--", lw=1)
    ax.annotate("η_min", (lo, 0), (0, 10), textcoords="offset points", ha="center", fontsize=7)
    ax.annotate("η_max", (hi, 0), (0, -16), textcoords="offset points", ha="center", fontsize=7)
    ax.annotate("‖z−x''‖/2", (half, 0.6), ha="center", fontsize=7, color="#1971c2")
    span = max(lo, hi, half) * 1.15 or 1.0
    ax.set_xlim(0, span)
    ax.set_ylim(-1, 1)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("η_min ≤ η_max: " + ("да" if lo <= hi else "нет"), fontsize=7, color=color)


def draw_histogram(ax, res: Assessment) -> None:
    ax.set_title("расстояния по реализациям", fontsize=8)
    if not res.d_to_z:
        ax.text(0.5, 0.5, "нет данных", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_yticks([])
        return
    dz, dg = np.asarray(res.d_to_z), np.asarray(res.d_to_graft)
    bins = np.linspace(0, max(dz.max(), dg.max()) * 1.05 or 1.0, 15)
    ax.hist(dz, bins=bins, alpha=0.7, label="d(φ(y''), z)", color="#d9480f")
    ax.hist(dg, bins=bins, alpha=0.7, label="d(φ(y''), x'')", color="#1971c2")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)


def draw_preview(fig: Figure, x_det: NDArray, pinv: NDArray, x_proj: NDArray, mask: NDArray[np.bool_],
                 ratio: float) -> None:
    fig.clear()
    axes = fig.subplots(1, 3)
    box = _zoom_box(mask, margin=10)
    for ax, img, title in zip(axes, (x_det, pinv, x_proj), ("x_det", "A⁺A x_det (видимая часть)", "x_det^proj (невидимая)"),
                              strict=True):
        arr = np.asarray(img)[box]
        lim = float(np.max(np.abs(arr))) or 1.0
        ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.contour(mask[box].astype(float), levels=[0.5], colors=["black"], linewidths=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"‖A x_det^proj‖ / ‖A x_det‖ = {ratio:.2e}", fontsize=8)
