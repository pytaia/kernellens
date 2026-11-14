"""Источник детали и запуск Алгоритма A со стадиями для демо (вычисления — в nullgraft)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation
from skimage.restoration import inpaint_biharmonic
from skimage.transform import resize

from nullgraft.assess import AlwaysRealistic, ArchivePatchRealism, Assessment, AssessmentConfig, DetailAssessor
from nullgraft.graft import Grafter, extract_detail
from nullgraft.nullspace import NullspaceProjector
from nullgraft_demo.compute import ReconResult, get_operator, make_decoder
from nullgraft_demo.state import DetailParams, ReconParams
from nullgraft_demo.workers import CancelToken, ProgressFn

logger = logging.getLogger(__name__)

STAGES = ("consistency", "projection", "blend", "decoder", "classify")
STAGE_TITLES = {
    "consistency": "Согласованность ‖A z − y‖ < ε",
    "projection": "Проекция детали на ядро",
    "blend": "Блендинг и проверка Δ ≤ ε/2",
    "decoder": "Декодер на M реализациях шума",
    "classify": "η_min, η_max и классификация",
}
_STAGE_PROGRESS = {"consistency": 0.05, "projection": 0.35, "blend": 0.4, "decoder": 0.95, "classify": 1.0}
_PROJECTORS: dict[str, NullspaceProjector] = {}


def projector_for(params: ReconParams, shape: tuple[int, int]) -> NullspaceProjector:
    """Проектор с LRU-кэшем на геометрию: повторная проекция той же детали мгновенна."""
    op = get_operator(shape, params.n_angles, params.beam)
    key = op.fingerprint()
    if key not in _PROJECTORS:
        _PROJECTORS[key] = NullspaceProjector(op)
    return _PROJECTORS[key]


def detail_source(
    detail: DetailParams,
    z: NDArray,
    mask: NDArray[np.bool_],
    other_slice: NDArray | None = None,
    offset: tuple[int, int] = (0, 0),
    file_image: NDArray | None = None,
) -> NDArray[np.float32]:
    """Изображение-источник для ``x_det = (source − z)·1_R``.

    * ``mode="remove"`` — содержимое ``z`` в ``R`` заменяется гладким заполнением от границы
      (бигармонический инпейнтинг): проверяется, могла ли деталь отсутствовать;
    * ``source="slice"`` — другой срез архива, сдвинутый на ``offset`` (участок с другого места);
    * ``source="file"`` — изображение из файла, вписанное в ограничивающий прямоугольник ``R``.
    """
    z = np.asarray(z, np.float32)
    if detail.mode == "remove":
        hole = binary_dilation(mask, iterations=1)
        return inpaint_biharmonic(z.astype(np.float64), hole).astype(np.float32)
    if detail.source == "file":
        if file_image is None:
            raise ValueError("файл детали не выбран")
        rows, cols = np.nonzero(mask)
        h, w = int(np.ptp(rows)) + 1, int(np.ptp(cols)) + 1
        return resize(np.asarray(file_image, np.float32), (h, w), anti_aliasing=True, preserve_range=True)
    if other_slice is None:
        raise ValueError("срез-источник не задан")
    src = np.asarray(other_slice, np.float32)
    if offset != (0, 0):
        # участок источника из точки R + offset переносится в R
        rows, cols = np.mgrid[: src.shape[0], : src.shape[1]]
        r2 = np.clip(rows + offset[0], 0, src.shape[0] - 1)
        c2 = np.clip(cols + offset[1], 0, src.shape[1] - 1)
        src = src[r2, c2]
    return src


@dataclass(frozen=True)
class DetailPreview:
    x_det: NDArray[np.float32]
    pinv: NDArray[np.float32]
    x_proj: NDArray[np.float32]
    kernel_ratio: float
    seconds: float


def preview_detail(
    z: NDArray, source: NDArray, mask: NDArray[np.bool_], params: ReconParams, projection: str = "local",
    *, token: CancelToken, progress: ProgressFn,
) -> DetailPreview:
    """``x_det``, ``A⁺A x_det``, ``x_det^proj`` и ``‖A x_det^proj‖ / ‖A x_det‖`` (A x_det — только по лучам через R)."""
    t0 = time.perf_counter()
    progress(0.1, "проекция детали на ядро (LSQR)")
    x_det = extract_detail(z, source, mask)
    projector = projector_for(params, mask.shape)
    res = projector.project_local(x_det, mask) if projection == "local" else projector.project(x_det, support=mask)
    token.check()
    progress(1.0, "превью детали готово")
    return DetailPreview(x_det, res.pinv_component, res.x_proj, res.residual_ratio, time.perf_counter() - t0)


@dataclass(frozen=True)
class CheckResult:
    assessment: Assessment
    z: NDArray[np.float32]
    mask: NDArray[np.bool_]
    seconds: float
    first_stage_seconds: float


def run_check(
    recon: ReconResult,
    source: NDArray,
    mask: NDArray[np.bool_],
    params: ReconParams,
    detail: DetailParams,
    realism: Any | None = None,
    batch_size: int = 5,
    on_stage: Callable[[str, str, dict[str, Any]], None] | None = None,
    *,
    token: CancelToken,
    progress: ProgressFn,
) -> CheckResult:
    """Алгоритм A на полном разрешении; ``on_stage(stage, status, data)`` для чек-листа."""
    t0 = time.perf_counter()
    first: list[float] = []
    op = get_operator(mask.shape, params.n_angles, params.beam)
    decoder = make_decoder(params, op)
    cfg = AssessmentConfig(eps=recon.eps, n_noise=params.n_noise, noise_sigma=params.sigma, seed=params.seed,
                           batch_size=batch_size)
    grafter = Grafter(op, projector_for(params, mask.shape), window=detail.window,  # type: ignore[arg-type]
                      window_width=detail.window_width, projection=detail.projection)  # type: ignore[arg-type]
    assessor = DetailAssessor(op, decoder, cfg, realism=realism or AlwaysRealistic(), grafter=grafter)
    notify = on_stage or (lambda *_: None)
    order = list(STAGES)
    notify(order[0], "running", {})

    def stage(name: str, data: dict[str, Any]) -> None:
        token.check()
        if not first:
            first.append(time.perf_counter() - t0)
        if name == "decoder-noise":
            progress(0.4, f"отклики декодера на шум (один раз): {data['done']} из {data['total']}")
            notify("decoder", "running", {})
            return
        if name == "decoder":
            frac = data["done"] / max(data["total"], 1)
            progress(0.4 + 0.55 * frac, f"декодер: {data['done']} из {data['total']} реализаций")
            notify("decoder", "done" if data["done"] >= data["total"] else "running", data)
            if data["done"] >= data["total"]:
                notify("classify", "running", {})
            return
        progress(_STAGE_PROGRESS[name], STAGE_TITLES[name])
        ok = not (name == "consistency" and not data.get("ok", True)) and not (
            name == "blend" and not data.get("valid", True))
        notify(name, "done" if ok else "failed", data)
        nxt = order.index(name) + 1
        if ok and nxt < len(order) and name != "classify":
            notify(order[nxt], "running", {})

    result = assessor.assess(recon.y, source, mask, z=recon.z, on_stage=stage)
    seconds = time.perf_counter() - t0
    logger.info("проверка детали: %s за %.2f с", result.outcome.value, seconds)
    return CheckResult(result, recon.z, np.asarray(mask, bool), seconds, first[0] if first else seconds)


def fit_realism(slices: list[NDArray], patch_size: int = 32) -> ArchivePatchRealism:
    """Модель реалистичности по срезам архива (без текущего)."""
    return ArchivePatchRealism(patch_size=patch_size).fit(slices, stride=max(4, patch_size // 4))
