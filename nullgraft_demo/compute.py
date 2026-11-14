"""Вычисления для демо поверх nullgraft: оператор, реконструкция с кэшем, кандидаты областей."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, maximum_filter

from nullgraft.api import decoder_from_name, decoder_signature, default_eps, detector_residual
from nullgraft.assess import select_point
from nullgraft.decoders import Decoder
from nullgraft.operators import RadonOperator, make_operator
from nullgraft_demo.state import ReconParams
from nullgraft_demo.workers import CancelToken, ProgressFn, ResultCache, cache_key

logger = logging.getLogger(__name__)

_OPS: dict[tuple, RadonOperator] = {}
_OPS_LOCK = threading.Lock()


def get_operator(shape: tuple[int, int], n_angles: int, beam: str) -> RadonOperator:
    """Оператор на геометрию; один экземпляр на процесс (матрица строится один раз)."""
    key = (tuple(shape), int(n_angles), beam)
    with _OPS_LOCK:
        if key not in _OPS:
            _OPS[key] = make_operator(shape, n_angles=n_angles, beam=beam)  # type: ignore[arg-type]
        return _OPS[key]


def make_decoder(params: ReconParams, op: RadonOperator) -> Decoder:
    if params.decoder == "neural":
        raise NotImplementedError("нейросетевой декодер подключается на этапе 6 (нужны PyTorch и файл весов)")
    return decoder_from_name(params.decoder, op, n_iter=params.decoder_iters)


@dataclass(frozen=True)
class ReconResult:
    y: NDArray[np.float32]
    z: NDArray[np.float32]
    det_residual: NDArray[np.float32]
    eps: float
    consistency: float
    key: str
    seconds: float
    from_cache: bool

    @property
    def consistent(self) -> bool:
        return self.consistency < self.eps


def reconstruct(
    x: NDArray[np.float32],
    params: ReconParams,
    cache: ResultCache | None,
    *,
    token: CancelToken,
    progress: ProgressFn,
) -> ReconResult:
    """``y = A x + e`` (seed), ``z = φ(y)``, ``|A z − y|``; кэш по (срез, геометрия, σ, seed, декодер)."""
    t0 = time.perf_counter()
    op = get_operator(x.shape, params.n_angles, params.beam)
    decoder = make_decoder(params, op)
    eps = params.eps or default_eps(op, params.sigma, params.eps_factor)
    key = cache_key("recon-v1", x, op.fingerprint(), params.sigma, params.seed, decoder_signature(decoder))
    if cache is not None and (hit := cache.get(key)) is not None:
        progress(1.0, "реконструкция из кэша")
        return ReconResult(hit["y"], hit["z"], hit["det_residual"], float(eps),
                           float(np.linalg.norm(op.forward(hit["z"]) - hit["y"])), key,
                           time.perf_counter() - t0, True)
    progress(0.05, "проекция A x")
    rng = np.random.default_rng(params.seed)
    y = (op.forward(x) + params.sigma * rng.standard_normal(op.sino_shape)).astype(np.float32)
    token.check()
    progress(0.15, f"декодер {params.decoder}")
    z = select_point(decoder.reconstruct(y)).astype(np.float32)
    token.check()
    progress(0.9, "невязка |A z − y|")
    det = detector_residual(op, z, y)
    if cache is not None:
        cache.put(key, {"y": y, "z": z, "det_residual": det})
    consistency = float(np.linalg.norm(op.forward(z) - y))
    progress(1.0, "реконструкция готова")
    return ReconResult(y, z, det, float(eps), consistency, key, time.perf_counter() - t0, False)


def suggest_candidates(
    z: NDArray, x: NDArray | None = None, k: int = 5, radius: int = 6
) -> list[tuple[int, int, int, float]]:
    """Кандидаты областей ``(row, col, radius, score)`` по пикам сглаженного остатка ``|z − x|``.

    Без эталона — по пикам высокочастотной энергии ``z``.
    """
    z = np.asarray(z, np.float64)
    if x is not None:
        score = gaussian_filter(np.abs(z - np.asarray(x, np.float64)), radius / 2)
    else:
        score = gaussian_filter((z - gaussian_filter(z, 2.0)) ** 2, radius / 2)
    m = radius + 1
    score[:m], score[-m:], score[:, :m], score[:, -m:] = 0, 0, 0, 0
    peaks = (score == maximum_filter(score, size=2 * radius + 1)) & (score > 0)
    rows, cols = np.nonzero(peaks)
    order = np.argsort(score[rows, cols])[::-1][:k]
    return [(int(rows[i]), int(cols[i]), int(radius), float(score[rows[i], cols[i]])) for i in order]
