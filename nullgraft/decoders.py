"""Декодеры φ: y → x с единым интерфейсом ``reconstruct(y) -> (L, H, W)``.

Подробности: docs/06_decoders.md.
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from nullgraft.operators import RadonOperator

logger = logging.getLogger(__name__)

Window = Literal["ramp", "hann", "shepp-logan", "cosine"]


class Decoder(ABC):
    """Отображение реконструкции; наследники реализуют ``_reconstruct_batch``."""

    n_samples: int = 1

    @property
    def stochastic(self) -> bool:
        return self.n_samples > 1

    def reconstruct(self, y: NDArray) -> NDArray[np.float32]:
        """``φ(y)`` для одной синограммы: ``(L, H, W)``."""
        return self.reconstruct_batch(np.asarray(y)[None])[0]

    def reconstruct_batch(self, ys: NDArray) -> NDArray[np.float32]:
        """``(M, N_θ, n_det) -> (M, L, H, W)``."""
        ys = np.asarray(ys, dtype=np.float32)
        if ys.ndim != 3:
            raise ValueError(f"expected (M, N_theta, n_det), got {ys.shape}")
        out = self._reconstruct_batch(ys)
        return np.asarray(out, dtype=np.float32)

    @abstractmethod
    def _reconstruct_batch(self, ys: NDArray[np.float32]) -> NDArray: ...


class _DeterministicDecoder(Decoder):
    def _reconstruct_batch(self, ys: NDArray[np.float32]) -> NDArray:
        return self._solve(ys)[:, None]

    @abstractmethod
    def _solve(self, ys: NDArray[np.float32]) -> NDArray: ...


def ramp_filter_response(n_det: int, spacing: float, window: Window = "ramp") -> NDArray[np.float64]:
    """АЧХ дискретного рамп-фильтра (Kak & Slaney, 3.61) с окном; дополнение нулями до ``>= 2 n_det``."""
    size = max(64, int(2 ** math.ceil(math.log2(2 * n_det))))
    n = np.concatenate([np.arange(0, size // 2 + 1), np.arange(-size // 2 + 1, 0)])
    h = np.zeros(size)
    h[0] = 1.0 / (4.0 * spacing**2)
    odd = n % 2 == 1
    h[odd] = -1.0 / (np.pi * n[odd] * spacing) ** 2
    response = np.real(np.fft.rfft(h)) * spacing
    f = np.fft.rfftfreq(size)
    if window == "hann":
        response *= 0.5 + 0.5 * np.cos(2 * np.pi * f)
    elif window == "shepp-logan":
        response *= np.sinc(f)
    elif window == "cosine":
        response *= np.cos(np.pi * f)
    elif window != "ramp":
        raise ValueError(f"unknown window {window!r}")
    return response


class FBP(_DeterministicDecoder):
    """Фильтрованная обратная проекция (параллельный и веерный пучок)."""

    def __init__(self, op: RadonOperator, window: Window = "ramp", nonneg: bool = False) -> None:
        self.op = op
        self.window = window
        self.nonneg = nonneg
        g = op.geometry
        n_det = op.n_det
        if g.beam == "parallel":
            self._iso_spacing = g.spacing
            self._preweight = np.ones(n_det)
            # формула обращения интегрирует по [0, π); более длинный обход покрывает его span/π раз
            span = g.angle_step * g.n_angles
            self._scale = g.angle_step * min(1.0, math.pi / span)
        else:
            self._iso_spacing = g.spacing * g.d_so / g.d_sd
            s = (np.arange(n_det) - (n_det - 1) / 2.0) * self._iso_spacing
            self._preweight = g.d_so / np.sqrt(g.d_so**2 + s**2)
            self._scale = 0.5 * g.angle_step
        self._response = ramp_filter_response(n_det, self._iso_spacing, window)

    def filter(self, ys: NDArray) -> NDArray[np.float64]:
        size = 2 * (self._response.size - 1)
        g = np.asarray(ys, np.float64) * self._preweight
        spec = np.fft.rfft(g, n=size, axis=-1) * self._response
        return np.fft.irfft(spec, n=size, axis=-1)[..., : g.shape[-1]]

    def _solve(self, ys: NDArray[np.float32]) -> NDArray:
        x = self.op.fbp_backproject(self.filter(ys)) * self._scale
        return np.maximum(x, 0) if self.nonneg else x


class SART(_DeterministicDecoder):
    """SART в одноблочной (SIRT) форме: ``x ← x + λ C Aᵀ R (y − A x)``."""

    def __init__(self, op: RadonOperator, n_iter: int = 50, relaxation: float = 1.0, nonneg: bool = True) -> None:
        self.op = op
        self.n_iter = n_iter
        self.relaxation = relaxation
        self.nonneg = nonneg
        row = op.forward(np.ones(op.image_shape, np.float32)).astype(np.float64)
        col = op.adjoint(np.ones(op.sino_shape, np.float32)).astype(np.float64)
        self._r = np.divide(1.0, row, out=np.zeros_like(row), where=row > 1e-8)
        self._c = np.divide(1.0, col, out=np.zeros_like(col), where=col > 1e-8)

    def _solve(self, ys: NDArray[np.float32]) -> NDArray:
        x = np.zeros((ys.shape[0], *self.op.image_shape))
        for _ in range(self.n_iter):
            resid = (ys - self.op.forward(x)) * self._r
            x = x + self.relaxation * self._c * self.op.adjoint(resid)
            if self.nonneg:
                x = np.maximum(x, 0.0)
        return x


def _grad(x: NDArray) -> NDArray:
    gx = np.zeros((2, *x.shape))
    gx[0, ..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    gx[1, ..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return gx


def _grad_adjoint(g: NDArray) -> NDArray:
    """``Dᵀ`` для прямых разностей с условием Неймана."""
    out = np.zeros(g.shape[1:])
    out[..., :-1, :] -= g[0, ..., :-1, :]
    out[..., 1:, :] += g[0, ..., :-1, :]
    out[..., :, :-1] -= g[1, ..., :, :-1]
    out[..., :, 1:] += g[1, ..., :, :-1]
    return out


class TVADMM(_DeterministicDecoder):
    """``argmin ½‖Ax − y‖² + λ‖A‖² TV(x)`` методом ADMM; ``λ``, ``ρ`` задаются относительно ``‖A‖²``."""

    def __init__(
        self,
        op: RadonOperator,
        lam: float = 1e-3,
        rho: float = 1e-2,
        n_iter: int = 50,
        cg_iter: int = 8,
        nonneg: bool = True,
    ) -> None:
        self.op = op
        nrm2 = op.norm() ** 2
        self.lam = lam * nrm2
        self.rho = rho * nrm2
        self.n_iter = n_iter
        self.cg_iter = cg_iter
        self.nonneg = nonneg

    def _normal(self, x: NDArray) -> NDArray:
        return self.op.adjoint(self.op.forward(x)).astype(np.float64) + self.rho * _grad_adjoint(_grad(x))

    def _cg(self, rhs: NDArray, x: NDArray) -> NDArray:
        axes = (-2, -1)
        r = rhs - self._normal(x)
        p = r.copy()
        rs = np.sum(r * r, axis=axes, keepdims=True)
        for _ in range(self.cg_iter):
            ap = self._normal(p)
            denom = np.sum(p * ap, axis=axes, keepdims=True)
            alpha = np.divide(rs, denom, out=np.zeros_like(rs), where=denom > 0)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = np.sum(r * r, axis=axes, keepdims=True)
            beta = np.divide(rs_new, rs, out=np.zeros_like(rs), where=rs > 0)
            p = r + beta * p
            rs = rs_new
        return x

    def _solve(self, ys: NDArray[np.float32]) -> NDArray:
        aty = self.op.adjoint(ys).astype(np.float64)
        x = np.zeros_like(aty)
        z = _grad(x)
        u = np.zeros_like(z)
        thresh = self.lam / self.rho
        for _ in range(self.n_iter):
            x = self._cg(aty + self.rho * _grad_adjoint(z - u), x)
            if self.nonneg:
                x = np.maximum(x, 0.0)
            dx = _grad(x)
            v = dx + u
            mag = np.sqrt(np.sum(v**2, axis=0, keepdims=True))
            z = np.maximum(1.0 - np.divide(thresh, mag, out=np.full_like(mag, np.inf), where=mag > 0), 0.0) * v
            u = u + dx - z
        return x


class PerturbedDecoder(Decoder):
    """Игрушечный стохастический декодер: ``L`` семплов с гладким шумом, воспроизводимых по ``y``."""

    def __init__(self, base: Decoder, n_samples: int = 8, sigma: float = 0.01, smooth: float = 1.5,
                 seed: int = 0) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.base = base
        self.n_samples = n_samples
        self.sigma = sigma
        self.smooth = smooth
        self.seed = seed

    def _reconstruct_batch(self, ys: NDArray[np.float32]) -> NDArray:
        base = self.base.reconstruct_batch(ys)[:, :1]
        out = np.repeat(base, self.n_samples, axis=1).astype(np.float64)
        for m, y in enumerate(ys):
            digest = hashlib.sha1(np.ascontiguousarray(y).tobytes()).digest()
            rng = np.random.default_rng([self.seed, int.from_bytes(digest[:8], "little")])
            noise = rng.standard_normal(out.shape[1:])
            out[m] += self.sigma * gaussian_filter(noise, sigma=(0, self.smooth, self.smooth))
        return out


class TorchPostProcessingDecoder(Decoder):
    """Заглушка нейросетевого декодера ``φ(y) = net(init(y))``; при ``n_samples > 1`` — MC dropout."""

    def __init__(
        self,
        model_factory: Callable[[], Any],
        weights_path: str | Path | None,
        init: Decoder,
        n_samples: int = 1,
        device: str | None = None,
        precision: Literal["fp32", "fp16", "bf16"] = "fp32",
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("TorchPostProcessingDecoder requires PyTorch (pip install nullgraft[torch])") from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_factory()
        if weights_path is not None:
            state = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.init = init
        self.n_samples = n_samples
        self.precision = precision
        self.batch_size = batch_size
        self.seed = seed
        if n_samples > 1:
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                    module.train()

    def _reconstruct_batch(self, ys: NDArray[np.float32]) -> NDArray:
        torch = self._torch
        z0 = self.init.reconstruct_batch(ys)[:, 0]
        inputs = np.repeat(z0[:, None], self.n_samples, axis=1).reshape(-1, 1, *z0.shape[-2:])
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        use_amp = dtype is not None and self.device.startswith("cuda")
        torch.manual_seed(self.seed)
        outs = []
        with torch.inference_mode():
            for start in range(0, inputs.shape[0], self.batch_size):
                chunk = torch.from_numpy(inputs[start : start + self.batch_size]).to(self.device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    pred = self.model(chunk)
                outs.append(pred.float().cpu().numpy())
        out = np.concatenate(outs)[:, 0]
        return out.reshape(z0.shape[0], self.n_samples, *z0.shape[-2:])


def build_decoder(cfg: Mapping[str, Any], op: RadonOperator) -> Decoder:
    """Декодер по конфигу, например ``{"type": "fbp", "window": "hann"}``."""
    params = dict(cfg)
    kind = str(params.pop("type")).lower()
    if kind == "fbp":
        return FBP(op, **params)
    if kind == "sart":
        return SART(op, **params)
    if kind in ("tv", "tv-admm", "tv_admm"):
        return TVADMM(op, **params)
    if kind == "perturbed":
        base = build_decoder(params.pop("base"), op)
        return PerturbedDecoder(base, **params)
    raise ValueError(f"unknown decoder type {kind!r} (neural decoders are built in code: TorchPostProcessingDecoder)")
