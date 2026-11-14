"""Прямой оператор КТ: 2D-преобразование Радона y = A x (параллельный и веерный пучок).

Подробности: docs/01_operator.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

logger = logging.getLogger(__name__)

Beam = Literal["parallel", "fan"]
BackendName = Literal["auto", "pixel", "astra", "skimage"]

# float64 + int32 индекс: матрицу float32 scipy перекастовывает на каждом умножении (в 3.4 раза медленнее)
_BYTES_PER_NNZ = 12
_SUPPORT_FRACTION_FOR_SPARSE_PATH = 0.25


@dataclass(frozen=True)
class Geometry:
    """Геометрия съёмки; длины в пикселях."""

    shape: tuple[int, int]
    n_angles: int
    beam: Beam = "parallel"
    n_det: int | None = None
    det_spacing: float | None = None
    angular_range: float | None = None
    angle_offset: float = 0.0
    source_origin: float | None = None
    origin_det: float | None = None
    pixel_size_mm: float = 1.0

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError(f"shape must be (H, W) with H, W >= 2, got {self.shape}")
        if self.n_angles < 1:
            raise ValueError("n_angles must be >= 1")
        if self.beam not in ("parallel", "fan"):
            raise ValueError(f"unknown beam {self.beam!r}")
        if self.beam == "fan" and self.d_so <= self.fov_radius:
            raise ValueError("source_origin must exceed the image half-diagonal")

    @property
    def fov_radius(self) -> float:
        h, w = self.shape
        return 0.5 * math.hypot(h, w)

    @property
    def d_so(self) -> float:
        return float(self.source_origin) if self.source_origin is not None else 2.0 * max(self.shape)

    @property
    def d_sd(self) -> float:
        d_od = float(self.origin_det) if self.origin_det is not None else self.d_so
        return self.d_so + d_od

    @property
    def spacing(self) -> float:
        if self.det_spacing is not None:
            return float(self.det_spacing)
        return 1.0 if self.beam == "parallel" else self.d_sd / self.d_so

    @property
    def detector_count(self) -> int:
        if self.n_det is not None:
            return int(self.n_det)
        r = self.fov_radius
        if self.beam == "parallel":
            half = r
        else:
            half = self.d_sd * r / math.sqrt(self.d_so**2 - r**2)
        return int(math.ceil(2.0 * half / self.spacing)) + 2

    @property
    def angles(self) -> NDArray[np.float64]:
        span = self.angular_range
        if span is None:
            span = math.pi if self.beam == "parallel" else 2.0 * math.pi
        return self.angle_offset + span * np.arange(self.n_angles, dtype=np.float64) / self.n_angles

    @property
    def angle_step(self) -> float:
        a = self.angles
        return float(a[1] - a[0]) if a.size > 1 else math.pi

    @property
    def sino_shape(self) -> tuple[int, int]:
        return (self.n_angles, self.detector_count)

    def fingerprint(self) -> str:
        """Хэш геометрии для ключей кэша."""
        payload = asdict(self) | {
            "resolved": [self.spacing, self.detector_count, self.d_so, self.d_sd],
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class _Backend(Protocol):
    name: str
    exact_adjoint: bool
    n_det: int

    def forward(self, x: NDArray[np.float64], idx: NDArray[np.int64] | None) -> NDArray[np.float64]: ...

    def adjoint(self, s: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def fbp_backproject(self, q: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class _AngleInterp:
    k0: NDArray[np.int64]
    w0: NDArray[np.float64]
    w1: NDArray[np.float64]
    scale: NDArray[np.float64]
    fbp_weight: NDArray[np.float64]


def _embed(x: NDArray[np.float64], idx: NDArray[np.int64], n_pix: int) -> NDArray[np.float64]:
    full = np.zeros((x.shape[0], n_pix))
    full[:, idx] = x
    return full


class PixelDrivenBackend:
    """Pixel-driven проектор без матрицы, с точным сопряжённым оператором."""

    name = "pixel"
    exact_adjoint = True

    def __init__(self, geometry: Geometry, matrix_budget_mb: float = 512.0) -> None:
        self.g = geometry
        h, w = geometry.shape
        xs = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
        ys = (h - 1) / 2.0 - np.arange(h, dtype=np.float64)
        px, py = np.meshgrid(xs, ys)
        self._px = px.ravel()
        self._py = py.ravel()
        self.n_pix = h * w
        self.n_det = geometry.detector_count
        angles = geometry.angles
        self._cos = np.cos(angles)
        self._sin = np.sin(angles)
        self._matrix: sparse.csr_matrix | None = None
        nnz_estimate = 2 * self.n_pix * geometry.n_angles
        if nnz_estimate * _BYTES_PER_NNZ <= matrix_budget_mb * 2**20:
            self._matrix = self._build_matrix()
            logger.debug("cached system matrix: nnz=%d", self._matrix.nnz)

    @property
    def has_matrix(self) -> bool:
        return self._matrix is not None

    def _interp(self, a: int, idx: NDArray[np.int64] | None) -> _AngleInterp:
        g = self.g
        px = self._px if idx is None else self._px[idx]
        py = self._py if idx is None else self._py[idx]
        c, s = float(self._cos[a]), float(self._sin[a])
        du = g.spacing
        if g.beam == "parallel":
            t = px * c + py * s
            scale = np.full_like(t, 1.0 / du)
            fbp_weight = np.ones_like(t)
        else:
            depth = g.d_so + px * c + py * s
            lateral = -px * s + py * c
            t = lateral * g.d_sd / depth
            magnification = g.d_sd / depth
            cos_gamma = depth / np.hypot(depth, lateral)
            scale = magnification / (du * cos_gamma)
            fbp_weight = (g.d_so / depth) ** 2
        f = t / du + (self.n_det - 1) / 2.0
        k0 = np.floor(f).astype(np.int64)
        w1 = f - k0
        return _AngleInterp(k0=k0, w0=1.0 - w1, w1=w1, scale=scale, fbp_weight=fbp_weight)

    def _taps(
        self, it: _AngleInterp, weight: NDArray[np.float64]
    ) -> Iterator[tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.float64]]]:
        for k, wk in ((it.k0, it.w0), (it.k0 + 1, it.w1)):
            valid = (k >= 0) & (k < self.n_det)
            yield valid, k[valid], (wk * weight)[valid]

    def _build_matrix(self) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        pix = np.arange(self.n_pix, dtype=np.int64)
        for a in range(self.g.n_angles):
            it = self._interp(a, None)
            for valid, k, wv in self._taps(it, it.scale):
                rows.append(a * self.n_det + k)
                cols.append(pix[valid])
                vals.append(wv)
        shape = (self.g.n_angles * self.n_det, self.n_pix)
        mat = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=shape,
        ).tocsr()
        mat.sum_duplicates()
        return mat

    def forward(self, x: NDArray[np.float64], idx: NDArray[np.int64] | None) -> NDArray[np.float64]:
        """``x``: (B, n_pix) или (B, len(idx)) — значения только в пикселях ``idx``."""
        batch = x.shape[0]
        use_support_path = idx is not None and idx.size < _SUPPORT_FRACTION_FOR_SPARSE_PATH * self.n_pix
        if idx is not None and not use_support_path:
            x, idx = _embed(x, idx, self.n_pix), None
        if self._matrix is not None and idx is None:
            return np.asarray((self._matrix @ x.T).T)
        out = np.zeros((batch, self.g.n_angles, self.n_det))
        offsets = (np.arange(batch, dtype=np.int64) * self.n_det)[:, None]
        for a in range(self.g.n_angles):
            it = self._interp(a, idx)
            for valid, k, wv in self._taps(it, it.scale):
                vals = x[:, valid] * wv
                flat = (offsets + k[None, :]).ravel()
                out[:, a, :] += np.bincount(flat, weights=vals.ravel(), minlength=batch * self.n_det).reshape(
                    batch, self.n_det
                )
        return out.reshape(batch, -1)

    def _gather(self, s: NDArray[np.float64], use_fbp_weight: bool) -> NDArray[np.float64]:
        batch = s.shape[0]
        sino = s.reshape(batch, self.g.n_angles, self.n_det)
        out = np.zeros((batch, self.n_pix))
        for a in range(self.g.n_angles):
            it = self._interp(a, None)
            weight = it.fbp_weight if use_fbp_weight else it.scale
            for valid, k, wv in self._taps(it, weight):
                out[:, valid] += sino[:, a, k] * wv
        return out

    def adjoint(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._matrix is not None:
            return np.asarray((self._matrix.T @ s.T).T)
        return self._gather(s, use_fbp_weight=False)

    def adjoint_matrix_free(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Эталонный ``A^T`` без кэшированной матрицы (для тестов)."""
        return self._gather(s, use_fbp_weight=False)

    def fbp_backproject(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Обратная проекция FBP с весами ``1`` (parallel) или ``(D_so/L)^2`` (fan)."""
        return self._gather(q, use_fbp_weight=True)


class SkimageBackend:
    """Бэкенд ``skimage.transform.radon``; сопряжённый оператор лишь приближённый."""

    name = "skimage"
    exact_adjoint = False

    def __init__(self, geometry: Geometry) -> None:
        h, w = geometry.shape
        if geometry.beam != "parallel" or h != w:
            raise ValueError("skimage backend supports parallel-beam square images only")
        self.g = geometry
        self.n_det = int(math.ceil(math.sqrt(2) * h))
        if geometry.n_det is not None and geometry.n_det != self.n_det:
            raise ValueError(f"skimage backend requires n_det={self.n_det}")
        self._theta_deg = np.rad2deg(geometry.angles)
        self.n_pix = h * w

    def forward(self, x: NDArray[np.float64], idx: NDArray[np.int64] | None) -> NDArray[np.float64]:
        from skimage.transform import radon

        h, w = self.g.shape
        if idx is not None:
            x = _embed(x, idx, self.n_pix)
        sinos = [radon(xi.reshape(h, w), theta=self._theta_deg, circle=False, preserve_range=True).T for xi in x]
        return np.stack(sinos).reshape(x.shape[0], -1)

    def adjoint(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        from skimage.transform import iradon

        h, _ = self.g.shape
        n_a = self.g.n_angles
        scale = 2.0 * n_a / math.pi
        out = [
            iradon(si.reshape(n_a, self.n_det).T, theta=self._theta_deg, filter_name=None, circle=False, output_size=h)
            * scale
            for si in s
        ]
        return np.stack(out).reshape(s.shape[0], -1)

    def fbp_backproject(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.adjoint(q)


class AstraBackend:
    """Бэкенд ASTRA Toolbox: CUDA при наличии GPU, иначе CPU-проекторы."""

    name = "astra"

    def __init__(self, geometry: Geometry) -> None:
        import astra

        self._astra = astra
        self.g = geometry
        h, w = geometry.shape
        self.n_pix = h * w
        self.n_det = geometry.detector_count
        vol = astra.create_vol_geom(h, w)
        if geometry.beam == "parallel":
            proj = astra.create_proj_geom("parallel", geometry.spacing, self.n_det, geometry.angles)
            cpu_kind = "linear"
        else:
            proj = astra.create_proj_geom(
                "fanflat", geometry.spacing, self.n_det, geometry.angles, geometry.d_so, geometry.d_sd - geometry.d_so
            )
            cpu_kind = "line_fanflat"
        use_cuda = bool(astra.astra.use_cuda())
        # обратная проекция CUDA — не побитно точное транспонирование
        self.exact_adjoint = not use_cuda
        self._proj_id = astra.create_projector("cuda" if use_cuda else cpu_kind, proj, vol)
        logger.info("ASTRA backend, cuda=%s", use_cuda)

    def forward(self, x: NDArray[np.float64], idx: NDArray[np.int64] | None) -> NDArray[np.float64]:
        h, w = self.g.shape
        if idx is not None:
            x = _embed(x, idx, self.n_pix)
        out = []
        for xi in x:
            sid, sino = self._astra.create_sino(xi.reshape(h, w).astype(np.float32), self._proj_id)
            self._astra.data2d.delete(sid)
            out.append(sino.astype(np.float64).ravel())
        return np.stack(out)

    def adjoint(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        out = []
        for si in s:
            bid, img = self._astra.create_backprojection(
                si.reshape(self.g.n_angles, self.n_det).astype(np.float32), self._proj_id
            )
            self._astra.data2d.delete(bid)
            out.append(img.astype(np.float64).ravel())
        return np.stack(out)

    def fbp_backproject(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.g.beam != "parallel":
            raise NotImplementedError("fan-beam FBP backprojection with ASTRA: use backend='pixel'")
        return self.adjoint(q) * self.g.spacing

    def __del__(self) -> None:
        try:
            self._astra.projector.delete(self._proj_id)
        except Exception:
            pass


def astra_available() -> bool:
    try:
        import astra  # noqa: F401
    except ImportError:
        return False
    return True


class RadonOperator:
    """Единый интерфейс: ``A`` (``forward``), ``A^T`` (``adjoint``) и ``LinearOperator``."""

    def __init__(
        self,
        geometry: Geometry,
        backend: BackendName = "auto",
        matrix_budget_mb: float = 512.0,
        dtype: type[np.floating] = np.float32,
    ) -> None:
        self.geometry = geometry
        self.dtype = dtype
        if backend == "auto":
            backend = "astra" if astra_available() else "pixel"
        if backend == "pixel":
            self._backend: _Backend = PixelDrivenBackend(geometry, matrix_budget_mb)
        elif backend == "astra":
            self._backend = AstraBackend(geometry)
        elif backend == "skimage":
            self._backend = SkimageBackend(geometry)
            logger.warning("skimage backend: adjoint is approximate; LSQR projections lose accuracy")
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.n_det = self._backend.n_det
        self._norm: float | None = None

    @property
    def backend(self) -> _Backend:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def exact_adjoint(self) -> bool:
        return self._backend.exact_adjoint

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.geometry.shape

    @property
    def sino_shape(self) -> tuple[int, int]:
        return (self.geometry.n_angles, self.n_det)

    @property
    def shape(self) -> tuple[int, int]:
        """Размер матрицы ``A``: ``(n_rays, n_pixels)``."""
        return (int(np.prod(self.sino_shape)), int(np.prod(self.image_shape)))

    def fingerprint(self) -> str:
        return f"{self._backend.name}:{self.geometry.fingerprint()}"

    def _as_batch(self, arr: NDArray, trailing: tuple[int, int]) -> tuple[NDArray[np.float64], bool]:
        arr = np.asarray(arr)
        if arr.shape[-2:] != tuple(trailing) or arr.ndim not in (2, 3):
            raise ValueError(f"expected shape (..., {trailing[0]}, {trailing[1]}), got {arr.shape}")
        single = arr.ndim == 2
        flat = arr.reshape(1 if single else arr.shape[0], -1).astype(np.float64, copy=False)
        return flat, single

    def forward(self, x: NDArray, support: NDArray[np.bool_] | None = None) -> NDArray:
        """``A x``; с маской ``support`` проецируются только её пиксели (точно, если вне маски ноль)."""
        flat, single = self._as_batch(x, self.image_shape)
        idx = None
        if support is not None:
            idx = np.flatnonzero(np.asarray(support, dtype=bool).ravel())
            flat = flat[:, idx]
        out = self._backend.forward(flat, idx).reshape(-1, *self.sino_shape).astype(self.dtype)
        return out[0] if single else out

    def adjoint(self, s: NDArray) -> NDArray:
        """``A^T s``."""
        flat, single = self._as_batch(s, self.sino_shape)
        out = self._backend.adjoint(flat).reshape(-1, *self.image_shape).astype(self.dtype)
        return out[0] if single else out

    A = forward
    At = adjoint

    def fbp_backproject(self, q: NDArray) -> NDArray:
        """Взвешенная обратная проекция для FBP (без множителя ``d theta``)."""
        flat, single = self._as_batch(q, self.sino_shape)
        out = self._backend.fbp_backproject(flat).reshape(-1, *self.image_shape).astype(self.dtype)
        return out[0] if single else out

    def ray_mask(self, support: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Бины детектора, лучи которых пересекают ``support``."""
        sino = self.forward(np.asarray(support, dtype=np.float32), support=support)
        return sino > 0

    def as_linear_operator(self, support: NDArray[np.bool_] | None = None) -> LinearOperator:
        """``LinearOperator`` (float64) для ``A`` или для ``A_R = A[:, R]`` при заданном ``support``."""
        backend = self._backend
        idx = None if support is None else np.flatnonzero(np.asarray(support, dtype=bool).ravel())
        n_cols = self.shape[1] if idx is None else int(idx.size)
        matrix = getattr(backend, "_matrix", None)
        if idx is not None and matrix is not None:
            # столбцы R вырезаются один раз: A_R в сотни раз разреженнее полного A (профиль: 512², LSQR на R)
            sub = matrix.tocsc()[:, idx].tocsr()
            sub_t = sub.T.tocsr()
            return LinearOperator(
                shape=(self.shape[0], n_cols), matvec=lambda v: sub @ np.asarray(v, np.float64).ravel(),
                rmatvec=lambda s: sub_t @ np.asarray(s, np.float64).ravel(),
                matmat=lambda vs: sub @ np.asarray(vs, np.float64), rmatmat=lambda ss: sub_t @ np.asarray(ss, np.float64),
                dtype=np.float64,
            )

        def _restrict(img: NDArray) -> NDArray:
            return img if idx is None else img[..., idx]

        def matvec(v: NDArray) -> NDArray:
            return backend.forward(np.asarray(v, np.float64).reshape(1, -1), idx)[0]

        def rmatvec(s: NDArray) -> NDArray:
            return _restrict(backend.adjoint(np.asarray(s, np.float64).reshape(1, -1))[0])

        def matmat(vs: NDArray) -> NDArray:
            return backend.forward(np.asarray(vs, np.float64).T, idx).T

        def rmatmat(ss: NDArray) -> NDArray:
            return _restrict(backend.adjoint(np.asarray(ss, np.float64).T)).T

        return LinearOperator(
            shape=(self.shape[0], n_cols), matvec=matvec, rmatvec=rmatvec, matmat=matmat, rmatmat=rmatmat,
            dtype=np.float64,
        )

    def norm(self, n_iter: int = 30, seed: int = 0) -> float:
        """Оценка ``||A||_2`` степенным методом по ``A^T A`` (кэшируется)."""
        if self._norm is None:
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.shape[1])
            v /= np.linalg.norm(v)
            lam = 0.0
            for _ in range(n_iter):
                w = self._backend.adjoint(self._backend.forward(v[None], None))[0]
                lam = float(np.linalg.norm(w))
                if lam == 0.0:
                    break
                v = w / lam
            self._norm = math.sqrt(lam)
        return self._norm


def make_operator(
    shape: tuple[int, int],
    n_angles: int,
    beam: Beam = "parallel",
    backend: BackendName = "auto",
    matrix_budget_mb: float = 512.0,
    **geometry_kwargs: object,
) -> RadonOperator:
    """Сокращение для ``RadonOperator(Geometry(shape, n_angles, beam, ...))``."""
    geometry = Geometry(shape=(int(shape[0]), int(shape[1])), n_angles=n_angles, beam=beam, **geometry_kwargs)
    return RadonOperator(geometry, backend=backend, matrix_budget_mb=matrix_budget_mb)
