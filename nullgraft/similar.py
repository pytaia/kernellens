"""Алгоритм B2: поиск в архиве участков, где тот же декодер давал похожие артефакты.

Подробности: docs/05_algorithm_B2.md.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.transform import resize

from nullgraft._io import atomic_save_npy, atomic_write_text, fingerprint, slices_digest
from nullgraft.assess import sample_noise, select_point
from nullgraft.decoders import Decoder
from nullgraft.feasible import patch_positions
from nullgraft.operators import RadonOperator

logger = logging.getLogger(__name__)

ResidualMode = Literal["highpass", "true"]
_TINY = 1e-12
_SEG_STRUCTURE_FRACTION = 0.1
_ARRAY_NAMES = ("descriptors", "meta", "artifacts", "contrast", "residual_energy")


@dataclass(frozen=True)
class DescriptorConfig:
    """Параметры дескриптора артефакта: размер сетки, число радиальных бинов, σ high-pass, веса блоков."""

    cells: int = 8
    n_radial: int = 8
    highpass_sigma: float = 2.0
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)


def artifact_estimate(z: NDArray, sigma: float) -> NDArray[np.float32]:
    """Оценка артефакта ``z − G_σ * z`` для ``(H, W)`` или ``(B, H, W)``."""
    z = np.asarray(z, np.float32)
    s = (0, sigma, sigma) if z.ndim == 3 else sigma
    return (z - gaussian_filter(z, s)).astype(np.float32)


def _radial_profile(patches: NDArray, n_bins: int) -> NDArray[np.float32]:
    s = patches.shape[-1]
    win = np.outer(np.hanning(s), np.hanning(s))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(patches * win), axes=(-2, -1)))
    yy, xx = np.mgrid[:s, :s] - s // 2
    rad = np.hypot(yy, xx) / (s / 2)
    bins = np.minimum((rad * n_bins).astype(int), n_bins)
    flat = spec.reshape(len(patches), -1)
    bins_flat = bins.ravel()
    prof = np.stack([flat[:, bins_flat == b].mean(axis=1) if np.any(bins_flat == b) else np.zeros(len(patches))
                     for b in range(n_bins)], axis=1)
    return (prof / (prof.sum(axis=1, keepdims=True) + _TINY)).astype(np.float32)


def _structure_stats(patches: NDArray) -> NDArray[np.float32]:
    gy, gx = np.gradient(patches, axis=(-2, -1))
    jxx, jyy, jxy = (np.mean(v, axis=(-2, -1)) for v in (gx * gx, gy * gy, gx * gy))
    trace = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (trace + _TINY)
    # удвоенный угол: ориентации θ и θ + π совпадают
    theta2 = np.arctan2(2 * jxy, jxx - jyy)
    energy = np.log10(np.mean(patches**2, axis=(-2, -1)) + _TINY)
    return np.stack([energy / 10.0, coherence * np.cos(theta2), coherence * np.sin(theta2), coherence], axis=1).astype(
        np.float32
    )


def describe(artifact_patches: NDArray, cfg: DescriptorConfig) -> NDArray[np.float32]:
    """Дескрипторы ``(B, d)`` патчей артефакта ``(B, s, s)``."""
    p = np.asarray(artifact_patches, np.float64)
    if p.ndim == 2:
        p = p[None]
    mean = p.mean(axis=(-2, -1), keepdims=True)
    std = p.std(axis=(-2, -1), keepdims=True)
    norm = (p - mean) / (std + _TINY)
    small = np.stack([resize(q, (cfg.cells, cfg.cells), anti_aliasing=True) for q in norm]).reshape(len(p), -1)
    w_patch, w_spec, w_stats = cfg.weights
    return np.concatenate(
        [w_patch * small / cfg.cells, w_spec * _radial_profile(p, cfg.n_radial), w_stats * _structure_stats(p)], axis=1
    ).astype(np.float32)


def ncc(a: NDArray, b: NDArray) -> NDArray[np.float64]:
    """Нормированная кросс-корреляция (с вычитанием среднего) стопок патчей."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    num = np.sum(a * b, axis=(-2, -1))
    den = np.sqrt(np.sum(a * a, axis=(-2, -1)) * np.sum(b * b, axis=(-2, -1))) + _TINY
    return num / den


def _core(patches: NDArray, central: float) -> NDArray:
    s = patches.shape[-1]
    m = max(1, int(round(s * (1 - central) / 2)))
    return np.asarray(patches)[..., m : s - m, m : s - m].reshape(len(patches), -1)


def truth_contrast(patches: NDArray, central: float = 0.5) -> NDArray[np.float32]:
    """Робастный контраст ``p95 − p5`` центральной части эталонных патчей."""
    core = _core(patches, central)
    return (np.percentile(core, 95, axis=1) - np.percentile(core, 5, axis=1)).astype(np.float32)


def central_fraction(masks: NDArray, central: float = 0.5) -> NDArray[np.float32]:
    """Доля размеченных пикселей в центральной части патчей сегментации."""
    return _core(np.asarray(masks, np.float32) > 0, central).mean(axis=1).astype(np.float32)


@dataclass(frozen=True)
class Match:
    index: int
    slice_index: int
    row: int
    col: int
    descriptor_distance: float
    ncc: float
    truth_contrast: float
    residual_energy: float
    is_structure: bool


@dataclass(frozen=True)
class SimilarArtifactSummary:
    n_matches: int
    fraction_real_structure: float
    fraction_background: float

    def __str__(self) -> str:
        return (f"{self.n_matches} similar artifacts: {self.fraction_real_structure:.1%} at real structures, "
                f"{self.fraction_background:.1%} on background")


class ArtifactIndex:
    """Архив артефактов декодера с KD-деревом по дескрипторам."""

    def __init__(
        self,
        descriptors: NDArray[np.float32],
        meta: NDArray[np.int32],
        artifacts: NDArray[np.float32],
        contrast: NDArray[np.float32],
        residual_energy: NDArray[np.float32],
        patch_size: int,
        cfg: DescriptorConfig,
        residual_mode: ResidualMode,
        structure_threshold: float,
        tree: cKDTree | None = None,
        fingerprint: str = "",
    ) -> None:
        self.fingerprint = fingerprint
        self.descriptors = descriptors
        self.meta = meta
        self.artifacts = artifacts
        self.contrast = contrast
        self.residual_energy = residual_energy
        self.patch_size = patch_size
        self.cfg = cfg
        self.residual_mode = residual_mode
        self.structure_threshold = structure_threshold
        self.tree = tree if tree is not None else cKDTree(np.asarray(descriptors))

    def __len__(self) -> int:
        return int(self.descriptors.shape[0])

    @classmethod
    def build(
        cls,
        slices: Sequence[NDArray],
        op: RadonOperator,
        decoder: Decoder,
        noise_sigma: float,
        eps: float = float("inf"),
        patch_size: int = 32,
        stride: int = 16,
        cfg: DescriptorConfig | None = None,
        residual_mode: ResidualMode = "highpass",
        structure_threshold: float | None = None,
        segmentations: Sequence[NDArray] | None = None,
        batch_slices: int = 8,
        out_dir: str | Path | None = None,
        seed: int = 0,
    ) -> ArtifactIndex:
        """Реконструкция срезов архива батчами, нарезка и описание патчей; сборка по отпечатку, с чекпоинтами."""
        cfg = cfg or DescriptorConfig()
        positions = patch_positions(np.asarray(slices[0]).shape, patch_size, stride)
        n_max = len(slices) * len(positions)
        build_fp = fingerprint({
            "op": op.fingerprint(), "decoder": _signature(decoder), "noise_sigma": noise_sigma, "eps": eps,
            "patch_size": patch_size, "stride": stride, "cfg": cfg.__dict__, "residual_mode": residual_mode,
            "structure_threshold": structure_threshold, "segmentations": segmentations is not None,
            "batch_slices": batch_slices, "seed": seed, "slices": slices_digest(slices),
        })
        # своя подпапка на отпечаток; out_dir/index.json указывает на текущую
        version_dir = None if out_dir is None else Path(out_dir) / f"fp_{build_fp[:16]}"
        if version_dir is not None and (version_dir / "index.json").exists():
            if json.loads((version_dir / "index.json").read_text()).get("fingerprint") == build_fp:
                logger.info("artifact index %s is up to date: loading", version_dir)
                _write_pointer(Path(out_dir), version_dir.name, build_fp)
                return cls.load(version_dir)
        d = describe(np.zeros((1, patch_size, patch_size)), cfg).shape[1]
        store = _Store(version_dir, n_max, patch_size, d, build_fp)
        half = patch_size // 2
        dyn_lo, dyn_hi = np.inf, -np.inf
        for start in range(store.done_slices, len(slices), batch_slices):
            xs = np.stack([np.asarray(s, np.float32) for s in slices[start : start + batch_slices]])
            dyn_lo, dyn_hi = min(dyn_lo, float(xs.min())), max(dyn_hi, float(xs.max()))
            rng = np.random.default_rng([seed, start])
            ys = op.forward(xs) + sample_noise(op.sino_shape, len(xs), noise_sigma, eps, rng)
            outs = decoder.reconstruct_batch(ys)
            zs = np.stack([select_point(o) for o in outs])
            res = zs - xs
            art_src = artifact_estimate(zs, cfg.highpass_sigma) if residual_mode == "highpass" else res
            for b, (x, a, r) in enumerate(zip(xs, art_src, res, strict=True)):
                si = start + b
                cut = positions
                arts = np.stack([a[rr - half : rr - half + patch_size, cc - half : cc - half + patch_size] for rr, cc in cut])
                truths = np.stack([x[rr - half : rr - half + patch_size, cc - half : cc - half + patch_size]
                                   for rr, cc in cut])
                resid = np.stack([r[rr - half : rr - half + patch_size, cc - half : cc - half + patch_size]
                                  for rr, cc in cut])
                contrast = truth_contrast(truths)
                if segmentations is not None:
                    seg = np.asarray(segmentations[si]) > 0
                    segp = np.stack([seg[rr - half : rr - half + patch_size, cc - half : cc - half + patch_size]
                                     for rr, cc in cut]).astype(np.float32)
                    labelled = central_fraction(segp) > _SEG_STRUCTURE_FRACTION
                    contrast = np.where(labelled, np.inf, contrast).astype(np.float32)
                store.append(describe(arts, cfg), np.array([(si, rr, cc) for rr, cc in cut], np.int32), arts,
                             contrast, np.sqrt(np.mean(resid**2, axis=(-2, -1))).astype(np.float32))
            store.checkpoint(start + len(xs), dyn_lo, dyn_hi)
        dyn_lo, dyn_hi = store.dynamic_range(dyn_lo, dyn_hi)
        thr = 0.1 * (dyn_hi - dyn_lo) if structure_threshold is None else structure_threshold
        index = cls(*store.arrays(), patch_size, cfg, residual_mode, float(thr), fingerprint=build_fp)
        if version_dir is not None:
            index.save(version_dir)
            _write_pointer(Path(out_dir), version_dir.name, build_fp)
        logger.info("artifact index: %d patches, structure threshold %.4g", len(index), thr)
        return index

    def query_from_image(self, z: NDArray, centers: Sequence[tuple[int, int]], k: int = 10) -> list[list[Match]]:
        """Top-``k`` похожих артефактов архива для патчей реконструкции ``z`` с центрами ``centers``."""
        if self.residual_mode != "highpass":
            raise ValueError("index built with residual_mode='true': use query(artifact_patches)")
        art = artifact_estimate(z, self.cfg.highpass_sigma)
        half = self.patch_size // 2
        padded = np.pad(art, half, mode="reflect")
        patches = np.stack([padded[r : r + self.patch_size, c : c + self.patch_size] for r, c in centers])
        return self.query(patches, k)

    def query(self, artifact_patches: NDArray, k: int = 10) -> list[list[Match]]:
        """Top-``k`` совпадений для батча патчей артефакта ``(B, s, s)``."""
        q = np.asarray(artifact_patches, np.float32)
        if q.ndim == 2:
            q = q[None]
        k = min(k, len(self))
        dist, idx = self.tree.query(describe(q, self.cfg), k=k, workers=-1)
        dist, idx = np.atleast_2d(dist).reshape(len(q), k), np.atleast_2d(idx).reshape(len(q), k)
        out = []
        for qi in range(len(q)):
            ids = idx[qi]
            corr = ncc(np.asarray(self.artifacts[ids]), q[qi][None])
            out.append([
                Match(int(n), int(self.meta[n, 0]), int(self.meta[n, 1]), int(self.meta[n, 2]), float(dist[qi, j]),
                      float(corr[j]), float(self.contrast[n]), float(self.residual_energy[n]),
                      bool(self.contrast[n] > self.structure_threshold))
                for j, n in enumerate(ids)
            ])
        return out

    @staticmethod
    def summarize(matches: Sequence[Sequence[Match]]) -> SimilarArtifactSummary:
        flat = [m for ms in matches for m in ms]
        if not flat:
            return SimilarArtifactSummary(0, float("nan"), float("nan"))
        real = sum(m.is_structure for m in flat) / len(flat)
        return SimilarArtifactSummary(len(flat), real, 1.0 - real)

    def save(self, out_dir: str | Path) -> None:
        """Атомарная запись массивов и затем ``index.json``; KD-дерево не сохраняется (без pickle)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in _ARRAY_NAMES:
            atomic_save_npy(out / f"{name}.npy", np.asarray(getattr(self, name)))
        atomic_write_text(out / "index.json", json.dumps({
            "patch_size": self.patch_size, "residual_mode": self.residual_mode,
            "structure_threshold": self.structure_threshold, "cfg": self.cfg.__dict__,
            "fingerprint": self.fingerprint,
        }))

    @classmethod
    def load(cls, out_dir: str | Path) -> ArtifactIndex:
        """Ленивая загрузка массивов (``mmap_mode="r"``) и пересборка KD-дерева."""
        out = Path(out_dir)
        info = json.loads((out / "index.json").read_text())
        if "current" in info:
            out = out / info["current"]
            info = json.loads((out / "index.json").read_text())
        cfg_d = info["cfg"]
        cfg = DescriptorConfig(cfg_d["cells"], cfg_d["n_radial"], cfg_d["highpass_sigma"], tuple(cfg_d["weights"]))
        arrays = [np.load(out / f"{n}.npy", mmap_mode="r", allow_pickle=False) for n in _ARRAY_NAMES]
        return cls(*arrays, info["patch_size"], cfg, info["residual_mode"], info["structure_threshold"],
                   fingerprint=info.get("fingerprint", ""))


def _write_pointer(out_dir: Path, current: str, build_fp: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "index.json", json.dumps({"current": current, "fingerprint": build_fp}))


def _signature(obj: object, depth: int = 0) -> object:
    """Описание декодера для отпечатка: класс и скалярные настройки (рекурсивно для обёрток)."""
    if depth > 3 or isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    items = {}
    for k, v in sorted(vars(obj).items()) if hasattr(obj, "__dict__") else ():
        if isinstance(v, (int, float, str, bool, tuple)) or v is None:
            items[k] = v
        elif isinstance(v, Decoder):
            items[k] = _signature(v, depth + 1)
    return {"class": type(obj).__name__, **items}


class _Store:
    """Массивы с дозаписью (в памяти или memmap) и возобновляемым файлом прогресса."""

    _names = _ARRAY_NAMES

    def __init__(self, out_dir: str | Path | None, n_max: int, s: int, d: int, build_fp: str = "") -> None:
        shapes = {"descriptors": ((n_max, d), np.float32), "meta": ((n_max, 3), np.int32),
                  "artifacts": ((n_max, s, s), np.float32), "contrast": ((n_max,), np.float32),
                  "residual_energy": ((n_max,), np.float32)}
        self.out = Path(out_dir) / "build" if out_dir is not None else None
        self.done_slices, self.count = 0, 0
        self._dyn = (np.inf, -np.inf)
        self.build_fp = build_fp
        if self.out is None:
            self.arr = {n: np.zeros(sh, dt) for n, (sh, dt) in shapes.items()}
            return
        self.out.mkdir(parents=True, exist_ok=True)
        prog = self.out / "progress.json"
        mode = "w+"
        if prog.exists():
            state = json.loads(prog.read_text())
            if state.get("fingerprint") == build_fp:
                self.done_slices, self.count = state["done_slices"], state["count"]
                self._dyn = (state["dyn_lo"], state["dyn_hi"])
                mode = "r+"
                logger.info("resuming artifact index at slice %d", self.done_slices)
            else:
                logger.warning("partial artifact index in %s has a different fingerprint: restarting", self.out)
        self.arr = {n: np.lib.format.open_memmap(self.out / f"{n}.npy", mode=mode, dtype=dt, shape=sh)
                    for n, (sh, dt) in shapes.items()}

    def append(self, *blocks: NDArray) -> None:
        n = len(blocks[0])
        for name, block in zip(self._names, blocks, strict=True):
            self.arr[name][self.count : self.count + n] = block
        self.count += n

    def checkpoint(self, done_slices: int, lo: float, hi: float) -> None:
        self.done_slices = done_slices
        self._dyn = (min(self._dyn[0], lo), max(self._dyn[1], hi))
        if self.out is None:
            return
        for a in self.arr.values():
            a.flush()
        atomic_write_text(self.out / "progress.json", json.dumps(
            {"done_slices": done_slices, "count": self.count, "dyn_lo": self._dyn[0], "dyn_hi": self._dyn[1],
             "fingerprint": self.build_fp}))

    def dynamic_range(self, lo: float, hi: float) -> tuple[float, float]:
        return min(self._dyn[0], lo), max(self._dyn[1], hi)

    def arrays(self) -> list[NDArray]:
        return [self.arr[n][: self.count] for n in self._names]
