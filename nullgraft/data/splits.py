"""Разбиение train/test без утечки: блоками соседних срезов или по анатомическим областям."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Split:
    train: NDArray[np.int64]
    test: NDArray[np.int64]

    def __post_init__(self) -> None:
        if np.intersect1d(self.train, self.test).size:
            raise ValueError("train and test overlap")


def block_split(n_slices: int, block_size: int = 50, test_fraction: float = 0.2, gap: int = 5,
                seed: int = 0) -> Split:
    """Целые блоки по ``block_size`` срезов идут в train или test; срезы в зазоре ``gap`` у границы отбрасываются."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    n_blocks = int(np.ceil(n_slices / block_size))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_fraction * n_blocks)))
    test_blocks = set(rng.choice(n_blocks, n_test, replace=False).tolist())
    block_of = np.arange(n_slices) // block_size
    is_test = np.isin(block_of, list(test_blocks))
    keep = np.ones(n_slices, bool)
    boundaries = np.flatnonzero(np.diff(is_test.astype(int)) != 0)
    for b in boundaries:
        keep[max(0, b + 1 - gap) : b + 1 + gap] = False
    idx = np.arange(n_slices)
    return Split(idx[keep & ~is_test], idx[keep & is_test])


def region_split(region_labels: Sequence[str], test_regions: Sequence[str]) -> Split:
    """Разбиение по метке анатомической области среза (например, "head", "thorax")."""
    labels = np.asarray(region_labels)
    unknown = set(test_regions) - set(labels.tolist())
    if unknown:
        raise ValueError(f"unknown regions {sorted(unknown)}")
    is_test = np.isin(labels, list(test_regions))
    idx = np.arange(labels.size)
    return Split(idx[~is_test], idx[is_test])
