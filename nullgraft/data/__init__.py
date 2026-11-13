"""Данные: синтетические фантомы, Visible Human Project, разбиения без утечки."""

from nullgraft.data.splits import Split, block_split, region_split
from nullgraft.data.synthetic import iter_synthetic, random_ellipse_phantom, shepp_logan, synthetic_volume

__all__ = ["Split", "block_split", "iter_synthetic", "random_ellipse_phantom", "region_split", "shepp_logan",
           "synthetic_volume"]
