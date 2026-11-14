"""Инструменты выбора области: растеризация фигур и редактор маски с историей."""

from nullgraft_demo.tools.mask import (
    MaskEditor,
    disk_region,
    ellipse_region,
    polygon_region,
    rect_region,
    shift_mask,
)

__all__ = ["MaskEditor", "disk_region", "ellipse_region", "polygon_region", "rect_region", "shift_mask"]
