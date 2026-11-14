from __future__ import annotations

import numpy as np
import pytest

from nullgraft.api import local_resolution
from nullgraft.operators import make_operator
from nullgraft_demo.tools import (
    MaskEditor,
    disk_region,
    ellipse_region,
    polygon_region,
    rect_region,
    shift_mask,
)

SHAPE = (20, 30)


def test_rect_region_uses_pixel_centres_and_any_corner_order() -> None:
    r = rect_region(SHAPE, 5.2, 2.0, 8.9, 3.0)
    rows, cols = np.nonzero(r)
    assert (cols.min(), cols.max(), rows.min(), rows.max()) == (2, 5, 3, 8)


def test_ellipse_and_disk_regions() -> None:
    e = ellipse_region(SHAPE, 5, 15, 5, 11)
    assert e[8, 10] and not e[5, 5]
    d = disk_region(SHAPE, 10, 15, 4)
    assert d.sum() == pytest.approx(np.pi * 16, rel=0.15)
    assert ellipse_region(SHAPE, 3, 3, 4, 4)[4, 3]  # вырожденный эллипс — один пиксель


def test_polygon_region() -> None:
    tri = polygon_region(SHAPE, [(2, 2), (12, 2), (2, 12)])
    assert tri[3, 3] and not tri[11, 11]
    assert not polygon_region(SHAPE, [(1, 1), (5, 5)]).any()


def test_boolean_operations_and_area() -> None:
    ed = MaskEditor(SHAPE)
    ed.apply(rect_region(SHAPE, 0, 9, 0, 9), "add")
    ed.apply(rect_region(SHAPE, 5, 14, 0, 9), "add")
    assert ed.area_px == 10 * 15
    ed.apply(rect_region(SHAPE, 0, 4, 0, 9), "subtract")
    assert ed.area_px == 10 * 10
    assert ed.area_mm2(0.5) == pytest.approx(25.0)
    ed.apply(disk_region(SHAPE, 10, 10, 2), "replace")
    assert ed.area_px == disk_region(SHAPE, 10, 10, 2).sum()
    with pytest.raises(ValueError):
        ed.apply(np.ones((3, 3), bool))
    with pytest.raises(ValueError):
        ed.apply(rect_region(SHAPE, 0, 1, 0, 1), "xor")  # type: ignore[arg-type]


def test_undo_redo_and_history_limit() -> None:
    ed = MaskEditor(SHAPE, history=3)
    states = []
    for i in range(5):
        ed.apply(rect_region(SHAPE, i, i, 0, 0), "add")
        states.append(ed.mask.copy())
    assert ed.undo() and np.array_equal(ed.mask, states[3])
    assert ed.redo() and np.array_equal(ed.mask, states[4])
    while ed.undo():
        pass
    assert np.array_equal(ed.mask, states[1])  # хранится только 3 шага
    ed.apply(rect_region(SHAPE, 20, 20, 5, 5), "add")
    assert not ed.can_redo  # новая правка сбрасывает redo


def test_clear_and_noop_do_not_pollute_history() -> None:
    ed = MaskEditor(SHAPE)
    ed.clear()
    assert not ed.can_undo
    ed.apply(rect_region(SHAPE, 1, 2, 1, 2))
    ed.apply(rect_region(SHAPE, 1, 2, 1, 2))
    ed.clear()
    assert ed.empty and ed.undo() and ed.area_px == 4
    assert ed.undo() and ed.empty and not ed.undo()  # повторная одинаковая правка не дала лишнего шага


def test_move_is_one_undo_step_and_clips_at_border() -> None:
    ed = MaskEditor(SHAPE)
    ed.apply(rect_region(SHAPE, 0, 3, 0, 3))
    before = ed.mask.copy()
    ed.begin_move()
    ed.preview_move(1, 1)
    ed.preview_move(2, 5)
    ed.end_move()
    assert np.array_equal(ed.mask, shift_mask(before, 2, 5))
    assert ed.undo() and np.array_equal(ed.mask, before)
    assert shift_mask(before, -10, 0).sum() == 0


def test_listeners_and_readonly_view() -> None:
    ed = MaskEditor(SHAPE)
    seen: list[int] = []
    ed.subscribe(lambda m: seen.append(int(m.sum())))
    ed.apply(rect_region(SHAPE, 0, 1, 0, 1))
    ed.undo()
    assert seen == [4, 0]
    with pytest.raises(ValueError):
        ed.mask[0, 0] = True
    assert ed.centroid() is None


def test_local_resolution_grows_with_distance_from_centre() -> None:
    op = make_operator((64, 64), n_angles=30, backend="pixel")
    centre = local_resolution(op, (31.5, 31.5))
    edge = local_resolution(op, (5, 31.5))
    assert centre == pytest.approx(1.0)
    assert edge == pytest.approx(26.5 * np.pi / 30, rel=1e-6)
