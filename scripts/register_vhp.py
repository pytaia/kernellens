"""Регистрация криосрезов Visible Human на сетку КТ (аффинная + B-spline, SimpleITK).

Запуск:
    python scripts/register_vhp.py --ct CT_DIR --cryo CRYO_DIR --cryo-pixel-mm 0.33 --out OUT_DIR
        [--landmarks landmarks.csv] [--pairs pairs.csv] [--slices 100:400]

--pairs      CSV ct_index,cryo_file — соответствие срезов КТ и криосрезов (без него — по порядку сортировки).
--landmarks  CSV ct_index,ct_row,ct_col,cryo_row,cryo_col — ориентиры для TRE (мм) до и после регистрации.

Выход в OUT_DIR: ct.npy (μ), cryo.npy (зарегистрированный криосрез на сетке КТ), registration_log.json
(Dice масок тела, TRE, значения метрики). Ничего не скачивается. Подробности: docs/08_data.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from nullgraft.data.vhp import cryo_to_ct_grid, cryo_to_gray, hu_to_mu, list_cryo, read_ct_slice, scan_ct_series
from nullgraft.metrics import dice

logger = logging.getLogger("register_vhp")


def _otsu_mask(img: np.ndarray) -> np.ndarray:
    from skimage.filters import threshold_otsu

    return img > threshold_otsu(img)


def _register(fixed: np.ndarray, moving: np.ndarray, bspline_mesh: int, iterations: int):
    import SimpleITK as sitk

    f = sitk.GetImageFromArray(fixed.astype(np.float32))
    m = sitk.GetImageFromArray(moving.astype(np.float32))
    # аффинная регистрация по взаимной информации: у КТ и фотографии разный контраст
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, iterations)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)
    init = sitk.CenteredTransformInitializer(f, m, sitk.AffineTransform(2),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg.SetInitialTransform(init, inPlace=False)
    affine = reg.Execute(f, m)
    affine_metric = reg.GetMetricValue()
    m_aff = sitk.Resample(m, f, affine, sitk.sitkLinear, 0.0)
    bsp = sitk.BSplineTransformInitializer(f, [bspline_mesh, bspline_mesh])
    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    reg2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iterations)
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetInitialTransform(bsp, inPlace=True)
    reg2.Execute(f, m_aff)
    composite = sitk.CompositeTransform([affine, bsp])
    out = sitk.GetArrayFromImage(sitk.Resample(m, f, composite, sitk.sitkLinear, 0.0))
    return out, composite, {"affine_metric": affine_metric, "bspline_metric": reg2.GetMetricValue()}


def _tre(transform, landmarks: list[tuple[float, float, float, float]], pixel_mm: tuple[float, float]) -> dict:
    before, after = [], []
    for ct_r, ct_c, cr_r, cr_c in landmarks:
        # SimpleITK переводит точки КТ в точки криосреза и ждёт (x, y), а массивы индексируются (строка, столбец)
        mapped = transform.TransformPoint((ct_c, ct_r))
        after.append(np.hypot((mapped[1] - cr_r) * pixel_mm[0], (mapped[0] - cr_c) * pixel_mm[1]))
        before.append(np.hypot((ct_r - cr_r) * pixel_mm[0], (ct_c - cr_c) * pixel_mm[1]))
    return {"tre_before_mm": float(np.mean(before)), "tre_after_mm": float(np.mean(after)), "n": len(landmarks)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True, type=Path)
    ap.add_argument("--cryo", required=True, type=Path)
    ap.add_argument("--cryo-pixel-mm", type=float, required=True, help="in-plane cryosection resolution (check it)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--landmarks", type=Path)
    ap.add_argument("--slices", default=None, help="python slice over CT indices, e.g. 100:400")
    ap.add_argument("--bspline-mesh", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        logger.error("SimpleITK is required: pip install nullgraft[registration]")
        return 2

    series = scan_ct_series(args.ct)
    report = series.report()
    logger.info("CT metadata found in files: %s", report)
    cryo_files = list_cryo(args.cryo)
    if args.pairs:
        with open(args.pairs, newline="") as fh:
            pairs = [(int(r["ct_index"]), args.cryo / r["cryo_file"]) for r in csv.DictReader(fh)]
    else:
        logger.warning("no --pairs given: matching CT and cryo by sorted order")
        pairs = list(enumerate(cryo_files[: len(series.slices)]))
    if args.slices:
        lo, hi = (int(v) if v else None for v in args.slices.split(":"))
        pairs = [p for p in pairs if (lo is None or p[0] >= lo) and (hi is None or p[0] < hi)]
    if not pairs:
        logger.error("no CT/cryo pairs to register")
        return 1
    landmarks: dict[int, list] = {}
    if args.landmarks:
        with open(args.landmarks, newline="") as fh:
            for r in csv.DictReader(fh):
                landmarks.setdefault(int(r["ct_index"]), []).append(
                    tuple(float(r[k]) for k in ("ct_row", "ct_col", "cryo_row", "cryo_col")))

    from skimage.io import imread

    args.out.mkdir(parents=True, exist_ok=True)
    shape = series.slices[pairs[0][0]].shape
    ct_mm = np.lib.format.open_memmap(args.out / "ct.npy", mode="w+", dtype=np.float32, shape=(len(pairs), *shape))
    cryo_mm = np.lib.format.open_memmap(args.out / "cryo.npy", mode="w+", dtype=np.float32,
                                        shape=(len(pairs), *shape))
    log: dict = {"ct_report": report, "cryo_pixel_mm": args.cryo_pixel_mm, "slices": []}
    for j, (ci, cryo_path) in enumerate(pairs):
        s = series.slices[ci]
        ct_mu = hu_to_mu(read_ct_slice(s))
        try:
            gray = cryo_to_ct_grid(cryo_to_gray(imread(cryo_path)), args.cryo_pixel_mm, s.pixel_spacing_mm, shape)
            reg, transform, metrics = _register(ct_mu, gray, args.bspline_mesh, args.iterations)
        except Exception as exc:  # один неудачный срез не должен обрывать весь прогон
            logger.error("slice %d (%s) failed: %s", ci, cryo_path.name, exc)
            ct_mm[j], cryo_mm[j] = ct_mu, np.nan
            log["slices"].append({"ct_index": ci, "cryo_file": cryo_path.name, "error": repr(exc), "excluded": True})
            (args.out / "registration_log.json").write_text(json.dumps(log, indent=1))
            continue
        entry = {"ct_index": ci, "cryo_file": cryo_path.name, **metrics,
                 "dice_before": dice(_otsu_mask(ct_mu), _otsu_mask(gray)),
                 "dice_after": dice(_otsu_mask(ct_mu), _otsu_mask(reg))}
        if ci in landmarks:
            entry |= _tre(transform, landmarks[ci], s.pixel_spacing_mm)
        ct_mm[j], cryo_mm[j] = ct_mu, reg
        log["slices"].append(entry)
        tre = f", TRE {entry['tre_before_mm']:.2f} -> {entry['tre_after_mm']:.2f} mm" if ci in landmarks else ""
        logger.info("slice %d: Dice %.3f -> %.3f%s", ci, entry["dice_before"], entry["dice_after"], tre)
        (args.out / "registration_log.json").write_text(json.dumps(log, indent=1))
    ct_mm.flush()
    cryo_mm.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
