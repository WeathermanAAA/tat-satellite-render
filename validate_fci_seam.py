#!/usr/bin/env python3
"""Ring-seam validation for the MTG FCI true-color member (the same protocol
the 2026-07 truecolor rebuild used for GOES-E/W and GK-2A-vs-AHI overlaps).

Co-registers the FCI and GOES-East true-color renders of ONE common slot on
a shared lat/lon grid in the Atlantic overlap and compares matched CLASSES,
not raw pixels (per-pixel diffs are dominated by parallax + the two nav
grids; class means are what the eye sees across a feathered blend seam):

  - bright NEUTRAL cloud tops (min(R,G,B) > 0.70, max-min < 0.10):
      per-channel |d mean| must be <= 0.020  (rebuild measured ~0.015-0.02)
  - matched-class ocean/mid-tones (max < 0.35, B >= R, sunlit):
      per-channel |d mean| must be <= 0.040  (rebuild measured ~0.02-0.04)

Both renders route through I.produce_recipe_imagery -- the PRODUCTION
dispatcher (never a per-family produce_* directly: the harness-trap lesson).
Geometry masks: both satellite zenith angles < 65 deg (the ring's own
GEO_MAX_ZENITH_DEG) and SZA < 70 deg so the classes are lit and inside both
feather windows.

Run on the box (needs EUMETSAT creds + the accepted licence + satpy):

  docker compose -p tat-s2 -f docker-compose.s2.yml run --rm \
      --entrypoint python emit validate_fci_seam.py

Exit 0 = seam holds within the ring tolerances; exit 1 = numbers printed for
diagnosis, do NOT ship the sector into the ring until resolved. A PNG pair
(fci_seam_check.png) lands beside the script for the eyeball step.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np

import s2_imagery as I
import s2_registry as R
import s2_meteosat as MET

UTC = dt.timezone.utc

# Atlantic overlap box: far enough off both limbs that the ring would blend
# here, wide enough to always contain both classes in daylight.
BOX_W, BOX_S, BOX_E, BOX_N = -55.0, -40.0, -15.0, 40.0
GRID_STEP_DEG = 0.05
TOL_CLOUDTOP = 0.020
TOL_MIDTONE = 0.040
MAX_SAT_ZENITH_DEG = 65.0
MAX_SZA_DEG = 70.0
NADIR_GOES_E = -75.2
NADIR_MTG = 0.0
MIN_CLASS_PX = 2000          # a class this thin is a non-representative slot


def _resample_to_box(img, lons, lats):
    """Bilinear-sample an ImageryResult's RGB (float 0..1) at query lon/lat.
    The rasters are regular lat/lon grids over img.bounds (row 0 = north)."""
    from scipy.ndimage import map_coordinates
    W, S, E, N = img.bounds
    rgba = img.rgba.astype(np.float32) / 255.0
    h, w = rgba.shape[:2]
    cols = (lons - W) / (E - W) * (w - 1)
    rows = (N - lats) / (N - S) * (h - 1)
    out = np.empty(lons.shape + (3,), np.float32)
    alpha = map_coordinates(rgba[..., 3], [rows, cols], order=1, cval=0.0)
    for c in range(3):
        out[..., c] = map_coordinates(rgba[..., c], [rows, cols], order=1,
                                      cval=np.nan)
    out[alpha < 0.99] = np.nan          # feather/off-disk stays out of stats
    return out


def _zenith_ok(lons, lats, sub_lon, when):
    """Satellite zenith bound via the pipeline's own geometry helper."""
    import truecolor
    sat_zen, _az = truecolor.satellite_geometry(lats, lons, sub_lon, when)
    return sat_zen < MAX_SAT_ZENITH_DEG


def _sza_ok(lons, lats, when):
    import truecolor
    _cos, sza, _az = truecolor.solar_geometry(lats, lons, when)
    return sza < MAX_SZA_DEG


def _class_stats(rgb, mask, name):
    sel = rgb[mask]
    print(f"  [{name}] n={sel.shape[0]}  mean RGB = "
          + np.array2string(np.nanmean(sel, axis=0), precision=4))
    return np.nanmean(sel, axis=0), sel.shape[0]


def main() -> int:
    slot = MET.newest_fci_slot()
    if slot is None:
        print("FAIL: no licence-compliant FCI slot to validate against")
        return 1
    print(f"[seam] common slot (FCI sensing end): {slot.isoformat()}")

    e_fci = R.REGISTRY_BY_ID["mtgi1-fd-truecolor"]
    e_goes = R.REGISTRY_BY_ID["goes19-fd-truecolor"]
    img_fci = I.produce_recipe_imagery(e_fci, time=slot, nearest=True)
    img_goes = I.produce_recipe_imagery(e_goes, time=slot, nearest=True)
    print(f"[seam] FCI stamp {img_fci.stamp}  GOES-E stamp {img_goes.stamp}")

    lon = np.arange(BOX_W, BOX_E, GRID_STEP_DEG, np.float32)
    lat = np.arange(BOX_N, BOX_S, -GRID_STEP_DEG, np.float32)
    LON, LAT = np.meshgrid(lon, lat)

    a = _resample_to_box(img_fci, LON, LAT)      # FCI
    b = _resample_to_box(img_goes, LON, LAT)     # GOES-East
    geom = (_zenith_ok(LON, LAT, NADIR_MTG, slot)
            & _zenith_ok(LON, LAT, NADIR_GOES_E, slot)
            & _sza_ok(LON, LAT, slot)
            & np.isfinite(a[..., 0]) & np.isfinite(b[..., 0]))
    print(f"[seam] co-registered sunlit dual-view pixels: {int(geom.sum())}")
    if geom.sum() < 10 * MIN_CLASS_PX:
        print("FAIL: overlap box mostly dark/off-disk -- rerun near 12-15 UTC "
              "when the Atlantic overlap is sunlit")
        return 1

    def classes(rgb):
        mx = np.nanmax(rgb, axis=-1)
        mn = np.nanmin(rgb, axis=-1)
        cloudtop = geom & (mn > 0.70) & ((mx - mn) < 0.10)
        midtone = geom & (mx < 0.35) & (rgb[..., 2] >= rgb[..., 0])
        return cloudtop, midtone

    ct_a, mt_a = classes(a)
    ct_b, mt_b = classes(b)

    print("[seam] FCI classes:")
    m_ct_a, n1 = _class_stats(a, ct_a, "cloudtop")
    m_mt_a, n2 = _class_stats(a, mt_a, "ocean/mid")
    print("[seam] GOES-East classes:")
    m_ct_b, n3 = _class_stats(b, ct_b, "cloudtop")
    m_mt_b, n4 = _class_stats(b, mt_b, "ocean/mid")

    ok = True
    if min(n1, n2, n3, n4) < MIN_CLASS_PX:
        print(f"FAIL: a class is too thin (<{MIN_CLASS_PX}px) for stable "
              "stats -- rerun at a slot with more daytime overlap cloud")
        ok = False

    d_ct = np.abs(m_ct_a - m_ct_b)
    d_mt = np.abs(m_mt_a - m_mt_b)
    print(f"[seam] cloudtop |d mean RGB| = "
          + np.array2string(d_ct, precision=4) + f"  (tol {TOL_CLOUDTOP})")
    print(f"[seam] ocean/mid |d mean RGB| = "
          + np.array2string(d_mt, precision=4) + f"  (tol {TOL_MIDTONE})")
    if np.any(d_ct > TOL_CLOUDTOP):
        print("FAIL: bright neutral cloud tops diverge beyond the ring "
              "tolerance -- the seam would show")
        ok = False
    if np.any(d_mt > TOL_MIDTONE):
        print("FAIL: ocean/mid-tones diverge beyond the ring tolerance")
        ok = False

    try:
        from PIL import Image
        half = np.where(LON[..., None] < (BOX_W + BOX_E) / 2.0, a, b)
        panel = np.concatenate(
            [np.nan_to_num(a), np.nan_to_num(half), np.nan_to_num(b)], axis=1)
        Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)).save(
            "fci_seam_check.png")
        print("[seam] wrote fci_seam_check.png (FCI | split | GOES-E)")
    except Exception as e:  # noqa: BLE001 -- the PNG is the eyeball aid only
        print(f"[seam] panel save skipped: {e}")

    print("[seam] " + ("PASS -- FCI disappears into the ring blends"
                       if ok else "FAIL -- see numbers above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
