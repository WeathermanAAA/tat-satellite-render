"""mosaic.py — wide-area (hemisphere / global) day-Vis / night-SWIR backdrop.

Composites the three geostationary disks already in the registry — GOES-East
(~75W), GOES-West (~137W), Himawari-9 (~140E) — into ONE chrome-free,
georeferenced, TRANSPARENT-alpha equirectangular grayscale raster for the wide
ASCAT/MW views (the single-disk basin backdrops in floater_poller can't fill a
hemisphere/global extent).

Per-pixel day/night: VISIBLE where the sun is up, SHORT-WAVE IR (ch7) where it's
night, with a smooth VECTORIZED SOLAR-ZENITH terminator blend (pivot 85deg, to
match the single-disk ``backdrop_band``). Each satellite is dual-fetched (vis +
ch7).

Frame: PACIFIC-CENTERED equirectangular, lon 0..360E (col 0 = 0E). The only
coverage seam falls in the GOES/Himawari gap (~0-70E: Africa, Europe, W Indian
Ocean; no Meteosat in the registry), left TRANSPARENT (the consumer's basemap
shows through), never black — so the gap reads as "no satellite coverage," not
broken.

Implementation: each disk-channel is SCATTER-rasterized (pure numpy, lon mod 360)
into the target grid, then vis/swir are blended by the target-grid solar zenith
and the disks are composited by a per-pixel view-quality weight (cos of the angle
from the sub-sat point) so overlaps blend toward the better-viewing satellite and
limbs fade — no antimeridian gore, no pcolormesh limb streaks, clean alpha. EACH
DISK IS FETCHED IN A FRESH SUBPROCESS that returns only the small scattered grids,
so the ~1.6 GB per-disk fetch footprint is fully reclaimed before the next disk
(the whole 3-disk build otherwise trips a ~2 GB memory cap).

Output: a lossy WebP WITH alpha + WGS84 bounds [0, -LAT_LIM, 360, LAT_LIM]. The
GOES/Himawari -> Meteosat gap closer (a EUMETSAT MTG key) is a flagged follow-up.
"""
from __future__ import annotations

import asyncio
import gc
import io
import math
import datetime as dt
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

import satellites as sat_mod
from colormaps import get_enhancement, enhancement_norm, normalize_visible

# Output raster: a dimmed wide-area backdrop, so a modest resolution is plenty.
TARGET_W = 1800
# Geostationary imagery degrades badly past ~65deg and TCs live well equatorward;
# capping the fetched/output lat band here also keeps each full-disk fetch under
# the ~2 GB per-process memory ceiling (the limb above 65deg is the costly part).
LAT_LIM = 65.0
TARGET_H = int(round(TARGET_W * (2 * LAT_LIM) / 360.0))   # ~1 px/deg

# Day/night terminator (matches floater_poller NIGHT_ZENITH_DEG=85): vis when the
# sun is up, SWIR when down, blended across +-FEATHER_DEG of the pivot.
PIVOT_DEG = 85.0
FEATHER_DEG = 8.0

SRC_STRIDE = 2                 # source downsample (limb coverage thins with stride)
PER_DISK_TIMEOUT_S = 200.0     # a hung S3 fetch never blocks the whole build

# The three disks + a generous fetch bbox per disk (the projection masks off-disk;
# GOES-West / Himawari cross +-180 so their bboxes are given in crossing form w>e).
# fetch_vis: GOES ABI visible reads strided/lazy (cheap over a wide bbox), so GOES
# sectors get true day-VIS / night-SWIR. Himawari AHI band-3 visible is 0.5 km and
# SEGMENT-downloaded (~300 MB/segment), so a wide crop pulls multiple GB before
# decimation -> infeasible here; the Himawari (W-Pacific) sector therefore uses
# SHORTWAVE-IR day AND night (clouds still read, just not the brighter VIS look).
# Tiled Himawari-visible fetching to restore VIS by day there is a flagged
# follow-up alongside the 0-70E Meteosat gap.
MOSAIC_SATS = [
    (sat_mod.GOES_EAST, [-142.0, -LAT_LIM, -5.0, LAT_LIM], True),
    (sat_mod.GOES_WEST, [150.0, -LAT_LIM, -62.0, LAT_LIM], True),
    (sat_mod.HIMAWARI_PACIFIC, [66.0, -LAT_LIM, -148.0, LAT_LIM], False),
]
VIS_CHANNEL = "visible_red"
SWIR_CHANNEL = "shortwave_ir"


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def solar_zenith_grid(lats: np.ndarray, lons: np.ndarray,
                      when: dt.datetime) -> np.ndarray:
    """Vectorized solar zenith angle (deg) — array form of
    floater_poller.solar_zenith_deg (NOAA low-precision, ~1deg). ``lons``
    east-positive."""
    when = when.astimezone(dt.timezone.utc)
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    gamma = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    tst = hour * 60.0 + (eqtime + 4.0 * lons)
    ha = np.radians(tst / 4.0 - 180.0)
    latr = np.radians(lats)
    cos_zen = (math.sin(decl) * np.sin(latr)
               + math.cos(decl) * np.cos(latr) * np.cos(ha))
    return np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))


def _gray_field(data, *, is_visible: bool) -> np.ndarray:
    """Per-pixel grayscale [0,1] for a fetched disk-channel; NaN where no data."""
    if is_visible:
        g = np.asarray(normalize_visible(data.cmi), dtype=float)
        return np.where(np.isfinite(data.cmi), g, np.nan)
    bt = np.asarray(data.cmi, dtype=float)
    if getattr(data, "units", "") in ("C", "celsius", "degC"):
        bt = bt + 273.15
    bt_c = np.ma.masked_invalid(bt - 273.15)
    cmap = get_enhancement("grayscale")["cmap"]
    rgba = cmap(enhancement_norm("grayscale")(bt_c))     # H×W×4, gray
    g = np.asarray(rgba[..., 0], dtype=float)
    return np.where(np.ma.getmaskarray(bt_c), np.nan, g)


def _scatter(gray: np.ndarray, lats: np.ndarray, lons: np.ndarray,
             acc: np.ndarray, cnt: np.ndarray) -> None:
    """Bin a disk-channel's pixels into the 0..360 target grid (in place)."""
    fin = np.isfinite(gray) & np.isfinite(lats) & np.isfinite(lons)
    g = gray[fin]
    lo = np.mod(lons[fin], 360.0)
    la = lats[fin]
    col = np.floor(lo / 360.0 * TARGET_W).astype(np.int64)
    row = np.floor((LAT_LIM - la) / (2.0 * LAT_LIM) * TARGET_H).astype(np.int64)
    m = (row >= 0) & (row < TARGET_H) & (col >= 0) & (col < TARGET_W)
    np.add.at(acc, (row[m], col[m]), g[m])
    np.add.at(cnt, (row[m], col[m]), 1.0)


async def _fetch_layer(sat, channel: str, bbox: list, when: dt.datetime):
    resolved = await sat.find_file(when, channel, bbox, False)
    return await sat.fetch(resolved, bbox, channel)


def _disk_grids(sat_idx: int, when: dt.datetime):
    """SUBPROCESS entry: fetch + scatter ONE disk's vis + ch7 into the target grid
    and return (gv, cv, gs, cs) — small arrays. All the heavy fetch memory is
    reclaimed when this process exits. Returns None if the disk could not be
    fetched at all."""
    # Fresh S3 filesystem in this subprocess: the poller is multi-threaded, and a
    # fork can inherit an s3fs singleton whose lock another thread held at fork
    # time. Resetting it forces _get_fs() to build a clean one here.
    sat_mod._fs = None
    sat, bbox, fetch_vis = MOSAIC_SATS[sat_idx]
    gv = np.zeros((TARGET_H, TARGET_W)); cv = np.zeros((TARGET_H, TARGET_W))
    gs = np.zeros((TARGET_H, TARGET_W)); cs = np.zeros((TARGET_H, TARGET_W))

    async def run():
        ok = 0
        layers = [(SWIR_CHANNEL, False, gs, cs)]
        if fetch_vis:
            layers.append((VIS_CHANNEL, True, gv, cv))
        for channel, is_vis, acc, cnt in layers:
            try:
                data = await _fetch_layer(sat, channel, bbox, when)
            except Exception as e:  # noqa: BLE001
                print(f"mosaic: {sat.__class__.__name__} {channel} fetch failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            st = SRC_STRIDE
            _scatter(_gray_field(data, is_visible=is_vis)[::st, ::st],
                     np.asarray(data.lats)[::st, ::st],
                     np.asarray(data.lons)[::st, ::st], acc, cnt)
            del data
            gc.collect()
            ok += 1
        return ok

    n = asyncio.run(run())
    if n == 0:
        return None
    if not fetch_vis:
        # SWIR-only disk (Himawari): use SWIR on the day side too.
        gv, cv = gs.copy(), cs.copy()
    return gv, cv, gs, cs


def _view_quality(lat_grid: np.ndarray, lon_grid: np.ndarray,
                  subsat: float) -> np.ndarray:
    """cos of the great-circle angle from the sub-sat point (0,subsat), clipped
    and sharpened — ~1 head-on, ->0 at the limb."""
    dlon = np.radians(lon_grid - subsat)
    cosc = np.cos(np.radians(lat_grid)) * np.cos(dlon)
    return np.clip(cosc, 0.0, 1.0) ** 2


def build_global_mosaic(when: dt.datetime):
    """Build the day/night composite, return (webp_bytes, bounds[W,S,E,N],
    n_disks). Each disk is fetched+scattered in a FRESH subprocess (bounded
    memory). Raises if NO disk could be used (caller keeps last-known-good)."""
    lon_1d = (np.arange(TARGET_W) + 0.5) / TARGET_W * 360.0
    lat_1d = LAT_LIM - (np.arange(TARGET_H) + 0.5) / TARGET_H * (2 * LAT_LIM)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    zen = solar_zenith_grid(lat_grid, lon_grid, when)
    day_w = 1.0 - _smoothstep(PIVOT_DEG - FEATHER_DEG, PIVOT_DEG + FEATHER_DEG, zen)

    num = np.zeros((TARGET_H, TARGET_W), dtype=float)
    den = np.zeros((TARGET_H, TARGET_W), dtype=float)
    cov = np.zeros((TARGET_H, TARGET_W), dtype=float)
    n_ok = 0

    for sat_idx, (sat, _bbox, _fv) in enumerate(MOSAIC_SATS):
        try:
            # A fresh single-task process per disk -> its entire fetch footprint is
            # freed on exit (a 3-disk build in one process fragments past the
            # memory ceiling). fork is safe here: the poller is single-threaded
            # (only a RateLimiter Lock, never held across this call), and each
            # child resets its own S3 filesystem (_disk_grids).
            with ProcessPoolExecutor(max_workers=1) as ex:
                grids = ex.submit(_disk_grids, sat_idx, when).result(
                    timeout=PER_DISK_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — best effort per disk
            print(f"mosaic: {sat.__class__.__name__} subprocess failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        if grids is None:
            continue
        gv, cv, gs, cs = grids
        vis_grid = np.where(cv > 0, gv / np.maximum(cv, 1e-9), np.nan)
        swir_grid = np.where(cs > 0, gs / np.maximum(cs, 1e-9), np.nan)
        sat_gray = day_w * np.nan_to_num(vis_grid) + (1 - day_w) * np.nan_to_num(swir_grid)
        only_vis = np.isfinite(vis_grid) & ~np.isfinite(swir_grid)
        only_swir = np.isfinite(swir_grid) & ~np.isfinite(vis_grid)
        sat_gray = np.where(only_vis, np.nan_to_num(vis_grid), sat_gray)
        sat_gray = np.where(only_swir, np.nan_to_num(swir_grid), sat_gray)
        sat_alpha = ((cv > 0) | (cs > 0)).astype(float)
        w = _view_quality(lat_grid, lon_grid, float(sat.sub_sat_lon)) * sat_alpha
        num += w * sat_gray
        den += w
        cov = np.maximum(cov, sat_alpha)
        n_ok += 1
        print(f"mosaic: {sat.__class__.__name__} ok "
              f"(coverage {float((cov > 0).mean()):.0%})", flush=True)

    if n_ok == 0:
        raise RuntimeError("mosaic: no disk could be fetched/rendered")

    gray = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    # Brighten midtones (gamma < 1) so cloud detail reads when the viewer draws the
    # mosaic dimmed at ~40% opacity — a global frame is mostly night ocean / low
    # evening sun, which is genuinely dark, but the day-side cloud structure should
    # still pop under the barbs. Blacks stay black, whites stay white.
    gray = np.clip(gray, 0.0, 1.0) ** 0.7
    rgb = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    # Alpha = the view-quality-weighted coverage (den), so the OBLIQUE LIMB of each
    # disk (low view quality -> sparse scatter -> ring/moire artifacts) fades to
    # transparent into the basemap instead of showing rings, while head-on /
    # overlapped pixels stay fully opaque. The uncovered gap (den=0) stays
    # transparent. Lifted slightly (den*1.4) so the well-viewed interior is solid.
    alpha = (np.clip(den * 1.4, 0.0, 1.0) * 255).astype(np.uint8)
    out = np.dstack([rgb, rgb, rgb, alpha])
    buf = io.BytesIO()
    Image.fromarray(out, mode="RGBA").save(buf, "WEBP", quality=80, method=6)
    return buf.getvalue(), [0.0, -LAT_LIM, 360.0, LAT_LIM], n_ok
