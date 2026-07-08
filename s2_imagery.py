#!/usr/bin/env python3
"""Stage-2 IMAGERY producer -- the FROZEN-renderer-reuse half of the pyramid
emitter (Phase 2a). Kept SEPARATE from the pure-tiling `s2_pyramid` so the tiler
stays testable with synthetic rasters and no heavy deps.

It fetches ONE GOES clean-IR (band 13) frame with the UNCHANGED `satellites.py`
crop/fetch, then reproduces the renderer's imagery layer EXACTLY -- the same
``pcolormesh(lons, lats, bt_c)`` with the same FROZEN `tat_palettes` colortable +
Normalize (SATELLITE-REARCH §2/§6.3) -- but CHROME-FREE (no title strip, no
colorbar, no watermark, no coastlines/graticule) and on a transparent
background. The result is a regular equirectangular (PlateCarree) RGBA raster
whose pixels map LINEARLY to (lon,lat) inside its bounds -- ready for
``s2_pyramid.cut_pyramid``.

WHY reproduce pcolormesh rather than call render_png:
  * render_png bakes ALL chrome into one composed Figure with NO suppress flag
    (verified) -- its output would tile chrome into the map (§6.3).
  * the fetch returns a CURVILINEAR geostationary scan-angle grid, NOT a regular
    lon/lat raster -- so the renderer's own pcolormesh onto a PlateCarree axes is
    what turns it into the displayed equirectangular image. Reproducing that
    single pcolormesh (same cmap+norm) is byte-faithful to the displayed imagery
    and reuses zero-visual-change colortables; we only DROP the chrome overlays.
  * off-data/space becomes TRANSPARENT (a map overlay wants it) rather than the
    floater renderer's opaque DARK_BG fill -- a deliberate, documented delta.

NO renderer edit: we import the colortable objects (the frozen `tat_palettes`
SSOT) and satellites' fetch; render.py / truecolor.py are untouched.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import os
import shutil
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

# The FROZEN colortable SSOT (tat_palettes, re-exported by colormaps -- the exact
# objects render.py assigns to plot_cmap/plot_cnorm on the IR path).
from colormaps import get_enhancement, enhancement_norm  # noqa: E402
from satellites import GOESEastSatellite  # noqa: E402

UTC = dt.timezone.utc


class ImageryResult:
    """A chrome-free equirectangular RGBA raster + the geo/time metadata the
    pyramid emitter needs."""
    __slots__ = ("rgba", "bounds", "scan_start", "product", "bucket", "s3_key",
                 "generic_channel", "enhancement", "bt_grid", "bt_dims")

    def __init__(self, rgba, bounds, scan_start, product, bucket, s3_key,
                 generic_channel, enhancement):
        self.rgba = rgba                 # HxWx4 uint8
        self.bounds = bounds             # (W,S,E,N) degrees
        self.scan_start = scan_start     # tz-aware UTC (== the slot stamp source)
        self.product = product           # 'CMIPC' | 'CMIPF'
        self.bucket = bucket
        self.s3_key = s3_key
        self.generic_channel = generic_channel
        self.enhancement = enhancement
        self.bt_grid = None      # HxW float BT (deg C, NaN off-data), equirect
        self.bt_dims = None      # (w, h)

    @property
    def stamp(self) -> str:
        # STAMP_FMT = '%Y%m%dT%H%M%SZ' (same as s1_slots / satellites._parse_scan_start)
        return self.scan_start.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


async def _fetch_async(sat, entry, t, nearest):
    bbox = list(entry.sector_bbox)
    band = entry.bands[0] if entry.bands else 13
    if (entry.render_product_hint or "").lower() == "fd":
        # Bypass find_file's CONUS-first ordering; force the full-disk sector.
        # _pick_full_disk returns None on a data gap (no CMIPF in current/prev
        # hour) -- guard it so a no-data condition raises a clean, descriptive
        # error instead of an AttributeError on None.s3_key inside fetch (the
        # find_file branch already raises RuntimeError on no-files).
        resolved = await sat._pick_full_disk(entry.bucket, bbox, band, t, nearest)
        if resolved is None:
            raise RuntimeError(
                f"no GOES full-disk (CMIPF) file for {entry.product_id} "
                f"bbox={bbox} near {t.isoformat()}")
    else:
        # find_file picks CMIPC for a bbox fully inside the CONUS footprint.
        resolved = await sat.find_file(t, entry.render_channel, bbox,
                                       nearest_to_target=nearest)
    result = await sat.fetch(resolved, bbox, entry.render_channel)
    return resolved, result


def _to_bt_celsius(cmi: np.ndarray, units: str) -> np.ndarray:
    """Match render.py exactly: Kelvin source -> subtract 273.15; if the file is
    already tagged °C, render.py adds 273.15 first (net identity), so bt_c==cmi."""
    bt = cmi
    if units in ("C", "celsius", "degC"):
        bt = bt + 273.15
    return bt - 273.15


def render_imagery_rgba(cmi: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                        enhancement: str, pyramid_px: int, units: str = "K"):
    """Chrome-free reproduction of render.py's IR imagery pcolormesh.

    Returns (rgba HxWx4 uint8, bounds (W,S,E,N)). The output long edge is
    ``pyramid_px``; the aspect preserves PlateCarree degrees (1° lon : 1° lat),
    matching the renderer's PlateCarree axes. Off-data (NaN/off-disk) is
    transparent.
    """
    from render import _fill_coord_nan  # reuse the FROZEN renderer's coord fill (parity)

    bt_c = _to_bt_celsius(cmi, units)
    field = np.ma.masked_invalid(bt_c)

    # Bounds from the finite coords (nanmin/nanmax ignore off-disk NaN).
    W = float(np.nanmin(lons)); E = float(np.nanmax(lons))
    S = float(np.nanmin(lats)); N = float(np.nanmax(lats))

    # render.py:335-340 exactly: off-disk pixels inverse-project to NaN lat/lon
    # which pcolormesh hard-rejects. Mask those cells in the field and
    # nearest-fill the coords so the phantom quads collapse to zero area.
    coord_bad = ~(np.isfinite(lats) & np.isfinite(lons))
    if coord_bad.any():
        field = np.ma.masked_where(coord_bad, field)
        lats = _fill_coord_nan(lats)
        lons = _fill_coord_nan(lons)
    span_x = max(E - W, 1e-6); span_y = max(N - S, 1e-6)
    if span_x >= span_y:
        w_out = int(pyramid_px); h_out = max(1, round(pyramid_px * span_y / span_x))
    else:
        h_out = int(pyramid_px); w_out = max(1, round(pyramid_px * span_x / span_y))

    cmap = get_enhancement(enhancement)["cmap"]
    norm = enhancement_norm(enhancement)

    dpi = 100
    fig = Figure(figsize=(w_out / dpi, h_out / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    ax.set_xlim(W, E)
    ax.set_ylim(S, N)   # y increases north; the pyramid flips to top-left origin below
    import warnings
    with warnings.catch_warnings():
        # 2D curvilinear coords are non-monotonic (same as render.py's pcolormesh);
        # the cell-edge inference warning is benign -- silence it for clean logs.
        warnings.filterwarnings("ignore", message=".*not monotonically increasing.*")
        ax.pcolormesh(lons, lats, field, cmap=cmap, norm=norm, shading="auto")
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())  # (h, w, 4), row 0 = TOP of the axes.

    # Agg's row 0 is the top of the y-axis, i.e. the NORTH edge (ylim high) --
    # exactly the pyramid's top-left origin (y increases southward). No flip.
    rgba = np.ascontiguousarray(buf)
    return rgba, (W, S, E, N)


BT_PX = 1280   # long-edge of the calibrated BT data raster (compact; §6 inspector)


def resample_bt_equirect(cmi: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                         bounds, out_w: int = BT_PX, units: str = "K"):
    """Resample the source (curvilinear) brightness temperature onto a regular
    equirectangular grid (deg C, NaN off-data) for the pixel/BT inspector. This
    is a DATA resample (values), separate from the colorized pcolormesh. Returns
    (bt_grid HxW float, (w, h)). The viewer maps lon/lat -> col/row linearly."""
    from scipy.interpolate import griddata
    bt_c = _to_bt_celsius(cmi, units)
    W, S, E, N = bounds
    span_x = max(E - W, 1e-6); span_y = max(N - S, 1e-6)
    out_h = max(1, round(out_w * span_y / span_x))
    # Downsample the source before triangulating (keep griddata fast + robust).
    st = max(1, int(max(cmi.shape) // 700))
    sl = (slice(None, None, st), slice(None, None, st))
    plon = np.asarray(lons[sl]).ravel()
    plat = np.asarray(lats[sl]).ravel()
    pval = np.asarray(bt_c[sl]).ravel()
    m = np.isfinite(plon) & np.isfinite(plat) & np.isfinite(pval)
    glon = np.linspace(W, E, out_w)
    glat = np.linspace(N, S, out_h)          # row 0 = north (matches tile origin)
    gx, gy = np.meshgrid(glon, glat)
    try:
        grid = griddata((plon[m], plat[m]), pval[m], (gx, gy), method="linear")
    except Exception:                        # degenerate triangulation -> nearest
        grid = griddata((plon[m], plat[m]), pval[m], (gx, gy), method="nearest")
    return grid.astype(np.float32), (out_w, out_h)


def produce_imagery(entry, time: Optional[dt.datetime] = None,
                    nearest: bool = True) -> ImageryResult:
    """End-to-end: fetch the newest (or nearest-to-`time`) clean-IR frame for a
    tiled ProductEntry and render its chrome-free equirectangular RGBA raster."""
    if not entry.tiled:
        raise ValueError(f"{entry.product_id} is not a tiled product")
    if not entry.sector_bbox:
        raise ValueError(f"{entry.product_id} has no sector_bbox")
    sat = GOESEastSatellite()
    t = time or dt.datetime.now(UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    # Opt this tiled product into a finer-than-default fetch so the pyramid can
    # reach the sensor's native resolution (satellites.py reads the env at crop
    # time). Restore after so we never perturb a co-tenant meso/floater fetch.
    _prev = os.environ.get("SAT_MAX_PX_PER_AXIS")
    if entry.fetch_max_px:
        os.environ["SAT_MAX_PX_PER_AXIS"] = str(entry.fetch_max_px)
    try:
        resolved, r = asyncio.run(_fetch_async(sat, entry, t, nearest))
    finally:
        if entry.fetch_max_px:
            if _prev is None:
                os.environ.pop("SAT_MAX_PX_PER_AXIS", None)
            else:
                os.environ["SAT_MAX_PX_PER_AXIS"] = _prev
    units = getattr(r, "units", "K")
    rgba, bounds = render_imagery_rgba(
        r.cmi, r.lats, r.lons, entry.render_enhancement, entry.pyramid_px, units=units)
    res = ImageryResult(
        rgba=rgba, bounds=bounds, scan_start=r.scan_start, product=r.product,
        bucket=r.bucket, s3_key=getattr(resolved, "s3_key", ""),
        generic_channel=entry.render_channel, enhancement=entry.render_enhancement)
    # Calibrated BT data raster for the pixel/BT inspector (bounds MUST match the
    # imagery so lon/lat probes line up). Non-fatal: a failure just disables the
    # inspector for this frame, never blocks the imagery.
    try:
        res.bt_grid, res.bt_dims = resample_bt_equirect(
            r.cmi, r.lats, r.lons, bounds, units=units)
    except Exception:   # noqa: BLE001
        res.bt_grid, res.bt_dims = None, None
    return res


# ============================================================================
# Multi-product imagery suite (Phase 3) -- the recipe engine's fetch+render.
#
# Same FROZEN-renderer-reuse contract as the clean-IR path above, applied to
# multi-band products: co-registration is the EXACT technique satellites.py's
# fetch_true_color has always used (per-band geos crops sampled onto ONE
# regular lat/lon target grid via _latlon_to_xy + _sample_geos), and the
# emissive single channels are colorized with the SAME frozen tat_palettes
# cmap+norm objects the meso/floater products render with. satellites.py /
# render.py / truecolor.py stay untouched; the two resolution caps we need
# (SAT_MAX_PX_PER_AXIS, TRUECOLOR_MAX_PX) are already env-overridable.
# ============================================================================
import s2_recipes


@contextlib.contextmanager
def _env_override(**pairs):
    """Temporarily set env vars (skips None values); always restores."""
    prev = {}
    try:
        for k, v in pairs.items():
            if v is None:
                continue
            prev[k] = os.environ.get(k)
            os.environ[k] = str(v)
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _open_crop_band(sat, resolved, bbox):
    """Download + geos-crop one band file (mirrors _compose_true_color_sync's
    per-band worker; the tmp dir is always cleaned)."""
    ds, tmp_dir = sat.open(resolved)
    try:
        return sat._crop_to_bbox(ds, bbox)
    finally:
        ds.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def fetch_band_crops(entry, bands, time=None, nearest=True, cache=None):
    """Fetch co-temporal geos crops for `bands` of one CONUS scan.

    Anchors on the clean-IR (C13) file to resolve product+scan, then locates
    every sibling band at that SAME scan (satellites._find_band_at -- the
    frozen co-temporality guarantee fetch_true_color relies on). Returns
    (anchor ResolvedFile, {band: (cmi, x, y)}, proj) where proj =
    (lon_origin, H, r_eq, r_pol). `cache` (optional dict) memoizes crops
    across products of one suite emit, keyed (band, stamp)."""
    from concurrent.futures import ThreadPoolExecutor
    from satellites import GOESEastSatellite

    sat = GOESEastSatellite()
    t = time or dt.datetime.now(UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    bbox = list(entry.sector_bbox)

    async def _resolve():
        if (entry.render_product_hint or "").lower() == "fd":
            # Bypass find_file's CONUS-first ordering (same guard as the
            # single-band fd path in _fetch_async above).
            anchor = await sat._pick_full_disk(entry.bucket, bbox, 13, t, nearest)
            if anchor is None:
                raise RuntimeError(
                    f"no GOES full-disk (CMIPF) C13 file for {entry.product_id} "
                    f"near {t.isoformat()}")
        else:
            anchor = await sat.find_file(t, "clean_ir", bbox, nearest_to_target=nearest)
        others = [b for b in bands if b != 13]
        sibs = await asyncio.gather(*(
            sat._find_band_at(anchor.bucket, anchor.product, b, anchor.scan_start)
            for b in others))
        files = dict(zip(others, sibs))
        if 13 in bands:
            files[13] = anchor
        return anchor, files

    anchor, files = asyncio.run(_resolve())
    stamp = anchor.scan_start.strftime("%Y%m%dT%H%M%SZ")

    crops = {}
    missing = []
    for b in bands:
        got = cache.get((b, stamp)) if cache is not None else None
        if got is not None:
            crops[b] = got
        else:
            missing.append(b)
    proj = None
    if missing:
        with _env_override(SAT_MAX_PX_PER_AXIS=(entry.fetch_max_px or None)):
            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures = {b: pool.submit(_open_crop_band, sat, files[b], bbox)
                           for b in missing}
                results = {b: f.result() for b, f in futures.items()}
        for b, (cmi, x, y, lon_origin, H, r_eq, r_pol) in results.items():
            crops[b] = (cmi, x, y)
            proj = (lon_origin, H, r_eq, r_pol)
            if cache is not None:
                cache[(b, stamp)] = crops[b]
                cache["_proj"] = proj
    if proj is None:
        proj = cache["_proj"] if cache is not None else None
    if proj is None:
        raise RuntimeError("no band crops fetched and no cached projection")
    return anchor, crops, proj


def sample_bands_to_grid(crops, proj, bbox, out_w):
    """Sample every band crop onto ONE regular lat/lon grid over `bbox`
    (row 0 = north, exactly the pyramid/tile origin). This is the frozen
    fetch_true_color co-registration, generalized to N bands. Returns
    ({band: HxW float32}, bounds, (out_w, out_h))."""
    from satellites import _latlon_to_xy, _sample_geos

    W, S, E, N = bbox
    span_x = max(E - W, 1e-6)
    span_y = max(N - S, 1e-6)
    out_h = max(1, round(out_w * span_y / span_x))
    tgt_lons = np.linspace(W, E, out_w)
    tgt_lats = np.linspace(N, S, out_h)          # row 0 = north
    TLON, TLAT = np.meshgrid(tgt_lons, tgt_lats)
    lon_origin, H, r_eq, r_pol = proj
    TX, TY = _latlon_to_xy(TLAT, TLON, lon_origin, H, r_eq, r_pol)
    sampled = {b: _sample_geos(c, x, y, TX, TY) for b, (c, x, y) in crops.items()}
    return sampled, (W, S, E, N), (out_w, out_h)


def _colorize_bt(bt_k, enhancement):
    """Kelvin field -> float RGB 0..1 via a FROZEN tat_palettes enhancement
    (the exact cmap+norm objects the meso/floater renders use; NaN -> NaN)."""
    bt_c = np.asarray(bt_k, dtype=np.float64) - 273.15
    cmap = get_enhancement(enhancement)["cmap"]
    norm = enhancement_norm(enhancement)
    rgba = cmap(norm(np.ma.masked_invalid(bt_c)))       # HxWx4 float, bad -> cmap bad
    rgb = np.asarray(rgba[..., :3], dtype=np.float32)
    rgb[~np.isfinite(bt_c)] = np.nan                    # keep off-data out of alpha
    return rgb


def _decimate_bt(grid_c, bounds, out_w=BT_PX):
    """Nearest-index decimation of the (already regular) target-grid BT field
    to the compact inspector raster. Exact values, no interpolation, no NaN
    bleed. Returns (bt HxW float32, (w, h))."""
    Hh, Ww = grid_c.shape
    ow = min(out_w, Ww)
    W, S, E, N = bounds
    oh = max(1, round(ow * max(N - S, 1e-6) / max(E - W, 1e-6)))
    oh = min(oh, Hh)
    rows = np.round(np.linspace(0, Hh - 1, oh)).astype(int)
    cols = np.round(np.linspace(0, Ww - 1, ow)).astype(int)
    return grid_c[rows][:, cols].astype(np.float32), (ow, oh)


def produce_recipe_imagery(entry, time=None, nearest=True,
                           band_cache=None):
    """End-to-end recipe product: co-temporal band fetch -> one regular grid ->
    declarative band math (s2_recipes) -> chrome-free RGBA + optional BT raster.
    Same ImageryResult contract as produce_imagery, so the pyramid emitter and
    Q7 --max-zoom tiering apply unchanged."""
    if not entry.recipe_id:
        raise ValueError(f"{entry.product_id} has no recipe_id")
    recipe = s2_recipes.RECIPES_BY_KEY[entry.recipe_id]
    if recipe.kind == "truecolor":
        return produce_truecolor(entry, time=time, nearest=nearest)

    anchor, crops, proj = fetch_band_crops(
        entry, recipe.bands, time=time, nearest=nearest, cache=band_cache)
    sampled, bounds, _ = sample_bands_to_grid(
        crops, proj, entry.sector_bbox, entry.pyramid_px)

    if recipe.kind == "rgb_guns":
        rgb = s2_recipes.compute_rgb(recipe, sampled)
        rgba = s2_recipes.rgba_from_rgb(rgb)
    elif recipe.kind == "single_palette":
        rgb = _colorize_bt(sampled[recipe.band], recipe.enhancement)
        rgba = s2_recipes.rgba_from_rgb(rgb)
    elif recipe.kind == "sandwich":
        ir_rgb = _colorize_bt(sampled[13], "rainbow_ir")
        rgb = s2_recipes.sandwich_rgb(sampled[2], ir_rgb)
        rgba = s2_recipes.rgba_from_rgb(rgb)
    else:
        raise ValueError(f"unknown recipe kind {recipe.kind!r}")

    res = ImageryResult(
        rgba=rgba, bounds=bounds, scan_start=anchor.scan_start,
        product=anchor.product, bucket=anchor.bucket, s3_key=anchor.s3_key,
        generic_channel=recipe.key,
        enhancement=recipe.enhancement or recipe.kind)
    if recipe.bt_band:
        try:
            res.bt_grid, res.bt_dims = _decimate_bt(
                sampled[recipe.bt_band] - 273.15, bounds)
        except Exception:   # noqa: BLE001  (inspector is best-effort, never blocks)
            res.bt_grid, res.bt_dims = None, None
    return res


def produce_truecolor(entry, time=None, nearest=True):
    """True color / GeoColor-lite through the FROZEN pipeline VERBATIM:
    satellites.fetch_true_color -> truecolor.assemble_truecolor (CIMSS synthetic
    green, CIRA Rayleigh order, tone curve, night IR fade). Its output is
    already a regular lat/lon grid over the bbox (row 0 = north), so it maps
    straight into the pyramid. Only the resolution caps are raised (both are
    designed env overrides); zero code-path changes."""
    from satellites import GOESEastSatellite

    sat = GOESEastSatellite()
    t = time or dt.datetime.now(UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    bbox = list(entry.sector_bbox)

    async def _go():
        resolved = await sat.find_file(t, "visible_red", bbox,
                                       nearest_to_target=nearest)
        return resolved, await sat.fetch_true_color(bbox, resolved)

    # TRUECOLOR_MAX_PX is a module-level constant read at import (unlike
    # SAT_MAX_PX_PER_AXIS, which is re-read per crop), so raise it by patching
    # the module attribute -- _compose_true_color_sync resolves the global at
    # call time. Restored in `finally`; the floater/meso default is untouched.
    import satellites as _sats
    _prev_cap = _sats.TRUECOLOR_MAX_PX
    _sats.TRUECOLOR_MAX_PX = int(entry.pyramid_px)
    try:
        with _env_override(SAT_MAX_PX_PER_AXIS=(entry.fetch_max_px or None)):
            resolved, r = asyncio.run(_go())
    finally:
        _sats.TRUECOLOR_MAX_PX = _prev_cap

    rgb = np.asarray(r.cmi, dtype=np.float32)
    rgba = s2_recipes.rgba_from_rgb(rgb, valid=r.geom_valid)
    W, S, E, N = bbox
    return ImageryResult(
        rgba=rgba, bounds=(W, S, E, N), scan_start=r.scan_start,
        product=r.product, bucket=r.bucket,
        s3_key=getattr(resolved, "s3_key", ""),
        generic_channel="truecolor", enhancement="tat_neon")
