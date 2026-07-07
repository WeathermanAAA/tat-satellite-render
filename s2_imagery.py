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
import datetime as dt
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
                 "generic_channel", "enhancement")

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
    resolved, r = asyncio.run(_fetch_async(sat, entry, t, nearest))
    rgba, bounds = render_imagery_rgba(
        r.cmi, r.lats, r.lons, entry.render_enhancement, entry.pyramid_px,
        units=getattr(r, "units", "K"))
    return ImageryResult(
        rgba=rgba, bounds=bounds, scan_start=r.scan_start, product=r.product,
        bucket=r.bucket, s3_key=getattr(resolved, "s3_key", ""),
        generic_channel=entry.render_channel, enhancement=entry.render_enhancement)
