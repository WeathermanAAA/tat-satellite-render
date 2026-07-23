"""Matplotlib + cartopy render pipeline.

Takes a FetchResult (cropped CMI on the geos grid with companion lat/lon arrays),
projects to PlateCarree, applies the requested enhancement, and produces a
clean dark-themed PNG with title strip, a labeled right-side colorbar, and a
footer credit.
"""

from __future__ import annotations

import io
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from matplotlib.colors import Normalize
from shapely.geometry import shape

from colormaps import get_enhancement, enhancement_norm, normalize_visible
from satellites import FetchResult

log = logging.getLogger("tat-satellite.render")

# admin_1 state/province boundary LINES, vendored as NE 10m geojson (the same
# asset the CycloLab basemap ships). Loaded locally instead of via cartopy's
# runtime downloader, whose admin_1 URL 404s on the deploy host (coastline +
# admin_0 borders download fine there; admin_1 does not).
# .resolve() so a relative __file__ (uvicorn import from CWD) still yields the
# absolute path next to render.py — mirrors cyclolab_basemap's proven pattern.
_STATE_LINES_PATH = Path(__file__).resolve().with_name("cyclolab_ne_10m_states.geojson")


@lru_cache(maxsize=1)
def _state_lines_feature() -> Optional[ShapelyFeature]:
    """admin_1 boundary LINES as a cartopy feature, read once from the vendored
    geojson (no network). Returns None if the asset is missing/empty."""
    if not _STATE_LINES_PATH.exists():
        return None
    gj = json.loads(_STATE_LINES_PATH.read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in gj.get("features", []) if f.get("geometry")]
    if not geoms:
        return None
    return ShapelyFeature(geoms, ccrs.PlateCarree())


def state_lines_status() -> str:
    """Diagnostic: did the vendored admin_1 layer load on this host? Surfaced
    via the X-State-Lines response header to verify the deploy at runtime."""
    p = _STATE_LINES_PATH
    try:
        return "loaded" if _state_lines_feature() is not None else f"absent(exists={p.exists()},p={p})"
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}:{e}"[:160]

DARK_BG = "#0a0d12"
GRID_COLOR = "#3a4252"
# Coast/border style (2026-07-12, Andrew's call — OVERRIDES the 2026-07-11
# cyan/white restyle): coastlines, borders, state lines and the halo are all
# BLACK. The halo under-stroke geometry is kept (it just reads as a slightly
# heavier black line), so a future restyle is a constants-only change again.
COAST_COLOR = "#000000"   # black — landmass outlines
BORDER_COLOR = "#000000"  # black — political borders
LINE_HALO = "#000000"     # black halo under both (thin: +~1.2 px)
TEXT_COLOR = "#e8eef5"
ACCENT_COLOR = "#79f0d6"
MUTED_COLOR = "#9199a4"


def _fill_coord_nan(a: np.ndarray) -> np.ndarray:
    """Nearest-finite fill for inverse-projection coordinate arrays.

    A meso sector steered onto the disk limb (e.g. GOES-18 M2 over the
    Bering Sea) has bbox corners past the geometric horizon — those pixels
    have no lat/lon. pcolormesh requires finite coords everywhere, so fill
    NaNs with the nearest finite value along each row (then column). The
    affected cells are masked in the data, and repeating a coordinate makes
    the phantom quads zero-area, so nothing fictional is drawn."""
    out = a.astype(np.float64, copy=True)
    for _ in range(2):          # pass 1: along rows; pass 2: along columns
        cols = np.arange(out.shape[1])
        rows = np.arange(out.shape[0])[:, None]
        # Forward fill: index of the last finite column so far (a leading
        # NaN run pulls col 0's value, which is NaN — the mirrored pass
        # below catches it).
        idx = np.where(np.isfinite(out), cols, 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = out[rows, idx]
        # Backward fill = forward fill on the mirror.
        rev = out[:, ::-1]
        idx = np.where(np.isfinite(rev), cols, 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = rev[rows, idx][:, ::-1]
        out = out.T
    return out


def _effective_extent(
    lats: np.ndarray,
    lons: np.ndarray,
    valid: "np.ndarray | None",
    bbox: list,
    lon_span_req: float,
) -> tuple[float, float, float, float]:
    """(lon_lo, lon_hi, lat_lo, lat_hi) of valid-data ∩ bbox, with longitudes
    UNWRAPPED from the bbox's west edge (monotonic even across ±180).

    A disk-limb meso sector (e.g. GOES-18 M2 over the Bering) carries a big
    off-disk corner in its lat/lon bounding box; framing the full request
    renders mostly-empty black margins. Cropping to the valid bounding box
    keeps the frame tight around real imagery while a genuine limb edge
    inside it still shows. Sectors whose data fills the request (every
    normal sector + storm floater) clip to exactly the request — unchanged
    by construction."""
    lon_min, lat_min, lon_max, lat_max = bbox
    fallback = (lon_min, lon_min + lon_span_req, lat_min, lat_max)
    if valid is None or not valid.any():
        return fallback
    vlat = lats[valid]
    vlon_uw = lon_min + ((lons[valid] - lon_min) % 360.0)
    lo = max(lon_min, float(vlon_uw.min()))
    hi = min(lon_min + lon_span_req, float(vlon_uw.max()))
    la = max(lat_min, float(vlat.min()))
    lb = min(lat_max, float(vlat.max()))
    if hi - lo < 0.5 or lb - la < 0.5:   # degenerate crop -> keep request
        return fallback
    return lo, hi, la, lb


def _gridline_xlocs(lon_min: float, lon_max_uw: float, step: float) -> np.ndarray:
    """Meridian locations for the graticule. ``lon_max_uw`` may exceed 180 (the
    unwrapped east edge of an antimeridian-crossing bbox) — locs are generated
    in unwrapped space so the sequence is continuous across the dateline, then
    wrapped into [-180, 180] for cartopy's Gridliner (which speaks wrapped
    PlateCarree longitudes). Non-crossing boxes reproduce the legacy arange."""
    locs = np.arange(np.floor(lon_min / step) * step, lon_max_uw + step, step)
    if lon_max_uw > 180.0:
        locs = np.unique(((locs + 180.0) % 360.0) - 180.0)
    return locs


def _gridline_step(span: float) -> float:
    """Pick a sane gridline interval (degrees) for the given bbox span."""
    if span <= 2:
        return 0.5
    if span <= 5:
        return 1.0
    if span <= 12:
        return 2.0
    if span <= 25:
        return 5.0
    return 10.0


def _map_geometry(bbox: list[float]):
    """Crossing-aware map geometry for a ``[W, S, E, N]`` bbox.

    Returns ``(projection, extent, lon_span, crossing)``:
      * ``projection`` — the AXES projection. PlateCarree(central_longitude=180)
        when the bbox crosses the antimeridian (normalized convention:
        lon_max < lon_min), so the extent is continuous in axes space; the
        plain PlateCarree otherwise (existing behavior, byte-identical).
      * ``extent`` — [lon_min, lon_max_unwrapped, lat_min, lat_max] to pass to
        ``set_extent(..., crs=ccrs.PlateCarree())``; cartopy transforms the
        unwrapped east edge into the crossing projection correctly.
      * ``lon_span`` — positive unwrapped span (drives aspect / coast scale /
        gridline step).
    Data plotting stays ``transform=ccrs.PlateCarree()`` in both cases: with a
    180-centered axes projection the transformed mesh is continuous across the
    dateline (the ±180 jump in wrapped source longitudes lands at the axes
    center, not at a seam), so pcolormesh needs no special handling.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    crossing = lon_max < lon_min
    lon_max_uw = lon_max + 360.0 if crossing else lon_max
    proj = ccrs.PlateCarree(central_longitude=180.0) if crossing else ccrs.PlateCarree()
    extent = [lon_min, lon_max_uw, lat_min, lat_max]
    return proj, extent, lon_max_uw - lon_min, crossing


def _unwrap_lons(lons, lon_min: float):
    """Make source longitudes CONTINUOUS across the dateline for a crossing
    bbox. Wrapped [-180, 180] lons (the AHI inverse projection wraps; GOES-West
    emits sub-sat-relative values) carry a ±360 jump at the dateline column —
    cartopy's pcolormesh wrap-detection then mangles the jump-spanning cells
    into misplaced horizontal smears (observed on the first live swpac
    backdrop), and _guard_mesh_coords' nanmean fill lands far outside the
    window. Everything west of the bbox's west edge is really the >180 side:
    shift it up by 360. PlateCarree transforms are periodic, so unwrapped
    values >180 plot exactly right."""
    lons = np.asarray(np.ma.filled(lons, np.nan), dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(lons < lon_min - 1e-9, lons + 360.0, lons)


def _coast_resolution(span_deg: float) -> str:
    """Natural Earth scale matched to bbox size.

    10m up to 90°: every meso sector (incl. the ~70°-wide GOES-18 M2 limb
    box, the view that exposed the blockiness) and every storm floater gets
    crisp coastlines — 50m at these zooms reads visibly blocky, especially
    at high latitude. The old jagged path-clipping concern applies to
    genuinely wide (near-disk) views, which still step down. The 10m
    geometry caches in cartopy after the first draw (~+6 s once, then ~0).
    """
    if span_deg < 90:
        return "10m"
    if span_deg < 180:
        return "50m"
    return "110m"


# Saffir-Simpson + tropical-status colors for the title-strip storm badge.
# Per Andrew's spec (2026-05-28): TD neon blue, TS lime, C1 yellow, C2 amber,
# C3 red, C4 pink, C5 magenta/purple. Tuned to read clearly against DARK_BG
# (each is used as a tinted/translucent bbox face for the text, not a fill).
_SS_COLORS: dict[str, str] = {
    "TD": "#3b82f6",   # tropical depression — neon blue
    "TS": "#84cc16",   # tropical storm — lime green
    "C1": "#fde047",   # cat 1 — yellow
    "C2": "#f59e0b",   # cat 2 — amber
    "C3": "#dc2626",   # cat 3 — red
    "C4": "#ec4899",   # cat 4 — pink
    "C5": "#a855f7",   # cat 5 — magenta / purple
    "EX": "#9199a4",   # extratropical / post-tropical / remnant low — gray
}


def _ss_category(nature: Optional[str], wind_kt: Optional[float]) -> str:
    """Map (nature, wind) -> short Saffir-Simpson-ish category label."""
    n = (nature or "").upper()
    # Non-tropical natures keep their own label (and the gray color).
    if n in ("EX", "PT", "DB", "WV", "LO", "SD", "SS"):
        return n or "EX"
    w = wind_kt or 0.0
    if w < 34:
        return "TD"
    if w < 64:
        return "TS"
    if w < 83:
        return "C1"
    if w < 96:
        return "C2"
    if w < 113:
        return "C3"
    if w < 137:
        return "C4"
    return "C5"


def render_png(
    data: FetchResult,
    bbox: list[float],
    channel: int,
    time_str: str,
    enhancement: str,
    downsample: int = 1,
    storm: Optional[dict] = None,
    coastlines: bool = True,
    gridlines: bool = True,
    dpi: int = 110,
) -> bytes:
    # ``coastlines`` draws coastlines + political borders; ``gridlines`` draws the
    # labeled lat/lon graticule. Both default True (the standard look); the custom-
    # zoom page can switch either off for clean imagery. (The floater/meso loop
    # frames never pass them -> always on, unchanged.)
    # True-color composites carry an H×W×3 RGB array in ``cmi`` (units="rgb")
    # and don't go through the scalar normalize/cmap path.
    is_rgb = data.units == "rgb"
    is_visible = data.units == "1"
    enh = None if is_rgb else get_enhancement(enhancement)

    # Pixel-budget stride. App-layer (compute_downsample_factor) sets this
    # based on raw bbox×channel so output_pixels ≤ the tier budget. For SCALAR
    # products the stride happens HERE; for TRUE COLOR it is applied in the fetch
    # (satellites._stride_tc_grids, before the per-pixel recipe) and this is
    # called with downsample=1, so the RGB is never double-strided.
    # ``cmi[::d, ::d]`` strides the first two axes for both 2D and 3D arrays.
    cmi = data.cmi
    lats = data.lats
    lons = data.lons
    if downsample > 1:
        log.info(
            "downsampled bbox by factor %d to stay within pixel budget "
            "(in shape %s -> out shape %s)",
            downsample,
            cmi.shape,
            cmi[::downsample, ::downsample].shape,
        )
        cmi = cmi[::downsample, ::downsample]
        lats = lats[::downsample, ::downsample]
        lons = lons[::downsample, ::downsample]

    # ---- Build the scalar plot field (skipped for true-color RGB) ----------
    # IR/WV  -> plot real brightness temperature in °C with the enhancement's
    #           cmap + a fresh Normalize over its °C domain. The colorbar ticks
    #           are then physical °C with no remap.
    # visible -> sqrt-stretched reflectance (0..1) in grayscale; colorbar shows
    #           reflectance %.
    plot_field = None
    plot_cmap = None
    plot_cnorm = None
    cbar_ticks = None
    cbar_ticklabels = None
    cbar_label = None
    bt_min_c = bt_max_c = None  # IR/WV only -> bottom-left min/max overlay
    if not is_rgb:
        if is_visible:
            refl = normalize_visible(cmi)
            plot_field = np.ma.masked_invalid(refl)
            plot_cmap = plt.get_cmap("gray")
            plot_cnorm = Normalize(vmin=0.0, vmax=1.0)
            cbar_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
            cbar_ticklabels = ["0", "25", "50", "75", "100"]
            cbar_label = "Reflectance (%)"
        else:
            # IR/WV: brightness temperature. Source is Kelvin unless tagged C.
            bt = cmi
            if data.units in ("C", "celsius", "degC"):
                bt = bt + 273.15
            bt_c = bt - 273.15
            # ---- DEGENERATE-FRAME GUARD (scalar IR/WV) --------------------
            # A partial satellite-segment fetch (missing AHI/ABI tiles -- the
            # same s3fs listings-cache race that bit truecolor) leaves most of
            # the bbox NaN, so the frame renders as a mostly-empty "strip" and
            # the poller would upload it -> glitchy loop. Mirror the truecolor
            # guard: raise so /render returns 500 and the poller retries 3x
            # then skips, and the next scan cycle (cache aged out) renders
            # cleanly. Threshold 55% NaN: a storm floater fully inside the disk
            # is ~0% NaN, a partial fetch is ~80%+, so this only trips on the
            # broken frames (off-disk on-demand boxes are already 422'd by the
            # satellite picker before reaching here).
            # Counted over GEOMETRICALLY VALID pixels only: a disk-limb meso
            # sector (e.g. GOES-18 M2 over the Bering) legitimately has a
            # big off-disk corner whose coords are NaN — that's geometry,
            # not a broken fetch, and must not permanently 500 the sector.
            valid_geom = np.isfinite(lats) & np.isfinite(lons)
            n_valid = float(valid_geom.sum())
            if n_valid == 0:
                raise RuntimeError("bbox has no on-disk pixels to render")
            nan_frac = float(
                (~np.isfinite(bt_c) & valid_geom).sum() / n_valid
            )
            if nan_frac > 0.55:
                log.warning(
                    "scalar IR/WV degenerate (nan=%.0f%%) -- bailing out so the "
                    "poller doesn't ship a partial-fetch frame", nan_frac * 100.0,
                )
                raise RuntimeError(
                    f"scalar render produced a mostly-NaN field "
                    f"(nan={nan_frac * 100:.0f}%) -- likely a partial satellite "
                    f"segment fetch; the next scan cycle will re-render"
                )
            # --------------------------------------------------------------
            if np.isfinite(bt_c).any():
                bt_min_c = float(np.nanmin(bt_c))
                bt_max_c = float(np.nanmax(bt_c))
            plot_field = np.ma.masked_invalid(bt_c)
            plot_cmap = enh["cmap"]
            plot_cnorm = enhancement_norm(enhancement)  # fresh, not shared
            cbar_ticks = list(enh["ticks"])
            cbar_ticklabels = [str(t) for t in enh["ticks"]]
            cbar_label = enh.get("cbar_label", "Brightness Temperature (°C)")

    lon_min, lat_min, lon_max, lat_max = bbox
    # Antimeridian crossing (lon_max < lon_min, e < w convention): draw in a
    # PlateCarree frame re-centered on the bbox so the ±180 seam falls on the
    # far side of the planet. Data keeps transform=PlateCarree(0) — cartopy
    # wraps it into the shifted frame continuously, so a GOES-18 meso sector
    # steered over the Bering Sea renders seamlessly.
    crosses = lon_max < lon_min
    lon_span_req = (lon_max - lon_min) % 360.0 or 360.0

    # ---- per-pixel validity (degenerate guards + extent crop) -------------
    if is_rgb:
        # ---- DEGENERATE-RGB GUARD --------------------------------------
        # Truecolor pulls 5 input bands (R/G/B/veggie/clean-IR) and a
        # transient cache race on ANY of them can leave the composite
        # mostly-NaN -- nan_to_num at draw time would then paint those
        # regions pure black and the render would ship a 200 OK with a
        # black PNG. Detect this and raise so /render returns 500: the
        # poller's retry/skip path discards the frame instead of uploading
        # it, and the next 10-min scan cycle (with the s3fs listings
        # cache aged out) re-renders cleanly. Threshold: >50% NaN or
        # mean-of-valid-channel-max <0.04 (~almost-pure-black even where
        # not NaN) counts as degenerate.
        finite_mask = np.isfinite(cmi).all(axis=-1)
        # NaN fraction counted over geometrically-valid (on-disk) pixels
        # only, when the fetch supplied the mask: a disk-limb sector's big
        # off-disk corner is geometry, not a broken fetch (mirrors the
        # scalar guard). The mask strides with the data above.
        geom = data.geom_valid
        if geom is not None and downsample > 1:
            geom = geom[::downsample, ::downsample]
        if geom is not None:
            n_geom = float(geom.sum())
            if n_geom == 0:
                raise RuntimeError("bbox has no on-disk pixels to render")
            nan_frac = float((~finite_mask & geom).sum() / n_geom)
        else:
            nan_frac = float((~finite_mask).mean())
        if finite_mask.any():
            valid_max = float(
                np.clip(cmi[finite_mask], 0.0, 1.0).max(axis=-1).mean()
            )
        else:
            valid_max = 0.0
        if nan_frac > 0.50 or valid_max < 0.04:
            log.warning(
                "truecolor RGB degenerate (nan=%.0f%%, mean valid max=%.3f) "
                "-- bailing out so the poller doesn't ship a black frame",
                nan_frac * 100.0, valid_max,
            )
            raise RuntimeError(
                f"truecolor render produced degenerate RGB "
                f"(nan={nan_frac * 100:.0f}%, mean_valid_max={valid_max:.3f}) "
                f"-- likely a cache race on one of the input band listings; "
                f"the next scan cycle will re-render"
            )
        # ----------------------------------------------------------------
        valid_px = finite_mask & geom if geom is not None else finite_mask
    else:
        # Disk-limb sectors: off-disk pixels inverse-project to NaN lat/lon,
        # which pcolormesh hard-rejects. Mask those cells in the field and
        # nearest-fill the coords — the repeated coordinates collapse the
        # phantom quads to zero area, so only real data draws.
        coord_bad = ~(np.isfinite(lats) & np.isfinite(lons))
        valid_px = ~np.ma.getmaskarray(plot_field) & ~coord_bad
        if coord_bad.any():
            plot_field = np.ma.masked_where(coord_bad, plot_field)
            lats = _fill_coord_nan(lats)
            lons = _fill_coord_nan(lons)

    # ---- frame geometry: tighten the view to the valid-data extent --------
    eff_lo, eff_hi, eff_lat_lo, eff_lat_hi = _effective_extent(
        lats, lons, valid_px, bbox, lon_span_req
    )
    lon_span = eff_hi - eff_lo
    lat_span = eff_lat_hi - eff_lat_lo
    aspect = lon_span / max(lat_span, 1e-6)

    # Figure size: fixed 12 in wide, height by aspect. The OUTPUT RESOLUTION is
    # the per-tier ``dpi`` at savefig (default 110 -> ~1320 px; low ~70 -> ~840 px;
    # high ~200 -> ~2400 px). figsize is held constant so layout proportions +
    # font sizes scale uniformly and ALL chrome (vector text/lines) renders crisp
    # at the tier dpi -- never bitmap-resized. default dpi 110 == today (byte-
    # identical); the webp LOOP path always passes 110 (then transcodes to 1056).
    fig_w = 12.0
    fig_h = max(4.0, fig_w / max(aspect, 0.3))
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)

    # Layout: title strip on top (~6%), main map fills the rest with a small
    # bottom margin for gridline labels. A labeled vertical colorbar sits in a
    # reserved right margin for every scalar (non-RGB) product; true color has
    # no colorbar so the map uses the full width.
    title_h = 0.06
    bottom_pad = 0.04  # leaves room for x-axis gridline labels
    map_h = 1.0 - title_h - bottom_pad
    show_cbar = not is_rgb
    map_w = 0.84 if show_cbar else 0.92

    # The map frame is the EFFECTIVE (valid-data) extent; the image keeps its
    # own full-request extent (img_extent) so imshow pixel registration is
    # untouched and set_extent simply crops the view.
    if crosses:
        center_uw = lon_min + lon_span_req / 2.0
        center_lon = ((center_uw + 180.0) % 360.0) - 180.0
        map_crs = ccrs.PlateCarree(central_longitude=center_lon)
        view_extent = [eff_lo - center_uw, eff_hi - center_uw,
                       eff_lat_lo, eff_lat_hi]
        img_extent = [-lon_span_req / 2.0, lon_span_req / 2.0,
                      lat_min, lat_max]
    else:
        map_crs = ccrs.PlateCarree()
        view_extent = [eff_lo, eff_hi, eff_lat_lo, eff_lat_hi]
        img_extent = [lon_min, lon_max, lat_min, lat_max]

    ax = fig.add_axes([0.04, bottom_pad, map_w, map_h], projection=map_crs)
    ax.set_facecolor(DARK_BG)
    ax.set_extent(view_extent, crs=map_crs)

    mesh = None
    # Plot with the (lats, lons) arrays we computed via inverse projection.
    if is_rgb:
        # True-color RGB is resampled onto a REGULAR lat/lon grid (see
        # _compose_true_color_sync), so imshow with a PlateCarree extent is
        # exact — no curvilinear warp to honor, and it sidesteps cartopy's
        # GeoQuadMesh.set_array(None) limitation for RGB pcolormesh. NaNs
        # (off-disk) -> black. origin "upper" because row 0 = lat_max.
        # ``img_extent`` in map_crs coords: the composite grid is regular over
        # the UNWRAPPED lon range (see _compose_true_color_sync), which in the
        # re-centered crossing frame is exactly [-span/2, +span/2] — identity
        # transform either way, so imshow stays a fast non-warping draw.
        rgb = np.clip(np.nan_to_num(cmi, nan=0.0).astype(np.float32), 0.0, 1.0)
        ax.imshow(
            rgb,
            origin="upper",
            extent=img_extent,
            transform=map_crs,
            interpolation="nearest",
            zorder=1,
        )
    else:
        # Masked / non-finite geolocation (off-disk limb; early-era ABI
        # sectors) would raise inside pcolormesh -- same guard as the
        # backdrop path (one guard, both call sites).
        if crossing:
            lons = _unwrap_lons(lons, lon_min)
        lons, lats, plot_field = _guard_mesh_coords(lons, lats, plot_field)
        # ---- REGULAR-GRID FAST PATH (Time Machine archive tiers) --------
        # GridSat-B1 / GridSat-GOES / MergIR (and GOES fetch_regular) are
        # uniform lat/lon grids that the fetchers meshgrid into 2-D coords;
        # pcolormesh then transforms MILLIONS of quad vertices per frame
        # (seconds each -- the dominant server cost of a 25-frame archive
        # window). A separable uniform grid renders identically via imshow
        # with a cell-EDGE extent (shading="auto" places edges at centers
        # ± half-step; matched exactly here) in milliseconds. The native
        # geos path (floaters / meso / live sectors -- genuinely
        # curvilinear 2-D coords) can never satisfy the detector, so the
        # frozen-renderer guarantee is untouched: this branch simply never
        # fires for it. Antimeridian-crossing boxes stay on pcolormesh
        # (axes central_longitude=180 vs a PlateCarree(0) image transform
        # would trigger cartopy's warp -- the same reason the true-color
        # imshow path is 422-gated for crossing).
        axes1d = None if crossing else _regular_grid_axes(lons, lats)
        if axes1d is not None:
            lon1, lat1 = axes1d
            dlon = (lon1[-1] - lon1[0]) / max(len(lon1) - 1, 1)
            dlat = (lat1[-1] - lat1[0]) / max(len(lat1) - 1, 1)
            ext_img = [
                lon1[0] - dlon / 2.0, lon1[-1] + dlon / 2.0,
                min(lat1[0], lat1[-1]) - abs(dlat) / 2.0,
                max(lat1[0], lat1[-1]) + abs(dlat) / 2.0,
            ]
            mesh = ax.imshow(
                np.ma.masked_invalid(plot_field),
                origin="lower" if dlat > 0 else "upper",
                extent=ext_img,
                cmap=plot_cmap,
                norm=plot_cnorm,
                transform=ccrs.PlateCarree(),
                interpolation="nearest",
                zorder=1,
            )
        else:
            mesh = ax.pcolormesh(
                lons,
                lats,
                plot_field,
                cmap=plot_cmap,
                norm=plot_cnorm,
                shading="auto",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

    # Coastlines + borders. Resolution scales with bbox; zorder explicitly
    # above pcolormesh (which defaults to ~1.5 in cartopy) so cyan coast
    # never gets painted over by hot cloud tops; full alpha for legibility.
    if coastlines:
        coast_scale = _coast_resolution(max(lon_span, lat_span))
        # halo under-strokes first (same geometry, slightly wider, below)…
        ax.add_feature(
            cfeature.COASTLINE.with_scale(coast_scale),
            linewidth=2.2, edgecolor=LINE_HALO, alpha=0.9, zorder=3,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale(coast_scale),
            linewidth=1.6, edgecolor=LINE_HALO, alpha=0.9, zorder=3,
        )
        # …then the legible light strokes on top
        ax.add_feature(
            cfeature.COASTLINE.with_scale(coast_scale),
            linewidth=1.0, edgecolor=COAST_COLOR, alpha=1.0, zorder=3.02,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale(coast_scale),
            linewidth=0.7, edgecolor=BORDER_COLOR, alpha=1.0, zorder=3.02,
        )
        # State/province (admin_1) boundary LINES — the internal-boundary-only
        # dataset (not the admin_1 *lakes* polygons), so it never re-traces the
        # coastline. Same halo'd off-white as the country borders, a touch
        # thinner so US/MX/AU state lines read as subtle landfall context.
        # Loaded from the vendored geojson (see _state_lines_feature) so it
        # works on the deploy host. Guarded: a missing asset degrades to "no
        # state lines" rather than a failed frame.
        try:
            states = _state_lines_feature()
            if states is not None:
                ax.add_feature(
                    states, linewidth=1.1, edgecolor=LINE_HALO,
                    facecolor="none", alpha=0.9, zorder=3,
                )
                ax.add_feature(
                    states, linewidth=0.45, edgecolor=BORDER_COLOR,
                    facecolor="none", alpha=1.0, zorder=3.02,
                )
        except Exception as e:  # noqa: BLE001 — never let admin_1 break a frame
            log.warning("state borders skipped: %s", e)

    # Dashed gridlines auto-spaced (toggleable) over the EFFECTIVE extent;
    # crossing frames lay xlocs on the unwrapped range then wrap to ±180.
    if gridlines:
        # Dashed gridlines auto-spaced over the EFFECTIVE extent. Crossing frames
        # lay xlocs out on the unwrapped lon range then wrap to ±180 (true
        # longitudes, so the gridliner labels them correctly in the re-centered
        # frame); plain frames keep the raw values — wrapping would map a bbox
        # edge at exactly 180 to -180 and silently drop that meridian's gridline.
        step = _gridline_step(max(lon_span, lat_span))
        xlocs = np.arange(
            np.floor(eff_lo / step) * step, eff_hi + step, step
        )
        if crosses:
            xlocs = ((xlocs + 180.0) % 360.0) - 180.0
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            linestyle="--",
            color=GRID_COLOR,
            alpha=0.7,
            xlocs=xlocs,
            ylocs=np.arange(np.floor(eff_lat_lo / step) * step,
                            eff_lat_hi + step, step),
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"color": TEXT_COLOR, "size": 8}
        gl.ylabel_style = {"color": TEXT_COLOR, "size": 8}

    # Right-side colorbar (every scalar product). Lives in the reserved right
    # margin; physical °C ticks for IR/WV, reflectance % for visible.
    if show_cbar and mesh is not None:
        cbar_ax = fig.add_axes([0.905, bottom_pad + 0.04, 0.016, map_h - 0.08])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8, length=3)
        cbar.outline.set_edgecolor(GRID_COLOR)
        cbar.set_label(cbar_label, color=MUTED_COLOR, fontsize=8)

    # Title strip
    title_ax = fig.add_axes([0, 1.0 - title_h, 1.0, title_h])
    title_ax.set_facecolor(DARK_BG)
    title_ax.axis("off")
    # Sensor label: read off FetchResult so it works for both ABI (GOES) and
    # AHI (Himawari) without per-family branching here. The GridSat-B1 deep
    # archive gets an HONEST era title — actual source, channel, cadence and
    # resolution — so an old frame can never imply modern imagery.
    is_gridsat = data.bucket.startswith("noaa-cdr-gridsat")
    is_mergir = data.bucket == "gesdisc-mergir"
    is_gg = data.bucket == "ncei-gridsat-goes"
    if is_gg:
        # per-satellite GOES-era tier: name the ACTUAL satellite + channel +
        # cadence + resolution (native 1 km GVAR is order-staged only — never
        # imply it; the visible channel says its 4 km is subsampled)
        sensor_label = "GOES Imager (GridSat-GOES)"
        gg_chan = {1: "0.65 µm visible (4 km, from 1 km)",
                   2: "3.9 µm shortwave IR",
                   3: "6.5 µm water vapor",
                   4: "10.7 µm IR window"}.get(channel, f"ch{channel}")
        center_title = (
            f"{data.sat_name} · GridSat-GOES · {gg_chan} · hourly · ~4 km "
            f"· {time_str} UTC")
    elif is_mergir:
        sensor_label = "merged geostationary IR"
        center_title = (
            f"NASA MergIR · 11 µm IR window · 30-min · ~4 km · {time_str} UTC")
    elif is_gridsat:
        sensor_label = "geostationary IR composite"
        gs_chan = "11 µm IR window" if channel == 1 else "6.7 µm water vapor"
        center_title = (
            f"GridSat-B1 · {gs_chan} · 3-hourly · ~8 km · {time_str} UTC")
    else:
        sensor_label = "AHI" if data.bucket.startswith("noaa-himawari") else "ABI"
        center_title = (
            f"{data.sat_name} {sensor_label} True Color · {time_str} UTC"
            if is_rgb
            else f"{data.sat_name} {sensor_label} Channel {channel:02d} · {time_str} UTC"
        )
    title_ax.text(
        0.5, 0.5,
        center_title,
        ha="center", va="center",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.99, 0.5,
        f"{data.product} · {'true color' if is_rgb else enhancement}",
        ha="right", va="center",
        color=ACCENT_COLOR, fontsize=9,
        transform=title_ax.transAxes,
    )

    # Storm badge (left of title strip) — only when /render is called with
    # storm context (poller path). Format:
    #   JANGMI · TS · 35 kt · 998 mb
    # Color-coded by Saffir-Simpson category as a tinted background pill.
    if storm:
        name = (storm.get("name") or "").upper()[:18]
        wind_kt = storm.get("wind_kt")
        pressure_mb = storm.get("pressure_mb")
        nature = storm.get("nature")
        cat = _ss_category(nature, wind_kt)
        cat_color = _SS_COLORS.get(cat, _SS_COLORS["EX"])
        parts = [name, cat]
        if wind_kt is not None:
            parts.append(f"{int(round(wind_kt))} kt")
        if pressure_mb is not None:
            parts.append(f"{int(round(pressure_mb))} mb")
        badge_text = "  ·  ".join(p for p in parts if p)
        title_ax.text(
            0.01, 0.5,
            badge_text,
            ha="left", va="center",
            color=TEXT_COLOR, fontsize=10, fontweight="bold",
            transform=title_ax.transAxes,
            bbox=dict(
                facecolor=cat_color, alpha=0.22, edgecolor=cat_color,
                linewidth=1.0, boxstyle="round,pad=0.35",
            ),
        )

    # Watermark: top-left of the map axes, mirroring the title strip's
    # right-aligned product label so the two corners balance visually.
    # Translucent dark backing rect keeps it legible over hot pixels.
    source_label = ("NASA" if is_mergir
                    else "NOAA NCEI" if is_gg
                    else "NOAA CDR" if is_gridsat
                    else "JMA" if data.bucket.startswith("noaa-himawari")
                    else "NOAA")
    ax.text(
        0.01, 0.99,
        f"@WeathermanAAA_  ·  {source_label} {data.sat_name} {sensor_label}",
        ha="left", va="top",
        color=ACCENT_COLOR, fontsize=9,
        transform=ax.transAxes,
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=4),
        zorder=10,
    )

    # Brightness-temperature min/max readout: bottom-left of the map, the
    # diagonal mirror of the top-left watermark. IR/WV only -- bt_min_c/
    # bt_max_c stay None for visible + true-color, so this no-ops on those
    # paths. Displayed in °C (analyst-standard for cloud-top temps).
    if bt_min_c is not None and bt_max_c is not None:
        ax.text(
            0.01, 0.01,
            f"min: {bt_min_c:.0f}°C  ·  max: {bt_max_c:.0f}°C",
            ha="left", va="bottom",
            color=ACCENT_COLOR, fontsize=9,
            transform=ax.transAxes,
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=4),
            zorder=10,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _erode1(mask: np.ndarray) -> np.ndarray:
    """Erode a boolean mask by one 4-connected cell (True = keep). A kept cell
    that touches a dropped cell in any of the 4 directions is itself dropped --
    so the surviving cells never border the off-disk fill region (see
    _guard_mesh_coords). Pure numpy (no scipy)."""
    e = mask.copy()
    e[1:, :] &= mask[:-1, :]
    e[:-1, :] &= mask[1:, :]
    e[:, 1:] &= mask[:, :-1]
    e[:, :-1] &= mask[:, 1:]
    return e


def _regular_grid_axes(lons, lats):
    """(lon1, lat1) 1-D center axes when the 2-D coord arrays are a UNIFORM,
    SEPARABLE lat/lon meshgrid — else None.

    The archive tiers (GridSat-B1 0.07°, GridSat-GOES 0.04°, MergIR ~0.036°)
    and GOES fetch_regular all build their 2-D coords as np.meshgrid of
    uniform 1-D axes; the render fast path (imshow) is exact only for that
    shape. Full-array separability checks (every lons row identical, every
    lats column identical) cost ~ms on a 5M-cell grid — cheap insurance that
    a curvilinear geos grid (which varies per row/column by construction)
    can NEVER slip through. Uniform-step tolerance absorbs float32 coord
    wobble (~1e-5°) while rejecting real curvature (degrees). Masked
    geolocation = limb handling = never regular. Ascending lons only (the
    crop functions emit ascending, antimeridian-unwrapped axes; unwrapped
    >180 grids are handled by the caller's `crossing` gate)."""
    if getattr(lons, "ndim", 0) != 2 or getattr(lats, "ndim", 0) != 2:
        return None
    if lons.shape != lats.shape or lons.shape[0] < 2 or lons.shape[1] < 2:
        return None
    if isinstance(lons, np.ma.MaskedArray) or isinstance(lats, np.ma.MaskedArray):
        return None
    lon1 = np.asarray(lons[0], dtype=np.float64)
    lat1 = np.asarray(lats[:, 0], dtype=np.float64)
    if not (np.isfinite(lon1).all() and np.isfinite(lat1).all()):
        return None
    if not ((lons == lons[0]).all() and (lats == lats[:, :1]).all()):
        return None
    dlon = np.diff(lon1)
    dlat = np.diff(lat1)
    if not (dlon > 0).all():
        return None
    if not ((dlat > 0).all() or (dlat < 0).all()):
        return None
    tol = 1e-3
    if (dlon.max() - dlon.min()) > tol or (np.abs(dlat).max() - np.abs(dlat).min()) > tol:
        return None
    return lon1, lat1


def _guard_mesh_coords(lons, lats, plot_field):
    """Make (lons, lats, plot_field) safe for pcolormesh when the geolocation
    carries masked / non-finite cells.

    Geostationary lat/lon grids carry NaN at the OFF-DISK limb (and early-era
    ABI sectors -- e.g. GOES-16 CONUS from the 89.5W checkout slot in 2017 --
    carry off-earth pixels INSIDE the sector), and pcolormesh REJECTS
    non-finite values in its x/y (coord) arrays ("x and y arguments ... cannot
    have non-finite values or be of type numpy.ma.MaskedArray"). Mask the
    field at those cells (-> transparent), replace the bad coords with a
    finite in-extent fill so pcolormesh accepts the grid, and erode the valid
    region by one cell so an on-disk edge cell never stretches a quad out to a
    filled coord (shading="auto" averages neighbouring centres -> limb streaks
    otherwise). Coords are coerced to plain NaN-filled float arrays FIRST -- a
    fetch may hand back MASKED lat/lon, and pcolormesh rejects both masked
    coords and non-finite values, so np.where alone (which can stay masked) is
    not enough. Shared by the main scalar render and the backdrop render (one
    guard, both call sites)."""
    lons = np.asarray(np.ma.filled(lons, np.nan), dtype=float)
    lats = np.asarray(np.ma.filled(lats, np.nan), dtype=float)
    xy_ok = np.isfinite(lons) & np.isfinite(lats)
    if not xy_ok.all():
        valid = _erode1(xy_ok & ~np.ma.getmaskarray(plot_field))
        plot_field = np.ma.masked_array(np.ma.getdata(plot_field), mask=~valid)
        fill_lon = float(np.nanmean(lons)) if np.isfinite(lons).any() else 0.0
        fill_lat = float(np.nanmean(lats)) if np.isfinite(lats).any() else 0.0
        lons = np.where(xy_ok, lons, fill_lon)
        lats = np.where(xy_ok, lats, fill_lat)
    return lons, lats, plot_field


def render_backdrop_webp(
    data,
    bbox,
    *,
    enhancement: str = "grayscale",
    downsample: int = 1,
    dpi: int = 110,
    quality: int = 82,
) -> bytes:
    """Bare GRAYSCALE Vis/SWIR satellite backdrop cutout for the ASCAT + MW viewers.

    Day scenes arrive as a VISIBLE channel (reflectance, units "1") and render via
    the sqrt-stretched grayscale recipe; night scenes arrive as SHORT-WAVE IR (or
    clean IR, brightness temperature) and render via the gray BT table. The
    day/night CHOICE of band is made upstream (pick_backdrop_band); only the units
    distinguish the two paths here. ZERO baked chrome either way: ONE full-bleed
    PlateCarree axes (set_aspect('auto') so the data fills the frame edge-to-edge),
    and no coastlines / gridlines / colorbar / title strip / storm badge /
    watermark / min-max overlay. Returns an OPAQUE WebP georeferenced to ``bbox``
    ([W, S, E, N]); the consumer draws it into those exact WGS84 corner bounds and
    owns the single shared graticule, coastline, colorbar, legend and watermark.
    Grayscale ONLY so the colored barbs / MW over it stay legible. Raises
    RuntimeError on a mostly-NaN (degenerate / partial-fetch) field and ValueError
    on a non-gray enhancement on the thermal path.
    """
    cmi = data.cmi
    lats = data.lats
    lons = data.lons
    if downsample > 1:
        cmi = cmi[::downsample, ::downsample]
        lats = lats[::downsample, ::downsample]
        lons = lons[::downsample, ::downsample]

    if getattr(data, "units", "") == "1":
        # VIS (day): visible reflectance -> sqrt-stretched grayscale (mirror of
        # render_png's is_visible branch).
        field = np.asarray(normalize_visible(cmi), dtype=float)
        plot_field = np.ma.masked_invalid(field)
        plot_cmap = plt.get_cmap("gray")
        plot_cnorm = Normalize(vmin=0.0, vmax=1.0)
    else:
        # SWIR / clean IR (night): brightness temperature in °C on the gray table.
        enh = get_enhancement(enhancement)
        if enh.get("kind") != "gray":
            raise ValueError(
                f"render_backdrop_webp requires a grayscale enhancement on the "
                f"thermal path (got {enhancement!r}, kind={enh.get('kind')!r})"
            )
        bt = cmi
        if data.units in ("C", "celsius", "degC"):
            bt = bt + 273.15
        field = np.asarray(bt - 273.15, dtype=float)
        plot_field = np.ma.masked_invalid(field)
        plot_cmap = enh["cmap"]
        plot_cnorm = enhancement_norm(enhancement)  # fresh, not shared

    nan_frac = float(np.isnan(field).mean()) if field.size else 1.0
    if nan_frac > 0.55:
        raise RuntimeError(
            f"backdrop render produced a mostly-NaN field (nan={nan_frac:.0%}) "
            "— bailing so a partial fetch never publishes a near-empty backdrop"
        )

    # Off-disk / masked geolocation guard -- shared with the main scalar
    # render (see _guard_mesh_coords; without it every basin backdrop 500s).
    if bbox[2] < bbox[0]:   # antimeridian-crossing: continuous lons first
        lons = _unwrap_lons(lons, bbox[0])
    lons, lats, plot_field = _guard_mesh_coords(lons, lats, plot_field)

    lon_min, lat_min, lon_max, lat_max = bbox
    proj, extent, lon_span, _crossing = _map_geometry(bbox)
    lat_span = lat_max - lat_min
    aspect = lon_span / max(lat_span, 1e-6)

    # Pixel proportions track the bbox aspect; set_aspect("auto") then fills the
    # axes edge-to-edge so the image corners ARE the bbox corners (no letterbox).
    fig_w = 10.0
    fig_h = max(2.0, fig_w / max(aspect, 0.2))
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_facecolor(DARK_BG)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.axis("off")
    ax.pcolormesh(
        lons, lats, plot_field,
        cmap=plot_cmap, norm=plot_cnorm, shading="auto",
        transform=ccrs.PlateCarree(), rasterized=True,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    return encode_webp(buf.getvalue(), quality)


def encode_webp(png: bytes, quality: int) -> bytes:
    """Re-encode a rendered PNG as lossy WebP at its NATIVE size (no resize).

    The custom-zoom "low" tier uses this for a small download while keeping the
    chrome crisp: the figure is already rendered small (low dpi), so there is no
    bitmap downscale of the composited plot (that is what used to pixelate the
    title/coastlines/colorbar). Only the codec changes. Distinct from
    ``transcode_frame``, which DOWNSCALES the 1320 px loop render to the fixed
    WEBP_FRAME_WIDTH and is the floater/meso poller path -- untouched here.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    out = io.BytesIO()
    im.save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()


def transcode_frame(png: bytes, width: int, quality: int) -> bytes:
    """Downscale a rendered PNG and re-encode it as lossy WebP loop-frame bytes.

    The figure is still rendered at the full 1320 px and downscaled here, NOT
    rendered natively at the target width: Lanczos from the oversampled render
    is exactly what browsers were already doing client-side (1320 -> 1050
    device px inside the 525 CSS px frame box), so this path changes the codec
    and the transfer weight, not the displayed look. Frames are opaque
    (DARK_BG facecolor) -- encode RGB, no alpha.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    if width < im.width:
        height = max(1, round(im.height * width / im.width))
        # Loop frames are square-by-design products (the 12deg storm floater, the
        # square meso sectors). The upstream cartopy figure height occasionally
        # rounds 1px off (a sub-0.1% bbox-aspect wobble as the storm drifts),
        # flipping a frame between e.g. 1056x1056 and 1056x1055 for a multi-hour
        # block. With the live player resizing its <canvas> to each frame's
        # native size, that made the loop visibly jump every cycle (the
        # "seizure"). Snap a near-square result to EXACTLY square so every frame
        # of a loop matches: a <=3px nudge is sub-0.3% (imperceptible), while a
        # genuinely non-square product (>3px off) is left untouched.
        if abs(height - width) <= 3:
            height = width
        im = im.resize((width, height), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()
