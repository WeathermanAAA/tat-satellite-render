"""Matplotlib + cartopy render pipeline.

Takes a FetchResult (cropped CMI on the geos grid with companion lat/lon arrays),
projects to PlateCarree, applies the requested enhancement, and produces a
clean dark-themed PNG with title strip + footer credit.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from colormaps import (
    get_enhancement,
    normalize_ir,
    normalize_visible,
)
from satellites import FetchResult

log = logging.getLogger("tat-satellite.render")

DARK_BG = "#0a0d12"
GRID_COLOR = "#3a4252"
COAST_COLOR = "#7eb6c9"   # cyan — coastlines stand out against IR/visible
BORDER_COLOR = "#e8eef5"  # near-white — political borders, slightly subordinate
TEXT_COLOR = "#e8eef5"
ACCENT_COLOR = "#79f0d6"


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


def _coast_resolution(span_deg: float) -> str:
    """Natural Earth scale matched to bbox size.

    10m on a wide view is the most likely cause of jagged/fragmented
    coastlines because the dense polyline gets aggressively path-clipped
    by matplotlib at viewport scale, producing visible polyline gaps and
    stair-stepping along long edges. Step down to 50m / 110m for wide views.
    """
    if span_deg < 5:
        return "10m"
    if span_deg < 30:
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
) -> bytes:
    # True-color composites carry an H×W×3 RGB array in ``cmi`` (units="rgb")
    # and don't go through the scalar normalize/cmap path.
    is_rgb = data.units == "rgb"
    is_visible = data.units == "1"
    enh = None if is_rgb else get_enhancement(enhancement)

    # Pixel-budget stride. App-layer (compute_downsample_factor) sets this
    # based on raw bbox×channel so output_pixels ≤ PIXEL_BUDGET. This
    # composes with the goes.py fetch-time stride: both layers cap output
    # size, the more aggressive of the two wins. ``cmi[::d, ::d]`` strides the
    # first two axes for both 2D (scalar) and 3D (RGB) arrays.
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

    norm = None
    cmap = None
    bt_min = bt_max = None  # set on the IR/WV (Kelvin) branch -> bottom-left overlay
    if not is_rgb:
        # Normalize the CMI to 0..1 according to data kind + enhancement.
        # We branch on units (not numeric channel) because the visible-red band
        # is 2 on GOES and 3 on AHI — units="1" tags reflectance regardless.
        if is_visible:
            # Visible: reflectance 0..1
            norm = normalize_visible(cmi)
            # Visible imagery is rendered grayscale regardless of enhancement;
            # IR colormaps on reflectance data don't make physical sense.
            cmap = plt.get_cmap("gray")
        else:
            # IR: brightness temperature in Kelvin (or convert from C if needed)
            bt = cmi
            if data.units in ("C", "celsius", "degC"):
                bt = bt + 273.15
            # Capture brightness-temperature extremes over the rendered bbox
            # (raw Kelvin, post-crop/downsample, pre-normalize) for the
            # bottom-left min/max overlay. Guard the fully-off-disk case so
            # np.nanmin/np.nanmax don't warn + return NaN on an all-NaN slice.
            if np.isfinite(bt).any():
                bt_min = float(np.nanmin(bt))
                bt_max = float(np.nanmax(bt))
            if enh["kind"] == "gray":
                # grayscale on IR: invert (cold=white) for readable IR
                t_warm, t_cold = 303.0, 183.0
                x = (t_warm - bt) / (t_warm - t_cold)
                norm = np.clip(x, 0.0, 1.0)
                cmap = plt.get_cmap("gray")
            else:
                norm = normalize_ir(bt, enh["range_k"])
                cmap = enh["cmap"]

        # Mask NaNs (off-disk + invalid pixels)
        norm = np.ma.masked_invalid(norm)

    lon_min, lat_min, lon_max, lat_max = bbox
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    aspect = lon_span / max(lat_span, 1e-6)

    # Figure size: target ~1400 px wide, height by aspect, dpi=110
    fig_w = 12.0
    fig_h = max(4.0, fig_w / max(aspect, 0.3))
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)

    # Layout: title strip on top (~6%), main map fills the rest with a
    # small bottom margin for gridline labels. Watermark moved into the
    # map's top-left corner (see ax.text below) so we no longer reserve a
    # bottom footer strip.
    title_h = 0.06
    bottom_pad = 0.04  # leaves room for x-axis gridline labels
    map_h = 1.0 - title_h - bottom_pad

    ax = fig.add_axes(
        [0.04, bottom_pad, 0.92, map_h], projection=ccrs.PlateCarree()
    )
    ax.set_facecolor(DARK_BG)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Plot with the (lats, lons) arrays we computed via inverse projection.
    if is_rgb:
        # True-color RGB is resampled onto a REGULAR lat/lon grid (see
        # _compose_true_color_sync), so imshow with a PlateCarree extent is
        # exact — no curvilinear warp to honor, and it sidesteps cartopy's
        # GeoQuadMesh.set_array(None) limitation for RGB pcolormesh. NaNs
        # (off-disk) -> black. origin "upper" because row 0 = lat_max.
        # ---- DEGENERATE-RGB GUARD --------------------------------------
        # Truecolor pulls 5 input bands (R/G/B/veggie/clean-IR) and a
        # transient cache race on ANY of them can leave the composite
        # mostly-NaN -- the nan_to_num below would then paint those
        # regions pure black and the render would ship a 200 OK with a
        # black PNG. Detect this and raise so /render returns 500: the
        # poller's retry/skip path discards the frame instead of uploading
        # it, and the next 10-min scan cycle (with the s3fs listings
        # cache aged out) re-renders cleanly. Threshold: >50% NaN or
        # mean-of-valid-channel-max <0.04 (~almost-pure-black even where
        # not NaN) counts as degenerate.
        finite_mask = np.isfinite(cmi).all(axis=-1)
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
            plt.close(fig)
            raise RuntimeError(
                f"truecolor render produced degenerate RGB "
                f"(nan={nan_frac * 100:.0f}%, mean_valid_max={valid_max:.3f}) "
                f"-- likely a cache race on one of the input band listings; "
                f"the next scan cycle will re-render"
            )
        # ----------------------------------------------------------------
        rgb = np.clip(np.nan_to_num(cmi, nan=0.0).astype(np.float32), 0.0, 1.0)
        ax.imshow(
            rgb,
            origin="upper",
            extent=[lon_min, lon_max, lat_min, lat_max],
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            zorder=1,
        )
    else:
        mesh = ax.pcolormesh(
            lons,
            lats,
            norm,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )

    # Coastlines + borders. Resolution scales with bbox; zorder explicitly
    # above pcolormesh (which defaults to ~1.5 in cartopy) so cyan coast
    # never gets painted over by hot cloud tops; full alpha for legibility.
    coast_scale = _coast_resolution(max(lon_span, lat_span))
    ax.add_feature(
        cfeature.COASTLINE.with_scale(coast_scale),
        linewidth=0.6, edgecolor=COAST_COLOR, alpha=1.0, zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(coast_scale),
        linewidth=0.5, edgecolor=BORDER_COLOR, alpha=1.0, zorder=3,
    )

    # Dashed gridlines auto-spaced
    step = _gridline_step(max(lon_span, lat_span))
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        linestyle="--",
        color=GRID_COLOR,
        alpha=0.7,
        xlocs=np.arange(np.floor(lon_min / step) * step, lon_max + step, step),
        ylocs=np.arange(np.floor(lat_min / step) * step, lat_max + step, step),
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"color": TEXT_COLOR, "size": 8}
    gl.ylabel_style = {"color": TEXT_COLOR, "size": 8}

    # Title strip
    title_ax = fig.add_axes([0, 1.0 - title_h, 1.0, title_h])
    title_ax.set_facecolor(DARK_BG)
    title_ax.axis("off")
    # Sensor label: read off FetchResult so it works for both ABI (GOES) and
    # AHI (Himawari) without per-family branching here.
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
    source_label = "JMA" if data.bucket.startswith("noaa-himawari") else "NOAA"
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
    # diagonal mirror of the top-left watermark. IR/WV (Kelvin) renders only
    # -- bt_min/bt_max stay None for visible + true-color, so this no-ops on
    # those paths. Displayed in degrees Celsius (analyst-standard for
    # cloud-top temps); bt_min/bt_max are raw Kelvin, so subtract 273.15.
    if bt_min is not None and bt_max is not None:
        ax.text(
            0.01, 0.01,
            f"min: {bt_min - 273.15:.0f}°C  ·  max: {bt_max - 273.15:.0f}°C",
            ha="left", va="bottom",
            color=ACCENT_COLOR, fontsize=9,
            transform=ax.transAxes,
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=4),
            zorder=10,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
