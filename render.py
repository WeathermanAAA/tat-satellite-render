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
from goes import FetchResult, goes_sat_label

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


def render_png(
    data: FetchResult,
    bbox: list[float],
    channel: int,
    time_str: str,
    enhancement: str,
    downsample: int = 1,
) -> bytes:
    enh = get_enhancement(enhancement)

    # Pixel-budget stride. App-layer (compute_downsample_factor) sets this
    # based on raw bbox×channel so output_pixels ≤ PIXEL_BUDGET. This
    # composes with the goes.py fetch-time stride: both layers cap output
    # size, the more aggressive of the two wins.
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

    # Normalize the CMI to 0..1 according to channel + enhancement
    if channel == 2:
        # Visible: reflectance 0..1
        norm = normalize_visible(cmi)
        if enh["kind"] == "ir":
            # IR colormap on visible doesn't make physical sense; use grayscale instead.
            cmap = plt.get_cmap("gray")
        else:
            cmap = plt.get_cmap("gray")
    else:
        # IR: brightness temperature in Kelvin (or convert from C if needed)
        bt = cmi
        if data.units in ("C", "celsius", "degC"):
            bt = bt + 273.15
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

    # Plot pcolormesh with the (lats, lons) arrays we computed via inverse proj
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
    title_ax.text(
        0.5, 0.5,
        f"{goes_sat_label(data.bucket)} ABI Channel {channel:02d} · {time_str} UTC",
        ha="center", va="center",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.99, 0.5,
        f"{data.product} · {enhancement}",
        ha="right", va="center",
        color=ACCENT_COLOR, fontsize=9,
        transform=title_ax.transAxes,
    )

    # Watermark: top-left of the map axes, mirroring the title strip's
    # right-aligned product label so the two corners balance visually.
    # Translucent dark backing rect keeps it legible over hot pixels.
    ax.text(
        0.01, 0.99,
        f"@WeathermanAAA_  ·  NOAA {goes_sat_label(data.bucket)} ABI",
        ha="left", va="top",
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
