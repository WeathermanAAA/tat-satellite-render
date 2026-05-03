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
COAST_COLOR = "#e8eef5"
BORDER_COLOR = "#7eb6c9"
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


def render_png(
    data: FetchResult,
    bbox: list[float],
    channel: int,
    time_str: str,
    enhancement: str,
) -> bytes:
    enh = get_enhancement(enhancement)

    # Normalize the CMI to 0..1 according to channel + enhancement
    cmi = data.cmi
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

    # Layout: title strip on top (~6%), main map, footer (~3%)
    title_h = 0.06
    footer_h = 0.035
    map_h = 1.0 - title_h - footer_h

    ax = fig.add_axes(
        [0.04, footer_h, 0.92, map_h], projection=ccrs.PlateCarree()
    )
    ax.set_facecolor(DARK_BG)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Plot pcolormesh with the (lats, lons) arrays we computed via inverse proj
    mesh = ax.pcolormesh(
        data.lons,
        data.lats,
        norm,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )

    # Coastlines + borders (cartopy 10m)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor=COAST_COLOR, alpha=0.9)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, edgecolor=BORDER_COLOR, alpha=0.7)

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

    # Footer credit
    footer_ax = fig.add_axes([0, 0, 1.0, footer_h])
    footer_ax.set_facecolor(DARK_BG)
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"@WeathermanAAA_  ·  NOAA {goes_sat_label(data.bucket)} ABI",
        ha="center", va="center",
        color=TEXT_COLOR, fontsize=8, alpha=0.85,
        transform=footer_ax.transAxes,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
