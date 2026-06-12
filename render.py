"""Matplotlib + cartopy render pipeline.

Takes a FetchResult (cropped CMI on the geos grid with companion lat/lon arrays),
projects to PlateCarree, applies the requested enhancement, and produces a
clean dark-themed PNG with title strip, a labeled right-side colorbar, and a
footer credit.
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
from matplotlib.colors import Normalize

from colormaps import get_enhancement, enhancement_norm, normalize_visible
from satellites import FetchResult

log = logging.getLogger("tat-satellite.render")

DARK_BG = "#0a0d12"
GRID_COLOR = "#3a4252"
COAST_COLOR = "#000000"   # deep bold black — landmass outlines (Andrew's pref)
BORDER_COLOR = "#000000"  # black — political borders, to match the coastlines
TEXT_COLOR = "#e8eef5"
ACCENT_COLOR = "#79f0d6"
MUTED_COLOR = "#9199a4"


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
            nan_frac = float((~np.isfinite(bt_c)).mean())
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
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    aspect = lon_span / max(lat_span, 1e-6)

    # Figure size: target ~1400 px wide, height by aspect, dpi=110
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

    ax = fig.add_axes(
        [0.04, bottom_pad, map_w, map_h], projection=ccrs.PlateCarree()
    )
    ax.set_facecolor(DARK_BG)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    mesh = None
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
    coast_scale = _coast_resolution(max(lon_span, lat_span))
    ax.add_feature(
        cfeature.COASTLINE.with_scale(coast_scale),
        linewidth=1.2, edgecolor=COAST_COLOR, alpha=1.0, zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(coast_scale),
        linewidth=0.8, edgecolor=BORDER_COLOR, alpha=1.0, zorder=3,
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
    fig.savefig(buf, format="png", dpi=110, facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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
        im = im.resize((width, height), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()
