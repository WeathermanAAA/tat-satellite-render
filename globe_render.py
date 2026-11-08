"""Globe-style loop frames for the storm floaters and mesoscale sectors.

Each frame is a square orthographic view centred on the storm (floater) or on
the sector (meso), so the curvature of the Earth shows at the corners. Chrome:
storm class line + big name top-left, date + HH:MM:SS UTC top-right, lat/lon
labels outside the map, a thin vertical colorbar on the right, intensity and a
source line along the bottom, @WeathermanAAA_ bottom-right.

The source field is resampled straight onto the output pixel grid (nearest
source pixel in projected space) and drawn with imshow, so a frame costs about
the same as the classic PlateCarree frame even at 500 m true color.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import font_manager
from scipy.spatial import cKDTree

from colormaps import get_enhancement, enhancement_norm, normalize_visible
from satellites import FetchResult

log = logging.getLogger("globe_render")

BG = "#000000"
INK = "#f2f4f7"
DIM = "#8d96a3"
FRAME = "#3b4250"
GRAT = "#c9d1dc"
COAST = "#e8edf3"

FIG_IN = 10.8      # 10.8 in x 100 dpi = 1080 px square
DPI = 100
MAP = [0.062, 0.088, 0.836, 0.836]   # left, bottom, width, height (square)
GRID_PX = 900                         # resample grid (map is ~903 px wide)

_FONT_DIR = Path(__file__).parent / "vendor" / "fonts"


@lru_cache(maxsize=1)
def _fonts() -> dict:
    out = {}
    for w in (400, 500, 700, 800):
        p = _FONT_DIR / f"Metropolis-{w}.ttf"
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            out[w] = font_manager.FontProperties(fname=str(p))
    return out


def _fp(weight: int, size: float):
    f = _fonts().get(weight)
    if f is None:
        return {"fontsize": size, "fontweight": "bold" if weight >= 700 else "normal"}
    fp = f.copy()
    fp.set_size(size)
    return {"fontproperties": fp}


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
_BAND_NAMES = {
    "clean_ir": "IR", "ir_window": "IR", "wv_upper": "UPPER WV",
    "wv_lower": "LOWER WV", "shortwave_ir": "SHORTWAVE IR",
}
_ENH_NAMES = {"dvorak": "DVORAK BD", "rainbow_ir": None, "wv_tat": None,
              "grayscale": None, "ir_gray": None}


def storm_class(nature: Optional[str], wind_kt: Optional[float],
                basin: Optional[str]) -> str:
    """Header class line, e.g. CATEGORY 5 HURRICANE / SUPER TYPHOON."""
    n = (nature or "").upper()
    b = (basin or "").upper()
    if n in ("EX", "PT"):
        return "POST-TROPICAL CYCLONE"
    if n in ("LO", "DB", "WV"):
        return "DISTURBANCE"
    if n == "SS":
        return "SUBTROPICAL STORM"
    if n == "SD":
        return "SUBTROPICAL DEPRESSION"
    w = wind_kt or 0.0
    if w < 34:
        return "TROPICAL DEPRESSION"
    if w < 64:
        return "TROPICAL STORM"
    if b == "WP":
        return "SUPER TYPHOON" if w >= 130 else "TYPHOON"
    if b in ("IO", "SH", "NI", "SI", "AU", "SP"):
        return "TROPICAL CYCLONE"
    cat = 1 if w < 83 else 2 if w < 96 else 3 if w < 113 else 4 if w < 137 else 5
    return f"CATEGORY {cat} HURRICANE"


def _band_label(data: FetchResult, channel: int, enhancement: str) -> str:
    if data.units == "rgb":
        return "TRUE COLOR"
    gen = getattr(data, "generic_channel", "") or ""
    name = _BAND_NAMES.get(gen, "VISIBLE" if data.units == "1" else "")
    s = f"BAND {channel} {name}".strip()
    extra = _ENH_NAMES.get(enhancement)
    return f"{s} · {extra}" if extra else s


def _product_word(product: str) -> str:
    p = (product or "").upper()
    if p.startswith("CMIPM") or "TARGET" in p:
        return "MESOSCALE"
    if p in ("FLDK",) or p.startswith("CMIPF"):
        return "FULL DISK"
    if p.startswith("CMIPC"):
        return "CONUS"
    if p.startswith("CMIPP"):
        return "PACUS"
    return p


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _center(bbox: list[float]) -> tuple[float, float]:
    w, s, e, n = bbox
    span = (e - w) % 360.0 or 360.0
    lon = ((w + span / 2.0 + 180.0) % 360.0) - 180.0
    return lon, (s + n) / 2.0


def _resample(field: np.ndarray, lats: np.ndarray, lons: np.ndarray,
              proj: ccrs.Projection, n_out: int, ext: list[float]):
    """Nearest-source-pixel resample of a curvilinear field onto a square
    n_out x n_out grid covering ``ext`` in ``proj`` coords."""
    field = np.ma.filled(field, np.nan) if np.ma.isMaskedArray(field) else field
    ok = np.isfinite(lats) & np.isfinite(lons)
    # keep the source density near 2x the output so the tree stays small
    h, w = lats.shape
    stride = max(1, int(min(h, w) / (2.0 * n_out)))
    if stride > 1:
        field = field[::stride, ::stride]
        lats = lats[::stride, ::stride]
        lons = lons[::stride, ::stride]
        ok = ok[::stride, ::stride]
    pc = ccrs.PlateCarree()
    xyz = proj.transform_points(pc, lons[ok].astype(np.float64),
                                lats[ok].astype(np.float64))
    xs, ys = xyz[:, 0], xyz[:, 1]
    vis = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[vis], ys[vis]
    vals = field[ok][vis]
    if xs.size == 0:
        raise RuntimeError("no source pixels visible from this globe view")
    # local source spacing (projected metres) -> gap tolerance. Measured
    # between row neighbours near the middle of the source grid, so the
    # compression toward the horizon only widens the tolerance there.
    half = (ext[1] - ext[0]) / 2.0
    cell = 2.0 * half / n_out
    tree = cKDTree(np.column_stack([xs, ys]))
    probe = np.column_stack([xs[:: max(1, xs.size // 4000)], ys[:: max(1, xs.size // 4000)]])
    d2, _ = tree.query(probe, k=2, workers=-1)
    spacing = float(np.nanpercentile(d2[:, 1], 90))
    tol = 1.5 * max(spacing, cell)
    gx = ext[0] + (np.arange(n_out) + 0.5) * cell
    gy = ext[3] - (np.arange(n_out) + 0.5) * cell
    GX, GY = np.meshgrid(gx, gy)
    dist, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]),
                           distance_upper_bound=tol, workers=-1)
    hit = np.isfinite(dist)
    if vals.ndim == 1:
        img = np.full(GX.size, np.nan, dtype=np.float32)
        img[hit] = vals[idx[hit]]
        img = img.reshape(n_out, n_out)
    else:
        img = np.full((GX.size, vals.shape[-1]), np.nan, dtype=np.float32)
        img[hit] = vals[idx[hit]]
        img = img.reshape(n_out, n_out, vals.shape[-1])
    return img


R_EARTH = 6371000.0
FILL = 0.92       # square half-side as a fraction of the box's shorter half-span
                  # (meso sectors are not exact lat/lon boxes; this keeps the
                  # disc inside the scanned footprint)
HORIZON = 1.35    # camera: projected horizon / square half-side (sets curvature;
                  # the centre is drawn ~1.75x the scale of the frame edge)
DISC = 1.0        # visible disc radius / square half-side: a full round disc


def _camera_height(a: float) -> float:
    """Satellite height (m) whose horizon lands at HORIZON x the projected
    radius of a point ``a`` radians from the centre."""
    def ratio(P: float) -> float:
        th = np.arccos(1.0 / P)
        r = lambda p: np.sin(p) * (P - 1.0) / (P - np.cos(p))  # noqa: E731
        return r(th) / r(a)
    lo, hi = 1.0 / np.cos(a) + 1e-9, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < HORIZON:
            lo = mid
        else:
            hi = mid
    return R_EARTH * (0.5 * (lo + hi) - 1.0)


def _view(bbox: list[float], lon0: float, lat0: float):
    """Near-side perspective camera centred on the box. Returns (projection,
    square extent, half-side in degrees)."""
    w, s, e, n = bbox
    lon_half = (((e - w) % 360.0) or 360.0) / 2.0 * np.cos(np.radians(lat0))
    lat_half = (n - s) / 2.0
    a = FILL * min(lon_half, lat_half)               # great-circle degrees
    h = _camera_height(np.radians(a))
    proj = ccrs.NearsidePerspective(central_longitude=lon0,
                                    central_latitude=lat0, satellite_height=h)
    pt = proj.transform_point(lon0, min(lat0 + a, 89.9), ccrs.PlateCarree())
    half = abs(pt[1])
    return proj, [-half, half, -half, half], a


def _nice_step(span_deg: float) -> float:
    for s in (1, 2, 5, 10, 15, 20):
        if span_deg / s <= 7:
            return float(s)
    return 30.0


def _fmt_lat(v: float) -> str:
    v = round(v, 4)
    return "0°" if v == 0 else f"{abs(v):g}°{'N' if v > 0 else 'S'}"


def _fmt_lon(v: float) -> str:
    v = ((v + 180.0) % 360.0) - 180.0
    v = round(v, 4)
    if v in (0.0, -180.0) or abs(v) == 180.0:
        return f"{abs(v):g}°"
    return f"{abs(v):g}°{'E' if v > 0 else 'W'}"


def _graticule(ax, fig, proj, ext, lon0, lat0, half_deg):
    """Dotted lat/lon lines + labels outside the map (lat left, lon bottom),
    each label placed where its line actually meets that map edge."""
    pc = ccrs.PlateCarree()
    step = _nice_step(2.0 * half_deg)
    span = 2.5 * half_deg / max(np.cos(np.radians(min(abs(lat0) + half_deg, 85.0))), 0.1)
    lat_lo = max(-88.0, np.floor((lat0 - span) / step) * step)
    lat_hi = min(88.0, np.ceil((lat0 + span) / step) * step)
    lon_lo = np.floor((lon0 - span) / step) * step
    lon_hi = np.ceil((lon0 + span) / step) * step
    xmin, xmax, ymin, ymax = ext
    kw = dict(color=GRAT, linewidth=0.7, alpha=0.55, linestyle=(0, (1.0, 2.2)),
              zorder=4)
    tlab = _fp(500, 8.6)
    left_used, bottom_used = [], []
    t = np.linspace(-180.0, 180.0, 1441)
    for la in np.arange(lat_lo, lat_hi + step / 2, step):
        lo = lon0 + t
        la_arr = np.full_like(lo, la)
        xy = proj.transform_points(pc, lo, la_arr)
        ax.plot(xy[:, 0], xy[:, 1], **kw)
        # left-edge crossing
        x, y = xy[:, 0], xy[:, 1]
        fin = np.isfinite(x) & np.isfinite(y)
        sgn = np.sign(x - xmin)
        for i in np.where(fin[:-1] & fin[1:] & (sgn[:-1] != sgn[1:]))[0]:
            fy = y[i] + (y[i + 1] - y[i]) * (xmin - x[i]) / (x[i + 1] - x[i])
            if ymin < fy < ymax:
                fyn = MAP[1] + MAP[3] * (fy - ymin) / (ymax - ymin)
                if all(abs(fyn - u) > 0.03 for u in left_used):
                    left_used.append(fyn)
                    fig.text(MAP[0] - 0.006, fyn, _fmt_lat(la), ha="right",
                             va="center", color=DIM, **tlab)
                break
    for lo in np.arange(lon_lo, lon_hi + step / 2, step):
        la = np.linspace(-89.0, 89.0, 1441)
        lo_arr = np.full_like(la, lo)
        xy = proj.transform_points(pc, lo_arr, la)
        ax.plot(xy[:, 0], xy[:, 1], **kw)
        x, y = xy[:, 0], xy[:, 1]
        fin = np.isfinite(x) & np.isfinite(y)
        sgn = np.sign(y - ymin)
        for i in np.where(fin[:-1] & fin[1:] & (sgn[:-1] != sgn[1:]))[0]:
            fx = x[i] + (x[i + 1] - x[i]) * (ymin - y[i]) / (y[i + 1] - y[i])
            if xmin < fx < xmax:
                fxn = MAP[0] + MAP[2] * (fx - xmin) / (xmax - xmin)
                if all(abs(fxn - u) > 0.06 for u in bottom_used):
                    bottom_used.append(fxn)
                    fig.text(fxn, MAP[1] - 0.008, _fmt_lon(lo), ha="center",
                             va="top", color=DIM, **tlab)
                break


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def render_globe_png(
    data: FetchResult,
    bbox: list[float],
    channel: int,
    scan_time: dt.datetime,
    enhancement: str,
    downsample: int = 1,
    storm: Optional[dict] = None,
    caption: Optional[str] = None,
) -> bytes:
    """One square globe frame. ``storm`` (floater) carries name / basin /
    nature / wind_kt / pressure_mb and optionally atcf + fix_time; ``caption``
    (meso) is the sector name used as the headline when there is no storm."""
    is_rgb = data.units == "rgb"
    is_vis = data.units == "1"
    cmi, lats, lons = data.cmi, data.lats, data.lons
    if downsample > 1:
        cmi = cmi[::downsample, ::downsample]
        lats = lats[::downsample, ::downsample]
        lons = lons[::downsample, ::downsample]
    if lats.ndim == 1 and lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    bt_min_c = None
    if is_rgb:
        field = np.clip(np.asarray(cmi, dtype=np.float32), 0.0, 1.0)
        geom = data.geom_valid
        if geom is not None and downsample > 1:
            geom = geom[::downsample, ::downsample]
        fin = np.isfinite(field).all(axis=-1)
        if geom is not None and geom.shape == fin.shape:
            n_geom = float(geom.sum())
            nan_frac = float((~fin & geom).sum() / n_geom) if n_geom else 1.0
        else:
            nan_frac = float((~fin).mean())
        vmax = float(field[fin].max(axis=-1).mean()) if fin.any() else 0.0
        if nan_frac > 0.50 or vmax < 0.04:
            raise RuntimeError(
                f"truecolor render produced degenerate RGB (nan={nan_frac * 100:.0f}%, "
                f"mean_valid_max={vmax:.3f})")
        cmap = norm = None
    elif is_vis:
        field = np.asarray(normalize_visible(cmi), dtype=np.float32)
        cmap = plt.get_cmap("gray")
        from matplotlib.colors import Normalize
        norm = Normalize(0.0, 1.0)
    else:
        bt = np.asarray(cmi, dtype=np.float32)
        if data.units in ("C", "celsius", "degC"):
            bt = bt + 273.15
        bt_c = bt - 273.15
        geo_ok = np.isfinite(lats) & np.isfinite(lons)
        n_ok = float(geo_ok.sum())
        if n_ok == 0:
            raise RuntimeError("bbox has no on-disk pixels to render")
        nan_frac = float((~np.isfinite(bt_c) & geo_ok).sum() / n_ok)
        if nan_frac > 0.55:
            raise RuntimeError(
                f"scalar render produced a mostly-NaN field (nan={nan_frac * 100:.0f}%)")
        if np.isfinite(bt_c).any():
            bt_min_c = float(np.nanmin(bt_c))
        field = bt_c
        enh = get_enhancement(enhancement)
        cmap = enh["cmap"]
        norm = enhancement_norm(enhancement)

    lon0, lat0 = _center(bbox)
    proj, ext, half_deg = _view(bbox, lon0, lat0)
    img = _resample(field, lats, lons, proj, GRID_PX, ext)

    fig = plt.figure(figsize=(FIG_IN, FIG_IN), dpi=DPI, facecolor=BG)
    ax = fig.add_axes(MAP, projection=proj)
    ax.set_facecolor(BG)
    ax.set_extent(ext, crs=proj)
    # the projection boundary is the horizon circle; hide it and frame the
    # square map instead
    try:
        ax.spines["geo"].set_visible(False)
    except Exception:  # noqa: BLE001
        pass
    fig.patches.append(matplotlib.patches.Rectangle(
        (MAP[0], MAP[1]), MAP[2], MAP[3], transform=fig.transFigure,
        fill=False, edgecolor=FRAME, linewidth=0.8, zorder=20))

    half = ext[1]
    yy, xx = np.mgrid[0:GRID_PX, 0:GRID_PX]
    rr = np.hypot((xx + 0.5) / GRID_PX * 2.0 - 1.0, (yy + 0.5) / GRID_PX * 2.0 - 1.0)
    outside = rr > DISC
    if img.ndim == 3:
        img[outside] = np.nan
    else:
        img = np.where(outside, np.nan, img)

    mappable = None
    if is_rgb:
        rgb = np.nan_to_num(img, nan=0.0)
        alpha = np.isfinite(img).all(axis=-1).astype(np.float32)
        rgba = np.dstack([rgb, alpha])
        ax.imshow(rgba, origin="upper", extent=ext, transform=proj,
                  interpolation="nearest", zorder=1)
    else:
        mappable = ax.imshow(np.ma.masked_invalid(img), origin="upper",
                             extent=ext, transform=proj, cmap=cmap, norm=norm,
                             interpolation="nearest", zorder=1)

    span = max(((bbox[2] - bbox[0]) % 360.0) or 360.0, bbox[3] - bbox[1])
    scale = "10m" if span < 25 else "50m"
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), edgecolor=COAST,
                   linewidth=0.6, alpha=0.55, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), edgecolor=COAST,
                   linewidth=0.45, alpha=0.4, zorder=3)
    _graticule(ax, fig, proj, ext, lon0, lat0, half_deg)
    # space outside the disc: a black square with a round hole, over the
    # coastlines and graticule
    from matplotlib.path import Path
    ring = Path.circle((0.0, 0.0), DISC * half)
    sq = Path([(-half, -half), (half, -half), (half, half), (-half, half),
               (-half, -half)], [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY])
    hole = Path(ring.vertices[::-1], ring.codes)
    ax.add_patch(matplotlib.patches.PathPatch(
        Path.make_compound_path(sq, hole), facecolor=BG, edgecolor="none",
        transform=proj, zorder=8))

    # colorbar (scalar products)
    if mappable is not None:
        cax = fig.add_axes([0.918, 0.20, 0.0115, 0.62])
        cb = fig.colorbar(mappable, cax=cax)
        if is_vis:
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cb.set_ticklabels(["0", "25", "50", "75", "100"])
            unit = "%"
        else:
            enh = get_enhancement(enhancement)
            ticks = list(enh["ticks"])
            if len(ticks) > 14:
                ticks = ticks[::2]
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{t:g}" for t in ticks])
            unit = "°C"
        cb.outline.set_edgecolor(FRAME)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(colors=DIM, length=2.5, width=0.6, pad=3)
        for lab in cb.ax.get_yticklabels():
            lab.set_fontproperties(_fp(500, 8.2).get("fontproperties"))
            lab.set_color(DIM)
        fig.text(0.918 + 0.0058, 0.828, unit, ha="center", va="bottom",
                 color=DIM, **_fp(500, 9))

    # header
    ts = scan_time.astimezone(dt.timezone.utc) if scan_time.tzinfo else scan_time
    date_s = ts.strftime("%d %b %Y").upper()
    time_s = ts.strftime("%H:%M:%S UTC")
    if storm:
        cls = storm_class(storm.get("nature"), storm.get("wind_kt"), storm.get("basin"))
        atcf = (storm.get("atcf") or "").upper()
        line1 = f"{cls} · {atcf}" if atcf else cls
        name = (storm.get("name") or "").strip()
        headline = name.title() if name.isupper() and not name.startswith("INVEST") else name
    else:
        line1 = f"{_product_word(data.product)} SECTOR" if caption else data.sat_name.upper()
        headline = caption or f"{data.sat_name} {_product_word(data.product).title()}"
    fig.text(MAP[0], 0.975, line1, ha="left", va="center", color=DIM, **_fp(500, 11))
    fig.text(MAP[0], 0.945, headline, ha="left", va="center", color=INK, **_fp(800, 27))
    right = MAP[0] + MAP[2]
    fig.text(right, 0.975, date_s, ha="right", va="center", color=DIM, **_fp(500, 11))
    fig.text(right, 0.945, time_s, ha="right", va="center", color=INK, **_fp(800, 22))

    # footer
    meta = [data.sat_name.upper(), _band_label(data, channel, enhancement),
            _product_word(data.product)]
    if bt_min_c is not None:
        meta.append(f"MIN {bt_min_c:.1f}°C")
    if storm and storm.get("fix_time"):
        meta.append(str(storm["fix_time"]).upper())
    stats = []
    if storm:
        if storm.get("wind_kt") is not None:
            stats.append(f"{int(round(storm['wind_kt']))} KT")
        if storm.get("pressure_mb") is not None:
            stats.append(f"{int(round(storm['pressure_mb']))} MB")
    if stats:
        fig.text(MAP[0], 0.040, "   ".join(stats), ha="left", va="center",
                 color=INK, **_fp(800, 20))
        fig.text(MAP[0], 0.014, "  ·  ".join(m for m in meta if m), ha="left",
                 va="center", color=DIM, **_fp(500, 8.8))
    else:
        fig.text(MAP[0], 0.030, "  ·  ".join(m for m in meta if m), ha="left",
                 va="center", color=DIM, **_fp(500, 9.5))
    fig.text(right, 0.040 if stats else 0.030, "@WeathermanAAA_", ha="right",
             va="center", color=INK, **_fp(700, 15))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    return buf.getvalue()
