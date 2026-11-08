"""Globe loop frames for the storm floaters and mesoscale sectors.

Ported from the globe-loop series (the Melissa GOES-19 and Champi Himawari-9
loops): the same general vertical perspective camera, globe basemap (navy
ocean, charcoal land, darkening toward the limb, limb glow, night side shaded),
the house enhancement composited over the globe with warm clear sky letting the
globe show through, dotted 2-degree graticule labelled where it meets the
panel, and the same chrome (stage and ATCF number, name, date and HH:MM:SS UTC,
vertical degrees-C bar, intensity, product line, watermark).

Data: the storm box (``data``: the 1-minute mesoscale sector, Himawari target
area or the floater crop) is feathered over a wider ``context`` field of the
same band (full disk / CONUS), which fills the rest of the globe.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import io
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from colormaps import get_enhancement, enhancement_norm, normalize_visible
from satellites import FetchResult

log = logging.getLogger("globe_render")

# layout (pixels, the 1400-px square of the loop series)
W, H = 1400, 1400
PANEL = (90, 118, 1150, 1150)          # globe panel x, y, w, h (top-left origin)
SPAN_KM = 2000.0                        # ground distance across the panel
LIMB_FRAC = 0.86                        # horizon at this fraction of the corner radius
FEATHER_PX = 30                         # storm box fades into the context field

BLACK, KEY, WHITE, GREY = "#000000", "#5a6578", "#f2f4f8", "#8d97a8"
R_EARTH_KM = 6371.0088
COL_SPACE = np.array([0.020, 0.022, 0.030])
COL_OCEAN = np.array([0.055, 0.085, 0.150])
COL_LAND = np.array([0.150, 0.155, 0.165])
COL_GLOW = np.array([0.35, 0.50, 0.75])

_FONT_DIR = Path(__file__).parent / "vendor" / "fonts"


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def solve_camera(span_km: float, limb_frac: float, aspect: float) -> tuple[float, float]:
    """Camera distance P (Earth radii from the centre) and image-plane half width
    (km) so that ground distance span/2 from the centre lands on the left/right
    frame edge and the horizon sits at limb_frac of the corner radius."""
    c = (span_km / 2.0) / R_EARTH_KM
    corner = math.sqrt(1.0 + aspect * aspect)

    def f(P):
        rh = R_EARTH_KM * math.sqrt((P - 1) / (P + 1))
        Wd = rh / (limb_frac * corner)
        proj = R_EARTH_KM * math.sin(c) * (P - 1) / (P - math.cos(c))
        return proj - Wd

    Ps = np.geomspace(1.0005, 200.0, 2000)
    vals = np.array([f(p) for p in Ps])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) == 0:
        raise RuntimeError("could not solve camera distance")
    lo, hi = Ps[idx[-1]], Ps[idx[-1] + 1]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sign(f(mid)) == np.sign(f(lo)):
            lo = mid
        else:
            hi = mid
    P = 0.5 * (lo + hi)
    rh = R_EARTH_KM * math.sqrt((P - 1) / (P + 1))
    return P, rh / (limb_frac * corner)


class Perspective:
    def __init__(self, P: float, half_w_km: float, width: int, height: int):
        self.P = P
        self.half_w = half_w_km
        self.half_h = half_w_km * height / width
        self.w, self.h = width, height
        self.scale = 2 * half_w_km / width  # km per pixel
        self.rho_h = R_EARTH_KM * math.sqrt((P - 1) / (P + 1))
        self.ext = (-self.half_w, self.half_w, -self.half_h, self.half_h)
        xs = (np.arange(width) + 0.5 - width / 2) * self.scale
        ys = (height / 2 - np.arange(height) - 0.5) * self.scale
        self.X, self.Y = np.meshgrid(xs, ys)
        self.rho_km = np.hypot(self.X, self.Y)
        rho = self.rho_km / R_EARTH_KM
        self.visible = rho <= math.sqrt((P - 1) / (P + 1)) * (1 - 1e-9)
        rho_s = np.where(self.visible, rho, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            arg = 1.0 - rho_s * rho_s * (P + 1) / (P - 1)
            num = P - np.sqrt(np.clip(arg, 0, None))
            den = (P - 1) / np.where(rho_s > 0, rho_s, 1e-12) + rho_s / (P - 1)
            c = np.arcsin(np.clip(num / den, -1, 1))
        c[rho_s == 0] = 0.0
        self.sin_c, self.cos_c = np.sin(c), np.cos(c)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.xr = np.where(rho_s > 0, self.X / self.rho_km, 0.0)
            self.yr = np.where(rho_s > 0, self.Y / self.rho_km, 0.0)

    def inverse(self, lat0: float, lon0: float):
        """Pixel grid -> (lat, lon) in degrees; lon continuous around lon0."""
        p1, l0 = math.radians(lat0), math.radians(lon0)
        sp, cp = math.sin(p1), math.cos(p1)
        lat = np.arcsin(np.clip(self.cos_c * sp + self.yr * self.sin_c * cp, -1, 1))
        lon = l0 + np.arctan2(self.xr * self.sin_c, cp * self.cos_c - self.yr * sp * self.sin_c)
        return np.degrees(lat), np.degrees(lon)

    def forward(self, lat, lon, lat0: float, lon0: float):
        """(lat, lon) degrees -> (x, y) km in the image plane + visible mask.
        Hidden points are pushed to the horizon along their azimuth."""
        P = self.P
        p1, l0 = math.radians(lat0), math.radians(lon0)
        phi = np.radians(np.asarray(lat, float))
        dl = np.radians(np.asarray(lon, float)) - l0
        cos_c = math.sin(p1) * np.sin(phi) + math.cos(p1) * np.cos(phi) * np.cos(dl)
        vis = cos_c >= 1.0 / P
        kp = (P - 1) / (P - cos_c)
        x = R_EARTH_KM * kp * np.cos(phi) * np.sin(dl)
        y = R_EARTH_KM * kp * (math.cos(p1) * np.sin(phi) - math.sin(p1) * np.cos(phi) * np.cos(dl))
        ax = np.cos(phi) * np.sin(dl)
        ay = math.cos(p1) * np.sin(phi) - math.sin(p1) * np.cos(phi) * np.cos(dl)
        an = np.hypot(ax, ay)
        an = np.where(an > 0, an, 1.0)
        x = np.where(vis, x, self.rho_h * 1.02 * ax / an)
        y = np.where(vis, y, self.rho_h * 1.02 * ay / an)
        return x, y, vis


@lru_cache(maxsize=2)
def _camera(span_km: float = SPAN_KM) -> Perspective:
    P, half_w = solve_camera(span_km, LIMB_FRAC, PANEL[3] / PANEL[2])
    return Perspective(P, half_w, PANEL[2], PANEL[3])


def view_radius_deg(span_km: float = SPAN_KM) -> float:
    """Great-circle radius (deg) of the visible disc: the area a context field must cover."""
    P, _ = solve_camera(span_km, LIMB_FRAC, PANEL[3] / PANEL[2])
    return math.degrees(math.acos(1.0 / P))


# --------------------------------------------------------------------------- #
# basemap
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _ne_polys():
    """Natural Earth 50 m land rings as (lon/lat array, bounds)."""
    import cartopy.io.shapereader as shp
    out = []
    for g in shp.Reader(shp.natural_earth("50m", "physical", "land")).geometries():
        for poly in getattr(g, "geoms", [g]):
            a = np.asarray(poly.exterior.coords, float)
            out.append((a, (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())))
    return out


@lru_cache(maxsize=1)
def _ne_coast():
    import cartopy.io.shapereader as shp
    out = []
    for g in shp.Reader(shp.natural_earth("10m", "physical", "coastline")).geometries():
        for line in getattr(g, "geoms", [g]):
            a = np.asarray(line.coords, float)
            out.append((a, (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())))
    return out


def _in_view(items, lat0, lon0, rad_deg):
    """Rings/lines whose bounds touch the view, shifted by 360 where needed."""
    la_lo, la_hi = lat0 - rad_deg, lat0 + rad_deg
    lon_rad = rad_deg / max(math.cos(math.radians(min(abs(lat0) + rad_deg, 85.0))), 0.08)
    lo_lo, lo_hi = lon0 - lon_rad, lon0 + lon_rad
    out = []
    for a, (x0, y0, x1, y1) in items:
        if y1 < la_lo or y0 > la_hi:
            continue
        for shift in (-360.0, 0.0, 360.0):
            if x1 + shift < lo_lo or x0 + shift > lo_hi:
                continue
            b = a.copy()
            b[:, 0] += shift
            out.append(b)
    return out


def _project_lines(proj, lines, lat0, lon0):
    segs = []
    for arr in lines:
        x, y, vis = proj.forward(arr[:, 1], arr[:, 0], lat0, lon0)
        if not vis.any():
            continue
        idx = np.where(~vis)[0]
        for s, e in zip(np.r_[0, idx + 1], np.r_[idx, len(x)]):
            if e - s >= 2:
                segs.append(np.column_stack([x[s:e], y[s:e]]))
    return segs


def _land_mask(proj, lat0, lon0, rad_deg) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(proj.w / 100.0, proj.h / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(proj.ext[0], proj.ext[1])
    ax.set_ylim(proj.ext[2], proj.ext[3])
    ax.axis("off")
    fig.patch.set_facecolor("white")
    rings = []
    for r in _in_view(_ne_polys(), lat0, lon0, rad_deg):
        x, y, _ = proj.forward(r[:, 1], r[:, 0], lat0, lon0)
        rings.append(np.column_stack([x, y]))
    if rings:
        ax.add_collection(PolyCollection(rings, facecolors="black", edgecolors="black",
                                         linewidths=0.3, antialiased=False))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, 0] < 128


def compose_base(proj: Perspective, land: np.ndarray, darken: float = 0.42) -> np.ndarray:
    img = np.empty((proj.h, proj.w, 3), np.float32)
    img[:] = COL_SPACE
    vis = proj.visible
    img[vis] = COL_OCEAN
    img[land & vis] = COL_LAND
    r = proj.rho_km / proj.rho_h
    img *= (1.0 - darken * np.clip(r, 0, 1) ** 3)[..., None]
    rim = np.exp(-np.clip(proj.rho_h - proj.rho_km, 0, None) / (0.015 * proj.rho_h)) * vis
    img += (0.16 * rim)[..., None] * COL_GLOW
    outside = np.clip(proj.rho_km - proj.rho_h, 0, None)
    img += (0.30 * np.exp(-outside / (0.05 * proj.rho_h)) * (~vis))[..., None] * COL_GLOW
    return np.clip(img, 0, 1)


def subsolar(t: dt.datetime) -> tuple[float, float]:
    """(lat, lon) of the subsolar point, degrees. Low-precision solar position (NOAA)."""
    n = (t - dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc)).total_seconds() / 86400.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360.0
    eot_deg = ((L - ra + 180.0) % 360.0) - 180.0
    ut_h = t.hour + t.minute / 60.0 + t.second / 3600.0
    lon = ((-15.0 * (ut_h - 12.0) - eot_deg + 180.0) % 360.0) - 180.0
    return dec, lon


def terminator_shade(lat, lon, t: dt.datetime, depth: float = 0.5) -> np.ndarray:
    dec, lon_s = subsolar(t)
    cosz = (np.sin(np.radians(lat)) * math.sin(math.radians(dec)) +
            np.cos(np.radians(lat)) * math.cos(math.radians(dec)) * np.cos(np.radians(lon - lon_s)))
    night = np.clip((-cosz + 0.02) / 0.16, 0.0, 1.0)
    night = night * night * (3 - 2 * night)
    return (1.0 - depth * night).astype(np.float32)


def edge_feather(mask: np.ndarray, width_px: int) -> np.ndarray:
    """1.0 deep inside a boolean mask, ramping linearly to 0 at its edge over width_px (0 outside)."""
    if width_px <= 0 or not mask.any():
        return mask.astype(np.float32)
    d = distance_transform_edt(mask)
    return np.clip(d / float(width_px), 0.0, 1.0).astype(np.float32) * mask


# --------------------------------------------------------------------------- #
# sampling a curvilinear satellite field onto the panel
# --------------------------------------------------------------------------- #
def sample(fr: FetchResult, proj: Perspective, lat0: float, lon0: float, field: np.ndarray) -> np.ndarray:
    """Nearest source pixel for each panel pixel (NaN where the source has no data)."""
    lats, lons = fr.lats, fr.lons
    if lats.ndim == 1 and lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)
    field = np.ma.filled(field, np.nan) if np.ma.isMaskedArray(field) else np.asarray(field)
    out_shape = (proj.h, proj.w) + field.shape[2:]
    out = np.full((proj.h * proj.w,) + field.shape[2:], np.nan, np.float32)
    h, w = lats.shape
    ok = np.isfinite(lats) & np.isfinite(lons)
    if not ok.any():
        return out.reshape(out_shape)
    x, y, vis = proj.forward(lats[ok], lons[ok], lat0, lon0)
    col = (x - proj.ext[0]) / proj.scale - 0.5
    row = (proj.ext[3] - y) / proj.scale - 0.5
    vals = field[ok]
    # source denser than ~1.4x the panel only costs time
    span_px = max(float(np.ptp(col[vis])) if vis.any() else 1.0, float(np.ptp(row[vis])) if vis.any() else 1.0)
    stride = max(1, int(min(h, w) / max(span_px * 1.4, 1.0)))
    if stride > 1:
        sub = np.zeros_like(ok)
        sub[::stride, ::stride] = True
        keep = sub[ok]
        col, row, vis, vals = col[keep], row[keep], vis[keep], vals[keep]
    inside = vis & (col > -2) & (col < proj.w + 1) & (row > -2) & (row < proj.h + 1)
    col, row, vals = col[inside], row[inside], vals[inside]
    if col.size < 4:
        return out.reshape(out_shape)
    tree = cKDTree(np.column_stack([col, row]))
    step = max(1, col.size // 3000)
    d2, _ = tree.query(np.column_stack([col[::step], row[::step]]), k=2)
    tol = max(1.5, 1.6 * float(np.nanpercentile(d2[:, 1], 90)))
    c0, c1 = max(0, int(col.min()) - 1), min(proj.w, int(col.max()) + 2)
    r0, r1 = max(0, int(row.min()) - 1), min(proj.h, int(row.max()) + 2)
    gc, gr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    dist, idx = tree.query(np.column_stack([gc.ravel(), gr.ravel()]), distance_upper_bound=tol, workers=2)
    hit = np.isfinite(dist)
    flat = gr.ravel() * proj.w + gc.ravel()
    out[flat[hit]] = vals[idx[hit]]
    return out.reshape(out_shape)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
_BAND_NAMES = {"clean_ir": "IR", "ir_window": "IR", "wv_upper": "UPPER WV",
               "wv_lower": "LOWER WV", "shortwave_ir": "SHORTWAVE IR"}


def stage_word(nature: Optional[str], v: Optional[float], basin: Optional[str]) -> str:
    n = (nature or "").upper()
    b = (basin or "").lower()
    if n in ("EX", "PT"):
        return "POST-TROPICAL CYCLONE"
    if n in ("LO", "DB", "WV"):
        return "DISTURBANCE"
    if n == "SS":
        return "SUBTROPICAL STORM"
    if n == "SD":
        return "SUBTROPICAL DEPRESSION"
    v = v or 0
    if b in ("al", "ep", "cp"):
        if v >= 64:
            cat = 5 if v >= 137 else 4 if v >= 113 else 3 if v >= 96 else 2 if v >= 83 else 1
            return f"CATEGORY {cat} HURRICANE"
        return "TROPICAL STORM" if v >= 34 else "TROPICAL DEPRESSION"
    if b == "wp":
        return ("SUPER TYPHOON" if v >= 130 else "TYPHOON" if v >= 64
                else "TROPICAL STORM" if v >= 34 else "TROPICAL DEPRESSION")
    return "TROPICAL CYCLONE" if v >= 64 else "TROPICAL STORM" if v >= 34 else "TROPICAL DEPRESSION"


def sector_word(fr: FetchResult) -> str:
    p = (fr.product or "").upper()
    if p.startswith("CMIPM"):
        return "MESOSCALE"
    if "TARGET" in p or p.startswith("R30"):
        return "TARGET AREA"
    if p.startswith("CMIPC"):
        return "CONUS"
    if p.startswith("CMIPP"):
        return "PACUS"
    return "FULL DISK"


def band_word(fr: FetchResult, channel: int, enhancement: str) -> str:
    if fr.units == "rgb":
        return "TRUE COLOR"
    name = _BAND_NAMES.get(getattr(fr, "generic_channel", "") or "", "VISIBLE" if fr.units == "1" else "")
    s = f"BAND {channel} {name}".strip()
    return f"{s}  ·  DVORAK BD" if enhancement == "dvorak" else s


def graticule(ax, proj, lat0, lon0, step=2.0):
    """Dotted lat/lon lines on the globe; returns label positions where each line meets the panel edge."""
    ext = proj.ext
    labels = []
    for k in (0, 1):
        vals = np.arange(-80, 81, step) if k == 0 else np.arange(-360, 360, step)
        for v in vals:
            if k == 0:
                lo = np.linspace(lon0 - 40, lon0 + 40, 800)
                la = np.full_like(lo, v)
            else:
                if abs(((v - lon0 + 180) % 360) - 180) > 40:
                    continue
                la = np.linspace(-80, 80, 800)
                lo = np.full_like(la, v)
            x, y, vis = proj.forward(la, lo, lat0, lon0)
            inside = vis & (x >= ext[0]) & (x <= ext[1]) & (y >= ext[2]) & (y <= ext[3])
            if inside.sum() < 2:
                continue
            ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="white", lw=0.7,
                    ls=(0, (1.2, 2.4)), alpha=0.35, zorder=4)
            out_edge = vis & ((x < ext[0]) if k == 0 else (y < ext[2]))
            cross = np.where((out_edge[:-1] & inside[1:]) | (inside[:-1] & out_edge[1:]))[0]
            if len(cross):
                i = cross[0]
                if k == 0:
                    f = (ext[0] - x[i]) / (x[i + 1] - x[i])
                    labels.append(("lat", float(y[i] + f * (y[i + 1] - y[i])), v))
                else:
                    f = (ext[2] - y[i]) / (y[i + 1] - y[i])
                    labels.append(("lon", float(x[i] + f * (x[i + 1] - x[i])), v))
    return labels


@lru_cache(maxsize=1)
def _fonts() -> dict:
    out = {}
    for role, name in (("black", "Black"), ("bold", "Bold"), ("medium", "Medium"), ("regular", "Regular")):
        p = _FONT_DIR / f"Metropolis-{name}.ttf"
        out[role] = font_manager.FontProperties(fname=str(p)) if p.exists() else font_manager.FontProperties()
    return out


# --------------------------------------------------------------------------- #
# frame
# --------------------------------------------------------------------------- #
def bbox_center(bbox: list[float]) -> tuple[float, float]:
    w, s, e, n = bbox
    span = (e - w) % 360.0 or 360.0
    return ((w + span / 2.0 + 180.0) % 360.0) - 180.0, (s + n) / 2.0


def _strided(fr: FetchResult, d: int) -> FetchResult:
    kw = {"cmi": fr.cmi[::d, ::d], "lats": fr.lats[::d, ::d], "lons": fr.lons[::d, ::d]}
    if fr.geom_valid is not None:
        kw["geom_valid"] = fr.geom_valid[::d, ::d]
    return dataclasses.replace(fr, **kw)


def _bt_c(fr: FetchResult) -> np.ndarray:
    bt = np.asarray(np.ma.filled(fr.cmi, np.nan) if np.ma.isMaskedArray(fr.cmi) else fr.cmi, np.float32)
    if fr.units in ("C", "celsius", "degC"):
        bt = bt + 273.15
    return bt - 273.15


def render_globe_png(
    data: FetchResult,
    bbox: list[float],
    channel: int,
    scan_time: dt.datetime,
    enhancement: str,
    downsample: int = 1,
    storm: Optional[dict] = None,
    caption: Optional[str] = None,
    context: Optional[FetchResult] = None,
) -> bytes:
    """One 1400-px square globe frame (PNG)."""
    if downsample > 1:
        data = _strided(data, downsample)
    t = scan_time if scan_time.tzinfo else scan_time.replace(tzinfo=dt.timezone.utc)
    t = t.astimezone(dt.timezone.utc)
    is_rgb, is_vis = data.units == "rgb", data.units == "1"
    proj = _camera()
    lon0, lat0 = bbox_center(bbox)
    lat, lon = proj.inverse(lat0, lon0)
    rad_deg = view_radius_deg() + 2.0
    land = _land_mask(proj, lat0, lon0, rad_deg)
    base = compose_base(proj, land) * terminator_shade(lat, lon, t)[..., None]

    enh = None if is_rgb else get_enhancement(enhancement)
    tmin = None
    if is_rgb:
        rgb_src = np.clip(np.asarray(data.cmi, np.float32), 0.0, 1.0)
        fin_src = np.isfinite(rgb_src).all(axis=-1)
        if not fin_src.any() or float(rgb_src[fin_src].max(axis=-1).mean()) < 0.04:
            raise RuntimeError("truecolor render produced degenerate RGB")
        v = sample(data, proj, lat0, lon0, rgb_src)
        fin = np.isfinite(v).all(axis=-1)
        if context is not None:
            # outside the true-color box: the clean-IR context in grey (cold
            # cloud white), over the globe like the IR frames
            c = sample(context, proj, lat0, lon0, _bt_c(context))
            okc = np.isfinite(c)
            grey = np.clip((25.0 - np.nan_to_num(c, nan=25.0)) / 95.0, 0.0, 1.0) ** 0.8
            ac = (np.where(okc, np.clip((22.0 - np.nan_to_num(c, nan=40.0)) / 17.0, 0.15, 1.0), 0.0) * proj.visible)[..., None]
            base = base * (1 - ac) + grey[..., None] * ac
        a = (edge_feather(fin, FEATHER_PX) * proj.visible)[..., None]
        rgb = base * (1 - a) + np.nan_to_num(v) * a
    else:
        if is_vis:
            fld = np.asarray(normalize_visible(data.cmi), np.float32)
        else:
            fld = _bt_c(data)
            geo = np.isfinite(data.lats) & np.isfinite(data.lons)
            if geo.any() and float((~np.isfinite(fld) & geo).sum() / geo.sum()) > 0.55:
                raise RuntimeError("scalar render produced a mostly-NaN field")
        v = sample(data, proj, lat0, lon0, fld)
        fin = np.isfinite(v)
        if not fin.any():
            raise RuntimeError("no data inside the globe view")
        if not is_vis:
            tmin = float(np.nanmin(v[fin]))
        fa = edge_feather(fin, FEATHER_PX)
        if context is not None and not is_vis:
            c = sample(context, proj, lat0, lon0, _bt_c(context))
            both = fin & np.isfinite(c)
            v = np.where(both, fa * v + (1 - fa) * c, np.where(fin, v, c))
            fade = np.ones_like(v)
        else:
            fade = fa
        ok = np.isfinite(v)
        if is_vis:
            rgba = np.repeat(np.nan_to_num(v, nan=0.0)[..., None], 3, axis=2)
            alpha = np.ones_like(v)
        else:
            rgba = np.asarray(enh["cmap"](enhancement_norm(enhancement)(np.nan_to_num(v, nan=40.0))), np.float32)[..., :3]
            if enhancement in ("rainbow_ir", "dvorak", "ir_gray"):
                alpha = np.clip((22.0 - v) / 17.0, 0.15, 1.0)   # warm clear sky lets the globe show through
            else:
                alpha = np.ones_like(v)                         # WV / SWIR: the whole field is the product
        a = (np.where(ok, alpha * fade, 0.0) * proj.visible)[..., None]
        rgb = base * (1 - a) + rgba * a

    fp = _fonts()
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BLACK)
    px, py, pw, ph = PANEL
    ax = fig.add_axes([px / W, 1 - (py + ph) / H, pw / W, ph / H])
    ax.set_facecolor(BLACK)
    ax.imshow(np.clip(rgb, 0, 1), extent=proj.ext, origin="upper", interpolation="bicubic")
    labels = graticule(ax, proj, lat0, lon0)
    coast = _project_lines(proj, _in_view(_ne_coast(), lat0, lon0, rad_deg), lat0, lon0)
    ax.add_collection(LineCollection(coast, colors=["#e8ecf2"], linewidths=0.8, alpha=0.55, zorder=5))
    ax.set_xlim(proj.ext[0], proj.ext[1])
    ax.set_ylim(proj.ext[2], proj.ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(KEY)
        sp.set_linewidth(1.2)
    dx, dy = (proj.ext[1] - proj.ext[0]), (proj.ext[3] - proj.ext[2])
    kept = []                                                      # clear of the corners, at least 6% apart
    for kind, pos, v_ in sorted(labels, key=lambda r: (r[0], r[1])):
        lo_, hi_ = (proj.ext[2], proj.ext[3]) if kind == "lat" else (proj.ext[0], proj.ext[1])
        span = hi_ - lo_
        if pos < lo_ + 0.08 * span or pos > hi_ - 0.08 * span:
            continue
        if kept and kept[-1][0] == kind and pos - kept[-1][1] < 0.06 * span:
            continue
        kept.append((kind, pos, v_))
    for kind, pos, v_ in kept:
        if kind == "lat":
            ax.text(proj.ext[0] - dx * 0.010, pos, f"{abs(v_):.0f}°{'N' if v_ >= 0 else 'S'}", ha="right", va="center",
                    fontproperties=fp["medium"], fontsize=12, color=GREY, clip_on=False)
        else:
            lo = ((v_ + 180) % 360) - 180
            ax.text(pos, proj.ext[2] - dy * 0.010, f"{abs(lo):.0f}°{'E' if lo >= 0 else 'W'}", ha="center", va="top",
                    fontproperties=fp["medium"], fontsize=12, color=GREY, clip_on=False)

    # top border: storm left, date and time right
    X0, X1 = px / W, (px + pw) / W
    if storm:
        kicker = stage_word(storm.get("nature"), storm.get("wind_kt"), storm.get("basin"))
        if storm.get("atcf"):
            kicker += f"  ·  {storm['atcf']}"
        name = (storm.get("name") or "").strip()
        title = name if name.upper().startswith("INVEST") else name.title()
    else:
        kicker = f"{data.sat_name}  ·  {sector_word(data)}"
        title = caption or data.sat_name
    fig.text(X0, 1 - 30 / H, kicker.upper(), fontproperties=fp["medium"], fontsize=14, color=GREY, ha="left", va="center")
    fig.text(X0, 1 - 72 / H, title, fontproperties=fp["black"], fontsize=38, color=WHITE, ha="left", va="center")
    fig.text(X1, 1 - 30 / H, f"{t:%d %b %Y}".upper(), fontproperties=fp["medium"], fontsize=14, color=GREY, ha="right", va="center")
    fig.text(X1, 1 - 72 / H, f"{t:%H:%M:%S} UTC", fontproperties=fp["bold"], fontsize=30, color=WHITE, ha="right", va="center")

    # bottom border: intensity, product, credit
    yb = 1 - (py + ph + 58) / H
    stats = []
    if storm and storm.get("wind_kt") is not None:
        stats.append(f"{int(round(storm['wind_kt']))} KT")
    if storm and storm.get("pressure_mb") is not None:
        stats.append(f"{int(round(storm['pressure_mb']))} MB")
    prod = [data.sat_name.upper(), band_word(data, channel, enhancement), sector_word(data)]
    if tmin is not None:
        prod.append(f"MIN {tmin:.1f}°C")
    if storm and storm.get("fix_time"):
        prod.append(str(storm["fix_time"]).upper())
    prod_s = "  ·  ".join(prod)
    if stats:
        fig.text(X0, yb, "   ".join(stats), fontproperties=fp["bold"], fontsize=22, color=WHITE, ha="left", va="center")
        fig.text(X0, yb - 30 / H, prod_s, fontproperties=fp["medium"], fontsize=11.5, color=GREY, ha="left", va="center")
    else:
        fig.text(X0, yb, prod_s, fontproperties=fp["medium"], fontsize=11.5, color=GREY, ha="left", va="center")
    fig.text(X1, yb, "@WeathermanAAA_", fontproperties=fp["bold"], fontsize=18, color=WHITE, ha="right", va="center")

    # right border: colour bar
    if not is_rgb:
        cb = fig.add_axes([(px + pw + 22) / W, 1 - (py + ph) / H + 0.12, 18 / W, ph / H - 0.24])
        if is_vis:
            vv = np.linspace(0, 1, 600)
            cb.imshow(np.repeat(vv[::-1, None, None], 3, axis=2), aspect="auto", extent=(0, 1, 0, 100))
            ticks = list(np.arange(0, 101, 20))
            lo_, hi_ = 0, 100
            unit = "%"
        else:
            ticks = sorted(float(x) for x in enh["ticks"])
            lo_, hi_ = ticks[0], ticks[-1]
            vals = np.linspace(lo_, hi_, 1200)
            cb.imshow(np.asarray(enh["cmap"](enhancement_norm(enhancement)(vals)))[::-1, None, :3],
                      aspect="auto", extent=(0, 1, lo_, hi_))
            unit = "°C"
        cb.set_yticks(ticks)
        cb.set_yticklabels([f"{x:g}" for x in ticks])
        cb.set_ylim(lo_, hi_)
        cb.set_xticks([])
        cb.yaxis.tick_right()
        cb.tick_params(colors=GREY, labelsize=11, length=3, width=0.8)
        for lab in cb.get_yticklabels():
            lab.set_fontproperties(fp["medium"])
            lab.set_fontsize(11)
        for sp in cb.spines.values():
            sp.set_color(KEY)
            sp.set_linewidth(0.8)
        cb.text(0.5, 1.02, unit, transform=cb.transAxes, fontproperties=fp["medium"], fontsize=12, color=GREY,
                ha="center", va="bottom")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=BLACK)
    plt.close(fig)
    return buf.getvalue()
