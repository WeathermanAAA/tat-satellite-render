"""Storm-centred multi-source CENTER-FIX plot — a server-rendered PNG.

WHAT IT SHOWS
    Panel 1  Grayscale IR with BD-style contours, and EVERY available centre
             estimate overlaid and unambiguously keyed:
               * ARCHER/ADT objective fixes  (the whole recent track, newest
                 emphasised, with the ARCHER 50%/95% position-certainty rings)
               * the official best-track position
               * the official forecast track
               * the floater / target box
             The diagnostic value is seeing where the OBJECTIVE fixes disagree
             with the OFFICIAL position -- so the key has to make each source
             unmistakable, and a rejected (low-confidence) ARCHER candidate has
             to look different from an accepted fix rather than quietly
             blending in.

    Panel 2  The same scene in the enhanced colour IR palette, with dmax/dmin
             brightness-temperature readouts tagged by BAND (IR / WV / SWIR).
             The tag is not decoration: an untagged -60 C invites reading a
             water-vapour frame as a cloud top.

THE COMPOSITE PLATE (render_composite)
    The same two panels plus the intensity diagnostic and an eye-structure
    panel, as ONE 2x2 PNG under one shared header, so the whole storm reads in
    a single copy-pasteable image. It is an ADDITIONAL product: the two-panel
    plot keeps publishing to its own key, unchanged.

    Bottom-left is the CycloLab wind/pressure + observed/projected ACE chart,
    captured from the REAL browser renderer by the collector lane (see
    wpace_headless.cjs) and pasted in as a raster. It is NOT recomputed here.
    That chart's ACE projection -- the 6-hourly regrid of forecast intensity,
    the resume-from-latest-observed-fix rule -- exists in exactly one place,
    and a matplotlib port of it would be a second copy drifting invisibly
    while both render plausible curves. Same reasoning as objfix_headless.py.
    When the capture is missing the panel says so; it never draws an empty
    frame that reads as "no ACE".

    Bottom-right is the eye-structure panel: a radial brightness-temperature
    profile about the working centre with the BD ladder behind it, and the
    ADT-style eye score (warmest eye pixel against the coldest eyewall ring)
    called out. See eye_score() for what in it is cited and what is ours.

HONESTY
    Everything objective here is an AUTOMATED SATELLITE ESTIMATE, never
    official. SATCON appears only as an INTENSITY readout beside the ADT
    intensity -- it is an intensity consensus and produces no position, so it
    is never drawn as a centre marker. When SATCON's own membership rule is
    unmet (>= 2 coincident members) the header says "no consensus" instead of
    relabelling the bare ADT.

    Data providers are credited; no third-party product is named.

INPUT
    ARCHER fixes come from cyclolab/objfix/<id>.json, published by
    objfix_headless.py, which runs the explorer's own objfix.js headlessly.
    There is exactly ONE implementation of the method and it is not here.
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import io
import json
import logging
import math
import os
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import Normalize              # noqa: E402
import numpy as np                                   # noqa: E402

log = logging.getLogger("centerfix")

CDN = os.environ.get("TAT_CDN", "https://cdn.triple-a-tropics.com")
R2_PREFIX = os.environ.get("CENTERFIX_R2_PREFIX", "cyclolab/centerfix")
WPACE_R2_PREFIX = os.environ.get("WPACE_R2_PREFIX", "cyclolab/wpace")
BOX_DEG = float(os.environ.get("CENTERFIX_BOX_DEG", "6.0"))

DARK_BG = "#0a1019"
TEXT_COLOR = "#e8eef7"
MUTED = "#8ea2bd"
GRID = "#22304a"

# Key colours. Deliberately four DISTINCT hues, none of them the IR palette's
# own, so no marker can be mistaken for imagery. Validated for CVD separation
# against the dark panel before use.
C_ARCHER = "#3fa4ff"     # objective ARCHER/ADT fixes
C_ARCHER_WEAK = "#7f8fa6"  # rejected / weak candidate — grey, never blue
C_OFFICIAL = "#ffe14d"    # official best-track position
C_FORECAST = "#ff6bd6"    # official forecast track
C_BOX = "#46c56a"         # floater / target box

#: The Dvorak BD enhancement's own grey-shade boundaries (CIMSS), in °C. Used
#: as panel-1 isotherms AND as the ladder behind the radial profile, so the two
#: panels are reading the same steps rather than two similar-looking ones.
BD_STEPS = [-80.0, -75.0, -69.0, -63.0, -53.0, -41.0, -30.0]

#: BD shade names for the ladder, warmest first. WMG (Warm Medium Grey) is the
#: warmest step, > +9 °C — a WMG pixel inside an eye is a deeply subsident,
#: cloud-free eye. Labels are the citable part; see eye_score() for what is not.
BD_LADDER = [
    ("WMG", 9.0, 40.0),
    ("OW", -30.0, 9.0),
    ("DG", -41.0, -30.0),
    ("MG", -53.0, -41.0),
    ("LG", -63.0, -53.0),
    ("B", -69.0, -63.0),
    ("W", -75.0, -69.0),
    ("CMG", -80.0, -75.0),
    ("CDG", -95.0, -80.0),
]

#: Is the official position contemporaneous enough with the emphasised
#: objective fix for their separation to mean anything? One synoptic step;
#: beyond that the gap is dominated by storm motion, not by method.
SEP_TOL_MIN = 90.0


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def _get_json(url: str) -> Optional[dict]:
    import urllib.request
    # The CDN 403s urllib's default User-Agent; send a real one. (Found the
    # hard way: every advisory/floater lookup came back 403 while the same
    # URLs served fine over curl.)
    req = urllib.request.Request(url, headers={
        "User-Agent": "tat-centerfix/1.0 (+https://triple-a-tropics.com)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - a missing input is a render skip
        log.warning("fetch failed %s: %s", url, e)
        return None


def _get_binary(url: str) -> tuple[Optional[bytes], Optional[dt.datetime]]:
    """Bytes plus the object's Last-Modified, or (None, None).

    The timestamp matters: a pasted-in raster from another lane has to be able
    to say how old it is, or the plate silently presents a stale chart beside
    fresh imagery under one VALID stamp.
    """
    import email.utils
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "tat-centerfix/1.0 (+https://triple-a-tropics.com)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            lm = r.headers.get("Last-Modified")
            when = None
            if lm:
                try:
                    when = email.utils.parsedate_to_datetime(lm)
                except Exception:      # noqa: BLE001 - a bad stamp is no stamp
                    when = None
            return raw, when
    except Exception as e:  # noqa: BLE001 - a missing input is a placeholder
        log.info("fetch failed %s: %s", url, e)
        return None, None


def load_fixes(storm_id: str) -> Optional[dict]:
    return _get_json(f"{CDN}/{os.environ.get('OBJFIX_R2_PREFIX', 'cyclolab/objfix')}"
                     f"/{storm_id}.json")


def load_advisory(storm_id: str) -> Optional[dict]:
    return _get_json(f"{CDN}/cyclolab/adv/{storm_id}.json")


def load_floater_manifest(slug: str) -> Optional[dict]:
    return _get_json(f"{CDN}/floaters/{slug}/manifest.json")


def load_wpace_png(storm_id: str) -> tuple[Optional[bytes], Optional[dt.datetime]]:
    """The CycloLab wind/pressure + ACE chart, as captured by the browser lane."""
    return _get_binary(f"{CDN}/{WPACE_R2_PREFIX}/{storm_id}.png")


def load_official_fix(storm_id: str) -> dict:
    """The official position AND the time it is valid at.

    The explorer's storm list (which rides along in the published track as
    storm_feed) carries lat/lon but NO timestamp, and a position without its
    valid time cannot be compared to an objective fix — the difference would
    be storm motion. The floaters INDEX carries last_fix, so that is where the
    time comes from.
    """
    man = _get_json(f"{CDN}/floaters/manifest.json") or {}
    for s in (man.get("storms") or []):
        if s.get("id") == storm_id:
            return s
    return {}


def target_box(man: Optional[dict]) -> Optional[dict]:
    """The floater's TARGET box from its newest frame.

    The manifest carries the BACKDROP bounds [W,S,E,N], which floater_poller
    widens to the render aspect. The target box the floater is actually
    centred on is the SQUARE of side (N-S) about the backdrop's lon centre --
    the same reconstruction objfix_sources.js does to georeference a frame, so
    the box drawn here is the box ARCHER was anchored to.
    """
    if not man:
        return None
    for band in ("ir", "irbd", "wv_up"):
        frames = ((man.get("bands") or {}).get(band) or {}).get("frames") or []
        if not frames:
            continue
        b = frames[-1].get("bounds")
        if not b or len(b) != 4:
            continue
        w, s, e, n = (float(v) for v in b)
        cx, span = (w + e) / 2.0, (n - s)
        return {"w": cx - span / 2.0, "s": s, "e": cx + span / 2.0, "n": n,
                "t": frames[-1].get("t")}
    return None


# ---------------------------------------------------------------------------
# imagery
# ---------------------------------------------------------------------------
async def _fetch_band(bbox: list[float], when: dt.datetime, generic: str):
    """One calibrated band over ``bbox`` using the render service's own path."""
    from satellites import pick_satellite
    sat = pick_satellite(bbox, when)
    resolved = await sat.find_file(when, generic, bbox, True)
    return await sat.fetch(resolved, bbox, generic)


def _bt_celsius(data) -> np.ndarray:
    bt = np.asarray(data.cmi, dtype="float64")
    if getattr(data, "units", "K") in ("C", "celsius", "degC"):
        return bt
    return bt - 273.15


def _extremes(bt_c: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    if not np.isfinite(bt_c).any():
        return None, None
    return float(np.nanmin(bt_c)), float(np.nanmax(bt_c))


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _norm_lon(lon: float, frame: float) -> float:
    while lon - frame > 180:
        lon -= 360
    while lon - frame < -180:
        lon += 360
    return lon


def _km_between(lat1, lon1, lat2, lon2) -> float:
    dlon = abs(lon1 - lon2)
    if dlon > 180:
        dlon = 360 - dlon
    x = math.radians(dlon) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return 6371.0 * math.hypot(x, y)


def _km_grid(lats, lons, clat: float, clon: float) -> np.ndarray:
    """Great-circle-ish distance in km from (clat, clon) to every grid cell.

    Vectorised twin of _km_between, on the same small-angle approximation --
    exact enough over a 6-degree box and it keeps the two distances consistent,
    which matters because the panel prints one and shades by the other.
    """
    la = np.asarray(lats, dtype="float64")
    lo = np.asarray(lons, dtype="float64")
    if la.ndim == 1 and lo.ndim == 1:
        lo, la = np.meshgrid(lo, la)
    dlon = np.abs(lo - clon)
    dlon = np.where(dlon > 180.0, 360.0 - dlon, dlon)
    x = np.radians(dlon) * np.cos(np.radians((la + clat) / 2.0))
    y = np.radians(la - clat)
    return 6371.0 * np.hypot(x, y)


# ---------------------------------------------------------------------------
# eye structure
# ---------------------------------------------------------------------------
#: Rings inside this radius cannot BE the eyewall. Without a floor the coldest
#: ring of a sheared, eyeless system sits at r~0 and the "eye" becomes an empty
#: set -- a number would still come out, and it would be meaningless.
EYE_MIN_EYEWALL_KM = 12.0
EYE_MAX_R_KM = 140.0
#: Floor on the ring width. The ACTUAL width is derived from the grid's own
#: sample spacing (see _ring_width_km): rings finer than a pixel hold one or
#: zero samples each, and the profile then reports sampling noise as structure
#: -- which is exactly the resolution-dependence this panel exists to avoid.
EYE_RING_MIN_KM = 4.0


def _ring_width_km(lats, lons, clat: float, clon: float) -> float:
    """Ring width from the grid's own spacing, floored at EYE_RING_MIN_KM."""
    la = np.asarray(lats, dtype="float64")
    lo = np.asarray(lons, dtype="float64")
    if la.ndim == 1 and lo.ndim == 1:
        dlat = float(np.median(np.abs(np.diff(la)))) if la.size > 1 else 0.0
        dlon = float(np.median(np.abs(np.diff(lo)))) if lo.size > 1 else 0.0
    else:
        dlat = float(np.median(np.abs(np.diff(la, axis=0)))) if la.shape[0] > 1 else 0.0
        dlon = float(np.median(np.abs(np.diff(lo, axis=1)))) if lo.shape[1] > 1 else 0.0
    km = max(dlat, dlon * max(0.2, math.cos(math.radians(clat)))) * 111.0
    if not np.isfinite(km) or km <= 0:
        return EYE_RING_MIN_KM
    # ONE pixel, not two. Structure finer than the sample spacing is not
    # resolvable, so that is the floor; going coarser than it throws away real
    # eyewall detail on a 2 km sensor for no gain.
    return float(max(EYE_RING_MIN_KM, km))


def eye_score(ir_c, lats, lons, clat: float, clon: float,
              max_r_km: float = EYE_MAX_R_KM,
              ring_km: Optional[float] = None) -> Optional[dict]:
    """ADT-style eye score: warmest eye pixel against the coldest eyewall ring.

    WHAT IS CITED AND WHAT IS OURS -- the number is only worth printing if its
    construction is stated, so:

    * The QUANTITY is the ADT's own documented relative: eye/eyewall brightness
      temperature CONTRAST. A larger positive value is a warmer, better-cleared
      eye inside a colder eyewall. It is a temperature difference in °C, so it
      is resolution-robust in a way a pixel COUNT is not -- the same eye counts
      differently on 2 km ABI than on a 4 km reprojection, which is why no
      count-based product is reported here.
    * The BD ladder the panel draws behind the profile is the citable
      discretisation (CIMSS): 9 steps from OW (> -30 °C) down to CDG (< -81),
      with WMG the warmest at > +9.
    * The REGIONS are OURS. ADT runs its own eye/eyewall search; here the
      eyewall is the coldest mean-BT ring of a radial profile about the working
      centre, and the eye is everything inside that radius. The radius cap, the
      ring width and the EYE_MIN_EYEWALL_KM floor are our construction too.
      They are printed on the panel rather than left implicit.

    Returns None when there is no eye signature to score -- the coldest ring is
    at the centre (no clearing), or the eye is not warmer than its eyewall. A
    withheld score is the honest output there; a negative "score" is not a
    weak eye, it is the absence of one.
    """
    bt = np.asarray(ir_c, dtype="float64")
    r = _km_grid(lats, lons, clat, clon)
    if bt.shape != r.shape or not np.isfinite(bt).any():
        return None
    if ring_km is None:
        ring_km = _ring_width_km(lats, lons, clat, clon)
    edges = np.arange(0.0, max_r_km + ring_km, ring_km)
    mids, means, mins, maxs = [], [], [], []
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1]) & np.isfinite(bt)
        if not m.any():
            continue
        v = bt[m]
        mids.append((edges[i] + edges[i + 1]) / 2.0)
        means.append(float(np.mean(v)))
        mins.append(float(np.min(v)))
        maxs.append(float(np.max(v)))
    if len(mids) < 4:
        return None
    mids = np.asarray(mids)
    means = np.asarray(means)
    prof = {"r_km": mids, "mean_c": means,
            "min_c": np.asarray(mins), "max_c": np.asarray(maxs),
            "ring_km": ring_km, "max_r_km": max_r_km,
            "score": None, "eyewall_r_km": None,
            "eye_warm_c": None, "eyewall_cold_c": None,
            "reason": None}

    if len(mids) < 8:
        # The profile is real and worth drawing; it is just too coarse for the
        # eyewall search to mean anything. Draw it, withhold the number.
        prof["reason"] = ("profile too coarse to locate an eyewall — "
                          f"{len(mids)} rings at {ring_km:.0f} km")
        return prof
    ok = mids >= EYE_MIN_EYEWALL_KM
    if not ok.any():
        prof["reason"] = "profile too short to locate an eyewall"
        return prof
    j = int(np.argmin(np.where(ok, means, np.inf)))
    ew_r = float(mids[j])
    # The coldest ring anywhere must BE that ring. If the profile is colder at
    # the centre than at ew_r, there is no clearing and nothing to score.
    if float(np.min(means)) < means[j] - 0.01:
        prof["reason"] = "no eye signature — coldest cloud is at the centre"
        return prof
    inside = r < ew_r
    if not (inside & np.isfinite(bt)).any():
        prof["reason"] = "no pixels inside the eyewall radius"
        return prof
    eye_warm = float(np.nanmax(bt[inside]))
    ann = (r >= ew_r - ring_km) & (r <= ew_r + ring_km) & np.isfinite(bt)
    eyewall_cold = float(np.nanmin(bt[ann])) if ann.any() else float(means[j])
    score = eye_warm - eyewall_cold
    if score <= 0:
        prof["reason"] = "eye is not warmer than its eyewall — no eye"
        return prof
    prof.update(score=score, eyewall_r_km=ew_r, eye_warm_c=eye_warm,
                eyewall_cold_c=eyewall_cold)
    return prof


# ---------------------------------------------------------------------------
# scene — everything both renderers derive from the same inputs, derived ONCE
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Scene:
    """The prepared inputs shared by the two-panel plot and the 2x2 plate.

    Both products must key off the SAME emphasised fix, the SAME comparability
    verdict and the SAME extremes, or the plate and the plot can disagree about
    the storm while carrying the same VALID stamp.
    """
    name: str
    sid: str
    storm: dict
    fixes: dict
    adv: Optional[dict]
    box: Optional[dict]
    satcon: Optional[dict]
    bbox: Optional[list]
    ir_c: Any
    ir_lat: Any
    ir_lon: Any
    ir_data: Any
    ir_min: Optional[float]
    ir_max: Optional[float]
    wv_min: Optional[float]
    wv_max: Optional[float]
    swir_min: Optional[float]
    swir_max: Optional[float]
    frame_lon: float
    pts: list
    newest: Optional[dict]
    acc: list
    rej: list
    emph: Optional[dict]
    sep_ok: bool
    sep_dt_min: Optional[float]


def prepare_scene(storm: dict, fixes: dict, adv: Optional[dict],
                  ir_data, wv_data, swir_data=None,
                  box: Optional[dict] = None, satcon: Optional[dict] = None,
                  bbox: Optional[list[float]] = None) -> Scene:
    ir_c = _bt_celsius(ir_data)
    ir_min, ir_max = _extremes(ir_c)
    wv_min = wv_max = swir_min = swir_max = None
    if wv_data is not None:
        wv_min, wv_max = _extremes(_bt_celsius(wv_data))
    if swir_data is not None:
        swir_min, swir_max = _extremes(_bt_celsius(swir_data))

    ir_lat, ir_lon = np.asarray(ir_data.lats), np.asarray(ir_data.lons)
    frame_lon = float(np.nanmean(ir_lon))
    pts = [p for p in (fixes.get("points") or [])
           if p.get("lat") is not None and p.get("lon") is not None]
    newest = pts[-1] if pts else None
    acc = [p for p in pts if p.get("fix")]
    rej = [p for p in pts if not p.get("fix")]
    # EMPHASIS + the disagreement measurement use the newest ACCEPTED fix.
    # The newest FRAME's candidate is often rejected (shear, no eye), and
    # hanging the headline crosshair, the certainty rings and the km label off
    # a rejected candidate would present a number ARCHER itself refused.
    # When they differ the header says so rather than quietly back-dating.
    emph = acc[-1] if acc else None
    sep_ok, sep_dt_min = False, None
    if emph and storm.get("last_fix"):
        try:
            ot = dt.datetime.fromisoformat(
                str(storm["last_fix"]).replace("Z", "+00:00"))
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=dt.timezone.utc)
            ft = dt.datetime.fromisoformat(emph["t"].replace("Z", "+00:00"))
            sep_dt_min = abs((ft - ot).total_seconds()) / 60.0
            sep_ok = sep_dt_min <= SEP_TOL_MIN
        except Exception:      # noqa: BLE001 - unparseable stamp -> no claim
            sep_ok, sep_dt_min = False, None
    return Scene(
        name=(storm.get("name") or "").upper(),
        sid=storm.get("id") or storm.get("sid") or "",
        storm=storm, fixes=fixes, adv=adv, box=box, satcon=satcon, bbox=bbox,
        ir_c=ir_c, ir_lat=ir_lat, ir_lon=ir_lon, ir_data=ir_data,
        ir_min=ir_min, ir_max=ir_max, wv_min=wv_min, wv_max=wv_max,
        swir_min=swir_min, swir_max=swir_max,
        frame_lon=frame_lon, pts=pts, newest=newest, acc=acc, rej=rej,
        emph=emph, sep_ok=sep_ok, sep_dt_min=sep_dt_min)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def _draw_field(ax, sc: Scene, field, cmap, norm, contours: bool):
    ax.pcolormesh(sc.ir_lon, sc.ir_lat, np.ma.masked_invalid(field),
                  cmap=cmap, norm=norm, shading="auto", zorder=1)
    if contours:
        # BD-style step contours: the Dvorak enhancement's own grey-shade
        # boundaries (CIMSS), drawn as isotherms over the grayscale so eye
        # / eyewall structure reads without colour.
        ax.contour(sc.ir_lon, sc.ir_lat, np.ma.masked_invalid(field),
                   levels=BD_STEPS, colors="#9fb3cc", linewidths=0.55,
                   alpha=0.75, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def _panel_label(ax, text: str, fontsize: float = 8.5):
    ax.text(0.012, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            color=TEXT_COLOR, fontsize=fontsize, fontweight="bold",
            bbox=dict(facecolor="#0a1019", alpha=0.62, edgecolor="none",
                      pad=3), zorder=12)


def panel_centres(ax1, sc: Scene, legend_fontsize: float = 7.4,
                  label_fontsize: float = 8.5):
    """Panel 1 — grayscale IR + every centre estimate, keyed."""
    storm, adv, box, bbox = sc.storm, sc.adv, sc.box, sc.bbox
    frame_lon, acc, rej, emph = sc.frame_lon, sc.acc, sc.rej, sc.emph
    sep_ok = sc.sep_ok

    _draw_field(ax1, sc, sc.ir_c, plt.get_cmap("gray_r"),
                Normalize(vmin=-90.0, vmax=30.0), contours=True)

    handles: list[tuple[str, dict]] = []

    # floater / target box
    if box:
        w, s, e, n = box["w"], box["s"], box["e"], box["n"]
        ax1.plot([w, e, e, w, w], [s, s, n, n, s], color=C_BOX, lw=1.3,
                 ls=(0, (6, 4)), zorder=5)
        # Only KEY it when an edge is actually inside the view. The floater
        # box is 12 deg and this window is BOX_DEG, so it is usually outside
        # the frame entirely - a legend entry for a line nobody can see sends
        # the reader hunting for it.
        if bbox and len(bbox) == 4:
            visible = (w > bbox[0] or e < bbox[2] or
                       s > bbox[1] or n < bbox[3])
        else:
            visible = True
        if visible:
            handles.append(("floater / target box", dict(color=C_BOX,
                                                         ls="--")))

    # ARCHER/ADT objective fixes — the whole recent track
    if len(acc) > 1:
        ax1.plot([_norm_lon(p["lon"], frame_lon) for p in acc],
                 [p["lat"] for p in acc], color=C_ARCHER, lw=1.1,
                 alpha=0.55, zorder=6)
    if rej:
        ax1.scatter([_norm_lon(p["lon"], frame_lon) for p in rej],
                    [p["lat"] for p in rej], s=26, marker="x",
                    c=C_ARCHER_WEAK, linewidths=1.2, zorder=6)
        handles.append(("ARCHER candidate — rejected (low confidence)",
                        dict(color=C_ARCHER_WEAK, marker="x")))
    if acc:
        ax1.scatter([_norm_lon(p["lon"], frame_lon) for p in acc],
                    [p["lat"] for p in acc], s=22, marker="o",
                    facecolors="none", edgecolors=C_ARCHER, linewidths=1.1,
                    zorder=7)
        handles.append(("ARCHER/ADT objective fix", dict(color=C_ARCHER,
                                                         marker="o")))
    if emph:
        nx, ny = _norm_lon(emph["lon"], frame_lon), emph["lat"]
        ax1.plot(nx, ny, marker="+", ms=17, mew=2.0, color=C_ARCHER, zorder=9)
        # ARCHER's own position-certainty rings
        for rk, style in ((emph.get("r50_km"), (0, (3, 2))),
                          (emph.get("r95_km"), (0, (1, 3)))):
            if not rk:
                continue
            rdeg = rk / 111.0
            th = np.linspace(0, 2 * np.pi, 181)
            ax1.plot(nx + rdeg * np.cos(th) /
                     max(0.2, math.cos(math.radians(ny))),
                     ny + rdeg * np.sin(th), color=C_ARCHER, lw=0.9,
                     ls=style, alpha=0.8, zorder=8)
        handles.append(("ARCHER 50% / 95% position certainty",
                        dict(color=C_ARCHER, ls=":")))

    # official best-track position
    off_lat, off_lon = storm.get("lat"), storm.get("lon")
    if off_lat is not None and off_lon is not None:
        ox = _norm_lon(float(off_lon), frame_lon)
        ax1.plot(ox, float(off_lat), marker="P", ms=13, color=C_OFFICIAL,
                 mec="#0a1019", mew=1.0, zorder=10)
        handles.append(("official best-track position",
                        dict(color=C_OFFICIAL, marker="P")))
        # The DISAGREEMENT is the point — but only when the two positions are
        # CONTEMPORANEOUS. The official position is a snapshot at its own fix
        # time; the objective fix is valid at its frame time. Measuring across
        # an 8 h offset reports the storm's own MOTION as method disagreement
        # (a 10 kt storm covers ~150 km in 8 h) and overstates it badly.
        # Within tolerance: draw the connector and the number. Outside it:
        # draw both markers, omit the number, and say why in the header.
        if emph:
            import matplotlib.patheffects as pe
            ex = _norm_lon(emph["lon"], frame_lon)
            d_km = _km_between(float(off_lat), float(off_lon),
                               emph["lat"], emph["lon"])
            ax1.plot([ox, ex], [float(off_lat), emph["lat"]],
                     color="#ffffff" if sep_ok else C_ARCHER_WEAK,
                     lw=0.9, ls=(0, (2, 2)),
                     alpha=0.85 if sep_ok else 0.5, zorder=9)
            if sep_ok:
                ax1.annotate(f"{d_km:.0f} km",
                             xy=((ox + ex) / 2,
                                 (float(off_lat) + emph["lat"]) / 2),
                             color="#ffffff", fontsize=9, fontweight="bold",
                             ha="center", va="bottom", zorder=11,
                             path_effects=[pe.withStroke(linewidth=3,
                                           foreground="#0a1019")])

    # official forecast track
    if adv and (adv.get("points") or []):
        fp = [(p["lat"], p["lon"]) for p in adv["points"]
              if p.get("lat") is not None and p.get("lon") is not None]
        if len(fp) > 1:
            ax1.plot([_norm_lon(p[1], frame_lon) for p in fp],
                     [p[0] for p in fp], color=C_FORECAST, lw=1.4,
                     ls=(0, (5, 3)), zorder=6, alpha=0.95)
            ax1.scatter([_norm_lon(p[1], frame_lon) for p in fp],
                        [p[0] for p in fp], s=14, c=C_FORECAST, zorder=6)
            handles.append(("official forecast track",
                            dict(color=C_FORECAST, ls="--")))

    # View window = the REQUESTED box, not the data array's own extent. A
    # geostationary grid this far off nadir is not axis-aligned in lat/lon, so
    # using the array's min/max frames the tilted parallelogram (with empty
    # corners) instead of the storm.
    if bbox and len(bbox) == 4:
        ax1.set_xlim(float(bbox[0]), float(bbox[2]))
        ax1.set_ylim(float(bbox[1]), float(bbox[3]))
    else:
        ax1.set_xlim(float(np.nanmin(sc.ir_lon)), float(np.nanmax(sc.ir_lon)))
        ax1.set_ylim(float(np.nanmin(sc.ir_lat)), float(np.nanmax(sc.ir_lat)))
    _panel_label(ax1, "IR WINDOW · GRAYSCALE + BD-STEP CONTOURS",
                 fontsize=label_fontsize)

    # legend — every source keyed, drawn as real proxies so nothing is
    # identified by position alone
    from matplotlib.lines import Line2D
    proxies, labels = [], []
    for lab, kw in handles:
        proxies.append(Line2D([0], [0], color=kw.get("color", "#fff"),
                              marker=kw.get("marker", None),
                              ls=kw.get("ls", "-" if not kw.get("marker") else "none"),
                              mfc="none" if kw.get("marker") == "o" else kw.get("color"),
                              mew=1.2, ms=8, lw=1.4))
        labels.append(lab)
    if proxies:
        leg = ax1.legend(proxies, labels, loc="lower left",
                         fontsize=legend_fontsize,
                         framealpha=0.82, facecolor="#0a1019",
                         edgecolor=GRID, labelcolor=TEXT_COLOR,
                         borderpad=0.6, handlelength=2.4)
        leg.set_zorder(13)


def panel_enhanced(ax2, sc: Scene, xlim, ylim, label_fontsize: float = 8.5,
                   readout_fontsize: float = 8.6):
    """Panel 2 — the same scene in enhanced colour, with BAND-TAGGED extremes."""
    try:
        # the SAME enhancement the floater renders use, so this panel reads
        # identically to the imagery elsewhere on the site
        from colormaps import get_enhancement, enhancement_norm
        enh = get_enhancement("tat_neon")
        cmap2, norm2 = enh["cmap"], enhancement_norm("tat_neon")
    except Exception:                      # palette import is best-effort
        cmap2, norm2 = plt.get_cmap("turbo"), Normalize(vmin=-95.0, vmax=40.0)
    _draw_field(ax2, sc, sc.ir_c, cmap2, norm2, contours=False)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    _panel_label(ax2, "IR WINDOW · ENHANCED COLOUR", fontsize=label_fontsize)
    # BAND-TAGGED, always. An unlabelled min/max invites reading a WV frame's
    # -60 C as a cloud top; a band that did not arrive says so rather than
    # leaving a gap the reader fills in with the band above it.
    bt_lines = []
    if sc.ir_min is not None:
        bt_lines.append(f"IR BT     min {sc.ir_min:+.1f} °C   max {sc.ir_max:+.1f} °C")
    if sc.wv_min is not None:
        bt_lines.append(f"WV BT     min {sc.wv_min:+.1f} °C   max {sc.wv_max:+.1f} °C")
    else:
        bt_lines.append("WV BT     unavailable this cycle")
    if sc.swir_min is not None:
        bt_lines.append(f"SWIR BT   min {sc.swir_min:+.1f} °C   max {sc.swir_max:+.1f} °C")
    else:
        bt_lines.append("SWIR BT   unavailable this cycle")
    ax2.text(0.012, 0.022, "\n".join(bt_lines), transform=ax2.transAxes,
             ha="left", va="bottom", color=TEXT_COLOR, fontsize=readout_fontsize,
             family="monospace",
             bbox=dict(facecolor="#0a1019", alpha=0.72, edgecolor=GRID,
                       pad=4), zorder=12)


def panel_wpace(ax, sc: Scene, png: Optional[bytes],
                captured: Optional[dt.datetime]):
    """Panel 3 — the CycloLab wind/pressure + observed/projected ACE chart.

    Pasted in as a raster from the browser lane's capture of the REAL renderer.
    Nothing about the ACE method is recomputed here; see the module docstring.
    """
    ax.set_facecolor(DARK_BG)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if png:
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(io.BytesIO(png), format="png")
            ax.imshow(img, aspect="equal", interpolation="antialiased",
                      zorder=1)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.set_frame_on(True)
            age = ""
            if captured is not None:
                mins = (dt.datetime.now(dt.timezone.utc) -
                        captured).total_seconds() / 60.0
                # Say the age whenever it is big enough to matter. A chart from
                # two cycles ago beside fresh imagery, under one VALID stamp,
                # is the failure mode this line exists to prevent.
                if mins >= 45:
                    age = f"   ·   CHART CAPTURED {mins / 60.0:.1f} H AGO"
            _panel_label(ax, "WIND, PRESSURE & ACE   ·   OBSERVED + OFFICIAL "
                             "FORECAST" + age)
            return
        except Exception as e:      # noqa: BLE001 - a bad raster is a placeholder
            log.warning("wpace raster unusable: %s", e)
    # HONEST PLACEHOLDER. An empty frame here would read as "this storm has no
    # ACE", which is a claim; "not captured" is the truth.
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _panel_label(ax, "WIND, PRESSURE & ACE")
    ax.text(0.5, 0.5,
            "chart not captured this cycle\n\n"
            "the wind/pressure + observed/projected ACE panel is rendered by\n"
            "CycloLab and captured from it; it is deliberately not recomputed\n"
            "here, so when the capture is missing the panel stays empty",
            transform=ax.transAxes, ha="center", va="center", color=MUTED,
            fontsize=9.0, linespacing=1.7)


def panel_eye(ax, sc: Scene, prof: Optional[dict]):
    """Panel 4 — radial BT profile about the working centre + the eye score."""
    ax.set_facecolor("#060a12")
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    if prof is None:
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "EYE STRUCTURE")
        why = ("no working centre this cycle — the profile is taken about the\n"
               "emphasised objective fix, and every candidate was rejected"
               if sc.emph is None else
               "no radial profile this cycle — the IR field could not be\n"
               "resolved into rings about the working centre")
        ax.text(0.5, 0.5, why + "\n\n(the score is withheld, not defaulted)",
                transform=ax.transAxes, ha="center", va="center", color=MUTED,
                fontsize=9.0, linespacing=1.7)
        return

    r, mean_c = prof["r_km"], prof["mean_c"]
    lo = float(min(np.nanmin(prof["min_c"]), -85.0))
    hi = float(max(np.nanmax(prof["max_c"]), 15.0))
    # Headroom at the top for the panel label + the score, and at the bottom
    # for the construction note, so neither is drawn over the profile.
    span = hi - lo
    lo -= span * 0.30
    hi += span * 0.22

    # BD ladder behind the profile — the citable discretisation, and the same
    # steps panel 1 contours, so the two panels read as one instrument.
    for i, (lab, a, b) in enumerate(BD_LADDER):
        a2, b2 = max(a, lo), min(b, hi)
        if b2 <= a2:
            continue
        ax.axhspan(a2, b2, color="#ffffff" if i % 2 else "#7f8fa6",
                   alpha=0.045 if i % 2 else 0.075, zorder=0)
        ax.axhline(a2, color=GRID, lw=0.5, alpha=0.8, zorder=1)
        if b2 - a2 > span * 0.055:      # skip labels with no room to sit in
            # LEFT edge: the profile's own minimum sits at the eyewall radius
            # and the construction note sits bottom-right, so the first few km
            # of the x axis is the only strip nothing else wants.
            ax.text(prof["max_r_km"] * 0.010, (a2 + b2) / 2.0, lab, ha="left",
                    va="center", color=MUTED, fontsize=6.8, alpha=0.7,
                    zorder=2)

    ax.fill_between(r, prof["min_c"], prof["max_c"], color=C_ARCHER,
                    alpha=0.13, lw=0, zorder=3)
    ax.plot(r, mean_c, color=C_ARCHER, lw=1.8, zorder=5)

    if prof.get("score") is not None:
        import matplotlib.patheffects as pe
        ew = prof["eyewall_r_km"]
        ax.axvline(ew, color=C_OFFICIAL, lw=1.1, ls=(0, (4, 3)), zorder=6)
        ax.plot([0.0], [prof["eye_warm_c"]], marker="o", ms=7,
                mfc="none", mec=C_OFFICIAL, mew=1.6, zorder=7)
        ax.plot([ew], [prof["eyewall_cold_c"]], marker="o", ms=7,
                mfc="none", mec=C_FORECAST, mew=1.6, zorder=7)
        # Sits BELOW the panel label, not under it, on a solid backing so a BD
        # rung label can never end up reading through the number.
        ax.text(0.012, 0.885, f"EYE SCORE  {prof['score']:.1f} °C",
                transform=ax.transAxes, color=TEXT_COLOR, fontsize=12.5,
                fontweight="bold", ha="left", va="top", zorder=9,
                bbox=dict(facecolor="#060a12", alpha=0.9, edgecolor="none",
                          pad=3),
                path_effects=[pe.withStroke(linewidth=3,
                              foreground="#060a12")])
        detail = (f"warmest eye pixel {prof['eye_warm_c']:+.1f} °C   ·   "
                  f"coldest eyewall ring {prof['eyewall_cold_c']:+.1f} °C\n"
                  f"eyewall radius {ew:.0f} km")
    else:
        detail = prof.get("reason") or "no eye signature — score withheld"

    ax.set_xlim(0.0, prof["max_r_km"])
    ax.set_ylim(lo, hi)
    ax.set_xlabel("radius from the objective centre (km)", color=MUTED,
                  fontsize=8.2, labelpad=2)
    ax.set_ylabel("brightness temperature (°C)", color=MUTED, fontsize=8.2,
                  labelpad=2)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.55, zorder=1)
    _panel_label(ax, "EYE STRUCTURE   ·   RADIAL IR PROFILE")
    # The construction, on the panel. A contrast number without its regions is
    # not reproducible, and the regions here are OURS, not the ADT's. Wrapped
    # to short lines so nothing runs off the panel's right edge.
    ax.text(0.988, 0.022,
            detail + "\n"
            f"mean (line) · min–max envelope (band) · {prof['ring_km']:.0f} km rings\n"
            f"eyewall = coldest mean ring beyond {EYE_MIN_EYEWALL_KM:.0f} km "
            "(our construction)\nBD ladder CIMSS",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED,
            fontsize=7.3, linespacing=1.55,
            bbox=dict(facecolor="#060a12", alpha=0.86, edgecolor=GRID, pad=4),
            zorder=10)


# ---------------------------------------------------------------------------
# shared chrome
# ---------------------------------------------------------------------------
def _draw_header(fig, sc: Scene, hdr_h: float, title: str,
                 title_fontsize: float = 17.0, body_fontsize: float = 9.2):
    """The ONE header both products carry: identity, valid time, forecast hour,
    the diagnostic state, and the intensity readouts."""
    hax = fig.add_axes([0, 1.0 - hdr_h, 1.0, hdr_h])
    hax.axis("off")
    hax.set_facecolor(DARK_BG)
    newest, emph, adv = sc.newest, sc.emph, sc.adv
    valid = newest["t"] if newest else None
    valid_txt = (valid.replace("T", " ").replace(".000Z", "Z")[:17] + "Z"
                 if valid else "—")
    hax.text(0.012, 0.78, f"{sc.name}  ·  {title}", ha="left",
             va="center", color=TEXT_COLOR, fontsize=title_fontsize,
             fontweight="bold", transform=hax.transAxes)
    # TWO EXPLICIT ROWS, not a greedy wrap. Row 1 is identity; row 2 is the
    # diagnostic state and the HONESTY NOTES. A greedy wrap pushed the
    # "separation not comparable" caveat onto a third row that then got
    # dropped -- losing precisely the line that must never be lost.
    row1 = [f"VALID {valid_txt}"]
    if adv and adv.get("points"):
        taus = [p.get("tau_h") for p in adv["points"]
                if p.get("tau_h") is not None]
        if taus:
            row1.append(f"FCST T+0 … T+{int(max(taus))}H")
        if adv.get("advisory") is not None:
            row1.append(f"ADVISORY {adv['advisory']}")
    if newest:
        row1.append(f"SCENE {newest.get('scene') or '—'}")
        if newest.get("confidence_score") is not None:
            row1.append(f"CONF {newest['confidence_score']:.2f}")
    row2 = []
    if newest and not newest.get("fix"):
        # The crosshair is on an OLDER accepted fix; never let that pass
        # silently, or the plot reads as a current objective centre.
        row2.append("NEWEST FRAME: CANDIDATE REJECTED")
        if emph:
            row2.append(f"CROSSHAIR {emph['t'][:16].replace('T', ' ')}Z")
    if sc.fixes.get("truncated"):
        row2.append("TRACK TRUNCATED (run budget)")
    if emph and sc.sep_dt_min is not None and not sc.sep_ok:
        row2.append(f"OFFICIAL FIX {sc.sep_dt_min / 60.0:.1f} H APART — "
                    f"SEPARATION NOT COMPARABLE")
    SEP = "   ·   "
    for i, row in enumerate((row1, row2)):
        if not row:
            continue
        hax.text(0.012, 0.44 - i * 0.30, SEP.join(row), ha="left",
                 va="center", color=MUTED, fontsize=body_fontsize,
                 transform=hax.transAxes)

    # intensity readouts: ADT and, when its membership rule is met, SATCON
    rows = []
    if newest:
        v = newest.get("vmax_kt")
        p_ = newest.get("mslp_mb")
        rows.append("ADT      " + (f"{v:.0f} kt" if v is not None else "— kt") +
                    (f"   {p_:.0f} mb" if p_ is not None else ""))
    satcon = sc.satcon
    if satcon and satcon.get("vmax") and satcon["vmax"].get("value") is not None:
        sv = satcon["vmax"]["value"]
        sp = (satcon.get("mslp") or {}).get("value")
        rows.append("SATCON   " + f"{sv:.0f} kt" +
                    (f"   {sp:.0f} mb" if sp is not None else ""))
    else:
        rows.append("SATCON   no consensus (needs ≥2 members)")
    off_kt = sc.storm.get("intensity_kt") or sc.storm.get("vmax")
    if off_kt:
        rows.append(f"OFFICIAL {float(off_kt):.0f} kt")
    hax.text(0.985, 0.5, "\n".join(rows), ha="right", va="center",
             color=TEXT_COLOR, fontsize=body_fontsize, family="monospace",
             transform=hax.transAxes, linespacing=1.5)


def _draw_footer(fig, sc: Scene, ftr_h: float):
    fax = fig.add_axes([0, 0, 1.0, ftr_h])
    fax.axis("off")
    fax.set_facecolor(DARK_BG)
    src = getattr(sc.ir_data, "sat_name", "geostationary")
    inp = sc.fixes.get("input") or ""
    # Two rows, not two columns: the disclosure and the provenance are both
    # long, and side-by-side they collided in the middle of the strip.
    fax.text(0.012, 0.68,
             "AUTOMATED OBJECTIVE SATELLITE ESTIMATE — experimental, not "
             "official. Centres from an ARCHER-style IR fix; intensity from an "
             "ADT-style estimate. See NHC / JTWC for official analyses.",
             ha="left", va="center", color=MUTED, fontsize=8,
             transform=fax.transAxes)
    fax.text(0.012, 0.24,
             f"@WeathermanAAA_   ·   imagery {src}" +
             (f"   ·   {inp}" if inp else ""),
             ha="left", va="center", color=MUTED, fontsize=7.6,
             transform=fax.transAxes)


# ---------------------------------------------------------------------------
# the plots
# ---------------------------------------------------------------------------
def render(storm: dict, fixes: dict, adv: Optional[dict],
           ir_data, wv_data, box: Optional[dict] = None,
           satcon: Optional[dict] = None, dpi: int = 110,
           bbox: Optional[list[float]] = None, swir_data=None,
           scene: Optional[Scene] = None) -> bytes:
    """Two-panel storm-centred centre-fix diagnostic -> PNG bytes."""
    sc = scene or prepare_scene(storm, fixes, adv, ir_data, wv_data,
                                swir_data=swir_data, box=box, satcon=satcon,
                                bbox=bbox)
    fig = plt.figure(figsize=(15.0, 8.2), facecolor=DARK_BG)
    # Fixed layout: header strip, two equal map panels, footer strip.
    hdr_h, ftr_h = 0.115, 0.055
    panel_y, panel_h = ftr_h + 0.035, 1.0 - hdr_h - ftr_h - 0.075
    axes = []
    for i in range(2):
        ax = fig.add_axes([0.035 + i * 0.483, panel_y, 0.455, panel_h])
        ax.set_facecolor("#060a12")
        axes.append(ax)
    panel_centres(axes[0], sc)
    panel_enhanced(axes[1], sc, axes[0].get_xlim(), axes[0].get_ylim())
    _draw_header(fig, sc, hdr_h, "OBJECTIVE CENTRE FIX")
    _draw_footer(fig, sc, ftr_h)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=DARK_BG)
    plt.close(fig)
    return buf.getvalue()


def render_composite(storm: dict, fixes: dict, adv: Optional[dict],
                     ir_data, wv_data, box: Optional[dict] = None,
                     satcon: Optional[dict] = None, dpi: int = 120,
                     bbox: Optional[list[float]] = None, swir_data=None,
                     wpace_png: Optional[bytes] = None,
                     wpace_captured: Optional[dt.datetime] = None,
                     scene: Optional[Scene] = None) -> bytes:
    """The 2x2 STORM DIAGNOSTIC PLATE -> PNG bytes.

        top-left      IR grayscale + BD contours + every centre estimate
        top-right     the same scene, enhanced colour, band-tagged extremes
        bottom-left   wind / pressure / observed + projected ACE (captured)
        bottom-right  eye structure — radial IR profile + the eye score

    One shared header, one shared footer. An additional product: render()
    keeps publishing the two-panel plot to its own key.
    """
    sc = scene or prepare_scene(storm, fixes, adv, ir_data, wv_data,
                                swir_data=swir_data, box=box, satcon=satcon,
                                bbox=bbox)
    # The bottom row is SHORTER than the top on purpose: the maps are square in
    # data and the two charts are wide, so equal rows would letterbox the
    # charts inside tall empty cells. 2:1 cells match the captured chart's own
    # 1000x480 frame almost exactly.
    fig = plt.figure(figsize=(15.0, 11.2), facecolor=DARK_BG)
    hdr_h, ftr_h = 0.094, 0.0446
    top_y, top_h = 0.397, 0.509
    # The bottom row starts ABOVE the footer, not on it: the eye panel carries
    # real tick labels and an x-label, and sitting the axes on ftr_h drew them
    # straight through the disclosure strip.
    bot_y, bot_h = 0.088, 0.262
    col_x = (0.035, 0.518)
    col_w = 0.447

    axes_top = []
    for x in col_x:
        ax = fig.add_axes([x, top_y, col_w, top_h])
        ax.set_facecolor("#060a12")
        axes_top.append(ax)
    axes_bot = []
    for x in col_x:
        ax = fig.add_axes([x, bot_y, col_w, bot_h])
        ax.set_facecolor("#060a12")
        axes_bot.append(ax)

    panel_centres(axes_top[0], sc)
    panel_enhanced(axes_top[1], sc, axes_top[0].get_xlim(),
                   axes_top[0].get_ylim())
    panel_wpace(axes_bot[0], sc, wpace_png, wpace_captured)

    prof = None
    if sc.emph is not None:
        prof = eye_score(sc.ir_c, sc.ir_lat, sc.ir_lon,
                         float(sc.emph["lat"]),
                         _norm_lon(float(sc.emph["lon"]), sc.frame_lon))
    panel_eye(axes_bot[1], sc, prof)

    _draw_header(fig, sc, hdr_h, "STORM DIAGNOSTIC PLATE",
                 title_fontsize=19.0, body_fontsize=9.6)
    _draw_footer(fig, sc, ftr_h)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=DARK_BG)
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_for_storm(entry: dict, sink=None, dpi: int = 110,
                    plate_dpi: int = 120) -> Optional[list]:
    """Fetch + render + publish one storm's plots.

    Returns a list of ``(key, png)`` — the two-panel centre-fix plot FIRST
    (unchanged, its own key) and the 2x2 plate SECOND. The plate is additive:
    if it fails to render, the two-panel plot has already been produced and is
    still returned.
    """
    sid = entry.get("id")
    slug = entry.get("slug")
    fixes = load_fixes(sid)
    if not fixes or not (fixes.get("points") or []):
        log.warning("%s: no published objfix track — skipped", sid)
        return None
    pts = [p for p in fixes["points"] if p.get("lat") is not None]
    if not pts:
        log.warning("%s: track has no positions — skipped", sid)
        return None
    # The INDEX entry carries only identity + counts; the OFFICIAL position
    # lives in the track's storm_feed snapshot. Without this merge the plot
    # silently loses the official marker AND the objective-vs-official
    # separation - i.e. the entire point of the panel - while still rendering
    # a confident-looking image. (Missed locally because the dev path passed
    # the storm_feed dict straight in.)
    feed = fixes.get("storm_feed") or {}
    for k, v in feed.items():
        entry.setdefault(k, v)
    # last_fix (the official position's VALID TIME) lives only in the floater
    # index; without it the separation cannot be shown as a like-for-like
    # measurement. Its own lat/lon win, since they are stamped with that time.
    off = load_official_fix(sid)
    for k in ("lat", "lon", "last_fix", "intensity_kt", "nature", "category"):
        if off.get(k) is not None:
            entry[k] = off[k]
    if entry.get("lat") is None or entry.get("lon") is None:
        log.warning("%s: no official position available — the separation "
                    "measurement will be omitted", sid)
    # Newest by TIME, not by position in the array: a truncated loop run can
    # leave the list ordered but short, and the plot dates itself off this.
    newest = max(pts, key=lambda p: p["t"])
    # tz-AWARE: the satellite resolver compares against aware epoch constants.
    when = dt.datetime.fromisoformat(newest["t"].replace("Z", "+00:00"))
    clat = float(entry.get("lat") if entry.get("lat") is not None
                 else newest["lat"])
    clon = float(entry.get("lon") if entry.get("lon") is not None
                 else newest["lon"])
    half = BOX_DEG / 2.0
    # bbox is [W, S, E, N] (lon first) — the render service's convention.
    bbox = [clon - half, clat - half, clon + half, clat + half]
    # FETCH wider than we DISPLAY. A geostationary grid is not axis-aligned in
    # lat/lon, so a fetch of exactly the display box comes back as a rotated
    # patch whose corners fall short of it — the panel then shows slanted data
    # edges. The margin is thrown away by the display crop.
    fm = half * 0.35
    fetch_bbox = [bbox[0] - fm, bbox[1] - fm, bbox[2] + fm, bbox[3] + fm]

    async def _fetch_all():
        ir = await _fetch_band(fetch_bbox, when, "clean_ir")
        # WV and SWIR are READOUTS, not the plot. Each is guarded on its own so
        # one missing band never costs the other, and never costs the render.
        try:
            wv = await _fetch_band(fetch_bbox, when, "wv_upper")
        except Exception as e:      # noqa: BLE001
            log.warning("%s: WV band unavailable (%s)", sid, e)
            wv = None
        try:
            swir = await _fetch_band(fetch_bbox, when, "shortwave_ir")
        except Exception as e:      # noqa: BLE001
            log.info("%s: SWIR band unavailable (%s)", sid, e)
            swir = None
        return ir, wv, swir

    ir, wv, swir = asyncio.run(_fetch_all())
    adv = load_advisory(sid)
    box = target_box(load_floater_manifest(slug)) if slug else None
    # ONE scene for both products, so the plate and the plot can never disagree
    # about the emphasised fix or the comparability verdict.
    sc = prepare_scene(entry, fixes, adv, ir, wv, swir_data=swir, box=box,
                       satcon=fixes.get("satcon"), bbox=bbox)
    out = []
    png = render(entry, fixes, adv, ir, wv, box=box,
                 satcon=fixes.get("satcon"), dpi=dpi, bbox=bbox,
                 swir_data=swir, scene=sc)
    key = f"{R2_PREFIX}/{sid}.png"
    if sink is not None:
        sink.write_png(key, png)
    out.append((key, png))
    try:
        wp_png, wp_when = load_wpace_png(sid)
        plate = render_composite(entry, fixes, adv, ir, wv, box=box,
                                 satcon=fixes.get("satcon"), dpi=plate_dpi,
                                 bbox=bbox, swir_data=swir,
                                 wpace_png=wp_png, wpace_captured=wp_when,
                                 scene=sc)
        pkey = f"{R2_PREFIX}/{sid}_plate.png"
        if sink is not None:
            sink.write_png(pkey, plate)
        out.append((pkey, plate))
    except Exception as e:          # noqa: BLE001 - the plate is ADDITIVE
        log.exception("%s: plate render failed (the two-panel plot still "
                      "published): %s", sid, e)
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--storm", help="only this storm id/name substring")
    ap.add_argument("--out-dir", help="write PNGs here instead of R2")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--plate-dpi", type=int, default=120)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    idx = _get_json(f"{CDN}/{os.environ.get('OBJFIX_R2_PREFIX', 'cyclolab/objfix')}"
                    f"/index.json") or {}
    storms = idx.get("storms") or []
    if not storms:
        log.info("no storms in the objfix index — nothing to render")
        return 0
    sink = None
    if not a.out_dir:
        from intensity_poller import R2Sink
        sink = R2Sink()
    else:
        os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for st in storms:
        if a.storm and a.storm.lower() not in json.dumps(st).lower():
            continue
        try:
            # PER-STORM ISOLATION: one storm's missing band or bad geometry
            # must never take the whole lane down with it.
            got = build_for_storm(dict(st), sink=sink, dpi=a.dpi,
                                  plate_dpi=a.plate_dpi)
            if not got:
                continue
            for key, png in got:
                if a.out_dir:
                    path = os.path.join(a.out_dir, os.path.basename(key))
                    with open(path, "wb") as fh:
                        fh.write(png)
                    log.info("%s -> %s (%d bytes)", st.get("name"), path,
                             len(png))
                else:
                    log.info("%s -> %s (%d bytes)", st.get("name"), key,
                             len(png))
            n += 1
        except Exception as e:      # noqa: BLE001
            log.exception("%s: render failed: %s", st.get("name"), e)
    log.info("rendered %d storm(s)", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
