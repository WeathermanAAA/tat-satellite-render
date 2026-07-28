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
             brightness-temperature readouts for the IR and water-vapour bands.

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


def load_fixes(storm_id: str) -> Optional[dict]:
    return _get_json(f"{CDN}/{os.environ.get('OBJFIX_R2_PREFIX', 'cyclolab/objfix')}"
                     f"/{storm_id}.json")


def load_advisory(storm_id: str) -> Optional[dict]:
    return _get_json(f"{CDN}/cyclolab/adv/{storm_id}.json")


def load_floater_manifest(slug: str) -> Optional[dict]:
    return _get_json(f"{CDN}/floaters/{slug}/manifest.json")


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


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------
def render(storm: dict, fixes: dict, adv: Optional[dict],
           ir_data, wv_data, box: Optional[dict] = None,
           satcon: Optional[dict] = None, dpi: int = 110,
           bbox: Optional[list[float]] = None) -> bytes:
    """Two-panel storm-centred centre-fix diagnostic -> PNG bytes."""
    name = (storm.get("name") or "").upper()
    sid = storm.get("id") or storm.get("sid") or ""

    ir_c = _bt_celsius(ir_data)
    ir_lat, ir_lon = np.asarray(ir_data.lats), np.asarray(ir_data.lons)
    ir_min, ir_max = _extremes(ir_c)
    wv_min = wv_max = None
    if wv_data is not None:
        wv_min, wv_max = _extremes(_bt_celsius(wv_data))

    frame_lon = float(np.nanmean(ir_lon))
    pts = [p for p in (fixes.get("points") or [])
           if p.get("lat") is not None and p.get("lon") is not None]
    newest = pts[-1] if pts else None

    fig = plt.figure(figsize=(15.0, 8.2), facecolor=DARK_BG)
    # Fixed layout: header strip, two equal map panels, footer strip.
    hdr_h, ftr_h = 0.115, 0.055
    panel_y, panel_h = ftr_h + 0.035, 1.0 - hdr_h - ftr_h - 0.075
    axes = []
    for i in range(2):
        ax = fig.add_axes([0.035 + i * 0.483, panel_y, 0.455, panel_h])
        ax.set_facecolor("#060a12")
        axes.append(ax)

    def _draw_field(ax, field, cmap, norm, contours: bool):
        ax.pcolormesh(ir_lon, ir_lat, np.ma.masked_invalid(field),
                      cmap=cmap, norm=norm, shading="auto", zorder=1)
        if contours:
            # BD-style step contours: the Dvorak enhancement's own grey-shade
            # boundaries (CIMSS), drawn as isotherms over the grayscale so eye
            # / eyewall structure reads without colour.
            levels = [-80.0, -75.0, -69.0, -63.0, -53.0, -41.0, -30.0]
            ax.contour(ir_lon, ir_lat, np.ma.masked_invalid(field),
                       levels=levels, colors="#9fb3cc", linewidths=0.55,
                       alpha=0.75, zorder=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)

    # ---- panel 1: grayscale IR + every centre estimate ------------------
    ax1 = axes[0]
    _draw_field(ax1, ir_c, plt.get_cmap("gray_r"),
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
    acc = [p for p in pts if p.get("fix")]
    rej = [p for p in pts if not p.get("fix")]
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
    # EMPHASIS + the disagreement measurement use the newest ACCEPTED fix.
    # The newest FRAME's candidate is often rejected (shear, no eye), and
    # hanging the headline crosshair, the certainty rings and the km label off
    # a rejected candidate would present a number ARCHER itself refused.
    # When they differ the header says so rather than quietly back-dating.
    emph = acc[-1] if acc else None
    # Is the official position contemporaneous enough with the emphasised fix
    # for their separation to mean anything? Tolerance is one synoptic step;
    # beyond that the gap is dominated by storm motion, not by method.
    SEP_TOL_MIN = 90.0
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
    elif emph:
        sep_dt_min = None
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
        ax1.set_xlim(float(np.nanmin(ir_lon)), float(np.nanmax(ir_lon)))
        ax1.set_ylim(float(np.nanmin(ir_lat)), float(np.nanmax(ir_lat)))
    ax1.text(0.012, 0.985, "IR WINDOW · GRAYSCALE + BD-STEP CONTOURS",
             transform=ax1.transAxes, ha="left", va="top", color=TEXT_COLOR,
             fontsize=8.5, fontweight="bold",
             bbox=dict(facecolor="#0a1019", alpha=0.62, edgecolor="none",
                       pad=3), zorder=12)

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
        leg = ax1.legend(proxies, labels, loc="lower left", fontsize=7.4,
                         framealpha=0.82, facecolor="#0a1019",
                         edgecolor=GRID, labelcolor=TEXT_COLOR,
                         borderpad=0.6, handlelength=2.4)
        leg.set_zorder(13)

    # ---- panel 2: enhanced colour IR + BT extremes -----------------------
    ax2 = axes[1]
    try:
        # the SAME enhancement the floater renders use, so this panel reads
        # identically to the imagery elsewhere on the site
        from colormaps import get_enhancement, enhancement_norm
        enh = get_enhancement("tat_neon")
        cmap2, norm2 = enh["cmap"], enhancement_norm("tat_neon")
    except Exception:                      # palette import is best-effort
        cmap2, norm2 = plt.get_cmap("turbo"), Normalize(vmin=-95.0, vmax=40.0)
    _draw_field(ax2, ir_c, cmap2, norm2, contours=False)
    ax2.set_xlim(*ax1.get_xlim())
    ax2.set_ylim(*ax1.get_ylim())
    ax2.text(0.012, 0.985, "IR WINDOW · ENHANCED COLOUR",
             transform=ax2.transAxes, ha="left", va="top", color=TEXT_COLOR,
             fontsize=8.5, fontweight="bold",
             bbox=dict(facecolor="#0a1019", alpha=0.62, edgecolor="none",
                       pad=3), zorder=12)
    bt_lines = []
    if ir_min is not None:
        bt_lines.append(f"IR BT   min {ir_min:+.1f} °C   max {ir_max:+.1f} °C")
    if wv_min is not None:
        bt_lines.append(f"WV BT   min {wv_min:+.1f} °C   max {wv_max:+.1f} °C")
    else:
        bt_lines.append("WV BT   unavailable this cycle")
    ax2.text(0.012, 0.022, "\n".join(bt_lines), transform=ax2.transAxes,
             ha="left", va="bottom", color=TEXT_COLOR, fontsize=8.6,
             family="monospace",
             bbox=dict(facecolor="#0a1019", alpha=0.72, edgecolor=GRID,
                       pad=4), zorder=12)

    # ---- header ----------------------------------------------------------
    hax = fig.add_axes([0, 1.0 - hdr_h, 1.0, hdr_h])
    hax.axis("off")
    hax.set_facecolor(DARK_BG)
    valid = newest["t"] if newest else None
    valid_txt = (valid.replace("T", " ").replace(".000Z", "Z")[:17] + "Z"
                 if valid else "—")
    hax.text(0.012, 0.78, f"{name}  ·  OBJECTIVE CENTRE FIX", ha="left",
             va="center", color=TEXT_COLOR, fontsize=17, fontweight="bold",
             transform=hax.transAxes)
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
    if fixes.get("truncated"):
        row2.append("TRACK TRUNCATED (run budget)")
    if emph and sep_dt_min is not None and not sep_ok:
        row2.append(f"OFFICIAL FIX {sep_dt_min / 60.0:.1f} H APART — "
                    f"SEPARATION NOT COMPARABLE")
    # Wrap the subhead rather than letting it run under the right-hand
    # intensity block. The honesty notes ("separation not comparable", "track
    # truncated") are the longest strings on the line and are exactly the ones
    # that must stay readable.
    SEP = "   ·   "
    for i, row in enumerate((row1, row2)):
        if not row:
            continue
        hax.text(0.012, 0.44 - i * 0.30, SEP.join(row), ha="left",
                 va="center", color=MUTED, fontsize=9.2,
                 transform=hax.transAxes)

    # intensity readouts: ADT and, when its membership rule is met, SATCON
    rows = []
    if newest:
        v = newest.get("vmax_kt")
        p_ = newest.get("mslp_mb")
        rows.append("ADT      " + (f"{v:.0f} kt" if v is not None else "— kt") +
                    (f"   {p_:.0f} mb" if p_ is not None else ""))
    if satcon and satcon.get("vmax") and satcon["vmax"].get("value") is not None:
        sv = satcon["vmax"]["value"]
        sp = (satcon.get("mslp") or {}).get("value")
        rows.append("SATCON   " + f"{sv:.0f} kt" +
                    (f"   {sp:.0f} mb" if sp is not None else ""))
    else:
        rows.append("SATCON   no consensus (needs ≥2 members)")
    off_kt = storm.get("intensity_kt") or storm.get("vmax")
    if off_kt:
        rows.append(f"OFFICIAL {float(off_kt):.0f} kt")
    hax.text(0.985, 0.5, "\n".join(rows), ha="right", va="center",
             color=TEXT_COLOR, fontsize=9.2, family="monospace",
             transform=hax.transAxes, linespacing=1.5)

    # ---- footer ----------------------------------------------------------
    fax = fig.add_axes([0, 0, 1.0, ftr_h])
    fax.axis("off")
    fax.set_facecolor(DARK_BG)
    src = getattr(ir_data, "sat_name", "geostationary")
    inp = fixes.get("input") or ""
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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=DARK_BG)
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_for_storm(entry: dict, sink=None, dpi: int = 110) -> Optional[str]:
    """Fetch + render + publish one storm's centre-fix plot. Returns the key."""
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
        try:
            wv = await _fetch_band(fetch_bbox, when, "wv_upper")
        except Exception as e:      # noqa: BLE001 - WV is a readout, not the plot
            log.warning("%s: WV band unavailable (%s)", sid, e)
            wv = None
        return ir, wv

    ir, wv = asyncio.run(_fetch_all())
    adv = load_advisory(sid)
    box = target_box(load_floater_manifest(slug)) if slug else None
    png = render(entry, fixes, adv, ir, wv, box=box,
                 satcon=fixes.get("satcon"), dpi=dpi, bbox=bbox)
    key = f"{R2_PREFIX}/{sid}.png"
    if sink is not None:
        sink.write_png(key, png)
    return key, png


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--storm", help="only this storm id/name substring")
    ap.add_argument("--out-dir", help="write PNGs here instead of R2")
    ap.add_argument("--dpi", type=int, default=110)
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
            got = build_for_storm(dict(st), sink=sink, dpi=a.dpi)
            if not got:
                continue
            key, png = got
            if a.out_dir:
                path = os.path.join(a.out_dir, os.path.basename(key))
                with open(path, "wb") as fh:
                    fh.write(png)
                log.info("%s -> %s (%d bytes)", st.get("name"), path, len(png))
            else:
                log.info("%s -> %s (%d bytes)", st.get("name"), key, len(png))
            n += 1
        except Exception as e:      # noqa: BLE001
            log.exception("%s: render failed: %s", st.get("name"), e)
    log.info("rendered %d plot(s)", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
