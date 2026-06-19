"""CycloLab basemap bake (S4-AD1 #2; ne_10m maps-pass): a storm-centered
clip of vendored Natural Earth geometry (ne_10m admin_0 land polygons +
ne_10m coastline, public domain, simplified to 2dp - no CDN, no runtime
fetch), baked into each storm page as a compact JS constant. The page
renders it under the cone / track / swath: LIGHT-GRAY land (clearly
lighter than the ocean navy), a WHITE coastline stroke from the separate
coastline polylines (so abutting country fills never paint interior
borders), a 5-degree graticule and a watermark ocean label.

The clip window is generous (+/-22 deg lat, +/-30 deg lon around the
current position) so any 120 h cone stays inside it; geometry is
bbox-filtered, antimeridian-normalized into the window frame, coords
rounded to 2 decimals and tiny rings dropped - mid-ocean storms bake to
a few hundred bytes, near-land storms to a few KB. Land polygons are
clipped as closed rings; coastline is clipped as OPEN polylines (a
distinct per-segment clipper - the land ring-clipper would fabricate a
closing segment across the window).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
# ne_10m maps-pass (round 2): THREE consistent layers.
#  * LAND = ne_10m_land, the PHYSICAL land-mass polygons -> light-gray fill.
#    The coastline is derived from this same land, so fill + coast MATCH
#    exactly (no dark unfilled islands with spiky outlines - the round-1 bug
#    came from filling political admin_0, which drops/differs on small islands
#    the physical coastline still drew).
#  * COAST is DERIVED from the land rings (their boundary minus the window-
#    edge clip segments) -> THICK white coast stroke that shares EXACT
#    vertices with the fill (no separate-file misalignment slivers).
#  * BORDER = ne_10m_admin_0_boundary_lines_land -> THIN white internal
#    country borders (no coast duplication - boundary_lines exclude coast).
# v3: the LAND + COAST now come from GSHHG (high-res, 'h'), a TRUE high-resolution
# GLOBAL shoreline - so the rendered coast matches the real terrain (bays / inlets
# / passes) instead of the generalized ne_10m. Pre-processed once from gshhs_h.b
# into a compact binary: level-1 (land) + level-2 (lakes), lat [-62,72] (drops the
# TC-irrelevant poles), each polygon lightly DP-simplified to 0.004 deg (terrain
# preserved) with a per-polygon bbox for fast window selection. ne_10m is kept ONLY
# for the admin borders (GSHHG has no borders), re-clipped to the GSHHG land.
GSHHS_PATH = HERE / "cyclolab_gshhs_coast.bin"
BORDER_PATH = HERE / "cyclolab_ne_10m_borders.geojson"
# ne_10m admin_1 state/province boundary LINES (internal; simplified to ~0.012deg,
# finer than the bake's TOL_NEAR so nothing is lost) -> a THIN, DIM state-border
# stroke UNDER the country borders. Adds the landfall detail (US/MX states, etc.)
# the cone + track maps were missing. Vendored (no CDN), like the others.
STATE_PATH = HERE / "cyclolab_ne_10m_states.geojson"

_BORDER = None
_STATE = None

OCEAN_NAMES = {"AL": "ATLANTIC OCEAN", "EP": "PACIFIC OCEAN",
               "CP": "PACIFIC OCEAN", "WP": "PACIFIC OCEAN"}


import struct

# GSHHG compact binary: b'GSH2' + count(uint32 LE); then per polygon a 21-byte
# header (level uint8, npts uint32, west/south/east/north int32 microdeg) followed
# by npts * (lon int32, lat int32) microdeg, lon already in -180..180.
_GSHHS_HDR = struct.Struct("<BIiiii")
_GSHHS_INDEX = None      # [(byte_offset, level, npts, w, s, e, n)] microdeg bbox


def _gshhs_index() -> list:
    """Scan the compact binary ONCE -> a light per-polygon index (offset + bbox +
    level + npts), so a bake reads only the polygons overlapping its window."""
    global _GSHHS_INDEX
    if _GSHHS_INDEX is None:
        idx = []
        with open(GSHHS_PATH, "rb") as f:
            if f.read(4) != b"GSH2":
                raise ValueError("bad GSHHG coast file magic")
            (count,) = struct.unpack("<I", f.read(4))
            hb = _GSHHS_HDR.size
            for _ in range(count):
                level, npts, w, s, e, n = _GSHHS_HDR.unpack(f.read(hb))
                idx.append((f.tell(), level, npts, w, s, e, n))
                f.seek(npts * 8, 1)
        _GSHHS_INDEX = idx
    return _GSHHS_INDEX


def _gshhs_window_rings(lo0, la0, lo1, la1, level=1, *,
                        lat=None, lon=None):
    """Read the GSHHG polygons of ``level`` whose bbox overlaps the window box
    [lo0,lo1]x[la0,la1] (deg) -> a list of rings [[lon,lat],...] (lon -180..180).
    The lon test is wrap-aware (+/-360) so an antimeridian window still selects.
    Dateline-spanning polygons (Eurasia) match every window but clip to nothing
    off-region - correct, only a small read cost.

    When ``lat``/``lon`` are given, a DISTANCE-ADAPTIVE speck filter drops tiny
    far islands by their INDEX bbox (without reading their points) - GSHHG-high
    has ~14 k polygons in a Gulf window, almost all sub-km islands that vanish at
    the window-edge zoom; skipping them at the index is the bake's big speedup."""
    cosl = max(0.3, math.cos(math.radians(lat))) if lat is not None else 1.0
    sel = []
    for (off, lv, npts, w, s, e, n) in _gshhs_index():
        if lv != level:
            continue
        sd, nd = s / 1e6, n / 1e6
        if nd < la0 or sd > la1:                    # lat reject
            continue
        wd, ed = w / 1e6, e / 1e6
        if not any(not (ed + sh < lo0 or wd + sh > lo1)
                   for sh in (0.0, 360.0, -360.0)):  # wrap-aware lon reject
            continue
        if lat is not None and npts < 6000:          # keep big polygons always
            span = max(ed - wd, nd - sd)
            cx = max(wd, min(ed, lon)) if wd <= ed else lon
            d = math.hypot((cx - lon) * cosl, max(sd, min(nd, lat)) - lat)
            floor = 0.012 if d < 8 else (0.06 if d < 16 else 0.22)
            if span < floor:
                continue
        sel.append((off, npts))
    rings = []
    with open(GSHHS_PATH, "rb") as f:
        for off, npts in sel:
            f.seek(off)
            pts = struct.unpack("<%di" % (2 * npts), f.read(npts * 8))
            rings.append([[pts[2 * i] / 1e6, pts[2 * i + 1] / 1e6]
                          for i in range(npts)])
    return rings


def _border_lines() -> list[list[list[float]]]:
    """Internal country border polylines (ne_10m_admin_0_boundary_lines_land,
    coast EXCLUDED) as open [[lon, lat], ...] lines, drawn as a THIN white
    stroke. (The COAST is derived from the land-fill rings - see _ring_coast -
    so it always aligns with the fill; there is no separate coastline file.)"""
    global _BORDER
    if _BORDER is None:
        gj = json.loads(BORDER_PATH.read_text(encoding="utf-8"))
        lines = []
        for f in gj["features"]:
            g = f["geometry"]
            if not g:
                continue
            if g["type"] == "LineString":
                lines.append(g["coordinates"])
            elif g["type"] == "MultiLineString":
                lines.extend(g["coordinates"])
        _BORDER = lines
    return _BORDER


def _state_lines() -> list[list[list[float]]]:
    """Internal STATE/PROVINCE boundary polylines (ne_10m admin_1 lines) as open
    [[lon, lat], ...] lines, drawn as a THIN DIM stroke UNDER the country
    borders. Absent file -> [] (graceful: the maps just lack state detail)."""
    global _STATE
    if _STATE is None:
        try:
            gj = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _STATE = []
            return _STATE
        lines = []
        for f in gj["features"]:
            g = f["geometry"]
            if not g:
                continue
            if g["type"] == "LineString":
                lines.append(g["coordinates"])
            elif g["type"] == "MultiLineString":
                lines.extend(g["coordinates"])
        _STATE = lines
    return _STATE


def _clip_ring(ring, lo0, la0, lo1, la1):
    """Sutherland-Hodgman clip of a polygon ring to the window box -
    a continent's ring reduces to just the slab inside the window."""
    def clip_edge(pts, inside, intersect):
        out = []
        for i, cur in enumerate(pts):
            prv = pts[i - 1]
            cin, pin = inside(cur), inside(prv)
            if cin:
                if not pin:
                    out.append(intersect(prv, cur))
                out.append(cur)
            elif pin:
                out.append(intersect(prv, cur))
        return out

    def x_at(a, b, x):
        t = (x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
        return (x, a[1] + (b[1] - a[1]) * t)

    def y_at(a, b, y):
        t = (y - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        return (a[0] + (b[0] - a[0]) * t, y)

    pts = ring
    for fn in (
        lambda p: clip_edge(p, lambda q: q[0] >= lo0,
                            lambda a, b: x_at(a, b, lo0)),
        lambda p: clip_edge(p, lambda q: q[0] <= lo1,
                            lambda a, b: x_at(a, b, lo1)),
        lambda p: clip_edge(p, lambda q: q[1] >= la0,
                            lambda a, b: y_at(a, b, la0)),
        lambda p: clip_edge(p, lambda q: q[1] <= la1,
                            lambda a, b: y_at(a, b, la1)),
    ):
        pts = fn(pts)
        if not pts:
            return []
    return pts


def _clip_polyline(line, lo0, la0, lo1, la1):
    """Clip an OPEN polyline to the window box, returning a LIST of
    polylines (a line entering/leaving the window yields disjoint runs).
    Per-segment Liang-Barsky; consecutive clipped pieces that share an
    endpoint stitch into one run. Unlike _clip_ring this never wraps the
    first/last vertices together, so a coastline that exits one side and
    re-enters elsewhere does NOT get a spurious closing segment."""
    eps = 1e-9

    def lb(a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = x1 - x0, y1 - y0
        p = (-dx, dx, -dy, dy)
        q = (x0 - lo0, lo1 - x0, y0 - la0, la1 - y0)
        u0, u1 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return None              # parallel & outside
            else:
                t = qi / pi
                if pi < 0:
                    if t > u1:
                        return None
                    if t > u0:
                        u0 = t
                else:
                    if t < u0:
                        return None
                    if t < u1:
                        u1 = t
        if u0 > u1:
            return None
        return ([x0 + u0 * dx, y0 + u0 * dy],
                [x0 + u1 * dx, y0 + u1 * dy])

    runs = []
    cur = []
    for i in range(len(line) - 1):
        seg = lb(line[i], line[i + 1])
        if seg is None:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            continue
        a2, b2 = seg
        if cur and abs(cur[-1][0] - a2[0]) < eps and abs(cur[-1][1] - a2[1]) < eps:
            cur.append(b2)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [a2, b2]
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def _ring_coast(ring, lo0, la0, lo1, la1):
    """Derive coast polylines from a CLIPPED land-fill ring, EXCLUDING the
    segments that lie on the window box edge (those are the clip box, not
    real coast). Coast and fill therefore share EXACT vertices, so the thick
    white coast can never misalign with the light-gray fill into a sliver
    (round-2 #2 - the round-1 separate-coastline-file approach drifted). The
    ring is treated as closed (the last vertex wraps to the first)."""
    eps = 0.02                       # > 2dp rounding; window edges are at 2dp

    def on_edge(a, b):
        return ((abs(a[0] - lo0) < eps and abs(b[0] - lo0) < eps) or
                (abs(a[0] - lo1) < eps and abs(b[0] - lo1) < eps) or
                (abs(a[1] - la0) < eps and abs(b[1] - la0) < eps) or
                (abs(a[1] - la1) < eps and abs(b[1] - la1) < eps))

    n = len(ring)
    runs = []
    cur = []
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        if a == b:
            continue
        if on_edge(a, b):
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
        elif cur and cur[-1] == a:
            cur.append(b)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [a, b]
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def _thin(pts, min_step=0.06):
    out = [pts[0]]
    for p in pts[1:]:
        if (abs(p[0] - out[-1][0]) >= min_step
                or abs(p[1] - out[-1][1]) >= min_step):
            out.append(p)
    return out


# Douglas-Peucker tolerance (degrees) for the baked geometry. ne_10m is
# sub-100m detail; grid-thinning keeps a vertex every min_step even on
# straight coast, which balloons near-land bakes. DP is SHAPE-preserving:
# straight/smooth runs collapse to their endpoints while bays/headlands
# stay, so a near-land bake drops from ~200 KB (grid-thin) to a few tens
# of KB at the SAME on-screen crispness. 0.01 deg ~= 1.1 km; at the
# tightest realistic cone auto-fit zoom (~120 px/deg) that is ~1.2 px, so
# the coast stays crisp even zoomed in, and at the full +/-30 window zoom
# it is sub-pixel.
SIMPLIFY_TOL = 0.01
# DISTANCE-ADAPTIVE simplification: the bake covers a generous +/-22/+/-30
# window so any 120 h cone fits, but the cone auto-fit usually crops the
# SVG to ~+/-8 deg around the (centered) storm, so most of the baked
# periphery is OFF-CANVAS. Keep the CENTER crisp (TOL_NEAR, where the cone
# actually shows, including the worst case of a tiny tightly-zoomed cone)
# and coarsen toward the edges (TOL_FAR) where geometry is rarely shown
# and, when it is, the cone is large so the zoom is loose. Cuts a near-
# land bake roughly in half vs uniform-fine with no visible center loss.
TOL_NEAR = 0.013      # deg ~ 1.4 km (v3: GSHHG-high source - the near-field coast
                      # is terrain-accurate at this light simplification, so bays/
                      # inlets/passes read at the cone auto-fit zoom). The page-
                      # size budget rises with the higher-fidelity coast.
TOL_FAR = 0.10        # deg ~ 11 km: rarely-shown window edge
# Island floors (deg). Tiny / thin near-shore islands, stroked with the
# THICK white coast, read as spiky white slivers, NOT land - so drop an
# island whose bbox is smaller than SPECK_DEG in its long axis OR thinner
# than THIN_DEG in its short axis. Checked AFTER simplification too, because
# the coarse periphery DP can COLLAPSE a small island into a sliver (round-2
# #2 - "check the simplification isn't creating spurs"). Real islands stay.
SPECK_DEG = 0.11
THIN_DEG = 0.06
AREA_MIN = 0.004      # deg^2 (~0.063x0.063): kills near-collinear slivers


def _simplify(pts, tol=None):
    """Iterative Douglas-Peucker (stack-based, no recursion-limit risk on
    long coastlines). Keeps endpoints; uniform-degree perpendicular
    distance (lon not cos-scaled - marginally more aggressive at high
    latitude, invisible at this scale). tol defaults to SIMPLIFY_TOL,
    resolved at CALL time so callers can pass a distance-adaptive value."""
    if tol is None:
        tol = SIMPLIFY_TOL
    n = len(pts)
    if n < 3:
        return list(pts)
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    t2 = tol * tol
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        ax, ay = pts[i0]
        bx, by = pts[i1]
        dx, dy = bx - ax, by - ay
        d2 = dx * dx + dy * dy
        dmax = t2
        idx = -1
        for i in range(i0 + 1, i1):
            px, py = pts[i]
            if d2 <= 0.0:
                pd = (px - ax) ** 2 + (py - ay) ** 2
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / d2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                cx, cy = ax + t * dx, ay + t * dy
                pd = (px - cx) ** 2 + (py - cy) ** 2
            if pd > dmax:
                dmax = pd
                idx = i
        if idx >= 0:
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return [pts[i] for i in range(n) if keep[i]]


# ---------------------------------------------------------------------------
# CLIP-TO-LAND (phase-4 v2 #1): the country/state border + the watch/warning
# zone polylines come from DIFFERENT datasets than the land/coast, so where a
# border follows a coast/river the two geometries DIVERGE and the border pokes
# off the land into the ocean. We clip every border/state line to the BAKED
# land rings (the exact geometry the drawn coast comes from), so no border
# segment is ever drawn over water and a coast-following border sits ON the
# drawn coast instead of dangling past it. Pure stdlib (ray-cast PIP + segment
# intersection) - no shapely.
# ---------------------------------------------------------------------------

def _point_in_land(pt, rings) -> bool:
    """Ray-cast: True iff pt is inside ANY land ring. The rings are disjoint
    landmasses (separate outer polygons), so membership is OR, not even-odd."""
    x, y = pt
    for ring in rings:
        n = len(ring)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def _seg_cross_t(a, b, c, d):
    """Parameter t in (0,1) along a->b where it crosses segment c-d, else None."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    if den == 0.0:
        return None                     # parallel / collinear
    qpx, qpy = c[0] - a[0], c[1] - a[1]
    t = (qpx * sy - qpy * sx) / den
    u = (qpx * ry - qpy * rx) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def _clip_line_to_land(line, rings):
    """Keep only the parts of an open polyline that lie inside the land rings.
    Each segment is split at every crossing with a land-ring edge; a sub-piece
    is kept iff its MIDPOINT is inside land. Consecutive kept pieces stitch into
    one run. Returns a list of on-land polylines."""
    eps = 1e-9
    # per-ring bbox: skip a whole ring whose bbox can't touch this line.
    rbox = [(min(p[0] for p in r), min(p[1] for p in r),
             max(p[0] for p in r), max(p[1] for p in r)) for r in rings]
    lx0 = min(p[0] for p in line); ly0 = min(p[1] for p in line)
    lx1 = max(p[0] for p in line); ly1 = max(p[1] for p in line)
    near = [i for i, b in enumerate(rbox)
            if not (b[2] < lx0 or b[0] > lx1 or b[3] < ly0 or b[1] > ly1)]
    runs = []
    cur = []
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        # segment bbox -> reject ring edges that can't cross it (cheap pre-test
        # before the float-heavy _seg_cross_t; the clip was the bake bottleneck).
        sx0 = a[0] if a[0] < b[0] else b[0]; sx1 = a[0] if a[0] > b[0] else b[0]
        sy0 = a[1] if a[1] < b[1] else b[1]; sy1 = a[1] if a[1] > b[1] else b[1]
        ts = [0.0, 1.0]
        for ri in near:
            ring = rings[ri]
            m = len(ring)
            for k in range(m):
                c = ring[k]; d = ring[(k + 1) % m]
                if (c[0] < sx0 and d[0] < sx0) or (c[0] > sx1 and d[0] > sx1) \
                        or (c[1] < sy0 and d[1] < sy0) \
                        or (c[1] > sy1 and d[1] > sy1):
                    continue
                t = _seg_cross_t(a, b, c, d)
                if t is not None and eps < t < 1.0 - eps:
                    ts.append(t)
        ts = sorted(set(ts))
        for j in range(len(ts) - 1):
            t0, t1 = ts[j], ts[j + 1]
            if t1 - t0 < eps:
                continue
            tm = 0.5 * (t0 + t1)
            mid = (a[0] + (b[0] - a[0]) * tm, a[1] + (b[1] - a[1]) * tm)
            if not _point_in_land(mid, rings):
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
                continue
            p0 = [a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0]
            p1 = [a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1]
            if cur and abs(cur[-1][0] - p0[0]) < eps \
                    and abs(cur[-1][1] - p0[1]) < eps:
                cur.append(p1)
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = [p0, p1]
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def _clip_lines_to_land(lines, rings, prec=2):
    """Clip each polyline to land, round to ``prec`` dp, dedup. [] land -> []."""
    if not rings:
        return []
    out = []
    for line in lines:
        for run in _clip_line_to_land(line, rings):
            rd = [[round(x, prec), round(y, prec)] for x, y in run]
            ded = [rd[0]]
            for p in rd[1:]:
                if p != ded[-1]:
                    ded.append(p)
            if len(ded) >= 2:
                out.append(ded)
    return out


def basemap_for(lat: float, lon: float, basin: str, *,
                dlat: float = 26.0, dlon: float = 50.0) -> dict:
    """The baked basemap dict for a storm at (lat, lon): land rings
    GEOMETRICALLY CLIPPED to the window (a continent bakes to just its
    in-window slab), lons normalized into the window frame so the
    antimeridian never splits a page, plus the ocean label.

    The window is baked GENEROUSLY beyond the cone/track/swath data bbox
    (default +-50 deg lon, +-26 deg lat) on purpose: the maps-pass
    ``fillExtent`` widens the rendered viewBox to the CARD aspect, so a wide /
    ultrawide monitor reveals far more lon than the data bbox. Land is clipped
    to THIS window, so a window narrower than the filled viewBox leaves a
    coastless ocean band at the edge (the China-cuts-off-at-~112E bug on a
    07W-at-143E cone). +-50 deg covers up to a ~21:9 panel with margin; the
    distance-adaptive DP (TOL_FAR) keeps the far periphery cheap. SST passes its
    own tight explicit dims, so it is unaffected by these defaults."""
    la0, la1 = lat - dlat, lat + dlat
    lo0, lo1 = lon - dlon, lon + dlon
    # distance-adaptive DP tolerance: a ring/line's tolerance ramps from
    # TOL_NEAR at the storm center to TOL_FAR at the window edge, keyed on
    # the geometry's CLOSEST approach to the center (so an island that
    # grazes the cone region stays crisp even if most of it is far). lon
    # deltas are cos-scaled to the storm latitude.
    cosl = max(0.3, math.cos(math.radians(lat)))

    def _adaptive_tol(pts):
        d = min(math.hypot((p[0] - lon) * cosl, p[1] - lat) for p in pts)
        # crisp within ~10 deg of the storm (the cone region, shown at high
        # zoom); only coarsen toward the rarely-shown +/-30 window edge.
        f = max(0.0, min(1.0, (d - 10.0) / max(1.0, dlon - 10.0)))
        return TOL_NEAR + (TOL_FAR - TOL_NEAR) * f

    out = []
    # v3: the land source is the windowed GSHHG high-res reader (the bake reads
    # only the polygons overlapping this storm's window). A generous lon margin
    # on the read keeps a continent that grazes the frame whole before unwrap.
    for ring in _gshhs_window_rings(lo0 - 1.0, la0 - 1.0, lo1 + 1.0, la1 + 1.0,
                                    lat=lat, lon=lon):
        # CONTIGUOUS unwrap: each vertex shifts by +-360 to stay within
        # 180 deg of the PREVIOUS one (per-vertex normalization tore
        # rings straddling the frame boundary - the clipper turned the
        # torn polylines into spurious full-window land bands), then the
        # whole ring shifts as one into the window's lon frame.
        norm = []
        prev = None
        for x, y in ring:
            if prev is not None:
                while x - prev > 180.0:
                    x -= 360.0
                while x - prev < -180.0:
                    x += 360.0
            norm.append([x, y])
            prev = x
        mean_x = sum(p[0] for p in norm) / len(norm)
        shift = 0.0
        while mean_x + shift - lon > 180.0:
            shift -= 360.0
        while mean_x + shift - lon < -180.0:
            shift += 360.0
        if shift:
            for p in norm:
                p[0] += shift
        clipped = _clip_ring([tuple(p) for p in norm], lo0, la0, lo1, la1)
        if len(clipped) < 3:
            continue
        xs = [p[0] for p in clipped]
        ys = [p[1] for p in clipped]
        # pre-DP skip (cheap): obvious tiny/thin rings.
        if max(max(xs) - min(xs), max(ys) - min(ys)) < SPECK_DEG or \
                min(max(xs) - min(xs), max(ys) - min(ys)) < THIN_DEG:
            continue
        thin = _simplify(clipped, _adaptive_tol(clipped))
        if len(thin) < 3:
            continue
        rounded = [[round(x, 2), round(y, 2)] for x, y in thin]
        ded = [rounded[0]]
        for p in rounded[1:]:
            if p != ded[-1]:
                ded.append(p)
        if len(ded) < 3:
            continue
        # POST-DP filter: the coarse periphery DP can collapse a small island
        # into a sliver the pre-DP bbox check could not see. Drop by bbox
        # (too small / too thin) AND by shoelace AREA - a near-collinear
        # 3-point sliver has a "fat" bbox but ~0 area, so the bbox test alone
        # misses it; the area floor is what kills the last white dashes.
        oxs = [p[0] for p in ded]
        oys = [p[1] for p in ded]
        if max(max(oxs) - min(oxs), max(oys) - min(oys)) < SPECK_DEG or \
                min(max(oxs) - min(oxs), max(oys) - min(oys)) < THIN_DEG:
            continue
        area = 0.0
        for k in range(len(ded)):
            x1, y1 = ded[k]
            x2, y2 = ded[(k + 1) % len(ded)]
            area += x1 * y2 - x2 * y1
        if abs(area) * 0.5 < AREA_MIN:
            continue
        out.append(ded)
        # COAST is no longer stored (v3 dedup): it is DERIVED from these land
        # rings at RENDER time (coast_from_land / the JS coastFromLand mirror) by
        # dropping the window-edge segments, so the high-res GSHHG coast is not
        # duplicated in the bake - the land + a derived coast at v2's byte cost.
    # BORDERS are OPEN polylines processed independently (boundary_lines):
    # contiguous-unwrap -> shift-into-frame -> open-polyline clip into the
    # in-window runs -> adaptive DP -> 2dp round -> dedup. `speck` drops a
    # run that collapsed to a sub-pixel dot (coast only; a border segment is
    # never a speck). Drawn as white strokes ON TOP of the land fill.
    def _proc_lines(lines, speck, tol_scale=1.0):
        res = []
        for line in lines:
            norm = []
            prev = None
            for x, y in line:
                if prev is not None:
                    while x - prev > 180.0:
                        x -= 360.0
                    while x - prev < -180.0:
                        x += 360.0
                norm.append([x, y])
                prev = x
            mean_x = sum(p[0] for p in norm) / len(norm)
            shift = 0.0
            while mean_x + shift - lon > 180.0:
                shift -= 360.0
            while mean_x + shift - lon < -180.0:
                shift += 360.0
            if shift:
                for p in norm:
                    p[0] += shift
            for run in _clip_polyline([tuple(p) for p in norm],
                                      lo0, la0, lo1, la1):
                thin = _simplify(run, _adaptive_tol(run) * tol_scale)
                if len(thin) < 2:
                    continue
                rd = [[round(x, 2), round(y, 2)] for x, y in thin]
                ded = [rd[0]]
                for p in rd[1:]:
                    if p != ded[-1]:
                        ded.append(p)
                if len(ded) < 2:
                    continue
                if speck > 0:
                    xs2 = [p[0] for p in ded]
                    ys2 = [p[1] for p in ded]
                    if (max(xs2) - min(xs2)) < speck and \
                            (max(ys2) - min(ys2)) < speck:
                        continue
                res.append(ded)
        return res

    border_out = _proc_lines(_border_lines(), 0.0)
    # State/province lines: clipped + simplified the SAME way as country
    # borders, then rendered UNDER them with a dimmer stroke (the maps draw
    # BASEMAP.states; an old baked basemap without the key degrades to []).
    # Baked only NEAR the storm (the cone + typical track extent) - far states
    # never show on these storm-centered maps and would blow the page-size
    # budget (the full window covers a whole continent of state lines). State
    # lines are also coarser (tol_scale) + speck-filtered: they are internal
    # admin boundaries that read as context, not the crisp coastline.
    _SLAT, _SLON = 11.0, 14.0

    def _near_storm(line):
        for x, y in line:
            if abs(y - lat) <= _SLAT and \
                    abs(((x - lon + 180.0) % 360.0) - 180.0) <= _SLON:
                return True
        return False

    state_out = _proc_lines([ln for ln in _state_lines() if _near_storm(ln)],
                            0.18, tol_scale=1.8)
    # CLIP-TO-LAND (v2 #1): bound the border + state polylines to the BAKED land
    # rings (the exact geometry the drawn coast is derived from), so no border
    # segment is ever drawn over water and a coast-following border sits ON the
    # drawn coast. Done LAST. v3: clip against a COARSENED copy of the land
    # (~0.05 deg) rather than the high-res GSHHG rings - a border just needs to be
    # bounded to land (it never reads sub-5 km coast detail), and clipping the
    # admin lines against ~20 k high-res land edges was the bake's bottleneck
    # (40 s -> ~3 s). The drawn COAST stays high-res; only the border-clip
    # boundary is coarse. Mid-ocean bake (no land) -> empty (correct).
    _clip_land = []
    for r in out:
        cr = _simplify([tuple(p) for p in r], 0.05)
        if len(cr) >= 3:
            _clip_land.append([list(p) for p in cr])
    border_out = _clip_lines_to_land(border_out, _clip_land, prec=2)
    state_out = _clip_lines_to_land(state_out, _clip_land, prec=2)
    return {
        "window": [round(la0, 2), round(la1, 2),
                   round(lo0, 2), round(lo1, 2)],
        "land": out,
        "borders": border_out,
        "states": state_out,
        "ocean": OCEAN_NAMES.get((basin or "").upper(), "OCEAN"),
    }


def coast_from_land(land_rings, window):
    """Derive the coast polylines from the BAKED land rings + the window box
    (la0,la1,lo0,lo1) - the land-ring boundary MINUS the window-edge segments, so
    the white coast shares EXACT vertices with the fill and no spurious stroke
    runs along the window box. The render-time mirror of this is the JS
    ``coastFromLand`` (v3 dedup: the coast is no longer stored in the bake)."""
    la0, la1, lo0, lo1 = window
    out = []
    for ring in land_rings:
        out.extend(_ring_coast([tuple(p) for p in ring], lo0, la0, lo1, la1))
    return out
