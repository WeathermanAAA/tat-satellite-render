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
# ne_10m maps-pass: country polygons -> land FILL (outer rings); the
# separate coastline file -> WHITE coast stroke. Two layers so the land
# fill can be stroke-free (no interior country borders) while the coast
# is a clean true-coast white line.
LAND_PATH = HERE / "cyclolab_ne_10m_land.geojson"
COAST_PATH = HERE / "cyclolab_ne_10m_coastline.geojson"

_LAND = None
_COAST = None

OCEAN_NAMES = {"AL": "ATLANTIC OCEAN", "EP": "PACIFIC OCEAN",
               "CP": "PACIFIC OCEAN", "WP": "PACIFIC OCEAN"}


def _land_rings() -> list[list[list[float]]]:
    """All land rings as [[lon, lat], ...] lists (outer rings only -
    holes are inland lakes, irrelevant at this scale)."""
    global _LAND
    if _LAND is None:
        gj = json.loads(LAND_PATH.read_text(encoding="utf-8"))
        rings = []
        for f in gj["features"]:
            g = f["geometry"]
            if g["type"] == "Polygon":
                rings.append(g["coordinates"][0])
            elif g["type"] == "MultiPolygon":
                rings.extend(p[0] for p in g["coordinates"])
        _LAND = rings
    return _LAND


def _coast_lines() -> list[list[list[float]]]:
    """All coastline polylines as [[lon, lat], ...] OPEN lines (the
    ne_10m_coastline features are LineString / MultiLineString - the
    coords ARE the line, never indexed [0] like a polygon ring)."""
    global _COAST
    if _COAST is None:
        gj = json.loads(COAST_PATH.read_text(encoding="utf-8"))
        lines = []
        for f in gj["features"]:
            g = f["geometry"]
            if not g:
                continue
            if g["type"] == "LineString":
                lines.append(g["coordinates"])
            elif g["type"] == "MultiLineString":
                lines.extend(g["coordinates"])
        _COAST = lines
    return _COAST


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
TOL_NEAR = 0.015      # deg ~ 1.7 km: crisp to ~2 px at a 130 px/deg zoom
TOL_FAR = 0.10        # deg ~ 11 km: periphery, usually off-canvas


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


def basemap_for(lat: float, lon: float, basin: str, *,
                dlat: float = 22.0, dlon: float = 30.0) -> dict:
    """The baked basemap dict for a storm at (lat, lon): land rings
    GEOMETRICALLY CLIPPED to the window (a continent bakes to just its
    in-window slab), lons normalized into the window frame so the
    antimeridian never splits a page, plus the ocean label."""
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
        f = min(1.0, d / dlon)
        return TOL_NEAR + (TOL_FAR - TOL_NEAR) * f

    out = []
    for ring in _land_rings():
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
        if max(xs) - min(xs) < 0.4 and max(ys) - min(ys) < 0.4:
            continue                      # speck islands: invisible here
        thin = _simplify(clipped, _adaptive_tol(clipped))
        if len(thin) >= 3:
            out.append([[round(x, 2), round(y, 2)] for x, y in thin])
    # COAST: same contiguous-unwrap + shift-into-frame, then OPEN-polyline
    # clip (may split one coastline into several in-window runs). Drawn as
    # a white stroke ON TOP of the light-gray land fill.
    coast_out = []
    for line in _coast_lines():
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
            thin = _simplify(run, _adaptive_tol(run))
            if len(thin) < 2:
                continue
            rd = [[round(x, 2), round(y, 2)] for x, y in thin]
            # 2dp rounding can collapse a short run to repeated points; dedup
            # and then DROP zero/sub-pixel-extent runs - a [[x,y],[x,y]] line
            # renders as a ~1px white DOT on open water (round linecap). The
            # land loop drops specks too; the coast keeps real short segments.
            ded = [rd[0]]
            for p in rd[1:]:
                if p != ded[-1]:
                    ded.append(p)
            if len(ded) < 2:
                continue
            xs = [p[0] for p in ded]
            ys = [p[1] for p in ded]
            if (max(xs) - min(xs)) < 0.02 and (max(ys) - min(ys)) < 0.02:
                continue
            coast_out.append(ded)
    return {
        "window": [round(la0, 2), round(la1, 2),
                   round(lo0, 2), round(lo1, 2)],
        "land": out,
        "coast": coast_out,
        "ocean": OCEAN_NAMES.get((basin or "").upper(), "OCEAN"),
    }
