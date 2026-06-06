"""CycloLab basemap bake (S4-AD1 #2): a storm-centered clip of vendored
Natural Earth land polygons (ne_50m admin_0, public domain, copied from
the house hafs_render bundle - no CDN, no runtime fetch), baked into
each storm page as a compact JS constant. The page renders it under THE
CONE: land fill slightly lighter than the ocean navy, subtle coast
stroke, a 5-degree graticule and a watermark ocean label.

The clip window is generous (+/-22 deg lat, +/-30 deg lon around the
current position) so any 120 h cone stays inside it; polygons are
bbox-filtered, antimeridian-normalized into the window frame, coords
rounded to 2 decimals and tiny rings dropped - mid-ocean storms bake to
a few hundred bytes, near-land storms to a few KB.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAND_PATH = HERE / "cyclolab_ne_50m_land.geojson"

_LAND = None

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


def _thin(pts, min_step=0.06):
    out = [pts[0]]
    for p in pts[1:]:
        if (abs(p[0] - out[-1][0]) >= min_step
                or abs(p[1] - out[-1][1]) >= min_step):
            out.append(p)
    return out


def basemap_for(lat: float, lon: float, basin: str, *,
                dlat: float = 22.0, dlon: float = 30.0) -> dict:
    """The baked basemap dict for a storm at (lat, lon): land rings
    GEOMETRICALLY CLIPPED to the window (a continent bakes to just its
    in-window slab), lons normalized into the window frame so the
    antimeridian never splits a page, plus the ocean label."""
    la0, la1 = lat - dlat, lat + dlat
    lo0, lo1 = lon - dlon, lon + dlon
    out = []
    for ring in _land_rings():
        norm = []
        for x, y in ring:
            while x - lon > 180.0:
                x -= 360.0
            while x - lon < -180.0:
                x += 360.0
            norm.append((x, y))
        clipped = _clip_ring(norm, lo0, la0, lo1, la1)
        if len(clipped) < 3:
            continue
        xs = [p[0] for p in clipped]
        ys = [p[1] for p in clipped]
        if max(xs) - min(xs) < 0.4 and max(ys) - min(ys) < 0.4:
            continue                      # speck islands: invisible here
        thin = _thin(clipped)
        if len(thin) >= 3:
            out.append([[round(x, 2), round(y, 2)] for x, y in thin])
    return {
        "window": [round(la0, 2), round(la1, 2),
                   round(lo0, 2), round(lo1, 2)],
        "land": out,
        "ocean": OCEAN_NAMES.get((basin or "").upper(), "OCEAN"),
    }
