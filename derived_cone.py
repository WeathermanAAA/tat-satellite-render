"""derived_cone - the WP derived uncertainty envelope (CYCLOLAB_DESIGN.md §8.4).

JTWC issues a forecast track but no official "cone of uncertainty". A WP
storm gets the forecast positions buffered by JTWC's *published average
track-forecast error* at each lead time and the corridor swept around the
track. The radii are REAL (downloaded, image-verified) and ship as a
method-versioned blob (``cyclolab_radii_jtwc_wpac_mean_2015.json``) so a
re-pin to a newer verification year is a data edit, not a code edit.

Radii are applied **1:1** (radius = mean error) - we deliberately do NOT
apply NHC's 2/3 percentile scaling (NHC's number is a 67th-percentile of
its own per-case distribution, not 2/3 x mean; scaling a *mean* by it
would fabricate a too-small band). See §8.4.

GEOMETRY (pure stdlib ``math``, NO shapely):

  Forward great-circle (destination) solution on a sphere of radius
  ``R_NM`` n mi. A buffer radius ``r`` n mi maps to angular distance
  ``delta = r / R_NM`` radians; sampling a circle at ``CIRCLE_SAMPLES``
  bearings draws the buffer of one point.

CONSTRUCTION CHOICE - the CORRIDOR WALK (deterministic, concave-safe):

  The track is an ordered polyline ``p0..pn`` with per-point radius
  ``r_i``. We build a single closed ring by walking:

    1. LEFT rail forward: for i = 0..n, the point offset ``r_i`` to the
       LEFT of the local track bearing at ``p_i``.
    2. HALF-ARC around the terminal circle ``p_n``: from the left rail
       bearing, sweep the outer semicircle to the right rail bearing
       (caps the far end - keeps the apex/terminus rounded, not a
       chord).
    3. RIGHT rail backward: for i = n..0, the point offset ``r_i`` to the
       RIGHT of the local track bearing at ``p_i``.
    4. HALF-ARC around the start circle ``p_0`` back to the first left
       point, closing the ring.

  "Left bearing" at ``p_i`` = local track course - 90 deg; "right" =
  +90 deg. Local course at an interior point is the mean of the
  incoming/outgoing segment azimuths; at the ends it is the single
  adjacent segment azimuth. The walk yields the *swept corridor envelope*
  - the union outline of all buffer circles along a (possibly curving)
  track - WITHOUT a convex hull, so a recurving WP track keeps its
  concave inner edge instead of being bridged over. (A convex-hull union
  would be simpler but would over-claim area on the concave side of a
  recurver; the corridor walk is the honest envelope. Its one limitation:
  for a track that doubles back on itself within a buffer width the rails
  can self-cross - WP forecast tracks are monotone-ish over 120 h and do
  not, but a renderer should treat the ring as even-odd filled.)

  Degenerate ``tau 0`` radius is 0 in the table; a 0-radius apex would
  collapse the start cap to a point. We FLOOR the apex (and any) radius
  at ``MIN_RADIUS_NM`` = 10 n mi so the terminus/apex stays a visible
  rounded cap. Documented, deterministic, and small relative to the 24 h
  radius (39 n mi).

  SINGLE POINT: no track axis exists, so the ring is the full circle
  sampled at ``CIRCLE_SAMPLES`` bearings (closed).

The output ring is ``[[lon, lat], ...]`` (GeoJSON lon,lat order) and is
CLOSED (last vertex == first).
"""
from __future__ import annotations

import math

# Sphere radius in nautical miles (mean Earth radius 6371.0088 km / 1.852).
R_NM: float = 3440.065

# Buffer-circle sampling resolution (bearings around one point's circle and
# the per-cap semicircle arc density).
CIRCLE_SAMPLES: int = 72

# tau-0 radius is 0 in the JTWC table; floor every radius here so the apex
# cap is a visible rounded end rather than a degenerate point. 10 n mi is
# small vs the 24 h radius (39 n mi) and never widens a real lead time.
MIN_RADIUS_NM: float = 10.0


def _dest_point(lat: float, lon: float, bearing_deg: float,
                dist_nm: float) -> tuple[float, float]:
    """Great-circle destination from (lat, lon) along ``bearing_deg`` for
    ``dist_nm`` n mi. Returns (lon, lat) in degrees (GeoJSON order)."""
    delta = dist_nm / R_NM            # angular distance, radians
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    sin_phi2 = (math.sin(phi1) * math.cos(delta)
                + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))
    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * sin_phi2
    lam2 = lam1 + math.atan2(y, x)
    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0   # normalize
    return (lon2, math.degrees(phi2))


def _gc_dist_nm(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Great-circle distance (n mi) - haversine on the R_NM sphere."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2)
    return 2.0 * R_NM * math.asin(min(1.0, math.sqrt(a)))


def _initial_bearing(lat1: float, lon1: float,
                     lat2: float, lon2: float) -> float:
    """Initial great-circle bearing (deg, 0..360) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _mean_bearing(b1: float, b2: float) -> float:
    """Circular mean of two bearings (deg) - averages course at an interior
    point across the incoming/outgoing segments, wrap-safe."""
    x = math.cos(math.radians(b1)) + math.cos(math.radians(b2))
    y = math.sin(math.radians(b1)) + math.sin(math.radians(b2))
    if abs(x) < 1e-12 and abs(y) < 1e-12:   # antiparallel; fall back to b1
        return b1 % 360.0
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _radius_for_tau(tau_h: float, radii_nm_by_tau: dict[int, float]) -> float:
    """Linear-interpolated radius (n mi) at ``tau_h`` from the table.

    Below the smallest table tau, interpolate from a (tau=0, r=0) origin so
    the apex grows smoothly; above the largest table tau, clamp to the last
    value (extrapolating an error envelope past the verification horizon
    would be unsupported by the source). The result is floored at
    ``MIN_RADIUS_NM``."""
    taus = sorted(radii_nm_by_tau)
    if not taus:
        return MIN_RADIUS_NM
    if tau_h <= taus[0]:
        # interpolate from origin (0,0) up to the first table point
        t0, r0 = taus[0], radii_nm_by_tau[taus[0]]
        r = r0 * (tau_h / t0) if t0 else r0
    elif tau_h >= taus[-1]:
        r = radii_nm_by_tau[taus[-1]]            # clamp at horizon
    else:
        lo = max(t for t in taus if t <= tau_h)
        hi = min(t for t in taus if t >= tau_h)
        if lo == hi:
            r = radii_nm_by_tau[lo]
        else:
            rlo, rhi = radii_nm_by_tau[lo], radii_nm_by_tau[hi]
            frac = (tau_h - lo) / (hi - lo)
            r = rlo + frac * (rhi - rlo)
    return max(r, MIN_RADIUS_NM)


def _circle_ring(lat: float, lon: float, radius_nm: float) -> list[list[float]]:
    """A single buffer circle as a closed [[lon,lat],...] ring."""
    ring = [list(_dest_point(lat, lon, 360.0 * k / CIRCLE_SAMPLES, radius_nm))
            for k in range(CIRCLE_SAMPLES)]
    ring.append(ring[0][:])   # close
    return ring


def _arc(lat: float, lon: float, radius_nm: float,
         from_brg: float, to_brg: float) -> list[list[float]]:
    """Outer-cap arc points around (lat,lon) sweeping CLOCKWISE (increasing
    bearing) from ``from_brg`` to ``to_brg``. Density tracks CIRCLE_SAMPLES.
    Endpoints are included so caps tie cleanly into the rails."""
    sweep = (to_brg - from_brg) % 360.0
    n = max(1, int(round(sweep / (360.0 / CIRCLE_SAMPLES))))
    pts = []
    for k in range(n + 1):
        b = from_brg + sweep * k / n
        pts.append(list(_dest_point(lat, lon, b, radius_nm)))
    return pts


def derive_cone(points: list[dict],
                radii_nm_by_tau: dict[int, float]) -> list[list[float]]:
    """Sweep the corridor envelope of the forecast track. ``points`` is the
    §8.3 forecast-point list (each carries ``tau_h``/``lat``/``lon``);
    ``radii_nm_by_tau`` maps integer tau -> radius n mi (the blob's
    ``nm_values`` with int keys). Returns a CLOSED ``[[lon,lat],...]`` ring.

    Algorithm: the TANGENT-CHAIN swept envelope - external tangent
    segments between consecutive tau circles, bridged by arcs on each
    circle between its incoming/outgoing contact bearings, with the far
    end capped by the last circle's outer arc and the start by the
    first circle's back arc. This is the geometrically exact boundary
    of the swept discs (C1-smooth, monotonically widening for growing
    radii); the previous corridor-walk chords bent at every tau and
    read as scallops. Single point -> full circle. Deterministic."""
    if not points:
        raise ValueError("derive_cone: empty points list")

    # Order by tau so the rails follow forecast time even if the caller
    # passes them out of order.
    pts = sorted(points, key=lambda p: p["tau_h"])
    coords = [(float(p["lat"]), float(p["lon"])) for p in pts]
    radii = [_radius_for_tau(float(p["tau_h"]), radii_nm_by_tau) for p in pts]

    if len(coords) == 1:
        (lat, lon), r = coords[0], radii[0]
        return _circle_ring(lat, lon, r)

    # DOMINATION FILTER: a slow-moving storm's growing radii can engulf
    # earlier circles entirely (d + r_i <= r_j); engulfed circles have
    # no external tangent and corrupt the chain. The swept envelope of
    # the family equals the envelope of the non-dominated subset, so
    # drop any circle fully inside another (order preserved).
    keep = []
    for i in range(len(coords)):
        dominated = False
        for j in range(len(coords)):
            if i == j:
                continue
            d_ij = _gc_dist_nm(coords[i][0], coords[i][1],
                               coords[j][0], coords[j][1])
            if d_ij + radii[i] <= radii[j] + 0.1:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    coords = [coords[i] for i in keep]
    radii = [radii[i] for i in keep]
    if len(coords) == 1:
        (lat, lon), r = coords[0], radii[0]
        return _circle_ring(lat, lon, r)

    n = len(coords)

    # ---- LOCAL PLANAR FRAME (S4-AD1 #9) -------------------------------
    # The tangent-chain construction is exact, simple 2D geometry, so we
    # run it in a local equirectangular plane (n mi units; lon scaled by
    # cos of the mid latitude) and map back. Over basin-scale cones the
    # planar error is ~1-2% - immaterial for an uncertainty band - and
    # in exchange every contact, crossing and arc is EXACT:
    #   * outward contact normal between consecutive circles leans BACK
    #     by gamma = asin((r_j - r_i)/d) (it touches BOTH circles at the
    #     same normal);
    #   * where consecutive tangent lines CROSS (radius growth, or a
    #     bend into this side), the envelope vertex is their
    #     intersection - an arc there would backtrack;
    #   * where the turn opens OUTWARD, the junction circle's arc
    #     bridges the contacts;
    #   * far end capped by the last circle's outer arc (sweep
    #     180 + 2*gamma: the widening cone bulges past the half
    #     circle), start capped by the first circle's back arc.
    lat0 = sum(c[0] for c in coords) / n
    lon0 = coords[0][1]
    coslat = max(0.2, math.cos(math.radians(lat0)))

    def to_xy(lat: float, lon: float) -> tuple[float, float]:
        dl = (lon - lon0 + 540.0) % 360.0 - 180.0   # antimeridian-safe
        return (dl * 60.0 * coslat, (lat - lat0) * 60.0)

    def to_ll(x: float, y: float) -> list[float]:
        return [lon0 + x / (60.0 * coslat), lat0 + y / 60.0]

    P = [to_xy(la, lo) for la, lo in coords]
    R = radii

    def ang(v):
        return math.atan2(v[1], v[0])

    def arc_pts(c, r, a0, a1, cw):
        # sample the arc from angle a0 to a1 in the given direction,
        # endpoints EXCLUDED (contacts/junctions are emitted explicitly)
        if cw:
            while a1 > a0:
                a1 -= 2.0 * math.pi
            sweep = a0 - a1
        else:
            while a1 < a0:
                a1 += 2.0 * math.pi
            sweep = a1 - a0
        steps = max(1, int(sweep / math.radians(5.0)))
        out = []
        for k in range(1, steps):
            a = a0 - sweep * k / steps if cw else a0 + sweep * k / steps
            out.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
        return out

    def side_chain(sign: float):
        # sign = +1 left of track, -1 right. Returns the planar vertex
        # chain from circle 0's contact to circle n-1's contact.
        normals = []
        for i in range(n - 1):
            vx, vy = P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]
            d = max(math.hypot(vx, vy), 1e-9)
            ux, uy = vx / d, vy / d
            s = max(-0.99985, min(0.99985, (R[i + 1] - R[i]) / d))
            g = math.asin(s)
            # outward perpendicular for this side, leaned BACK by gamma
            px, py = -uy * sign, ux * sign
            cg, sg = math.cos(g), math.sin(g)
            normals.append((px * cg - ux * sg, py * cg - uy * sg))
        chain = [(P[0][0] + R[0] * normals[0][0],
                  P[0][1] + R[0] * normals[0][1])]
        for i in range(1, n - 1):
            n_in, n_out = normals[i - 1], normals[i]
            cin = (P[i][0] + R[i] * n_in[0], P[i][1] + R[i] * n_in[1])
            cout = (P[i][0] + R[i] * n_out[0], P[i][1] + R[i] * n_out[1])
            cross = n_in[0] * n_out[1] - n_in[1] * n_out[0]
            # outward bridge direction: left side bridges around the
            # circle when the normal rotates CW (cross < 0); right side
            # when CCW. Otherwise the tangent lines cross - emit their
            # intersection as the (single) envelope vertex.
            bridges = (cross < 0) if sign > 0 else (cross > 0)
            if abs(cross) < 1e-6:
                chain.append(cout)
            elif bridges:
                chain.append(cin)
                chain += arc_pts(P[i], R[i], ang(n_in), ang(n_out),
                                 cw=(sign > 0))
                chain.append(cout)
            else:
                # tangent directions (along travel) for both lines
                t_in = (-n_in[1] * sign, n_in[0] * sign)
                t_out = (-n_out[1] * sign, n_out[0] * sign)
                den = t_in[0] * (-t_out[1]) - t_in[1] * (-t_out[0])
                if abs(den) < 1e-9:
                    chain.append(cout)
                else:
                    bx, by = cout[0] - cin[0], cout[1] - cin[1]
                    t = (bx * (-t_out[1]) - by * (-t_out[0])) / den
                    chain.append((cin[0] + t_in[0] * t,
                                  cin[1] + t_in[1] * t))
        chain.append((P[-1][0] + R[-1] * normals[-1][0],
                      P[-1][1] + R[-1] * normals[-1][1]))
        return chain, normals

    left, lnorm = side_chain(+1.0)
    right, rnorm = side_chain(-1.0)

    ring_xy = list(left)
    # The ring as a whole travels CLOCKWISE (left flank forward keeps
    # the corridor on the traveler's right), so BOTH caps sweep cw:
    # terminal from the left contact around the FAR side to the right
    # contact, start from the right contact around the BACK side to
    # the left one. (The first cut swept these ccw - across the NEAR
    # side - and drove the boundary ~90 nm through the terminal
    # circle. Probe-caught.)
    ring_xy += arc_pts(P[-1], R[-1], ang(lnorm[-1]), ang(rnorm[-1]),
                       cw=True)
    ring_xy += list(reversed(right))
    ring_xy += arc_pts(P[0], R[0], ang(rnorm[0]), ang(lnorm[0]), cw=True)

    ring = [to_ll(x, y) for x, y in ring_xy]

    # Close the ring.
    if ring[0] != ring[-1]:
        ring.append(ring[0][:])
    return ring


def build_derived_advisory_json(sid: str, forecast_points: list[dict],
                                radii_blob: dict) -> dict:
    """Assemble the §8.3 cached-advisory contract for a WP derived cone.

    ``source`` is always ``"jtwc"``; ``method`` is the blob's
    ``method_version`` with the ``derived-mean-error-`` prefix
    (``derived-mean-error-jtwc-wpac-mean-2015``); ``cone`` comes from
    :func:`derive_cone`; ``points`` pass through verbatim; ``text`` is
    ``None`` (JTWC ships no parsed TCP/TCD here). Advisory number and
    issuance are read from the first forecast point's metadata if present,
    else left null. ``provenance`` records the radii method version and
    source doc so a rendered cone is self-describing."""
    nm_values = {int(k): float(v)
                 for k, v in radii_blob["nm_values"].items()}
    cone = derive_cone(forecast_points, nm_values)

    method_version = radii_blob["method_version"]
    issued = None
    advisory = None
    if forecast_points:
        p0 = min(forecast_points, key=lambda p: p["tau_h"])
        issued = p0.get("valid_utc")
        advisory = p0.get("advisory")

    return {
        "sid": sid,
        "advisory": advisory,
        "issued_utc": issued,
        "source": "jtwc",
        "method": f"derived-mean-error-{method_version}",
        "cone": cone,
        "points": list(forecast_points),
        "text": None,
        "provenance": {
            "radii_method_version": method_version,
            "radii_source_doc": radii_blob.get("source_doc"),
            "radii_source_md5": radii_blob.get("source_md5"),
            "min_radius_nm_floor": MIN_RADIUS_NM,
            "construction": "tangent-chain-swept-envelope",
        },
    }
