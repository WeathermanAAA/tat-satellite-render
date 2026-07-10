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

GEOMETRY:

  Forward great-circle (destination) solution on a sphere of radius
  ``R_NM`` n mi. A buffer radius ``r`` n mi maps to angular distance
  ``delta = r / R_NM`` radians; sampling a circle at ``CIRCLE_SAMPLES``
  bearings draws the buffer of one point.

CONSTRUCTION CHOICE - the UNION-OF-DISKS BOUNDARY (the swept envelope,
topology-robust):

  A forecast cone must EXPAND with forecast time - ~0 width at the
  current position (NOW) growing monotonically to its widest at the last
  forecast hour - and it must not self-pinch on a recurve or loop. The
  honest envelope of "the storm centre is within ``r(tau)`` of the
  forecast track" is the OUTER BOUNDARY OF THE UNION of the forecast-
  time-scaled uncertainty disks, which has exactly those properties by
  construction (it cannot fold inward through a turn the way a fixed-
  offset corridor / tangent chain does - those bent or self-crossed on
  hard recurvers, the failure this construction replaces).

    1. Densely interpolate the centerline through the forecast points
       (great-circle), one disk every few n mi.
    2. Assign each interpolated centre a radius ``r(tau)`` from the
       per-lead-time mean-error table (small near NOW, large at +120 h;
       see ``_radius_for_tau``). ``r ~= 0`` at NOW.
    3. The cone = the outer boundary of the union of those disks. We
       evaluate the union's signed-distance field on a local planar grid
       (``f = max_k (r_k - dist)``; ``f >= 0`` is the union) and trace
       its zero-contour (marching squares, via ``contourpy``); the outer
       (largest-area) contour is the cone. This AUTOMATICALLY (a) grows
       from a point at NOW to the widest disk at the terminus and (b)
       handles any recurve/loop without a self-pinch - a fully-engulfed
       early disk simply never reaches the boundary, so no domination
       filter is needed.
    4. The traced boundary is resampled to uniform arc length and lightly
       smoothed so it renders as a clean curve (the front-end re-smooths
       with a centripetal Catmull-Rom for display).

  Degenerate ``tau 0`` radius is 0 in the table; a 0-radius apex would
  collapse the start cap to a point. We FLOOR the apex (and any) radius
  at ``MIN_RADIUS_NM`` = 10 n mi so the terminus/apex stays a visible
  rounded cap. Documented, deterministic, and small relative to the 24 h
  radius (39 n mi).

  SINGLE POINT: no track axis exists, so the ring is the full circle
  sampled at ``CIRCLE_SAMPLES`` bearings (closed).

Determinism: the grid, the contour and the resample are all data-driven
(no randomness), so a given input always yields the same ring; inputs are
sorted by tau, so the ring is independent of input order.

The output ring is ``[[lon, lat], ...]`` (GeoJSON lon,lat order) and is
CLOSED (last vertex == first).

Dependencies: ``numpy`` + ``contourpy`` (the marching-squares engine that
ships with matplotlib - both pinned in requirements.txt; the advisories
poller runs in the web service that installs them).
"""
from __future__ import annotations

import math

import numpy as np
import contourpy

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


def _gc_interp(p: tuple[float, float], q: tuple[float, float],
               f: float) -> tuple[float, float]:
    """Great-circle interpolation between (lat,lon) ``p`` and ``q`` at fraction
    ``f`` in [0,1] (slerp on the unit sphere). Returns (lat, lon). Densifies
    the centerline so the swept union follows the true great-circle path, not
    a chord, between sparse forecast points."""
    la1, lo1 = math.radians(p[0]), math.radians(p[1])
    la2, lo2 = math.radians(q[0]), math.radians(q[1])
    d = 2.0 * math.asin(min(1.0, math.sqrt(
        math.sin((la2 - la1) / 2.0) ** 2
        + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2.0) ** 2)))
    if d < 1e-9:
        return p
    a = math.sin((1.0 - f) * d) / math.sin(d)
    b = math.sin(f * d) / math.sin(d)
    x = a * math.cos(la1) * math.cos(lo1) + b * math.cos(la2) * math.cos(lo2)
    y = a * math.cos(la1) * math.sin(lo1) + b * math.cos(la2) * math.sin(lo2)
    z = a * math.sin(la1) + b * math.sin(la2)
    return (math.degrees(math.atan2(z, math.hypot(x, y))),
            math.degrees(math.atan2(y, x)))


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


# --- union-of-disks construction tuning (all n mi; see module docstring) ---
# Centre spacing when densifying the centerline. Adapts down with the local
# radius (``0.5 * r``) so the small NOW/apex disks still overlap into a smooth
# union, floored at MIN_DENS_NM so a long leg never explodes the centre count.
MIN_DENS_NM: float = 3.0
# Hard cap on densified centres - the signed-distance field loops over them, so
# a very long slow track is down-sampled to this many to keep the build bounded.
MAX_CENTERS: int = 700
# SDF grid cell, clamped to [GRID_MIN, GRID_MAX] and scaled to the cone span so
# a big cone stays a reasonable grid while a small one resolves the apex cap.
GRID_MIN_NM: float = 1.5
GRID_MAX_NM: float = 4.0
GRID_SPAN_DIV: float = 450.0
# Hard ceiling on SDF grid cells; a span wide enough to exceed it coarsens the
# cell beyond GRID_MAX so the build stays sub-second on a fast long-track cone.
CELL_BUDGET: int = 260_000
# Output boundary spacing (uniform arc length) after the contour is traced.
OUT_STEP_NM: float = 5.0


def _resample_closed(P: "np.ndarray", step: float) -> "np.ndarray":
    """Uniform arc-length resample of a closed polyline ``P`` (Nx2, NOT
    repeating the first vertex), at ~``step`` spacing. Returns the resampled
    open list (caller closes)."""
    closed = np.vstack([P, P[0]])
    seg = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return P
    m = max(12, int(round(total / step)))
    d = np.linspace(0.0, total, m, endpoint=False)
    idx = np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1)
    frac = (d - cum[idx]) / np.maximum(seg[idx], 1e-9)
    return closed[idx] + (closed[idx + 1] - closed[idx]) * frac[:, None]


def _smooth_closed(P: "np.ndarray", win: int) -> "np.ndarray":
    """Circular moving-average smooth of a closed polyline ``P`` (Nx2) with a
    box window of ``win`` samples - kills the marching-squares grid staircase
    (wavelength ~one cell) while leaving real cap/turn curvature (>> a cell)
    intact. ``win`` small vs the boundary length, so the inward pull on convex
    arcs is sub-tenth-nm (well under the uncertainty band's tolerance)."""
    n = len(P)
    if n < 5 or win < 3:
        return P
    k = np.ones(win) / win
    pad = win
    out = np.empty_like(P)
    for c in (0, 1):
        wrapped = np.concatenate([P[-pad:, c], P[:, c], P[:pad, c]])
        out[:, c] = np.convolve(wrapped, k, "same")[pad:pad + n]
    return out


def derive_cone(points: list[dict],
                radii_nm_by_tau: dict[int, float]) -> list[list[float]]:
    """The forecast-uncertainty cone as the OUTER BOUNDARY OF THE UNION of the
    forecast-time-scaled error disks. ``points`` is the §8.3 forecast-point
    list (each carries ``tau_h``/``lat``/``lon``); ``radii_nm_by_tau`` maps
    integer tau -> radius n mi (the blob's ``nm_values`` with int keys).
    Returns a CLOSED ``[[lon,lat],...]`` ring.

    The centerline is densely interpolated (great-circle), each interpolated
    centre is buffered by ``r(tau)`` (~0 at NOW, widest at the last hour), and
    the cone is the zero-contour of the union's signed-distance field
    ``f = max_k (r_k - dist)`` traced by marching squares. This expands
    monotonically with forecast time and cannot self-pinch on a recurve/loop
    (a fixed-offset corridor / tangent chain bent or self-crossed there). A
    fully-engulfed early disk never reaches the boundary, so no domination
    filter is needed. Single point -> full circle. Deterministic; input order
    independent (sorted by tau)."""
    if not points:
        raise ValueError("derive_cone: empty points list")

    # Order by tau so the centerline follows forecast time even if the caller
    # passes them out of order.
    pts = sorted(points, key=lambda p: p["tau_h"])
    coords = [(float(p["lat"]), float(p["lon"])) for p in pts]
    taus = [float(p["tau_h"]) for p in pts]

    if len(coords) == 1:
        (lat, lon) = coords[0]
        return _circle_ring(lat, lon, _radius_for_tau(taus[0], radii_nm_by_tau))

    # ---- densify the centerline -> a disk every few n mi -------------------
    centers: list[tuple[float, float]] = []
    rads: list[float] = []
    for i in range(len(coords) - 1):
        seg = _gc_dist_nm(coords[i][0], coords[i][1],
                          coords[i + 1][0], coords[i + 1][1])
        r_lo = min(_radius_for_tau(taus[i], radii_nm_by_tau),
                   _radius_for_tau(taus[i + 1], radii_nm_by_tau))
        dens = max(MIN_DENS_NM, 0.5 * r_lo)
        nstep = max(1, int(math.ceil(seg / dens)))
        for k in range(nstep):
            f = k / nstep
            la, lo = _gc_interp(coords[i], coords[i + 1], f)
            tau = taus[i] + (taus[i + 1] - taus[i]) * f
            centers.append((la, lo))
            rads.append(_radius_for_tau(tau, radii_nm_by_tau))
    centers.append(coords[-1])
    rads.append(_radius_for_tau(taus[-1], radii_nm_by_tau))

    # Bound the centre count (a very long slow track) by even down-sampling,
    # always keeping the first and last disk.
    if len(centers) > MAX_CENTERS:
        keep = sorted(set(
            [int(round(j)) for j in
             np.linspace(0, len(centers) - 1, MAX_CENTERS)]))
        centers = [centers[j] for j in keep]
        rads = [rads[j] for j in keep]

    # ---- local planar frame (n mi; lon scaled by cos of the mid latitude) --
    lat0 = sum(c[0] for c in coords) / len(coords)
    lon0 = coords[0][1]
    coslat = max(0.2, math.cos(math.radians(lat0)))

    def to_xy(la: float, lo: float) -> tuple[float, float]:
        dl = (lo - lon0 + 540.0) % 360.0 - 180.0   # antimeridian-safe
        return (dl * 60.0 * coslat, (la - lat0) * 60.0)

    def to_ll(x: float, y: float) -> list[float]:
        # NOT re-normalised to [-180,180]: vertices stay consistent relative to
        # lon0 (the front-end re-wraps to its basemap window), so a cone that
        # straddles the antimeridian is never split into a stray seam line.
        return [float(lon0 + x / (60.0 * coslat)), float(lat0 + y / 60.0)]

    C = np.array([to_xy(la, lo) for la, lo in centers])
    R = np.asarray(rads, dtype=float)

    # ---- signed-distance field of the union on a local grid ----------------
    # Grid cell scales with cone span, clamped to [GRID_MIN, GRID_MAX], THEN
    # coarsened further if the cell count would exceed CELL_BUDGET. A fast
    # long-track 120 h cone spans thousands of n mi; left uncapped, the grid x
    # centre evaluation reached ~1 M cells and stalled the advisories poller
    # for seconds. A 6000 nm uncertainty band tolerates a coarser sample, and
    # the boundary is smoothed downstream regardless.
    span = max(float(np.ptp(C[:, 0])), float(np.ptp(C[:, 1]))) + 2.0 * R.max()
    grid = min(GRID_MAX_NM, max(GRID_MIN_NM, span / GRID_SPAN_DIV))
    pad = R.max() + grid * 3.0
    cells = (((C[:, 0].max() - C[:, 0].min()) + 2.0 * pad) / grid
             * (((C[:, 1].max() - C[:, 1].min()) + 2.0 * pad) / grid))
    if cells > CELL_BUDGET:
        grid *= math.sqrt(cells / CELL_BUDGET)
        pad = R.max() + grid * 3.0
    xs = np.arange(C[:, 0].min() - pad, C[:, 0].max() + pad, grid)
    ys = np.arange(C[:, 1].min() - pad, C[:, 1].max() + pad, grid)
    gx, gy = np.meshgrid(xs, ys)
    # f = max_k (r_k - dist to centre k). Vectorised over centres in BLOCKS so
    # a big grid never churns one full-grid temporary per centre (the per-
    # centre Python loop was the stall); the block size bounds the transient.
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    field = np.full(flat.shape[0], -1.0e18)
    block = max(1, int(1_500_000 // max(1, flat.shape[0])))
    for s0 in range(0, len(R), block):
        cb, rb = C[s0:s0 + block], R[s0:s0 + block]
        dx = flat[:, 0][None, :] - cb[:, 0][:, None]
        dy = flat[:, 1][None, :] - cb[:, 1][:, None]
        cand = rb[:, None] - np.sqrt(dx * dx + dy * dy)
        np.maximum(field, cand.max(axis=0), out=field)
    field = field.reshape(gx.shape)

    # ---- trace the union boundary (f = 0) and keep the outer contour -------
    lines = contourpy.contour_generator(xs, ys, field).lines(0.0)
    if not lines:                      # numerically impossible (centres are
        raise RuntimeError("derive_cone: empty union contour")   # f>0 cells

    def _area(p: "np.ndarray") -> float:
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                         - np.dot(p[:, 1], np.roll(p[:, 0], -1)))

    outer = max(lines, key=_area)
    if outer.shape[0] > 1 and np.allclose(outer[0], outer[-1]):
        outer = outer[:-1]

    # ---- de-staircase: resample to uniform arc length, smooth, resample ----
    boundary = _resample_closed(outer, max(GRID_MIN_NM, grid))
    win = max(3, int(round(5.0 / max(GRID_MIN_NM, grid))))
    boundary = _smooth_closed(boundary, win)
    boundary = _resample_closed(boundary, OUT_STEP_NM)

    ring = [to_ll(x, y) for x, y in boundary]
    ring.append(ring[0][:])            # close
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
            "construction": "union-of-time-scaled-error-disks",
        },
    }
