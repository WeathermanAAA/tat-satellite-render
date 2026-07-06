#!/usr/bin/env python3
"""Stable common-extent ("anchor") logic for the meso poller.

WHY THIS EXISTS
---------------
Each meso sector's *source* extent (the box the operators have GOES M1/M2 or the
Himawari Target pointed at) is read fresh per scan from the data
(``geospatial_lat_lon_extent`` for GOES, the HSD nav for Himawari -- see
``meso_poller.discover_*_extent``). The poller used to render **every band frame
to that live per-scan box**. The viewer just stacks the rendered frames with no
geo-referencing, so the loop is only as steady as the source extent. Measured
reality (2026-06-20):

  * GOES M1/M2: byte-identical scan-to-scan **within an operator position**
    (0 px wander over 45 min), with 1-2 *genuine* large repositions per 24 h
    (12-30 deg jumps as the operators re-steer onto new weather).
  * Himawari Target: the JMA Target is **continuously** re-steered to keep an
    active storm centred, so it creeps every slot (~0.1-0.3 deg/h, ~3.6 deg over
    24 h) -- a constant frame-to-frame crawl.
  * /render emits a fixed-WIDTH (1056 px) frame whose HEIGHT tracks the bbox
    aspect, so a drifting/repositioning box ALSO changes the frame's pixel
    dimensions -> vertical jitter / letterbox shift on top of the geographic
    drift.

THE FIX
-------
Pin each sector's loop to ONE reference extent (the "anchor") and render every
frame to it, so consecutive frames are geo-aligned AND pixel-identical -> the
loop is locked. The anchor is HELD (frames stay pixel-for-pixel stable) until
the live data box no longer covers it (the operators genuinely re-steered far
enough) or its span changes (an operator zoom), at which point the anchor SNAPS
to the new position. This converts Himawari's continuous per-frame crawl into a
handful of discrete, deliberate re-centres per day with pixel-locked eras
between them, and leaves GOES (already stable within a position) effectively
unchanged.

KEY GUARANTEES
--------------
  * **No blank edges.** The chosen anchor is always fully contained in the live
    data box (``plan_anchor`` verifies this and falls back to an inset of the
    live box if a candidate would poke outside it). For continuously-drifting
    sources (Himawari) the anchor is INSET from the live box by a per-family
    margin, so the box can drift up to that margin before the anchor needs to
    move -- the margin is the hysteresis that keeps the loop locked AND the
    coverage buffer that keeps every pixel backed by real data.
  * **GOES reduces to the old behaviour.** With margin 0 (the GOES default) the
    anchor equals the live box and snaps only on a genuine reposition -- exactly
    what the poller did before -- so this is zero-visual-change for GOES within a
    position.
  * **Constant dimensions within an era.** A drift re-anchor keeps the previous
    anchor's *span* and merely re-centres, so the frame's pixel dimensions don't
    change across a drift snap (only a genuine operator *zoom* changes them).

This module is PURE + stdlib-only (no s3fs/xarray/numpy), so it imports and unit
-tests without the heavy discovery deps -- mirroring the lazy-import discipline
in ``meso_poller``. All longitude math is antimeridian-safe (a box may cross
+/-180 with e < w, e.g. a Bering Sea GOES-18 M2 box).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math


# ---------------------------------------------------------------------------
# Longitude helpers (antimeridian-safe). bbox = [lon_w, lat_s, lon_e, lat_n];
# lon_e < lon_w is a VALID +/-180 crossing (the operators steer boxes across the
# dateline), so spans/centres/containment all use wrapped math.
# ---------------------------------------------------------------------------

def norm_lon(lon: float) -> float:
    """Wrap a longitude into [-180, 180]."""
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def lon_span(bbox) -> float:
    """Longitudinal width in degrees, wrap-aware. e < w wraps to a positive
    span; a zero remainder is the full-width [-180, 180] box -> 360."""
    return (bbox[2] - bbox[0]) % 360.0 or 360.0


def center_lon(bbox) -> float:
    """Centre longitude, +/-180-crossing aware (e < w -> the box wraps)."""
    w, e = bbox[0], bbox[2]
    if e < w:  # crosses +/-180
        c = w + ((e + 360.0) - w) / 2.0
        return norm_lon(c)
    return (w + e) / 2.0


def bbox_to_cwh(bbox) -> tuple[float, float, float, float]:
    """(centre_lon, centre_lat, width_lon, height_lat) -- a wrap-free shape that
    makes inset/recentre trivial. width/height are always positive degrees."""
    return (center_lon(bbox), (bbox[1] + bbox[3]) / 2.0,
            lon_span(bbox), bbox[3] - bbox[1])


def cwh_to_bbox(clon: float, clat: float, wlon: float, hlat: float) -> list[float]:
    """Inverse of :func:`bbox_to_cwh`. Re-wraps the lon edges into [-180, 180];
    a box wider than its position naturally comes back with e < w (a crossing)."""
    wlon = min(wlon, 360.0)
    return [round(norm_lon(clon - wlon / 2.0), 3), round(clat - hlat / 2.0, 3),
            round(norm_lon(clon + wlon / 2.0), 3), round(clat + hlat / 2.0, 3)]


def bbox_contains(outer, inner, buffer: float = 0.0) -> bool:
    """Wrap-aware: is ``inner`` fully inside ``outer`` (+/- buffer)? Either box
    may cross +/-180. Mirrors ``satellites._bbox_inside`` so the anchor's
    coverage test matches the render service's geometry exactly: the inner west
    edge's wrapped offset east of the outer west edge must be >= -buffer and the
    inner's east edge must not run past the outer's east edge."""
    if inner[1] < outer[1] - buffer or inner[3] > outer[3] + buffer:
        return False
    o_span = (outer[2] - outer[0]) % 360.0
    i_span = (inner[2] - inner[0]) % 360.0
    i_off = (inner[0] - outer[0]) % 360.0
    if i_off > 180.0:  # inner west edge slightly WEST of outer's -> small negative
        i_off -= 360.0
    return i_off >= -buffer and i_off + i_span <= o_span + buffer


def coverage_poke(outer, inner) -> float:
    """How far ``inner`` extends OUTSIDE ``outer`` on its worst edge, in degrees
    (0 if fully contained). Wrap-aware. This is the largest blank strip a frame
    rendered to ``inner`` would show when the data covers only ``outer`` -- the
    quantity the anchor's coverage guarantee bounds."""
    lat_over = max(0.0, outer[1] - inner[1], inner[3] - outer[3])
    o_span = (outer[2] - outer[0]) % 360.0
    i_span = (inner[2] - inner[0]) % 360.0
    i_off = (inner[0] - outer[0]) % 360.0
    if i_off > 180.0:
        i_off -= 360.0
    lon_over = max(0.0, -i_off, i_off + i_span - o_span)
    return max(lat_over, lon_over)


def inset_bbox(bbox, margin_lon: float, margin_lat: float) -> list[float]:
    """Shrink ``bbox`` inward by the given margins on every edge (wrap-safe).
    The margins are clamped to <= 40 % of the box dimension so the result can
    never invert or collapse."""
    clon, clat, wlon, hlat = bbox_to_cwh(bbox)
    ml = min(max(margin_lon, 0.0), 0.4 * wlon)
    mla = min(max(margin_lat, 0.0), 0.4 * hlat)
    return cwh_to_bbox(clon, clat, wlon - 2.0 * ml, hlat - 2.0 * mla)


def recenter_keep_span(prev_bbox, discovered_bbox) -> list[float]:
    """A box with ``prev_bbox``'s span re-centred on ``discovered_bbox``'s centre.
    Used for a *drift* re-anchor: the view pans to follow the operators without
    changing the frame's span (so its pixel dimensions stay constant across the
    snap)."""
    _, _, pw, ph = bbox_to_cwh(prev_bbox)
    dclon, dclat, _, _ = bbox_to_cwh(discovered_bbox)
    return cwh_to_bbox(dclon, dclat, pw, ph)


# ---------------------------------------------------------------------------
# The anchor state
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Anchor:
    """The stable reference extent the poller renders every frame of a sector to.

    ``bbox`` is the rendered box ([lon_w, lat_s, lon_e, lat_n]); ``source_scan``
    is the discovered scan time it was last (re-)anchored from; ``set_utc`` is
    when it was set; ``reason`` records why (init/reposition/zoom/recenter)."""
    bbox: list[float]
    source_scan: dt.datetime
    set_utc: dt.datetime
    reason: str = "init"

    def to_json(self) -> dict:
        return {"bbox": [round(float(v), 3) for v in self.bbox],
                "source_scan": self.source_scan.strftime("%Y%m%dT%H%M%SZ"),
                "set_utc": self.set_utc.strftime("%Y%m%dT%H%M%SZ"),
                "reason": self.reason}

    @classmethod
    def from_json(cls, d: dict) -> "Anchor | None":
        """Rebuild from a persisted dict; None on any malformed field (a corrupt
        anchor file must never crash the poller -- it just re-anchors fresh)."""
        try:
            bb = [float(v) for v in d["bbox"]]
            # Structural validity: 4 FINITE edges, lat ordered, and a real
            # (non-zero, non-full-globe) wrapped lon span. NaN/Inf pass float()
            # but must be rejected; a zero-width box (w==e) would otherwise sail
            # through lon_span's "% 360 or 360" full-disk convention. (Only an
            # externally-corrupted anchor reaches here; a bad one just re-anchors
            # fresh on the next discovery.)
            raw_lon_span = (bb[2] - bb[0]) % 360.0
            if (len(bb) != 4 or not all(math.isfinite(v) for v in bb)
                    or not (bb[1] < bb[3]) or not (0.0 < raw_lon_span < 360.0)):
                return None

            def _p(s: str) -> dt.datetime:
                return dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=dt.timezone.utc)
            return cls(bbox=bb, source_scan=_p(d["source_scan"]),
                       set_utc=_p(d["set_utc"]), reason=str(d.get("reason", "load")))
        except (KeyError, ValueError, TypeError):
            return None


def center_shift_deg(a, b) -> float:
    """Great-circle-free centre offset between two boxes, in degrees, combining
    the (wrap-aware) lon and the lat components. Used only for logging the size
    of a re-anchor snap."""
    aclon, aclat, _, _ = bbox_to_cwh(a)
    bclon, bclat, _, _ = bbox_to_cwh(b)
    dlon = ((bclon - aclon + 180.0) % 360.0) - 180.0
    return (dlon * dlon + (bclat - aclat) ** 2) ** 0.5


def plan_anchor(prev_bbox, discovered_bbox, *, margin_lon: float, margin_lat: float,
                cover_buffer: float, span_tol: float,
                drift_limit: float = 0.0,
                max_span: float = 360.0) -> tuple[list[float], str]:
    """Decide the anchor box to render to, given the previous anchor (or None)
    and the freshly-discovered live box.

    Returns ``(anchor_bbox, reason)`` where reason is one of:
      * ``"init"``      -- first anchor for this sector (inset of the live box).
      * ``"hold"``      -- keep the previous anchor unchanged (the loop stays
                           pixel-locked); the live box still covers it and its
                           span is unchanged.
      * ``"recenter"``  -- a small *drift* re-anchor: pan to the live centre
                           keeping the previous span (constant frame dimensions).
      * ``"zoom"``      -- the live span changed beyond tolerance (operator zoom)
                           -> re-inset the live box (new dimensions).
      * ``"reposition"``-- a LARGE move (centre shift > ``drift_limit``) or a
                           coverage backstop -> adopt the inset live box.

    ``drift_limit`` separates a small drift (the Himawari Target creeping just
    past its margin -> pan, keep the span so dimensions don't change) from a
    genuine relocation (an operator re-steering M1/M2 far, or the Target jumping
    -> adopt the new box). With ``margin_lon == 0`` (GOES) the limit is 0, so any
    move is a reposition that adopts the live box EXACTLY -- the old behaviour.

    ``max_span`` is a plausibility ceiling (deg): if the discovered box is
    implausibly large (a glitchy scan whose lat/lon bounding box ballooned past a
    real meso sector -- ~5-35 deg), the previous anchor is HELD rather than let
    the anchor balloon to a near-global box (defense-in-depth behind discovery's
    own validation; a meso sector is never tens of degrees wide).

    Invariant: the returned box is ALWAYS fully contained in ``discovered_bbox``
    (no blank edges) -- ``hold`` returns the prev box only after re-verifying it
    is still covered; every other branch is coverage-checked + backstopped."""
    if prev_bbox is None:
        return inset_bbox(discovered_bbox, margin_lon, margin_lat), "init"

    # Plausibility guard: never balloon the anchor to a glitchy near-global box.
    # Keep the last-known-good anchor (it self-heals on the next good scan).
    if (lon_span(discovered_bbox) > max_span
            or (discovered_bbox[3] - discovered_bbox[1]) > max_span):
        return prev_bbox, "hold"

    covered = bbox_contains(discovered_bbox, prev_bbox, cover_buffer)
    d_span = lon_span(discovered_bbox)
    # The live span the previous anchor was inset from (prev span + the margins
    # we trimmed). Comparing the CURRENT live span to this detects an operator
    # zoom independently of drift.
    held_src_span = lon_span(prev_bbox) + 2.0 * margin_lon
    zoom = abs(d_span - held_src_span) > span_tol * max(held_src_span, 1e-6)

    if covered and not zoom:
        return prev_bbox, "hold"

    shift = center_shift_deg(prev_bbox, discovered_bbox)
    if (not zoom) and shift <= drift_limit:
        # Small drift past the margin: pan to the new centre keeping the span,
        # so the frame's pixel dimensions don't change across the snap.
        cand = recenter_keep_span(prev_bbox, discovered_bbox)
        reason = "recenter"
    else:
        # Operator zoom or a genuine large relocation: adopt the (inset) box.
        cand = inset_bbox(discovered_bbox, margin_lon, margin_lat)
        reason = "zoom" if zoom else "reposition"

    # Correctness backstop: the rendered box MUST be backed by data everywhere.
    # If a candidate would poke outside the live box, fall back to a plain inset
    # of the live box (always covered).
    if not bbox_contains(discovered_bbox, cand, cover_buffer):
        cand = inset_bbox(discovered_bbox, margin_lon, margin_lat)
        reason = "reposition"
    return cand, reason
