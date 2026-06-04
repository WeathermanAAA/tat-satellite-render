#!/usr/bin/env python3
"""Storm-following satellite floater poller.

A long-running worker (a SECOND Railway service in the tat-satellite-render
repo, alongside the existing /render web service). It keeps R2 continuously
fresh with storm-centered satellite crops for every active TC/TD/invest, so
the Triple-A-Tropics /satellite/ page can show sub-minute-fresh floaters
without an Actions cron (the 5-min cron floor is too slow).

Pipeline, per cycle:
  1. Refresh the active-storm list from the Pages-origin tracks JSON
     (https://triple-a-tropics.com/{wp,al,ep}_tracks_data.json), filtering on
     the hardened is_active flag (recency + tropical nature + valid wind).
  2. For each active storm x band, on the band's cadence, call /render with an
     12 deg storm-centered bbox + time=latest.
  3. sha256 the returned PNG; if it differs from the last uploaded frame for
     that (storm, band), upload it to R2 and append the per-storm manifest.
     X-Cache:HIT / a source-scan-time header (if present) are used only as a
     cheap short-circuit; the content hash is the source of truth.
  4. Maintain a top-level floaters/manifest.json (active storms) so the
     frontend self-hides when empty.

Design notes:
  * /render is called over Railway PRIVATE networking (RENDER_BASE_URL set to
    http://<render-service>.railway.internal:PORT). That bypasses the public
    10 req/min/IP limiter, so each band can poll at its 60 s target. The
    public-URL cadence formula below is kept as a safety net: set
    RATE_MIN_SPACING_S=7 (~8.5/min) if you ever point RENDER_BASE_URL at the
    public https URL.
  * HOT bands (clean_ir, clean_ir+dvorak_bd) always target 60 s -- the live
    TC-structure diagnostics. COLD bands (WV x2, true_color, shortwave IR)
    stretch as the storm count grows so the rate budget protects the hot
    bands. true_color is daytime-only (skipped when the storm center is in
    night, solar zenith >= NIGHT_ZENITH_DEG).
  * Never crashes: /render errors retry+backoff then skip the unit; R2 errors
    log+skip (manifest only references successfully-uploaded keys); no active
    storms -> idle. On restart it resyncs last-hash state from the R2
    manifests, so it neither duplicates frames nor loses history.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from typing import Iterable, Optional

import boto3
import requests
from botocore.config import Config as BotoConfig

# ---------------------------------------------------------------------------
# Config (env-driven; safe defaults)
# ---------------------------------------------------------------------------

def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# /render base. Prefer Railway private networking, e.g.
#   http://tat-satellite-render.railway.internal:8080
# Falls back to the public URL (then set RATE_MIN_SPACING_S=7).
RENDER_BASE_URL = _env("RENDER_BASE_URL", "https://web-production-b88d.up.railway.app").rstrip("/")
RENDER_URL = RENDER_BASE_URL + "/render"

# R2 / S3
R2_ENDPOINT = _env("R2_ENDPOINT")
R2_BUCKET = _env("R2_BUCKET", "triple-a-tropics-media")
R2_ACCESS_KEY_ID = _env("R2_ACCESS_KEY_ID") or _env("AWS_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _env("R2_SECRET_ACCESS_KEY") or _env("AWS_SECRET_ACCESS_KEY")
R2_PREFIX = _env("R2_PREFIX", "floaters").strip("/")

# Where the hardened active-storm list lives. Use the LIVE R2 feed (poller-
# written, refreshed ~every 2 min) rather than the git-committed Pages origin
# (cron-written, refreshed at best every 6 h - and GitHub drops those scheduled
# runs under load). Reading the stale Pages feed pinned a storm's bbox to an old
# fix so the storm drifted off-frame (Jangmi stuck at its 00Z position while it
# had moved ~7 deg E by 18Z). The live feed carries the same
# {basin}_tracks_data.json schema, so this is a source swap only.
TRACKS_BASE = _env("TRACKS_BASE", "https://cdn.triple-a-tropics.com/feeds").rstrip("/")
TRACKS_BASINS = ("wp", "al", "ep")

# Invests (numbered 90-99, pre-classification disturbances) -- INDEPENDENT
# floater sources, one per basin family. Wholly separate from the
# {basin}_tracks_data.json feeds: nothing here reads, writes, or affects
# ACE / tracks / climatology data, and an invest must never enter season counts.
#
#   * NHC basins (AL/EP/CP): list the public btk directory and parse each
#     invest b-deck's latest fix (the path that discovered 90E).
#   * WP/JTWC: the knackwx ATCF API -- the SAME source the site's tracks
#     pipeline (cron + intensity poller) uses for its invest cards, so the
#     floater discovers a new WP invest as fast as the rest of the site. The
#     JTWC b-deck mirrors (proxy/natyphoon) are NOT usable here: invest
#     numbers recycle within a season and the mirrors keep serving the
#     RETIRED deck (observed 2026-06-04: bwp912026.dat on both mirrors ended
#     at the April 91W's 04-30 fix while the June 91W was live at 19.1N
#     118.8E), so a mirror fetch would float a month-old ghost. knackwx
#     carries the CURRENT fix (atcf_id, analysis_time, position, winds,
#     cyclone_nature) plus ``transitioned_from`` on named entries -- an
#     EXPLICIT invest->named handoff signal ("01E" said it came from "90E").
INVESTS_ENABLED = (_env("INVESTS_ENABLED", "1") or "1").lower() not in ("0", "false", "no")
INVEST_BTK_BASE = _env("INVEST_BTK_BASE", "https://ftp.nhc.noaa.gov/atcf/btk").rstrip("/")
INVEST_INDEX_URL = INVEST_BTK_BASE + "/"
# basin two-letter -> ATCF designation suffix (90 + "E" -> "90E").
INVEST_BASIN_LETTER = {"al": "L", "ep": "E", "cp": "C"}
# WP invests (knackwx). WP_INVESTS_ENABLED is the per-source kill switch;
# INVESTS_ENABLED still gates all invest sources globally.
KNACKWX_ATCF_URL = _env("KNACKWX_ATCF_URL", "https://api.knackwx.com/atcf/v2")
WP_INVESTS_ENABLED = (_env("WP_INVESTS_ENABLED", "1") or "1").lower() not in ("0", "false", "no")

# NHC CurrentStorms.json: the AUTHORITATIVE list of currently-NAMED NHC TCs
# (AL/EP/CP). Re-read each cycle so a just-DESIGNATED system (an invest promoted
# to a TD/TS, e.g. 90E -> One-E) is floated as the NAMED storm with its real
# status, and its retired invest floater is dropped (never both at once).
CURRENT_STORMS_URL = _env("CURRENT_STORMS_URL",
                          "https://www.nhc.noaa.gov/CurrentStorms.json")
NHC_BASIN_PREFIXES = {"al": "AL", "ep": "EP", "cp": "CP"}
# An invest counts as DESIGNATED (-> dropped) when a named NHC storm sits in the
# same basin within this many degrees of the invest's latest fix (it became that
# TC). Generous enough to span a few 6-hourly fixes of drift between the two.
INVEST_DESIGNATION_DEG = float(_env("INVEST_DESIGNATION_DEG", "5.0"))

# Geometry + cadence
BBOX_DEG = float(_env("BBOX_DEG", "12"))           # square floater width (deg)
CADENCE_TARGET_S = float(_env("CADENCE_TARGET_S", "60"))   # hot-band target
TRACKS_REFRESH_S = float(_env("TRACKS_REFRESH_S", "600"))  # storms update every 6 h
RATE_MIN_SPACING_S = float(_env("RATE_MIN_SPACING_S", "1.0"))  # min gap between /render calls
ACTIVE_WINDOW_HOURS = float(_env("ACTIVE_WINDOW_HOURS", "60"))
NIGHT_ZENITH_DEG = float(_env("NIGHT_ZENITH_DEG", "85"))   # >= this => skip daytime-only bands
EXTRAPOLATE_MAX_H = float(_env("EXTRAPOLATE_MAX_H", "6"))  # cap motion extrapolation

# Retention (R2) -- native recent + thinned history.
RECENT_WINDOW_H = float(_env("RECENT_WINDOW_H", "6"))      # keep native cadence within this
HISTORY_WINDOW_H = float(_env("HISTORY_WINDOW_H", "24"))   # thin to THIN_SPACING beyond recent
THIN_SPACING_S = float(_env("THIN_SPACING_S", "300"))      # 5 min
DEACTIVATE_GRACE_H = float(_env("DEACTIVATE_GRACE_H", "24"))  # keep frames after storm ends

# HTTP timeouts / retries
RENDER_TIMEOUT_S = float(_env("RENDER_TIMEOUT_S", "45"))
RENDER_MAX_RETRIES = int(_env("RENDER_MAX_RETRIES", "3"))
CIRCUIT_TRIP_FAILS = int(_env("CIRCUIT_TRIP_FAILS", "8"))   # consecutive fails -> cool down
CIRCUIT_COOLDOWN_S = float(_env("CIRCUIT_COOLDOWN_S", "60"))

# Tracks-JSON fetch (Pages origin can be slow under load -- the read timeout
# must tolerate a sluggish response or the basin a storm lives in silently
# stops refreshing). Tuple timeout: fast connect, generous read. Retried with
# exponential backoff so a single slow response is not treated as a failure.
TRACKS_CONNECT_TIMEOUT_S = float(_env("TRACKS_CONNECT_TIMEOUT_S", "10"))
TRACKS_READ_TIMEOUT_S = float(_env("TRACKS_READ_TIMEOUT_S", "45"))
TRACKS_MAX_RETRIES = int(_env("TRACKS_MAX_RETRIES", "3"))   # retries AFTER the first attempt

CACHE_FRAME = "public, max-age=31536000, immutable"
CACHE_MANIFEST = "max-age=30"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("floater")


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Band:
    key: str            # path/manifest key + URL ?band=
    channel: str        # /render channel
    enhancement: str    # /render enhancement
    label: str          # UI label
    hot: bool           # hot=always 60 s; cold=stretches with storm count
    daytime_only: bool  # skip when storm center is in night


BANDS: tuple[Band, ...] = (
    Band("ir",        "clean_ir",     "rainbow_ir", "Clean IR",         hot=True,  daytime_only=False),
    Band("irbd",      "clean_ir",     "dvorak",    "IR (Dvorak)",       hot=True,  daytime_only=False),
    Band("wv_up",     "wv_upper",     "wv_tat",    "Upper WV",          hot=False, daytime_only=False),
    Band("wv_low",    "wv_lower",     "wv_tat",    "Lower WV",          hot=False, daytime_only=False),
    # Truecolor now runs 24/7. The render-side GeoColor-lite night blend
    # (truecolor.night_blend in tat-satellite-render) fades the true-color
    # composite to grayscale clean-IR across the terminator, so the band
    # shows photographic color by day and IR cloud structure at night --
    # no more day-only dropout. Render-side guards catch any transient
    # degenerate RGB so we don't ship a black frame mid-day.
    Band("truecolor", "true_color",   "tat_neon",  "Visible (true color)", hot=False, daytime_only=False),
    Band("swir",      "shortwave_ir", "grayscale", "Shortwave IR",      hot=False, daytime_only=False),
)
BANDS_BY_KEY = {b.key: b for b in BANDS}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> dt.datetime | None:
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d


def storm_slug(sid: str, basin: str) -> str:
    """sid like 'JTWC_WP062026' -> 'wp06'; fallback to a sanitized sid."""
    m = re.search(r"([A-Z]{2})(\d{2})\d{4}$", sid or "")
    if m:
        return (m.group(1) + m.group(2)).lower()
    b = (basin or "").lower()[:2]
    digits = re.sub(r"\D", "", sid or "")
    if b and digits:
        return f"{b}{digits[:2]}"
    return re.sub(r"[^a-z0-9]+", "-", (sid or "storm").lower()).strip("-") or "storm"


def solar_zenith_deg(lat: float, lon: float, when: dt.datetime) -> float:
    """Approximate solar zenith angle (deg). NOAA low-precision algorithm;
    good to ~1 deg, plenty for a day/night gate."""
    when = when.astimezone(dt.timezone.utc)
    # Fractional year (radians)
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    gamma = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    # Equation of time (min) and solar declination (rad)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    time_offset = eqtime + 4 * lon            # lon east-positive
    tst = hour * 60 + time_offset             # true solar time (min)
    ha = math.radians(tst / 4 - 180)          # hour angle (rad)
    latr = math.radians(lat)
    cos_zen = (math.sin(latr) * math.sin(decl)
               + math.cos(latr) * math.cos(decl) * math.cos(ha))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    return math.degrees(math.acos(cos_zen))


def norm_lon(lon: float) -> float:
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


# ---------------------------------------------------------------------------
# Active-storm discovery
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Storm:
    sid: str
    slug: str
    name: str
    basin: str
    lat: float
    lon: float
    category: str
    intensity_kt: float | None    # PEAK wind so far (kept for back-compat / sort)
    last_fix: str                 # ISO of latest fix
    # Current-fix snapshot used to color-code the title badge in /render.
    # ``current_wind_kt`` / ``current_pressure_mb`` come from the LATEST fix
    # point (peak fields would mislabel a weakening storm). ``nature``
    # (TS/TD/HU/EX/...) lets the render side pick the right color family
    # for non-tropical statuses.
    current_wind_kt: Optional[float] = None
    current_pressure_mb: Optional[float] = None
    nature: Optional[str] = None


def _extrapolate(points: list[dict], now: dt.datetime) -> tuple[float, float]:
    """Center the floater on the storm's position *now*, linearly extrapolated
    from its last two 6-hourly fixes (capped at EXTRAPOLATE_MAX_H so a stale
    JSON can't fling the bbox off the storm)."""
    last = points[-1]
    lat, lon = float(last["lat"]), float(last["lon"])
    lt = parse_iso(last["t"])
    if len(points) < 2 or lt is None:
        return lat, lon
    prev = points[-2]
    pt = parse_iso(prev["t"])
    if pt is None or pt >= lt:
        return lat, lon
    dt_h = (lt - pt).total_seconds() / 3600.0
    if dt_h <= 0:
        return lat, lon
    dlat = (lat - float(prev["lat"])) / dt_h
    dlon = (norm_lon(lon - float(prev["lon"]))) / dt_h
    ahead_h = max(0.0, min(EXTRAPOLATE_MAX_H, (now - lt).total_seconds() / 3600.0))
    return lat + dlat * ahead_h, norm_lon(lon + dlon * ahead_h)


def _fetch_tracks_json(session: requests.Session, url: str, basin: str) -> Optional[dict]:
    """Fetch one basin's tracks JSON, surviving a slow origin.

    Uses a tuple (connect, read) timeout so a sluggish Pages response gets up
    to TRACKS_READ_TIMEOUT_S to deliver, and retries TRACKS_MAX_RETRIES times
    with 2s/4s/8s exponential backoff. Returns the parsed JSON, or ``None``
    only after every attempt is exhausted. The shared ``session`` keeps a
    keep-alive pool so the three basin fetches reuse one connection.
    """
    timeout = (TRACKS_CONNECT_TIMEOUT_S, TRACKS_READ_TIMEOUT_S)
    total = TRACKS_MAX_RETRIES + 1
    last_exc: Exception | None = None
    for attempt in range(1, total + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - never crash on a bad fetch
            last_exc = e
            if attempt < total:
                backoff = 2 ** attempt  # 2s, 4s, 8s
                log.warning(
                    "tracks fetch attempt %d/%d failed (%s): %s -- retrying in %.0fs",
                    attempt, total, basin, e, backoff,
                )
                time.sleep(backoff)
    log.warning("tracks fetch failed for %s after %d attempts: %s",
                basin, total, last_exc)
    return None


def fetch_active_storms(session: requests.Session) -> dict[str, Optional[list[Storm]]]:
    """Return active storms keyed by basin.

    Each basin maps to its (possibly empty) active-storm list, or to ``None``
    if that basin's tracks fetch failed after all retries. Returning per-basin
    results lets the caller refresh the basins that succeeded and preserve only
    the failing basin's last-known-good storms -- one slow basin no longer
    freezes the whole top manifest. An empty list for a basin means it fetched
    cleanly and reported no active storms (genuine quiescence).
    """
    now = utcnow()
    cutoff = now - dt.timedelta(hours=ACTIVE_WINDOW_HOURS)
    results: dict[str, Optional[list[Storm]]] = {}
    for basin in TRACKS_BASINS:
        url = f"{TRACKS_BASE}/{basin}_tracks_data.json"
        data = _fetch_tracks_json(session, url, basin)
        if data is None:
            results[basin] = None
            continue
        storms: list[Storm] = []
        for s in data.get("storms", []):
            # Trust the hardened is_active baked into the JSON, but re-check
            # recency here so a stale JSON can't keep a storm "active" forever.
            pts = s.get("points") or []
            if not s.get("is_active") or not pts:
                continue
            lt = parse_iso(pts[-1].get("t", ""))
            if lt is None or lt < cutoff:
                continue
            # Use the LATEST FIX position directly (no extrapolation). The
            # bbox is computed from storm.lat/lon downstream, so this pins
            # the floater crop to the most recent JTWC fix point -- new
            # fixes arrive every ~6 h, so the bbox shifts only when an
            # actual fix lands rather than drifting continuously between
            # fixes (which created visible per-frame jitter in the loop).
            last_pt = pts[-1]
            lat = float(last_pt.get("lat", 0.0))
            lon = float(last_pt.get("lon", 0.0))
            sid = s.get("sid") or ""
            # Invest-range entries (90-99) ride the dedicated invest sources
            # (NHC b-decks / knackwx) with INVEST labeling -- never the named
            # path, even if the feed ever marks one active. Belt-and-suspenders:
            # today the feed carries invests with is_active=false.
            if re.search(r"[A-Z]{2}9\d\d{4}$", sid):
                continue
            # Latest-fix wind/pressure/nature for the color-coded title badge.
            # ``peak_wind_kt`` (storm-level) stays as ``intensity_kt`` for
            # legacy callers, but the badge uses the CURRENT fix values so a
            # weakening storm shows its current category, not its peak.
            cur_wind = last_pt.get("wind_kt")
            cur_pres = last_pt.get("pressure_mb")
            nature = last_pt.get("nature")
            storms.append(Storm(
                sid=sid,
                slug=storm_slug(sid, basin),
                name=s.get("name") or "UNNAMED",
                basin=basin.upper(),
                lat=round(lat, 2),
                lon=round(norm_lon(lon), 2),
                category=s.get("current_category") or "TD",
                intensity_kt=s.get("peak_wind_kt"),
                last_fix=last_pt.get("t", ""),
                current_wind_kt=float(cur_wind) if cur_wind is not None else None,
                current_pressure_mb=float(cur_pres) if cur_pres is not None else None,
                nature=nature if isinstance(nature, str) else None,
            ))
        results[basin] = storms
    return results


# ---------------------------------------------------------------------------
# Invest discovery (NHC ATCF b-decks) -- an INDEPENDENT source
# ---------------------------------------------------------------------------
# Invests are floater-only: nothing here reads or writes the tracks / ACE /
# climatology data, and an invest never enters season counts. The flow mirrors
# the named-storm path (list -> parse latest fix -> Storm); the invest Storms
# are then merged into the floater manifest alongside named storms.

def _fetch_text(session: requests.Session, url: str, label: str) -> Optional[str]:
    """GET a text resource, surviving a slow/transient origin.

    Same hardening as the tracks fetch (tuple connect/read timeout + 2s/4s/8s
    exponential-backoff retries). Returns the body text, or ``None`` once every
    attempt is exhausted. Never raises -- a bad fetch is a None, not a crash.
    """
    timeout = (TRACKS_CONNECT_TIMEOUT_S, TRACKS_READ_TIMEOUT_S)
    total = TRACKS_MAX_RETRIES + 1
    last_exc: Exception | None = None
    for attempt in range(1, total + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001 - never crash on a bad fetch
            last_exc = e
            if attempt < total:
                time.sleep(2 ** attempt)  # 2s, 4s, 8s
    log.warning("%s fetch failed after %d attempts: %s", label, total, last_exc)
    return None


def _bdeck_latlon(tok: str) -> Optional[float]:
    """ATCF tenths-of-degree + hemisphere -> signed degrees.
    '94N' -> 9.4, '1257W' -> -125.7. Returns None on a malformed token."""
    tok = (tok or "").strip()
    if len(tok) < 2:
        return None
    hemi = tok[-1].upper()
    try:
        val = int(tok[:-1]) / 10.0
    except ValueError:
        return None
    if hemi in ("S", "W"):
        return -val
    if hemi in ("N", "E"):
        return val
    return None


def _bdeck_time(tok: str) -> Optional[dt.datetime]:
    """ATCF YYYYMMDDHH (UTC) -> aware datetime. None on a malformed token."""
    tok = (tok or "").strip()
    if len(tok) < 10:
        return None
    try:
        return dt.datetime.strptime(tok[:10], "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _parse_invest_bdeck(text: str, basin2: str, num: str, year: str,
                        cutoff: dt.datetime) -> Optional[Storm]:
    """Parse one invest b-deck into a Storm from its LATEST best-track fix.

    Returns None if there is no usable fix, or if the latest fix is older than
    ``cutoff`` (a dissipated/named invest whose b-deck lingers in the directory).
    The peak wind across all fixes is kept as ``intensity_kt`` (sort / legacy)
    while the badge uses the latest fix's wind/pressure. Never raises.
    """
    best: list[str] | None = None
    best_t: dt.datetime | None = None
    peak_wind: float | None = None
    for line in text.splitlines():
        f = [c.strip() for c in line.split(",")]
        if len(f) < 11:
            continue
        t = _bdeck_time(f[2])
        if t is None:
            continue
        try:
            w = float(f[8]) if f[8] else None
        except ValueError:
            w = None
        if w is not None and w > 0:
            peak_wind = w if peak_wind is None else max(peak_wind, w)
        if best_t is None or t > best_t:
            best, best_t = f, t
    if best is None or best_t is None or best_t < cutoff:
        return None
    lat = _bdeck_latlon(best[6])
    lon = _bdeck_latlon(best[7])
    if lat is None or lon is None:
        return None
    try:
        vmax = float(best[8]) if best[8] else None
    except ValueError:
        vmax = None
    if vmax is not None and vmax <= 0:
        vmax = None
    try:
        mslp = float(best[9]) if best[9] else None
    except ValueError:
        mslp = None
    if mslp is not None and not (850 <= mslp <= 1050):
        mslp = None  # 0 / out-of-range placeholder -> omit from the badge
    status = (best[10] or "DB").strip().upper() or "DB"
    letter = INVEST_BASIN_LETTER.get(basin2, basin2.upper()[:1])
    designation = f"{num}{letter}"            # "90E"
    return Storm(
        sid=f"{basin2.upper()}{num}{year}",   # "EP902026" (stable id)
        slug=f"{basin2}{num}",                # "ep90" (never collides with named 01-49)
        name=f"INVEST {designation}",         # "INVEST 90E"
        basin=basin2.upper(),                 # "EP"
        lat=round(lat, 2),
        lon=round(norm_lon(lon), 2),
        # The frontend shows ``category`` as plain text and defaults a falsy
        # value to "TD"; "INVEST" keeps it honest (pre-classification, no
        # Saffir-Simpson category).
        category="INVEST",
        intensity_kt=peak_wind,
        last_fix=iso_z(best_t),
        current_wind_kt=vmax,
        current_pressure_mb=mslp,
        # Real ATCF status (DB/LO/WV/...): render.py maps these to a neutral GRAY
        # badge with no Saffir-Simpson category -- exactly the invest treatment.
        nature=status,
    )


def fetch_active_invests(session: requests.Session) -> Optional[list[Storm]]:
    """Active NHC invests (90-99) parsed from the ATCF best-track b-decks.

    Lists the public btk directory, then fetches + parses each invest b-deck,
    keeping those whose latest fix is within ACTIVE_WINDOW_HOURS. Returns:
      * ``None`` only if the directory listing itself fails after all retries
        (the caller then preserves the last-known-good invests);
      * an empty list if invests are disabled, or the listing succeeded but no
        invest is currently active;
      * a list of invest Storms otherwise.
    A single invest's fetch/parse failure is swallowed (that invest is skipped)
    so one bad b-deck never sinks the rest. NEVER raises.
    """
    if not INVESTS_ENABLED:
        return []
    index = _fetch_text(session, INVEST_INDEX_URL, "invest-index")
    if index is None:
        return None
    # Directory hrefs look like ``bep902026.dat`` -> (basin2, num, year). Each
    # filename appears twice (href + link text); dedup keeps one per invest.
    seen: set[tuple[str, str, str]] = set()
    ids: list[tuple[str, str, str]] = []
    for basin2, num, year in re.findall(r"b(al|ep|cp)(9\d)(20\d{2})\.dat", index, flags=re.I):
        k = (basin2.lower(), num, year)
        if k not in seen:
            seen.add(k)
            ids.append(k)
    cutoff = utcnow() - dt.timedelta(hours=ACTIVE_WINDOW_HOURS)
    out: list[Storm] = []
    for basin2, num, year in ids:
        url = f"{INVEST_BTK_BASE}/b{basin2}{num}{year}.dat"
        text = _fetch_text(session, url, f"invest-{basin2}{num}")
        if text is None:
            continue
        try:
            s = _parse_invest_bdeck(text, basin2, num, year, cutoff)
        except Exception as e:  # noqa: BLE001 - one bad deck never sinks the source
            log.warning("invest parse failed for b%s%s%s: %s", basin2, num, year, e)
            continue
        if s is not None:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# WP invest discovery (knackwx ATCF API) -- an INDEPENDENT source
# ---------------------------------------------------------------------------
# Mirrors the NHC invest path's contract exactly (None = fetch failure -> the
# caller preserves last-known-good; [] = genuine quiescence; one bad entry is
# skipped, never sinks the source) so the per-source isolation guarantees stay
# uniform across basins.

# ATCF statuses render.py's _ss_category gives the neutral GRAY (no
# Saffir-Simpson) badge. knackwx's cyclone_nature is an independent analysis
# classification that CAN read TD/TS/TY on a system still numbered 90-99 (a
# developing pre-designation disturbance -- the sibling intensity_poller parser
# even falls back to TS for an invest with wind); the invest convention is NO
# Saffir-Simpson pill, so anything outside the neutral set is coerced to DB.
# (The NHC b-deck path needs no coercion: ATCF decks structurally tag 90-99
# systems DB/LO/WV.)
INVEST_NEUTRAL_NATURES = frozenset({"DB", "LO", "WV", "SD", "SS", "EX", "PT"})


def _parse_knackwx_invest(it: dict, cutoff: dt.datetime) -> Optional[Storm]:
    """One knackwx ATCF entry -> a WP invest Storm, or None.

    None when the entry is not a WP invest (atcf_id not 9xW), is missing or
    garbling a required field, or its latest fix is older than ``cutoff`` (a
    dissipated invest knackwx hasn't dropped yet -- the freshness lifecycle,
    same ACTIVE_WINDOW_HOURS the NHC invest path applies). Labeling follows
    the established invest convention: name "INVEST 91W", category "INVEST"
    (frontend plain-text pill, no Saffir-Simpson category), nature = the ATCF
    status coerced into INVEST_NEUTRAL_NATURES (render.py gives those the
    neutral GRAY badge -- never a Saffir-Simpson pill on an invest).
    """
    m = re.fullmatch(r"(9\d)W", (str(it.get("atcf_id") or "")).strip().upper())
    if m is None:
        return None
    num = m.group(1)
    t = parse_iso(str(it.get("analysis_time") or ""))
    if t is None or t < cutoff:
        return None
    try:
        lat = float(it.get("latitude"))
        lon = float(it.get("longitude"))
    except (TypeError, ValueError):
        return None

    def _pos(v) -> Optional[float]:
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    vmax = _pos(it.get("winds"))
    mslp = _pos(it.get("pressure"))
    if mslp is not None and not (850 <= mslp <= 1050):
        mslp = None  # 0 / out-of-range placeholder -> omit from the badge
    status = (str(it.get("cyclone_nature") or "")).strip().upper() or "DB"
    if status not in INVEST_NEUTRAL_NATURES:
        status = "DB"  # never a Saffir-Simpson-colored pill on an invest
    yid = (str(it.get("long_atcf_id") or "")).strip().lower()
    year = yid[-4:] if re.fullmatch(r"wp9\d20\d\d", yid) else str(t.year)
    return Storm(
        sid=f"WP{num}{year}",                 # "WP912026" (stable id)
        slug=f"wp{num}",                      # "wp91" (never collides with named 01-49)
        name=f"INVEST {num}W",                # "INVEST 91W"
        basin="WP",
        lat=round(lat, 2),
        lon=round(norm_lon(lon), 2),
        category="INVEST",
        intensity_kt=vmax,                    # knackwx carries the latest fix only
        last_fix=iso_z(t),
        current_wind_kt=vmax,
        current_pressure_mb=mslp,
        nature=status,
    )


def fetch_wp_invests(session: requests.Session) -> Optional[tuple[list[Storm], set[str]]]:
    """Active WP invests (90-99) + the retired-designation set, from knackwx.

    Returns ``None`` when the API is unreachable or unparseable after all
    retries (the caller preserves last-known-good), else ``(invests, retired)``:
      * ``invests`` -- WP invest Storms with a fix within ACTIVE_WINDOW_HOURS
        (empty = invests disabled, or genuine quiescence);
      * ``retired`` -- invest designations that NAMED entries (numbers 01-49,
        ANY basin) report in ``transitioned_from`` ("90E" on 01E): the explicit
        invest->named handoff signal, consumed by refresh_storms to drop a
        just-designated invest's floater the cycle the link appears.
    A single malformed entry is skipped (logged), never sinks the source.
    NEVER raises.
    """
    if not (INVESTS_ENABLED and WP_INVESTS_ENABLED):
        return [], set()
    text = _fetch_text(session, KNACKWX_ATCF_URL, "knackwx-atcf")
    if text is None:
        return None
    try:
        data = json.loads(text)
    except ValueError as e:
        log.warning("knackwx: malformed JSON (%s) -- treating as fetch failure", e)
        return None
    if not isinstance(data, list):
        log.warning("knackwx: unexpected payload type %s -- treating as fetch failure",
                    type(data).__name__)
        return None
    cutoff = utcnow() - dt.timedelta(hours=ACTIVE_WINDOW_HOURS)
    invests: list[Storm] = []
    retired: set[str] = set()
    for it in data:
        if not isinstance(it, dict):
            continue
        try:
            aid = (str(it.get("atcf_id") or "")).strip().upper()
            tf = (str(it.get("transitioned_from") or "")).strip().upper()
            m = re.fullmatch(r"(\d\d)[A-Z]", aid)
            if m is not None and 1 <= int(m.group(1)) <= 49:
                if tf:
                    retired.add(tf)           # named entry born from an invest
                continue
            s = _parse_knackwx_invest(it, cutoff)
        except Exception as e:  # noqa: BLE001 - one bad entry never sinks the source
            log.warning("knackwx: bad entry skipped (%s): %s", it.get("atcf_id"), e)
            continue
        if s is not None:
            invests.append(s)
    return invests, retired


def _deg_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in degrees between two lat/lon points, with longitude wrap.
    Good enough for the invest-designation co-location test (a few degrees)."""
    dlon = ((lon1 - lon2 + 180.0) % 360.0) - 180.0
    return math.hypot(lat1 - lat2, dlon)


def fetch_current_named(session: requests.Session) -> Optional[dict[str, Storm]]:
    """Named NHC TCs from NHC CurrentStorms.json (AL/EP/CP), keyed by slug.

    The authoritative current-named list: a just-DESIGNATED system appears here
    as its TD/TS the cycle it is named, before the derived {basin}_tracks_data
    feed catches up. Used to (a) surface the named storm immediately and (b) drop
    the retired invest it was promoted from. Returns None on fetch failure (the
    caller preserves last-known-good); an empty dict when the fetch succeeds with
    no active named NHC storm. Never raises.
    """
    data = _fetch_tracks_json(session, CURRENT_STORMS_URL, "nhc-current")
    if data is None:
        return None

    def _pos_num(v):
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    out: dict[str, Storm] = {}
    for s in (data.get("activeStorms") or []):
        sid = (s.get("id") or "").strip()                  # "ep012026"
        basin2 = sid[:2].lower()
        if basin2 not in NHC_BASIN_PREFIXES:
            continue
        try:
            lat = float(s.get("latitudeNumeric"))
            lon = norm_lon(float(s.get("longitudeNumeric")))
        except (TypeError, ValueError):
            continue
        cls = (s.get("classification") or "").strip().upper() or "TD"   # TD/TS/HU...
        slug = storm_slug(sid.upper(), basin2)             # "ep01"
        out[slug] = Storm(
            sid=sid, slug=slug,
            name=(s.get("name") or slug).strip().upper(),  # "ONE-E"
            basin=NHC_BASIN_PREFIXES[basin2],
            lat=round(lat, 2), lon=round(lon, 2),
            category=cls, intensity_kt=_pos_num(s.get("intensity")),
            last_fix=(s.get("lastUpdate") or ""),
            current_wind_kt=_pos_num(s.get("intensity")),
            current_pressure_mb=_pos_num(s.get("pressure")),
            nature=cls,
        )
    return out


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------

class R2:
    def __init__(self) -> None:
        self.s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )

    def get_json(self, key: str) -> dict | None:
        try:
            obj = self.s3.get_object(Bucket=R2_BUCKET, Key=key)
            return json.loads(obj["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("R2 get_json %s failed: %s", key, e)
            return None

    def put_bytes(self, key: str, data: bytes, content_type: str, cache: str) -> bool:
        try:
            self.s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data,
                               ContentType=content_type, CacheControl=cache)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("R2 put %s failed: %s", key, e)
            return False

    def put_json(self, key: str, obj: dict, cache: str) -> bool:
        body = json.dumps(obj, separators=(",", ":")).encode()
        return self.put_bytes(key, body, "application/json", cache)

    def delete(self, keys: Iterable[str]) -> None:
        keys = [k for k in keys if k]
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            try:
                self.s3.delete_objects(
                    Bucket=R2_BUCKET,
                    Delete={"Objects": [{"Key": k} for k in batch]},
                )
            except Exception as e:  # noqa: BLE001
                log.warning("R2 delete batch failed: %s", e)


# ---------------------------------------------------------------------------
# Rate limiter (global min spacing between /render calls)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, min_spacing_s: float) -> None:
        self.min_spacing = min_spacing_s
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self.min_spacing - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Render call
# ---------------------------------------------------------------------------

class RenderError(Exception):
    pass


class RenderSkip(Exception):
    """422 / coverage / night -- expected, not an error."""


def call_render(session: requests.Session, bbox: list[float], channel: str,
                enhancement: str, storm: Optional[dict] = None) -> tuple[bytes, dict]:
    body: dict = {"bbox": bbox, "time": "latest", "channel": channel, "enhancement": enhancement}
    # When supplied, /render burns a color-coded intensity badge into the
    # rendered PNG's title strip (left side). Only sent from the poller path;
    # legacy draw-a-box /satellite/ UI omits it and gets the plain title.
    if storm is not None:
        body["storm"] = storm
    last_exc: Exception | None = None
    for attempt in range(RENDER_MAX_RETRIES):
        try:
            r = session.post(RENDER_URL, json=body, timeout=RENDER_TIMEOUT_S)
            if r.status_code == 422:
                raise RenderSkip(r.text[:200])
            if r.status_code == 429:
                # Public limiter tripped (shouldn't happen on private net) --
                # back off hard and retry.
                time.sleep(6 + attempt * 4)
                last_exc = RenderError("429 rate limited")
                continue
            r.raise_for_status()
            return r.content, dict(r.headers)
        except RenderSkip:
            raise
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    raise RenderError(str(last_exc))


# ---------------------------------------------------------------------------
# Manifest / frame state per (storm, band)
# ---------------------------------------------------------------------------

def frame_key(slug: str, band_key: str, ts: dt.datetime) -> str:
    return f"{R2_PREFIX}/{slug}/{band_key}/{ts.strftime('%Y%m%dT%H%MZ')}.png"


def storm_manifest_key(slug: str) -> str:
    return f"{R2_PREFIX}/{slug}/manifest.json"


def top_manifest_key() -> str:
    return f"{R2_PREFIX}/manifest.json"


def prune_frames(frames: list[dict], now: dt.datetime) -> tuple[list[dict], list[str]]:
    """Keep native cadence within RECENT_WINDOW_H; thin to THIN_SPACING_S out
    to HISTORY_WINDOW_H; drop older. Returns (kept, deleted_keys)."""
    recent_cut = now - dt.timedelta(hours=RECENT_WINDOW_H)
    history_cut = now - dt.timedelta(hours=HISTORY_WINDOW_H)
    kept: list[dict] = []
    deleted: list[str] = []
    last_kept_t: dt.datetime | None = None
    # Oldest -> newest so thinning spacing is stable.
    for f in sorted(frames, key=lambda x: x["t"]):
        t = parse_iso(f["t"])
        if t is None or t < history_cut:
            deleted.append(f["key"])
            continue
        if t >= recent_cut:
            kept.append(f)
            last_kept_t = t
            continue
        # history zone: thin
        if last_kept_t is None or (t - last_kept_t).total_seconds() >= THIN_SPACING_S:
            kept.append(f)
            last_kept_t = t
        else:
            deleted.append(f["key"])
    return kept, deleted


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Unit:
    storm: Storm
    band: Band
    next_due: float = 0.0          # monotonic time
    last_hash: str | None = None   # sha256 of last uploaded frame


class Poller:
    def __init__(self) -> None:
        self.r2 = R2()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "tat-floater-poller/1.0"
        self.limiter = RateLimiter(RATE_MIN_SPACING_S)
        self.units: dict[tuple[str, str], Unit] = {}
        self.storms: dict[str, Storm] = {}     # slug -> storm (combined active set)
        self.named: dict[str, Storm] = {}      # slug -> named storm (last-known-good)
        self.invests: dict[str, Storm] = {}    # slug -> NHC invest (last-known-good)
        self.wp_invests: dict[str, Storm] = {}  # slug -> WP/knackwx invest (last-known-good)
        # designation -> when we FIRST saw its transitioned_from link (LKG).
        # The timestamp makes the explicit handoff recycle-safe; see
        # refresh_storms.
        self.retired_invests: dict[str, dt.datetime] = {}
        self.current_named: dict[str, Storm] = {}  # slug -> NHC named (last-known-good)
        self._last_tracks_refresh = 0.0
        self._consec_render_fail = 0
        self._circuit_open_until = 0.0

    # ---- storm-set management ------------------------------------------

    def refresh_storms(self) -> None:
        # --- Named storms: per-basin tracks feeds, last-known-good on failure.
        # A refreshed basin replaces its storms outright; a failed basin keeps
        # the storms it had last cycle (matched on the uppercase Storm.basin).
        per_basin = fetch_active_storms(self.session)
        succeeded = [b for b, v in per_basin.items() if v is not None]
        failed = [b for b, v in per_basin.items() if v is None]
        if succeeded:
            failed_upper = {b.upper() for b in failed}
            named: dict[str, Storm] = {
                slug: s for slug, s in self.named.items() if s.basin in failed_upper
            }
            for basin in succeeded:
                for s in per_basin[basin] or []:
                    named[s.slug] = s
            self.named = named
            if failed:
                log.warning("tracks: refreshed %s; preserved last-known-good for %s",
                            ",".join(succeeded), ",".join(failed))
            else:
                log.info("tracks: refreshed all basins (%s)", ",".join(succeeded))
        else:
            # Every basin failed -> keep the named set we already have. Invests
            # below are an independent source and still refresh; the combined
            # manifest is rewritten so generated_utc advances and the widget
            # stays visible.
            log.warning(
                "tracks fetch failed for ALL basins (%s) -- preserving last-known-good "
                "named storms", ",".join(failed),
            )

        # --- NHC invests: independent NHC b-deck source, isolated from the named
        # feeds. A failed invest fetch preserves the last-known-good invests and
        # never touches the named storms (and vice versa) -- per-source guarding.
        invests = fetch_active_invests(self.session)
        if invests is None:
            log.warning("invests: fetch failed -- preserving last-known-good (%d)",
                        len(self.invests))
        else:
            self.invests = {s.slug: s for s in invests}
            if invests:
                log.info("invests: %d active (%s)", len(invests),
                         ", ".join(f"{s.name}/{s.slug}" for s in invests))

        # --- WP invests: knackwx ATCF API, the same source the site's tracks
        # pipeline uses for its invest cards -- per-source isolated exactly like
        # the NHC invests above. A knackwx failure preserves the last-known-good
        # WP invests AND the last-known retired-designation set, and never
        # touches the named storms or the NHC invests. Slugs (wp90-99) never
        # collide with NHC invests (al/ep/cp 90-99) or named storms (01-49).
        wp = fetch_wp_invests(self.session)
        if wp is None:
            log.warning("wp-invests: fetch failed -- preserving last-known-good (%d)",
                        len(self.wp_invests))
        else:
            wp_invests, retired = wp
            # Stamp each designation with when we FIRST saw its
            # transitioned_from link. The explicit handoff below only retires
            # an invest whose latest fix predates that moment, which makes it
            # recycle-safe: transitioned_from persists on a named entry for its
            # whole life (AMANDA carries "90E" weeks after designation), but a
            # RECYCLED 90E gets fixes NEWER than the first-seen stamp and
            # floats, while the old promoted invest's deck stopped at
            # designation and stays retired. A designation drops out of the
            # dict when no named entry reports it anymore.
            now = utcnow()
            self.retired_invests = {d: self.retired_invests.get(d, now)
                                    for d in retired}
            self.wp_invests = {s.slug: s for s in wp_invests}
            if wp_invests:
                log.info("wp-invests: %d active (%s)", len(wp_invests),
                         ", ".join(f"{s.name}/{s.slug}" for s in wp_invests))

        # --- NHC CurrentStorms.json: AUTHORITATIVE current-named list (AL/EP/CP),
        # re-read each cycle. It surfaces a just-DESIGNATED system as its named
        # TD/TS immediately and tells us which invests were promoted, so a retired
        # invest is REPLACED by its named successor (never shown alongside it).
        # WP/JTWC is untouched (NHC-only source); a fetch failure preserves the
        # last-known-good set and never crashes the cycle.
        current = fetch_current_named(self.session)
        if current is None:
            log.warning("CurrentStorms fetch failed -- preserving last-known-good (%d)",
                        len(self.current_named))
        else:
            self.current_named = current
            if current:
                log.info("CurrentStorms: %d named NHC TC(s) (%s)", len(current),
                         ", ".join(f"{s.name}/{s.slug}" for s in current.values()))

        # Named = tracks-feed named, with CurrentStorms overriding/adding the
        # authoritative NHC TCs (One-E shows the cycle it is designated, with its
        # real TD/TS status rather than waiting on the derived feed).
        named_combined: dict[str, Storm] = {**self.named, **self.current_named}

        # Designated-invest handoff (the 90E -> One-E pattern), two signals,
        # uniform across basins. Keep every invest that is STILL an invest --
        # the global track map's invest display is unaffected (separate source).
        #   1. EXPLICIT: knackwx named entries carry transitioned_from ("01E"
        #      came from "90E") -- drop that invest the cycle the link appears.
        #   2. CO-LOCATION: drop any invest within INVEST_DESIGNATION_DEG of a
        #      same-basin NAMED storm (tracks feeds + CurrentStorms -- the feeds
        #      cover WP/JTWC designations, which have no CurrentStorms entry).
        live_invests: dict[str, Storm] = {}
        for slug, inv in {**self.invests, **self.wp_invests}.items():
            desig = inv.name.split()[-1].upper()       # "INVEST 91W" -> "91W"
            retired_at = self.retired_invests.get(desig)
            if retired_at is not None:
                # Only fixes from BEFORE we first saw the transitioned_from
                # link belong to the promoted (retired) invest; a newer fix
                # means the number was RECYCLED for a new system -- float it.
                fix_t = parse_iso(inv.last_fix)
                if fix_t is None or fix_t <= retired_at:
                    log.info("invest %s/%s designated (knackwx transitioned_from); "
                             "dropping the invest floater", inv.name, inv.slug)
                    continue
            succ = next((s for s in named_combined.values()
                         if s.basin == inv.basin
                         and _deg_dist(s.lat, s.lon, inv.lat, inv.lon) <= INVEST_DESIGNATION_DEG),
                        None)
            if succ is not None:
                log.info("invest %s/%s designated -> %s/%s; dropping the invest floater",
                         inv.name, inv.slug, succ.name, succ.slug)
            else:
                live_invests[slug] = inv

        # --- Combine into the active floater set. Named (tracks + CurrentStorms)
        # plus the still-invests. Named and invest slugs never collide (90-99).
        combined: dict[str, Storm] = {**named_combined, **live_invests}
        self.storms = combined
        active = list(combined.values())
        active_slugs = set(combined)
        # New / updated storms -> ensure a Unit per band; refresh metadata.
        for s in active:
            for band in BANDS:
                key = (s.slug, band.key)
                if key not in self.units:
                    u = Unit(storm=s, band=band)
                    u.last_hash = self._resync_hash(s.slug, band.key)
                    self.units[key] = u
                else:
                    self.units[key].storm = s  # refresh position/metadata
        # Drop de-activated storms (frames stay on R2 for the grace window;
        # pruning ages them out). Remove their units so we stop polling.
        for key in list(self.units):
            if key[0] not in active_slugs:
                del self.units[key]
        self.write_top_manifest(active)
        log.info("active storms: %d (%s)", len(active),
                 ", ".join(f"{s.name}/{s.slug}" for s in active) or "none")

    def _resync_hash(self, slug: str, band_key: str) -> str | None:
        """On (re)start, recover the last uploaded frame's hash so we don't
        re-upload an unchanged frame after a restart."""
        man = self.r2.get_json(storm_manifest_key(slug))
        if not man:
            return None
        band = (man.get("bands") or {}).get(band_key) or {}
        return band.get("last_hash")

    # ---- manifests ------------------------------------------------------

    def write_top_manifest(self, active: list[Storm]) -> None:
        obj = {
            "generated_utc": iso_z(utcnow()),
            "storms": [
                {
                    "id": s.sid, "slug": s.slug, "name": s.name, "basin": s.basin,
                    "category": s.category,
                    # Pill intensity = the LATEST fix wind (pts[-1].wind_kt), so
                    # a strengthening/weakening storm shows its current value,
                    # not its season peak. Falls back to peak only when the JSON
                    # carries no fix-level wind yet.
                    "intensity_kt": (s.current_wind_kt
                                     if s.current_wind_kt is not None
                                     else s.intensity_kt),
                    "peak_wind_kt": s.intensity_kt,
                    "nature": s.nature,
                    "lat": s.lat, "lon": s.lon, "last_fix": s.last_fix,
                    "bands": [b.key for b in BANDS],
                    "manifest": f"{R2_PREFIX}/{s.slug}/manifest.json",
                }
                for s in active
            ],
        }
        self.r2.put_json(top_manifest_key(), obj, CACHE_MANIFEST)

    def append_frame(self, storm: Storm, band: Band, key: str, ts: dt.datetime,
                     content_hash: str) -> None:
        mkey = storm_manifest_key(storm.slug)
        man = self.r2.get_json(mkey) or {
            "id": storm.sid, "slug": storm.slug, "name": storm.name,
            "basin": storm.basin, "bands": {},
        }
        man["id"] = storm.sid
        man["name"] = storm.name
        man["basin"] = storm.basin
        man["generated_utc"] = iso_z(utcnow())
        bands = man.setdefault("bands", {})
        b = bands.setdefault(band.key, {"label": band.label, "frames": []})
        b["label"] = band.label
        b["frames"].append({"t": iso_z(ts), "key": key})
        kept, deleted = prune_frames(b["frames"], utcnow())
        b["frames"] = kept
        b["latest"] = kept[-1]["key"] if kept else key
        b["last_hash"] = content_hash
        b["updated_utc"] = iso_z(utcnow())
        if self.r2.put_json(mkey, man, CACHE_MANIFEST) and deleted:
            self.r2.delete(deleted)

    # ---- one unit -------------------------------------------------------

    def process_unit(self, u: Unit) -> None:
        storm, band = u.storm, u.band
        if band.daytime_only:
            zen = solar_zenith_deg(storm.lat, storm.lon, utcnow())
            if zen >= NIGHT_ZENITH_DEG:
                # Night: nothing to render; defer to next cadence slot.
                return
        half = BBOX_DEG / 2.0
        bbox = [round(norm_lon(storm.lon - half), 3), round(storm.lat - half, 3),
                round(norm_lon(storm.lon + half), 3), round(storm.lat + half, 3)]
        # Storm metadata for the rendered title badge (color-coded intensity).
        # Only the fields actually known get sent; missing pressure_mb is fine
        # (the badge omits the section). Empty when the JSON doesn't carry
        # a fix-level wind yet -- /render falls back to its plain title.
        storm_ctx = {
            "name": storm.name,
            "basin": storm.basin,
            "nature": storm.nature,
            "wind_kt": storm.current_wind_kt,
            "pressure_mb": storm.current_pressure_mb,
        }
        self.limiter.acquire()
        try:
            png, headers = call_render(
                self.session, bbox, band.channel, band.enhancement, storm=storm_ctx,
            )
        except RenderSkip as e:
            log.info("skip %s/%s: %s", storm.slug, band.key, e)
            self._consec_render_fail = 0
            return
        except RenderError as e:
            self._consec_render_fail += 1
            log.warning("render fail %s/%s (%d): %s",
                        storm.slug, band.key, self._consec_render_fail, e)
            if self._consec_render_fail >= CIRCUIT_TRIP_FAILS:
                self._circuit_open_until = time.monotonic() + CIRCUIT_COOLDOWN_S
                log.error("circuit OPEN: cooling down %ss", CIRCUIT_COOLDOWN_S)
            return
        self._consec_render_fail = 0

        h = hashlib.sha256(png).hexdigest()
        # X-Cache:HIT is a cheap hint that the source scan is unchanged, but
        # the content hash decides (handles HIT/MISS edge cases uniformly).
        if h == u.last_hash:
            return  # no new frame
        # Source scan time if exposed (more accurate loop time axis); else now.
        scan_hdr = (headers.get("X-Source-Time") or headers.get("X-Scan-Time")
                    or headers.get("X-Timestamp"))
        ts = parse_iso(scan_hdr) if scan_hdr else None
        ts = ts or utcnow()
        key = frame_key(storm.slug, band.key, ts)
        if not self.r2.put_bytes(key, png, "image/png", CACHE_FRAME):
            return  # upload failed -> do NOT touch manifest; retry next slot
        self.append_frame(storm, band, key, ts, h)
        u.last_hash = h
        log.info("uploaded %s (%d B, %s)", key, len(png),
                 headers.get("X-Satellite", "?"))

    # ---- cadence --------------------------------------------------------

    def cold_cadence(self) -> float:
        """Cold-band per-unit cadence. With private networking + small
        RATE_MIN_SPACING_S this floors at the 60 s target. On the public URL
        (RATE_MIN_SPACING_S=7) it stretches so the rate budget is respected
        and the hot bands stay fresh: C = max(60, U_cold * spacing)."""
        n_cold = sum(1 for u in self.units.values() if not u.band.hot)
        return max(CADENCE_TARGET_S, n_cold * self.limiter.min_spacing)

    def tick(self) -> None:
        now = time.monotonic()
        if now < self._circuit_open_until:
            return
        cold_c = self.cold_cadence()
        # Due units, hot first, then most-overdue, so the rate budget protects
        # the diagnostic IR bands under multi-storm contention.
        due = [u for u in self.units.values() if u.next_due <= now]
        due.sort(key=lambda u: (not u.band.hot, u.next_due))
        for u in due:
            if time.monotonic() < self._circuit_open_until:
                break
            cadence = CADENCE_TARGET_S if u.band.hot else cold_c
            self.process_unit(u)
            u.next_due = time.monotonic() + cadence

    # ---- main loop ------------------------------------------------------

    def run(self) -> None:
        log.info("floater poller starting | render=%s | bucket=%s | bbox=%g deg | spacing=%gs",
                 RENDER_URL, R2_BUCKET, BBOX_DEG, RATE_MIN_SPACING_S)
        while True:
            try:
                if time.monotonic() - self._last_tracks_refresh >= TRACKS_REFRESH_S \
                        or not self.units:
                    self.refresh_storms()
                    self._last_tracks_refresh = time.monotonic()
                if not self.units:
                    time.sleep(min(TRACKS_REFRESH_S, 60))  # idle: low CPU
                    continue
                self.tick()
                time.sleep(1.0)
            except KeyboardInterrupt:
                log.info("shutting down")
                return
            except Exception as e:  # noqa: BLE001 - the loop must never die
                log.exception("loop error (continuing): %s", e)
                time.sleep(5)


def _validate_env() -> None:
    missing = [n for n in ("R2_ENDPOINT",) if not _env(n)]
    if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        missing.append("R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY")
    if missing:
        log.error("missing required env: %s", ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    _validate_env()
    Poller().run()
