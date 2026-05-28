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
     8 deg storm-centered bbox + time=latest.
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

# Where the hardened active-storm list lives (Pages origin, not R2).
TRACKS_BASE = _env("TRACKS_BASE", "https://triple-a-tropics.com").rstrip("/")
TRACKS_BASINS = ("wp", "al", "ep")

# Geometry + cadence
BBOX_DEG = float(_env("BBOX_DEG", "8"))            # square floater width (deg)
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
    Band("ir",        "clean_ir",     "grayscale", "Clean IR",          hot=True,  daytime_only=False),
    Band("irbd",      "clean_ir",     "dvorak_bd", "IR (Dvorak BD)",    hot=True,  daytime_only=False),
    Band("wv_up",     "wv_upper",     "tat_neon",  "Upper WV",          hot=False, daytime_only=False),
    Band("wv_low",    "wv_lower",     "tat_neon",  "Lower WV",          hot=False, daytime_only=False),
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


def fetch_active_storms(session: requests.Session) -> Optional[list[Storm]]:
    """Return active storms across all basins.

    Returns ``None`` if ANY basin fetch failed -- a partial failure can
    silently drop the storms that live in the failing basin (we'd return
    an empty out from sibling-quiet basins), so the caller treats partial
    failure the same as full failure and preserves the last-known-good
    top manifest for one cycle (worst-case ~10 min stale). Returns ``[]``
    only when EVERY basin fetched cleanly and none reported active storms
    (genuine quiescence -- safe to clear the manifest).
    """
    now = utcnow()
    cutoff = now - dt.timedelta(hours=ACTIVE_WINDOW_HOURS)
    out: list[Storm] = []
    failed_basins: list[str] = []
    for basin in TRACKS_BASINS:
        url = f"{TRACKS_BASE}/{basin}_tracks_data.json"
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # noqa: BLE001 - never crash on a bad fetch
            log.warning("tracks fetch failed (%s): %s", basin, e)
            failed_basins.append(basin)
            continue
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
            # Latest-fix wind/pressure/nature for the color-coded title badge.
            # ``peak_wind_kt`` (storm-level) stays as ``intensity_kt`` for
            # legacy callers, but the badge uses the CURRENT fix values so a
            # weakening storm shows its current category, not its peak.
            cur_wind = last_pt.get("wind_kt")
            cur_pres = last_pt.get("pressure_mb")
            nature = last_pt.get("nature")
            out.append(Storm(
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
    if failed_basins:
        log.warning(
            "tracks fetch failed for %d/%d basins (%s) -- returning None to "
            "preserve last-known-good (avoids silently dropping storms that "
            "live in the failing basin)",
            len(failed_basins), len(TRACKS_BASINS), ",".join(failed_basins),
        )
        return None
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
        self.storms: dict[str, Storm] = {}     # slug -> storm
        self._last_tracks_refresh = 0.0
        self._consec_render_fail = 0
        self._circuit_open_until = 0.0

    # ---- storm-set management ------------------------------------------

    def refresh_storms(self) -> None:
        active = fetch_active_storms(self.session)
        # ``None`` => every basin's tracks JSON fetch failed (transient
        # origin/network hiccup). Don't touch the in-memory unit set OR the
        # top R2 manifest in that case -- preserve last-known-good so the
        # floater widget stays visible on the live site instead of self-
        # hiding for the full TOP_REFRESH_MS window on the frontend.
        if active is None:
            log.warning(
                "tracks fetch failed for ALL basins -- preserving last-known-good "
                "manifest (no storm set / unit changes this cycle)"
            )
            return
        active_slugs = {s.slug for s in active}
        # New / updated storms.
        for s in active:
            self.storms[s.slug] = s
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
        for slug in list(self.storms):
            if slug not in active_slugs:
                del self.storms[slug]
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
                    "category": s.category, "intensity_kt": s.intensity_kt,
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
