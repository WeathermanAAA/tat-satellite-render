#!/usr/bin/env python3
"""Mesoscale-sector satellite poller.

A long-running worker -- a SIBLING of floater_poller.py, fully ISOLATED from it
(separate process, separate state, separate R2 prefix ``meso/``, separate health
file). Where the floater poller follows storms, this poller follows the
operators' own steered mesoscale sectors: the GOES ABI mesoscale floaters
(CMIPM1 / CMIPM2 on GOES-19 + GOES-18) and the Himawari AHI Target sector
(Himawari-9). The five fixed sectors live in ``meso_sectors.MESO_SECTORS``.

Pipeline, per cycle, per sector (each sector fully isolated -- one sector's
fetch/render failure NEVER touches another sector or the floater poller):

  1. DISCOVER the sector's CURRENT extent. The operators steer M1/M2/Target
     around, so the bbox is not fixed: per scan we list the sector's latest scan
     of a cheap IR band and read its nav --
       * GOES (CMIPM1/CMIPM2): open the latest CMIPM*<band> netCDF's ATTRIBUTES
         ONLY (h5netcdf, no array load) and read geospatial_lat_min/max,
         geospatial_lon_min/max -- the same attrs satellites._check_meso_coverage
         reads. That rectangle IS the live meso box.
       * Himawari (AHI-L1b-Target): list the latest Target HSD segments for the
         band, download+decompress them (the Target sector is a small region, so
         segments are small), parse each segment's HSD header navigation, stitch
         the global col/line grid, and inverse-project its corners + edges to
         lat/lon via satellites._ahi_colline_to_latlon -> the live Target bbox.
     The discovered bbox is the change anchor for the sector (the bbox + its
     scan time): when the operators move the box or a new scan lands, every band
     re-renders.

  2. For the FULL band palette (the SAME BANDS table the floater poller uses, at
     full parity) call the LOCAL /render over private networking, exactly like
     floater_poller.call_render, with the discovered bbox + time=latest:
       * HOT bands (ir, irbd) target 60 s on their OWN lane thread + their own
         render container (RENDER_BASE_URL).
       * COLD bands (wv_up, wv_low, truecolor, swir) run on a SEPARATE lane
         thread against a SEPARATE render container (RENDER_BASE_URL_COLD) at
         the stretched cold cadence -- strict isolation, so a multi-second
         true-color render can never delay a due hot unit. The render-side
         night blend handles day/night inside /render -- no daytime-only
         special-casing here.
     sha256(png) is the new-frame source of truth (skip if unchanged).

  3. Write R2 under the ``meso/`` prefix:
       meso/{slug}/{band}/{YYYYMMDDTHHMMZ}.png   (immutable frames)
       meso/{slug}/manifest.json                 (per-sector bands body)
       meso/manifest.json                        (top index of sectors)
     Retention is RENDER-ONCE native within RECENT_WINDOW_H, thinned to
     THIN_SPACING_S out to HISTORY_WINDOW_H, dropped older (prune_frames,
     reused verbatim from the floater poller's policy).

Guards (the anti-freeze lessons the floater poller paid for, applied per sector):
  * MESO_ENABLED kill switch (default true). False -> the poller idles, writes
    NOTHING (no frames, no manifests), so an operator can stop it instantly.
  * Per-sector last-known-good: a sector that fails discovery or render keeps its
    last extent + frames and is flagged; it never disturbs another sector or the
    floater. There is no wholesale "preserve everything" path.
  * Source-freshness via poller_framework: each sector owns a SourceHealth, and
    every cycle ends by writing a health snapshot (per-sector latest-scan time +
    fresh/stale/failing classification + an overall healthy flag) to R2 and to a
    local file a compose healthcheck curls. Staleness is always DETECTABLE.

Never crashes: every sector is wrapped; /render errors retry+backoff then skip;
R2 errors log+skip (the manifest only references uploaded keys); on restart it
resyncs last-hash state from the R2 manifests so it neither duplicates frames nor
loses history.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import http.server
import json
import logging
import os
import random
import sys
import threading
import time
from typing import Iterable, Optional

import boto3
import requests
from botocore.config import Config as BotoConfig

from meso_sectors import MESO_SECTORS, MesoSector

# poller_framework is dependency-free (no requests/boto3); import its health +
# freshness primitives for the per-sector isolation + heartbeat. Imported at
# module load (cheap, stdlib-only) so a missing framework fails fast + loud.
from poller_framework import (
    FAILING,
    FRESH,
    SourceHealth,
    iso_z as pf_iso_z,
    process_mem_mb,
)


# ---------------------------------------------------------------------------
# Config (env-driven; safe defaults). Mirrors floater_poller's knobs so the two
# workers tune identically, but every default is independent (separate process).
# ---------------------------------------------------------------------------

def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    v = (_env(name, "1" if default else "0") or "").strip().lower()
    return v not in ("0", "false", "no", "off")


# Global kill switch. False -> idle, write nothing.
MESO_ENABLED = _env_bool("MESO_ENABLED", True)

# /render base. Prefer Railway private networking, e.g.
#   http://tat-satellite-render.railway.internal:8080
# Falls back to the public URL (then set RATE_MIN_SPACING_S=7).
RENDER_BASE_URL = _env("RENDER_BASE_URL", "https://web-production-b88d.up.railway.app").rstrip("/")
RENDER_URL = RENDER_BASE_URL + "/render"
# COLD-lane /render base. The compose stack points this at a SECOND render
# container (meso-render-cold) so a slow cold render (true-color ~spans
# seconds even optimized) physically cannot occupy the hot lane's render
# worker — the hot IR/IRBD 60 s cadence survives any cold-band weather.
# Defaults to the hot URL so a single-service deployment still works.
RENDER_BASE_URL_COLD = (_env("RENDER_BASE_URL_COLD") or RENDER_BASE_URL).rstrip("/")
RENDER_URL_COLD = RENDER_BASE_URL_COLD + "/render"

# R2 / S3 -- SAME bucket as the floater worker, but a distinct prefix so the two
# never touch each other's keys.
R2_ENDPOINT = _env("R2_ENDPOINT")
R2_BUCKET = _env("R2_BUCKET", "triple-a-tropics-media")
R2_ACCESS_KEY_ID = _env("R2_ACCESS_KEY_ID") or _env("AWS_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _env("R2_SECRET_ACCESS_KEY") or _env("AWS_SECRET_ACCESS_KEY")
R2_PREFIX = _env("R2_PREFIX", "meso").strip("/")

# Cadence + geometry. No BBOX_DEG here -- the bbox comes from the discovered
# sector extent, not a fixed storm crop.
CADENCE_TARGET_S = float(_env("CADENCE_TARGET_S", "60"))         # hot-band target (native GOES meso)
COLD_CADENCE_TARGET_S = float(_env("COLD_CADENCE_TARGET_S", "300"))  # cold-band target (stretched)
SECTORS_REFRESH_S = float(_env("SECTORS_REFRESH_S", "120"))      # re-discover extents
# Himawari render product: "target" -> the ~2.5-min AHI Target sub-scans
# (R301..R304) for the 5 scalar bands; "fldk" -> the 10-min full disk. Flip to
# fldk to revert Himawari to the old cadence WITHOUT a rebuild (env + restart).
# True-color always uses FLDK (the compositor can't mix a Target red with FLDK
# greens). GOES is unaffected (it always uses the CMIPM meso product).
MESO_HIMAWARI_PRODUCT = (_env("MESO_HIMAWARI_PRODUCT", "target")
                         or "target").strip().lower()
RATE_MIN_SPACING_S = float(_env("RATE_MIN_SPACING_S", "1.0"))    # min gap between /render calls
EXTENT_STALE_AFTER_S = float(_env("EXTENT_STALE_AFTER_S", "1800"))  # >this since last scan => stale

# Retention (R2) -- native recent + thinned history (render-once policy).
RECENT_WINDOW_H = float(_env("RECENT_WINDOW_H", "6"))
HISTORY_WINDOW_H = float(_env("HISTORY_WINDOW_H", "24"))
THIN_SPACING_S = float(_env("THIN_SPACING_S", "300"))
# Manifest==storage re-sync cadence. The manifest is rebuilt from an R2
# LISTING (storage is the source of truth) at startup and every RECONCILE_S,
# so a manifest can never keep advertising frames whose objects are gone
# (the viewer 404s on those). See reconcile_manifests.
RECONCILE_S = float(_env("MESO_RECONCILE_S", "900"))

# Frame codec requested from /render. "webp" = 1056 px lossy-WebP loop frames
# (~6x lighter than the 1320 px PNGs); "png" = the legacy full-res PNG and the
# instant rollback knob. The poller derives the uploaded extension and
# content-type from the RESPONSE (see frame_ext), so a render service that
# predates the format param -- or a rollback flip -- stays self-consistent:
# whatever /render actually sent is what gets keyed and served.
FRAME_FORMAT = (_env("FRAME_FORMAT", "webp") or "webp").strip().lower()

# HTTP timeouts / retries (render + extent discovery).
RENDER_TIMEOUT_S = float(_env("RENDER_TIMEOUT_S", "45"))
RENDER_MAX_RETRIES = int(_env("RENDER_MAX_RETRIES", "3"))
CIRCUIT_TRIP_FAILS = int(_env("CIRCUIT_TRIP_FAILS", "8"))
CIRCUIT_COOLDOWN_S = float(_env("CIRCUIT_COOLDOWN_S", "60"))
DISCOVER_RETRIES = int(_env("DISCOVER_RETRIES", "2"))            # retries after first attempt

# Health classification.
FAIL_THRESHOLD = int(_env("FAIL_THRESHOLD", "3"))

# Tiny health HTTP server (a compose healthcheck curls this).
HEALTH_PORT = int(_env("HEALTH_PORT", "8090"))
HEALTH_FILE = _env("HEALTH_FILE", "/tmp/meso_health.json")

CACHE_FRAME = "public, max-age=31536000, immutable"
CACHE_MANIFEST = "max-age=30"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("meso")


# ---------------------------------------------------------------------------
# Bands -- the SAME table the floater poller renders, at full parity. The render
# side owns the night blend for truecolor (no daytime_only gate here).
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Band:
    key: str            # path/manifest key
    channel: str        # /render channel
    enhancement: str    # /render enhancement
    label: str          # UI label
    hot: bool           # hot=always 60 s; cold=stretches with unit count


BANDS: tuple[Band, ...] = (
    Band("ir",        "clean_ir",     "rainbow_ir", "Clean IR",             hot=True),
    Band("irbd",      "clean_ir",     "dvorak",     "IR (Dvorak)",          hot=True),
    Band("wv_up",     "wv_upper",     "wv_tat",     "Upper WV",             hot=False),
    Band("wv_low",    "wv_lower",     "wv_tat",     "Lower WV",             hot=False),
    Band("truecolor", "true_color",   "tat_neon",   "Visible (true color)", hot=False),
    Band("swir",      "shortwave_ir", "grayscale",  "Shortwave IR",         hot=False),
)
BANDS_BY_KEY = {b.key: b for b in BANDS}


# ---------------------------------------------------------------------------
# Small helpers (mirror floater_poller / poller_framework time helpers)
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


def norm_lon(lon: float) -> float:
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def sector_family_hint(sector: MesoSector) -> str:
    """/render ``satellite`` hint for a sector -- the satellite FAMILY that
    owns the sector's imagery (mirrors satellites.SATELLITES_BY_FAMILY keys)."""
    if sector.family == "himawari":
        return "Himawari-Pacific"
    return "GOES-West" if sector.bucket == "noaa-goes18" else "GOES-East"


# ---------------------------------------------------------------------------
# Discovered sector extent (the per-scan result of extent discovery)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SectorExtent:
    """One discovered scan of a meso sector: its live bbox + scan time.

    ``bbox`` is [lon_min, lat_min, lon_max, lat_max] (the box the operators have
    the sector pointed at right now). ``scan_start`` is the scan's valid time
    (the change anchor + the loop time-axis). ``sat_name`` is reported in the
    manifest. ``key`` is a stable change-token: a new scan time OR a moved box
    both change it, so every band re-renders.
    """
    bbox: list[float]
    scan_start: dt.datetime
    sat_name: str

    @property
    def key(self) -> str:
        b = ",".join(f"{v:.3f}" for v in self.bbox)
        return f"{self.scan_start.strftime('%Y%m%dT%H%M%SZ')}|{b}"


# ---------------------------------------------------------------------------
# Extent discovery -- GOES (netCDF attrs) + Himawari (HSD nav). Heavy deps
# (s3fs / xarray / numpy) are imported lazily INSIDE the discovery functions so
# `python -c "import meso_poller"` works without them (the guards in the task).
# ---------------------------------------------------------------------------

class DiscoverError(Exception):
    """Sector extent could not be discovered this cycle (transient -- the sector
    keeps its last-known-good extent and is flagged)."""


# Map generic channel -> ABI / AHI native band (mirrors satellites.generic_to_band
# for the locate band only; kept local so discovery doesn't import the whole
# Satellite layer just for one lookup).
_GOES_GENERIC_TO_BAND = {"clean_ir": 13, "wv_upper": 8, "wv_lower": 10,
                         "shortwave_ir": 7, "visible_red": 2}
_AHI_GENERIC_TO_BAND = {"clean_ir": 13, "wv_upper": 8, "wv_lower": 10,
                        "shortwave_ir": 7, "visible_red": 3}


def _get_fs():
    """Anonymous s3fs handle for the public NOAA Open Data buckets. Reuses the
    SAME singleton + tuning satellites._get_fs uses so behavior matches the
    render service's S3 access exactly."""
    import satellites
    return satellites._get_fs()


def discover_goes_extent(sector: MesoSector) -> SectorExtent:
    """Discover a GOES mesoscale sector's (CMIPM1/CMIPM2) current extent.

    Lists the latest CMIPM*<locate_band> scan in the sector's bucket and reads
    ONLY its geospatial_lat/lon_min/max attributes (h5netcdf, no array load) --
    the same attrs satellites._check_meso_coverage_sync reads -- which IS the
    live meso box. Returns the bbox + scan time. Raises DiscoverError on any
    failure (no scan found / unreadable attrs)."""
    import xarray as xr
    import satellites

    band = _GOES_GENERIC_TO_BAND.get(sector.locate_band, 13)
    target = utcnow()
    # GOES L2 CMIP mesoscale lives under the SINGLE product prefix ABI-L2-CMIPM/;
    # M1 vs M2 is encoded in the FILENAME (OR_ABI-L2-CMIPM1-... / ...-CMIPM2-...),
    # NOT in separate ABI-L2-CMIPM1//ABI-L2-CMIPM2/ prefixes (those don't exist on
    # the bucket -- listing them returns []). So list the CMIPM hour (this hour +
    # the previous, for the hour-boundary edge) and filter keys to this sector's
    # M1/M2 token. (_list_hour already filters by the C{band} channel token.)
    sector_tok = f"-{sector.sector}-"   # "-CMIPM1-" / "-CMIPM2-"
    files = [f for f in (
                satellites._list_hour(sector.bucket, "CMIPM", band, target)
                + satellites._list_hour(sector.bucket, "CMIPM", band,
                                        target - dt.timedelta(hours=1)))
             if sector_tok in f]
    if not files:
        raise DiscoverError(f"no {sector.sector} scan found in {sector.bucket}")
    with_t = sorted(((f, satellites._parse_scan_start(f)) for f in files),
                    key=lambda p: p[1])
    s3_key, scan_start = with_t[-1]  # newest scan
    fs = _get_fs()
    try:
        with fs.open(s3_key, mode="rb") as f:
            ds = xr.open_dataset(f, decode_cf=False, engine="h5netcdf")
            try:
                # The meso extent is on the geospatial_lat_lon_extent VARIABLE's
                # attrs (west/south/east/north), NOT global geospatial_lat_min/max
                # (which don't exist on ABI files). One canon: satellites helper.
                lon_min, lat_min, lon_max, lat_max = \
                    satellites._goes_meso_extent_from_ds(ds)
            finally:
                ds.close()
    except Exception as e:  # noqa: BLE001
        raise DiscoverError(f"extent read failed for {s3_key}: {e}") from e
    bbox = [round(norm_lon(lon_min), 3), round(lat_min, 3),
            round(norm_lon(lon_max), 3), round(lat_max, 3)]
    # lon_e < lon_w is a VALID antimeridian crossing (operators steer M1/M2 across
    # the dateline -- e.g. a Bering/Aleutians sector; the render + frontend handle
    # e<w), NOT degenerate. Validate latitude order + a positive WRAPPED lon span
    # instead of lon_w < lon_e.
    lon_span = (bbox[2] - bbox[0]) % 360.0
    if not (bbox[1] < bbox[3] and 0.0 < lon_span < 359.0):
        raise DiscoverError(f"degenerate bbox {bbox} for {s3_key}")
    return SectorExtent(bbox=bbox, scan_start=scan_start,
                        sat_name=satellites.goes_sat_label(sector.bucket))


def discover_himawari_extent(sector: MesoSector) -> SectorExtent:
    """Discover the Himawari AHI Target sector's current extent.

    Lists the latest Target HSD segments for the locate band, downloads +
    decompresses them (the Target region is small, so the segments are small),
    parses each segment's HSD header navigation, stitches the global col/line
    grid (first_line_number + n_lines per segment), and inverse-projects the
    grid's corners + edge midpoints to lat/lon via
    satellites._ahi_colline_to_latlon. The finite lat/lon min/max IS the live
    Target bbox. Raises DiscoverError on any failure."""
    import bz2
    import numpy as np
    import satellites
    from vendor.ahi_hsd import parse_hsd_segment

    band = _AHI_GENERIC_TO_BAND.get(sector.locate_band, 13)
    fs = _get_fs()
    # AHI cycles every ~10 min; the Target sector cycles faster but its folders
    # share the FLDK 10-min timestamp layout. Floor to the most-recent published
    # 10-min slot and back off one slot for publishing latency (mirrors
    # HimawariPacificSatellite._snap_10min for nearest_to_target=False).
    base = utcnow().replace(second=0, microsecond=0)
    floored = base.replace(minute=(base.minute // 10) * 10)
    seg_paths: list[str] = []
    chosen_slot: Optional[dt.datetime] = None
    # back=0 first: Target sub-scans land every ~2.5 min, so the current
    # 10-min folder usually already has segments (matches find_file's
    # _resolve_target_sync, which also starts at the current slot).
    for back in range(0, 4):  # current slot + 3 back (publish latency / gaps)
        slot = floored - dt.timedelta(minutes=10 * back)
        prefix = (f"{sector.bucket}/AHI-L1b-Target/"
                  f"{slot.year:04d}/{slot.month:02d}/{slot.day:02d}/"
                  f"{slot.hour:02d}{slot.minute:02d}/")
        try:
            listing = fs.ls(prefix)
        except (FileNotFoundError, OSError):
            listing = []
        band_token = f"_B{band:02d}_"
        matches = sorted(f for f in listing
                         if band_token in f and f.endswith(".DAT.bz2"))
        if matches:
            seg_paths, chosen_slot = matches, slot
            break
    if not seg_paths or chosen_slot is None:
        raise DiscoverError(f"no Target scan found in {sector.bucket}")

    # Parse each segment's header navigation. We need the full segment buffer for
    # parse_hsd_segment, but Target segments are small (a ~1000-2000 km box), so
    # this is cheap. Stitch the global col/line extent from first_line_number.
    sub_lon = cfac = lfac = coff = loff = None
    n_columns = 0
    line_min = None
    line_max = None
    for p in seg_paths:
        try:
            with fs.open(p, mode="rb") as fh:
                raw = fh.read()
            seg = parse_hsd_segment(bz2.decompress(raw))
        except Exception as e:  # noqa: BLE001
            raise DiscoverError(f"HSD parse failed for {p}: {e}") from e
        sub_lon, cfac, lfac = seg.sub_lon, seg.cfac, seg.lfac
        coff, loff = seg.coff, seg.loff
        n_columns = max(n_columns, seg.n_columns)
        seg_line_lo = seg.first_line_number
        seg_line_hi = seg.first_line_number + seg.n_lines - 1
        line_min = seg_line_lo if line_min is None else min(line_min, seg_line_lo)
        line_max = seg_line_hi if line_max is None else max(line_max, seg_line_hi)
    if sub_lon is None or n_columns == 0 or line_min is None:
        raise DiscoverError(f"no usable Target segments in {sector.bucket}")

    # Inverse-project the col/line grid corners + edge midpoints to lat/lon.
    # Columns run 1..n_columns; lines run line_min..line_max (HSD is 1-based).
    cols = np.array([1, n_columns / 2.0, n_columns], dtype=np.float64)
    lines = np.array([line_min, (line_min + line_max) / 2.0, line_max],
                     dtype=np.float64)
    COL, LINE = np.meshgrid(cols, lines)
    lat, lon = satellites._ahi_colline_to_latlon(
        COL, LINE, float(sub_lon), int(cfac), int(lfac), float(coff), float(loff))
    finite = np.isfinite(lat) & np.isfinite(lon)
    if not finite.any():
        raise DiscoverError("Target grid corners project off-disk (no finite latlon)")
    lats = lat[finite]
    lons = lon[finite]
    bbox = [round(float(np.min(lons)), 3), round(float(np.min(lats)), 3),
            round(float(np.max(lons)), 3), round(float(np.max(lats)), 3)]
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise DiscoverError(f"degenerate Target bbox {bbox}")
    return SectorExtent(bbox=bbox, scan_start=chosen_slot, sat_name=sector.satellite)


def discover_extent(sector: MesoSector) -> SectorExtent:
    """Discover one sector's current extent, dispatching on family. Retries the
    discovery (transient S3 hiccups) with backoff before giving up. Raises
    DiscoverError on final failure (the caller preserves last-known-good)."""
    fn = discover_goes_extent if sector.family == "goes" else discover_himawari_extent
    last: Optional[Exception] = None
    for attempt in range(DISCOVER_RETRIES + 1):
        try:
            return fn(sector)
        except DiscoverError:
            raise
        except Exception as e:  # noqa: BLE001 - transient S3/parse -> retry
            last = e
            if attempt < DISCOVER_RETRIES:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    raise DiscoverError(str(last))


# ---------------------------------------------------------------------------
# R2 client (verbatim shape from floater_poller.R2)
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

    def list_keys(self, prefix: str) -> list[str]:
        """Every object key under ``prefix``. Raises on failure -- callers
        decide whether a listing error may blank state (it must not)."""
        keys: list[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys


# ---------------------------------------------------------------------------
# Rate limiter (verbatim from floater_poller)
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
# Render call (verbatim contract from floater_poller.call_render)
# ---------------------------------------------------------------------------

class RenderError(Exception):
    pass


class RenderSkip(Exception):
    """422 / coverage / night -- expected, not an error."""


def call_render(session: requests.Session, bbox: list[float], channel: str,
                enhancement: str, storm: Optional[dict] = None,
                product: Optional[str] = None, satellite: Optional[str] = None,
                url: str = RENDER_URL) -> tuple[bytes, dict]:
    body: dict = {"bbox": bbox, "time": "latest", "channel": channel,
                  "enhancement": enhancement, "format": FRAME_FORMAT}
    if storm is not None:
        body["storm"] = storm
    if product is not None:
        body["product"] = product
    if satellite is not None:
        body["satellite"] = satellite
    last_exc: Exception | None = None
    for attempt in range(RENDER_MAX_RETRIES):
        try:
            r = session.post(url, json=body, timeout=RENDER_TIMEOUT_S)
            if r.status_code == 422:
                raise RenderSkip(r.text[:200])
            if r.status_code == 429:
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
# Manifest / frame state per (sector, band) -- meso/ prefix
# ---------------------------------------------------------------------------

def header_get(headers: dict, name: str) -> Optional[str]:
    """Case-insensitive lookup in a plain header dict. call_render hands back
    dict(r.headers) with the SERVER's casing — uvicorn lowercases everything,
    so a literal .get("X-Scan-Time") silently missed and every frame fell
    back to the stale discovery slot time (new scans then OVERWROTE the
    previous frame under the old stamp, and Himawari Target sub-scans
    collapsed onto their 10-min slot)."""
    lname = name.lower()
    return next((v for k, v in headers.items() if k.lower() == lname), None)


def frame_ext(headers: dict) -> tuple[str, str]:
    """(key extension, content-type) for a /render response.

    Derived from the response rather than FRAME_FORMAT so the poller can never
    mislabel a frame across a mixed-version window: an old render service that
    ignores the format param answers image/png and gets keyed .png; the new
    service answers image/webp and gets keyed .webp. Header lookup is
    case-insensitive (call_render hands back a plain dict)."""
    ctype = next(
        (v for k, v in headers.items() if k.lower() == "content-type"), ""
    )
    ctype = (ctype or "").split(";")[0].strip().lower()
    if ctype == "image/webp":
        return ".webp", "image/webp"
    return ".png", "image/png"


def frame_key(slug: str, band_key: str, ts: dt.datetime, ext: str = ".png") -> str:
    # SECOND precision. GOES meso scans are ~60 s apart but jitter (the public
    # bucket shows 57-63 s gaps), so two DISTINCT scans can land in the same
    # wall-clock minute; a minute-resolution key would collide and the second
    # scan's frame would overwrite the first under the same key (the content-hash
    # dedup can't catch it -- different bytes both pass). Seconds make every scan
    # a distinct key. SectorExtent.key already uses %Y%m%dT%H%M%SZ.
    return f"{R2_PREFIX}/{slug}/{band_key}/{ts.strftime('%Y%m%dT%H%M%SZ')}{ext}"


def sector_manifest_key(slug: str) -> str:
    return f"{R2_PREFIX}/{slug}/manifest.json"


def top_manifest_key() -> str:
    return f"{R2_PREFIX}/manifest.json"


def health_key() -> str:
    return f"{R2_PREFIX}/health.json"


def frame_ts_from_key(key: str) -> Optional[dt.datetime]:
    """Recover a frame's scan time from its object key name -- frame_key()
    encodes it ({prefix}/{slug}/{band}/{stamp}.png or .webp), so a band's frame
    list is fully reconstructible from an R2 LISTING alone. Tolerates the
    legacy minute-precision stamps alongside the current second-precision ones,
    and both frame codecs (.png pre-cutover, .webp after)."""
    name = key.rsplit("/", 1)[-1]
    if name.endswith(".png"):
        stamp = name[: -len(".png")]
    elif name.endswith(".webp"):
        stamp = name[: -len(".webp")]
    else:
        return None
    # exact-length dispatch: strptime pads greedily, so a minute-precision
    # stamp would MIS-PARSE under the second-precision format (0441Z ->
    # 04:04:01) instead of falling through.
    if len(stamp) == 16:
        fmt = "%Y%m%dT%H%M%SZ"
    elif len(stamp) == 14:
        fmt = "%Y%m%dT%H%MZ"
    else:
        return None
    try:
        return dt.datetime.strptime(stamp, fmt).replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def prune_frames(frames: list[dict], now: dt.datetime) -> tuple[list[dict], list[str]]:
    """Keep native cadence within RECENT_WINDOW_H; thin to THIN_SPACING_S out to
    HISTORY_WINDOW_H; drop older. Returns (kept, deleted_keys). Reused verbatim
    from the floater poller's retention policy."""
    recent_cut = now - dt.timedelta(hours=RECENT_WINDOW_H)
    history_cut = now - dt.timedelta(hours=HISTORY_WINDOW_H)
    kept: list[dict] = []
    deleted: list[str] = []
    last_kept_t: dt.datetime | None = None
    for f in sorted(frames, key=lambda x: x["t"]):
        t = parse_iso(f["t"])
        if t is None or t < history_cut:
            deleted.append(f["key"])
            continue
        if t >= recent_cut:
            kept.append(f)
            last_kept_t = t
            continue
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
    """One (sector, band) render unit."""
    sector: MesoSector
    band: Band
    next_due: float = 0.0          # monotonic time
    last_hash: str | None = None   # sha256 of last uploaded frame


class Lane:
    """One scheduling lane (hot or cold), STRICTLY isolated from its sibling:
    its own units, its own requests.Session, its own rate limiter, its own
    circuit breaker, and its own /render URL — run on its own thread by
    MesoPoller._lane_loop. A cold render physically cannot delay a hot unit:
    nothing in the hot lane ever waits on cold state, and the compose stack
    points each lane at a separate render container."""

    def __init__(self, name: str, units: list[Unit], render_url: str,
                 min_spacing_s: float, drain_all: bool) -> None:
        self.name = name
        self.units = units
        self.render_url = render_url
        self.session = requests.Session()
        self.session.headers["User-Agent"] = f"tat-meso-poller/1.0 ({name})"
        self.limiter = RateLimiter(min_spacing_s)
        # drain_all: hot processes EVERY due unit per pass (the whole IR/IRBD
        # fleet must land inside the 60 s budget); cold takes only the single
        # most-overdue unit per pass, so cold throughput self-paces to the
        # cold service's actual render speed.
        self.drain_all = drain_all
        self.consec_fail = 0
        self.circuit_open_until = 0.0


class MesoPoller:
    def __init__(self) -> None:
        self.r2 = R2()
        self.units: dict[tuple[str, str], Unit] = {}
        # Per-sector last-known-good extent (preserved on a discovery failure).
        # Written by the main (discovery) thread, read by the lane threads --
        # whole-value dict assignment, so readers always see a complete extent.
        self.extents: dict[str, SectorExtent] = {}
        # Per-sector health -- the freshness heartbeat (poller_framework).
        self.health: dict[str, SourceHealth] = {
            s.slug: SourceHealth(name=s.slug) for s in MESO_SECTORS
        }
        self._last_sectors_refresh = 0.0
        self._last_reconcile = 0.0
        # MEMORY-AUTHORITATIVE per-sector manifests (seeded by
        # reconcile_manifests in run(); this poller is the sole writer).
        # append_frame must never GET-modify-PUT remote manifest state: a
        # stale read (e.g. a second poller container left running across a
        # rebuild) made thinning delete objects that another manifest copy
        # still listed -- the viewer then 404'd on scattered phantom frames.
        # Both lanes (+ the main reconcile loop) mutate manifest state, so
        # every mutation+upload holds _manifest_lock.
        self.manifests: dict[str, dict] = {}
        self._manifest_lock = threading.Lock()
        # Build the unit set once (sectors are fixed; only their extents move).
        # last_hash is seeded from the reconciled manifests in run().
        for sector in MESO_SECTORS:
            for band in BANDS:
                self.units[(sector.slug, band.key)] = Unit(sector=sector, band=band)
        # STRICT lane isolation (hot = IR/IRBD at the 60 s target; cold =
        # WV/true-color/SWIR on the stretched cadence). Separate threads,
        # sessions, limiters, circuits, and -- via RENDER_BASE_URL_COLD --
        # separate render containers, so a slow cold render can never starve
        # the hot lane.
        self.hot_lane = Lane(
            "hot", [u for u in self.units.values() if u.band.hot],
            RENDER_URL, RATE_MIN_SPACING_S, drain_all=True)
        self.cold_lane = Lane(
            "cold", [u for u in self.units.values() if not u.band.hot],
            RENDER_URL_COLD, RATE_MIN_SPACING_S, drain_all=False)

    # ---- extent discovery (per-sector, fully isolated) -----------------

    def refresh_extents(self) -> None:
        """Re-discover every sector's current extent. Each sector is isolated:
        a failure preserves THAT sector's last-known-good extent + flags it via
        its SourceHealth, and never touches another sector. Always ends by
        rewriting the top manifest + emitting health."""
        now = utcnow()
        for sector in MESO_SECTORS:
            h = self.health[sector.slug]
            h.last_attempt_utc = now
            h.total_polls += 1
            try:
                ext = discover_extent(sector)
            except Exception as e:  # noqa: BLE001 - isolation boundary
                h.consecutive_failures += 1
                h.total_failures += 1
                h.last_error = f"{type(e).__name__}: {e}"
                kept = self.extents.get(sector.slug)
                log.warning("discover %s FAILED (%s); %s", sector.slug, h.last_error,
                            "preserving last-known-good extent" if kept
                            else "no prior extent yet")
                continue
            h.last_success_utc = now
            h.last_valid_time = ext.scan_start
            prev = self.extents.get(sector.slug)
            h.last_signature = ext.key
            if prev is None or prev.key != ext.key:
                h.last_change_utc = now
            h.consecutive_failures = 0
            h.last_error = None
            self.extents[sector.slug] = ext
            # Refresh each band unit's sector view (extent read live in process).
            log.info("discover %s -> bbox=%s scan=%s",
                     sector.slug, ext.bbox, iso_z(ext.scan_start))
        self.write_top_manifest()
        self.emit_health()

    # ---- manifests ------------------------------------------------------

    def reconcile_manifests(self) -> None:
        """Rebuild every per-sector manifest from an R2 LISTING -- storage is
        the source of truth. frame_key encodes the scan time in the object
        name, so each band's frame list is reconstructed wholly from what
        actually exists; entries a previous manifest advertised but storage
        doesn't have ("phantom" frames, which the viewer 404s on) are dropped,
        and out-of-retention objects are pruned. Runs at startup and every
        RECONCILE_S, so manifest==storage self-heals even across writer
        accidents (e.g. two poller containers fighting across a rebuild).
        A band whose LISTING fails keeps its previous body -- a transient list
        error must never blank a healthy band."""
        now = utcnow()
        for sector in MESO_SECTORS:
            mkey = sector_manifest_key(sector.slug)
            prev = self.manifests.get(sector.slug) or self.r2.get_json(mkey) or {}
            old_bands = prev.get("bands") or {}
            man = {
                "slug": sector.slug, "satellite": sector.satellite,
                "sector": sector.sector, "label": sector.label,
                "generated_utc": iso_z(now), "bands": {},
            }
            to_delete: list[str] = []
            phantoms = 0
            counts: list[str] = []
            for band in BANDS:
                ob = old_bands.get(band.key) or {}
                try:
                    keys = self.r2.list_keys(
                        f"{R2_PREFIX}/{sector.slug}/{band.key}/")
                except Exception as e:  # noqa: BLE001 - per-band isolation
                    log.warning("reconcile list %s/%s failed: %s -- keeping "
                                "prior body", sector.slug, band.key, e)
                    if ob:
                        man["bands"][band.key] = ob
                    continue
                frames = []
                for k in keys:
                    t = frame_ts_from_key(k)
                    if t is not None:
                        frames.append({"t": iso_z(t), "key": k})
                frames.sort(key=lambda f: f["t"])
                kept, deleted = prune_frames(frames, now)
                to_delete.extend(deleted)
                stored = {f["key"] for f in frames}
                phantoms += sum(1 for f in (ob.get("frames") or [])
                                if f.get("key") not in stored)
                man["bands"][band.key] = {
                    "label": band.label,
                    "frames": kept,
                    "latest": kept[-1]["key"] if kept else ob.get("latest"),
                    "last_hash": ob.get("last_hash"),
                    "updated_utc": ob.get("updated_utc") or iso_z(now),
                }
                counts.append(f"{band.key}:{len(kept)}")
            # Hold the manifest lock across the swap+upload so a concurrent
            # lane append can't interleave between our PUT and state swap
            # (the listing above is a snapshot either way; the next
            # reconcile re-grounds whatever raced in).
            with self._manifest_lock:
                # Re-graft any frames a lane appended for this sector while
                # we were listing -- they're newer than the listing snapshot
                # and must not be dropped from the manifest. Membership in
                # the in-memory manifest (the sole writer's own state) is
                # the test; a scan-TIME cutoff would wrongly drop Himawari
                # FLDK frames, whose scan stamps trail wall clock by 10-20
                # minutes of mirror lag.
                dead = set(to_delete)
                cur = self.manifests.get(sector.slug) or {}
                for bk, cb in (cur.get("bands") or {}).items():
                    mb = man["bands"].setdefault(bk, {"label": cb.get("label"),
                                                      "frames": []})
                    known = {f["key"] for f in mb["frames"]}
                    fresh = [f for f in (cb.get("frames") or [])
                             if f["key"] not in known and f["key"] not in dead]
                    if fresh:
                        mb["frames"] = sorted(mb["frames"] + fresh,
                                              key=lambda f: f["t"])
                        mb["latest"] = mb["frames"][-1]["key"]
                        mb["last_hash"] = cb.get("last_hash") or mb.get("last_hash")
                # Re-prune with a FRESH clock inside the lock: a lane's
                # append_frame may have pruned (and deleted from R2) a
                # boundary-age frame AFTER our listing snapshot -- the
                # snapshot-era prune above would still advertise it.
                now_locked = utcnow()
                for bk, mb in man["bands"].items():
                    kept, deleted = prune_frames(mb.get("frames") or [],
                                                 now_locked)
                    mb["frames"] = kept
                    if kept:
                        mb["latest"] = kept[-1]["key"]
                    to_delete.extend(deleted)
                if self.r2.put_json(mkey, man, CACHE_MANIFEST):
                    self.manifests[sector.slug] = man
                    if to_delete:
                        self.r2.delete(to_delete)
            log.info("reconcile %s: %s%s%s", sector.slug, " ".join(counts),
                     f" | dropped {phantoms} phantom manifest entries"
                     if phantoms else "",
                     f" | pruned {len(to_delete)} objects" if to_delete else "")

    def write_top_manifest(self) -> None:
        """Top index mirrors floater write_top_manifest, storms->sectors:
        s.id->sec.slug, with sector metadata. Only sectors with a discovered
        extent are listed (an undiscovered sector has no bbox to advertise)."""
        sectors_out = []
        for sector in MESO_SECTORS:
            ext = self.extents.get(sector.slug)
            if ext is None:
                continue
            cx = round((ext.bbox[0] + ext.bbox[2]) / 2.0, 2)
            cy = round((ext.bbox[1] + ext.bbox[3]) / 2.0, 2)
            sectors_out.append({
                "slug": sector.slug,
                "satellite": ext.sat_name or sector.satellite,
                "sector": sector.sector,
                "label": sector.label,
                "lat": cy,
                "lon": cx,
                "bbox": ext.bbox,
                "scan": iso_z(ext.scan_start),
                "bands": [b.key for b in BANDS],
                "manifest": f"{R2_PREFIX}/{sector.slug}/manifest.json",
            })
        obj = {"generated_utc": iso_z(utcnow()), "sectors": sectors_out}
        self.r2.put_json(top_manifest_key(), obj, CACHE_MANIFEST)

    def append_frame(self, sector: MesoSector, band: Band, key: str,
                     ts: dt.datetime, content_hash: str) -> None:
        """Per-sector manifest with the BYTE-IDENTICAL bands body shape the
        floater poller uses: bands:{<k>:{label,frames:[{t,key}],latest,
        last_hash,updated_utc}}. MEMORY-AUTHORITATIVE: appends to this
        process's own manifest state and writes it out -- never re-reads the
        remote copy (a stale read is how phantom entries were born); the
        periodic reconcile re-grounds the state in an actual R2 listing."""
        with self._manifest_lock:
            man = self.manifests.get(sector.slug)
            if man is None:
                man = {"slug": sector.slug, "satellite": sector.satellite,
                       "sector": sector.sector, "label": sector.label, "bands": {}}
                self.manifests[sector.slug] = man
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
            if self.r2.put_json(sector_manifest_key(sector.slug), man,
                                CACHE_MANIFEST) and deleted:
                self.r2.delete(deleted)

    # ---- one unit -------------------------------------------------------

    def process_unit(self, u: Unit, lane: Lane) -> None:
        sector, band = u.sector, u.band
        ext = self.extents.get(sector.slug)
        if ext is None:
            return  # sector has no discovered extent yet -> nothing to render
        bbox = list(ext.bbox)
        # All meso sectors render from their rapid-scan product via the "meso"
        # hint: GOES -> CMIPM (find_file forces _pick_meso past the 12° span gate),
        # Himawari -> the ~2.5-min Target sub-scans. MESO_HIMAWARI_PRODUCT=fldk
        # reverts Himawari to the 10-min full disk (true-color is always FLDK,
        # guarded in find_file). GOES is unaffected by the flag.
        product = "meso"
        if sector.family == "himawari" and MESO_HIMAWARI_PRODUCT == "fldk":
            product = None
        # Tell /render WHICH satellite this sector's bbox came from -- the
        # picker's sub-point-distance tie-break lands antimeridian sectors on
        # the wrong satellite (Himawari "wins" a Bering Sea GOES-18 M2 box).
        sat_hint = sector_family_hint(sector)
        lane.limiter.acquire()
        _t_render = time.monotonic()
        try:
            png, headers = call_render(lane.session, bbox, band.channel,
                                       band.enhancement, product=product,
                                       satellite=sat_hint,
                                       url=lane.render_url)
        except RenderSkip as e:
            log.info("skip %s/%s: %s", sector.slug, band.key, e)
            lane.consec_fail = 0
            return
        except RenderError as e:
            lane.consec_fail += 1
            log.warning("[%s] render fail %s/%s (%d): %s", lane.name,
                        sector.slug, band.key, lane.consec_fail, e)
            if lane.consec_fail >= CIRCUIT_TRIP_FAILS:
                lane.circuit_open_until = time.monotonic() + CIRCUIT_COOLDOWN_S
                log.error("[%s] circuit OPEN: cooling down %ss",
                          lane.name, CIRCUIT_COOLDOWN_S)
            return
        lane.consec_fail = 0

        h = hashlib.sha256(png).hexdigest()
        if h == u.last_hash:
            return  # no new frame
        # Stamp the frame with what /render ACTUALLY rendered (X-Scan-Time) so
        # Himawari Target sub-scans (~2.5 min) get distinct timestamps instead of
        # collapsing onto the 10-min discovery slot; fall back to the discovered
        # scan time, then now.
        scan_hdr = (header_get(headers, "X-Scan-Time")
                    or header_get(headers, "X-Source-Time")
                    or header_get(headers, "X-Timestamp"))
        ts = (parse_iso(scan_hdr) if scan_hdr else None) or ext.scan_start or utcnow()
        # fext, not ext -- ext is the SectorExtent above.
        fext, ctype = frame_ext(headers)
        key = frame_key(sector.slug, band.key, ts, fext)
        if not self.r2.put_bytes(key, png, ctype, CACHE_FRAME):
            return  # upload failed -> do NOT touch manifest; retry next slot
        self.append_frame(sector, band, key, ts, h)
        u.last_hash = h
        log.info("[%s] uploaded %s (%d B, %s, render %.1fs)", lane.name, key,
                 len(png), header_get(headers, "X-Satellite") or "?",
                 time.monotonic() - _t_render)

    # ---- cadence --------------------------------------------------------

    def lane_cadence(self, lane: Lane) -> float:
        """Per-unit cadence for a lane. Hot = the native 60 s GOES meso scan
        target. Cold is DELIBERATELY stretched well past that so the cold
        bands (WV / true-color / SWIR) self-pace: floors at
        COLD_CADENCE_TARGET_S (default 300 s) and still widens with the rate
        budget when the lane is pointed at the public URL."""
        if lane.drain_all:
            return CADENCE_TARGET_S
        return max(COLD_CADENCE_TARGET_S,
                   len(lane.units) * lane.limiter.min_spacing)

    def _lane_loop(self, lane: Lane) -> None:
        """The lane's scheduler thread. Hot drains EVERY due unit per pass (the
        10-unit IR/IRBD fleet fits the 60 s budget at ~2.5 s/render against its
        dedicated render container); cold takes only the single most-overdue
        unit per pass, so a cold backlog self-paces to the cold container's
        real render speed. The two lanes share NOTHING but the discovered
        extents and the (locked) manifest writer -- the old single-threaded
        tick() let one ~18 s cold render push every hot unit past its 60 s
        slot, which is exactly the starvation this kills."""
        while True:
            try:
                now = time.monotonic()
                if now < lane.circuit_open_until or not self.extents:
                    time.sleep(1.0)
                    continue
                due = [u for u in lane.units if u.next_due <= now]
                if not due:
                    time.sleep(0.25)
                    continue
                if lane.drain_all:
                    due.sort(key=lambda u: u.next_due)
                else:
                    due = [min(due, key=lambda u: u.next_due)]
                for u in due:
                    if time.monotonic() < lane.circuit_open_until:
                        break
                    self.process_unit(u, lane)
                    u.next_due = time.monotonic() + self.lane_cadence(lane)
            except Exception as e:  # noqa: BLE001 - a lane must never die
                log.exception("[%s] lane error (continuing): %s", lane.name, e)
                time.sleep(5)

    # ---- health ---------------------------------------------------------

    def health_snapshot(self) -> dict:
        """Per-sector freshness + an overall healthy flag. Mirrors
        PollerEngine.health_snapshot: each sector classifies fresh/stale/failing
        via its SourceHealth; healthy is False if ANY sector is not fresh."""
        now = utcnow()
        sources: dict[str, dict] = {}
        healthy = True
        for sector in MESO_SECTORS:
            snap = self.health[sector.slug].snapshot(
                now, EXTENT_STALE_AFTER_S, FAIL_THRESHOLD)
            sources[sector.slug] = snap
            if snap["state"] != FRESH:
                healthy = False
        mono = time.monotonic()
        return {
            "poller": "meso",
            "enabled": MESO_ENABLED,
            "generated_utc": iso_z(now),
            "interval_s": SECTORS_REFRESH_S,
            "stale_after_s": EXTENT_STALE_AFTER_S,
            "fail_threshold": FAIL_THRESHOLD,
            "healthy": healthy,
            "process": process_mem_mb(),
            "sources": sources,
            # Per-lane scheduler visibility (strict hot/cold isolation): the
            # render endpoint each lane talks to, its breaker state, and how
            # late its most-overdue unit is running -- "is hot keeping its
            # 60 s promise" is readable straight off /health.
            "lanes": {
                lane.name: {
                    "render_url": lane.render_url,
                    "units": len(lane.units),
                    "consec_fail": lane.consec_fail,
                    "circuit_open": mono < lane.circuit_open_until,
                    "max_overdue_s": round(max(
                        (mono - u.next_due for u in lane.units), default=0.0), 1),
                }
                for lane in (self.hot_lane, self.cold_lane)
            },
        }

    def emit_health(self) -> None:
        """Write the health snapshot to R2 AND a local file (a compose
        healthcheck curls the local HTTP server, which serves this file).
        Emission failures never crash the loop."""
        snap = self.health_snapshot()
        _HEALTH_STATE["snapshot"] = snap
        try:
            with open(HEALTH_FILE, "w", encoding="utf-8") as f:
                json.dump(snap, f, separators=(",", ":"))
        except OSError as e:
            log.warning("health file write failed: %s", e)
        try:
            self.r2.put_json(health_key(), snap, CACHE_MANIFEST)
        except Exception as e:  # noqa: BLE001
            log.warning("health R2 write failed: %s", e)

    # ---- main loop ------------------------------------------------------

    def run(self) -> None:
        if not MESO_ENABLED:
            log.warning("MESO_ENABLED=false -- poller IDLE, writing nothing. "
                        "Emitting health only so the watcher sees a live, "
                        "intentionally-disabled poller.")
            while True:
                # Idle: still emit a (disabled) heartbeat so /health is truthful.
                self.emit_health()
                time.sleep(min(SECTORS_REFRESH_S, 60))

        log.info("meso poller starting | render hot=%s cold=%s | bucket=%s | "
                 "prefix=%s | sectors=%d | spacing=%gs | reconcile=%gs",
                 RENDER_URL, RENDER_URL_COLD, R2_BUCKET, R2_PREFIX,
                 len(MESO_SECTORS), RATE_MIN_SPACING_S, RECONCILE_S)
        # Storage-truth sync BEFORE any frame work: rebuild every manifest
        # from an R2 listing (drops phantom entries the viewer 404s on), then
        # seed the per-unit dedup hashes from the reconciled state.
        self.reconcile_manifests()
        self._last_reconcile = time.monotonic()
        for (slug, band_key), u in self.units.items():
            b = ((self.manifests.get(slug) or {}).get("bands") or {}) \
                .get(band_key) or {}
            u.last_hash = b.get("last_hash")
        # Lane scheduler threads -- started AFTER the manifests + dedup hashes
        # are seeded so neither lane can race the startup reconcile. Daemon:
        # they die with the main (discovery/health) thread.
        for lane in (self.hot_lane, self.cold_lane):
            threading.Thread(target=self._lane_loop, args=(lane,),
                             name=f"meso-{lane.name}", daemon=True).start()
        # Main thread owns discovery, reconcile, and the health heartbeat.
        while True:
            try:
                if time.monotonic() - self._last_sectors_refresh >= SECTORS_REFRESH_S \
                        or not self.extents:
                    self.refresh_extents()
                    self._last_sectors_refresh = time.monotonic()
                if time.monotonic() - self._last_reconcile >= RECONCILE_S:
                    self.reconcile_manifests()
                    self._last_reconcile = time.monotonic()
                time.sleep(2.0 if self.extents
                           else min(SECTORS_REFRESH_S, 60))
            except KeyboardInterrupt:
                log.info("shutting down")
                return
            except Exception as e:  # noqa: BLE001 - the loop must never die
                log.exception("loop error (continuing): %s", e)
                time.sleep(5)


# ---------------------------------------------------------------------------
# Tiny health HTTP server (a compose healthcheck curls GET /health on HEALTH_PORT)
# ---------------------------------------------------------------------------

_HEALTH_STATE: dict = {"snapshot": {"poller": "meso", "healthy": None,
                                    "enabled": MESO_ENABLED}}


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib handler name
        if self.path.rstrip("/") not in ("/health", ""):
            self.send_response(404)
            self.end_headers()
            return
        snap = _HEALTH_STATE.get("snapshot") or {}
        # 200 when healthy (or disabled-on-purpose); 503 when any sector is not
        # fresh, so the compose healthcheck flips the container to unhealthy.
        healthy = snap.get("healthy")
        ok = (healthy is not False)  # None (booting) / True / disabled -> 200
        body = json.dumps(snap, separators=(",", ":")).encode()
        self.send_response(200 if ok else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request logging
        return


def _start_health_server() -> None:
    """Background HTTP server exposing the latest health snapshot. Bound to all
    interfaces inside the container; compose maps/curls it. Never fatal."""
    try:
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", HEALTH_PORT), _HealthHandler)
    except OSError as e:
        log.warning("health server bind :%d failed: %s (continuing without it)",
                    HEALTH_PORT, e)
        return
    t = threading.Thread(target=srv.serve_forever, name="meso-health", daemon=True)
    t.start()
    log.info("health server on :%d (GET /health)", HEALTH_PORT)


def _validate_env() -> None:
    missing = [n for n in ("R2_ENDPOINT",) if not _env(n)]
    if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        missing.append("R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY")
    if missing:
        log.error("missing required env: %s", ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    _start_health_server()
    if MESO_ENABLED:
        _validate_env()
    MesoPoller().run()
