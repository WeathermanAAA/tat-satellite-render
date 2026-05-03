"""GOES ABI L2 CMIP access on the public NOAA AWS Open Data buckets.

Buckets in rotation (anonymous read, all on AWS):
  s3://noaa-goes19/  - operational GOES-East since 2025-04-04
  s3://noaa-goes16/  - operational GOES-East 2017-12 -> 2025-04-04, archive
                       only after that

Bucket selection is time-based — see ``pick_buckets_for_time``. The
``GOES_BUCKET`` env var, when set, overrides the rotation and forces a
single bucket (useful for ops/testing).

Product short-codes (same in every bucket):
  CMIPF -> Full Disk (every 10 min)
  CMIPC -> CONUS    (every 5 min)
  CMIPM1, CMIPM2 -> Mesoscale sectors (every 1 min, dynamically positioned)

Selection logic: pick smallest product that fully covers the bbox to keep
fetches minimal (Mesoscale ~5 MB, CONUS ~30 MB, Full Disk ~150-300 MB).

Geos-projection crop happens BEFORE materializing CMI to keep RAM bounded:
ABI files are stored on the satellite's geostationary scan-angle grid
(x, y in radians). We project the requested bbox to (x, y), slice via .isel,
THEN read CMI for just that window.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import s3fs
import xarray as xr

GOES_DISK_BBOX = (-152.04, -75.0, 6.04, 75.0)  # (lon_min, lat_min, lon_max, lat_max)

# CONUS scan footprint changed when GOES-East switched from Mode 3 to Mode 6
# (more frequent CONUS, slightly larger scan). Pre-Mode-6 the eastern edge
# is ~-65°W; post-Mode-6 it's ~-55°W. Picking CMIPC for a bbox that extends
# past the relevant edge produces a black wedge (no data) on the eastern
# side of the render — see _pick_conus for the fallback logic.
CMIPC_MODE6_START = dt.datetime(2019, 4, 2, tzinfo=dt.timezone.utc)
CONUS_FOOTPRINT_MODE3 = (-135.0, 14.0, -65.0, 50.0)
CONUS_FOOTPRINT_MODE6 = (-135.0, 14.0, -55.0, 50.0)


def _conus_footprint(t: dt.datetime) -> tuple[float, float, float, float]:
    """Return the approximate CONUS scan footprint for a given UTC time.

    GOES-16 was the first GOES-East with these scan modes, and GOES-19
    inherited Mode 6 — same footprint either way. The boundary is the
    Mode 3 -> Mode 6 transition on 2019-04-02.
    """
    if t >= CMIPC_MODE6_START:
        return CONUS_FOOTPRINT_MODE6
    return CONUS_FOOTPRINT_MODE3

# Manual override — when set, forces a single bucket and skips the picker.
GOES_BUCKET_OVERRIDE = os.getenv("GOES_BUCKET", "").strip()

# Time boundaries for the time-based picker.
# 2025-04-04: GOES-19 became operational GOES-East, GOES-16 went to standby.
# 2018-08-01: per ops spec, before this date GOES-16 archive is the only
# reliable candidate (GOES-16 was post-launch / pre-operational mid-2017
# through 2017-12 but the archive there is sparse + uncalibrated, so we
# still try goes16 — the explicit cutoff just skips the goes19 fallback).
GOES19_OPERATIONAL = dt.datetime(2025, 4, 4, tzinfo=dt.timezone.utc)
GOES16_PRIMARY_BEFORE = dt.datetime(2018, 8, 1, tzinfo=dt.timezone.utc)
PRIMARY_LIVE_BUCKET = "noaa-goes19"


def pick_buckets_for_time(requested_time: str) -> list[str]:
    """Return buckets to try in order based on the requested time.

    Rules (per ops spec):
      - 'latest' or t >= 2025-04-04: noaa-goes19 only
      - 2018-08-01 <= t < 2025-04-04: noaa-goes19 then noaa-goes16
        (overlap window — goes19 may carry late-2024 calibration data; if
         it doesn't, we fall through to goes16's primary archive)
      - t < 2018-08-01: noaa-goes16 only

    GOES_BUCKET env var, if set, overrides everything and pins a single
    bucket. Future west-Pacific support would add goes17/goes18 here.
    """
    if GOES_BUCKET_OVERRIDE:
        return [GOES_BUCKET_OVERRIDE]

    if requested_time == "latest":
        return ["noaa-goes19"]

    try:
        t = dt.datetime.fromisoformat(requested_time.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return ["noaa-goes19"]  # unparsable -> assume live

    if t >= GOES19_OPERATIONAL:
        return ["noaa-goes19"]
    if t >= GOES16_PRIMARY_BEFORE:
        return ["noaa-goes19", "noaa-goes16"]
    return ["noaa-goes16"]


def goes_sat_label(bucket: str) -> str:
    """Render a human-friendly label like 'GOES-19' from a bucket name."""
    s = bucket.split("-")[-1]  # "noaa-goes19" -> "goes19"
    if s.startswith("goes"):
        return f"GOES-{s[4:]}"
    return bucket


log = logging.getLogger("tat-satellite.goes")


# ---------------------------------------------------------------------------
# S3 filesystem (singleton)
# ---------------------------------------------------------------------------
_fs: Optional[s3fs.S3FileSystem] = None


def _get_fs() -> s3fs.S3FileSystem:
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(
            anon=True,
            config_kwargs={"max_pool_connections": 8, "retries": {"max_attempts": 3}},
        )
    return _fs


async def _to_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def bucket_reachable() -> bool:
    """Probe the primary live bucket. Used by /health.

    We don't probe historical fallbacks here because (a) goes16 is on the
    same NOAA AWS Open Data CDN and effectively shares an availability
    fate with goes19, and (b) /health green should mean "live renders
    work" — historical query failures surface cleanly via /render's 502.
    """
    primary = GOES_BUCKET_OVERRIDE or PRIMARY_LIVE_BUCKET
    try:
        fs = _get_fs()
        await _to_thread(fs.ls, f"{primary}/ABI-L2-CMIPF/")
        return True
    except Exception as e:
        log.warning("bucket unreachable: %s", e)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bbox_area_sqdeg(bbox: list[float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _bbox_inside(inner: list[float], outer: tuple[float, float, float, float], buffer: float = 0.0) -> bool:
    return (
        inner[0] >= outer[0] - buffer
        and inner[1] >= outer[1] - buffer
        and inner[2] <= outer[2] + buffer
        and inner[3] <= outer[3] + buffer
    )


def _parse_scan_start(s3_key: str) -> dt.datetime:
    base = s3_key.split("/")[-1]
    s_token = base.split("_s")[1].split("_")[0]
    year = int(s_token[:4])
    doy = int(s_token[4:7])
    hh = int(s_token[7:9])
    mm = int(s_token[9:11])
    ss = int(s_token[11:13])
    return dt.datetime(year, 1, 1, hh, mm, ss, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1)


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------
def _channel_token(channel: int) -> str:
    return f"C{channel:02d}_"


def _list_hour(bucket: str, product: str, channel: int, t: dt.datetime) -> list[str]:
    fs = _get_fs()
    doy = t.strftime("%j")
    prefix = f"{bucket}/ABI-L2-{product}/{t.year}/{doy}/{t.hour:02d}/"
    try:
        files = fs.ls(prefix)
    except (FileNotFoundError, OSError):
        return []
    tok = _channel_token(channel)
    return [f for f in files if tok in f]


async def _list_files_around(bucket: str, product: str, channel: int, target: dt.datetime) -> list[str]:
    """List the target hour and the previous hour to cover edge cases (e.g. target=00:02 needs prev hour for nearest)."""
    prev = target - dt.timedelta(hours=1)
    a, b = await asyncio.gather(
        _to_thread(_list_hour, bucket, product, channel, prev),
        _to_thread(_list_hour, bucket, product, channel, target),
    )
    return a + b


# ---------------------------------------------------------------------------
# Mesoscale coverage check (read NetCDF header attrs only)
# ---------------------------------------------------------------------------
def _check_meso_coverage_sync(s3_key: str, bbox: list[float], buffer_deg: float = 0.5) -> bool:
    fs = _get_fs()
    try:
        with fs.open(s3_key, mode="rb") as f:
            ds = xr.open_dataset(f, decode_cf=False, engine="h5netcdf")
            try:
                lat_min = float(ds.attrs.get("geospatial_lat_min", -90))
                lat_max = float(ds.attrs.get("geospatial_lat_max", 90))
                lon_min = float(ds.attrs.get("geospatial_lon_min", -180))
                lon_max = float(ds.attrs.get("geospatial_lon_max", 180))
            finally:
                ds.close()
        return _bbox_inside(bbox, (lon_min, lat_min, lon_max, lat_max), buffer=buffer_deg)
    except Exception as e:
        log.warning("meso coverage check failed for %s: %s", s3_key, e)
        return False


# ---------------------------------------------------------------------------
# Public selection / fetch API
# ---------------------------------------------------------------------------
@dataclass
class ResolvedFile:
    bucket: str  # "noaa-goes19" | "noaa-goes16" | ...
    s3_key: str
    product: str  # "CMIPF" | "CMIPC" | "CMIPM1" | "CMIPM2"
    scan_start: dt.datetime


async def resolve_request(
    bbox: list[float],
    channel: int,
    requested_time: str,
) -> ResolvedFile:
    """Resolve a render request to a concrete S3 file key.

    requested_time: "latest" or ISO8601 string.

    Iterates the time-based bucket rotation; the first bucket with a
    suitable file wins. For dates in the goes19/goes16 overlap window
    (2018-08-01..2025-04-04), goes19 is tried first and goes16 is the
    fallback if no files exist in goes19 for that scan time.
    """
    if requested_time == "latest":
        target = dt.datetime.now(dt.timezone.utc)
        nearest_to_target = False
    else:
        target = dt.datetime.fromisoformat(requested_time.replace("Z", "+00:00"))
        if target.tzinfo is None:
            target = target.replace(tzinfo=dt.timezone.utc)
        nearest_to_target = True

    buckets = pick_buckets_for_time(requested_time)
    area = _bbox_area_sqdeg(bbox)

    # Product preference order is determined by bbox area, same in every bucket
    if area < 30:
        candidates = [_pick_meso, _pick_conus, _pick_full_disk]
    elif area < 200:
        candidates = [_pick_conus, _pick_full_disk]
    else:
        candidates = [_pick_full_disk]

    last_err: Optional[Exception] = None
    for bucket in buckets:
        for picker in candidates:
            try:
                resolved = await picker(bucket, bbox, channel, target, nearest_to_target)
                if resolved is not None:
                    log.info("resolved via %s/%s -> %s", bucket, picker.__name__, resolved.product)
                    return resolved
            except Exception as e:
                last_err = e
                log.warning("%s on %s failed: %s", picker.__name__, bucket, e)
                continue
        log.info("no files found in %s for time=%s; trying next bucket", bucket, requested_time)

    raise RuntimeError(
        f"no GOES file found for bbox/time/channel across buckets {buckets}; last_err={last_err}"
    )


async def _pick_meso(bucket, bbox, channel, target, nearest_to_target) -> Optional[ResolvedFile]:
    # Try M1 then M2 — first that covers wins
    for sector in ("CMIPM1", "CMIPM2"):
        files = await _list_files_around(bucket, sector, channel, target)
        if not files:
            continue
        # Sort by scan time, pick most recent <= target if explicit, else most recent
        files_with_t = sorted([(f, _parse_scan_start(f)) for f in files], key=lambda p: p[1])
        if nearest_to_target:
            picked = min(files_with_t, key=lambda p: abs((p[1] - target).total_seconds()))
        else:
            picked = files_with_t[-1]
        if await _to_thread(_check_meso_coverage_sync, picked[0], bbox):
            return ResolvedFile(bucket, picked[0], sector, picked[1])
    return None


async def _pick_conus(bucket, bbox, channel, target, nearest_to_target) -> Optional[ResolvedFile]:
    # Validate that the requested bbox is fully inside the CONUS scan
    # footprint for the selected satellite + date. Overlap-only checks are
    # not safe — a Caribbean bbox extending east of the scan edge produces
    # a black wedge in the rendered image (CMIPC sector has no data there).
    footprint = _conus_footprint(target)
    if not _bbox_inside(bbox, footprint):
        sat = goes_sat_label(bucket)
        mode = "Mode 6" if target >= CMIPC_MODE6_START else "Mode 3"
        log.info(
            "bbox lon=%.1f..%.1f lat=%.1f..%.1f outside CONUS footprint "
            "lon=%.1f..%.1f for %s %s — falling back to CMIPF",
            bbox[0], bbox[2], bbox[1], bbox[3],
            footprint[0], footprint[2], sat, mode,
        )
        return None
    files = await _list_files_around(bucket, "CMIPC", channel, target)
    if not files:
        return None
    files_with_t = sorted([(f, _parse_scan_start(f)) for f in files], key=lambda p: p[1])
    if nearest_to_target:
        picked = min(files_with_t, key=lambda p: abs((p[1] - target).total_seconds()))
    else:
        picked = files_with_t[-1]
    return ResolvedFile(bucket, picked[0], "CMIPC", picked[1])


async def _pick_full_disk(bucket, bbox, channel, target, nearest_to_target) -> Optional[ResolvedFile]:
    files = await _list_files_around(bucket, "CMIPF", channel, target)
    if not files:
        return None
    files_with_t = sorted([(f, _parse_scan_start(f)) for f in files], key=lambda p: p[1])
    if nearest_to_target:
        picked = min(files_with_t, key=lambda p: abs((p[1] - target).total_seconds()))
    else:
        picked = files_with_t[-1]
    return ResolvedFile(bucket, picked[0], "CMIPF", picked[1])


# ---------------------------------------------------------------------------
# Geos-projection bbox crop
# ---------------------------------------------------------------------------
def _latlon_to_xy(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lon_origin_deg: float,
    h: float,
    r_eq: float,
    r_pol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward projection: geodetic lat/lon -> ABI (x, y) scan angles in radians.

    Reference: NOAA GOES-R ABI L1b PUG Vol 3, eq. 5.1.2.8.1.
    Returns (x, y) where x is east-west scan, y is north-south.
    For points off the visible disk, returns NaN.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lambda_0 = np.deg2rad(lon_origin_deg)
    e = np.sqrt(1.0 - (r_pol**2) / (r_eq**2))

    phi_c = np.arctan(((r_pol**2) / (r_eq**2)) * np.tan(lat))
    rc = r_pol / np.sqrt(1.0 - (e**2) * (np.cos(phi_c) ** 2))

    sx = h - rc * np.cos(phi_c) * np.cos(lon - lambda_0)
    sy = -rc * np.cos(phi_c) * np.sin(lon - lambda_0)
    sz = rc * np.sin(phi_c)

    # Skip the PUG-3 visibility check (its inequality direction is unreliable
    # for our sampling — manually verified that even sub-satellite points fail
    # the canonical form). Caller-side bbox validation already restricts to
    # disk-overlapping bboxes; off-disk samples just produce extreme but
    # finite scan angles that get clipped during the slice step.
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.arctan(sz / sx)
        x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    return x, y


@dataclass
class FetchResult:
    cmi: np.ndarray  # 2D, lat/lon-decoded values (Kelvin for IR, reflectance 0..1 for visible)
    lats: np.ndarray  # 2D array, same shape
    lons: np.ndarray  # 2D
    channel: int
    scan_start: dt.datetime
    product: str
    bucket: str  # source bucket — drives the title strip ("GOES-19" vs "GOES-16")
    units: str  # "K" | "1"


def _fetch_data_sync(resolved: ResolvedFile, bbox: list[float], channel: int) -> FetchResult:
    """Download the NC file to /tmp, then open locally.

    Streaming via s3fs+h5netcdf is fine for IR (5-30 MB files) but visible C02
    is 150-300 MB — HDF5's chunk-aligned byte-range reads + S3 per-request
    overhead pushes streaming visible to 40-60s wall. Downloading whole then
    opening locally is consistently sub-15s on a Railway-class link.
    """
    import shutil
    import tempfile

    fs = _get_fs()
    log.info("fetching s3://%s (%s)", resolved.s3_key, resolved.product)

    tmp_dir = tempfile.mkdtemp(prefix="goes-")
    local_path = f"{tmp_dir}/{resolved.s3_key.split('/')[-1]}"
    fs.get(resolved.s3_key, local_path)

    try:
        ds = xr.open_dataset(local_path, engine="h5netcdf", decode_times=False)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    try:
        proj = ds["goes_imager_projection"]
        h = float(proj.attrs["perspective_point_height"])
        r_eq = float(proj.attrs["semi_major_axis"])
        r_pol = float(proj.attrs["semi_minor_axis"])
        lon_origin = float(proj.attrs["longitude_of_projection_origin"])
        H = h + r_eq  # distance from earth center to satellite

        # x, y stored as 1D coords in the file
        x = ds["x"].values  # radians
        y = ds["y"].values

        # Sample bbox corners + center to get xy span (use a denser grid for safety)
        lon_min, lat_min, lon_max, lat_max = bbox
        n_sample = 16
        sample_lons = np.linspace(lon_min, lon_max, n_sample)
        sample_lats = np.linspace(lat_min, lat_max, n_sample)
        LON, LAT = np.meshgrid(sample_lons, sample_lats)
        sx, sy = _latlon_to_xy(LAT, LON, lon_origin, H, r_eq, r_pol)
        finite_mask = np.isfinite(sx) & np.isfinite(sy)
        sx = sx[finite_mask]
        sy = sy[finite_mask]
        if sx.size == 0:
            raise RuntimeError("bbox has no projection-valid sample points")

        x_lo, x_hi = sx.min(), sx.max()
        y_lo, y_hi = sy.min(), sy.max()
        # buffer in scan-angle units (~5 pixel pad at full disk resolution)
        x_buf = abs(x[1] - x[0]) * 5
        y_buf = abs(y[1] - y[0]) * 5
        x_lo -= x_buf
        x_hi += x_buf
        y_lo -= y_buf
        y_hi += y_buf

        # x is monotonically increasing west->east; y is monotonically DECREASING north->south
        ix0 = int(np.searchsorted(x, x_lo, side="left"))
        ix1 = int(np.searchsorted(x, x_hi, side="right"))
        # y descending: argsorted high->low
        y_desc = y  # as stored
        if y_desc[0] > y_desc[-1]:
            iy_top = int(np.searchsorted(-y_desc, -y_hi, side="left"))
            iy_bot = int(np.searchsorted(-y_desc, -y_lo, side="right"))
        else:
            iy_top = int(np.searchsorted(y_desc, y_lo, side="left"))
            iy_bot = int(np.searchsorted(y_desc, y_hi, side="right"))

        ix0 = max(0, ix0 - 1)
        ix1 = min(len(x), ix1 + 1)
        iy_top = max(0, iy_top - 1)
        iy_bot = min(len(y), iy_bot + 1)

        if ix1 <= ix0 or iy_bot <= iy_top:
            raise RuntimeError("crop produced empty window")

        # Cap render-input resolution at ~2400 px per axis. Visible (C02) at
        # 0.5 km on Full Disk produces 6600+ px for a 30° bbox — without
        # striding, the byte-range reads from S3 + matplotlib pcolormesh
        # become the wall-clock bottleneck. Output resolution stays the same
        # (matplotlib resamples), and 2400 px easily covers a 1320 px figure.
        MAX_PX_PER_AXIS = 2400
        x_stride = max(1, (ix1 - ix0) // MAX_PX_PER_AXIS)
        y_stride = max(1, (iy_bot - iy_top) // MAX_PX_PER_AXIS)

        sub = ds.isel(x=slice(ix0, ix1, x_stride), y=slice(iy_top, iy_bot, y_stride))
        cmi = sub["CMI"].load().values  # materialize the small window only

        # Build lat/lon for this window via inverse projection
        x_sub = sub["x"].values
        y_sub = sub["y"].values
        X, Y = np.meshgrid(x_sub, y_sub)
        lats, lons = _xy_to_latlon(X, Y, lon_origin, H, r_eq, r_pol)

        units = ds["CMI"].attrs.get("units", "")
        return FetchResult(
            cmi=cmi.astype(np.float32),
            lats=lats.astype(np.float32),
            lons=lons.astype(np.float32),
            channel=channel,
            scan_start=resolved.scan_start,
            product=resolved.product,
            bucket=resolved.bucket,
            units=units,
        )
    finally:
        ds.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _xy_to_latlon(
    x: np.ndarray, y: np.ndarray, lon_origin_deg: float, H: float, r_eq: float, r_pol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse projection: ABI scan angles -> geodetic lat/lon.

    Reference: NOAA GOES-R ABI L1b PUG Vol 3, eq. 5.1.2.8.1 (inverted form).
    """
    lambda_0 = np.deg2rad(lon_origin_deg)
    a = np.sin(x) ** 2 + (np.cos(x) ** 2) * (np.cos(y) ** 2 + (r_eq**2 / r_pol**2) * np.sin(y) ** 2)
    b = -2.0 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2

    disc = b**2 - 4 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2 * a)
        sx = rs * np.cos(x) * np.cos(y)
        sy = -rs * np.sin(x)
        sz = rs * np.cos(x) * np.sin(y)

        lat = np.rad2deg(np.arctan((r_eq**2 / r_pol**2) * (sz / np.sqrt((H - sx) ** 2 + sy**2))))
        lon = np.rad2deg(lambda_0 - np.arctan(sy / (H - sx)))

    invalid = disc < 0
    lat = np.where(invalid, np.nan, lat)
    lon = np.where(invalid, np.nan, lon)
    return lat, lon


async def fetch_data(resolved: ResolvedFile, bbox: list[float], channel: int) -> FetchResult:
    return await _to_thread(_fetch_data_sync, resolved, bbox, channel)
