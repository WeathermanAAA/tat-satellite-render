"""Satellite-agnostic abstraction layer.

A ``Satellite`` knows how to:
  - resolve which physical hardware was operational at a given time,
  - check whether a bbox overlaps its visible disk,
  - find the smallest-product S3 file for a (time, generic-channel, bbox) request,
  - open that file and project it onto a regular lat/lon grid.

Generic channels (``clean_ir``, ``wv_upper``, ...) decouple the public API
from instrument-specific band numbers. A1 ships ``GOES_EAST``; A2 will
add Himawari (and later METEOSAT) plugged into the same interface.

This file replaces the legacy ``goes.py``. All GOES-East CMIPM/CMIPC/CMIPF
picker logic, CONUS scan-footprint validation, and ABI geos crop are
preserved unchanged inside ``GOESEastSatellite`` — behavior parity is the
primary goal of this refactor.
"""

from __future__ import annotations

import abc
import asyncio
import datetime as dt
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import s3fs
import xarray as xr

log = logging.getLogger("tat-satellite.satellites")


# ---------------------------------------------------------------------------
# Generic channel definitions
# ---------------------------------------------------------------------------
GENERIC_CHANNELS: dict[str, dict] = {
    "visible_red":   {"goes": 2,  "wavelength": "0.64 µm",  "label": "Visible (red)",         "native_km": 0.5},
    "shortwave_ir":  {"goes": 7,  "wavelength": "3.9 µm",   "label": "Shortwave IR",          "native_km": 2.0},
    "wv_upper":      {"goes": 8,  "wavelength": "6.2 µm",   "label": "Upper-tropospheric WV", "native_km": 2.0},
    "wv_lower":      {"goes": 10, "wavelength": "7.3 µm",   "label": "Lower-tropospheric WV", "native_km": 2.0},
    "clean_ir":      {"goes": 13, "wavelength": "10.4 µm",  "label": "Clean longwave IR",     "native_km": 2.0},
    "ir_window":     {"goes": 14, "wavelength": "11.2 µm",  "label": "IR window",             "native_km": 2.0},
}


_GOES_BAND_TO_GENERIC: dict[int, str] = {
    spec["goes"]: name for name, spec in GENERIC_CHANNELS.items()
}


def goes_band_to_generic(band: int) -> Optional[str]:
    return _GOES_BAND_TO_GENERIC.get(band)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CoverageError(Exception):
    """No active satellite can see the requested bbox."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ResolvedSatellite:
    name: str            # "GOES-19", "GOES-16", later "Himawari-9"/"Himawari-8"
    bucket: str          # "noaa-goes19", "noaa-goes16", ...
    sub_sat_lon: float   # signed degrees east


@dataclass
class ResolvedFile:
    bucket: str
    s3_key: str
    product: str  # GOES: "CMIPF" | "CMIPC" | "CMIPM1" | "CMIPM2"
    scan_start: dt.datetime
    sat_name: str
    sub_sat_lon: float


@dataclass
class FetchResult:
    cmi: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    channel: int            # native band number — kept for back-compat with render.py
    generic_channel: str    # e.g. "clean_ir"
    scan_start: dt.datetime
    product: str
    bucket: str             # source bucket — drives the title strip ("GOES-19" vs "GOES-16")
    sat_name: str
    sub_sat_lon: float
    units: str              # "K" | "1"


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


def _bbox_overlaps(a: list[float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _bbox_inside(inner: list[float], outer: tuple[float, float, float, float], buffer: float = 0.0) -> bool:
    return (
        inner[0] >= outer[0] - buffer
        and inner[1] >= outer[1] - buffer
        and inner[2] <= outer[2] + buffer
        and inner[3] <= outer[3] + buffer
    )


# ---------------------------------------------------------------------------
# Antimeridian-safe bbox center
# ---------------------------------------------------------------------------
def antimeridian_safe_center_lon(bbox: list[float]) -> float:
    """Return the bbox's center longitude, correctly handling ±180° crossings.

    Convention: a bbox that crosses ±180° has e < w (e.g. [170, -30, -170, -20]).
    """
    w, e = bbox[0], bbox[2]
    if e < w:  # crosses ±180°
        span = (e + 360) - w
        c = w + span / 2
        return ((c + 180) % 360) - 180
    return (w + e) / 2


# ---------------------------------------------------------------------------
# Satellite ABC
# ---------------------------------------------------------------------------
class Satellite(abc.ABC):
    family: ClassVar[str]
    sensor: ClassVar[str]
    generic_to_band: ClassVar[dict[str, int]]
    disk_bbox: ClassVar[tuple[float, float, float, float]]
    primary_live_bucket: ClassVar[str]

    def can_see(self, bbox: list[float], time: dt.datetime) -> bool:
        return _bbox_overlaps(bbox, self.disk_bbox)

    @abc.abstractmethod
    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        """Return which physical hardware was operational at ``time``."""

    @abc.abstractmethod
    async def find_file(
        self,
        time: dt.datetime,
        generic_channel: str,
        bbox: list[float],
        nearest_to_target: bool,
    ) -> ResolvedFile:
        """Locate the smallest product covering ``bbox`` at ``time``."""

    @abc.abstractmethod
    def open(self, resolved: ResolvedFile) -> tuple[xr.Dataset, str]:
        """Download + open the file. Returns (dataset, tmp_dir).

        ``tmp_dir`` MUST be removed by the caller after the dataset is closed
        — typically by ``fetch``.
        """

    @abc.abstractmethod
    def project_to_latlon(
        self,
        ds: xr.Dataset,
        bbox: list[float],
        resolved: ResolvedFile,
        generic_channel: str,
    ) -> FetchResult:
        """Geos-crop + inverse-project to lat/lon. Loads only the cropped window."""

    async def fetch(
        self,
        resolved: ResolvedFile,
        bbox: list[float],
        generic_channel: str,
    ) -> FetchResult:
        return await _to_thread(self._fetch_sync, resolved, bbox, generic_channel)

    def _fetch_sync(
        self,
        resolved: ResolvedFile,
        bbox: list[float],
        generic_channel: str,
    ) -> FetchResult:
        ds, tmp_dir = self.open(resolved)
        try:
            return self.project_to_latlon(ds, bbox, resolved, generic_channel)
        finally:
            ds.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# GOES-East implementation
# ---------------------------------------------------------------------------
GOES_DISK_BBOX = (-152.04, -75.0, 6.04, 75.0)  # (lon_min, lat_min, lon_max, lat_max)

# CONUS scan footprint changed when GOES-East switched from Mode 3 to Mode 6
# (more frequent CONUS, slightly larger scan). Pre-Mode-6 the eastern edge
# is ~-65°W; post-Mode-6 it's ~-55°W. Picking CMIPC for a bbox that extends
# past the relevant edge produces a black wedge (no data) on the eastern
# side of the render — see _pick_conus for the fallback logic.
CMIPC_MODE6_START = dt.datetime(2019, 4, 2, tzinfo=dt.timezone.utc)
CONUS_FOOTPRINT_MODE3 = (-135.0, 14.0, -65.0, 50.0)
CONUS_FOOTPRINT_MODE6 = (-135.0, 14.0, -55.0, 50.0)

GOES_BUCKET_OVERRIDE = os.getenv("GOES_BUCKET", "").strip()

# Time boundaries for the time-based picker.
# 2025-04-04: GOES-19 became operational GOES-East, GOES-16 went to standby.
# 2018-08-01: per ops spec, before this date GOES-16 archive is the only
# reliable candidate.
GOES19_OPERATIONAL = dt.datetime(2025, 4, 4, tzinfo=dt.timezone.utc)
GOES16_PRIMARY_BEFORE = dt.datetime(2018, 8, 1, tzinfo=dt.timezone.utc)

PRIMARY_LIVE_BUCKET = "noaa-goes19"
GOES_EAST_SUB_SAT_LON = -75.2

# CMIPM (mesoscale) sectors are ~1000 km on a side — viable only for tightly
# zoomed renders. Anything larger skips meso entirely and starts at CMIPC.
MESO_PER_AXIS_DEG_MAX = 12.0


def goes_sat_label(bucket: str) -> str:
    """Render a human-friendly label like 'GOES-19' from a bucket name."""
    s = bucket.split("-")[-1]  # "noaa-goes19" -> "goes19"
    if s.startswith("goes"):
        return f"GOES-{s[4:]}"
    return bucket


def _conus_footprint(t: dt.datetime) -> tuple[float, float, float, float]:
    if t >= CMIPC_MODE6_START:
        return CONUS_FOOTPRINT_MODE6
    return CONUS_FOOTPRINT_MODE3


def _channel_token(channel: int) -> str:
    return f"C{channel:02d}_"


def _parse_scan_start(s3_key: str) -> dt.datetime:
    base = s3_key.split("/")[-1]
    s_token = base.split("_s")[1].split("_")[0]
    year = int(s_token[:4])
    doy = int(s_token[4:7])
    hh = int(s_token[7:9])
    mm = int(s_token[9:11])
    ss = int(s_token[11:13])
    return dt.datetime(year, 1, 1, hh, mm, ss, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1)


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


def _pick_buckets_for_time_dt(t: dt.datetime) -> list[str]:
    """Bucket fallback chain for a parsed UTC datetime.

    Rules (per ops spec):
      - t >= 2025-04-04: noaa-goes19 only
      - 2018-08-01 <= t < 2025-04-04: noaa-goes19 then noaa-goes16
      - t < 2018-08-01: noaa-goes16 only
    """
    if GOES_BUCKET_OVERRIDE:
        return [GOES_BUCKET_OVERRIDE]
    if t >= GOES19_OPERATIONAL:
        return ["noaa-goes19"]
    if t >= GOES16_PRIMARY_BEFORE:
        return ["noaa-goes19", "noaa-goes16"]
    return ["noaa-goes16"]


def pick_buckets_for_time(requested_time: str) -> list[str]:
    """String form of the bucket picker — used by /health for ops visibility."""
    if GOES_BUCKET_OVERRIDE:
        return [GOES_BUCKET_OVERRIDE]
    if requested_time == "latest":
        return ["noaa-goes19"]
    try:
        t = dt.datetime.fromisoformat(requested_time.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return ["noaa-goes19"]
    return _pick_buckets_for_time_dt(t)


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

    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.arctan(sz / sx)
        x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    return x, y


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


class GOESEastSatellite(Satellite):
    family = "GOES-East"
    sensor = "ABI"
    generic_to_band = {
        "visible_red": 2,
        "shortwave_ir": 7,
        "wv_upper": 8,
        "wv_lower": 10,
        "clean_ir": 13,
        "ir_window": 14,
    }
    disk_bbox = GOES_DISK_BBOX
    primary_live_bucket = PRIMARY_LIVE_BUCKET

    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        """Return the primary operational hardware at ``time``.

        The actually-fetched satellite may differ during the goes19/goes16
        overlap window (2018-08-01..2025-04-04) if the primary doesn't have
        the file — that fallback is handled inside ``find_file`` and reflected
        on the ``ResolvedFile`` it returns. ``resolve()`` itself reports the
        ideal/primary hardware for the moment.
        """
        if GOES_BUCKET_OVERRIDE:
            bucket = GOES_BUCKET_OVERRIDE
        elif time >= GOES19_OPERATIONAL:
            bucket = "noaa-goes19"
        else:
            bucket = "noaa-goes16"
        return ResolvedSatellite(
            name=goes_sat_label(bucket),
            bucket=bucket,
            sub_sat_lon=GOES_EAST_SUB_SAT_LON,
        )

    async def bucket_reachable(self) -> bool:
        """Probe the primary live bucket. Used by /health.

        We don't probe historical fallbacks here because (a) goes16 is on the
        same NOAA AWS Open Data CDN and effectively shares an availability
        fate with goes19, and (b) /health green should mean "live renders
        work" — historical query failures surface cleanly via /render's 502.
        """
        primary = GOES_BUCKET_OVERRIDE or self.primary_live_bucket
        try:
            fs = _get_fs()
            await _to_thread(fs.ls, f"{primary}/ABI-L2-CMIPF/")
            return True
        except Exception as e:
            log.warning("bucket unreachable: %s", e)
            return False

    async def find_file(
        self,
        time: dt.datetime,
        generic_channel: str,
        bbox: list[float],
        nearest_to_target: bool,
    ) -> ResolvedFile:
        if generic_channel not in self.generic_to_band:
            raise ValueError(
                f"unknown generic channel for {self.family}: {generic_channel!r}"
            )
        band = self.generic_to_band[generic_channel]

        buckets = _pick_buckets_for_time_dt(time)
        lon_w = bbox[2] - bbox[0]
        lat_h = bbox[3] - bbox[1]

        # Product preference: smallest sector that could plausibly cover
        # the bbox. CMIPM is only viable for ≤12°×12°; anything larger
        # starts at CMIPC, whose internal footprint check falls through to
        # CMIPF when the bbox lies outside the CONUS scan footprint.
        if lon_w <= MESO_PER_AXIS_DEG_MAX and lat_h <= MESO_PER_AXIS_DEG_MAX:
            candidates = [self._pick_meso, self._pick_conus, self._pick_full_disk]
        else:
            candidates = [self._pick_conus, self._pick_full_disk]

        last_err: Optional[Exception] = None
        for bucket in buckets:
            for picker in candidates:
                try:
                    resolved = await picker(bucket, bbox, band, time, nearest_to_target)
                    if resolved is not None:
                        log.info(
                            "resolved via %s/%s -> %s", bucket, picker.__name__, resolved.product
                        )
                        return resolved
                except Exception as e:
                    last_err = e
                    log.warning("%s on %s failed: %s", picker.__name__, bucket, e)
                    continue
            log.info("no files found in %s for time=%s; trying next bucket", bucket, time)

        raise RuntimeError(
            f"no GOES file found for bbox/time/channel across buckets {buckets}; last_err={last_err}"
        )

    async def _pick_meso(self, bucket, bbox, channel, target, nearest_to_target):
        # Try M1 then M2 — first that covers wins
        for sector in ("CMIPM1", "CMIPM2"):
            files = await _list_files_around(bucket, sector, channel, target)
            if not files:
                continue
            files_with_t = sorted([(f, _parse_scan_start(f)) for f in files], key=lambda p: p[1])
            if nearest_to_target:
                picked = min(files_with_t, key=lambda p: abs((p[1] - target).total_seconds()))
            else:
                picked = files_with_t[-1]
            if await _to_thread(_check_meso_coverage_sync, picked[0], bbox):
                return self._make_resolved(bucket, picked[0], sector, picked[1])
        return None

    async def _pick_conus(self, bucket, bbox, channel, target, nearest_to_target):
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
        return self._make_resolved(bucket, picked[0], "CMIPC", picked[1])

    async def _pick_full_disk(self, bucket, bbox, channel, target, nearest_to_target):
        files = await _list_files_around(bucket, "CMIPF", channel, target)
        if not files:
            return None
        files_with_t = sorted([(f, _parse_scan_start(f)) for f in files], key=lambda p: p[1])
        if nearest_to_target:
            picked = min(files_with_t, key=lambda p: abs((p[1] - target).total_seconds()))
        else:
            picked = files_with_t[-1]
        return self._make_resolved(bucket, picked[0], "CMIPF", picked[1])

    def _make_resolved(self, bucket: str, s3_key: str, product: str, scan_start: dt.datetime) -> ResolvedFile:
        # Both GOES-19 and GOES-16, in the time ranges we'd ever resolve them
        # for this family, sit at the East slot (~-75.2°). GOES-16 only began
        # drifting after handover, and we never query goes16 for post-handover
        # times. So a single sub-sat-lon constant is correct here.
        return ResolvedFile(
            bucket=bucket,
            s3_key=s3_key,
            product=product,
            scan_start=scan_start,
            sat_name=goes_sat_label(bucket),
            sub_sat_lon=GOES_EAST_SUB_SAT_LON,
        )

    def open(self, resolved: ResolvedFile) -> tuple[xr.Dataset, str]:
        """Download the NC file to /tmp, then open locally.

        Streaming via s3fs+h5netcdf is fine for IR (5-30 MB files) but visible
        C02 is 150-300 MB — HDF5's chunk-aligned byte-range reads + S3
        per-request overhead pushes streaming visible to 40-60s wall.
        Downloading whole then opening locally is consistently sub-15s on a
        Railway-class link.
        """
        fs = _get_fs()
        log.info("fetching s3://%s (%s)", resolved.s3_key, resolved.product)
        tmp_dir = tempfile.mkdtemp(prefix="goes-")
        local_path = f"{tmp_dir}/{resolved.s3_key.split('/')[-1]}"
        try:
            fs.get(resolved.s3_key, local_path)
            ds = xr.open_dataset(local_path, engine="h5netcdf", decode_times=False)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return ds, tmp_dir

    def project_to_latlon(
        self,
        ds: xr.Dataset,
        bbox: list[float],
        resolved: ResolvedFile,
        generic_channel: str,
    ) -> FetchResult:
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
            channel=self.generic_to_band[generic_channel],
            generic_channel=generic_channel,
            scan_start=resolved.scan_start,
            product=resolved.product,
            bucket=resolved.bucket,
            sat_name=resolved.sat_name,
            sub_sat_lon=resolved.sub_sat_lon,
            units=units,
        )


# ---------------------------------------------------------------------------
# Singletons + picker
# ---------------------------------------------------------------------------
GOES_EAST = GOESEastSatellite()

# Order matters for ties in pick_satellite: candidates are filtered by can_see
# then sorted by |sub_sat_lon - center_lon|. With one entry today this is moot,
# but A2 will append HIMAWARI_PACIFIC here.
ALL_SATELLITES: list[Satellite] = [GOES_EAST]


def pick_satellite(bbox: list[float], time: dt.datetime) -> Satellite:
    """Pick the best satellite for ``bbox`` at ``time``.

    Filter by visible-disk overlap, then break ties by minimum
    |sub-sat-lon − bbox-center-lon|.

    Raises ``CoverageError`` if no satellite can see the bbox.
    """
    center_lon = antimeridian_safe_center_lon(bbox)
    candidates = [s for s in ALL_SATELLITES if s.can_see(bbox, time)]
    if not candidates:
        raise CoverageError(
            f"bbox center {center_lon:.1f}° not visible from any active satellite. "
            f"GOES-East: -135° to -5°. Western Pacific (Himawari) coming soon."
        )
    resolved = [(s, s.resolve(time)) for s in candidates]
    return min(resolved, key=lambda pair: abs(pair[1].sub_sat_lon - center_lon))[0]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def parse_request_time(requested_time: str) -> tuple[dt.datetime, bool]:
    """Parse the API ``time`` string. Returns (utc_dt, is_explicit_target).

    ``"latest"`` -> (now_utc, False). ISO8601 (with or without Z) -> (parsed, True).
    """
    if requested_time == "latest":
        return dt.datetime.now(dt.timezone.utc), False
    t = dt.datetime.fromisoformat(requested_time.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t, True


# ---------------------------------------------------------------------------
# Module-level convenience for legacy callers
# ---------------------------------------------------------------------------
async def bucket_reachable() -> bool:
    """GOES-East /health probe — preserved as a top-level for app.py."""
    return await GOES_EAST.bucket_reachable()
