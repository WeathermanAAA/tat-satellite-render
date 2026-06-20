"""Satellite-agnostic abstraction layer.
A ``Satellite`` knows how to:
  - resolve which physical hardware was operational at a given time,
  - check whether a bbox overlaps its visible disk,
  - find the smallest-product S3 file for a (time, generic-channel, bbox) request,
  - open that file and project it onto a regular lat/lon grid.
Generic channels (``clean_ir``, ``wv_upper``, ...) decouple the public API
from instrument-specific band numbers. A1 shipped ``GOES_EAST``; A2 adds
``HIMAWARI_PACIFIC`` (H-9 since 2022-12-13, H-8 archive back to 2017-01-01).
This file replaces the legacy ``goes.py``. All GOES-East CMIPM/CMIPC/CMIPF
picker logic, CONUS scan-footprint validation, and ABI geos crop are
preserved unchanged inside ``GOESEastSatellite``.
Himawari archive cutoff
-----------------------
``noaa-himawari8`` AWS Open Data: earliest verified data 2017-01-01 00:00 UTC
(spot-checked 2025-11). NOAA's bucket distributes two segment layouts:
  - Older H-8 timestamps (Yutu 2018, Hagibis 2019): single-file repack with
    ``S0101`` suffix — one file per band per timestep.
  - Recent H-8 + all H-9: native 10-segment FLDK with ``S0110`` ... ``S1010``.
The HSD reader (``vendor/ahi_hsd.py``) is layout-agnostic — it reads
``total_segments`` and ``first_line_number`` from each segment's Block #7
and stitches accordingly.
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
    "visible_red":   {"goes": 2,  "ahi": 3,  "wavelength": "0.64 µm",  "label": "Visible (red)",         "native_km": 0.5},
    # Blue + veggie/NIR back the true-color recipe. They're real generic
    # channels (selectable + downsampled like any other) but the dropdown
    # only surfaces them via the `true_color` product, not as standalone
    # grayscale options. ABI green is synthesized (no native band); AHI has a
    # native green (band 2) handled inside its fetch_true_color, so green is
    # deliberately NOT a generic channel (it can't map to a GOES band).
    "visible_blue":  {"goes": 1,  "ahi": 1,  "wavelength": "0.47 µm",  "label": "Visible (blue)",        "native_km": 1.0},
    "veggie":        {"goes": 3,  "ahi": 4,  "wavelength": "0.86 µm",  "label": "Veggie / NIR",          "native_km": 1.0},
    "shortwave_ir":  {"goes": 7,  "ahi": 7,  "wavelength": "3.9 µm",   "label": "Shortwave IR",          "native_km": 2.0},
    "wv_upper":      {"goes": 8,  "ahi": 8,  "wavelength": "6.2 µm",   "label": "Upper-tropospheric WV", "native_km": 2.0},
    "wv_lower":      {"goes": 10, "ahi": 10, "wavelength": "7.3 µm",   "label": "Lower-tropospheric WV", "native_km": 2.0},
    "clean_ir":      {"goes": 13, "ahi": 13, "wavelength": "10.4 µm",  "label": "Clean longwave IR",     "native_km": 2.0},
    "ir_window":     {"goes": 14, "ahi": 14, "wavelength": "11.2 µm",  "label": "IR window",             "native_km": 2.0},
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
class UnsupportedTimeError(Exception):
    """Requested time falls outside the satellite's archive coverage."""
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
    units: str              # "K" | "1" | "rgb" (true-color composite: cmi is H×W×3, 0..1)
    # True-color only: per-pixel cos(solar zenith), exposed so the render path
    # can apply the day/night terminator (GeoColor-lite fade to IR at night).
    cos_sza: Optional[np.ndarray] = None
    # True-color only: which target-grid pixels are geometrically ON the
    # satellite's visible disk. The render-side degenerate-RGB guard counts
    # NaNs over these pixels only, so a disk-limb sector (big legitimate
    # off-disk corner) isn't mistaken for a broken fetch.
    geom_valid: Optional[np.ndarray] = None
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
            listings_expiry_time=30,
        )
    return _fs
async def _to_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
def _bbox_overlaps(a: list[float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])
def bbox_lon_span(bbox) -> float:
    """Longitudinal width in degrees; lon_max < lon_min is an antimeridian
    crossing (e < w convention) and wraps to the positive span. A zero
    remainder is the full-width [-180, 180] box -> 360."""
    return (bbox[2] - bbox[0]) % 360.0 or 360.0
def _bbox_inside(inner: list[float], outer: tuple[float, float, float, float], buffer: float = 0.0) -> bool:
    """Wrap-aware containment: either box may cross the antimeridian
    (lon_max < lon_min). Longitudes compare as the inner box's wrapped
    offset east of the outer's west edge; a crossing inner box can never
    sit inside a non-crossing outer (the offset math rejects it)."""
    if inner[1] < outer[1] - buffer or inner[3] > outer[3] + buffer:
        return False
    o_span = (outer[2] - outer[0]) % 360.0
    i_span = (inner[2] - inner[0]) % 360.0
    i_off = (inner[0] - outer[0]) % 360.0
    if i_off > 180.0:  # inner west edge slightly WEST of outer's -> small negative
        i_off -= 360.0
    return i_off >= -buffer and i_off + i_span <= o_span + buffer
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
    # Sub-sat longitude used by ``pick_satellite`` for angular-distance
    # tie-breaking. Subclasses set this as a class var.
    sub_sat_lon: ClassVar[float]
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
        product_hint: "str | None" = None,
    ) -> ResolvedFile:
        """Locate the smallest product covering ``bbox`` at ``time``.

        ``product_hint`` is an optional caller preference (e.g. "target" for the
        AHI mesoscale Target sector); satellites that don't recognise it ignore it.
        """
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
    async def fetch_true_color(
        self, bbox: list[float], red_resolved: ResolvedFile
    ) -> FetchResult:
        """Fetch a multi-band RGB true-color composite, given the already-
        resolved red-band file (which pins product + scan time so the RGB bands
        are co-temporal). Implemented per family (ABI synthesizes green; AHI
        uses its native green). Default raises so an as-yet-unsupported family
        surfaces a clear message, not AttributeError."""
        raise NotImplementedError(
            f"true color is not yet available for {self.family}"
        )
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
# Shared GOES (ABI) plumbing — used by both GOES-East and GOES-West
# ---------------------------------------------------------------------------
# GOES-East "well-imageable" rectangle. The literal visible disk from
# sub-sat -75.2° extends to ~+10°E and ~-160°W, but image quality and
# limb-distortion grow steep past ±60°. The CoverageError message advertises
# "-135° to -5°"; we use that as the disk_bbox so the picker's overlap check
# matches what users see in the message and pushes far-east bboxes (Africa /
# Europe) into the METEOSAT-coming-soon path instead of grabbing an unusable
# limb slice from GOES.
GOES_DISK_BBOX = (-135.0, -75.0, -5.0, 75.0)  # (lon_min, lat_min, lon_max, lat_max)
# CONUS scan footprint changed when GOES-East switched from Mode 3 to Mode 6
# (more frequent CONUS, slightly larger scan). Pre-Mode-6 the eastern edge
# is ~-65°W; post-Mode-6 it's ~-55°W. Picking CMIPC for a bbox that extends
# past the relevant edge produces a black wedge (no data) on the eastern
# side of the render — see _pick_conus for the fallback logic.
CMIPC_MODE6_START = dt.datetime(2019, 4, 2, tzinfo=dt.timezone.utc)
CONUS_FOOTPRINT_MODE3 = (-135.0, 14.0, -65.0, 50.0)
CONUS_FOOTPRINT_MODE6 = (-135.0, 14.0, -55.0, 50.0)
# GOES-West CMIPC ("PACUS") scan footprint. GOES-17/18 both operate in Mode 6
# from launch, so a single footprint covers all GOES-West-era times. The PACUS
# sector sits over the eastern Pacific / western Americas; extents below come
# from NOAA GOES-West product specs (lon ~-152°W to ~-77°W, lat ~14°N to
# ~51°N). Bboxes outside this rect skip CMIPC and fall through to CMIPF.
PACUS_FOOTPRINT = (-152.0, 14.0, -77.0, 51.0)
GOES_BUCKET_OVERRIDE = os.getenv("GOES_BUCKET", "").strip()
# Time boundaries for the GOES-East time-based picker.
# 2025-04-04: GOES-19 became operational GOES-East, GOES-16 went to standby.
# 2018-08-01: per ops spec, before this date GOES-16 archive is the only
# reliable candidate.
GOES19_OPERATIONAL = dt.datetime(2025, 4, 4, tzinfo=dt.timezone.utc)
GOES16_PRIMARY_BEFORE = dt.datetime(2018, 8, 1, tzinfo=dt.timezone.utc)
# Time boundaries for the GOES-West time-based picker.
# 2023-01-04: GOES-18 became operational GOES-West, GOES-17 went to standby.
# 2019-02-12: GOES-17 became operational GOES-West (earliest GOES-West time
# we'll resolve at all).
GOES18_OPERATIONAL = dt.datetime(2023, 1, 4, tzinfo=dt.timezone.utc)
GOES17_OPERATIONAL = dt.datetime(2019, 2, 12, tzinfo=dt.timezone.utc)
PRIMARY_LIVE_BUCKET = "noaa-goes19"
GOES_EAST_SUB_SAT_LON = -75.2
GOES_WEST_SUB_SAT_LON = -137.2
# Angular distance (deg) from sub-sat point inside which we still consider
# the GOES-West disk usable. Same 85° threshold the AHI disk model uses.
GOES_WEST_DISK_HALF_LON = 85.0
GOES_WEST_DISK_LAT_LIMIT = 75.0
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
    """GOES-East CONUS sector footprint — date-dependent (Mode 3 vs Mode 6)."""
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
def _goes_meso_extent_from_ds(ds) -> tuple[float, float, float, float]:
    """(lon_w, lat_s, lon_e, lat_n) for a GOES ABI mesoscale scan.

    The operators steer M1/M2, so the extent is per-file. ABI L1b/L2 files carry
    it on the ``geospatial_lat_lon_extent`` VARIABLE's attributes
    (geospatial_{west,east}bound_longitude / {south,north}bound_latitude) -- NOT
    as global ``geospatial_lat_min/max`` attrs (those do not exist on these
    files). Reading the missing globals defaulted to the whole globe, which made
    coverage checks pass for ANY bbox (so M1 was wrongly chosen for M2)."""
    a = ds["geospatial_lat_lon_extent"].attrs
    return (float(a["geospatial_westbound_longitude"]),
            float(a["geospatial_southbound_latitude"]),
            float(a["geospatial_eastbound_longitude"]),
            float(a["geospatial_northbound_latitude"]))


def _check_meso_coverage_sync(s3_key: str, bbox: list[float], buffer_deg: float = 0.5) -> bool:
    fs = _get_fs()
    try:
        with fs.open(s3_key, mode="rb") as f:
            ds = xr.open_dataset(f, decode_cf=False, engine="h5netcdf")
            try:
                lon_w, lat_s, lon_e, lat_n = _goes_meso_extent_from_ds(ds)
            finally:
                ds.close()
        return _bbox_inside(bbox, (lon_w, lat_s, lon_e, lat_n), buffer=buffer_deg)
    except Exception as e:
        log.warning("meso coverage check failed for %s: %s", s3_key, e)
        return False
def _pick_buckets_for_time_dt(t: dt.datetime) -> list[str]:
    """GOES-East bucket fallback chain for a parsed UTC datetime.
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
def _pick_west_buckets_for_time_dt(t: dt.datetime) -> list[str]:
    """GOES-West bucket fallback chain.
    Rules:
      - t >= 2023-01-04: noaa-goes18 only
      - 2019-02-12 <= t < 2023-01-04: noaa-goes17 only
      - t < 2019-02-12: raises UnsupportedTimeError (handled in resolve()).
    Each operational window owns its bucket exclusively, so no fallback chain
    is needed (unlike East, which has a goes19/goes16 overlap window).
    """
    if t >= GOES18_OPERATIONAL:
        return ["noaa-goes18"]
    if t >= GOES17_OPERATIONAL:
        return ["noaa-goes17"]
    raise UnsupportedTimeError(
        f"GOES-West coverage starts 2019-02-12 (GOES-17 operational date); "
        f"requested {t.isoformat()}"
    )
def pick_buckets_for_time(requested_time: str) -> list[str]:
    """String form of the GOES-East bucket picker — used by /health for ops
    visibility. Always reports the GOES-East live bucket (primary live render
    path); GOES-West availability is a separate concern surfaced through the
    /health ``satellites`` list.
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
def _sample_geos(
    data: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_q: np.ndarray,
    y_q: np.ndarray,
) -> np.ndarray:
    """Bilinear-sample ``data`` (on its 1D geos grid x_src,y_src) at the query
    scan-angles (x_q, y_q). Used to put every true-color band onto one regular
    lat/lon target grid: the caller forward-projects the target lat/lon mesh to
    each band's (x, y) and samples here, so all bands co-register exactly and
    the result is a regular grid (clean to imshow, no curvilinear warp). ABI
    ``y`` descends north→south, so flip to the ascending order
    ``RegularGridInterpolator`` requires. Off-disk queries (NaN) -> NaN.
    """
    from scipy.interpolate import RegularGridInterpolator
    ys = y_src[::-1]
    d = data[::-1, :]
    interp = RegularGridInterpolator(
        (ys, x_src), d, bounds_error=False, fill_value=np.nan, method="linear"
    )
    out = interp(np.stack([y_q.ravel(), x_q.ravel()], axis=-1)).reshape(x_q.shape)
    return out.astype(np.float32)
# True-color target grid long-axis cap. The figure it feeds is 1320 px wide
# (12 in @ 110 dpi), so 1600 px still oversamples the final raster — the old
# 2400 px default bought no visible sharpness and paid ~2.3x the pixels in
# interpolation, Rayleigh, and solar/satellite geometry per frame.
TRUECOLOR_MAX_PX = int(os.getenv("TRUECOLOR_MAX_PX", "1600"))
def _truecolor_target_dims(lon_span: float, lat_span: float, max_px: int = None) -> tuple[int, int]:
    """Pixel (W, H) for a true-color target grid at ~0.5 km red GSD, long axis
    capped at ``max_px`` (defaults to TRUECOLOR_MAX_PX so output size and
    render time stay bounded)."""
    if max_px is None:
        max_px = TRUECOLOR_MAX_PX
    deg_per_px = 0.5 / 111.0
    nat_w = lon_span / deg_per_px
    nat_h = lat_span / deg_per_px
    scale = min(1.0, max_px / max(nat_w, nat_h, 1.0))
    return max(16, int(round(nat_w * scale))), max(16, int(round(nat_h * scale)))
class GOESBaseSatellite(Satellite):
    """Shared GOES (ABI) plumbing for GOES-East and GOES-West.
    Both families speak the same NetCDF format, share the same band layout
    (visible 2 / SWIR 7 / WV 8,10 / clean IR 13 / IR window 14), and use
    identical CMIPM/CMIPC/CMIPF product structure on their respective NOAA
    Open Data buckets. The only per-family knobs are:
      * ``sub_sat_lon`` (-75.2 vs -137.2)
      * which buckets to search at a given time
      * which CONUS-class sector footprint applies (East has Mode 3/6 cutover;
        West is Mode 6 from launch)
      * the visible-disk model used by ``can_see``
    Subclasses override the four hooks marked below.
    """
    sensor = "ABI"
    generic_to_band = {
        "visible_red": 2,
        "visible_blue": 1,
        "veggie": 3,
        "shortwave_ir": 7,
        "wv_upper": 8,
        "wv_lower": 10,
        "clean_ir": 13,
        "ir_window": 14,
    }
    # Bands the true-color recipe pulls. ABI has no green -> synthesized from
    # veggie; `green_band` is None to signal that to fetch_true_color.
    truecolor_bands = {"red": 2, "blue": 1, "veggie": 3}
    green_band = None
    sub_sat_lon: ClassVar[float]
    # --- per-family hooks --------------------------------------------------
    def _buckets_for_time(self, t: dt.datetime) -> list[str]:
        raise NotImplementedError
    def _conus_sector_footprint(self, t: dt.datetime) -> tuple[float, float, float, float]:
        """The CMIPC sector's geographic footprint for this satellite at
        ``time``. ``_pick_conus`` rejects bboxes that aren't fully inside
        this rect (otherwise the render gets a black wedge of no-data).
        """
        raise NotImplementedError
    def _conus_sector_label(self, t: dt.datetime) -> str:
        """Human-readable label for the CMIPC sector — used in fallback
        log lines (``CONUS Mode 6`` vs ``PACUS`` etc.)."""
        return "CONUS"
    # ----------------------------------------------------------------------
    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        """Return the primary operational hardware at ``time``.
        Default impl uses ``_buckets_for_time(t)[0]`` as the primary bucket;
        subclasses can override if they need different reporting behavior.
        """
        bucket = self._buckets_for_time(time)[0]
        return ResolvedSatellite(
            name=goes_sat_label(bucket),
            bucket=bucket,
            sub_sat_lon=self.sub_sat_lon,
        )
    async def bucket_reachable(self) -> bool:
        """Probe the primary live bucket. Used by /health.
        We don't probe historical fallbacks here because the goes16/17/18
        archives are on the same NOAA AWS Open Data CDN and effectively
        share an availability fate with the live primary, and /health
        green should mean "live renders work" — historical query failures
        surface cleanly via /render's 502.
        """
        primary = self.primary_live_bucket
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
        product_hint: "str | None" = None,   # "meso" forces the CMIPM product
    ) -> ResolvedFile:
        if generic_channel not in self.generic_to_band:
            raise ValueError(
                f"unknown generic channel for {self.family}: {generic_channel!r}"
            )
        band = self.generic_to_band[generic_channel]
        buckets = self._buckets_for_time(time)
        lon_w = bbox_lon_span(bbox)
        lat_h = bbox[3] - bbox[1]
        # Product preference: smallest sector that could plausibly cover
        # the bbox. CMIPM is only viable for ≤12°×12°; anything larger
        # starts at CMIPC, whose internal footprint check falls through to
        # CMIPF when the bbox lies outside the CONUS/PACUS scan footprint.
        # product_hint=="meso" (the meso-sector poller) forces _pick_meso FIRST
        # regardless of span: a mesoscale tile's lat/lon BOUNDING box runs wider
        # than 12° at off-nadir / high latitude (e.g. CMIPM1 over the US can be
        # ~21°×15°), even though it's the same 1000 km sector. _pick_meso's own
        # coverage check still falls through to CONUS/full-disk if it doesn't fit.
        if (product_hint == "meso"
                or (lon_w <= MESO_PER_AXIS_DEG_MAX and lat_h <= MESO_PER_AXIS_DEG_MAX)):
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
        # GOES L2 CMIP mesoscale lives under the SINGLE product prefix
        # ABI-L2-CMIPM/; M1 vs M2 is in the FILENAME (...-CMIPM1-... / ...-CMIPM2-...),
        # NOT separate ABI-L2-CMIPM1//ABI-L2-CMIPM2/ prefixes (those don't exist --
        # listing them returns [], so meso silently fell back to CONUS). List the
        # real CMIPM prefix and filter to the sector token. Try M1 then M2 — first
        # that covers wins. `sector` stays the product label for _make_resolved.
        for sector in ("CMIPM1", "CMIPM2"):
            tok = f"-{sector}-"
            files = [f for f in await _list_files_around(bucket, "CMIPM", channel, target)
                     if tok in f]
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
        # Validate that the requested bbox is fully inside the CMIPC scan
        # footprint for the selected satellite + date. Overlap-only checks are
        # not safe — a bbox extending past the scan edge produces a black
        # wedge in the rendered image (CMIPC sector has no data there).
        footprint = self._conus_sector_footprint(target)
        if not _bbox_inside(bbox, footprint):
            sat = goes_sat_label(bucket)
            label = self._conus_sector_label(target)
            log.info(
                "bbox lon=%.1f..%.1f lat=%.1f..%.1f outside %s footprint "
                "lon=%.1f..%.1f for %s — falling back to CMIPF",
                bbox[0], bbox[2], bbox[1], bbox[3],
                label, footprint[0], footprint[2], sat,
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
        # All currently-resolvable East buckets sit near -75.2°W; all
        # currently-resolvable West buckets sit near -137.2°W. (Old retired
        # hardware drifts to other slots after handover, but we never query
        # those time-ranges for that hardware.) So a single per-family
        # sub-sat-lon constant is correct here.
        return ResolvedFile(
            bucket=bucket,
            s3_key=s3_key,
            product=product,
            scan_start=scan_start,
            sat_name=goes_sat_label(bucket),
            sub_sat_lon=self.sub_sat_lon,
        )
    async def _find_band_at(
        self, bucket: str, product: str, band: int, scan_start: dt.datetime
    ) -> ResolvedFile:
        """Locate band ``band``'s file for the SAME product + scan_start as an
        already-resolved sibling band. All bands of one ABI scan share the
        scan-start token, so the RGB bands are guaranteed co-temporal.

        Mesoscale gotcha (same one _pick_meso handles): CMIPM1/CMIPM2 are NOT
        S3 prefixes — everything lives under ABI-L2-CMIPM/ with the sector in
        the FILENAME. Listing "CMIPM1" returns [] and broke every GOES meso
        true-color render, so list CMIPM and filter to the sector token."""
        list_product, sector_tok = product, None
        if product.startswith("CMIPM") and product != "CMIPM":
            list_product, sector_tok = "CMIPM", f"-{product}-"

        def _list(t: dt.datetime) -> list[str]:
            files = _list_hour(bucket, list_product, band, t)
            if sector_tok:
                files = [f for f in files if sector_tok in f]
            return files

        files = await _to_thread(_list, scan_start)
        if not files:
            # The scan can straddle an hour boundary for the previous-hour edge.
            files = await _to_thread(_list, scan_start - dt.timedelta(hours=1))
        if not files:
            raise RuntimeError(f"no band {band} file for {product} at {scan_start.isoformat()}")
        with_t = [(f, _parse_scan_start(f)) for f in files]
        exact = [f for f, t in with_t if t == scan_start]
        chosen = exact[0] if exact else min(with_t, key=lambda p: abs((p[1] - scan_start).total_seconds()))[0]
        return self._make_resolved(bucket, chosen, product, scan_start)
    async def fetch_true_color(
        self, bbox: list[float], red_resolved: ResolvedFile
    ) -> FetchResult:
        """Fetch the RGB true-color composite given the resolved red-band file.
        Pulls the other true-color bands from that SAME product/scan so they're
        co-temporal, crops each, resamples the 1 km bands onto the 0.5 km red
        grid (shared geos x/y → exact co-registration), and hands off to
        truecolor.assemble_truecolor. ABI green is synthesized inside.
        """
        # Sibling-band lookups run concurrently (S3 listings). Clean-IR
        # (band 13) backs the GeoColor-lite night fade; same product + scan
        # so everything co-registers with the visible bands.
        roles = [(role, band) for role, band in self.truecolor_bands.items()
                 if role != "red"]
        roles.append(("ir", self.generic_to_band["clean_ir"]))
        resolved_list = await asyncio.gather(*(
            self._find_band_at(red_resolved.bucket, red_resolved.product,
                               band, red_resolved.scan_start)
            for _, band in roles
        ))
        band_files: dict[str, ResolvedFile] = {"red": red_resolved}
        band_files.update(zip((r for r, _ in roles), resolved_list))
        return await _to_thread(self._compose_true_color_sync, band_files, bbox, red_resolved)
    def _compose_true_color_sync(
        self, band_files: dict[str, ResolvedFile], bbox: list[float], red_resolved: ResolvedFile
    ) -> FetchResult:
        from concurrent.futures import ThreadPoolExecutor
        import truecolor

        def _open_crop(rf: ResolvedFile) -> tuple:
            ds, tmp_dir = self.open(rf)
            try:
                return self._crop_to_bbox(ds, bbox)
            finally:
                ds.close()
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # Each band downloads its own file to its own tmp dir — fully
        # independent, so fetch+crop all of them concurrently instead of
        # serializing four S3 downloads per frame.
        with ThreadPoolExecutor(max_workers=len(band_files)) as pool:
            futures = {role: pool.submit(_open_crop, rf)
                       for role, rf in band_files.items()}
            results = {role: f.result() for role, f in futures.items()}
        crops: dict[str, tuple] = {}
        proj = None
        for role, (cmi, x_sub, y_sub, lon_origin, H, r_eq, r_pol) in results.items():
            crops[role] = (cmi, x_sub, y_sub)
            proj = (lon_origin, H, r_eq, r_pol)
        r_cmi, r_x, r_y = crops["red"]
        lon_origin, H, r_eq, r_pol = proj
        H_pix, W_pix = r_cmi.shape
        # Cap the target grid like the AHI path: the figure raster is 1320 px,
        # so a 2000 px meso red crop only inflates interpolation + Rayleigh +
        # geometry cost with no visible sharpness.
        cap_scale = min(1.0, TRUECOLOR_MAX_PX / max(H_pix, W_pix, 1))
        if cap_scale < 1.0:
            H_pix = max(16, int(round(H_pix * cap_scale)))
            W_pix = max(16, int(round(W_pix * cap_scale)))
        # Regular lat/lon target grid over the bbox at the red pixel count.
        # Image row order = north→south (row 0 = lat_max) for imshow origin
        # "upper". Antimeridian (lon_max < lon_min) is unwrapped so the target
        # longitudes stay monotonic, then re-wrapped to ±180 for projection.
        lon_min, lat_min, lon_max, lat_max = bbox
        lon_max_uw = lon_max + 360.0 if lon_max < lon_min else lon_max
        tgt_lons = ((np.linspace(lon_min, lon_max_uw, W_pix) + 180.0) % 360.0) - 180.0
        tgt_lats = np.linspace(lat_max, lat_min, H_pix)
        TLON, TLAT = np.meshgrid(tgt_lons, tgt_lats)
        TX, TY = _latlon_to_xy(TLAT, TLON, lon_origin, H, r_eq, r_pol)
        def grid(role: str):
            c, xs, ys = crops[role]
            return _sample_geos(c, xs, ys, TX, TY)
        red = grid("red")
        blue = grid("blue")
        green = grid("green") if "green" in crops else None
        veggie = grid("veggie") if "veggie" in crops else None
        ir_bt = grid("ir") if "ir" in crops else None  # clean-IR (K) for night fade
        # "Could have data" mask: on-disk (forward projection finite) AND
        # inside the red crop's scan-angle window. A limb-grazing meso
        # sector's lat/lon bounding box is mostly OUTSIDE its actual scan
        # quadrilateral — only pixels the scan could cover should count
        # toward the degenerate-frame NaN statistic.
        geom_valid = (
            np.isfinite(TX) & np.isfinite(TY)
            & (TX >= r_x.min()) & (TX <= r_x.max())
            & (TY >= r_y.min()) & (TY <= r_y.max())
        )
        lats, lons = TLAT.astype(np.float32), TLON.astype(np.float32)
        platform_name = self._pyspectral_platform(red_resolved.bucket)
        rgb, cos_sza = truecolor.assemble_truecolor(
            red, green, blue, veggie, lats, lons,
            when=red_resolved.scan_start,
            sub_sat_lon=red_resolved.sub_sat_lon,
            platform_name=platform_name,
            sensor="abi",
            ir_bt=ir_bt,
        )
        return FetchResult(
            cmi=rgb,
            lats=lats.astype(np.float32),
            lons=lons.astype(np.float32),
            channel=self.truecolor_bands["red"],
            generic_channel="true_color",
            scan_start=red_resolved.scan_start,
            product=red_resolved.product,
            bucket=red_resolved.bucket,
            sat_name=red_resolved.sat_name,
            sub_sat_lon=red_resolved.sub_sat_lon,
            units="rgb",
            cos_sza=cos_sza,
            geom_valid=geom_valid,
        )
    @staticmethod
    def _pyspectral_platform(bucket: str) -> str:
        """pyspectral platform name (e.g. 'GOES-19') from a bucket name."""
        return goes_sat_label(bucket)
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
    def _crop_to_bbox(self, ds: xr.Dataset, bbox: list[float]):
        """Geos-crop ``ds["CMI"]`` to ``bbox``.
        Returns ``(cmi, x_sub, y_sub, lon_origin, H, r_eq, r_pol)``. Shared by
        the single-band ``project_to_latlon`` and the multi-band
        ``fetch_true_color`` paths. All ABI bands of one satellite share the
        same geos projection, so the returned (x, y) scan-angle frame is a
        *common coordinate system* across bands — co-registering them for an
        RGB composite is just interpolation onto the red band's (x, y) grid,
        with no reprojection.
        """
        proj = ds["goes_imager_projection"]
        h = float(proj.attrs["perspective_point_height"])
        r_eq = float(proj.attrs["semi_major_axis"])
        r_pol = float(proj.attrs["semi_minor_axis"])
        lon_origin = float(proj.attrs["longitude_of_projection_origin"])
        H = h + r_eq  # distance from earth center to satellite
        # x, y stored as 1D coords in the file
        x = ds["x"].values  # radians
        y = ds["y"].values
        # Sample bbox corners + center to get xy span (use a denser grid for
        # safety). Antimeridian crossing (lon_max < lon_min): unwrap the east
        # edge so the sample sweep covers the bbox, not the far side of the
        # planet — _latlon_to_xy is trig-based, so lons past 180 are fine.
        lon_min, lat_min, lon_max, lat_max = bbox
        lon_max_uw = lon_max + 360.0 if lon_max < lon_min else lon_max
        n_sample = 16
        sample_lons = np.linspace(lon_min, lon_max_uw, n_sample)
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
        return (
            cmi.astype(np.float32),
            sub["x"].values,
            sub["y"].values,
            lon_origin, H, r_eq, r_pol,
        )
    def project_to_latlon(
        self,
        ds: xr.Dataset,
        bbox: list[float],
        resolved: ResolvedFile,
        generic_channel: str,
    ) -> FetchResult:
        cmi, x_sub, y_sub, lon_origin, H, r_eq, r_pol = self._crop_to_bbox(ds, bbox)
        # Build lat/lon for this window via inverse projection
        X, Y = np.meshgrid(x_sub, y_sub)
        lats, lons = _xy_to_latlon(X, Y, lon_origin, H, r_eq, r_pol)
        units = ds["CMI"].attrs.get("units", "")
        return FetchResult(
            cmi=cmi,
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
class GOESEastSatellite(GOESBaseSatellite):
    """GOES-East satellite (sub-sat -75.2°W).
    Resolves to GOES-19 for time >= 2025-04-04; GOES-16 otherwise. The
    goes19/goes16 fallback chain inside ``find_file`` (when
    GOES16_PRIMARY_BEFORE <= time < GOES19_OPERATIONAL) handles the brief
    overlap window where either satellite may have a given file first.
    """
    family = "GOES-East"
    sub_sat_lon = GOES_EAST_SUB_SAT_LON
    disk_bbox = GOES_DISK_BBOX
    primary_live_bucket = PRIMARY_LIVE_BUCKET
    def _buckets_for_time(self, t: dt.datetime) -> list[str]:
        return _pick_buckets_for_time_dt(t)
    def _conus_sector_footprint(self, t: dt.datetime) -> tuple[float, float, float, float]:
        return _conus_footprint(t)
    def _conus_sector_label(self, t: dt.datetime) -> str:
        return "CONUS Mode 6" if t >= CMIPC_MODE6_START else "CONUS Mode 3"
    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        """East ``resolve()`` reports the *primary* operational hardware at
        ``time``. The actually-fetched satellite may differ during the
        goes19/goes16 overlap (2018-08-01..2025-04-04) if the primary
        doesn't have the file — that fallback is handled inside
        ``find_file`` and reflected on the ``ResolvedFile`` it returns.
        ``resolve()`` itself reports the ideal/primary hardware for the
        moment, so a probe like /health gets a deterministic answer.
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
        # Honor GOES_BUCKET_OVERRIDE for the live probe — same behavior as
        # before the GOESBaseSatellite refactor.
        primary = GOES_BUCKET_OVERRIDE or self.primary_live_bucket
        try:
            fs = _get_fs()
            await _to_thread(fs.ls, f"{primary}/ABI-L2-CMIPF/")
            return True
        except Exception as e:
            log.warning("bucket unreachable: %s", e)
            return False
class GOESWestSatellite(GOESBaseSatellite):
    """GOES-West satellite (sub-sat -137.2°W).
    Resolves to GOES-18 for time >= 2023-01-04; GOES-17 for
    2019-02-12 <= time < 2023-01-04. Pre-2019-02-12 raises
    ``UnsupportedTimeError`` (GOES-17 wasn't operational GOES-West yet).
    NOTE on GOES-17 archive: GOES-17 had a known ABI cooling system fault
    that degraded several IR channels during local AM hours. Brightness
    temperatures may show artifacts on bands 8-16 around local sunrise.
    Users should be aware when interpreting historical 2019-2022 imagery.
    Reference: https://www.goes-r.gov/users/abiCoolingFault.html
    """
    family = "GOES-West"
    sub_sat_lon = GOES_WEST_SUB_SAT_LON
    # disk_bbox is supplied for ABC compatibility but not used — can_see is
    # overridden because the West disk crosses ±180° on the west edge.
    disk_bbox = (-180.0, -GOES_WEST_DISK_LAT_LIMIT, 180.0, GOES_WEST_DISK_LAT_LIMIT)
    primary_live_bucket = "noaa-goes18"
    def can_see(self, bbox: list[float], time: dt.datetime) -> bool:
        if bbox[1] <= -GOES_WEST_DISK_LAT_LIMIT or bbox[3] >= GOES_WEST_DISK_LAT_LIMIT:
            return False
        center_lon = antimeridian_safe_center_lon(bbox)
        lon_offset = abs(((center_lon - self.sub_sat_lon + 180.0) % 360.0) - 180.0)
        return lon_offset < GOES_WEST_DISK_HALF_LON
    def _buckets_for_time(self, t: dt.datetime) -> list[str]:
        # Raises UnsupportedTimeError if t < GOES17_OPERATIONAL.
        return _pick_west_buckets_for_time_dt(t)
    def _conus_sector_footprint(self, t: dt.datetime) -> tuple[float, float, float, float]:
        # GOES-17/18 both ship Mode 6 from launch; PACUS footprint is constant.
        return PACUS_FOOTPRINT
    def _conus_sector_label(self, t: dt.datetime) -> str:
        return "PACUS"
    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        if time >= GOES18_OPERATIONAL:
            bucket = "noaa-goes18"
        elif time >= GOES17_OPERATIONAL:
            bucket = "noaa-goes17"
        else:
            raise UnsupportedTimeError(
                f"GOES-West coverage starts 2019-02-12 (GOES-17 operational date); "
                f"requested {time.isoformat()}"
            )
        return ResolvedSatellite(
            name=goes_sat_label(bucket),
            bucket=bucket,
            sub_sat_lon=GOES_WEST_SUB_SAT_LON,
        )
# ---------------------------------------------------------------------------
# Himawari-Pacific implementation
# ---------------------------------------------------------------------------
HIMAWARI_SUB_SAT_LON = 140.7
HIMAWARI_DISK_HALF_LON = 85.0   # geos visible-disk extent from sub-sat point
HIMAWARI_DISK_LAT_LIMIT = 75.0  # bbox lat must lie strictly inside ±75°
# Hardware boundaries (per ops):
#   2022-12-13 16:00 UTC: H-9 promoted to operational; H-8 went to standby.
#   ~2017-01-01 00:00 UTC: earliest H-8 data on the noaa-himawari8 bucket.
H9_OPERATIONAL_DATE = dt.datetime(2022, 12, 13, 0, 0, tzinfo=dt.timezone.utc)
H8_ARCHIVE_START = dt.datetime(2017, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
def _ahi_latlon_to_xy_deg(
    lat_deg: np.ndarray, lon_deg: np.ndarray, sub_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """AHI geos scan angles (degrees) for WGS84 lat/lon — the expensive trig
    half of the forward projection. Band-INDEPENDENT: every AHI band shares
    the same viewing geometry and differs only in the linear CFAC/LFAC/COFF/
    LOFF scaling, so a multi-band composite computes this ONCE and applies
    each band's scaling via _ahi_xy_deg_to_colline (the per-band recompute
    was ~8 s of pure duplicate trig per true-color frame)."""
    R_s = 42164.0           # km, satellite-Earth-center distance
    r_eq = 6378.1370        # km, WGS84
    r_pol = 6356.7523       # km, WGS84
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sub_lon_r = np.deg2rad(sub_lon)
    c_lat = np.arctan((r_pol * r_pol) / (r_eq * r_eq) * np.tan(lat))
    R_l = r_pol / np.sqrt(1.0 - (1.0 - (r_pol * r_pol) / (r_eq * r_eq)) * np.cos(c_lat) ** 2)
    R1 = R_s - R_l * np.cos(c_lat) * np.cos(lon - sub_lon_r)
    R2 = -R_l * np.cos(c_lat) * np.sin(lon - sub_lon_r)
    R3 = R_l * np.sin(c_lat)
    R_n = np.sqrt(R1 * R1 + R2 * R2 + R3 * R3)
    with np.errstate(invalid="ignore"):
        x = np.arctan2(-R2, R1)
        y = np.arcsin(-R3 / R_n)
    return np.rad2deg(x), np.rad2deg(y)
def _ahi_xy_deg_to_colline(
    x_deg: np.ndarray, y_deg: np.ndarray,
    cfac: int, lfac: int, coff: float, loff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one band's linear scaling to shared AHI scan angles."""
    col = coff + cfac / (2 ** 16) * x_deg
    line = loff + lfac / (2 ** 16) * y_deg
    return col, line
def _ahi_latlon_to_colline(
    lat_deg: np.ndarray, lon_deg: np.ndarray,
    sub_lon: float, cfac: int, lfac: int, coff: float, loff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward AHI geos projection: WGS84 lat/lon (deg) -> column/line indices.
    Reference: CGMS LRIT/HRIT Global Specification §4.4 (cited by JMA HSD spec §3).
    Both ``CFAC`` and ``LFAC`` are positive on AHI, matching the convention
    where x-scan-angle increases east (col 1 = west) and y-scan-angle
    increases north (line 1 = south of disk center). Column 1 / line 1 sit
    in the disk corner; ``COFF`` / ``LOFF`` (~2750.5 for 2 km bands)
    locate the sub-satellite point.
    """
    x_deg, y_deg = _ahi_latlon_to_xy_deg(lat_deg, lon_deg, sub_lon)
    return _ahi_xy_deg_to_colline(x_deg, y_deg, cfac, lfac, coff, loff)
def _ahi_colline_to_latlon(
    col: np.ndarray, line: np.ndarray,
    sub_lon: float, cfac: int, lfac: int, coff: float, loff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse AHI geos projection: column/line -> WGS84 lat/lon (deg).
    Returns NaN where the (col, line) pair lies outside the visible disk
    (discriminant < 0).
    """
    R_s = 42164.0
    r_eq = 6378.1370
    r_pol = 6356.7523
    ratio_sq = (r_eq * r_eq) / (r_pol * r_pol)
    sub_lon_r = np.deg2rad(sub_lon)
    x_deg = (col - coff) * (2 ** 16) / cfac
    y_deg = (line - loff) * (2 ** 16) / lfac
    x = np.deg2rad(x_deg)
    y = np.deg2rad(y_deg)
    cos_x = np.cos(x); sin_x = np.sin(x)
    cos_y = np.cos(y); sin_y = np.sin(y)
    a = cos_y * cos_y + ratio_sq * sin_y * sin_y
    b = R_s * cos_x * cos_y
    sd_sq = b * b - a * (R_s * R_s - r_eq * r_eq)
    valid = sd_sq >= 0
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.maximum(sd_sq, 0.0))
        sn = (b - sd) / a
        s1 = R_s - sn * cos_x * cos_y
        s2 = sn * sin_x * cos_y
        s3 = -sn * sin_y
        sxy = np.sqrt(s1 * s1 + s2 * s2)
        lon_rad = np.arctan2(s2, s1) + sub_lon_r
        lat_rad = np.arctan(ratio_sq * s3 / sxy)
    lon_deg = ((np.rad2deg(lon_rad) + 180.0) % 360.0) - 180.0
    lat_deg = np.rad2deg(lat_rad)
    return (
        np.where(valid, lat_deg, np.nan),
        np.where(valid, lon_deg, np.nan),
    )
class HimawariPacificSatellite(Satellite):
    family = "Himawari-Pacific"
    sensor = "AHI"
    generic_to_band = {
        "visible_red": 3,
        "visible_blue": 1,
        "veggie": 4,
        "shortwave_ir": 7,
        "wv_upper": 8,
        "wv_lower": 10,
        "clean_ir": 13,
        "ir_window": 14,
    }
    # AHI has a native green (band 2, 0.51 µm), so true color uses it directly
    # — no synthesis. Veggie (band 4) is fetched too for optional green
    # correction but the v1 recipe uses native green as-is.
    truecolor_bands = {"red": 3, "green": 2, "blue": 1, "veggie": 4}
    green_band = 2
    # disk_bbox isn't usable here (the disk crosses ±180° on the east side),
    # so we override can_see to use angular distance from sub-sat point.
    disk_bbox = (-180.0, -HIMAWARI_DISK_LAT_LIMIT, 180.0, HIMAWARI_DISK_LAT_LIMIT)
    primary_live_bucket = "noaa-himawari9"
    sub_sat_lon = HIMAWARI_SUB_SAT_LON
    def can_see(self, bbox: list[float], time: dt.datetime) -> bool:
        if bbox[1] < -HIMAWARI_DISK_LAT_LIMIT or bbox[3] > HIMAWARI_DISK_LAT_LIMIT:
            return False
        center_lon = antimeridian_safe_center_lon(bbox)
        lon_offset = abs(((center_lon - HIMAWARI_SUB_SAT_LON + 180.0) % 360.0) - 180.0)
        return lon_offset < HIMAWARI_DISK_HALF_LON
    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        if time >= H9_OPERATIONAL_DATE:
            return ResolvedSatellite("Himawari-9", "noaa-himawari9", HIMAWARI_SUB_SAT_LON)
        return ResolvedSatellite("Himawari-8", "noaa-himawari8", HIMAWARI_SUB_SAT_LON)
    async def find_file(
        self,
        time: dt.datetime,
        generic_channel: str,
        bbox: list[float],
        nearest_to_target: bool,
        product_hint: "str | None" = None,
    ) -> ResolvedFile:
        if generic_channel not in self.generic_to_band:
            raise ValueError(
                f"unknown generic channel for {self.family}: {generic_channel!r}"
            )
        # Mesoscale "Target" path: the AHI Target sector (Region 3) scans ~every
        # 2.5 min (sub-scans R301..R304 per 10-min slot) vs FLDK's 10 min. When the
        # caller hints product=="meso" (the meso poller), resolve the latest Target
        # sub-scan. True-color rides Target too (visible_red pins the composite's
        # product, and _compose_true_color_sync loads EVERY sub-band off that same
        # Target sub-scan -- no Target/FLDK mix), so it refreshes at ~2.5 min like
        # the IR/WV bands. Set MESO_TC_FLDK=true to keep JUST true-color on the
        # 10-min FLDK full disk if the B03 0.5 km Target bandwidth is tight. Falls
        # back to FLDK if no Target slot lands.
        tc_fldk = os.environ.get("MESO_TC_FLDK", "").strip().lower() in (
            "1", "true", "yes", "on")
        if product_hint == "meso" and not (generic_channel == "visible_red"
                                           and tc_fldk):
            band = self.generic_to_band[generic_channel]
            resolved = await _to_thread(
                self._resolve_target_sync, time, band, nearest_to_target)
            if resolved is not None:
                return resolved
        # AHI cycles every 10 minutes; snap target time to the nearest 10-min slot.
        snapped = self._snap_10min(time, nearest_to_target)
        if not nearest_to_target:
            # LIVE path: the snapped slot may not have landed yet — NOAA's
            # bucket trails JMA by ~8-12 min for FLDK, so "floored − 10 min"
            # is often still empty and every consumer 500'd ("no AHI
            # segments found"). Probe slots (one listing each) until one
            # actually has this band's segments; for a true-color resolve
            # (visible_red) require the WHOLE recipe's bands so the
            # compositor can't trip over a half-published slot.
            need = [self.generic_to_band[generic_channel]]
            if generic_channel == "visible_red":
                need = sorted({*self.truecolor_bands.values(),
                               self.generic_to_band["clean_ir"]})
            # Probe from the FLOORED slot (snapped already backed off 10 min):
            # if the current slot has somehow fully landed, use it.
            probed = await _to_thread(
                self._first_available_fldk_slot_sync,
                snapped + dt.timedelta(minutes=10), need)
            if probed is not None:
                snapped = probed
        resolved_sat = self.resolve(snapped)
        # ``s3_key`` holds the time-folder prefix; the loader globs band segments
        # off that prefix at open() time so that the same ResolvedFile shape works
        # for both NOAA layouts (1 file or 10 segments per band).
        prefix = (
            f"{resolved_sat.bucket}/AHI-L1b-FLDK/"
            f"{snapped:%Y/%m/%d/%H%M}/"
        )
        return ResolvedFile(
            bucket=resolved_sat.bucket,
            s3_key=prefix,
            product="FLDK",
            scan_start=snapped,
            sat_name=resolved_sat.name,
            sub_sat_lon=HIMAWARI_SUB_SAT_LON,
        )

    @staticmethod
    def _fldk_band_complete(listing: "list[str]", band: int) -> bool:
        """True iff EVERY segment of ``band`` is present in the slot listing.
        NOAA uploads segments sequentially over minutes; the _SkkLL filename
        token carries (segment, total), so presence-of-any is NOT enough —
        a half-published B03 stitches to a shorter line window and would
        ship a part-black frame straight past the degenerate guard (which
        counts NaNs inside the stitched window only)."""
        token = f"_B{band:02d}_"
        segs: set[int] = set()
        total = None
        for f in listing:
            if token not in f or not f.endswith(".DAT.bz2"):
                continue
            try:
                s_part = f.rsplit("_S", 1)[1].split(".", 1)[0]
                seq, tot = int(s_part[:2]), int(s_part[2:])
            except (IndexError, ValueError):
                continue
            segs.add(seq)
            total = tot
        return total is not None and len(segs) >= total

    def _first_available_fldk_slot_sync(
        self, snapped: dt.datetime, bands: "list[int]", max_back: int = 4,
    ) -> "dt.datetime | None":
        """Newest FLDK slot at/before ``snapped`` with a COMPLETE segment set
        for EVERY band in ``bands``. One fs.ls per probed slot. None if no
        slot qualifies (caller keeps the snapped guess and the load surfaces
        the error as before)."""
        fs = _get_fs()
        for back in range(0, max_back + 1):
            slot = snapped - dt.timedelta(minutes=10 * back)
            bucket = self.resolve(slot).bucket
            prefix = f"{bucket}/AHI-L1b-FLDK/{slot:%Y/%m/%d/%H%M}/"
            try:
                listing = fs.ls(prefix)
            except (FileNotFoundError, OSError):
                continue
            if all(self._fldk_band_complete(listing, b) for b in bands):
                return slot
        return None

    def _resolve_target_sync(self, time: dt.datetime, band: int,
                             nearest_to_target: bool):
        """Resolve the latest available AHI Target sub-scan near ``time`` to a
        ResolvedFile (product 'Target<sub>', scan_start = the sub-scan's approx obs
        time so distinct sub-scans get distinct frame timestamps). Returns None if
        no recent Target slot has a sub-scan -> caller falls back to FLDK. Sync
        (s3 listing); call via _to_thread."""
        from vendor.ahi_loader import latest_target_subscan
        fs = _get_fs()
        base = time.replace(second=0, microsecond=0)
        floored = base.replace(minute=(base.minute // 10) * 10)
        # Start at the CURRENT slot (back=0): Target sub-scans land every
        # ~2.5 min with low publish latency, so the current 10-min folder
        # usually already has the freshest R30x. Skipping straight to back=1
        # (the old behavior) quantized Himawari freshness to 10+ minutes —
        # the whole point of the Target product is the 2.5-min cadence.
        for back in range(0, 4):
            slot = floored - dt.timedelta(minutes=10 * back)
            resolved_sat = self.resolve(slot)
            sub = latest_target_subscan(fs, resolved_sat.bucket, slot, band)
            if sub:
                # sub-scan obs time ~ slot + (sub-1)*2.5 min (R301..R304)
                scan_t = slot + dt.timedelta(seconds=(sub - 1) * 150)
                return ResolvedFile(
                    bucket=resolved_sat.bucket,
                    s3_key=(f"{resolved_sat.bucket}/AHI-L1b-Target/"
                            f"{slot:%Y/%m/%d/%H%M}/"),
                    product=f"Target{sub}",
                    scan_start=scan_t,
                    sat_name=resolved_sat.name,
                    sub_sat_lon=HIMAWARI_SUB_SAT_LON,
                )
        return None
    @staticmethod
    def _snap_10min(time: dt.datetime, nearest_to_target: bool) -> dt.datetime:
        """Snap to the most relevant 10-min slot.
        ``nearest_to_target=True`` rounds to the nearest slot; False (live
        ``latest`` queries) floors to the most-recent published slot — and
        backs off another 10 min so that segments have time to land in S3
        (publishing latency is a few minutes).
        """
        base = time.replace(second=0, microsecond=0)
        floor_min = (base.minute // 10) * 10
        floored = base.replace(minute=floor_min)
        if nearest_to_target:
            if base.minute - floor_min >= 5:
                return floored + dt.timedelta(minutes=10)
            return floored
        return floored - dt.timedelta(minutes=10)
    def open(self, resolved: ResolvedFile):
        # HimawariPacific overrides ``_fetch_sync`` directly so that the
        # full-disk calibration + bbox crop happen in one synchronous flow
        # without the GOES-style (Dataset, tmp_dir) handoff.
        raise NotImplementedError("HimawariPacificSatellite uses _fetch_sync directly")
    def project_to_latlon(self, ds, bbox, resolved, generic_channel):
        raise NotImplementedError("HimawariPacificSatellite uses _fetch_sync directly")
    @staticmethod
    def _target_slot_sub(resolved: ResolvedFile):
        """If ``resolved`` is an AHI Target sub-scan (product 'Target<sub>'),
        return (sub:int, slot:datetime): ``slot`` is the 10-min folder the Target
        segments live in (``scan_start`` is the sub-scan obs time, floored back to
        its slot). Returns None for an FLDK (or any non-Target) resolution, so the
        caller falls back to the FLDK loader. Shared by the single-band fetch and
        the true-color compositor so both recover the slot/sub identically."""
        if not resolved.product.startswith("Target"):
            return None
        sub = int(resolved.product[len("Target"):])
        slot = resolved.scan_start.replace(second=0, microsecond=0)
        slot = slot.replace(minute=(slot.minute // 10) * 10)
        return sub, slot

    def _fetch_sync(
        self,
        resolved: ResolvedFile,
        bbox: list[float],
        generic_channel: str,
    ) -> FetchResult:
        from vendor.ahi_loader import load_band_sync, load_target_band_sync
        band = self.generic_to_band[generic_channel]
        fs = _get_fs()
        # Pass bbox so the loader can drop irrelevant FLDK segments before download
        # (a Target sub-scan is a single small regional segment -- B03 0.5 km is
        # ~3 MB -- so the filter is a no-op there but matters for FLDK full disk).
        ts = self._target_slot_sub(resolved)
        if ts is not None:
            sub, slot = ts
            disk = load_target_band_sync(
                fs, resolved.bucket, slot, band, sub, bbox=tuple(bbox)
            )
        else:
            disk = load_band_sync(
                fs, resolved.bucket, resolved.scan_start, band, bbox=tuple(bbox)
            )
        # Antimeridian-safe bbox handling: if e < w (crossing), unwrap east edge.
        lon_min, lat_min, lon_max, lat_max = bbox
        unwrap = lon_max < lon_min
        lon_max_uw = lon_max + 360.0 if unwrap else lon_max
        n_sample = 16
        sample_lons_uw = np.linspace(lon_min, lon_max_uw, n_sample)
        sample_lons = ((sample_lons_uw + 180.0) % 360.0) - 180.0
        sample_lats = np.linspace(lat_min, lat_max, n_sample)
        LON, LAT = np.meshgrid(sample_lons, sample_lats)
        # Forward projection produces GLOBAL (full-disk) col/line indices.
        col_g, line_g = _ahi_latlon_to_colline(
            LAT, LON, disk.sub_lon, disk.cfac, disk.lfac, disk.coff, disk.loff
        )
        finite_mask = np.isfinite(col_g) & np.isfinite(line_g)
        col_g = col_g[finite_mask]
        line_g = line_g[finite_mask]
        if col_g.size == 0:
            raise RuntimeError("bbox has no projection-valid sample points")
        # Convert to local indices for slicing into ``disk.data`` (the disk
        # may be line- AND column-banded when the loader pre-cropped to bbox).
        ic_lo_g = int(np.floor(col_g.min())) - 5
        ic_hi_g = int(np.ceil(col_g.max())) + 5
        ic_lo = max(0, ic_lo_g - disk.col_offset)
        ic_hi = min(disk.n_columns, ic_hi_g - disk.col_offset)
        il_lo_g = int(np.floor(line_g.min())) - 5
        il_hi_g = int(np.ceil(line_g.max())) + 5
        il_lo = max(0, il_lo_g - disk.line_offset)
        il_hi = min(disk.n_lines, il_hi_g - disk.line_offset)
        if ic_hi <= ic_lo or il_hi <= il_lo:
            raise RuntimeError(
                f"crop produced empty window (line_offset={disk.line_offset}, "
                f"global lines {il_lo_g}..{il_hi_g}, local span 0..{disk.n_lines})"
            )
        MAX_PX_PER_AXIS = 2400
        x_stride = max(1, (ic_hi - ic_lo) // MAX_PX_PER_AXIS)
        y_stride = max(1, (il_hi - il_lo) // MAX_PX_PER_AXIS)
        sub_data = disk.data[il_lo:il_hi:y_stride, ic_lo:ic_hi:x_stride]
        # Inverse projection takes GLOBAL col/line — add the offsets back.
        sub_cols_global = np.arange(
            ic_lo + disk.col_offset, ic_hi + disk.col_offset, x_stride,
            dtype=np.float64,
        )[: sub_data.shape[1]]
        sub_lines_global = np.arange(
            il_lo + disk.line_offset, il_hi + disk.line_offset, y_stride, dtype=np.float64
        )[: sub_data.shape[0]]
        COL, LINE = np.meshgrid(sub_cols_global, sub_lines_global)
        lats, lons = _ahi_colline_to_latlon(
            COL, LINE, disk.sub_lon, disk.cfac, disk.lfac, disk.coff, disk.loff
        )
        return FetchResult(
            cmi=sub_data.astype(np.float32),
            lats=lats.astype(np.float32),
            lons=lons.astype(np.float32),
            channel=band,
            generic_channel=generic_channel,
            scan_start=resolved.scan_start,
            product=resolved.product,
            bucket=resolved.bucket,
            sat_name=resolved.sat_name,
            sub_sat_lon=resolved.sub_sat_lon,
            units=disk.units,
        )
    async def fetch_true_color(
        self, bbox: list[float], red_resolved: ResolvedFile
    ) -> FetchResult:
        """AHI true color: native green (band 2), no synthesis. ``red_resolved``
        pins the product (FLDK 10-min slot or a Target sub-scan) + bucket; every
        band loads from that same product so the composite is co-temporal."""
        return await _to_thread(self._compose_true_color_sync, bbox, red_resolved)
    def _compose_true_color_sync(
        self, bbox: list[float], red_resolved: ResolvedFile
    ) -> FetchResult:
        from concurrent.futures import ThreadPoolExecutor
        from vendor.ahi_loader import load_band_sync, load_target_band_sync
        from scipy.interpolate import RegularGridInterpolator
        import truecolor
        fs = _get_fs()
        # Load every band's calibrated disk CONCURRENTLY (segment-filtered to
        # the bbox's line band so B03's 0.5 km segments don't blow memory).
        # The five sequential loads were the single biggest cost of a
        # true-color frame; bz2 decompression releases the GIL, so threads
        # genuinely overlap download + decompress. EVERY band loads off the same
        # product the red file resolved to -- a Target sub-scan red pairs with
        # Target green/blue/veggie/IR (never an FLDK mix at a different scan
        # time); mirrors the single-band dispatch in _fetch_sync.
        ts = self._target_slot_sub(red_resolved)
        if ts is not None:
            sub, slot = ts

        def _load_tc(band: int):
            if ts is not None:
                return load_target_band_sync(
                    fs, red_resolved.bucket, slot, band, sub, bbox=tuple(bbox))
            return load_band_sync(
                fs, red_resolved.bucket, red_resolved.scan_start, band,
                bbox=tuple(bbox))

        roles = dict(self.truecolor_bands)
        roles["ir"] = self.generic_to_band["clean_ir"]  # GeoColor-lite night fade
        with ThreadPoolExecutor(max_workers=len(roles)) as pool:
            futures = {role: pool.submit(_load_tc, band)
                       for role, band in roles.items()}
            disks: dict[str, object] = {r: f.result() for r, f in futures.items()}
        # Regular lat/lon target grid (same scheme as GOES); antimeridian
        # unwrap keeps target longitudes monotonic for the AHI disk (centered
        # at 140.7°E, so W-Pac bboxes routinely cross ±180°).
        lon_min, lat_min, lon_max, lat_max = bbox
        lon_max_uw = lon_max + 360.0 if lon_max < lon_min else lon_max
        W_pix, H_pix = _truecolor_target_dims(lon_max_uw - lon_min, lat_max - lat_min)
        tgt_lons = ((np.linspace(lon_min, lon_max_uw, W_pix) + 180.0) % 360.0) - 180.0
        tgt_lats = np.linspace(lat_max, lat_min, H_pix)
        TLON, TLAT = np.meshgrid(tgt_lons, tgt_lats)
        # Scan angles are band-independent (all AHI bands share the viewing
        # geometry; only the linear CFAC/COFF scaling differs per resolution),
        # so the expensive trig runs ONCE for the target mesh.
        x_deg, y_deg = _ahi_latlon_to_xy_deg(
            TLAT, TLON, disks["red"].sub_lon
        )
        def grid(role: str) -> np.ndarray:
            d = disks[role]
            # This band's GLOBAL col/line off the shared angles, then shift
            # both axes into the local (line- and column-banded) window.
            col_g, line_g = _ahi_xy_deg_to_colline(
                x_deg, y_deg, d.cfac, d.lfac, d.coff, d.loff
            )
            line_local = line_g - d.line_offset
            col_local = col_g - d.col_offset
            interp = RegularGridInterpolator(
                (np.arange(d.n_lines), np.arange(d.n_columns)),
                d.data, bounds_error=False, fill_value=np.nan, method="linear",
            )
            out = interp(np.stack([line_local.ravel(), col_local.ravel()], axis=-1))
            return out.reshape(TLON.shape).astype(np.float32)
        red = grid("red")
        green = grid("green")
        blue = grid("blue")
        veggie = grid("veggie") if "veggie" in disks else None
        ir_bt = grid("ir") if "ir" in disks else None  # clean-IR (K) for night fade
        # "Could have data" mask: inside the red disk's (possibly line- and
        # column-banded) data window. Counting NaNs only there keeps the
        # degenerate-frame guard meaningful for limb sectors whose lat/lon
        # bounding box mostly misses the actual scan region.
        d_red = disks["red"]
        col_r, line_r = _ahi_xy_deg_to_colline(
            x_deg, y_deg, d_red.cfac, d_red.lfac, d_red.coff, d_red.loff)
        geom_valid = (
            np.isfinite(col_r) & np.isfinite(line_r)
            & (col_r - d_red.col_offset >= 0)
            & (col_r - d_red.col_offset <= d_red.n_columns - 1)
            & (line_r - d_red.line_offset >= 0)
            & (line_r - d_red.line_offset <= d_red.n_lines - 1)
        )
        lats, lons = TLAT.astype(np.float32), TLON.astype(np.float32)
        rgb, cos_sza = truecolor.assemble_truecolor(
            red, green, blue, veggie, lats, lons,
            when=red_resolved.scan_start,
            sub_sat_lon=red_resolved.sub_sat_lon,
            platform_name=red_resolved.sat_name,  # "Himawari-9"/"Himawari-8"
            sensor="ahi",
            ir_bt=ir_bt,
        )
        return FetchResult(
            cmi=rgb,
            lats=lats,
            lons=lons,
            channel=self.truecolor_bands["red"],
            generic_channel="true_color",
            scan_start=red_resolved.scan_start,
            product=red_resolved.product,
            bucket=red_resolved.bucket,
            sat_name=red_resolved.sat_name,
            sub_sat_lon=red_resolved.sub_sat_lon,
            units="rgb",
            cos_sza=cos_sza,
            geom_valid=geom_valid,
        )
# ---------------------------------------------------------------------------
# Singletons + picker
# ---------------------------------------------------------------------------
GOES_EAST = GOESEastSatellite()
GOES_WEST = GOESWestSatellite()
HIMAWARI_PACIFIC = HimawariPacificSatellite()
# Order matters for ties in pick_satellite: candidates are filtered by can_see
# then sorted by |sub_sat_lon - center_lon|.
ALL_SATELLITES: list[Satellite] = [GOES_EAST, GOES_WEST, HIMAWARI_PACIFIC]
SATELLITES_BY_FAMILY = {s.family.lower(): s for s in ALL_SATELLITES}
def pick_satellite(bbox: list[float], time: dt.datetime,
                   family_hint: "str | None" = None) -> Satellite:
    """Pick the best satellite for ``bbox`` at ``time``.
    Filter by visible-disk overlap, then break ties by minimum angular
    distance between the satellite's sub_sat-lon and the bbox center.
    ``family_hint`` (e.g. "GOES-West") names the satellite the caller KNOWS
    owns this imagery — the meso poller discovered its bbox from a GOES-18
    CMIPM2 scan, so /render must not re-guess. Center-distance picking gets
    antimeridian sectors wrong: the Bering Sea M2 box sits 2.5° closer to
    Himawari's sub-point than to GOES-West's, but only GOES-West actually
    images it. A hinted satellite still must pass its own can_see (422
    otherwise); an unknown hint falls back to the normal picker.
    Raises ``CoverageError`` if no satellite can see the bbox.
    """
    center_lon = antimeridian_safe_center_lon(bbox)
    if family_hint:
        hinted = SATELLITES_BY_FAMILY.get(family_hint.strip().lower())
        if hinted is not None:
            if not hinted.can_see(bbox, time):
                raise CoverageError(
                    f"bbox center {center_lon:.1f}° not visible from the "
                    f"requested satellite {hinted.family}"
                )
            hinted.resolve(time)  # may raise UnsupportedTimeError — propagate
            return hinted
    candidates = [s for s in ALL_SATELLITES if s.can_see(bbox, time)]
    if not candidates:
        raise CoverageError(
            f"bbox center {center_lon:.1f}° not visible from any active satellite. "
            f"GOES-East: -135° to -5°. GOES-West: +160° to -65° (wraps the antimeridian). "
            f"Himawari-Pacific: +60°E to +220°E. "
            f"METEOSAT (Atlantic east / Africa / Europe) coming soon."
        )
    def _angular_dist(sub_lon: float) -> float:
        return abs(((sub_lon - center_lon + 180.0) % 360.0) - 180.0)
    # Sort by angular distance ASC and pick the best-fit. We deliberately do
    # NOT silently fall back to a worse-angle satellite when the best fit
    # raises UnsupportedTimeError (e.g. EPac bbox at a pre-2019 time, where
    # GOES-East could technically see the bbox but at a 47°+ look angle that
    # produces unusable limb imagery). Surfacing the time error is more
    # honest than producing a low-quality fallback the user didn't ask for.
    candidates.sort(key=lambda s: _angular_dist(s.sub_sat_lon))
    best = candidates[0]
    best.resolve(time)  # may raise UnsupportedTimeError — let it propagate
    return best
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
