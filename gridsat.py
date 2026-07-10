"""GridSat-B1 deep-archive tier for /render (Time Machine, 1980 -> present).

NOAA Climate Data Record of geostationary IR: ONE merged product spanning
every geostationary generation (GOES/Meteosat/GMS/MTSAT/Himawari), so the
deep archive needs no per-satellite handling. AWS Open Data, anonymous:

    s3://noaa-cdr-gridsat-b1-pds/data/{YYYY}/GRIDSAT-B1.{Y}.{M}.{D}.{HH}.v02r01.nc

3-hourly (00/03/../21 UTC), regular 0.07 deg lat/lon grid (~8 km), lat
-70..70. Channels served: ``irwin_cdr`` (11 um IR window brightness
temperature, the CDR-quality primary) and ``irwvp`` (6.7 um water vapor BT,
present where the source satellites carried WV). Values are Kelvin
(xarray applies the netCDF scale/offset on decode).

HONESTY: this tier exists so the Time Machine can reach 1980 without
implying modern resolution -- render.py labels these frames
"GridSat-B1 · 11 um IR window · 3-hourly · ~8 km" and the frontend greys
multi-band fields pre-2017. It is ADDITIVE: /render dispatches here only
for explicit-time requests before ABI_CUTOVER (dates that previously 502'd)
-- every live/2017+ code path is untouched.

Also home to ``encode_bt_png``: the calibrated-BT u16 PNG encoder
(format="btpng" on /render, any era) that feeds the objfix/TC-Diagnostics
archive reanalysis. Encoding matches the S2 suite's bt.png exactly
(scale 0.01, offset -120, R=hi byte, G=lo byte, alpha=validity) so the
frontend BTProbe formula decodes both.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
from typing import Optional

import numpy as np

from satellites import (
    CoverageError,
    FetchResult,
    ResolvedFile,
    ResolvedSatellite,
    Satellite,
    UnsupportedTimeError,
    _get_fs,
    _to_thread,
)

log = logging.getLogger("tat-satellite.gridsat")

GRIDSAT_BUCKET = "noaa-cdr-gridsat-b1-pds"
GRIDSAT_VERSION = "v02r01"
GRIDSAT_START = dt.datetime(1980, 1, 1, tzinfo=dt.timezone.utc)
GRIDSAT_LAT_LIMIT = 70.0
GRIDSAT_STEP_S = 3 * 3600            # 3-hourly cadence

# Explicit-time /render requests BEFORE this dispatch to GridSat-B1: the
# native GOES-R/Himawari archives are reliable from here forward. (A future
# MergIR tier can claim 2000..cutover at 4 km/30-min; until then GridSat
# serves those dates honestly labeled.)
ABI_CUTOVER = dt.datetime(2017, 3, 1, tzinfo=dt.timezone.utc)

# BT PNG encoding constants -- MUST match the S2 suite's s2_bt.py descriptor
# (the frontend BTProbe/objfix decoder hardcodes the same formula).
BT_SCALE = 0.01
BT_OFFSET = -120.0

_CHANNEL_VARS = {"clean_ir": "irwin_cdr", "wv_upper": "irwvp"}


def slot_for(t: dt.datetime) -> dt.datetime:
    """Nearest 3-hour GridSat slot to ``t``."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    epoch = t.timestamp()
    snapped = round(epoch / GRIDSAT_STEP_S) * GRIDSAT_STEP_S
    return dt.datetime.fromtimestamp(snapped, tz=dt.timezone.utc)


def key_for(slot: dt.datetime) -> str:
    return (f"{GRIDSAT_BUCKET}/data/{slot:%Y}/"
            f"GRIDSAT-B1.{slot:%Y.%m.%d.%H}.{GRIDSAT_VERSION}.nc")


def candidate_slots(t: dt.datetime, max_steps: int = 4) -> list:
    """Slots ordered by |delta t|: nearest, then +-3h, +-6h ... (archive gaps
    are rare but real; the resolver takes the first that exists)."""
    base = slot_for(t)
    out = [base]
    for k in range(1, max_steps + 1):
        out.append(base - dt.timedelta(seconds=k * GRIDSAT_STEP_S))
        out.append(base + dt.timedelta(seconds=k * GRIDSAT_STEP_S))
    return out


def _crop_indices(coord: np.ndarray, lo: float, hi: float) -> slice:
    idx = np.where((coord >= lo) & (coord <= hi))[0]
    if idx.size == 0:
        return slice(0, 0)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def load_crop_sync(key: str, var: str, bbox) -> tuple:
    """Lazy-open one GridSat file and crop ``var`` to bbox (wrap-aware: the
    e<w convention unwraps the eastern part by +360 so longitudes stay
    monotonic for pcolormesh). Returns (data K float32, lats 1-D, lons 1-D
    possibly unwrapped)."""
    import xarray as xr
    fs = _get_fs()
    W, S, E, N = bbox
    S2, N2 = max(S, -GRIDSAT_LAT_LIMIT), min(N, GRIDSAT_LAT_LIMIT)
    if S2 >= N2:
        raise CoverageError(
            "GridSat-B1 covers 70°S–70°N — the requested box lies outside "
            "the geostationary archive.")
    with fs.open(key, mode="rb") as fh:
        ds = xr.open_dataset(fh, engine="h5netcdf")
        try:
            da = ds[var]
            if "time" in da.dims:
                da = da.isel(time=0)
            lat = ds["lat"].values
            lon = ds["lon"].values
            lat_sl = _crop_indices(lat, S2, N2)
            if E >= W:
                lon_sl = _crop_indices(lon, W, E)
                sub = da.isel(lat=lat_sl, lon=lon_sl).values
                lons = lon[lon_sl]
            else:
                # antimeridian crossing: [W..180] + [-180..E], east part +360
                sl_w = _crop_indices(lon, W, 180.0)
                sl_e = _crop_indices(lon, -180.0, E)
                a = da.isel(lat=lat_sl, lon=sl_w).values
                b = da.isel(lat=lat_sl, lon=sl_e).values
                sub = np.concatenate([a, b], axis=1)
                lons = np.concatenate([lon[sl_w], lon[sl_e] + 360.0])
            lats = lat[lat_sl]
        finally:
            ds.close()
    if sub.size == 0:
        raise CoverageError("requested box has no GridSat-B1 grid cells")
    return np.asarray(sub, dtype=np.float32), lats.astype(np.float64), lons.astype(np.float64)


class GridSatB1Satellite(Satellite):
    """The deep-archive 'satellite': one merged geostationary IR record."""
    family = "GridSat"
    sensor = "GEO-IR"
    # band numbers are synthetic (the CDR has variables, not bands); render.py
    # branches on the bucket for the honest header, never on these numbers.
    generic_to_band = {"clean_ir": 1, "wv_upper": 2}
    disk_bbox = (-180.0, -GRIDSAT_LAT_LIMIT, 180.0, GRIDSAT_LAT_LIMIT)
    primary_live_bucket = GRIDSAT_BUCKET
    sub_sat_lon = 0.0

    def can_see(self, bbox, time) -> bool:
        return bbox[1] < GRIDSAT_LAT_LIMIT and bbox[3] > -GRIDSAT_LAT_LIMIT

    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        if time < GRIDSAT_START:
            raise UnsupportedTimeError(
                "the GridSat-B1 archive begins 1980-01-01; requested "
                + time.isoformat())
        return ResolvedSatellite("GridSat-B1", GRIDSAT_BUCKET, 0.0)

    async def find_file(self, time, generic_channel, bbox, nearest_to_target,
                        product_hint=None) -> ResolvedFile:
        if generic_channel not in _CHANNEL_VARS:
            raise ValueError(
                f"{generic_channel!r} is not in the GridSat-B1 archive -- "
                "the deep tier is a single-channel 11 µm IR (+ 6.7 µm WV) record")
        self.resolve(time)
        return await _to_thread(self._find_file_sync, time)

    def _find_file_sync(self, time: dt.datetime) -> ResolvedFile:
        fs = _get_fs()
        last_err: Optional[str] = None
        for slot in candidate_slots(time):
            if slot < GRIDSAT_START:
                continue
            key = key_for(slot)
            try:
                if fs.exists(key):
                    return ResolvedFile(
                        bucket=GRIDSAT_BUCKET, s3_key=key, product="GRIDSAT-B1",
                        scan_start=slot, sat_name="GridSat-B1", sub_sat_lon=0.0)
            except OSError as e:   # transient listing error: try the next slot
                last_err = str(e)
        raise RuntimeError(
            f"no GridSat-B1 file within ±12 h of {time.isoformat()}"
            + (f" (last error: {last_err})" if last_err else ""))

    def open(self, resolved):   # pragma: no cover - fetch path is _fetch_sync
        raise NotImplementedError("GridSatB1Satellite uses _fetch_sync directly")

    def project_to_latlon(self, ds, bbox, resolved, generic_channel):  # pragma: no cover
        raise NotImplementedError("GridSatB1Satellite uses _fetch_sync directly")

    def _fetch_sync(self, resolved: ResolvedFile, bbox, generic_channel) -> FetchResult:
        var = _CHANNEL_VARS[generic_channel]
        data, lats1, lons1 = load_crop_sync(resolved.s3_key, var, bbox)
        finite_frac = float(np.isfinite(data).mean()) if data.size else 0.0
        # honest failure thresholds: the 11 µm CDR is near-complete (a sparse
        # crop means a real gap); WV exists only where the era's source
        # satellites carried a water-vapor channel, so demand real coverage
        # rather than letting the render's generic NaN guard produce a
        # misleading 'partial fetch' error downstream
        min_frac = 0.40 if var == "irwvp" else 0.02
        if finite_frac < min_frac:
            raise RuntimeError(
                f"GridSat-B1 {'6.7 µm water vapor' if var == 'irwvp' else var} "
                f"is not present for this era/region ({finite_frac:.0%} valid) "
                "-- the 11 µm IR window is the reliable deep-archive channel")
        LON, LAT = np.meshgrid(lons1, lats1)
        return FetchResult(
            cmi=data,
            lats=LAT.astype(np.float32),
            lons=LON.astype(np.float32),
            channel=self.generic_to_band[generic_channel],
            generic_channel=generic_channel,
            scan_start=resolved.scan_start,
            product="GRIDSAT-B1",
            bucket=GRIDSAT_BUCKET,
            sat_name="GridSat-B1",
            sub_sat_lon=0.0,
            units="K",
        )


GRIDSAT = GridSatB1Satellite()


# ---------------------------------------------------------------------------
# Calibrated-BT PNG (format="btpng"): era-agnostic objfix/diagnostics input.
# ---------------------------------------------------------------------------
def encode_bt_png(bt_c: np.ndarray) -> bytes:
    """Brightness temperature (°C, NaN = no data) -> lossless u16-in-RGBA PNG.
    Decode (the frontend BTProbe formula): (R*256 + G) * 0.01 - 120 °C,
    alpha 0 = no data. Row 0 must be NORTH -- callers flip if needed."""
    from PIL import Image
    bt = np.asarray(bt_c, dtype=np.float64)
    valid = np.isfinite(bt)
    q = np.zeros(bt.shape, dtype=np.uint16)
    q[valid] = np.clip(np.round((bt[valid] - BT_OFFSET) / BT_SCALE), 0, 65535).astype(np.uint16)
    out = np.zeros(bt.shape + (4,), dtype=np.uint8)
    out[..., 0] = (q >> 8).astype(np.uint8)
    out[..., 1] = (q & 0xFF).astype(np.uint8)
    out[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out, "RGBA").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def bt_png_from_fetch(data: FetchResult) -> bytes:
    """BT PNG for any single-channel FetchResult (K or °C source; row 0 north).
    True color has no scalar BT -- callers must gate it out."""
    cmi = np.asarray(data.cmi, dtype=np.float64)
    bt_c = cmi - 273.15 if data.units == "K" else cmi
    # row 0 = north: FetchResult grids may be ascending-lat (GridSat) or
    # geos-native (ABI/AHI curvilinear -- row order follows the scan). Flip
    # when the first row is south of the last (regular ascending grids).
    lats = np.asarray(data.lats)
    if lats.ndim == 2 and np.nanmean(lats[0]) < np.nanmean(lats[-1]):
        bt_c = bt_c[::-1]
    return encode_bt_png(bt_c)
