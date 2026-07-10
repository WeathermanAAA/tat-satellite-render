"""NASA MergIR archive tier for /render (Time Machine, 2000-02 -> ~present).

NCEP/CPC 4-km Merged IR (GPM_MERGIR.1): global-merged geostationary 11 µm
brightness temperature, 30-minute cadence, 4-km grid, 60°S-60°N. THE
best-available middle tier: dates >= MERGIR_START render here at ~4 km /
30-min instead of GridSat-B1's ~8 km / 3-hourly; GridSat stays the deep tier
for pre-2000 (and for the 6.7 µm WV channel, which MergIR does not carry).

ACCESS (verified 2026-07-10): there is NO anonymous path to this dataset.
The AWS Open Data listing (registry.opendata.aws/nasa-gpmmergir) points at
``gesdisc-cumulus-prod-protected`` -- Controlled Access, requester-pays,
anonymous S3 -> PermissionError; the GES DISC HTTPS archive 302s to URS
login; NASA PPS does not mirror the Tb granules (tree walked). So this tier
authenticates against GES DISC over HTTPS with the project's Earthdata
credentials (env: EARTHDATA_TOKEN bearer preferred, else EARTHDATA_USERNAME
+ EARTHDATA_PASSWORD via the URS redirect flow). WITHOUT credentials the
/render selector falls back to GridSat-B1 HONESTLY -- the burned-in era
header always names the source that actually rendered the frame.

File layout (one netCDF4 per hour, two half-hourly grids inside):
  https://data.gesdisc.earthdata.nasa.gov/data/MERGED_IR/GPM_MERGIR.1/
      {YYYY}/{DDD}/merg_{YYYYMMDDHH}_4km-pixel.nc4
  variable Tb(time=2, lat=3298, lon=9896), Kelvin; lat -59.98..59.98,
  lon -179.98..179.98 (regular ~0.03638° grid).

Note for later (deferred by decision): for even sharper pre-2017 over the
Americas the native GOES archives (NOAA CLASS / AWS) are the deeper option.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
import tempfile
from typing import Optional

import numpy as np

from satellites import (
    CoverageError,
    FetchResult,
    ResolvedFile,
    ResolvedSatellite,
    Satellite,
    UnsupportedTimeError,
    _to_thread,
)

log = logging.getLogger("tat-satellite.mergir")

MERGIR_BASE = "https://data.gesdisc.earthdata.nasa.gov/data/MERGED_IR/GPM_MERGIR.1"
MERGIR_START = dt.datetime(2000, 2, 7, tzinfo=dt.timezone.utc)   # record start (GES DISC)
MERGIR_LAT_LIMIT = 60.0
MERGIR_STEP_S = 30 * 60          # half-hourly grids (two per hourly file)


def have_credentials() -> bool:
    """Can this process authenticate to GES DISC? (the /render selector only
    prefers MergIR when it can actually fetch it -- honest fallback)."""
    return bool(os.getenv("EARTHDATA_TOKEN")
                or (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")))


def slot_for(t: dt.datetime) -> dt.datetime:
    """Nearest 30-min MergIR slot to ``t``."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    snapped = round(t.timestamp() / MERGIR_STEP_S) * MERGIR_STEP_S
    return dt.datetime.fromtimestamp(snapped, tz=dt.timezone.utc)


def url_for(slot: dt.datetime) -> tuple[str, int]:
    """(hourly granule URL, half-hour index 0|1) for a 30-min slot."""
    hour = slot.replace(minute=0)
    return (f"{MERGIR_BASE}/{hour:%Y}/{hour:%j}/merg_{hour:%Y%m%d%H}_4km-pixel.nc4",
            0 if slot.minute < 30 else 1)


def candidate_slots(t: dt.datetime, max_steps: int = 4) -> list:
    """Slots ordered by |delta t| (gap tolerance, mirrors gridsat)."""
    base = slot_for(t)
    out = [base]
    for k in range(1, max_steps + 1):
        out.append(base - dt.timedelta(seconds=k * MERGIR_STEP_S))
        out.append(base + dt.timedelta(seconds=k * MERGIR_STEP_S))
    return out


def _session():
    import requests
    s = requests.Session()
    tok = os.getenv("EARTHDATA_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    else:
        # the URS redirect flow: requests re-sends basic auth only to the
        # hosts in its redirect chain; URS accepts it and sets the cookie
        s.auth = (os.getenv("EARTHDATA_USERNAME", ""), os.getenv("EARTHDATA_PASSWORD", ""))
    return s


def download_granule(url: str, timeout: int = 120) -> bytes:
    if not have_credentials():
        raise RuntimeError(
            "MergIR needs Earthdata credentials (EARTHDATA_TOKEN or "
            "EARTHDATA_USERNAME/EARTHDATA_PASSWORD) -- no anonymous path exists")
    with _session() as s:
        r = s.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 404:
            raise FileNotFoundError(url)
        if r.status_code in (401, 403):
            raise RuntimeError(
                f"GES DISC auth failed ({r.status_code}) -- check the Earthdata "
                "credentials and that the 'NASA GESDISC DATA ARCHIVE' app is "
                "authorized on the profile (urs.earthdata.nasa.gov -> Applications)")
        r.raise_for_status()
        return r.content


def crop_from_bytes(raw: bytes, half_idx: int, bbox) -> tuple:
    """Decode one granule + crop Tb[half_idx] to bbox (wrap-aware, e<w
    unwraps east by +360). Pure function of bytes -- the tier's testable
    seam (the sampler-artifact / synthetic-fixture verification path).
    Returns (tb K float32, lats 1-D, lons 1-D possibly unwrapped)."""
    import xarray as xr
    W, S, E, N = bbox
    S2, N2 = max(S, -MERGIR_LAT_LIMIT), min(N, MERGIR_LAT_LIMIT)
    if S2 >= N2:
        raise CoverageError(
            "MergIR covers 60°S–60°N — the requested box lies outside the "
            "merged-IR archive.")
    # h5netcdf needs a real file for some builds; a spooled tmp keeps it cheap
    with tempfile.NamedTemporaryFile(suffix=".nc4") as tf:
        tf.write(raw)
        tf.flush()
        ds = xr.open_dataset(tf.name, engine="h5netcdf")
        try:
            da = ds["Tb"]
            if "time" in da.dims:
                da = da.isel(time=min(half_idx, da.sizes["time"] - 1))
            lat = ds["lat"].values
            lon = ds["lon"].values
            lat_idx = np.where((lat >= S2) & (lat <= N2))[0]
            if lat_idx.size == 0:
                raise CoverageError("no MergIR grid rows in the requested box")
            lat_sl = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
            if E >= W:
                li = np.where((lon >= W) & (lon <= E))[0]
                if li.size == 0:
                    raise CoverageError("no MergIR grid columns in the requested box")
                sub = da.isel(lat=lat_sl, lon=slice(int(li[0]), int(li[-1]) + 1)).values
                lons = lon[slice(int(li[0]), int(li[-1]) + 1)]
            else:
                wi = np.where(lon >= W)[0]
                ei = np.where(lon <= E)[0]
                a = da.isel(lat=lat_sl, lon=slice(int(wi[0]), int(wi[-1]) + 1)).values
                b = da.isel(lat=lat_sl, lon=slice(int(ei[0]), int(ei[-1]) + 1)).values
                sub = np.concatenate([a, b], axis=1)
                lons = np.concatenate([lon[wi], lon[ei] + 360.0])
            lats = lat[lat_sl]
        finally:
            ds.close()
    return np.asarray(sub, dtype=np.float32), lats.astype(np.float64), lons.astype(np.float64)


class MergIRSatellite(Satellite):
    """The middle archive tier: one merged global 11 µm record, 2000-02+."""
    family = "MergIR"
    sensor = "merged geo-IR"
    generic_to_band = {"clean_ir": 1}      # 11 µm only -- MergIR has NO WV
    disk_bbox = (-180.0, -MERGIR_LAT_LIMIT, 180.0, MERGIR_LAT_LIMIT)
    primary_live_bucket = "gesdisc-mergir"
    sub_sat_lon = 0.0

    def can_see(self, bbox, time) -> bool:
        return bbox[1] < MERGIR_LAT_LIMIT and bbox[3] > -MERGIR_LAT_LIMIT

    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        if time < MERGIR_START:
            raise UnsupportedTimeError(
                "the MergIR record begins 2000-02-07; requested " + time.isoformat())
        return ResolvedSatellite("MergIR", "gesdisc-mergir", 0.0)

    async def find_file(self, time, generic_channel, bbox, nearest_to_target,
                        product_hint=None) -> ResolvedFile:
        if generic_channel not in self.generic_to_band:
            raise ValueError(
                f"{generic_channel!r} is not in the MergIR record -- it is an "
                "11 µm IR-window product (WV comes from GridSat-B1)")
        self.resolve(time)
        # resolution is deterministic (no listing): gap tolerance happens at
        # fetch time (404 -> next candidate slot)
        slot = slot_for(time)
        url, _half = url_for(slot)
        return ResolvedFile(bucket="gesdisc-mergir", s3_key=url,
                            product="MERGIR", scan_start=slot,
                            sat_name="MergIR", sub_sat_lon=0.0)

    def open(self, resolved):   # pragma: no cover -- fetch path is _fetch_sync
        raise NotImplementedError("MergIRSatellite uses _fetch_sync directly")

    def project_to_latlon(self, ds, bbox, resolved, generic_channel):  # pragma: no cover
        raise NotImplementedError("MergIRSatellite uses _fetch_sync directly")

    def _fetch_sync(self, resolved: ResolvedFile, bbox, generic_channel) -> FetchResult:
        last_err = None
        for slot in candidate_slots(resolved.scan_start):
            if slot < MERGIR_START:
                continue
            url, half = url_for(slot)
            try:
                raw = download_granule(url)
            except FileNotFoundError as e:
                last_err = e
                continue
            tb, lats1, lons1 = crop_from_bytes(raw, half, bbox)
            finite_frac = float(np.isfinite(tb).mean()) if tb.size else 0.0
            if finite_frac < 0.02:
                last_err = RuntimeError(f"{url} half {half}: {finite_frac:.0%} valid")
                continue
            LON, LAT = np.meshgrid(lons1, lats1)
            return FetchResult(
                cmi=tb, lats=LAT.astype(np.float32), lons=LON.astype(np.float32),
                channel=1, generic_channel=generic_channel,
                scan_start=slot, product="MERGIR", bucket="gesdisc-mergir",
                sat_name="MergIR", sub_sat_lon=0.0, units="K")
        raise RuntimeError(
            f"no MergIR granule within ±2 h of {resolved.scan_start.isoformat()}"
            + (f" (last: {last_err})" if last_err else ""))


MERGIR = MergIRSatellite()
