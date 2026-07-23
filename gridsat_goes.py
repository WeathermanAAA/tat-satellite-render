"""NCEI GridSat-GOES tier for /render (Time Machine, 1994-10 -> 2017-12).

THE per-satellite deep tier for the GOES era: NOAA NCEI's GridSat-GOES
(Knapp, 2017) -- one netCDF per satellite per HOUR, 0.04 deg (~4 km)
equal-angle, SIX imager channels, un-blended (each file is one GOES's own
view). Against the tiers it upgrades: GridSat-B1 is 8 km / 3-hourly / one
blended IR+WV; MergIR is 4 km / 30-min but IR-only, 2000+, and
credential-gated. GridSat-GOES adds VISIBLE (ch1, calibrated reflectance,
subsampled from the native 1 km), shortwave 3.9 (ch2), 6.5 um WV (ch3),
10.7 um IR window (ch4), and 12.0/13.3 um split-window/CO2 (ch5/ch6) --
hourly, anonymous, direct HTTPS.

ACCESS (verified 2026-07-11 against the live archive, incl. the Katrina
scene GridSat-GOES.goes12.2005.08.28.1800.v01.nc, 63 MB, Accept-Ranges):
    https://www.ncei.noaa.gov/data/gridsat-goes/access/goes/{Y}/{M}/
        GridSat-GOES.goes{NN}.{Y}.{M}.{DD}.{HHMM}.v01.nc
Reads are RANGED (fsspec https + h5netcdf): a bbox render pulls only the
chunks it needs, not the 63 MB file.

HONESTY CONTRACT: this is the sharpest render-on-demand data that exists
for the era -- TRUE native GVAR (1 km vis / 4 km IR raw) is order-staged
only (NOAA CLASS) or licence-fenced (SSEC McFetch, .edu non-commercial), so
it is NOT served here and never implied. The burned-in header names the
actual satellite + product + resolution ("GOES-12 · GridSat-GOES · 10.7 um
IR · hourly · ~4 km"). Coverage is per-satellite (a disk, not the globe):
requests outside every available disk raise CoverageError so the /render
selector falls back to MergIR/GridSat-B1 -- the label always tells the
truth about which record drew the frame.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import threading
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

log = logging.getLogger("tat-satellite.gridsat_goes")

GG_BASE = "https://www.ncei.noaa.gov/data/gridsat-goes/access/goes"
GG_BUCKET = "ncei-gridsat-goes"          # render.py branches headers on this
GG_START = dt.datetime(1994, 10, 1, tzinfo=dt.timezone.utc)
GG_END = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)   # record frozen at 2017-12
GG_STEP_S = 3600                          # hourly ('goes' domain)
GG_LAT_LIMIT = 75.0                       # grid spans 75S..75N
GG_MAX_ZENITH_DEG = 75.0                  # usable disk reach for coverage tests

# generic /render channel -> GridSat-GOES variable
_CHANNEL_VARS = {
    "clean_ir": "ch4",       # 10.7 um IR window
    "wv_upper": "ch3",       # 6.5 um water vapor
    "visible_red": "ch1",    # 0.65 um visible reflectance (4 km, from 1 km)
    "shortwave_ir": "ch2",   # 3.9 um
}

# Operational sub-satellite longitudes by satellite + era (NOAA operational
# history; used only to RANK candidates from the month listing -- the fetch
# validates real coverage and falls through candidates honestly). Entries:
# (sat_number, start, end, sub_lon).
_POSITIONS = (
    (8,  "1995-01", "2003-05", -75.0),    # East
    (12, "2003-04", "2010-05", -75.0),    # East
    (13, "2010-04", "2018-01", -75.0),    # East
    (9,  "1996-01", "1998-08", -135.0),   # West
    (10, "1998-07", "2006-07", -135.0),   # West
    (11, "2006-06", "2011-12", -135.0),   # West
    (15, "2011-12", "2018-01", -135.0),   # West
    (9,  "2003-04", "2005-12", 155.0),    # GMS backup over the W Pacific
    (10, "2006-11", "2009-12", -60.0),    # South America
    (12, "2010-05", "2013-09", -60.0),    # South America
    (14, "2012-09", "2013-11", -105.0),   # standby fill-ins
)


def _ym(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m").replace(tzinfo=dt.timezone.utc)


def positions_at(t: dt.datetime) -> dict:
    """{sat_number: sub_lon} plausibly operational at ``t`` (table above)."""
    out = {}
    for num, a, b, sub in _POSITIONS:
        if _ym(a) <= t < _ym(b):
            out.setdefault(num, sub)
    return out


def slot_for(t: dt.datetime) -> dt.datetime:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    snapped = round(t.timestamp() / GG_STEP_S) * GG_STEP_S
    return dt.datetime.fromtimestamp(snapped, tz=dt.timezone.utc)


def candidate_slots(t: dt.datetime, max_steps: int = 3) -> list:
    base = slot_for(t)
    out = [base]
    for k in range(1, max_steps + 1):
        out.append(base - dt.timedelta(seconds=k * GG_STEP_S))
        out.append(base + dt.timedelta(seconds=k * GG_STEP_S))
    return out


def url_for(sat_num: int, slot: dt.datetime) -> str:
    return (f"{GG_BASE}/{slot:%Y}/{slot:%m}/"
            f"GridSat-GOES.goes{sat_num:02d}.{slot:%Y.%m.%d.%H%M}.v01.nc")


# ---- month listing (one HTTPS index per month, cached) ---------------------
_listing_cache: dict = {}
_listing_lock = threading.Lock()
_FNAME_RE = re.compile(
    r"GridSat-GOES\.goes(\d{2})\.(\d{4})\.(\d{2})\.(\d{2})\.(\d{4})\.v01\.nc")


def month_listing(year: int, month: int) -> set:
    """{(sat_num, 'YYYY.MM.DD.HHMM')} present in the month's directory index.
    ONE ranged-free GET per month, cached for the process lifetime (the
    record is frozen)."""
    key = (year, month)
    with _listing_lock:
        hit = _listing_cache.get(key)
    if hit is not None:
        return hit
    import requests
    url = f"{GG_BASE}/{year:04d}/{month:02d}/"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            entries: set = set()
        else:
            r.raise_for_status()
            entries = set()
            for m in _FNAME_RE.finditer(r.text):
                entries.add((int(m.group(1)),
                             f"{m.group(2)}.{m.group(3)}.{m.group(4)}.{m.group(5)}"))
    except Exception as e:  # noqa: BLE001 -- listing failure = tier unavailable
        raise RuntimeError(f"GridSat-GOES month index failed ({url}): {e}")
    with _listing_lock:
        _listing_cache[key] = entries
    return entries


def _zenith_ok(bbox, sub_lon: float) -> bool:
    """Does a satellite at ``sub_lon`` plausibly see the bbox center?"""
    W, S, E, N = bbox
    if E < W:
        E += 360.0
    clon = (W + E) / 2.0
    clat = min(max((S + N) / 2.0, -GG_LAT_LIMIT), GG_LAT_LIMIT)
    dlon = abs(((clon - sub_lon) + 180.0) % 360.0 - 180.0)
    import math
    cosp = math.cos(math.radians(clat)) * math.cos(math.radians(dlon))
    psi = math.degrees(math.acos(max(-1.0, min(1.0, cosp))))
    return psi <= GG_MAX_ZENITH_DEG


def rank_candidates(bbox, slot: dt.datetime) -> list:
    """Satellites listed for this slot that plausibly cover bbox, best
    (smallest center zenith) first. [(sat_num, sub_lon)]."""
    import math
    listing = month_listing(slot.year, slot.month)
    stamp = f"{slot:%Y.%m.%d.%H%M}"
    have = {num for num, s in listing if s == stamp}
    pos = positions_at(slot)
    W, S, E, N = bbox
    if E < W:
        E += 360.0
    clon, clat = (W + E) / 2.0, (S + N) / 2.0
    ranked = []
    for num in sorted(have):
        sub = pos.get(num)
        if sub is None:
            continue                      # not in the ops table: skip ranking
        if not _zenith_ok(bbox, sub):
            continue
        dlon = abs(((clon - sub) + 180.0) % 360.0 - 180.0)
        cosp = (math.cos(math.radians(min(max(clat, -75.0), 75.0)))
                * math.cos(math.radians(dlon)))
        ranked.append((math.degrees(math.acos(max(-1.0, min(1.0, cosp)))), num, sub))
    ranked.sort()
    # table-less satellites (rare listing oddities) go last as blind fallbacks
    for num in sorted(have):
        if num not in [r[1] for r in ranked] and num not in pos:
            ranked.append((89.9, num, None))
    return [(num, sub) for _psi, num, sub in ranked]


def load_crop_sync(url: str, var: str, bbox) -> tuple:
    """Ranged-open one GridSat-GOES file and crop ``var`` to bbox.
    Returns (data float32, lats 1-D ascending?, lons 1-D). Values: ch1 is
    reflectance (0..1), ch2..ch6 Kelvin (xarray applies scale/offset)."""
    import fsspec
    import xarray as xr
    W, S, E, N = bbox
    S2, N2 = max(S, -GG_LAT_LIMIT), min(N, GG_LAT_LIMIT)
    if S2 >= N2:
        raise CoverageError("GridSat-GOES covers 75°S-75°N")
    fs = fsspec.filesystem("https", block_size=1 << 21)
    with fs.open(url, mode="rb") as fh:
        ds = xr.open_dataset(fh, engine="h5netcdf")
        try:
            da = ds[var]
            if "time" in da.dims:
                da = da.isel(time=0)
            lat = ds["lat"].values
            lon = ds["lon"].values
            def _sl(coord, lo, hi):
                idx = np.where((coord >= lo) & (coord <= hi))[0]
                return slice(int(idx[0]), int(idx[-1]) + 1) if idx.size else slice(0, 0)
            lat_sl = _sl(lat, S2, N2)
            # the grid's lon axis spans -210..+5 style unwrapped domains: try
            # the raw window, then the +-360 aliases (covers GOES-9 at 155E)
            for shift in (0.0, -360.0, 360.0):
                lon_sl = _sl(lon, W + shift, (E if E >= W else E + 360.0) + shift)
                if lon_sl.stop > lon_sl.start:
                    break
            sub = da.isel(lat=lat_sl, lon=lon_sl).values
            lats = lat[lat_sl]
            lons = lon[lon_sl] - shift
        finally:
            ds.close()
    if sub.size == 0:
        raise CoverageError("requested box has no GridSat-GOES grid cells")
    return (np.asarray(sub, dtype=np.float32),
            lats.astype(np.float64), lons.astype(np.float64))


class GridSatGoesSatellite(Satellite):
    """Per-satellite GOES-era deep tier (see module docstring)."""
    family = "GridSat-GOES"
    sensor = "GOES Imager"
    generic_to_band = {"clean_ir": 4, "wv_upper": 3, "visible_red": 1,
                       "shortwave_ir": 2}
    disk_bbox = (-180.0, -GG_LAT_LIMIT, 180.0, GG_LAT_LIMIT)
    primary_live_bucket = GG_BUCKET
    sub_sat_lon = -75.0

    def can_see(self, bbox, time) -> bool:
        try:
            return bool(rank_candidates(bbox, slot_for(time)))
        except Exception:  # noqa: BLE001 -- listing trouble: can't promise
            return False

    def resolve(self, time: dt.datetime) -> ResolvedSatellite:
        if not (GG_START <= time < GG_END):
            raise UnsupportedTimeError(
                "GridSat-GOES spans 1994-10 .. 2017-12; requested "
                + time.isoformat())
        return ResolvedSatellite("GridSat-GOES", GG_BUCKET, -75.0)

    async def find_file(self, time, generic_channel, bbox, nearest_to_target,
                        product_hint=None) -> ResolvedFile:
        if generic_channel not in _CHANNEL_VARS:
            raise ValueError(
                f"{generic_channel!r} is not in the GridSat-GOES record "
                "(channels: visible, 3.9 µm, 6.5 µm WV, 10.7 µm IR)")
        self.resolve(time)
        return await _to_thread(self._find_file_sync, time, bbox)

    def _find_file_sync(self, time: dt.datetime, bbox) -> ResolvedFile:
        last = None
        for slot in candidate_slots(time):
            if not (GG_START <= slot < GG_END):
                continue
            try:
                cands = rank_candidates(bbox, slot)
            except RuntimeError as e:
                last = str(e)
                continue
            if not cands:
                continue
            num, sub = cands[0]
            return ResolvedFile(
                bucket=GG_BUCKET, s3_key=url_for(num, slot),
                product="GridSat-GOES", scan_start=slot,
                sat_name=f"GOES-{num}",
                sub_sat_lon=(sub if sub is not None else -75.0))
        raise CoverageError(
            "no GridSat-GOES satellite covers this box within ±3 h of "
            + time.isoformat() + (f" ({last})" if last else "")
            + " — falling back to the blended archive is the honest option")

    def open(self, resolved):   # pragma: no cover - fetch path is _fetch_sync
        raise NotImplementedError("GridSatGoesSatellite uses _fetch_sync directly")

    def project_to_latlon(self, ds, bbox, resolved, generic_channel):  # pragma: no cover
        raise NotImplementedError("GridSatGoesSatellite uses _fetch_sync directly")

    def _fetch_sync(self, resolved: ResolvedFile, bbox, generic_channel) -> FetchResult:
        var = _CHANNEL_VARS[generic_channel]
        data, lats1, lons1 = load_crop_sync(resolved.s3_key, var, bbox)
        finite_frac = float(np.isfinite(data).mean()) if data.size else 0.0
        # per-satellite disks: an off-disk crop is mostly NaN -- fail HONESTLY
        # so the selector's fallback (MergIR/GridSat-B1) takes the render. The
        # visible channel is additionally night-empty; 2% keeps dawn edges.
        min_frac = 0.02 if var == "ch1" else 0.10
        if finite_frac < min_frac:
            raise CoverageError(
                f"GridSat-GOES {resolved.sat_name} has no usable "
                f"{'visible (night side?)' if var == 'ch1' else var} data over "
                f"this box ({finite_frac:.0%} valid)")
        LON, LAT = np.meshgrid(lons1, lats1)
        return FetchResult(
            cmi=data,
            lats=LAT.astype(np.float32),
            lons=LON.astype(np.float32),
            channel=self.generic_to_band[generic_channel],
            generic_channel=generic_channel,
            scan_start=resolved.scan_start,
            product="GridSat-GOES",
            bucket=GG_BUCKET,
            sat_name=resolved.sat_name,
            sub_sat_lon=resolved.sub_sat_lon,
            units=("1" if var == "ch1" else "K"),
        )


GRIDSAT_GOES = GridSatGoesSatellite()
