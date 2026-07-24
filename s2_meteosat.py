#!/usr/bin/env python3
"""Meteosat members for the GEO-RING (Stage-2): SEVIRI BT + MTG FCI true color.

Fills the ring's Africa / Europe / Indian-Ocean gap with the two operational
SEVIRI services: Meteosat 0° prime-backup (EO:EUM:DAT:MSG:HRSEVIRI, currently
Meteosat-10) and Meteosat IODC at 45.5°E (EO:EUM:DAT:MSG:HRSEVIRI-IODC,
currently Meteosat-9) -- BT (IR/WV) members only: SEVIRI has no blue/green
band, so it CANNOT do true color (a SEVIRI visible composite is "natural
color" and stays out of the true-color ring by policy).

The TRUE-COLOR Meteosat member is MTG-I1 FCI (``fetch_fci_disk`` below,
COLLECTION_FCI): vis_04/05/06/08 + ir_105 through the shared
truecolor.assemble_truecolor pipeline with sensor="fci" (NDVI-hybrid green,
GK-2A/GOES/Himawari-identical tone curve). Dormant until EUMETSAT creds land
on the box AND satpy (requirements-s2-geo.txt) is installed in the emit
image -- both are queued manual/activation steps; until then every fetch
raises and the sector stays an honest transparent gap.

ACCESS (verified against the EUMETSAT Data Store, 2026-07-11; auth path
re-verified 2026-07-24):
  - Free EUMETSAT account -> consumer key/secret at api.eumetsat.int/api-key/;
    token via POST /token (OAuth2 client credentials). Env:
    EUMETSAT_CONSUMER_KEY + EUMETSAT_CONSUMER_SECRET.
  - AUTH MIGRATION NOTE (2026-07-24): the api-key page banners a "new
    authentication method" -- that is the v2 Data Access Services flow
    (OAuth2 Authorization Code + PKCE against user.eumetsat.int/cas, live
    since 2026-06-30). It requires an INTERACTIVE browser login to bootstrap
    a 30-day refresh-token chain, so it is unsuitable for this headless box
    today. Our client-credentials flow at POST /token is the documented v1
    method, still fully supported ("will be depreciated in due time" -- no
    date anywhere), and is exactly what the official eumdac client (3.1.1,
    2025-12) still ships. Stay on v1 until eumdac gains v2 support or
    EUMETSAT announces a deadline; the v2 delta is small (endpoints 1.0.0 ->
    2.0.0 + the refresh-token mint) and is documented in the Data Store
    detailed guide on user.eumetsat.int.
  - LICENCE GATE: collection downloads 403 with "GeneralLicense required to
    access this collection" until the account holder accepts the EUMETSAT
    General Licence on user.eumetsat.int (self-service; activation can lag
    up to 1 h). One acceptance covers FCI + both SEVIRI services. Search
    works without it, so slot pinning succeeds and every download honestly
    degrades until the click.
  - The account must have accepted the "Meteosat Level 1 data with latency
    >= 1 hour" licence (free, self-service on user.eumetsat.int). That
    licence allows use FOR ANY PURPOSE with attribution but not
    redistribution of the original numerical data -- we publish derived
    rendered tiles, which is fine, and we do NOT publish the .nat files.
  - TRUE NRT (<1 h) needs the paid NRT/Service-Provider licence -- so this
    module fetches the newest slot at least EUMETSAT_DELAY_MIN (default 60)
    minutes old. The composite blends members at different valid times
    honestly: per-member stamps ride the manifest (`members`).

If credentials or satpy are absent, fetches raise and the composite's
per-member degrade keeps the sector an HONEST transparent gap -- exactly the
pre-Meteosat behavior. Nothing is ever faked.

Attribution (licence requirement): "Contains EUMETSAT Meteosat data".
"""
from __future__ import annotations

import base64
import datetime as dt
import glob
import os
import shutil
import tempfile
import time as _time
import zipfile

import numpy as np
import requests

UTC = dt.timezone.utc

API = "https://api.eumetsat.int"
COLLECTION_0DEG = "EO:EUM:DAT:MSG:HRSEVIRI"
COLLECTION_IODC = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"

# MTG-I1 (Meteosat-12) FCI L1C FDHSI at 0 deg -- the TRUE-COLOR-capable
# Meteosat (SEVIRI has no blue/green; any SEVIRI visible product is "natural
# color" and stays OUT of the true-color ring). Verified live on the Data
# Store 2026-07-24: ~93k products, 10-min repeat cycles, ~470 MB zips of
# chunked netCDF read by satpy's fci_l1c_nc reader.
COLLECTION_FCI = "EO:EUM:DAT:0662"

# generic field key -> SEVIRI dataset name (satpy seviri_l1b_native)
SEVIRI_DATASETS = {"ir": "IR_108", "irbd": "IR_108", "wv": "WV_062"}
_EAGER_DATASETS = ("IR_108", "WV_062")   # one .nat download serves all fields

# FCI datasets for the shared true-color pipeline (truecolor.assemble_truecolor
# sensor="fci": red vis_06, REAL 0.51um green vis_05, blue vis_04, NIR vis_08
# for the NDVI-hybrid green, ir_105 for the terminator IR cross-fade).
FCI_VIS_DATASETS = ("vis_04", "vis_05", "vis_06", "vis_08")
FCI_IR_DATASET = "ir_105"
FCI_TRUECOLOR_ROLES = {"blue": "vis_04", "green": "vis_05", "red": "vis_06",
                       "veggie": "vis_08", "ir": FCI_IR_DATASET}
FCI_PLATFORM = "Meteosat-12"   # pyspectral RSR platform name (MTG-I1)

# Registry/recipe numeric convention (mirrors AMI/AHI: 1=blue, 2=green,
# 3=red, 4=veggie NIR, 13=clean IR) so the mtgi1-fd rows read like every
# other suite and the explorer's cross-satellite key mapping holds.
FCI_BAND_TOKENS = {1: "vis_04", 2: "vis_05", 3: "vis_06", 4: "vis_08",
                   13: FCI_IR_DATASET}

# FCI FDHSI repeat cycle is 10 min; the emit backfill grid + slot-covered
# tolerance both key off this.
FCI_CADENCE_MIN = 10

# COMPLETENESS GATE (never-miss): one FDHSI FD product is a zip of 41
# chunked netCDFs (40 CHK-BODY strips + 1 CHK-TRAIL) and satpy's fci_l1c_nc
# needs ALL of them -- a partial set decodes into a disk with silent missing
# strips. fetch_fci_disk refuses to decode unless the extracted chunk set is
# contiguous and complete, so a truncated download or a mid-upload product
# can never render. Env override for schema drift, never for convenience.
FCI_EXPECTED_CHUNKS = int(os.getenv("EUMETSAT_FCI_EXPECTED_CHUNKS", "41"))

_token_cache = {"token": None, "expires": 0.0}


class _FatalDownloadError(RuntimeError):
    """A download failure that retrying cannot fix (4xx: licence gate,
    vanished product). Propagates out of the retry loop with the server's
    own message intact."""


def credentials():
    k = os.getenv("EUMETSAT_CONSUMER_KEY", "").strip()
    s = os.getenv("EUMETSAT_CONSUMER_SECRET", "").strip()
    return (k, s) if k and s else None


def available() -> bool:
    """Creds present AND satpy importable -- the gate the ring member checks."""
    if credentials() is None:
        return False
    try:
        import satpy  # noqa: F401
        return True
    except ImportError:
        return False


def _token() -> str:
    now = _time.time()
    if _token_cache["token"] and now < _token_cache["expires"] - 60:
        return _token_cache["token"]
    creds = credentials()
    if creds is None:
        # _FatalDownloadError so the download retry loop fails fast instead
        # of burning attempts on a condition retrying can never fix (still a
        # RuntimeError to every existing caller's contract)
        raise _FatalDownloadError("EUMETSAT_CONSUMER_KEY/SECRET not set")
    basic = base64.b64encode(f"{creds[0]}:{creds[1]}".encode()).decode()
    r = requests.post(f"{API}/token",
                      headers={"Authorization": f"Basic {basic}"},
                      data={"grant_type": "client_credentials"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    _token_cache["token"] = j["access_token"]
    _token_cache["expires"] = now + float(j.get("expires_in", 3600))
    return _token_cache["token"]


def _search_latest(collection: str, not_after: dt.datetime,
                   lookback_h: float = 6.0) -> dict:
    """Newest product in `collection` whose sensing END <= not_after.
    Returns the OpenSearch feature (id + times). Raises if none."""
    dtstart = (not_after - dt.timedelta(hours=lookback_h))
    r = requests.get(
        f"{API}/data/search-products/1.0.0/os",
        params={"format": "json", "pi": collection,
                "dtstart": dtstart.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dtend": not_after.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sort": "start,time,0", "c": 10},
        timeout=60)
    r.raise_for_status()
    feats = (r.json().get("features") or [])
    for f in feats:
        # date field: "start/end" ISO pair; require end <= not_after (the
        # dtend filter matches on START, so re-check honestly)
        try:
            date = f["properties"]["date"]
            end = dt.datetime.fromisoformat(
                date.split("/")[1].replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001 -- schema drift -> skip, try next
            continue
        if end <= not_after:
            return f
    raise RuntimeError(f"{collection}: no product ends before "
                       f"{not_after.isoformat()} in the last {lookback_h} h")


def _download_product(collection: str, product_id: str, dest_dir: str,
                      pattern: str = "*.nat", attempts: int = 3) -> list[str]:
    """Download one product zip and return the extracted paths matching
    ``pattern`` (SEVIRI Native = one .nat; FCI L1C = 41 chunked *.nc).

    An FCI product is ~800 MB over one HTTP stream, so a dropped connection
    mid-transfer is a matter of time, not chance. The stream lands in a
    .part file; a short read RESUMES with a Range request from the byte
    already on disk (the Data Store supports Range -- verified 2026-07-24),
    and a zip that still fails to open after completion is discarded for a
    clean full retry. The zip only becomes product.zip once its size checks
    out, so extraction never sees a half-written file."""
    from urllib.parse import quote
    url = (f"{API}/data/download/1.0.0/collections/"
           f"{quote(collection, safe='')}/products/{quote(product_id, safe='')}")
    part = os.path.join(dest_dir, "product.zip.part")
    zpath = os.path.join(dest_dir, "product.zip")
    total = None            # Content-Length of the full object, once known
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            got = os.path.getsize(part) if os.path.exists(part) else 0
            if total is not None and got >= total:
                pass        # transfer already complete; go verify the zip
            else:
                headers = {"Authorization": f"Bearer {_token()}"}
                if got:
                    headers["Range"] = f"bytes={got}-"
                with requests.get(url, headers=headers, stream=True,
                                  timeout=900) as r:
                    if got and r.status_code == 200:
                        got = 0     # server ignored Range: restart the file
                    if 400 <= r.status_code < 500:
                        # a 4xx never heals on retry, and its body carries
                        # the actionable message (e.g. the licence gate's
                        # "GeneralLicense required to access this
                        # collection") -- surface it verbatim
                        raise _FatalDownloadError(
                            f"{product_id}: HTTP {r.status_code}: "
                            f"{r.text[:300]}")
                    r.raise_for_status()
                    if total is None:
                        cl = r.headers.get("Content-Length")
                        if cl and r.status_code == 200:
                            total = int(cl)
                        elif cl and r.status_code == 206:
                            total = got + int(cl)
                    with open(part, "ab" if got else "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
            size = os.path.getsize(part) if os.path.exists(part) else 0
            if total is not None and size < total:
                raise IOError(f"short read: {size}/{total} bytes")
            os.replace(part, zpath)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(dest_dir)
            break
        except zipfile.BadZipFile as e:
            # complete-length but corrupt: resume can't fix it -- start over
            last_err = e
            for p in (part, zpath):
                if os.path.exists(p):
                    os.remove(p)
            total = None
        except _FatalDownloadError:
            raise               # 4xx: retrying cannot help, message matters
        except Exception as e:  # noqa: BLE001 -- timeouts, resets, short reads
            last_err = e        # .part stays for the Range resume
            if os.path.exists(zpath) and not os.path.exists(part):
                os.replace(zpath, part)   # extraction failed post-rename
    else:
        raise RuntimeError(
            f"{product_id}: download failed after {attempts} attempts: {last_err}")
    os.remove(zpath)
    paths = sorted(glob.glob(os.path.join(dest_dir, "**", pattern), recursive=True))
    if not paths:
        raise RuntimeError(f"{product_id}: zip contained no {pattern} file")
    return paths


class SeviriDisk:
    """Calibrated SEVIRI channel grids + the pyresample area for sampling."""
    __slots__ = ("values", "area", "scan_end", "sat_name", "collection")

    def __init__(self, values, area, scan_end, sat_name, collection):
        self.values = values          # {dataset_name: float32 BT Kelvin grid}
        self.area = area              # pyresample AreaDefinition
        self.scan_end = scan_end      # tz-aware UTC
        self.sat_name = sat_name
        self.collection = collection

    def sample_bt(self, dataset: str, TLAT: np.ndarray, TLON: np.ndarray) -> np.ndarray:
        """Bilinear-sample a channel at lat/lon query points; NaN off-disk."""
        return _sample_area(self.area, self.values[dataset], TLAT, TLON)


def _sample_area(area, vals: np.ndarray, TLAT: np.ndarray, TLON: np.ndarray) -> np.ndarray:
    """Bilinear-sample a pyresample-area grid at lat/lon query points; NaN
    off-disk. Shared by the SEVIRI BT member and the FCI true-color member
    (the sentinel-poisoning trick keeps a bilinear touch of any space pixel
    honest NaN instead of a half-real value)."""
    from scipy.ndimage import map_coordinates
    h, w = vals.shape
    cols, rows = _float_indices(area, TLON, TLAT)
    inb = (np.isfinite(cols) & np.isfinite(rows) &
           (cols >= 0) & (cols <= w - 1) & (rows >= 0) & (rows <= h - 1))
    out = np.full(TLAT.shape, np.nan, np.float32)
    if inb.any():
        coords = np.stack([np.where(inb, rows, 0).ravel(),
                           np.where(inb, cols, 0).ravel()])
        samp = map_coordinates(np.nan_to_num(vals, nan=-1e9), coords,
                               order=1, mode="nearest").reshape(TLAT.shape)
        # a bilinear touch of any space-pixel sentinel poisons the sample
        out = np.where(inb & (samp > -1e8), samp, np.nan).astype(np.float32)
    return out


def _float_indices(area, lons, lats):
    """Float (col, row) image indices for lon/lat -- prefers pyresample's
    float API, falls back to projection math via area attributes."""
    try:
        cols, rows = area.get_array_coordinates_from_lonlat(lons, lats)
        return (np.ma.filled(cols, np.nan).astype(np.float64),
                np.ma.filled(rows, np.nan).astype(np.float64))
    except AttributeError:
        pass
    try:  # integer API (older pyresample): good to +-0.5 px (3 km) -- fine
        cols, rows = area.get_array_indices_from_lonlat(lons, lats)
        return (np.ma.filled(cols.astype(np.float64), np.nan),
                np.ma.filled(rows.astype(np.float64), np.nan))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"pyresample lonlat->index API unavailable: {e}")


def _sweep_stale_tmp(prefix: str, max_age_s: float = 7200.0) -> None:
    """An OOM-kill mid-fetch (mem_limit SIGKILL bypasses ``finally``)
    strands a multi-GB temp dir in the long-lived emit container, and
    nothing else ever sweeps it. Reap same-prefix siblings older than
    max_age_s before starting a new fetch -- an active fetch is always
    younger than that."""
    now = _time.time()
    for p in glob.glob(os.path.join(tempfile.gettempdir(), prefix + "*")):
        try:
            if now - os.path.getmtime(p) > max_age_s:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def fetch_seviri_disk(collection: str, time=None, delay_min=None) -> SeviriDisk:
    """Fetch + decode the newest licence-compliant SEVIRI slot for a service.

    time: pin near this UTC time (archive use); None = newest allowed.
    delay_min: minimum data age in minutes (default env EUMETSAT_DELAY_MIN
    or 60 -- the free >=1 h-latency licence). Set 0 only with an NRT licence.
    """
    delay = float(os.getenv("EUMETSAT_DELAY_MIN", "60")) if delay_min is None \
        else float(delay_min)
    now = dt.datetime.now(UTC)
    not_after = now - dt.timedelta(minutes=delay)
    if time is not None:
        t = time if time.tzinfo else time.replace(tzinfo=UTC)
        not_after = min(not_after, t)

    from satpy import Scene   # fail BEFORE the ~270 MB transfer, not after

    feat = _search_latest(collection, not_after)
    pid = feat["properties"]["identifier"]
    date = feat["properties"]["date"]
    scan_end = dt.datetime.fromisoformat(date.split("/")[1].replace("Z", "+00:00"))

    _sweep_stale_tmp("seviri_")
    tmp = tempfile.mkdtemp(prefix="seviri_")
    try:
        nats = _download_product(collection, pid, tmp)
        scn = Scene(filenames=nats, reader="seviri_l1b_native")
        scn.load(list(_EAGER_DATASETS), calibration="brightness_temperature")
        values = {}
        area = None
        for ds in _EAGER_DATASETS:
            da = scn[ds]
            values[ds] = np.asarray(da.values, np.float32)
            area = da.attrs.get("area", area)
        sat_name = str(scn[_EAGER_DATASETS[0]].attrs.get("platform_name", "MSG"))
        if area is None:
            raise RuntimeError("seviri_l1b_native returned no area definition")
        return SeviriDisk(values, area, scan_end, sat_name, collection)
    finally:
        # arrays are eagerly materialized above -- the .nat can go at once
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# MTG FCI (Meteosat-12): the TRUE-COLOR Meteosat ring member
# ---------------------------------------------------------------------------
def _verify_fci_chunks(paths: list[str], product_id: str) -> None:
    """The never-miss completeness gate: refuse to decode a partial FDHSI
    chunk set. Chunk files end in _NNNN.nc (body strips numbered from 1,
    plus the trailer); the set must be CONTIGUOUS from 1 and total
    FCI_EXPECTED_CHUNKS files, else this slot does not render (the emit
    backfill retries it next tick -- a gap is honest, a half disk is not)."""
    import re
    idx = []
    for p in paths:
        m = re.search(r"_(\d{4})\.nc$", os.path.basename(p))
        if m:
            idx.append(int(m.group(1)))
    n = len(paths)
    if n != FCI_EXPECTED_CHUNKS:
        raise RuntimeError(
            f"{product_id}: incomplete FCI chunk set -- {n} *.nc files, "
            f"expected {FCI_EXPECTED_CHUNKS} (env EUMETSAT_FCI_EXPECTED_CHUNKS "
            f"overrides if the product schema ever changes)")
    if idx:
        want = set(range(1, max(idx) + 1))
        missing = sorted(want - set(idx))
        if missing:
            raise RuntimeError(
                f"{product_id}: FCI chunk numbering has gaps -- missing "
                f"indices {missing[:8]}{'...' if len(missing) > 8 else ''}")


def newest_fci_slot(time=None, delay_min=None):
    """Sensing-END datetime of the newest licence-compliant FCI repeat cycle
    (the suite pin -- mirrors s2_gk2a.newest_complete_slot). With ``time``,
    resolves the cycle covering that backfill slot: newest product ending
    <= time + FCI_CADENCE_MIN/2 (the emit grid's own covered-tolerance), so
    a slot the licence delay still embargoes resolves to an older, already-
    emitted cycle and dedups instead of erroring. Returns None if the
    search window is empty (caller decides how loud to be). Needs NO
    credentials -- OpenSearch is open, so pinning works even while the
    licence gate still 403s the download (honest-degrade per product)."""
    delay = (float(os.getenv("EUMETSAT_FCI_DELAY_MIN",
                             os.getenv("EUMETSAT_DELAY_MIN", "60")))
             if delay_min is None else float(delay_min))
    not_after = dt.datetime.now(UTC) - dt.timedelta(minutes=delay)
    if time is not None:
        t = time if time.tzinfo else time.replace(tzinfo=UTC)
        not_after = min(not_after, t + dt.timedelta(minutes=FCI_CADENCE_MIN / 2.0))
    try:
        feat = _search_latest(COLLECTION_FCI, not_after)
    except (RuntimeError, requests.RequestException):
        # empty window OR a transient OpenSearch failure (5xx, timeout,
        # reset) -- both mean "no pin this tick", never an uncaught abort
        return None
    date = feat["properties"]["date"]
    return dt.datetime.fromisoformat(date.split("/")[1].replace("Z", "+00:00"))


class FciDisk:
    """Calibrated FCI L1C fields + per-resolution pyresample areas.

    ``values`` holds float32 grids keyed by dataset name: the four solar
    bands as TOA reflectance FACTOR 0..1 (satpy calibration='reflectance'
    yields percent; divided by 100 here so the shared truecolor pipeline
    gets the same units as ABI CMI / AHI albedo -- NOT sun-normalized), and
    ir_105 as brightness temperature (Kelvin). FDHSI feeds every solar band
    at 1 km class, so the true-color self-sharpen is a documented no-op
    (truecolor.SHARPEN_BLOCK['fci'] == 1)."""
    __slots__ = ("values", "areas", "scan_end", "sat_name", "collection")

    def __init__(self, values, areas, scan_end, sat_name, collection):
        self.values = values          # {dataset: float32 grid}
        self.areas = areas            # {dataset: pyresample AreaDefinition}
        self.scan_end = scan_end      # tz-aware UTC
        self.sat_name = sat_name      # e.g. "Meteosat-12" / "MTG-I1"
        self.collection = collection

    def sample(self, dataset: str, TLAT: np.ndarray, TLON: np.ndarray) -> np.ndarray:
        """Bilinear-sample one field at lat/lon query points; NaN off-disk."""
        return _sample_area(self.areas[dataset], self.values[dataset], TLAT, TLON)


def fetch_fci_disk(time=None, delay_min=None,
                   datasets=FCI_VIS_DATASETS + (FCI_IR_DATASET,),
                   slot_tolerance_min=None) -> FciDisk:
    """Fetch + decode the newest licence-compliant MTG FCI L1C repeat cycle.

    Mirrors fetch_seviri_disk: same creds/token/OpenSearch/download client,
    same honest-degrade contract (raises when creds/satpy absent -- the
    caller keeps the sector transparent, nothing is faked). The product is a
    ~470 MB zip of ~40 chunked netCDFs; satpy's ``fci_l1c_nc`` reader
    assembles the full disk. Solar bands load as reflectance percent ->
    stored /100; ir_105 as brightness temperature (K). Memory note: five
    eager FDHSI fields is ~2.5 GB of float32 -- activation on the geo lane
    should re-check the emit container's mem_limit before joining the cron.

    time: pin near this UTC time (archive use); None = newest allowed.
    delay_min: minimum data age in minutes (default env EUMETSAT_FCI_DELAY_MIN
    or EUMETSAT_DELAY_MIN or 60). Set 0 only with an NRT-tier licence.
    slot_tolerance_min: with ``time``, refuse a product whose sensing end is
    further than this from the requested slot (the cron producer passes the
    FCI cadence so a pin/fetch race can never render the wrong cycle);
    None = legacy nearest-before behavior for archive use.
    """
    delay = (float(os.getenv("EUMETSAT_FCI_DELAY_MIN",
                             os.getenv("EUMETSAT_DELAY_MIN", "60")))
             if delay_min is None else float(delay_min))
    now = dt.datetime.now(UTC)
    not_after = now - dt.timedelta(minutes=delay)
    if time is not None:
        t = time if time.tzinfo else time.replace(tzinfo=UTC)
        not_after = min(not_after, t)

    from satpy import Scene   # fail BEFORE the ~800 MB transfer, not after

    feat = _search_latest(COLLECTION_FCI, not_after)
    pid = feat["properties"]["identifier"]
    date = feat["properties"]["date"]
    scan_end = dt.datetime.fromisoformat(date.split("/")[1].replace("Z", "+00:00"))
    if slot_tolerance_min is not None and time is not None:
        t = time if time.tzinfo else time.replace(tzinfo=UTC)
        if abs((scan_end - t).total_seconds()) > slot_tolerance_min * 60.0:
            raise RuntimeError(
                f"FCI: no repeat cycle within {slot_tolerance_min} min of "
                f"{t.isoformat()} (nearest ends {scan_end.isoformat()})")

    _sweep_stale_tmp("fci_")
    tmp = tempfile.mkdtemp(prefix="fci_")
    try:
        chunks = _download_product(COLLECTION_FCI, pid, tmp, pattern="*.nc")
        _verify_fci_chunks(chunks, pid)
        scn = Scene(filenames=chunks, reader="fci_l1c_nc")
        vis = [d for d in datasets if d != FCI_IR_DATASET]
        if vis:
            scn.load(vis, calibration="reflectance")
        if FCI_IR_DATASET in datasets:
            scn.load([FCI_IR_DATASET], calibration="brightness_temperature")
        values, areas = {}, {}
        for ds in datasets:
            da = scn[ds]
            arr = np.asarray(da.values, np.float32)
            if ds != FCI_IR_DATASET:
                arr = arr / 100.0          # percent -> reflectance factor
            values[ds] = arr
            areas[ds] = da.attrs.get("area")
            if areas[ds] is None:
                raise RuntimeError(f"fci_l1c_nc returned no area for {ds}")
        sat_name = str(scn[datasets[0]].attrs.get("platform_name", FCI_PLATFORM))
        return FciDisk(values, areas, scan_end, sat_name, COLLECTION_FCI)
    finally:
        # arrays are eagerly materialized above -- the chunk zip can go at once
        shutil.rmtree(tmp, ignore_errors=True)
