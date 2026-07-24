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

ACCESS (verified against the EUMETSAT Data Store, 2026-07-11):
  - Free EUMETSAT account -> consumer key/secret at api.eumetsat.int/api-key/;
    token via POST /token (OAuth2 client credentials). Env:
    EUMETSAT_CONSUMER_KEY + EUMETSAT_CONSUMER_SECRET.
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

_token_cache = {"token": None, "expires": 0.0}


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
        raise RuntimeError("EUMETSAT_CONSUMER_KEY/SECRET not set")
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
                      pattern: str = "*.nat") -> list[str]:
    """Download one product zip and return the extracted paths matching
    ``pattern`` (SEVIRI Native = one .nat; FCI L1C = ~40 chunked *.nc)."""
    from urllib.parse import quote
    url = (f"{API}/data/download/1.0.0/collections/"
           f"{quote(collection, safe='')}/products/{quote(product_id, safe='')}")
    zpath = os.path.join(dest_dir, "product.zip")
    with requests.get(url, headers={"Authorization": f"Bearer {_token()}"},
                      stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(zpath, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest_dir)
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

    feat = _search_latest(collection, not_after)
    pid = feat["properties"]["identifier"]
    date = feat["properties"]["date"]
    scan_end = dt.datetime.fromisoformat(date.split("/")[1].replace("Z", "+00:00"))

    tmp = tempfile.mkdtemp(prefix="seviri_")
    try:
        nats = _download_product(collection, pid, tmp)
        from satpy import Scene
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
                   datasets=FCI_VIS_DATASETS + (FCI_IR_DATASET,)) -> FciDisk:
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
    """
    delay = (float(os.getenv("EUMETSAT_FCI_DELAY_MIN",
                             os.getenv("EUMETSAT_DELAY_MIN", "60")))
             if delay_min is None else float(delay_min))
    now = dt.datetime.now(UTC)
    not_after = now - dt.timedelta(minutes=delay)
    if time is not None:
        t = time if time.tzinfo else time.replace(tzinfo=UTC)
        not_after = min(not_after, t)

    feat = _search_latest(COLLECTION_FCI, not_after)
    pid = feat["properties"]["identifier"]
    date = feat["properties"]["date"]
    scan_end = dt.datetime.fromisoformat(date.split("/")[1].replace("Z", "+00:00"))

    tmp = tempfile.mkdtemp(prefix="fci_")
    try:
        chunks = _download_product(COLLECTION_FCI, pid, tmp, pattern="*.nc")
        from satpy import Scene
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
