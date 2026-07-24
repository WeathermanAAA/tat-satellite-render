#!/usr/bin/env python3
"""GK-2A AMI L1B ingestion for the S2 true-color suite (Stage-2).

Adds GEO-KOMPSAT-2A (128.2E) to the GEO ring between Himawari (140.7E) and
Meteosat IODC (45.5E). AMI is an AHI sibling: same GEOS y-sweep viewing
geometry, a REAL 0.51 um green band (no synthesis), 10-minute full-disk
cadence. ``truecolor.assemble_truecolor`` already carries the "ami" sensor
row (VI006/VI005/VI004 + VI008 veggie), so this module only has to deliver
calibrated, co-registerable band grids -- the display pipeline is shared.

ACCESS (verified against the live bucket, 2026-07):
  - Public anonymous NOAA NODD bucket, no credentials, no licence delay:
      https://noaa-gk2a-pds.s3.amazonaws.com
    keys  AMI/L1B/FD/{YYYYMM}/{DD}/{HH}/
          gk2a_ami_le1b_{band}_fd{res}ge_{YYYYMMDDHHMM}.nc
    10-minute slots (..00, ..10, ...). One plain .nc per band per slot
    (no segment stitching, unlike AHI HSD). Listing is anonymous REST
    (?list-type=2), parsed with xml.etree -- NO boto3.
  - Sizes: ir105 fd020ge ~35 MB; vi005 fd010ge ~150 MB; vi006 fd005ge
    ~470 MB (a 22000x22000 uint16 grid). Downloads are streamed to a
    tempfile, decoded eagerly, and the tempfile is deleted at once
    (the s2_meteosat tempdir-hygiene pattern). For suite-scale work pass
    ``stride`` so the 0.5 km red never materializes at native size.

CALIBRATION -- from in-file attrs ONLY (no hardcoded lookup tables), per
the NMSC "GK-2A AMI L1B Data User Manual" formulas:
  VIS/NIR ("albedo"):  radiance = DN * DN_to_Radiance_Gain
                                  + DN_to_Radiance_Offset      [W m-2 sr-1 um-1]
                       albedo   = radiance * Radiance_to_Albedo_c, clipped >= 0.
    This is the TOA reflectance FACTOR (0..1, not sun-normalized, can
    exceed 1 in glint) -- exactly what assemble_truecolor expects; it does
    its own sun-zenith correction.
  IR ("bt"):  radiance = DN * gain + offset      [mW m-2 sr-1 (cm-1)-1]
              inverse Planck in wavenumber form using the file's own
              light_speed / Plank_constant_h / Boltzmann_constant_k and
              nu = 1 / channel_center_wavelength (the file carries the
              CENTER wavelength, not an effective wavenumber -- the
              Teff_to_Tbb_c0/c1/c2 band-correction polynomial is exactly
              the term that absorbs that difference):
                Teff = (h c nu / k) / ln(1 + 2 h c^2 nu^3 / L_SI)
                Tbb  = c0 + c1*Teff + c2*Teff^2
              with L_SI = radiance * 1e-5 (mW->W, per cm-1 -> per m-1).
  DN masking is self-described by the variable attrs: 16 bits total, the
  TOP number_of_data_quality_flag_bits_per_pixel (2) bits are the DQF
  (0 good, 1 conditionally usable, 2 out of scan, 3 error); the LOW
  number_of_valid_bits_per_pixel bits are the count. DQF>=2 -> NaN.
  Conditionally-usable pixels (DQF 1) are KEPT -- degraded data beats a
  speckle hole (satpy's default drops them; flagged as our departure).
  Non-positive IR radiance (counts past the calibration cold end) -> NaN.

TIME: ``observation_start_time`` is float seconds since the J2000 epoch
2000-01-01T12:00:00Z -- NOON, not midnight; the naive midnight epoch reads
12 h early. Verified against ``scene_acquisition_time`` ("YYYYMMDD_HHMMSS",
whole seconds) to the exact second on real ir105 + vi005 files
(838058432.1166971 -> 2026-07-23T06:00:32.117Z == "20260723_060032").
We parse observation_start_time (sub-second, numeric), cross-check it
against scene_acquisition_time, and fall back to the string if the two
disagree by more than 2 minutes (epoch drift would be a whole-file bug,
never a rounding one).

NAVIGATION (CGMS LRIT/HRIT Global Spec 4.4, y-sweep GEOS like AHI/SEVIRI,
NOT GOES ABI's x-sweep in satellites.py):
    col = COFF + x_deg * CFAC / 2**16,   row = LOFF + y_deg * LFAC / 2**16
  GK-2A stores CFAC > 0 and LFAC < 0 with x east-positive and y
  NORTH-positive scan angles (y = +arcsin(r3/rn); the AHI helper in
  satellites.py uses the sign-flipped pairing y = arcsin(-r3/rn) with a
  positive LFAC -- both make the row axis increase southward, row 1 north).
  Pixel numbering is 1-BASED CGMS pixel centers: the file's
  image_upperleft_x/y scan angles land on col/row exactly 1.0 for both the
  2 km and 1 km grids, so array index = col - 1. ``sub_longitude`` is in
  RADIANS in the file (2.2375121 rad = 128.2E) -- converted on decode.
  Earth-far-side points (horizon test cos(c_lat)*cos(dlon) < r_l/h) are
  NaN'd in the forward projection, so ``sample`` is honest off-disk.

Honest degrade: no fabrication anywhere. A missing slot, a failed
download, an off-disk query, an unparseable file -- each raises or NaNs;
the composite's per-member degrade turns that into a transparent gap.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import requests

log = logging.getLogger("tat-satellite.s2_gk2a")

UTC = dt.timezone.utc

S3_BASE = "https://noaa-gk2a-pds.s3.amazonaws.com"

AMI_PLATFORM = "GEO-KOMPSAT-2A"     # pyspectral / truecolor platform_name
AMI_SENSOR = "ami"                  # truecolor.SENSOR_BANDS key

# true-color role -> AMI band (mirrors HIMAWARI truecolor_bands + clean-IR
# night fade). VI005 is a REAL green -- no ABI-style synthesis.
TRUECOLOR_BANDS = {
    "red": "vi006",      # 0.64 um, 0.5 km
    "green": "vi005",    # 0.51 um, 1 km
    "blue": "vi004",     # 0.47 um, 1 km
    "veggie": "vi008",   # 0.86 um NIR, 1 km (hybrid-green correction)
    "ir": "ir105",       # 10.5 um clean IR, 2 km (terminator fade)
}

# band -> nominal-resolution token in the file name ("fd{res}ge"), i.e.
# resolution in units of 0.1 km. Full AMI set per the NMSC L1B manual;
# only the five TRUECOLOR_BANDS entries are exercised against the live
# bucket so far -- the rest are mechanical key-building.
BAND_RES = {
    "vi004": "010", "vi005": "010", "vi006": "005", "vi008": "010",
    "nr013": "020", "nr016": "020", "sw038": "020",
    "wv063": "020", "wv069": "020", "wv073": "020",
    "ir087": "020", "ir096": "020", "ir105": "020",
    "ir112": "020", "ir123": "020", "ir133": "020",
}

# observation_start_time epoch: J2000 = 2000-01-01T12:00:00Z (NOON -- see
# the TIME section of the module docstring before "fixing" this).
_EPOCH_J2000 = dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)

# GEOS constants (km) -- defaults equal to the in-file attrs, which win
# when present (earth_equatorial_radius / earth_polar_radius /
# nominal_satellite_height; the last is the GEOCENTRIC distance, 42164 km).
_R_EQ_KM = 6378.137
_R_POL_KM = 6356.7523
_H_KM = 42164.0

_SLOT_RE = re.compile(r"_(\d{12})\.nc$")


# ---------------------------------------------------------------------------
# registry/imagery glue: the suite layer speaks NATIVE BAND NUMBERS (the
# AHI-mirroring layout s2_recipes.AMI_BAND_NATIVE_KM documents); this module
# speaks file tokens. One table, one alias, one slot-pinning helper.
# ---------------------------------------------------------------------------
BAND_TOKENS = {1: "vi004", 2: "vi005", 3: "vi006", 4: "vi008",
               5: "nr013", 6: "nr016", 7: "sw038", 8: "wv063", 9: "wv069",
               10: "wv073", 11: "ir087", 12: "ir096", 13: "ir105",
               14: "ir112", 15: "ir123", 16: "ir133"}
TRUECOLOR_ROLE_BANDS = {"red": 3, "green": 2, "blue": 1, "veggie": 4, "ir": 13}
PLATFORM_NAME = AMI_PLATFORM

# suite-render read strides: the gk2a-fd truecolor renders at the 2 km-class
# raster (s2_registry), so band reads decimate to ~2 km effective on read --
# vi006 (0.5 km, ~1.9 GB at stride 1) becomes ~120 MB at stride 4. ir105 is
# 2 km native (stride 1).
SUITE_STRIDE = {"vi006": 4, "vi004": 2, "vi005": 2, "vi008": 2}


def newest_complete_slot(bands, time=None, nearest=True) -> dt.datetime:
    """Newest 10-min slot with a file present for EVERY band token in
    ``bands`` (the suite pin: rendering one product across two scans would
    split the suite in time). Lists the target hour + the previous (+ next
    for pinned times); raises RuntimeError when no common slot exists."""
    t = time or dt.datetime.now(UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    hours = [t - dt.timedelta(hours=1), t]
    if time is not None:
        hours.append(t + dt.timedelta(hours=1))
    per_band: dict = {b: {} for b in bands}
    for prefix in dict.fromkeys(_hour_prefix(h) for h in hours):
        keys = _list_keys(prefix)
        for b in bands:
            per_band[b].update(_band_slots(keys, b))
    common = set.intersection(*(set(v) for v in per_band.values())) \
        if per_band else set()
    if not common:
        raise RuntimeError(
            f"no complete GK-2A slot for bands {sorted(bands)} near "
            f"{t:%Y-%m-%d %H:%M}Z")
    return _pick_slot({s: "" for s in common}, t, nearest)


# ---------------------------------------------------------------------------
# bucket keys + listing
# ---------------------------------------------------------------------------

def slot_key(band: str, slot: dt.datetime) -> str:
    """Exact bucket key for one band at one 10-minute slot."""
    res = BAND_RES[band]
    return (f"AMI/L1B/FD/{slot:%Y%m}/{slot:%d}/{slot:%H}/"
            f"gk2a_ami_le1b_{band}_fd{res}ge_{slot:%Y%m%d%H%M}.nc")


def _hour_prefix(t: dt.datetime) -> str:
    return f"AMI/L1B/FD/{t:%Y%m}/{t:%d}/{t:%H}/"


def _list_keys(prefix: str, timeout: float = 60) -> list:
    """Anonymous REST listing (?list-type=2) of one prefix -> object keys.

    Namespace-agnostic XML walk (tags arrive as ``{ns}Key``); follows
    continuation tokens even though a full FD hour (6 slots x 16 bands)
    fits well inside one max-keys=1000 page.
    """
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    keys = []
    while True:
        r = requests.get(f"{S3_BASE}/", params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        token = None
        truncated = False
        for el in root.iter():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "Key" and el.text:
                keys.append(el.text)
            elif tag == "IsTruncated":
                truncated = (el.text or "").strip().lower() == "true"
            elif tag == "NextContinuationToken":
                token = el.text
        if not truncated or not token:
            return keys
        params["continuation-token"] = token


def _band_slots(keys, band: str) -> dict:
    """{slot datetime (UTC): key} for one band from a raw key listing."""
    token = f"_le1b_{band}_fd"
    out = {}
    for key in keys:
        if token not in key:
            continue
        m = _SLOT_RE.search(key)
        if not m:
            continue
        try:
            slot = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC)
        except ValueError:
            continue
        out[slot] = key
    return out


def _pick_slot(slots: dict, target: dt.datetime, nearest: bool) -> dt.datetime:
    """Choose a slot. nearest=True: closest to target, ties prefer the
    EARLIER slot (older is at least fully uploaded). nearest=False: the
    target's own 10-minute slot must exist exactly, else raise."""
    if not slots:
        raise RuntimeError("no GK-2A slots to pick from")
    if nearest:
        return min(slots, key=lambda s: (abs((s - target).total_seconds()), s))
    snapped = target.replace(minute=(target.minute // 10) * 10,
                             second=0, microsecond=0)
    if snapped not in slots:
        raise RuntimeError(
            f"GK-2A slot {snapped:%Y-%m-%d %H:%M} not on the bucket "
            f"(have {sorted(s.strftime('%H%M') for s in slots)})")
    return snapped


# ---------------------------------------------------------------------------
# GEOS navigation (CGMS y-sweep; north-positive y paired with LFAC < 0)
# ---------------------------------------------------------------------------

def _ami_latlon_to_xy_deg(lat_deg, lon_deg, sub_lon_deg,
                          r_eq_km=_R_EQ_KM, r_pol_km=_R_POL_KM, h_km=_H_KM):
    """WGS84 lat/lon (deg) -> AMI GEOS scan angles (deg), NaN beyond the
    horizon. Band-INDEPENDENT (the expensive trig): every AMI band shares
    the viewing geometry and differs only in the linear CFAC/LFAC/COFF/LOFF
    scaling, so multi-band composites compute this once (the AHI pattern).

    x is east-positive, y NORTH-positive (y = +arcsin(r3/rn)): the pairing
    GK-2A's stored negative LFAC expects, so row 1 lands at the north edge.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    sub = np.deg2rad(float(sub_lon_deg))
    rr = (r_pol_km * r_pol_km) / (r_eq_km * r_eq_km)
    c_lat = np.arctan(rr * np.tan(lat))
    cos_c = np.cos(c_lat)
    r_l = r_pol_km / np.sqrt(1.0 - (1.0 - rr) * cos_c * cos_c)
    dlon = lon - sub
    # horizon test: satellite local zenith <= 90 deg, i.e. the geocentric
    # angle to the point satisfies cos(c_lat)*cos(dlon) >= r_l / h.
    vis = cos_c * np.cos(dlon) >= r_l / h_km
    r1 = h_km - r_l * cos_c * np.cos(dlon)
    r2 = -r_l * cos_c * np.sin(dlon)
    r3 = r_l * np.sin(c_lat)
    rn = np.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    with np.errstate(invalid="ignore"):
        x = np.where(vis, np.arctan2(-r2, r1), np.nan)
        y = np.where(vis, np.arcsin(r3 / rn), np.nan)
    return np.rad2deg(x), np.rad2deg(y)


def _ami_xy_deg_to_latlon(x_deg, y_deg, sub_lon_deg,
                          r_eq_km=_R_EQ_KM, r_pol_km=_R_POL_KM, h_km=_H_KM):
    """Inverse GEOS (CGMS 4.4.3.2): scan angles (deg) -> WGS84 lat/lon
    (deg), NaN off-disk. Test/debug support -- validates the forward math
    independently; the render path only ever runs the forward direction."""
    x = np.deg2rad(np.asarray(x_deg, dtype=np.float64))
    y = np.deg2rad(np.asarray(y_deg, dtype=np.float64))
    cosx, cosy = np.cos(x), np.cos(y)
    sinx, siny = np.sin(x), np.sin(y)
    aa = (r_eq_km * r_eq_km) / (r_pol_km * r_pol_km)
    A = cosy * cosy + aa * siny * siny
    hc = h_km * cosx * cosy
    disc = hc * hc - A * (h_km * h_km - r_eq_km * r_eq_km)
    with np.errstate(invalid="ignore"):
        s_n = (hc - np.sqrt(disc)) / A          # NaN where the ray misses
        s1 = h_km - s_n * cosy * cosx
        s2 = s_n * cosy * sinx
        s3 = s_n * siny
        sxy = np.hypot(s1, s2)
        lon = np.rad2deg(np.arctan2(s2, s1)) + float(sub_lon_deg)
        lat = np.rad2deg(np.arctan(aa * s3 / sxy))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _ami_xy_deg_to_colline(x_deg, y_deg, cfac, lfac, coff, loff):
    """One band's linear CGMS scaling of shared scan angles -> 1-BASED
    pixel-center (col, row). Array index = value - 1."""
    col = coff + cfac / (2 ** 16) * np.asarray(x_deg, dtype=np.float64)
    row = loff + lfac / (2 ** 16) * np.asarray(y_deg, dtype=np.float64)
    return col, row


# ---------------------------------------------------------------------------
# calibration (in-file attrs only)
# ---------------------------------------------------------------------------

def _mask_dn(raw: np.ndarray, var_attrs: dict) -> np.ndarray:
    """Raw counts -> float32 DN with DQF>=2 (out-of-scan / error) NaN'd.

    Self-described layout: top ``number_of_data_quality_flag_bits_per_pixel``
    bits are the DQF, low ``number_of_valid_bits_per_pixel`` bits the count
    (the in-between bits are unused padding). Conditionally-usable (DQF 1)
    pixels are kept -- see the module docstring.
    """
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"AMI counts must be integer, got {raw.dtype} "
                         "(open with mask_and_scale=False)")
    total = int(var_attrs.get("number_of_total_bits_per_pixel", 16))
    qbits = int(var_attrs.get("number_of_data_quality_flag_bits_per_pixel", 2))
    vbits = int(var_attrs.get("number_of_valid_bits_per_pixel", total - qbits))
    qf = raw >> (total - qbits)
    dn = (raw & ((1 << vbits) - 1)).astype(np.float32)
    dn[qf >= 2] = np.nan
    return dn


def _vis_albedo(dn: np.ndarray, attrs: dict) -> np.ndarray:
    """DN -> TOA reflectance factor (>=0, NOT sun-normalized), float32."""
    rad = dn * float(attrs["DN_to_Radiance_Gain"]) \
        + float(attrs["DN_to_Radiance_Offset"])
    alb = rad * float(attrs["Radiance_to_Albedo_c"])
    return np.clip(alb, 0.0, None).astype(np.float32)   # clip keeps NaN


def _ir_bt(dn: np.ndarray, attrs: dict) -> np.ndarray:
    """DN -> brightness temperature (K), float32; NaN where the radiance
    is non-positive (counts past the calibration cold end)."""
    gain = float(attrs["DN_to_Radiance_Gain"])
    offset = float(attrs["DN_to_Radiance_Offset"])
    hp = float(attrs["Plank_constant_h"])       # attr name sic (NMSC typo)
    kb = float(attrs["Boltzmann_constant_k"])
    cl = float(attrs["light_speed"])
    c0 = float(attrs["Teff_to_Tbb_c0"])
    c1 = float(attrs["Teff_to_Tbb_c1"])
    c2 = float(attrs["Teff_to_Tbb_c2"])
    wl_um = float(attrs["channel_center_wavelength"])
    nu = 1.0 / (wl_um * 1e-6)                                # m-1
    rad_si = dn.astype(np.float64) * gain + offset           # mW m-2 sr-1 cm
    rad_si *= 1e-5                                           # -> SI per m-1
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        teff = (hp * cl * nu / kb) / np.log1p(2.0 * hp * cl * cl * nu ** 3 / rad_si)
        tbb = c0 + (c1 + c2 * teff) * teff
        tbb = np.where(rad_si > 0.0, tbb, np.nan)
    return tbb.astype(np.float32)


def _scan_start(attrs: dict) -> dt.datetime:
    """Scan start per the TIME section: observation_start_time on the
    J2000-NOON epoch, cross-checked against scene_acquisition_time."""
    scene = None
    raw = attrs.get("scene_acquisition_time")
    if raw:
        try:
            scene = dt.datetime.strptime(str(raw), "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            log.warning("unparseable scene_acquisition_time %r", raw)
    obs = attrs.get("observation_start_time")
    if obs is not None:
        t = _EPOCH_J2000 + dt.timedelta(seconds=float(obs))
        if scene is not None and abs((t - scene).total_seconds()) > 120:
            log.warning(
                "observation_start_time (%s) disagrees with "
                "scene_acquisition_time (%s) -- epoch assumption broken for "
                "this file; trusting the string", t.isoformat(), scene.isoformat())
            return scene
        return t
    if scene is not None:
        return scene
    raise ValueError("file carries neither observation_start_time nor "
                     "scene_acquisition_time")


# ---------------------------------------------------------------------------
# disk
# ---------------------------------------------------------------------------

class AmiDisk:
    """One calibrated AMI band grid + its GEOS navigation.

    ``data`` is float32, NaN where invalid; ``kind`` is "albedo" (TOA
    reflectance factor, VIS/NIR) or "bt" (Kelvin, IR/WV). ``cfac``/
    ``lfac``/``coff``/``loff`` ALWAYS describe the native full-resolution
    1-based grid; when the counts were decimated on load, ``stride``
    carries the factor and local index = (colrow - 1) / stride -- the
    stride==1 form is exactly the plain CGMS index math (the AHI
    CalibratedDisk convention).
    """
    __slots__ = ("band", "kind", "data", "units", "sub_lon", "cfac", "lfac",
                 "coff", "loff", "stride", "scan_start", "sat_name",
                 "wavelength_um", "resolution_km", "r_eq_km", "r_pol_km",
                 "h_km")

    def __init__(self, band, kind, data, units, sub_lon, cfac, lfac, coff,
                 loff, scan_start, sat_name, wavelength_um, resolution_km,
                 stride=1, r_eq_km=_R_EQ_KM, r_pol_km=_R_POL_KM, h_km=_H_KM):
        self.band = band
        self.kind = kind                  # "albedo" | "bt"
        self.data = data                  # float32 (rows north->south)
        self.units = units                # "1" | "K"
        self.sub_lon = sub_lon            # DEGREES east (converted from rad)
        self.cfac = cfac
        self.lfac = lfac                  # negative on GK-2A
        self.coff = coff
        self.loff = loff
        self.stride = stride
        self.scan_start = scan_start      # tz-aware UTC
        self.sat_name = sat_name          # "GK-2A"
        self.wavelength_um = wavelength_um
        self.resolution_km = resolution_km
        self.r_eq_km = r_eq_km
        self.r_pol_km = r_pol_km
        self.h_km = h_km

    def sample_xy(self, x_deg, y_deg) -> np.ndarray:
        """Bilinear-sample at precomputed scan angles (the shared-trig
        multi-band path); NaN off-window. Same map_coordinates +
        sentinel-poisoning trick as SeviriDisk.sample_bt: an invalid pixel
        at bilinear weight >= 0.1 poisons the sample to NaN (the -1e9
        sentinel against the -1e8 threshold) -- no half-real blends."""
        from scipy.ndimage import map_coordinates
        col, row = _ami_xy_deg_to_colline(
            x_deg, y_deg, self.cfac, self.lfac, self.coff, self.loff)
        st = self.stride or 1
        cols = (col - 1.0) / st           # 1-based CGMS -> local array index
        rows = (row - 1.0) / st
        h, w = self.data.shape
        inb = (np.isfinite(cols) & np.isfinite(rows) &
               (cols >= 0) & (cols <= w - 1) & (rows >= 0) & (rows <= h - 1))
        out = np.full(np.shape(x_deg), np.nan, np.float32)
        if inb.any():
            coords = np.stack([np.where(inb, rows, 0).ravel(),
                               np.where(inb, cols, 0).ravel()])
            samp = map_coordinates(np.nan_to_num(self.data, nan=-1e9), coords,
                                   order=1, mode="nearest").reshape(out.shape)
            out = np.where(inb & (samp > -1e8), samp, np.nan).astype(np.float32)
        return out

    def sample(self, TLAT, TLON) -> np.ndarray:
        """Bilinear-sample at lat/lon query points; NaN off-disk (both the
        far-side horizon test and the data window). Single-band
        convenience -- multi-band consumers should compute the scan angles
        once with _ami_latlon_to_xy_deg and call sample_xy per band."""
        x_deg, y_deg = _ami_latlon_to_xy_deg(
            TLAT, TLON, self.sub_lon, self.r_eq_km, self.r_pol_km, self.h_km)
        return self.sample_xy(x_deg, y_deg)


# ---------------------------------------------------------------------------
# decode + fetch
# ---------------------------------------------------------------------------

def ami_disk_from_dataset(ds, band=None, stride: int = 1) -> AmiDisk:
    """Decode one opened GK-2A L1B dataset -> AmiDisk (pure, no I/O).

    ``ds`` must carry raw counts (mask_and_scale=False). ``stride`` > 1
    decimates on read ([::stride, ::stride] of the native grid), keeping
    the native cfac/coff and recording the factor -- the backend reads
    only the strided selection, so a 470 MB vi006 never fully lands in RAM.
    """
    attrs = dict(ds.attrs)
    if "image_pixel_values" in ds:
        var = ds["image_pixel_values"]
    else:  # format drift guard: fall back to the sole 2-D variable
        two_d = [v for v in ds.data_vars if ds[v].ndim == 2]
        if len(two_d) != 1:
            raise ValueError(f"cannot identify the counts variable "
                             f"(2-D candidates: {two_d})")
        var = ds[two_d[0]]
    var_attrs = dict(var.attrs)
    stride = max(1, int(stride))
    sel = var[::stride, ::stride] if stride > 1 else var
    raw = np.asarray(sel.values)
    if raw.ndim != 2:
        raise ValueError(f"counts variable is {raw.ndim}-D, expected 2-D")

    dn = _mask_dn(raw, var_attrs)
    if "Radiance_to_Albedo_c" in attrs:
        kind, units, data = "albedo", "1", _vis_albedo(dn, attrs)
    elif "Teff_to_Tbb_c0" in attrs:
        kind, units, data = "bt", "K", _ir_bt(dn, attrs)
    else:
        raise ValueError("file carries neither Radiance_to_Albedo_c nor "
                         "Teff_to_Tbb_c0 -- unknown calibration kind")

    band_name = str(var_attrs.get("channel_name", "") or band or "").lower()
    return AmiDisk(
        band=band_name or "unknown",
        kind=kind, data=data, units=units,
        sub_lon=float(np.degrees(float(attrs["sub_longitude"]))),  # rad in file
        cfac=float(attrs["cfac"]), lfac=float(attrs["lfac"]),
        coff=float(attrs["coff"]), loff=float(attrs["loff"]),
        scan_start=_scan_start(attrs),
        sat_name=str(attrs.get("satellite_name", "GK-2A")),
        wavelength_um=float(attrs.get("channel_center_wavelength", 0.0) or 0.0),
        resolution_km=float(attrs.get("channel_spatial_resolution", 0.0) or 0.0),
        stride=stride,
        r_eq_km=float(attrs.get("earth_equatorial_radius", _R_EQ_KM * 1e3)) / 1e3,
        r_pol_km=float(attrs.get("earth_polar_radius", _R_POL_KM * 1e3)) / 1e3,
        h_km=float(attrs.get("nominal_satellite_height", _H_KM * 1e3)) / 1e3,
    )


def _decode_path(path: str, band=None, stride: int = 1) -> AmiDisk:
    """Open one downloaded .nc (h5netcdf preferred, default engine as the
    fallback) and decode eagerly so the file can be deleted at once."""
    import xarray as xr
    try:
        ds = xr.open_dataset(path, engine="h5netcdf",
                             mask_and_scale=False, decode_times=False)
    except (ImportError, ValueError, OSError) as e:
        log.debug("h5netcdf open failed (%s); trying the default engine", e)
        ds = xr.open_dataset(path, mask_and_scale=False, decode_times=False)
    try:
        return ami_disk_from_dataset(ds, band=band, stride=stride)
    finally:
        ds.close()


def fetch_ami_disk(band: str, time=None, nearest: bool = True,
                   timeout: float = 600, stride: int = 1) -> AmiDisk:
    """Fetch + calibrate the best GK-2A slot for one band -> AmiDisk.

    time: pin near this UTC time (archive/suite use); None = newest on the
    bucket. nearest: True picks the listed slot closest to the target;
    False requires the target's exact 10-minute slot. stride: decimate on
    read (>=2 strongly advised for vi006 outside floater-scale work).

    Lists the target hour + the previous hour (+ the next hour for pinned
    times, so a :58 pin can resolve forward); raises RuntimeError when the
    band has no slot there -- the caller's degrade path owns the gap.
    """
    band = str(band).lower()
    if band not in BAND_RES:
        raise ValueError(f"unknown AMI band {band!r} "
                         f"(expected one of {sorted(BAND_RES)})")
    t = time or dt.datetime.now(UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)

    hours = [t - dt.timedelta(hours=1), t]
    if time is not None:
        hours.append(t + dt.timedelta(hours=1))
    slots = {}
    for prefix in dict.fromkeys(_hour_prefix(h) for h in hours):
        slots.update(_band_slots(_list_keys(prefix, timeout=min(timeout, 60)),
                                 band))
    if not slots:
        raise RuntimeError(
            f"no GK-2A {band} slot on {S3_BASE} near {t:%Y-%m-%d %H:%M}Z")
    slot = _pick_slot(slots, t, nearest)
    key = slots[slot]

    tmp = tempfile.mkdtemp(prefix="gk2a_")
    try:
        path = os.path.join(tmp, os.path.basename(key))
        with requests.get(f"{S3_BASE}/{key}", stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        disk = _decode_path(path, band=band, stride=stride)
        log.info("gk2a %s %s: %s px stride %d",
                 band, slot.strftime("%Y-%m-%d %H:%MZ"),
                 "x".join(map(str, disk.data.shape))
                 if hasattr(disk, "data") else "?", stride)
        return disk
    finally:
        # the grid is eagerly materialized -- the ~35..470 MB .nc goes now
        shutil.rmtree(tmp, ignore_errors=True)
