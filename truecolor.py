"""True-color RGB recipe for geostationary ABI/AHI imagery.

Operates on already-fetched, **co-registered** top-of-atmosphere reflectance
bands (red / green / blue, each 0..1 on a common lat/lon grid) plus the
observation time, and produces a display-ready RGB array.

Recipe matches the documented CIMSS Natural True Color + CIRA GeoColor pipeline
(see ABIQuickGuide_CIMSSRGB_v2 and QuickGuide_CIRA_Geocolor):

  * ORDER (CIRA GeoColor): sun-normalize -> **Rayleigh-correct the real bands
    first** (blue 0.47, red 0.64, veggie 0.86) -> THEN synthesize green. Doing
    Rayleigh before the green synthesis is what keeps clear-sky/ocean a clean
    deep blue and the colors vibrant.
  * ABI has no green band -> synthesize the CIMSS Natural True Color green:
        G = 0.45*Red + 0.10*Veggie(0.86) + 0.45*Blue
    The big 0.45 BLUE share is the fix for the "magenta ocean" failure: over
    water veggie≈0, so a veggie-heavy green collapses and blue+red dominate ->
    magenta. The CIMSS blue-heavy mix keeps green up over water (deep blue),
    while the veggie term still lifts vegetation green over land. Himawari AHI
    has a native 0.51 green band and skips synthesis entirely.
  * Solar-zenith normalization: ABI CMI and our AHI albedo are raw TOA
    reflectance (NOT sun-angle normalized) -> divide by cos(SZA).
  * A CIRA/EUMETSAT tone curve (gamma-ish stretch) for natural brightness.
  * Red-band ratio sharpening lifts the 1 km green/blue toward the 0.5 km red.
  * Day/night via the cos(SZA) field (GeoColor-lite: fade to clean-IR at night).

Geometry (sun + geostationary satellite zenith/azimuth) comes from pyorbital.
"""

from __future__ import annotations

import logging
import datetime as dt
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("tat-satellite.truecolor")

# CIMSS Natural True Color synthetic-green fractions (Red, Veggie/NIR, Blue).
# From ABIQuickGuide_CIMSSRGB_v2: G = 0.45*Red + 0.10*Veggie + 0.45*Blue.
# (Previously 0.40/0.40/0.20 — the veggie-heavy / blue-light mix that turned
# open ocean magenta because synth-green collapsed where veggie≈0.)
GREEN_FRACTIONS = (0.45, 0.10, 0.45)

# Rayleigh subtraction strength (0..1). CIRA GeoColor applies FULL Rayleigh
# correction to maximize clear/cloud contrast and color vibrancy. With the
# CIMSS blue-heavy green this no longer exposes a magenta imbalance, so we run
# it at full strength.
RAYLEIGH_SCALE = 1.0

# Sun-correction floor: never divide by less than this cos(SZA), so the
# terminator doesn't explode to white. ~84.3 deg SZA.
COS_SZA_FLOOR = 0.10

# Geostationary satellite height above the surface (km) for pyorbital's
# observer-look geometry. 42164 km geocentric - 6378 km mean Earth radius.
GEO_SAT_ALT_KM = 35786.0

# Central wavelengths (um) used to look up Rayleigh correction per channel.
WL_RED = 0.64
WL_GREEN = 0.51
WL_BLUE = 0.47
WL_VEGGIE = 0.86


# ---------------------------------------------------------------------------
# Geometry (pyorbital)
# ---------------------------------------------------------------------------
# Stride for the angle-geometry fields (solar + satellite zenith/azimuth).
# These vary over hundreds of km, so computing them on a coarse subgrid and
# bilinear-upsampling is visually exact while cutting pyorbital's per-pixel
# trig (~3 s/frame at full res) to noise. 1 disables the shortcut.
GEOMETRY_STRIDE = 8


def _upsample_to(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize a 2D field to ``shape`` (smooth angle fields only)."""
    from scipy.ndimage import zoom
    if field.shape == shape:
        return field
    zy = shape[0] / field.shape[0]
    zx = shape[1] / field.shape[1]
    return zoom(field, (zy, zx), order=1, mode="nearest", grid_mode=True,
                output=np.float64)[: shape[0], : shape[1]]


def _upsample_azimuth(az_deg: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Upsample an azimuth field via its sin/cos components — bilinear on the
    raw degrees would tear wherever the field crosses the 0/360 wrap (routine
    for sun azimuth in the tropics near local noon)."""
    rad = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    s = _upsample_to(np.sin(rad), shape)
    c = _upsample_to(np.cos(rad), shape)
    return np.rad2deg(np.arctan2(s, c)) % 360.0


def solar_geometry(lats: np.ndarray, lons: np.ndarray, when: dt.datetime):
    """(cos_sza, sun_zenith_deg, sun_azimuth_deg) for each pixel. Angles are
    computed on a GEOMETRY_STRIDE subgrid and bilinear-upsampled — sun
    geometry is smooth at the ~10 km scale of the stride."""
    from pyorbital.astronomy import sun_zenith_angle, get_alt_az

    s = GEOMETRY_STRIDE
    cl_lats, cl_lons = lats[::s, ::s], lons[::s, ::s]
    sza_c = sun_zenith_angle(when, cl_lons, cl_lats)  # degrees
    alt, az = get_alt_az(when, cl_lons, cl_lats)      # radians
    sza = _upsample_to(np.asarray(sza_c, dtype=np.float64), lats.shape)
    sun_az = _upsample_azimuth(np.rad2deg(az), lats.shape)
    cos_sza = np.cos(np.deg2rad(sza))
    return cos_sza, sza, sun_az


def satellite_geometry(lats: np.ndarray, lons: np.ndarray, sub_sat_lon: float, when: dt.datetime):
    """(sat_zenith_deg, sat_azimuth_deg) for a geostationary bird at
    sub_sat_lon — same coarse-subgrid + upsample scheme as solar_geometry."""
    from pyorbital.orbital import get_observer_look

    s = GEOMETRY_STRIDE
    cl_lats, cl_lons = lats[::s, ::s], lons[::s, ::s]
    sat_az_c, sat_elev_c = get_observer_look(
        sub_sat_lon, 0.0, GEO_SAT_ALT_KM, when, cl_lons, cl_lats, 0.0
    )
    sat_az = _upsample_azimuth(np.asarray(sat_az_c), lats.shape)
    sat_elev = _upsample_to(np.asarray(sat_elev_c, dtype=np.float64),
                            lats.shape)
    return 90.0 - sat_elev, sat_az


# ---------------------------------------------------------------------------
# Reflectance conditioning
# ---------------------------------------------------------------------------
def sun_correct(refl: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """Normalize TOA reflectance by the solar illumination (divide by cos SZA)."""
    denom = np.clip(cos_sza, COS_SZA_FLOOR, None)
    return refl / denom


def synth_green(red: np.ndarray, veggie: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Synthesize the missing ABI green (CIMSS Natural True Color)."""
    fr, fv, fb = GREEN_FRACTIONS
    return fr * red + fv * veggie + fb * blue


# Rayleigh correctors cached per (platform, sensor): construction re-reads the
# pyspectral LUT HDF5 from disk, which is pure waste when every truecolor frame
# rebuilds it. Only SUCCESSFUL constructions cache, so a transient init failure
# (e.g. LUTs still downloading) retries on the next frame instead of pinning
# the degraded no-Rayleigh path forever.
_RAYLEIGH_CACHE: dict = {}
_RAYLEIGH_LOCK = threading.Lock()


def _make_rayleigh(platform_name: str, sensor: str):
    """Build (or reuse) a pyspectral Rayleigh corrector, or None if unavailable."""
    key = (platform_name, sensor)
    with _RAYLEIGH_LOCK:
        if key in _RAYLEIGH_CACHE:
            return _RAYLEIGH_CACHE[key]
        try:
            from pyspectral.rayleigh import Rayleigh
        except Exception as e:  # pragma: no cover - import guard
            log.warning("pyspectral unavailable (%s); skipping Rayleigh correction", e)
            return None
        try:
            corrector = Rayleigh(platform_name, sensor)
        except Exception as e:
            log.warning("Rayleigh(%s,%s) init failed (%s); skipping", platform_name, sensor, e)
            return None
        _RAYLEIGH_CACHE[key] = corrector
        return corrector


def rayleigh_band(
    band: np.ndarray,
    wl: float,
    sun_zenith: np.ndarray,
    sat_zenith: np.ndarray,
    azidiff: np.ndarray,
    red_ref_pct: np.ndarray,
    corrector,
    scale: float = RAYLEIGH_SCALE,
) -> np.ndarray:
    """Subtract molecular (Rayleigh) scattering from a single band via pyspectral.

    Done on the real measured bands BEFORE green synthesis (CIRA GeoColor order).
    ``red_ref_pct`` (sun-corrected red, in reflectance percent) is the aerosol
    scaling reference. Returns the band unchanged (with a warning) if the
    corrector is missing or the call fails — degrade hazier, never fail.
    pyspectral works in reflectance percent (0..100).
    """
    if corrector is None:
        return band
    try:
        corr_pct = corrector.get_reflectance(sun_zenith, sat_zenith, azidiff, wl, red_ref_pct)
    except Exception as e:
        log.warning("Rayleigh get_reflectance failed for wl=%.2f (%s); skipping band", wl, e)
        return band
    return np.clip(band - scale * corr_pct / 100.0, 0.0, 1.0)


def ratio_sharpen(rgb: np.ndarray, red_hires: np.ndarray) -> np.ndarray:
    """Lift the (1 km) green/blue toward the 0.5 km red via the red ratio."""
    base = rgb[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(base > 0.02, red_hires / base, 1.0)
    ratio = np.clip(ratio, 0.5, 2.0)
    out = rgb.copy()
    out[..., 0] = red_hires
    out[..., 1] = np.clip(rgb[..., 1] * ratio, 0.0, 1.0)
    out[..., 2] = np.clip(rgb[..., 2] * ratio, 0.0, 1.0)
    return out


# GeoColor-lite night: clean-IR grayscale range (cold = white) and the
# terminator blend window in cos(SZA).
IR_T_WARM = 303.0
IR_T_COLD = 183.0
COS_SZA_FULL_DAY = 0.14    # sun ≳ 8° up -> 100% true color
COS_SZA_FULL_NIGHT = -0.05  # sun ≳ 3° below -> 100% IR


def _clean_ir_gray(bt_kelvin: np.ndarray) -> np.ndarray:
    """Clean-IR brightness temperature -> grayscale 0..1 (cold cloud = white)."""
    x = (IR_T_WARM - bt_kelvin) / (IR_T_WARM - IR_T_COLD)
    return np.clip(x, 0.0, 1.0)


def night_blend(day_rgb: np.ndarray, ir_bt: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """GeoColor-lite: fade true color (day) to grayscale clean-IR (night)."""
    w_day = np.clip(
        (cos_sza - COS_SZA_FULL_NIGHT) / (COS_SZA_FULL_DAY - COS_SZA_FULL_NIGHT),
        0.0, 1.0,
    )[..., None]
    night_gray = _clean_ir_gray(ir_bt)
    night_rgb = np.repeat(night_gray[..., None], 3, axis=-1)
    return np.clip(day_rgb * w_day + night_rgb * (1.0 - w_day), 0.0, 1.0)


def tone_curve(rgb: np.ndarray) -> np.ndarray:
    """CIRA/Polar2Grid-style nonlinear stretch for natural true-color brightness."""
    x = np.array([0.0, 0.0030, 0.0110, 0.0190, 0.0290, 0.0440, 0.0720, 0.1010,
                  0.1260, 0.1560, 0.1900, 0.2300, 0.2800, 0.3400, 0.4000,
                  0.4700, 0.5400, 0.6400, 0.7400, 0.8600, 1.0000], dtype=np.float64)
    y = np.array([0.0, 0.0500, 0.1090, 0.1590, 0.2030, 0.2650, 0.3370, 0.3970,
                  0.4380, 0.4830, 0.5240, 0.5670, 0.6130, 0.6630, 0.7010,
                  0.7430, 0.7820, 0.8380, 0.8880, 0.9530, 1.0000], dtype=np.float64)
    clipped = np.clip(rgb, 0.0, 1.0)
    out = np.interp(clipped.ravel(), x, y).reshape(rgb.shape)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------
def assemble_truecolor(
    red: np.ndarray,
    green: Optional[np.ndarray],
    blue: np.ndarray,
    veggie: Optional[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    when: dt.datetime,
    sub_sat_lon: float,
    platform_name: str,
    sensor: str,
    ir_bt: Optional[np.ndarray] = None,
    do_rayleigh: bool = True,
    do_ratio_sharpen: bool = True,
):
    """Build a display RGB (H,W,3 float32 0..1) + the cos(SZA) field.

    ``green`` is None for ABI (synthesized from veggie); supplied for AHI.
    ``ir_bt`` (clean-IR brightness temp, K, co-registered) enables the
    GeoColor-lite night fade. All inputs are co-registered TOA reflectance on
    the same (red-res) grid. Returns (rgb, cos_sza).
    """
    cos_sza, sun_zen, sun_az = solar_geometry(lats, lons, when)
    sat_zen, sat_az = satellite_geometry(lats, lons, sub_sat_lon, when)

    # 1) Sun-angle normalize every measured band.
    red_c = sun_correct(red, cos_sza)
    blue_c = sun_correct(blue, cos_sza)
    veggie_c = sun_correct(veggie, cos_sza) if veggie is not None else None
    green_c = sun_correct(green, cos_sza) if green is not None else None

    # 2) Rayleigh-correct the REAL bands first (CIRA GeoColor order), so the
    #    synthesized green is built from already-corrected red/veggie/blue.
    if do_rayleigh:
        azidiff = np.abs(sun_az - sat_az)
        azidiff = np.where(azidiff > 180.0, 360.0 - azidiff, azidiff)
        corrector = _make_rayleigh(platform_name, sensor)
        red_ref_pct = np.clip(red_c, 0.0, 1.0) * 100.0
        red_c = rayleigh_band(red_c, WL_RED, sun_zen, sat_zen, azidiff, red_ref_pct, corrector)
        blue_c = rayleigh_band(blue_c, WL_BLUE, sun_zen, sat_zen, azidiff, red_ref_pct, corrector)
        if veggie_c is not None:
            veggie_c = rayleigh_band(veggie_c, WL_VEGGIE, sun_zen, sat_zen, azidiff, red_ref_pct, corrector)
        if green_c is not None:
            green_c = rayleigh_band(green_c, WL_GREEN, sun_zen, sat_zen, azidiff, red_ref_pct, corrector)

    # 3) Synthesize green for ABI (CIMSS) from the corrected bands; AHI uses its
    #    native (already-corrected) green.
    if green_c is None:
        if veggie_c is None:
            raise ValueError("ABI true color needs a veggie band to synthesize green")
        green_c = synth_green(red_c, veggie_c, blue_c)

    rgb = np.clip(np.dstack([red_c, green_c, blue_c]), 0.0, 1.0)

    if do_ratio_sharpen:
        rgb = ratio_sharpen(rgb, np.clip(red_c, 0.0, 1.0))

    rgb = tone_curve(rgb)

    # GeoColor-lite: fade to clean-IR at night (after the day side is developed).
    if ir_bt is not None:
        rgb = night_blend(rgb, ir_bt, cos_sza)

    return rgb.astype(np.float32), cos_sza.astype(np.float32)
