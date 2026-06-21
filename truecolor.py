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

# --- Terminator (sunrise/sunset golden-hour) handling ----------------------
# Near the terminator FULL Rayleigh subtraction over-corrects at the limb --
# stripping the warm forward-scattered light, the synth-green then dominates ->
# a green cast. Two smoothly-gated corrections fix it WITHOUT touching daytime.
#
# Both are confined to a NARROW low-sun band that stays BELOW full daylight:
# COS_SZA_FULL_DAY is 0.14, so the gates live under it. The first ship started
# both at cos_sza 0.30 (sun ~17 deg up -- full daylight), which painted a warm/
# sepia WALL across the whole afternoon (worst over Saharan dust). Now:
#   * warm tint: ZERO at/above WARM_TINT_DAY_COS (0.12), peaking AT the horizon
#     -- a thin sunrise/sunset band, half the old amplitude (subtle golden, not
#     sepia). Never fires in full daylight.
#   * Rayleigh: stays FULL across the ENTIRE day side; only in the last few
#     degrees (below RAYLEIGH_TAPER_START_COS 0.10) does it soften -- and only to
#     RAYLEIGH_FLOOR, NEVER to 0, so clear-sky/ocean stays blue and the broad
#     warm bleed is gone.
# Both are EXACT no-ops above their start cos (smoothstep saturates to 1.0 -> the
# taper is exactly RAYLEIGH_SCALE and the warm weight exactly 0.0), so midday true
# color stays byte-identical. Applied to the day-side RGB BEFORE the clean-IR
# night fade so they develop then fade into night. All tunable here.
RAYLEIGH_TAPER_START_COS = 0.10  # cos_sza >= this -> FULL Rayleigh (1.0): whole day side unchanged
RAYLEIGH_FLOOR = 0.4             # Rayleigh strength AT the horizon (cos_sza<=0) -- never 0

WARM_TINT_DAY_COS = 0.12    # cos_sza >= this -> NO warm tint (0.0): never in full daylight
WARM_TINT_PEAK_COS = 0.0    # cos_sza <= this (the horizon + below) -> full warm tint
WARM_TINT_STRENGTH = 1.0    # overall multiplier on the warm tint (0 disables)
WARM_TINT_RED_ADD = 0.05    # additive red boost at full tint (halved: subtle golden, not sepia)
WARM_TINT_GREEN_GAIN = 0.08  # multiplicative green attenuation at full tint (halved)
WARM_TINT_BLUE_GAIN = 0.17   # multiplicative blue attenuation at full tint (halved; most -> orange)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep, clamped to [0,1]: 0 for x<=edge0, 1 for x>=edge1.
    Saturates EXACTLY to 1.0/0.0 outside the edges (t is clipped first), which is
    what makes the daytime taper/tint exact no-ops -> byte-identical midday."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rayleigh_scale_field(cos_sza: np.ndarray) -> np.ndarray:
    """Per-pixel Rayleigh strength: FULL (RAYLEIGH_SCALE) across the ENTIRE day
    side, softening only in the last few degrees near the horizon and only down to
    RAYLEIGH_FLOOR (never 0 -- keeps clear-sky/ocean blue + kills the broad warm
    bleed). EXACTLY RAYLEIGH_SCALE for cos_sza >= RAYLEIGH_TAPER_START_COS, so
    daytime is byte-identical."""
    return RAYLEIGH_FLOOR + (RAYLEIGH_SCALE - RAYLEIGH_FLOOR) * _smoothstep(
        0.0, RAYLEIGH_TAPER_START_COS, cos_sza
    )


def warm_terminator_tint(rgb: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """Warm the day-side RGB toward red/orange near the terminator (sunrise /
    sunset). The weight is EXACTLY 0 for cos_sza >= WARM_TINT_DAY_COS (midday
    byte-identical) and ramps to 1 by WARM_TINT_PEAK_COS, held through the
    terminator; the clean-IR night fade (applied after) then takes over."""
    w = (1.0 - _smoothstep(WARM_TINT_PEAK_COS, WARM_TINT_DAY_COS, cos_sza))
    w = w * WARM_TINT_STRENGTH
    out = rgb.copy()
    out[..., 0] = np.clip(rgb[..., 0] + w * WARM_TINT_RED_ADD, 0.0, 1.0)
    out[..., 1] = np.clip(rgb[..., 1] * (1.0 - w * WARM_TINT_GREEN_GAIN), 0.0, 1.0)
    out[..., 2] = np.clip(rgb[..., 2] * (1.0 - w * WARM_TINT_BLUE_GAIN), 0.0, 1.0)
    return out  # already rgb.dtype (the in-place clips downcast to out's dtype)


# ---------------------------------------------------------------------------
# Geometry (pyorbital)
# ---------------------------------------------------------------------------
def solar_geometry(lats: np.ndarray, lons: np.ndarray, when: dt.datetime):
    """(cos_sza, sun_zenith_deg, sun_azimuth_deg) for each pixel."""
    from pyorbital.astronomy import sun_zenith_angle, get_alt_az

    sza = sun_zenith_angle(when, lons, lats)          # degrees
    alt, az = get_alt_az(when, lons, lats)            # radians
    sun_az = np.rad2deg(az) % 360.0
    cos_sza = np.cos(np.deg2rad(sza))
    return cos_sza, sza, sun_az


def satellite_geometry(lats: np.ndarray, lons: np.ndarray, sub_sat_lon: float, when: dt.datetime):
    """(sat_zenith_deg, sat_azimuth_deg) for a geostationary bird at sub_sat_lon."""
    from pyorbital.orbital import get_observer_look

    sat_az, sat_elev = get_observer_look(
        sub_sat_lon, 0.0, GEO_SAT_ALT_KM, when, lons, lats, 0.0
    )
    sat_zenith = 90.0 - sat_elev
    return sat_zenith, sat_az % 360.0


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


def _make_rayleigh(platform_name: str, sensor: str):
    """Build a pyspectral Rayleigh corrector, or None if unavailable."""
    try:
        from pyspectral.rayleigh import Rayleigh
    except Exception as e:  # pragma: no cover - import guard
        log.warning("pyspectral unavailable (%s); skipping Rayleigh correction", e)
        return None
    try:
        return Rayleigh(platform_name, sensor)
    except Exception as e:
        log.warning("Rayleigh(%s,%s) init failed (%s); skipping", platform_name, sensor, e)
        return None


def rayleigh_band(
    band: np.ndarray,
    wl: float,
    sun_zenith: np.ndarray,
    sat_zenith: np.ndarray,
    azidiff: np.ndarray,
    red_ref_pct: np.ndarray,
    corrector,
    scale: "float | np.ndarray" = RAYLEIGH_SCALE,
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
        # Per-pixel Rayleigh strength: full in daytime, tapered toward 0 near the
        # terminator (it over-corrects at the limb and greens the sunrise/sunset).
        # Exactly 1.0 for daytime pixels -> byte-identical midday.
        ray_scale = rayleigh_scale_field(cos_sza)
        red_c = rayleigh_band(red_c, WL_RED, sun_zen, sat_zen, azidiff, red_ref_pct, corrector, scale=ray_scale)
        blue_c = rayleigh_band(blue_c, WL_BLUE, sun_zen, sat_zen, azidiff, red_ref_pct, corrector, scale=ray_scale)
        if veggie_c is not None:
            veggie_c = rayleigh_band(veggie_c, WL_VEGGIE, sun_zen, sat_zen, azidiff, red_ref_pct, corrector, scale=ray_scale)
        if green_c is not None:
            green_c = rayleigh_band(green_c, WL_GREEN, sun_zen, sat_zen, azidiff, red_ref_pct, corrector, scale=ray_scale)

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

    # Golden-hour: warm the day-side RGB toward red/orange near the terminator
    # (sunrise/sunset) BEFORE the night fade. Exact no-op in daytime -> midday
    # byte-identical; develops at the terminator then fades into the IR night.
    rgb = warm_terminator_tint(rgb, cos_sza)

    # GeoColor-lite: fade to clean-IR at night (after the day side is developed).
    if ir_bt is not None:
        rgb = night_blend(rgb, ir_bt, cos_sza)

    return rgb.astype(np.float32), cos_sza.astype(np.float32)
