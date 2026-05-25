"""True-color RGB recipe for geostationary ABI/AHI imagery.

Operates on already-fetched, **co-registered** top-of-atmosphere reflectance
bands (red / green / blue, each 0..1 on a common lat/lon grid) plus the
observation time, and produces a display-ready RGB array.

The make-or-break details (see module-level constants + functions):

  * GOES ABI has no green band -> synthesize one (Bah et al. 2018 / CIMSS
    operational fractional combination, the same default satpy uses):
        G = 0.45*Red + 0.45*Veggie(0.86um) + 0.10*Blue
    Using raw veggie *as* green makes vegetation read orange; the mix fixes
    it. Himawari AHI has a native green band and skips this.
  * Solar-zenith normalization: both ABI CMI and our AHI albedo are raw TOA
    reflectance (NOT sun-angle normalized), so we divide by cos(SZA) or the
    disk darkens badly away from the subsolar point.
  * Rayleigh / atmospheric correction (pyspectral) removes the blue molecular
    scattering haze so oceans read deep blue, not milky. Guarded: if pyspectral
    or its RSR data is unavailable the recipe still renders (just hazier).
  * A tone curve (gamma-ish stretch) for natural brightness without blown tops.
  * Red-band ratio sharpening lifts the 1 km green/blue toward the 0.5 km red.
  * Day/night handled by the caller via the cos(SZA) field this module exposes.

Geometry (sun + geostationary satellite zenith/azimuth) comes from pyorbital,
which has closed-form helpers for both — no TLEs needed for a fixed geo bird.
"""

from __future__ import annotations

import logging
import datetime as dt
from typing import Optional

import numpy as np

log = logging.getLogger("tat-satellite.truecolor")

# Synthetic-green fractions (Red, Veggie/NIR, Blue). Canonical Bah et al.
# (2018) / satpy SimulatedGreen is (0.45, 0.45, 0.10); we nudge a little blue
# in because veggie≈0 over open water drags synth-green below red there, which
# (especially under Rayleigh removal) tints deep ocean magenta. The small blue
# share lifts green over water to keep the ocean a neutral deep blue while
# leaving land vegetation visibly green. Validated on GOES-19 over FL/Bahamas.
GREEN_FRACTIONS = (0.40, 0.40, 0.20)

# Rayleigh subtraction strength (0..1). Full removal (1.0) over-darkens open
# ocean and exposes the synth-green imbalance as magenta; 0 leaves a milky blue
# haze. 0.6 removes most of the haze for a deep-blue ocean without the magenta.
RAYLEIGH_SCALE = 0.6

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
    """Synthesize the missing ABI green from red + veggie(NIR) + blue."""
    fr, fv, fb = GREEN_FRACTIONS
    return fr * red + fv * veggie + fb * blue


def rayleigh_correct(
    rgb: np.ndarray,
    sun_zenith: np.ndarray,
    sat_zenith: np.ndarray,
    sun_azimuth: np.ndarray,
    sat_azimuth: np.ndarray,
    platform_name: str,
    sensor: str,
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract molecular (Rayleigh) scattering per band via pyspectral.

    ``scale`` (0..1) attenuates the subtraction — see RAYLEIGH_SCALE.

    Returns the RGB unchanged (with a warning) if pyspectral or its RSR data
    for ``platform_name``/``sensor`` isn't available — the recipe degrades to
    a hazier-but-valid image rather than failing the render.

    pyspectral works in reflectance percent; we scale 0..1 <-> 0..100 around
    the call. The red band is passed as the aerosol-scaling reference.
    """
    try:
        from pyspectral.rayleigh import Rayleigh
    except Exception as e:  # pragma: no cover - import guard
        log.warning("pyspectral unavailable (%s); skipping Rayleigh correction", e)
        return rgb

    azidiff = np.abs(sun_azimuth - sat_azimuth)
    azidiff = np.where(azidiff > 180.0, 360.0 - azidiff, azidiff)

    try:
        corrector = Rayleigh(platform_name, sensor)
    except Exception as e:
        log.warning("Rayleigh(%s,%s) init failed (%s); skipping", platform_name, sensor, e)
        return rgb

    out = rgb.copy()
    red_pct = np.clip(rgb[..., 0], 0.0, 1.0) * 100.0
    for idx, wl in ((0, WL_RED), (1, WL_GREEN), (2, WL_BLUE)):
        try:
            corr_pct = corrector.get_reflectance(sun_zenith, sat_zenith, azidiff, wl, red_pct)
        except Exception as e:
            log.warning("Rayleigh get_reflectance failed for wl=%.2f (%s); skipping band", wl, e)
            continue
        out[..., idx] = np.clip(rgb[..., idx] - scale * corr_pct / 100.0, 0.0, 1.0)
    return out


def ratio_sharpen(rgb: np.ndarray, red_hires: np.ndarray) -> np.ndarray:
    """Lift the (1 km) green/blue toward the 0.5 km red via the red ratio.

    ``rgb`` and ``red_hires`` are on the same (red-resolution) grid; rgb[...,0]
    is the upsampled-then-composited red, red_hires is the native red. Where
    they differ, the ratio injects red's fine structure into G and B. Clamped
    so a near-zero denominator can't blow up dark pixels.
    """
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
# terminator blend window in cos(SZA). cos(SZA)=0 is the horizon; we hold full
# true color until the sun is ~8° up and fade to full IR a few degrees past
# sunset so the day/night seam is a soft band, not a hard line.
IR_T_WARM = 303.0
IR_T_COLD = 183.0
COS_SZA_FULL_DAY = 0.14    # sun ≳ 8° up -> 100% true color
COS_SZA_FULL_NIGHT = -0.05  # sun ≳ 3° below -> 100% IR


def _clean_ir_gray(bt_kelvin: np.ndarray) -> np.ndarray:
    """Clean-IR brightness temperature -> grayscale 0..1 (cold cloud = white),
    matching the render.py grayscale-IR ramp."""
    x = (IR_T_WARM - bt_kelvin) / (IR_T_WARM - IR_T_COLD)
    return np.clip(x, 0.0, 1.0)


def night_blend(day_rgb: np.ndarray, ir_bt: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """GeoColor-lite: fade true color (day) to grayscale clean-IR (night) across
    the terminator, weighted by cos(SZA). Looks good 24/7 and reuses the IR we
    already know how to read; full GeoColor (IR microphysics + city lights) is a
    later upgrade."""
    w_day = np.clip(
        (cos_sza - COS_SZA_FULL_NIGHT) / (COS_SZA_FULL_DAY - COS_SZA_FULL_NIGHT),
        0.0, 1.0,
    )[..., None]
    night_gray = _clean_ir_gray(ir_bt)
    night_rgb = np.repeat(night_gray[..., None], 3, axis=-1)
    return np.clip(day_rgb * w_day + night_rgb * (1.0 - w_day), 0.0, 1.0)


def tone_curve(rgb: np.ndarray) -> np.ndarray:
    """CIRA/Polar2Grid-style nonlinear stretch for natural true-color brightness.

    Piecewise curve mapping input reflectance (post-sun-correction, 0..1) to
    display 0..1: near-linear in the shadows with a gentle toe, rolling into a
    sqrt-ish highlight compression so bright cloud tops don't clip to flat
    white. This is the standard EUMETSAT/CIRA true-color enhancement, sampled
    to control points and interpolated.
    """
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
    GeoColor-lite night fade; omit it for a daytime-only (dark at night) image.
    All inputs are co-registered TOA reflectance on the same (red-res) grid.
    Returns (rgb, cos_sza).
    """
    cos_sza, sun_zen, sun_az = solar_geometry(lats, lons, when)
    sat_zen, sat_az = satellite_geometry(lats, lons, sub_sat_lon, when)

    # Native red kept aside for ratio sharpening after the band ops.
    red_native = sun_correct(red, cos_sza)
    blue_c = sun_correct(blue, cos_sza)
    if green is None:
        if veggie is None:
            raise ValueError("ABI true color needs a veggie band to synthesize green")
        veggie_c = sun_correct(veggie, cos_sza)
        green_c = synth_green(red_native, veggie_c, blue_c)
    else:
        green_c = sun_correct(green, cos_sza)

    rgb = np.clip(np.dstack([red_native, green_c, blue_c]), 0.0, 1.0)

    if do_rayleigh:
        rgb = rayleigh_correct(rgb, sun_zen, sat_zen, sun_az, sat_az,
                               platform_name, sensor, scale=RAYLEIGH_SCALE)

    if do_ratio_sharpen:
        rgb = ratio_sharpen(rgb, np.clip(red_native, 0.0, 1.0))

    rgb = tone_curve(rgb)

    # GeoColor-lite: fade to clean-IR at night. Done AFTER the tone curve so the
    # day side is fully developed; the IR night layer has its own ramp.
    if ir_bt is not None:
        rgb = night_blend(rgb, ir_bt, cos_sza)

    return rgb.astype(np.float32), cos_sza.astype(np.float32)
