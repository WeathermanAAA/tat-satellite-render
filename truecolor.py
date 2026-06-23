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
  * Land-aware Rayleigh relax: FULL Rayleigh is tuned for dark ocean; over
    VEGETATED land it over-subtracts the blue path radiance (vegetation is dark
    in red, so pyspectral's red-keyed backoff never fires) and forest/savanna
    render muddy BROWN. An NDVI gate (NIR vs red) AND an absolute NIR floor scale
    the Rayleigh correction down over vegetation only -- EXACTLY a no-op over
    ocean (low NIR / NDVI<0), cloud/snow/glint (spectrally flat -> NDVI~0) and
    near-black water (NIR floor), so those stay byte-identical (LAND_* knobs).
  * ABI has no green band -> synthesize green from an **AHI-DERIVED** linear
    model (green_synth_ahi.json), learned from Himawari's REAL 0.51um green over
    co-registered AHI scenes (the CIRA GeoColor hybrid-green idea):
        G = c0 + cB*Blue + cR*Red + cV*Veggie(0.86)   (cB dominant)
    Green sits right beside blue (0.51 vs 0.47um) so it is mostly BLUE, with a
    small veggie lift for vegetation. This replaces the old fixed CIMSS mix
    (0.45*Red + 0.10*Veggie + 0.45*Blue), whose heavy RED share over-greened at
    low sun and produced the terminator GREEN CAST; the learned green matches the
    warm/correct AHI look at every sun angle while keeping ocean deep-blue and
    vegetation green. If the asset is missing it falls back to the fixed mix.
    Himawari AHI has a native 0.51 green band and skips synthesis entirely.
  * Solar-zenith normalization: ABI CMI and our AHI albedo are raw TOA
    reflectance (NOT sun-angle normalized) -> divide by cos(SZA).
  * A CIRA/EUMETSAT tone curve (gamma-ish stretch) for natural brightness.
  * Red-band ratio sharpening lifts the 1 km green/blue toward the 0.5 km red.
  * Terminator highlight rolloff: near the terminator the divide-by-cos(SZA)
    sun-correction (up to ~10x) blows bright cold cloud tops past 1.0 and the
    Rayleigh [0,1] clip flattens them to white. A sun-angle-gated highlight KNEE
    (HIGHLIGHT_* knobs) soft-compresses each SUN-CORRECTED band BEFORE that clip,
    so distinct tops keep distinct sub-1 values (texture) instead of clipping.
    ZERO in full day (exact no-op) -> midday byte-identical. ABI + AHI.
  * AHI vegetation vibrance: a green-biased, luminance-preserving saturation lift
    (AHI_VIBRANCE_* knobs) applied ONLY on the AHI native-green path, so Himawari
    land/vegetation reads livelier without oversaturating clouds (white) or ocean
    (blue). The GOES synth-green path is untouched.
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
# FALLBACK ONLY: used by synth_green when the AHI-derived green asset
# (green_synth_ahi.json) is absent. The PRIMARY green is the learned AHI model;
# this fixed mix is the legacy CIMSS Natural True Color green kept as a safety
# net. (Previously 0.40/0.40/0.20 — the veggie-heavy / blue-light mix that
# turned open ocean magenta because synth-green collapsed where veggie≈0.)
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

# --- Land-aware Rayleigh relax (vegetation de-browning) ---------------------
# FULL Rayleigh subtraction is tuned to reveal deep-blue OCEAN by stripping the
# blue-heavy molecular path radiance. Over VEGETATED LAND it over-subtracts: the
# only brightness-aware backoff pyspectral applies is keyed on the RED reference
# (it relaxes the correction where red>~20%, i.e. cloud/desert), but vegetation
# is DARK in red (chlorophyll absorption) so it receives the FULL ~9% blue
# subtraction -> blue is nearly removed and lush forest renders a dark muddy
# OLIVE/BROWN (savanna a rust red). Confirmed empirically as the lone shared
# stage (ABI + AHI) that browns land; ratio-sharpen + tone-curve do not.
#
# The fix mirrors pyspectral's own bright-scene relax but keys on the vegetation
# signal it misses -- NDVI from NIR (veggie 0.86um) vs red -- AND-gated with an
# absolute NIR floor. Over vegetation (high NDVI, high NIR) the whole Rayleigh
# correction is scaled DOWN UNIFORMLY across bands (band-balanced -> no
# blue/cyan/green cast, just less haze removal), so land keeps its blue and reads
# natural green/tan. The gate is EXACTLY 0 over:
#   * OCEAN -- liquid water absorbs NIR (NIR ~0.02-0.05), so the NIR floor closes
#     the gate. This also kills the failure mode where NDVI -- a RATIO -- reads a
#     spuriously HIGH value over near-black clear water (tiny red AND tiny NIR).
#     Real deep/teal/turbid water all keep NIR low; measured open ocean here sits
#     at NIR~0.024 and NDVI<0.
#   * CLOUD / SNOW / SUN GLINT -- bright but spectrally FLAT (NIR ~= red), so NDVI
#     ~0 closes the NDVI gate even though NIR is high (glint is a near-mirror
#     reflection of the solar disk, not a low-NIR event).
# So deep-blue ocean, white cloud, snow and glint stay byte-identical. The NIR
# floor additionally fades the relax out as land darkens toward the terminator
# (NIR -> 0), keeping the terminator a no-op. Sensor-shared: ABI veggie=band3,
# AHI veggie=band4; degrades to a no-op if no veggie band is present.
LAND_RAYLEIGH_RELAX = 0.6   # max fraction the Rayleigh correction is cut over full vegetation (0 = OFF)
LAND_NDVI_LO = 0.14         # NDVI <= this -> NO relax (ocean<0, cloud/snow/glint~0, bare soil): gate 0
LAND_NDVI_HI = 0.45         # NDVI >= this -> FULL NDVI gate (dense vegetation)
LAND_NIR_LO = 0.05          # sun-corrected NIR(veggie) <= this -> gate forced to 0 (water/shadow/near-black)
LAND_NIR_HI = 0.10          # NIR >= this -> NIR floor fully open (real vegetation NIR is reliably >=~0.15)

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
# DISABLED (0.0): sunrise/sunset should read as natural true-color RGB, not a warm
# cast. The window/amplitude tuning below is retained so the effect can be re-
# enabled by raising this knob, but by default no warm tint is applied -- the
# terminator is just the (Rayleigh-tapered) true color, which matches the day-side
# hue and avoids both the old crimson wall AND the full-Rayleigh green at the limb.
WARM_TINT_STRENGTH = 0.0    # overall multiplier on the warm tint (0 = OFF, natural RGB)
WARM_TINT_RED_ADD = 0.05    # additive red boost at full tint (halved: subtle golden, not sepia)
WARM_TINT_GREEN_GAIN = 0.08  # multiplicative green attenuation at full tint (halved)
WARM_TINT_BLUE_GAIN = 0.17   # multiplicative blue attenuation at full tint (halved; most -> orange)

# --- Terminator highlight rolloff (cold-cloud-top exposure) -----------------
# Sun-correction (sun_correct: divide by cos SZA, floored at COS_SZA_FLOOR -> up
# to ~10x at low sun) pushes bright/cold cloud tops FAR past 1.0 near the
# terminator; Rayleigh then hard-clips every band to [0,1], flattening those tops
# to texture-less white. A sun-angle-gated highlight KNEE soft-compresses values
# above HIGHLIGHT_KNEE back toward 1.0 with a tanh shoulder so distinct cloud tops
# keep distinct (sub-1) values -- texture survives instead of clipping. CRITICAL
# placement: it runs on each SUN-CORRECTED band BEFORE Rayleigh/clip (the only
# stage where the >1 texture still exists); doing it after the clip would only
# darken already-flattened white. The knee acts ONLY on near-clipping highlights
# (mid/low tones below the knee pass through untouched), and the cos gate confines
# it to the low-sun band -- double containment, so vegetation/ocean and daytime
# cloud are never touched. ZERO in full day: the gate smoothstep saturates EXACTLY
# to 0 at/above HIGHLIGHT_DAY_COS, so the bands are byte-identical there and
# Rayleigh / green-synthesis / the clip reproduce midday exactly. Per band, so it
# helps ABI (incl. its synth-green, built from the compressed bands) and AHI
# alike. All tunable here.
HIGHLIGHT_DAY_COS = 0.30   # cos_sza >= this -> NO rolloff (identity): never fires in full day
HIGHLIGHT_PEAK_COS = 0.20  # cos_sza <= this -> full-strength rolloff (the low-sun terminator band)
HIGHLIGHT_KNEE = 0.72      # sun-corrected level where compression begins (below it: untouched)
HIGHLIGHT_SHOULDER = 2.2   # tanh shoulder width above the knee (larger = gentler, keeps more tonal range)
HIGHLIGHT_STRENGTH = 1.0   # overall amount of the rolloff at full gate (0 = OFF)


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


def land_rayleigh_relax_field(red_c: np.ndarray, veggie_c: "np.ndarray | None") -> np.ndarray:
    """Per-pixel multiplier in [1-LAND_RAYLEIGH_RELAX, 1] that scales the Rayleigh
    correction DOWN over vegetated land while leaving ocean/cloud/snow/glint
    untouched.

    The land gate is the product of two smoothsteps over the SUN-CORRECTED,
    pre-Rayleigh NIR (``veggie_c``) and red (``red_c``) reflectances:
      * an NDVI gate  -- 0 for NDVI<=LAND_NDVI_LO (ocean<0; cloud/snow/glint ~0,
        being spectrally flat; bare soil), ramping to 1 by LAND_NDVI_HI; and
      * an absolute NIR floor -- 0 for NIR<=LAND_NIR_LO, ramping to 1 by
        LAND_NIR_HI. NDVI is a ratio, so near-black clear water (tiny red AND
        tiny NIR) can read a spuriously high NDVI; the NIR floor closes the gate
        there because liquid water absorbs NIR (NIR ~0.02-0.05) while vegetation
        NIR is reliably >=~0.15.
    The factor is therefore EXACTLY 1.0 (an exact no-op -> byte-identical) over
    ocean, cloud, snow and sun glint, and drops to 1-LAND_RAYLEIGH_RELAX over
    full, well-lit vegetation. NaN inputs (partial-tile edges) map to 1.0 so the
    relax never poisons an otherwise-valid pixel (matters on the AHI path, where
    veggie is not in the output RGB). Returns all-ones when no veggie band is
    available or the feature is disabled (LAND_RAYLEIGH_RELAX<=0)."""
    if veggie_c is None or LAND_RAYLEIGH_RELAX <= 0.0:
        return np.ones_like(red_c)
    r = np.clip(red_c, 0.0, None)
    v = np.clip(veggie_c, 0.0, None)
    ndvi = (v - r) / np.maximum(v + r, 1e-6)
    gate = _smoothstep(LAND_NDVI_LO, LAND_NDVI_HI, ndvi) * _smoothstep(LAND_NIR_LO, LAND_NIR_HI, v)
    factor = (1.0 - LAND_RAYLEIGH_RELAX * gate).astype(red_c.dtype, copy=False)
    factor[np.isnan(red_c) | np.isnan(veggie_c)] = 1.0   # never let a NaN edge poison a valid pixel
    return factor


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


def highlight_rolloff_field(cos_sza: np.ndarray) -> np.ndarray:
    """Per-pixel highlight-knee strength: 0 in full day (cos_sza >=
    HIGHLIGHT_DAY_COS), rising to HIGHLIGHT_STRENGTH at/below HIGHLIGHT_PEAK_COS
    (the deep terminator). The smoothstep saturates EXACTLY to 0 above
    HIGHLIGHT_DAY_COS, so the daytime knee is an exact no-op -> byte-identical."""
    return HIGHLIGHT_STRENGTH * (
        1.0 - _smoothstep(HIGHLIGHT_PEAK_COS, HIGHLIGHT_DAY_COS, cos_sza)
    )


def highlight_rolloff(band: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """Soft-compress a single SUN-CORRECTED band's highlights above HIGHLIGHT_KNEE
    with a tanh shoulder, gated by the sun angle (see HIGHLIGHT_* knobs).

    ``band`` is a sun-normalized reflectance band (H,W) and near the terminator
    MAY be far above 1 (sun_correct multiplies by up to ~10). It is applied
    BEFORE Rayleigh + the [0,1] clip, which is the only place the >1 cloud-top
    texture still exists: above the knee the value is pushed toward a shoulder
    ``soft(x) = knee + (1-knee)*tanh((x-knee)/shoulder)`` that maps [knee, +inf)
    -> [knee, 1), monotonically, so distinct cloud tops keep distinct values
    (= texture) instead of all clipping to flat white. The blend is
    ``out = x + s*(soft(x) - x)``: at full strength (s==STRENGTH) over-knee values
    land on the <1 shoulder so the downstream Rayleigh clip is a no-op there;
    below the knee, values pass through untouched. At s==0 (full day) the result
    is EXACTLY ``band`` (incl. >1 values), so Rayleigh / green-synthesis / the
    clip all see byte-identical input -> midday byte-identical. Monotonic in the
    input, so it never inverts tonal ordering."""
    s = highlight_rolloff_field(cos_sza)
    knee = HIGHLIGHT_KNEE
    over = band > knee
    soft = knee + (1.0 - knee) * np.tanh((band - knee) / HIGHLIGHT_SHOULDER)
    return np.where(over, band + s * (soft - band), band)


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


def _load_green_model():
    """AHI-derived synthetic-green coefficients (c0, cB, cR, cV) from
    green_synth_ahi.json, or None to fall back to the legacy fixed mix. The
    coeffs were fit from Himawari's REAL 0.51um green band over co-registered AHI
    scenes (the CIRA GeoColor hybrid-green idea) in this pipeline's post-sun-
    correct + Rayleigh space, so applying them to ABI makes GOES converge on the
    correct AHI look instead of the linear mix's low-sun green cast."""
    import json
    import pathlib
    try:
        a = json.loads(pathlib.Path(__file__).with_name("green_synth_ahi.json").read_text())
        if a.get("features") == ["1", "blue", "red", "veggie"] and len(a["coef"]) == 4:
            return tuple(float(x) for x in a["coef"])
        log.warning("green_synth_ahi.json has unexpected shape; using linear fallback")
    except Exception as e:  # noqa: BLE001 - missing/corrupt asset must not break rendering
        log.warning("green model asset unavailable (%s); using the linear fallback", e)
    return None


_GREEN_AHI = _load_green_model()   # (c0, cB, cR, cV) or None


def synth_green(red: np.ndarray, veggie: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Synthesize the missing ABI green.

    PRIMARY: an AHI-DERIVED green -- a linear model learned from Himawari's native
    0.51um green (G = c0 + cB*blue + cR*red + cV*veggie). Green is mostly BLUE
    (0.51um sits right next to 0.47um), with a small veggie lift for vegetation;
    the old fixed CIMSS mix (0.45*Red + 0.10*Veggie + 0.45*Blue) put far too much
    RED weight and over-greened at low sun -> the terminator green cast. This
    learned green matches the warm/correct AHI look at every sun angle. ABI only;
    AHI keeps its native green. FALLBACK: the legacy fixed mix if the asset is
    absent. Output is clipped to [0,1] downstream (assemble_truecolor)."""
    if _GREEN_AHI is not None:
        c0, cB, cR, cV = _GREEN_AHI
        return c0 + cB * blue + cR * red + cV * veggie
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


# --- AHI vegetation vibrance (Himawari native-green path) -------------------
# AHI's native 0.51um green renders true color correctly, but land/vegetation
# reads a touch dull/dark. A green-biased VIBRANCE lifts saturation only where a
# vegetation driver is positive: GREEN-OVER-BLUE penalized by RED-OVER-GREEN,
# (g - b) - max(r - g, 0). That is positive for vegetation (incl. dull olive
# where red ~ green) but ZERO for the things we must not touch:
#   * cloud / grey-white  -> r~=g~=b -> driver ~ 0      (clouds stay white)
#   * deep ocean          -> blue leads green -> g-b<0  (ocean stays blue)
#   * desert / bare soil  -> red leads green -> penalty  (no orange neon)
# (g-b) alone, however, is also positive for CYAN/TEAL water (turbid/shallow/
# sediment seas: green>=blue, red strongly absorbed) -- which would wrongly green
# the water. Vegetation reflects red MORE than blue (r>=b) while teal water absorbs
# red (b>r), so a WATER GUARD multiplies the driver down wherever blue exceeds red,
# protecting teal/turbid seas while leaving land (r>=b) untouched. The lift is
# tapered by (1 - saturation) so already-vivid greens don't go neon, capped at
# AHI_VIBRANCE_MAX, and luminance-preserving (Rec.601) so brightness is unchanged.
# AHI-ONLY: assemble_truecolor calls it only for sensor=='ahi', so the GOES
# synth-green path is untouched. Set AHI_VIBRANCE_STRENGTH to 0 to disable.
AHI_VIBRANCE_STRENGTH = 6.0   # gain applied to the green-driver vibrance (0 = OFF)
AHI_VIBRANCE_MAX = 0.32       # cap on the per-pixel saturation boost (prevents neon)
AHI_VIBRANCE_WATER_GUARD = 0.06  # ramp width: driver -> 0 once blue leads red by this (teal water)
AHI_VIBRANCE_LUMA = (0.299, 0.587, 0.114)  # Rec.601 luma weights (the saturation pivot)


def ahi_vegetation_vibrance(rgb: np.ndarray) -> np.ndarray:
    """Green-biased, luminance-preserving vibrance for the AHI true-color path.

    Saturation is boosted in proportion to a vegetation driver, green-over-blue
    minus red-over-green ``(g - b) - max(r - g, 0)``, times a WATER GUARD that
    ramps the driver to 0 wherever blue leads red (cyan/teal/turbid water absorbs
    red, so b>r; vegetation reflects red, so r>=b). The driver is thus positive
    for vegetation (lush and dull olive) and zero for grey/white cloud, blue-led
    deep ocean, red-led desert, AND green-led teal/turbid water -- so only land
    vegetation is lifted. The boost is tapered by ``(1 - saturation)`` (no neon on
    already-vivid greens), capped at AHI_VIBRANCE_MAX, and applied around the
    Rec.601 luma so brightness is unchanged (clouds keep their white). AHI only
    (assemble-gated)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)  # HSV saturation
    veg = (g - b) - np.maximum(r - g, 0.0)                # green-over-blue, red-penalized
    water_guard = np.clip(1.0 - (b - r) / AHI_VIBRANCE_WATER_GUARD, 0.0, 1.0)  # ->0 where blue>red
    green_drive = np.clip(veg * water_guard, 0.0, None)   # vegetation-positive, water-safe
    amt = np.clip(AHI_VIBRANCE_STRENGTH * green_drive * (1.0 - sat),
                  0.0, AHI_VIBRANCE_MAX)[..., None]
    wr, wg, wb = AHI_VIBRANCE_LUMA
    luma = (wr * r + wg * g + wb * b)[..., None]
    out = luma + (rgb - luma) * (1.0 + amt)
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype)


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

    # 1b) Terminator highlight rolloff -- on the SUN-CORRECTED bands, BEFORE
    #     Rayleigh's [0,1] clip. Sun-correction is what blows cold cloud tops past
    #     1.0; compressing here (gated, identity in full day) is the ONLY place the
    #     >1 cloud-top texture still exists, so by the time Rayleigh clips, the
    #     terminator values already sit <=1 and the clip keeps them textured instead
    #     of flattening to white. At full day the gate is 0 -> the bands are
    #     byte-identical, so Rayleigh / green-synthesis / the clip reproduce midday.
    red_c = highlight_rolloff(red_c, cos_sza)
    blue_c = highlight_rolloff(blue_c, cos_sza)
    if veggie_c is not None:
        veggie_c = highlight_rolloff(veggie_c, cos_sza)
    if green_c is not None:
        green_c = highlight_rolloff(green_c, cos_sza)

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
        # Land-aware relax: scale the Rayleigh correction DOWN over vegetated land
        # (NDVI-gated, from the sun-corrected NIR/red) so lush forest and savanna
        # keep their blue and read natural green/tan instead of an over-subtracted
        # muddy brown. EXACTLY 1.0 (no-op) over ocean (NDVI<0) and cloud (NDVI~0),
        # so deep-blue ocean and white cloud stay byte-identical. Applied to every
        # band (uniform, band-balanced) so no blue/cyan cast is introduced.
        ray_scale = ray_scale * land_rayleigh_relax_field(red_c, veggie_c)
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

    # AHI-only: green-biased vegetation vibrance (livelier land without touching
    # clouds/ocean). The GOES synth-green path is left untouched.
    if sensor.lower() == "ahi":
        rgb = ahi_vegetation_vibrance(rgb)

    # Golden-hour: warm the day-side RGB toward red/orange near the terminator
    # (sunrise/sunset) BEFORE the night fade. Exact no-op in daytime -> midday
    # byte-identical; develops at the terminator then fades into the IR night.
    rgb = warm_terminator_tint(rgb, cos_sza)

    # GeoColor-lite: fade to clean-IR at night (after the day side is developed).
    if ir_bt is not None:
        rgb = night_blend(rgb, ir_bt, cos_sza)

    return rgb.astype(np.float32), cos_sza.astype(np.float32)
