"""True-color RGB pipeline for the geostationary ring (ABI / AHI / AMI / FCI).

REBUILT 2026-07-23 to the authoritative operational recipe so every ring
sensor -- GOES-East/West (ABI), Himawari (AHI), GK-2A (AMI), Meteosat MTG
(FCI) -- renders through ONE shared pipeline and lands on the SAME look
(reference target: CIRA SLIDER GeoColor). Every algorithm and constant below
is lifted VERBATIM from Satpy 0.60's true_color composite chain and
pyspectral 0.14 (the operational implementations behind SLIDER-class true
color); the only intentional departures are listed at the end of this
docstring.

PIPELINE ORDER (per band role; Satpy modifier order):

  1. calibrate -> TOA reflectance (upstream: the fetchers hand us 0..1
     reflectance factor, NOT sun-normalized -- ABI CMI / AHI-AMI albedo /
     FCI L1c convention).
  2. SUN-ZENITH NORMALIZE (``sunz_correct``): divide by cos(SZA), correction
     CLAMPED at SZA 88 deg, then tapered to ZERO by 95 deg with Satpy's
     log-shaped falloff (satpy.modifiers.angles._sunzen_corr_cos_ndarray,
     limit=88, max_sza=95). This kills the old terminator blow-out at its
     source -- no highlight-knee compensation needed. The NIR/veggie band
     instead uses the Li & Shibata (2006) EFFECTIVE SOLAR PATHLENGTH
     correction (satpy.utils.atmospheric_path_length_correction) -- more
     accurate at high SZA for the longer NIR path, same 88/95 clamp+taper.
  3. RAYLEIGH CORRECTION (``rayleigh_band``): pyspectral Rayleigh LUT,
     atmosphere='us-standard', aerosol_type='rayleigh_only', applied to the
     REAL measured solar bands only -- blue, red, and the real ~0.51 um green
     where the sensor has one (AHI B02 / AMI VI005 / FCI vis_05). The NIR
     0.86 um veggie band is NOT Rayleigh-corrected (Satpy: sunz only;
     molecular scattering at 0.86 um is negligible). Per-band effective
     wavelengths are SENSOR-SPECIFIC: we pass the Satpy band NAME (C01, B02,
     VI005, vis_05, ...) and pyspectral resolves the SRF-weighted effective
     wavelength from that sensor's spectral response. The sun-corrected red
     band rides along as ``redband`` so pyspectral's cloud relax backs the
     correction off over bright cloud (no over-darkened cloud tops). At high
     sun zenith the correction itself is tapered (reduce_rayleigh_highzenith,
     70 deg -> 95 deg, strength 0.6 -- Satpy's documented geostationary
     day/night-edge configuration) so the limb never over-corrects into the
     old green/blue fringe.
  4. PER-SENSOR GREEN (``make_green``), each formulated to land on the same
     ~0.55 um vegetation-true green so the ring matches:
       * ABI  (no green band): SimulatedGreen, CIMSS/Bah hybrid
         G = 0.45*C01 + 0.45*C02 + 0.10*C03 on the corrected bands
         (satpy abi.yaml 'green'; Bah et al. 2018, 10.1029/2018EA000379).
       * AHI + AMI (0.51 um green misses the chlorophyll bump): HybridGreen
         G = 0.85*R(0.51) + 0.15*R(0.86) (satpy HybridGreen, fraction 0.15;
         Miller et al. 2016, 10.1175/BAMS-D-15-00154.2).
       * FCI: NDVIHybridGreen -- the NIR blend fraction varies per pixel with
         NDVI, limits [0.15, 0.05], strength 3.0 (satpy fci.yaml
         'ndvi_hybrid_green').
  5. SELF-SHARPEN (``self_sharpen``): ratio-sharpen green/blue to the red
     band's finer native resolution (satpy SelfSharpenedRGB: low-res red =
     2x2 block mean of the high-res red; ratio = red/lowres, non-finite or
     negative -> 1, clipped to [0, 1.5]).
  6. TONE CURVE (``cira_stretch``): the CIRA log stretch, identical for every
     band and EVERY sensor -- display = (log10(R) + 1.6517) / 1.9888 for R in
     0..1 reflectance (satpy enhancements 'cira_stretch'; R=100% maps to
     ~0.83, leaving real headroom for sun-corrected >100% cloud tops instead
     of clipping them flat). ONE curve across the ring: divergent tone curves
     are the #1 seam source between sensors.
  7. DAY/NIGHT: cross-fade the day RGB into grayscale clean-IR across the
     terminator (``night_blend``). The day weight uses the SAME 88->95 deg
     taper as the sun correction, so IR fills in exactly as fast as the
     day side fades out -- no hard edge, no double-dark band.
  8. SUN-GLINT TAME (``glint_tame``): inside the specular cone over open
     water, gently desaturate + compress the glare toward its luminance.
     Purely cosmetic, conservative, and OFF over land/cloud (NDVI + NIR
     gates); GLINT_TAME_STRENGTH=0 disables bit-exactly.

DOCUMENTED DEPARTURES from the working spec / Satpy:
  * The spec's ORDER line lists "Rayleigh -> SZA-normalize"; Satpy's actual
    operational chain is sunz_corrected THEN rayleigh_corrected (the LUT
    reflectance is a true bidirectional reflectance, subtracted in the
    sun-normalized space). We follow Satpy -- the spec's own reference.
  * FCI true_color in satpy adds 'sunz_reduced' (an extra darkening toward
    the terminator, satpy issue #2643). We deliberately DON'T: the ring's
    IR cross-fade covers the terminator, and one shared SZA treatment across
    all sensors is the harmonization point.
  * Sensors without a native green never call the AHI/FCI green paths; the
    old AHI-derived learned green, vegetation bumps, vibrance and land-aware
    Rayleigh relax stages are RETIRED -- with rayleigh_only aerosols +
    hybrid greens + cira_stretch they are unnecessary compensations (they
    were tuned against the old LUT tone curve and marine aerosol Rayleigh).

PHASE 2 HOOK (gold standard, not yet implemented): per-sensor 3x3 matrices
from each sensor's spectral response functions -> CIE 1931 XYZ -> sRGB (the
JMA True Color Reproduction / CIRA method) would make the ring identical BY
CONSTRUCTION. The place to apply it is immediately before ``cira_stretch``
(replace the per-sensor green ladder with the matrix product). See
OBJFIX-METHODS-style provenance doc in the repo runbook when built.

Geometry (sun + geostationary satellite zenith/azimuth) comes from pyorbital.
"""

from __future__ import annotations

import logging
import datetime as dt
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("tat-satellite.truecolor")

# ---------------------------------------------------------------------------
# Shared operational constants (Satpy 0.60 verbatim)
# ---------------------------------------------------------------------------

# Sun-zenith correction clamp/taper (satpy SunZenithCorrector defaults).
SUNZ_LIMIT_DEG = 88.0     # 1/cos correction applied up to here...
SUNZ_MAX_DEG = 95.0       # ...then log-tapered to exactly 0 here.

# Rayleigh LUT selection (satpy visir.yaml 'rayleigh_corrected').
RAYLEIGH_ATMOSPHERE = "us-standard"
RAYLEIGH_AEROSOL = "rayleigh_only"
# High-sun-zenith taper of the Rayleigh correction itself (satpy's documented
# geostationary day/night-edge configuration for PSPRayleighReflectance).
RAYLEIGH_REDUCE_LOW_DEG = 70.0
RAYLEIGH_REDUCE_HIGH_DEG = 95.0
RAYLEIGH_REDUCE_STRENGTH = 0.6

# ABI SimulatedGreen fractions (blue C01, red C02, veggie C03) -- the
# CIMSS/Bah hybrid (satpy abi.yaml documents this exact option).
ABI_GREEN_FRACTIONS = (0.45, 0.45, 0.10)
# AHI/AMI HybridGreen NIR fraction (satpy HybridGreen default; F=0.15).
HYBRID_GREEN_FRACTION = 0.15
# FCI NDVIHybridGreen (satpy fci.yaml 'ndvi_hybrid_green').
NDVI_GREEN_LIMITS = (0.15, 0.05)
NDVI_GREEN_STRENGTH = 3.0
NDVI_GREEN_NDVI_MIN = 0.0
NDVI_GREEN_NDVI_MAX = 1.0

# cira_stretch constants (satpy enhancements._cira_stretch, in 0..1 units:
# out = (log10(R) - log10(0.0223)) / ((1 - log10(0.0223)) * 0.75)).
_CIRA_LOG_ROOT = np.log10(0.0223)               # = -1.6517...
_CIRA_DENOM = (1.0 - _CIRA_LOG_ROOT) * 0.75     # = 1.9888...

# Self-sharpen: satpy _get_sharpening_ratio clips the ratio to [0, 1.5];
# SelfSharpenedRGB's low-res red is the 2x2 block mean of the high-res red.
SHARPEN_RATIO_MAX = 1.5
# Red-vs-green/blue native resolution ratio per sensor (0.5 km red vs 1 km
# for ABI/AHI/AMI; FCI FDHSI feeds all-1km -> no sharpening).
SHARPEN_BLOCK = {"abi": 2, "ahi": 2, "ami": 2, "fci": 1}

# Satpy band names per sensor role -- pyspectral resolves these to the
# sensor's SRF-weighted effective wavelength (spec item 1: per-band,
# sensor-specific). 'green' None = no native green (synthesized).
SENSOR_BANDS = {
    "abi": {"red": "C02", "green": None, "blue": "C01", "veggie": "C03"},
    "ahi": {"red": "B03", "green": "B02", "blue": "B01", "veggie": "B04"},
    "ami": {"red": "VI006", "green": "VI005", "blue": "VI004", "veggie": "VI008"},
    "fci": {"red": "vis_06", "green": "vis_05", "blue": "vis_04", "veggie": "vis_08"},
}
# Central wavelengths (um) -- FALLBACK ONLY, used when pyspectral cannot
# resolve the band name (e.g. an RSR download hiccup at runtime).
FALLBACK_WL = {
    "abi": {"red": 0.64, "blue": 0.47, "green": 0.51},
    "ahi": {"red": 0.64, "blue": 0.47, "green": 0.51},
    "ami": {"red": 0.64, "blue": 0.47, "green": 0.51},
    "fci": {"red": 0.64, "blue": 0.444, "green": 0.51},
}

# --- Sun-glint tame (order step 8) ------------------------------------------
# Specular geometry: glint angle = angle between the satellite view ray and
# the sun's mirror reflection off a flat water surface. Inside the cone the
# glare is pulled toward its luminance (desaturated) and softly compressed.
# Water-gated (NDVI < 0 and low NIR -- liquid water absorbs 0.86 um), so land
# and cloud are exact no-ops. Conservative by design; 0 disables bit-exactly.
GLINT_CONE_FULL_DEG = 12.0   # glint angle <= this -> full tame weight
GLINT_CONE_ZERO_DEG = 28.0   # glint angle >= this -> zero (smoothstep ramp)
GLINT_TAME_STRENGTH = 0.35   # overall strength (0 = OFF, bit-exact)
GLINT_DESAT = 0.6            # fraction of chroma removed at full weight
GLINT_DARKEN = 0.25          # fraction of luma-above-ocean-base removed
GLINT_WATER_NIR_MAX = 0.10   # sun-corrected NIR above this -> not water
_GLINT_LUMA = (0.299, 0.587, 0.114)   # Rec.601, same pivot the repo uses

# GeoColor-lite night: clean-IR grayscale range (cold = white).
IR_T_WARM = 303.0
IR_T_COLD = 183.0

# Geostationary satellite height above the surface (km) for pyorbital's
# observer-look geometry. 42164 km geocentric - 6378 km mean Earth radius.
GEO_SAT_ALT_KM = 35786.0


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep, clamped to [0,1]: 0 for x<=edge0, 1 for x>=edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Geometry (pyorbital) -- unchanged machinery
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
# Step 2: sun-zenith normalization (Satpy verbatim)
# ---------------------------------------------------------------------------
def _sunz_grad_factor(cos_sza: np.ndarray) -> np.ndarray:
    """Satpy's log-shaped falloff from 1 at SUNZ_LIMIT_DEG to 0 at
    SUNZ_MAX_DEG: grad = clip(1 - ln((sza-limit)/(max-limit) + 1)/ln 2, 0).
    (>=1 below the limit; callers gate on cos_sza > cos(limit).)"""
    limit_rad = np.deg2rad(SUNZ_LIMIT_DEG)
    max_rad = np.deg2rad(SUNZ_MAX_DEG)
    with np.errstate(invalid="ignore"):
        grad = (np.arccos(np.clip(cos_sza, -1.0, 1.0)) - limit_rad) / (max_rad - limit_rad)
        grad = 1.0 - np.log(grad + 1.0) / np.log(2.0)
    return np.clip(grad, 0.0, None)


def sunz_correct(refl: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """Standard 1/cos(SZA) reflectance normalization, clamped at 88 deg and
    log-tapered to 0 by 95 deg (satpy._sunzen_corr_cos_ndarray, limit=88,
    max_sza=95). NaN cos -> correction 0 (matches satpy's night forcing)."""
    limit_cos = np.cos(np.deg2rad(SUNZ_LIMIT_DEG))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = 1.0 / cos_sza
    grad = _sunz_grad_factor(cos_sza)
    corr = np.where(cos_sza > limit_cos, corr, grad / limit_cos)
    corr = np.where(np.isnan(cos_sza), 0.0, corr)
    return (refl * corr).astype(np.float32, copy=False)


def sunz_correct_pathlength(refl: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """Li & Shibata (2006) effective-solar-pathlength correction for the NIR
    term (satpy.utils.atmospheric_path_length_correction, limit=88,
    max_sza=95): corr = 24.35 / (2*cos + sqrt(498.5225*cos^2 + 1)), clamped
    at 88 deg and log-tapered to 0 by 95 deg like the visible correction."""
    def _li_shibata(c):
        return 24.35 / (2.0 * c + np.sqrt(498.5225 * c * c + 1.0))

    limit_cos = np.cos(np.deg2rad(SUNZ_LIMIT_DEG))
    corr = _li_shibata(np.clip(cos_sza, -1.0, 1.0))
    corr_lim = _li_shibata(limit_cos)
    grad = _sunz_grad_factor(cos_sza)
    corr = np.where(cos_sza > limit_cos, corr, grad * corr_lim)
    corr = np.where(np.isnan(cos_sza), 0.0, corr)
    return (refl * corr).astype(np.float32, copy=False)


def terminator_day_weight(cos_sza: np.ndarray) -> np.ndarray:
    """Day-side weight for the IR cross-fade: exactly 1 up to SZA 88 deg,
    falling to exactly 0 at 95 deg on the SAME log-shaped curve as the sun
    correction -- IR fills in precisely as fast as the day side fades."""
    limit_cos = np.cos(np.deg2rad(SUNZ_LIMIT_DEG))
    w = np.where(cos_sza > limit_cos, 1.0, np.clip(_sunz_grad_factor(cos_sza), 0.0, 1.0))
    return np.where(np.isnan(cos_sza), 0.0, w)


# ---------------------------------------------------------------------------
# Step 3: Rayleigh correction (pyspectral; satpy PSPRayleighReflectance)
# ---------------------------------------------------------------------------
# Correctors cached per (platform, sensor): construction re-reads the LUT
# HDF5 from disk. Only SUCCESSFUL constructions cache, so a transient init
# failure (e.g. LUTs still downloading) retries on the next frame instead of
# pinning the degraded no-Rayleigh path forever.
_RAYLEIGH_CACHE: dict = {}
_RAYLEIGH_LOCK = threading.Lock()


def _make_rayleigh(platform_name: str, sensor: str):
    """Build (or reuse) a pyspectral Rayleigh corrector with the operational
    LUT selection (us-standard atmosphere, rayleigh_only aerosol), or None."""
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
            corrector = Rayleigh(platform_name, sensor,
                                 atmosphere=RAYLEIGH_ATMOSPHERE,
                                 aerosol_type=RAYLEIGH_AEROSOL)
        except Exception as e:
            log.warning("Rayleigh(%s,%s) init failed (%s); skipping", platform_name, sensor, e)
            return None
        _RAYLEIGH_CACHE[key] = corrector
        return corrector


def rayleigh_band(
    band: np.ndarray,
    band_name,
    sun_zenith: np.ndarray,
    sat_zenith: np.ndarray,
    azidiff: np.ndarray,
    red_ref_pct: np.ndarray,
    corrector,
    fallback_wl: float = 0.64,
) -> np.ndarray:
    """Subtract molecular (Rayleigh) path reflectance from one sun-corrected
    band, satpy-style: band name first (pyspectral resolves the SENSOR'S
    SRF-weighted effective wavelength), central wavelength as fallback; the
    sun-corrected red in percent as ``redband`` (cloud relax); then the
    high-SZA taper (reduce_rayleigh_highzenith 70->95 deg, strength 0.6).
    Returns the band unchanged if the corrector is missing or fails --
    degrade hazier, never fail. pyspectral works in reflectance percent."""
    if corrector is None:
        return band
    try:
        try:
            corr_pct = corrector.get_reflectance(
                sun_zenith, sat_zenith, azidiff, band_name, red_ref_pct)
        except (KeyError, IOError, OSError):
            log.warning("Rayleigh band-name lookup failed for %s; using "
                        "central wavelength %.3f um", band_name, fallback_wl)
            corr_pct = corrector.get_reflectance(
                sun_zenith, sat_zenith, azidiff, fallback_wl, red_ref_pct)
        corr_pct = corrector.reduce_rayleigh_highzenith(
            sun_zenith, corr_pct,
            RAYLEIGH_REDUCE_LOW_DEG, RAYLEIGH_REDUCE_HIGH_DEG,
            RAYLEIGH_REDUCE_STRENGTH)
    except Exception as e:
        log.warning("Rayleigh get_reflectance failed for %s (%s); skipping band", band_name, e)
        return band
    return (band - corr_pct.astype(band.dtype, copy=False) / 100.0)


# ---------------------------------------------------------------------------
# Step 4: per-sensor green synthesis (Satpy compositors, verbatim math)
# ---------------------------------------------------------------------------
def make_green(sensor: str,
               red_c: np.ndarray,
               blue_c: np.ndarray,
               green_c: "np.ndarray | None",
               veggie_c: "np.ndarray | None") -> np.ndarray:
    """The ring's green ladder -- every formula targets the same ~0.55 um
    vegetation-true green so the sensors match:

      * abi      : SimulatedGreen 0.45*blue + 0.45*red + 0.10*veggie
      * ahi, ami : HybridGreen    0.85*green + 0.15*veggie
      * fci      : NDVIHybridGreen -- per-pixel NIR fraction from NDVI
                   (limits [0.15, 0.05], strength 3.0)

    All inputs are the corrected bands (sunz for all; Rayleigh on the real
    measured blue/red/green only -- never the NIR). Falls back sensibly when
    a band is missing (native green alone beats a broken hybrid)."""
    s = sensor.lower()
    if s == "abi":
        if veggie_c is None:
            raise ValueError("ABI true color needs the veggie band to synthesize green")
        fb, fr, fv = ABI_GREEN_FRACTIONS
        return fb * blue_c + fr * red_c + fv * veggie_c
    if green_c is None:
        raise ValueError(f"sensor {sensor!r} true color needs a native green band")
    if veggie_c is None:
        log.warning("%s: no veggie band; using native green without the hybrid NIR blend", sensor)
        return green_c
    if s in ("ahi", "ami"):
        f = HYBRID_GREEN_FRACTION
        return (1.0 - f) * green_c + f * veggie_c
    if s == "fci":
        lo, hi = NDVI_GREEN_LIMITS
        st = NDVI_GREEN_STRENGTH
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (veggie_c - red_c) / (veggie_c + red_c)
        ndvi = np.clip(ndvi, NDVI_GREEN_NDVI_MIN, NDVI_GREEN_NDVI_MAX)
        if st != 1.0:
            ndvi = ndvi ** st / (ndvi ** st + (1.0 - ndvi) ** st)
        frac = (ndvi - NDVI_GREEN_NDVI_MIN) / (NDVI_GREEN_NDVI_MAX - NDVI_GREEN_NDVI_MIN) \
            * (hi - lo) + lo
        return (1.0 - frac) * green_c + frac * veggie_c
    raise ValueError(f"unknown sensor {sensor!r} for green synthesis")


# ---------------------------------------------------------------------------
# Step 5: self-sharpen to the red band's native resolution (Satpy verbatim)
# ---------------------------------------------------------------------------
def self_sharpen(rgb: np.ndarray, sensor: str) -> np.ndarray:
    """Satpy SelfSharpenedRGB on the common grid: the 'low-res red' is the
    2x2 block mean of the (finer-native) red channel; green and blue are
    multiplied by ratio = red / lowres_red with satpy's exact guards
    (non-finite or negative -> 1; clip [0, SHARPEN_RATIO_MAX]). A no-op for
    sensors whose bands share one native resolution (block size 1)."""
    block = SHARPEN_BLOCK.get(sensor.lower(), 1)
    if block <= 1:
        return rgb
    from scipy.ndimage import uniform_filter
    red = rgb[..., 0]
    finite = np.isfinite(red)
    filled = np.where(finite, red, 0.0)
    # NaN-safe block mean: normalize the filtered field by the filtered mask
    # so off-disk NaN edges don't bleed darkness into the ratio.
    num = uniform_filter(filled, size=block, mode="nearest")
    den = uniform_filter(finite.astype(red.dtype), size=block, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        low = num / den
        ratio = red / low
    ratio = np.where(np.isfinite(ratio) & (ratio >= 0.0), ratio, 1.0)
    ratio = np.clip(ratio, 0.0, SHARPEN_RATIO_MAX)
    out = rgb.copy()
    out[..., 1] = rgb[..., 1] * ratio
    out[..., 2] = rgb[..., 2] * ratio
    return out


# ---------------------------------------------------------------------------
# Step 6: the ONE shared tone curve (CIRA stretch, Satpy verbatim)
# ---------------------------------------------------------------------------
def cira_stretch(rgb: np.ndarray) -> np.ndarray:
    """CIRA logarithmic stretch (satpy 'cira_stretch'), in 0..1 units:
    display = (log10(R) + 1.6517) / 1.9888, clipped to [0, 1]. Identical for
    R/G/B on every sensor -- the ring's single tone curve. NaN passes
    through (off-disk stays transparent downstream)."""
    x = np.clip(rgb, np.finfo(np.float32).eps, None)   # satpy clips at eps
    with np.errstate(invalid="ignore"):
        out = (np.log10(x) - _CIRA_LOG_ROOT) / _CIRA_DENOM
    out = np.where(np.isnan(rgb), np.nan, out)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Step 7: day/night cross-fade to clean IR (GeoColor-lite)
# ---------------------------------------------------------------------------
def _clean_ir_gray(bt_kelvin: np.ndarray) -> np.ndarray:
    """Clean-IR brightness temperature -> grayscale 0..1 (cold cloud = white)."""
    x = (IR_T_WARM - bt_kelvin) / (IR_T_WARM - IR_T_COLD)
    return np.clip(x, 0.0, 1.0)


def night_blend(day_rgb: np.ndarray, ir_bt: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """Cross-fade true color (day) to grayscale clean-IR (night) across the
    terminator. The day weight is terminator_day_weight -- the SAME 88->95
    deg taper as the sun correction, so there is never a hard edge nor a
    doubly-darkened band."""
    w_day = terminator_day_weight(cos_sza)[..., None]
    night_gray = _clean_ir_gray(ir_bt)
    night_rgb = np.repeat(night_gray[..., None], 3, axis=-1)
    return np.clip(day_rgb * w_day + night_rgb * (1.0 - w_day), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Step 8: sun-glint tame (cosmetic, water-gated, conservative)
# ---------------------------------------------------------------------------
def glint_angle_field(sun_zen: np.ndarray, sat_zen: np.ndarray,
                      azidiff: np.ndarray) -> np.ndarray:
    """Angle (deg) between the satellite view ray and the sun's specular
    reflection off a flat surface: cos g = cos(sza)cos(vza) -
    sin(sza)sin(vza)cos(azidiff). 0 deg = dead-center glint (equal zeniths,
    opposite azimuths)."""
    sz = np.deg2rad(sun_zen)
    vz = np.deg2rad(sat_zen)
    ad = np.deg2rad(azidiff)
    cosg = np.cos(sz) * np.cos(vz) - np.sin(sz) * np.sin(vz) * np.cos(ad)
    return np.rad2deg(np.arccos(np.clip(cosg, -1.0, 1.0)))


def glint_tame(rgb: np.ndarray, glint_deg: np.ndarray,
               red_c: np.ndarray, veggie_c: "np.ndarray | None") -> np.ndarray:
    """Gently pull specular ocean glare toward its luminance inside the
    glint cone. Water gate: NDVI < 0 AND sun-corrected NIR below
    GLINT_WATER_NIR_MAX (liquid water absorbs 0.86 um) -- land, vegetation
    and cloud (bright NIR) are exact no-ops. Strength 0 disables bit-exactly.
    Skipped when no veggie band is available (cannot gate safely)."""
    if GLINT_TAME_STRENGTH <= 0.0 or veggie_c is None:
        return rgb
    cone = 1.0 - _smoothstep(GLINT_CONE_FULL_DEG, GLINT_CONE_ZERO_DEG, glint_deg)
    if not np.any(cone > 0.0):
        return rgb
    r = np.clip(red_c, 0.0, None)
    v = np.clip(veggie_c, 0.0, None)
    ndvi = (v - r) / np.maximum(v + r, 1e-6)
    water = (ndvi < 0.0) & (v < GLINT_WATER_NIR_MAX) \
        & np.isfinite(red_c) & np.isfinite(veggie_c)
    w = (GLINT_TAME_STRENGTH * cone * water.astype(rgb.dtype))[..., None]
    if not np.any(w > 0.0):
        return rgb
    wr, wg, wb = _GLINT_LUMA
    luma = (wr * rgb[..., 0] + wg * rgb[..., 1] + wb * rgb[..., 2])[..., None]
    desat = luma + (rgb - luma) * (1.0 - GLINT_DESAT)
    tamed = desat * (1.0 - GLINT_DARKEN)
    out = rgb * (1.0 - w) + tamed * w
    # exact no-op outside the gated cone (w==0 -> original pixel bit-for-bit)
    out = np.where(w > 0.0, out, rgb)
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


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

    ``green`` is None for ABI (synthesized); the sensor's REAL ~0.51 um green
    for AHI/AMI/FCI -- never the 0.86 um NIR, which rides in ``veggie``.
    ``ir_bt`` (clean-IR brightness temp, K, co-registered) enables the
    terminator IR cross-fade. All inputs are co-registered TOA reflectance
    (0..1, not sun-normalized) on the same (red-res) grid. Returns
    (rgb, cos_sza). The pipeline and every constant are shared across ALL
    ring sensors -- see the module docstring for the order and provenance.
    """
    sensor_l = sensor.lower()
    band_names = SENSOR_BANDS.get(sensor_l)
    if band_names is None:
        raise ValueError(f"unknown sensor {sensor!r} (expected one of {sorted(SENSOR_BANDS)})")
    fb_wl = FALLBACK_WL.get(sensor_l, FALLBACK_WL["abi"])

    cos_sza, sun_zen, sun_az = solar_geometry(lats, lons, when)
    sat_zen, sat_az = satellite_geometry(lats, lons, sub_sat_lon, when)
    azidiff = np.abs(sun_az - sat_az)
    azidiff = np.where(azidiff > 180.0, 360.0 - azidiff, azidiff)

    # 2) Sun-zenith normalize: 88-deg clamp + 95-deg taper on the visible
    #    bands; Li & Shibata effective pathlength for the NIR term.
    red_c = sunz_correct(red, cos_sza)
    blue_c = sunz_correct(blue, cos_sza)
    green_c = sunz_correct(green, cos_sza) if green is not None else None
    veggie_c = sunz_correct_pathlength(veggie, cos_sza) if veggie is not None else None

    # 3) Rayleigh-correct the REAL measured bands (blue, red, real green) --
    #    NOT the 0.86 um NIR. Band names give pyspectral the sensor-specific
    #    SRF effective wavelength; the sun-corrected red rides along in
    #    percent for the cloud relax (satpy passes it unclipped).
    if do_rayleigh:
        corrector = _make_rayleigh(platform_name, sensor_l)
        red_ref_pct = red_c * 100.0
        red_c = rayleigh_band(red_c, band_names["red"], sun_zen, sat_zen,
                              azidiff, red_ref_pct, corrector, fb_wl["red"])
        blue_c = rayleigh_band(blue_c, band_names["blue"], sun_zen, sat_zen,
                               azidiff, red_ref_pct, corrector, fb_wl["blue"])
        if green_c is not None:
            green_c = rayleigh_band(green_c, band_names["green"], sun_zen, sat_zen,
                                    azidiff, red_ref_pct, corrector, fb_wl["green"])

    # 4) Per-sensor green, all aimed at the same ~0.55 um target.
    green_out = make_green(sensor_l, red_c, blue_c, green_c, veggie_c)

    rgb = np.dstack([red_c, green_out, blue_c]).astype(np.float32, copy=False)
    rgb = np.clip(rgb, 0.0, None)      # Rayleigh negatives -> 0; NaN passes through

    # 5) Self-sharpen green/blue to the red band's finer native resolution.
    if do_ratio_sharpen:
        rgb = self_sharpen(rgb, sensor_l)

    # 6) The ONE shared tone curve.
    rgb = cira_stretch(rgb)

    # 8) Sun-glint tame (before the IR fade so night is untouched).
    rgb = glint_tame(rgb, glint_angle_field(sun_zen, sat_zen, azidiff),
                     red_c, veggie_c)

    # 7) Terminator cross-fade to clean IR (the taper's exact complement).
    if ir_bt is not None:
        rgb = night_blend(rgb, ir_bt, cos_sza)

    return rgb.astype(np.float32), cos_sza.astype(np.float32)
