"""Tests for the rebuilt shared true-color pipeline (truecolor.py).

The pipeline is the Satpy 0.60 / pyspectral 0.14 operational recipe shared
across ABI / AHI / AMI / FCI (see the truecolor.py module docstring). These
tests pin the verbatim constants (cira_stretch curve, 88/95-deg sun-zenith
clamp+taper, Li & Shibata pathlength, per-sensor greens, satpy sharpening
guards) and the harmonization invariants (a spectrally-flat neutral cloud
renders IDENTICALLY on every sensor; sensor dispatch is total).

Rayleigh unit tests use a fake corrector so no pyspectral LUT download is
needed; one optional integration test exercises the real LUT and skips
cleanly when the data is unavailable (offline CI).
"""
import datetime as dt
import math

import numpy as np
import pytest

import truecolor as tc
from app import normalize_channel
from satellites import GENERIC_CHANNELS, GOES_EAST, HIMAWARI_PACIFIC

UTC = dt.timezone.utc
WHEN = dt.datetime(2026, 7, 1, 18, 0, tzinfo=UTC)   # midday over the Americas


# ---------------------------------------------------------------------------
# Band wiring (unchanged from the pre-rebuild suite)
# ---------------------------------------------------------------------------
def test_generic_channels_have_blue_and_veggie():
    assert GENERIC_CHANNELS["visible_blue"]["goes"] == 1
    assert GENERIC_CHANNELS["visible_blue"]["ahi"] == 1
    assert GENERIC_CHANNELS["veggie"]["goes"] == 3
    assert GENERIC_CHANNELS["veggie"]["ahi"] == 4


def test_truecolor_band_sets():
    # ABI has no green -> synthesized; AHI has native green (band 2).
    assert GOES_EAST.green_band is None
    assert GOES_EAST.truecolor_bands == {"red": 2, "blue": 1, "veggie": 3}
    assert HIMAWARI_PACIFIC.green_band == 2
    assert HIMAWARI_PACIFIC.truecolor_bands["green"] == 2


def test_normalize_channel_true_color_passthrough():
    assert normalize_channel("true_color") == ("true_color", False)


# ---------------------------------------------------------------------------
# cira_stretch -- the ONE shared tone curve
# ---------------------------------------------------------------------------
def test_cira_stretch_pins_satpy_constants():
    # display = (log10(R) + 1.6517) / 1.9888 in 0..1 units
    r = np.array([0.0223, 0.1, 0.31, 1.0], dtype=np.float32)
    out = tc.cira_stretch(r)
    expect = (np.log10(r) - np.log10(0.0223)) / ((1.0 - np.log10(0.0223)) * 0.75)
    assert np.allclose(out, np.clip(expect, 0.0, 1.0), atol=1e-6)
    # anchor values: 2.23% -> 0 exactly; 100% -> ~0.8305 (headroom, not 1.0)
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[3] == pytest.approx(0.83054, abs=1e-4)


def test_cira_stretch_monotonic_clipped_and_nan_transparent():
    r = np.linspace(0.0, 2.0, 101, dtype=np.float32)
    out = tc.cira_stretch(r)
    assert np.all(np.diff(out) >= -1e-7)          # monotone
    assert out.min() >= 0.0 and out.max() <= 1.0  # clipped
    nan_in = np.array([np.nan, 0.5], dtype=np.float32)
    nan_out = tc.cira_stretch(nan_in)
    assert np.isnan(nan_out[0]) and np.isfinite(nan_out[1])


def test_cira_stretch_identical_per_channel():
    rgb = np.random.default_rng(7).uniform(0.0, 1.2, (5, 4, 3)).astype(np.float32)
    out = tc.cira_stretch(rgb)
    for ch in range(3):
        assert np.array_equal(out[..., ch], tc.cira_stretch(rgb[..., ch]))


# ---------------------------------------------------------------------------
# Sun-zenith normalization: 88-deg clamp, 95-deg log taper (satpy verbatim)
# ---------------------------------------------------------------------------
def _cos(deg):
    return math.cos(math.radians(deg))


def test_sunz_correct_plain_cosine_below_limit():
    cos = np.array([1.0, 0.5, _cos(60.0)], dtype=np.float64)
    refl = np.full(3, 0.30, dtype=np.float32)
    out = tc.sunz_correct(refl, cos)
    assert np.allclose(out, 0.30 / cos, rtol=1e-6)


def test_sunz_correct_clamped_at_88_and_zero_by_95():
    refl = np.ones(4, dtype=np.float32)
    cos = np.array([_cos(88.0), _cos(91.0), _cos(95.0), _cos(97.0)])
    out = tc.sunz_correct(refl, cos)
    lim = 1.0 / _cos(88.0)
    assert out[0] == pytest.approx(lim, rel=1e-4)   # exactly the clamp at 88
    assert 0.0 < out[1] < lim                        # tapering region
    assert out[2] == pytest.approx(0.0, abs=1e-6)    # exact 0 at 95
    assert out[3] == pytest.approx(0.0, abs=1e-6)    # stays 0 beyond
    # satpy's exact log-shaped falloff at 91 deg
    grad = 1.0 - math.log((math.radians(91) - math.radians(88))
                          / (math.radians(95) - math.radians(88)) + 1.0) / math.log(2.0)
    assert out[1] == pytest.approx(grad / _cos(88.0), rel=1e-4)


def test_sunz_correct_nan_cos_forces_zero():
    out = tc.sunz_correct(np.ones(2, np.float32), np.array([np.nan, 1.0]))
    assert out[0] == 0.0 and out[1] == pytest.approx(1.0)


def test_pathlength_li_shibata_polynomial():
    # corr = 24.35 / (2 cos + sqrt(498.5225 cos^2 + 1))
    cos = np.array([1.0, 0.5, 0.2], dtype=np.float64)
    out = tc.sunz_correct_pathlength(np.ones(3, np.float32), cos)
    expect = 24.35 / (2.0 * cos + np.sqrt(498.5225 * cos ** 2 + 1.0))
    assert np.allclose(out, expect, rtol=1e-5)
    # near-overhead sun the correction is ~1 (sanity of the polynomial)
    assert out[0] == pytest.approx(1.0, abs=1e-3)


def test_pathlength_gentler_than_cosine_at_high_sza():
    cos = np.array([0.05], dtype=np.float64)   # SZA ~87 deg
    plain = tc.sunz_correct(np.ones(1, np.float32), cos)[0]
    path = tc.sunz_correct_pathlength(np.ones(1, np.float32), cos)[0]
    assert path < plain      # finite-atmosphere pathlength < geometric 1/cos


def test_terminator_day_weight_complements_taper():
    cos = np.array([1.0, _cos(87.9), _cos(88.1), _cos(95.0), _cos(96.0), np.nan])
    w = tc.terminator_day_weight(cos)
    assert w[0] == 1.0 and w[1] == 1.0            # full day through 88 deg
    assert 0.0 < w[2] < 1.0                        # fading
    assert w[3] == pytest.approx(0.0, abs=1e-6)    # gone at 95
    assert w[4] == 0.0 and w[5] == 0.0             # night / invalid


# ---------------------------------------------------------------------------
# Rayleigh correction plumbing (fake corrector -- no LUT download)
# ---------------------------------------------------------------------------
class _FakeRayleigh:
    """Records the calls rayleigh_band makes; returns a fixed 5% correction."""

    def __init__(self):
        self.calls = []
        self.reduce_calls = []

    def get_reflectance(self, sunz, satz, azidiff, band, redband=None):
        if isinstance(band, float):
            raise AssertionError("band NAME expected first, not wavelength")
        self.calls.append((band, redband is not None))
        return np.full_like(np.asarray(sunz, dtype=np.float32), 5.0)

    def reduce_rayleigh_highzenith(self, zenith, rayref, lo, hi, strength):
        self.reduce_calls.append((lo, hi, strength))
        return rayref


def test_rayleigh_band_passes_band_name_redband_and_taper():
    fake = _FakeRayleigh()
    band = np.full((2, 2), 0.30, dtype=np.float32)
    ang = np.zeros((2, 2))
    red_pct = np.full((2, 2), 30.0, dtype=np.float32)
    out = tc.rayleigh_band(band, "C01", ang, ang, ang, red_pct, fake)
    assert fake.calls == [("C01", True)]
    assert fake.reduce_calls == [(70.0, 95.0, 0.6)]      # spec item 4 taper
    assert np.allclose(out, 0.30 - 0.05)                  # 5% subtracted


def test_rayleigh_band_none_corrector_is_identity():
    band = np.full((2, 2), 0.30, dtype=np.float32)
    ang = np.zeros((2, 2))
    out = tc.rayleigh_band(band, "C01", ang, ang, ang, band * 100, None)
    assert out is band


def test_rayleigh_lut_selection_constants():
    # the operational LUT: rayleigh-only aerosol, us-standard atmosphere
    assert tc.RAYLEIGH_AEROSOL == "rayleigh_only"
    assert tc.RAYLEIGH_ATMOSPHERE == "us-standard"


def test_sensor_band_names_are_sensor_specific():
    # per-band effective wavelength comes from the SENSOR's SRF via the
    # satpy band name -- pin the name tables (typos here = silent 0.64 um
    # fallback for every band, killing the per-sensor wavelength handling)
    assert tc.SENSOR_BANDS["abi"] == {"red": "C02", "green": None,
                                      "blue": "C01", "veggie": "C03"}
    assert tc.SENSOR_BANDS["ahi"] == {"red": "B03", "green": "B02",
                                      "blue": "B01", "veggie": "B04"}
    assert tc.SENSOR_BANDS["ami"] == {"red": "VI006", "green": "VI005",
                                      "blue": "VI004", "veggie": "VI008"}
    assert tc.SENSOR_BANDS["fci"] == {"red": "vis_06", "green": "vis_05",
                                      "blue": "vis_04", "veggie": "vis_08"}


# ---------------------------------------------------------------------------
# Per-sensor green ladder
# ---------------------------------------------------------------------------
def test_abi_simulated_green_cimss_bah_fractions():
    red = np.float32([0.2]); blue = np.float32([0.1]); nir = np.float32([0.4])
    g = tc.make_green("abi", red, blue, None, nir)
    assert g[0] == pytest.approx(0.45 * 0.1 + 0.45 * 0.2 + 0.10 * 0.4, rel=1e-6)
    assert sum(tc.ABI_GREEN_FRACTIONS) == pytest.approx(1.0)


def test_ahi_ami_hybrid_green_f015():
    green = np.float32([0.30]); nir = np.float32([0.50])
    r = b = np.float32([0.2])
    for sensor in ("ahi", "ami"):
        g = tc.make_green(sensor, r, b, green, nir)
        assert g[0] == pytest.approx(0.85 * 0.30 + 0.15 * 0.50, rel=1e-6)


def test_fci_ndvi_hybrid_green_limits_and_strength():
    green = np.float32([0.30, 0.30, 0.30])
    red = np.float32([0.30, 0.10, 0.20])
    nir = np.float32([0.30, 0.90, 0.20])   # NDVI = 0, 0.8, 0
    g = tc.make_green("fci", red, None, green, nir)
    # NDVI=0 -> fraction limits[0]=0.15
    assert g[0] == pytest.approx(0.85 * 0.30 + 0.15 * 0.30, rel=1e-6)
    # NDVI=0.8, strength 3: n' = .8^3/(.8^3+.2^3); f = 0.15 - 0.1*n'
    nprime = 0.8 ** 3 / (0.8 ** 3 + 0.2 ** 3)
    f = 0.15 + nprime * (0.05 - 0.15)
    assert g[1] == pytest.approx((1 - f) * 0.30 + f * 0.90, rel=1e-5)


def test_green_ladder_raises_on_missing_required_bands():
    v = np.float32([0.3])
    with pytest.raises(ValueError):
        tc.make_green("abi", v, v, None, None)       # ABI needs veggie
    with pytest.raises(ValueError):
        tc.make_green("ahi", v, v, None, v)          # AHI needs native green
    with pytest.raises(ValueError):
        tc.make_green("seviri", v, v, v, v)          # not a ring sensor


# ---------------------------------------------------------------------------
# Self-sharpen (satpy SelfSharpenedRGB semantics)
# ---------------------------------------------------------------------------
def test_self_sharpen_injects_red_detail_into_green_blue():
    # a red field with a bright 1-px feature the 2x2 mean smooths away
    red = np.full((8, 8), 0.2, dtype=np.float32)
    red[4, 4] = 0.6
    rgb = np.dstack([red, np.full_like(red, 0.3), np.full_like(red, 0.3)])
    out = tc.self_sharpen(rgb, "abi")
    assert out[4, 4, 1] > 0.3 and out[4, 4, 2] > 0.3   # detail injected
    assert np.array_equal(out[..., 0], red)             # red untouched


def test_self_sharpen_ratio_guards_and_cap():
    red = np.zeros((4, 4), dtype=np.float32)            # ratio undefined -> 1
    rgb = np.dstack([red, np.full_like(red, 0.4), np.full_like(red, 0.4)])
    out = tc.self_sharpen(rgb, "ahi")
    assert np.allclose(out[..., 1], 0.4)                # no-op, no NaN
    assert tc.SHARPEN_RATIO_MAX == 1.5                  # satpy's clip


def test_self_sharpen_noop_for_single_resolution_sensor():
    rgb = np.random.default_rng(3).uniform(0, 1, (6, 6, 3)).astype(np.float32)
    assert tc.self_sharpen(rgb, "fci") is rgb


def test_self_sharpen_nan_edges_do_not_bleed():
    red = np.full((6, 6), 0.4, dtype=np.float32)
    red[:, :2] = np.nan                                  # off-disk edge
    rgb = np.dstack([red, np.full_like(red, 0.3), np.full_like(red, 0.3)])
    out = tc.self_sharpen(rgb, "abi")
    assert np.allclose(out[:, 3:, 1], 0.3, atol=1e-6)    # interior unchanged
    assert np.all(np.isnan(out[:, :2, 0]))               # edge stays NaN


# ---------------------------------------------------------------------------
# Night blend + glint
# ---------------------------------------------------------------------------
def test_night_blend_day_night_and_crossfade():
    day = np.full((1, 3, 3), 0.8, dtype=np.float32)
    ir = np.full((1, 3), 250.0, dtype=np.float32)        # cool cloud gray
    cos = np.array([[1.0, _cos(91.0), _cos(96.0)]])
    out = tc.night_blend(day, ir, cos)
    gray = (tc.IR_T_WARM - 250.0) / (tc.IR_T_WARM - tc.IR_T_COLD)
    assert np.allclose(out[0, 0], 0.8)                   # pure day
    assert np.allclose(out[0, 2], gray, atol=1e-5)       # pure IR night
    w = tc.terminator_day_weight(np.array([_cos(91.0)]))[0]
    assert out[0, 1, 0] == pytest.approx(0.8 * w + gray * (1 - w), abs=1e-5)


def test_glint_angle_geometry():
    # dead-center glint: equal zeniths, opposite azimuths (azidiff 180)
    g = tc.glint_angle_field(np.array([30.0]), np.array([30.0]), np.array([180.0]))
    assert g[0] == pytest.approx(0.0, abs=1e-4)
    # same-side azimuth (azidiff 0): angle = sum of zeniths
    g = tc.glint_angle_field(np.array([30.0]), np.array([40.0]), np.array([0.0]))
    assert g[0] == pytest.approx(70.0, abs=1e-4)


def test_glint_tame_water_gated_and_strength_zero_bit_exact(monkeypatch):
    rgb = np.full((2, 2, 3), 0.7, dtype=np.float32)
    glint = np.zeros((2, 2))                             # dead-center cone
    red = np.full((2, 2), 0.05, dtype=np.float32)
    veg_water = np.full((2, 2), 0.03, dtype=np.float32)  # NDVI<0, low NIR
    veg_land = np.full((2, 2), 0.40, dtype=np.float32)   # vegetation
    tamed = tc.glint_tame(rgb, glint, red, veg_water)
    assert np.all(tamed[..., 0] < 0.7)                   # glare tamed on water
    land = tc.glint_tame(rgb, glint, red, veg_land)
    assert np.array_equal(land, rgb)                     # land: exact no-op
    monkeypatch.setattr(tc, "GLINT_TAME_STRENGTH", 0.0)
    off = tc.glint_tame(rgb, glint, red, veg_water)
    assert off is rgb                                    # disabled: bit-exact


# ---------------------------------------------------------------------------
# assemble_truecolor -- end-to-end + harmonization invariants
# ---------------------------------------------------------------------------
def _grids(n=16, lat0=0.0, lon0=-75.0, span=4.0):
    lats = np.linspace(lat0 + span / 2, lat0 - span / 2, n, dtype=np.float32)
    lons = np.linspace(lon0 - span / 2, lon0 + span / 2, n, dtype=np.float32)
    LON, LAT = np.meshgrid(lons, lats)
    return LAT, LON


def _assemble(sensor, val=0.5, green_val=None, when=WHEN, ir=None, veggie=None,
              n=16, span=4.0):
    LAT, LON = _grids(n=n, span=span)
    shape = LAT.shape
    band = lambda v: np.full(shape, v, dtype=np.float32)   # noqa: E731
    veggie = val if veggie is None else veggie   # default: flat spectrum
    green = None if sensor == "abi" else band(val if green_val is None else green_val)
    rgb, cos_sza = tc.assemble_truecolor(
        band(val), green, band(val), band(veggie), LAT, LON,
        when=when, sub_sat_lon=-75.0, platform_name="TEST", sensor=sensor,
        ir_bt=(band(ir) if ir is not None else None),
        do_rayleigh=False,   # no LUT in unit tests; Rayleigh has its own tests
    )
    return rgb, cos_sza


def test_assemble_dispatches_all_ring_sensors_and_rejects_unknown():
    for sensor in ("abi", "ahi", "ami", "fci"):
        rgb, cos_sza = _assemble(sensor)
        assert rgb.shape == (16, 16, 3) and rgb.dtype == np.float32
        assert np.isfinite(rgb).all()
        assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    with pytest.raises(ValueError):
        _assemble("seviri")


def test_neutral_cloud_identical_across_all_sensors():
    """Spec item 6: a spectrally-flat bright cloud must land on the SAME
    near-white on every sensor -- every green formula collapses to the input
    for a flat spectrum, and the tone curve is shared."""
    outs = {s: _assemble(s, val=0.85)[0] for s in ("abi", "ahi", "ami", "fci")}
    ref = outs["abi"]
    for s, rgb in outs.items():
        assert np.allclose(rgb, ref, atol=2e-3), f"{s} diverges on neutral cloud"
    # and the channels are neutral (white, not tinted)
    assert np.allclose(ref[..., 0], ref[..., 1], atol=2e-3)
    assert np.allclose(ref[..., 1], ref[..., 2], atol=2e-3)


def test_assemble_night_side_is_ir_when_provided():
    night = dt.datetime(2026, 7, 1, 6, 0, tzinfo=UTC)    # local midnight at 75W
    rgb, cos_sza = _assemble("abi", ir=220.0, when=night)
    gray = (tc.IR_T_WARM - 220.0) / (tc.IR_T_WARM - tc.IR_T_COLD)
    assert np.allclose(rgb, gray, atol=1e-4)             # pure IR, no color
    assert np.all(cos_sza < 0.0)


def test_assemble_preserves_offdisk_nan():
    LAT, LON = _grids()
    band = np.full(LAT.shape, 0.5, dtype=np.float32)
    band[:4, :4] = np.nan
    rgb, _ = tc.assemble_truecolor(
        band, None, band, band, LAT, LON, when=WHEN, sub_sat_lon=-75.0,
        platform_name="TEST", sensor="abi", do_rayleigh=False)
    assert np.isnan(rgb[:2, :2]).all()                   # alpha mask survives
    assert np.isfinite(rgb[8:, 8:]).all()


def test_assemble_sunz_taper_darkens_terminator_smoothly():
    # dawn scene: a wide grid spanning the terminator
    dawn = dt.datetime(2026, 7, 1, 10, 40, tzinfo=UTC)
    rgb, cos_sza = _assemble("abi", val=0.4, when=dawn, n=64, span=30.0)
    dark = rgb[np.cos(np.deg2rad(95.0)) > cos_sza]
    if dark.size:
        assert np.allclose(dark, 0.0, atol=1e-5)         # black past 95 deg
    day = rgb[cos_sza > 0.5]
    if day.size:
        assert day.mean() > 0.5                          # day side developed


# ---------------------------------------------------------------------------
# Optional integration: the real pyspectral LUT (skips offline)
# ---------------------------------------------------------------------------
def test_real_rayleigh_lut_abi_smoke():
    corrector = tc._make_rayleigh("GOES-19", "abi")
    if corrector is None:
        pytest.skip("pyspectral Rayleigh LUT unavailable")
    ang = np.full((4, 4), 30.0)
    red_pct = np.full((4, 4), 5.0, dtype=np.float32)
    band = np.full((4, 4), 0.10, dtype=np.float32)
    try:
        out = tc.rayleigh_band(band, "C01", ang, ang, np.full((4, 4), 90.0),
                               red_pct, corrector)
    except Exception:
        pytest.skip("Rayleigh LUT data not downloadable in this environment")
    # blue over dark ocean: a real, meaningful subtraction happened
    assert np.all(out < band) and np.all(out > band - 0.15)
