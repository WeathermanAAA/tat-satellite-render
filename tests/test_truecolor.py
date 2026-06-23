"""Unit tests for the true-color recipe math (no network).

Covers the pieces that are easy to get subtly wrong: the synthetic-green mix,
sun-zenith normalization, the tone curve's monotonicity/endpoints, and the
GeoColor-lite day/night blend selecting the right layer on each side of the
terminator. The fetch/projection paths are exercised by live smoke tests, not
here.
"""
import numpy as np
import pytest

import truecolor as tc
from app import normalize_channel
from satellites import GENERIC_CHANNELS, GOES_EAST, HIMAWARI_PACIFIC


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


def test_synth_green_fractions_sum_to_one():
    assert abs(sum(tc.GREEN_FRACTIONS) - 1.0) < 1e-9


def test_synth_green_uses_ahi_derived_model():
    # The AHI-derived linear green is active (asset loaded): G = c0 + cB*blue +
    # cR*red + cV*veggie, dominated by BLUE (the 0.51um green band sits right next
    # to 0.47um blue) -- NOT the old fixed CIMSS mix that over-greened at low sun.
    assert tc._GREEN_AHI is not None
    c0, cB, cR, cV = tc._GREEN_AHI
    assert cB > cR and cB > cV                    # green is mostly blue
    red = np.array([0.3], np.float32)
    veg = np.array([0.5], np.float32)
    blue = np.array([0.1], np.float32)
    g = tc.synth_green(red, veg, blue)
    assert np.isclose(g[0], c0 + cB * 0.1 + cR * 0.3 + cV * 0.5, atol=1e-5)
    # and it genuinely differs from the legacy fixed mix (the whole point)
    fr, fv, fb = tc.GREEN_FRACTIONS
    assert not np.isclose(g[0], fr * 0.3 + fv * 0.5 + fb * 0.1, atol=1e-3)


def test_synth_green_falls_back_to_linear_mix(monkeypatch):
    # If the trained asset is unavailable, synth_green uses the legacy fixed mix
    # (rendering must never break on a missing asset).
    monkeypatch.setattr(tc, "_GREEN_AHI", None)
    red = np.array([0.3], np.float32)
    veg = np.array([0.5], np.float32)
    blue = np.array([0.1], np.float32)
    fr, fv, fb = tc.GREEN_FRACTIONS
    assert np.isclose(tc.synth_green(red, veg, blue)[0], fr * 0.3 + fv * 0.5 + fb * 0.1)


def test_sun_correct_floor():
    # Below the cos(SZA) floor we divide by the floor, not the tiny value,
    # so the terminator doesn't explode.
    refl = np.array([0.1], np.float32)
    out = tc.sun_correct(refl, np.array([1e-4], np.float32))
    assert np.isclose(out[0], 0.1 / tc.COS_SZA_FLOOR)


def test_tone_curve_monotonic_and_endpoints():
    x = np.linspace(0, 1, 256).astype(np.float32)
    y = tc.tone_curve(x)
    assert np.isclose(y[0], 0.0, atol=1e-3)
    assert np.isclose(y[-1], 1.0, atol=1e-3)
    assert np.all(np.diff(y) >= -1e-6)  # non-decreasing


def test_night_blend_picks_day_and_night_layers():
    day = np.ones((1, 2, 3), np.float32)          # bright "true color"
    ir_bt = np.full((1, 2), 200.0, np.float32)    # cold cloud -> near-white IR
    cos_sza = np.array([[0.5, -0.5]], np.float32)  # left day, right night
    out = tc.night_blend(day, ir_bt, cos_sza)
    # Day pixel keeps the true-color (white) layer.
    assert np.allclose(out[0, 0], 1.0, atol=1e-3)
    # Night pixel comes from the IR layer (grayscale, all 3 channels equal).
    assert np.isclose(out[0, 1, 0], out[0, 1, 1]) and np.isclose(out[0, 1, 1], out[0, 1, 2])


def test_ratio_sharpen_injects_red_detail():
    rgb = np.full((2, 2, 3), 0.4, np.float32)
    red_hi = np.array([[0.4, 0.8], [0.4, 0.2]], np.float32)
    out = tc.ratio_sharpen(rgb, red_hi)
    # Red channel becomes the hi-res red exactly.
    assert np.allclose(out[..., 0], red_hi)
    # Where red is brighter than the composite base, G/B are lifted.
    assert out[0, 1, 1] > rgb[0, 1, 1]


# --- Terminator (sunrise/sunset) golden-hour handling ----------------------

def test_rayleigh_scale_field_full_across_dayside_floor_at_horizon():
    cos = np.array([[1.0, 0.6, tc.RAYLEIGH_TAPER_START_COS, 0.05,
                     0.0, -0.1]], np.float64)
    s = tc.rayleigh_scale_field(cos)
    # EXACTLY full (==RAYLEIGH_SCALE) for cos_sza >= the START threshold (the
    # whole day side, well into twilight) -- this is what keeps midday byte-
    # identical AND prevents the broad warm bleed.
    assert s[0, 0] == tc.RAYLEIGH_SCALE
    assert s[0, 1] == tc.RAYLEIGH_SCALE
    assert s[0, 2] == tc.RAYLEIGH_SCALE       # at START_COS -> still full
    # Softens only in the last few degrees, and only toward the FLOOR (never 0).
    assert tc.RAYLEIGH_FLOOR < s[0, 3] < tc.RAYLEIGH_SCALE   # 0.05 -> partial
    assert abs(s[0, 4] - tc.RAYLEIGH_FLOOR) < 1e-9           # horizon -> floor
    assert abs(s[0, 5] - tc.RAYLEIGH_FLOOR) < 1e-9           # below -> floor
    assert s[0, 4] > 0.0                                     # NEVER zeroed
    # cos descends across the array, so the strength is non-increasing as the
    # sun sets toward the horizon.
    assert np.all(np.diff(s[0]) <= 1e-12)
    # the gate stays below full daylight
    assert tc.WARM_TINT_DAY_COS < tc.COS_SZA_FULL_DAY
    assert tc.RAYLEIGH_TAPER_START_COS < tc.COS_SZA_FULL_DAY


def test_warm_tint_disabled_by_default_natural_rgb():
    # Default contract: the warm tint is OFF, so sunrise/sunset reads as natural
    # true-color RGB -- the function is a no-op even AT the terminator.
    assert tc.WARM_TINT_STRENGTH == 0.0
    rgb = np.full((1, 1, 3), 0.5, np.float32)
    cos_term = np.array([[tc.WARM_TINT_PEAK_COS]], np.float32)   # the horizon
    assert np.array_equal(tc.warm_terminator_tint(rgb, cos_term), rgb)
    cos_day = np.full((1, 1), 0.8, np.float32)
    assert np.array_equal(tc.warm_terminator_tint(rgb, cos_day), rgb)


def test_warm_tint_when_enabled_is_gated_and_warms():
    # The knob is retained: if re-enabled, the tint stays daytime-gated (exact
    # no-op for cos_sza >= WARM_TINT_DAY_COS) and warms the terminator correctly
    # (R up, G/B down, blue down most -> orange).
    rgb = np.full((1, 1, 3), 0.5, np.float32)
    saved = tc.WARM_TINT_STRENGTH
    try:
        tc.WARM_TINT_STRENGTH = 1.0
        day = tc.warm_terminator_tint(rgb, np.full((1, 1), 0.8, np.float32))
        assert np.array_equal(day, rgb)   # daytime still untouched
        out = tc.warm_terminator_tint(rgb, np.array([[tc.WARM_TINT_PEAK_COS]], np.float32))
    finally:
        tc.WARM_TINT_STRENGTH = saved
    assert out[0, 0, 0] > 0.5 and out[0, 0, 1] < 0.5 and out[0, 0, 2] < 0.5
    assert out[0, 0, 2] < out[0, 0, 1]   # blue down MORE than green


def _midday_scene():
    """A small, solidly-daytime ABI scene (cos_sza well above the gate
    thresholds) -- for the midday byte-identity tests."""
    import datetime as dt
    H = W = 6
    rng = np.random.default_rng(0)
    red = rng.uniform(0.1, 0.6, (H, W)).astype(np.float32)
    blue = rng.uniform(0.1, 0.6, (H, W)).astype(np.float32)
    veggie = rng.uniform(0.1, 0.6, (H, W)).astype(np.float32)
    when = dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)  # noon @ lon 0
    lons, lats = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H))
    lats = lats.astype(np.float32); lons = lons.astype(np.float32)
    cos_sza, *_ = tc.solar_geometry(lats, lons, when)
    assert float(cos_sza.min()) >= tc.WARM_TINT_DAY_COS  # really all-daytime
    return red, blue, veggie, lats, lons, when


def test_assemble_truecolor_midday_warm_tint_is_noop():
    """A fully-daytime scene is byte-identical with the WARM TINT in place: at
    high cos_sza the tint weight is exactly 0, so the only post-tone-curve op
    added is a no-op. (Rayleigh off here -> no pyspectral dep; the taper no-op is
    covered by the do_rayleigh=True test below.)"""
    red, blue, veggie, lats, lons, when = _midday_scene()
    common = dict(when=when, sub_sat_lon=0.0, platform_name="GOES-19",
                  sensor="abi", do_rayleigh=False)
    rgb_new, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    saved = tc.WARM_TINT_STRENGTH
    try:
        tc.WARM_TINT_STRENGTH = 0.0
        rgb_off, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    finally:
        tc.WARM_TINT_STRENGTH = saved
    assert np.array_equal(rgb_new, rgb_off)


def test_assemble_truecolor_vegetation_green_ocean_blue():
    """End-to-end guard for the AHI-derived green: through the FULL assemble path
    (sun-correct -> synth green -> ratio-sharpen -> tone curve -> night blend), a
    vegetation pixel must stay GREEN-dominant (G>=R and G>=B) and an ocean pixel
    must stay BLUE-dominant (B>=G). This is the regression that pins the learned
    green model -- an ocean-only fit (or too-small veggie weight) renders forest
    yellow-brown; the matched land+ocean fit keeps both surfaces right. Mirrors
    the real GOES Amazon (G-R>0) + ocean (B>G) renders. (Rayleigh off -> no
    pyspectral dep; sun-correct/tone-curve/sharpen are the parts that matter.)"""
    import datetime as dt
    H = W = 4
    red = np.full((H, W), 0.05, np.float32)
    blue = np.full((H, W), 0.04, np.float32)
    veggie = np.full((H, W), 0.35, np.float32)
    red[0, 0], blue[0, 0], veggie[0, 0] = 0.035, 0.030, 0.45   # vegetation: chlorophyll absorbs red/blue, high NIR
    red[0, 1], blue[0, 1], veggie[0, 1] = 0.025, 0.090, 0.015  # ocean: blue-favored, ~zero NIR
    when = dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
    lons, lats = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H))
    lats = lats.astype(np.float32); lons = lons.astype(np.float32)
    assert tc._GREEN_AHI is not None, "AHI-derived green asset must be present for this guard"
    rgb, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, when=when,
                                   sub_sat_lon=0.0, platform_name="GOES-19",
                                   sensor="abi", do_rayleigh=False)
    veg, ocean = rgb[0, 0], rgb[0, 1]
    assert veg[1] >= veg[0] and veg[1] >= veg[2], f"vegetation not green-dominant: {veg}"
    assert ocean[2] >= ocean[1], f"ocean not blue-dominant: {ocean}"


def _have_pyspectral():
    try:
        import pyspectral  # noqa: F401
        return tc._make_rayleigh("GOES-19", "abi") is not None
    except Exception:
        return False


@pytest.mark.skipif(not _have_pyspectral(), reason="pyspectral Rayleigh unavailable")
def test_assemble_truecolor_midday_byte_identical_full_recipe():
    """The REAL production path (do_rayleigh=True): a fully-daytime scene is
    byte-identical with BOTH terminator hooks in place -- the Rayleigh taper is
    exactly RAYLEIGH_SCALE (so the per-pixel array scale == the old scalar 1.0)
    AND the warm tint is exactly 0. Compares the live recipe to the same pipeline
    with both terminator hooks neutralized to their pre-change form. (The
    land-aware Rayleigh relax is active in BOTH calls -- it is deterministic per
    input, so it cancels and does not affect this midday-taper identity.)"""
    red, blue, veggie, lats, lons, when = _midday_scene()
    common = dict(when=when, sub_sat_lon=0.0, platform_name="GOES-19",
                  sensor="abi", do_rayleigh=True)
    rgb_new, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    saved_strength, saved_field = tc.WARM_TINT_STRENGTH, tc.rayleigh_scale_field
    try:
        tc.WARM_TINT_STRENGTH = 0.0
        # old behavior: full Rayleigh everywhere (scalar RAYLEIGH_SCALE)
        tc.rayleigh_scale_field = lambda c: np.full_like(c, tc.RAYLEIGH_SCALE)
        rgb_old, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    finally:
        tc.WARM_TINT_STRENGTH, tc.rayleigh_scale_field = saved_strength, saved_field
    assert np.array_equal(rgb_new, rgb_old)


# --- Land-aware Rayleigh relax (vegetation de-browning) --------------------

def test_land_rayleigh_relax_field_gate():
    """The NDVI gate: EXACTLY 1.0 (no relax) over ocean (NDVI<0) and cloud
    (NDVI~0), dropping to 1-LAND_RAYLEIGH_RELAX over dense vegetation, with a
    monotone ramp for sparse veg in between."""
    #          ocean  cloud  sparse  dense-veg
    red = np.array([[0.05, 0.30, 0.05, 0.04]], np.float32)
    veg = np.array([[0.01, 0.32, 0.07, 0.45]], np.float32)
    f = tc.land_rayleigh_relax_field(red, veg)
    assert f[0, 0] == 1.0                                   # ocean: NDVI<0 -> no relax
    assert f[0, 1] == 1.0                                   # cloud: NDVI<=LO -> no relax
    assert abs(f[0, 3] - (1.0 - tc.LAND_RAYLEIGH_RELAX)) < 1e-6   # dense veg -> full relax
    assert (1.0 - tc.LAND_RAYLEIGH_RELAX) <= f[0, 2] <= 1.0       # sparse: partial
    # no veggie band -> all ones (exact no-op)
    assert np.array_equal(tc.land_rayleigh_relax_field(red, None), np.ones_like(red))
    # disabled (strength 0) -> all ones even over vegetation
    saved = tc.LAND_RAYLEIGH_RELAX
    try:
        tc.LAND_RAYLEIGH_RELAX = 0.0
        assert np.array_equal(tc.land_rayleigh_relax_field(red, veg), np.ones_like(red))
    finally:
        tc.LAND_RAYLEIGH_RELAX = saved


@pytest.mark.skipif(not _have_pyspectral(), reason="pyspectral Rayleigh unavailable")
def test_assemble_land_relax_ocean_cloud_byte_identical():
    """The de-browning MUST NOT touch ocean or cloud: a do_rayleigh=True midday
    OCEAN+CLOUD scene is byte-identical with the land relax on vs off, because the
    NDVI gate is exactly 0 there (deep-blue ocean / white cloud preserved)."""
    import datetime as dt
    H, W = 1, 4
    #             ocean ocean  cloud cloud   (NIR<=red everywhere -> NDVI<=0)
    red = np.array([[0.02, 0.03, 0.85, 0.80]], np.float32)
    blue = np.array([[0.09, 0.10, 0.85, 0.82]], np.float32)
    veggie = np.array([[0.01, 0.015, 0.84, 0.83]], np.float32)
    when = dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
    lons, lats = np.meshgrid(np.linspace(-2, 2, W), np.linspace(0, 0, H))
    lats = lats.astype(np.float32); lons = lons.astype(np.float32)
    common = dict(when=when, sub_sat_lon=0.0, platform_name="GOES-19",
                  sensor="abi", do_rayleigh=True)
    on, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    saved = tc.LAND_RAYLEIGH_RELAX
    try:
        tc.LAND_RAYLEIGH_RELAX = 0.0   # old behavior: full Rayleigh everywhere
        off, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    finally:
        tc.LAND_RAYLEIGH_RELAX = saved
    assert np.array_equal(on, off)


@pytest.mark.skipif(not _have_pyspectral(), reason="pyspectral Rayleigh unavailable")
def test_assemble_land_relax_debrowns_vegetation():
    """The regression the relax exists for: do_rayleigh=True over vegetation
    (high NDVI). Full Rayleigh over-subtracts the blue path radiance and the
    pixel turns red-dominant (brown); the land relax keeps that blue so the pixel
    is LESS red-dominant. A co-located ocean pixel (NDVI<0) stays byte-identical."""
    import datetime as dt
    H, W = 1, 2
    #            vegetation     ocean
    red = np.array([[0.07, 0.02]], np.float32)   # veg: red absorbed; ocean dark
    blue = np.array([[0.12, 0.09]], np.float32)  # veg: blue-heavy TOA haze
    veggie = np.array([[0.45, 0.01]], np.float32)  # veg: high NIR; ocean ~zero NIR
    when = dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
    lons, lats = np.meshgrid(np.linspace(-1, 1, W), np.linspace(0, 0, H))
    lats = lats.astype(np.float32); lons = lons.astype(np.float32)
    common = dict(when=when, sub_sat_lon=0.0, platform_name="GOES-19",
                  sensor="abi", do_rayleigh=True)
    on, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    saved = tc.LAND_RAYLEIGH_RELAX
    try:
        tc.LAND_RAYLEIGH_RELAX = 0.0   # old behavior: full Rayleigh everywhere
        off, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    finally:
        tc.LAND_RAYLEIGH_RELAX = saved
    veg_on, veg_off = on[0, 0], off[0, 0]
    # the relax preserves blue the full correction would have stripped...
    assert veg_on[2] > veg_off[2], f"blue not preserved: {veg_off} -> {veg_on}"
    # ...so the vegetation pixel is less red-dominant (less brown)
    assert (veg_on[0] - veg_on[2]) < (veg_off[0] - veg_off[2])
    # the ocean pixel (gate 0) is untouched
    assert np.array_equal(on[0, 1], off[0, 1])


def test_land_relax_nir_floor_protects_dark_and_glint_water():
    """The absolute NIR floor closes the gate where the NDVI gate alone would not:
    near-black clear water (tiny red AND tiny NIR -> spuriously HIGH NDVI) and sun
    glint (high NIR but spectrally flat). Both must yield factor 1.0 (no relax),
    while genuine well-lit vegetation still relaxes."""
    #              dark-water  glint       dense-veg
    red = np.array([[0.004, 0.50, 0.04]], np.float32)
    veg = np.array([[0.012, 0.50, 0.45]], np.float32)
    f = tc.land_rayleigh_relax_field(red, veg)
    # dark water: NDVI=(.012-.004)/.016=0.5 (high!) but NIR 0.012 < LAND_NIR_LO -> gated off
    assert f[0, 0] == 1.0, f"near-black water relaxed: NDVI high but NIR tiny -> {f[0,0]}"
    # glint: NIR high but NDVI~0 (flat spectrum) -> gated off
    assert f[0, 1] == 1.0, f"glint relaxed: {f[0,1]}"
    # real vegetation (high NDVI AND high NIR) still relaxes
    assert abs(f[0, 2] - (1.0 - tc.LAND_RAYLEIGH_RELAX)) < 1e-6


def test_land_rayleigh_relax_field_nan_safe():
    """NaN inputs (partial-tile edges) map to factor 1.0 -> the relax never
    poisons an otherwise-valid pixel; dtype is preserved (float32 -> ocean
    multiply by exactly 1.0 stays bit-identical)."""
    red = np.array([[np.nan, 0.04, 0.02]], np.float32)
    veg = np.array([[0.45, np.nan, 0.01]], np.float32)
    f = tc.land_rayleigh_relax_field(red, veg)
    assert f.dtype == np.float32
    assert f[0, 0] == 1.0 and f[0, 1] == 1.0   # any NaN input -> identity
    assert not np.any(np.isnan(f))


@pytest.mark.skipif(not _have_pyspectral(), reason="pyspectral Rayleigh unavailable")
def test_assemble_land_relax_glint_and_dark_water_byte_identical():
    """do_rayleigh=True midday scene of SUN GLINT + NEAR-BLACK water is byte-
    identical with the land relax on vs off (the NIR floor / flat-spectrum NDVI
    keep the gate at 0 there) -- the failure mode the robustness review flagged."""
    import datetime as dt
    H, W = 1, 3
    #             glint  dark-water near-black
    red = np.array([[0.55, 0.004, 0.010]], np.float32)
    blue = np.array([[0.55, 0.090, 0.060]], np.float32)
    veggie = np.array([[0.55, 0.012, 0.020]], np.float32)
    when = dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
    lons, lats = np.meshgrid(np.linspace(-2, 2, W), np.linspace(0, 0, H))
    lats = lats.astype(np.float32); lons = lons.astype(np.float32)
    common = dict(when=when, sub_sat_lon=0.0, platform_name="GOES-19",
                  sensor="abi", do_rayleigh=True)
    on, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    saved = tc.LAND_RAYLEIGH_RELAX
    try:
        tc.LAND_RAYLEIGH_RELAX = 0.0
        off, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    finally:
        tc.LAND_RAYLEIGH_RELAX = saved
    assert np.array_equal(on, off)


@pytest.mark.skipif(not _have_pyspectral(), reason="pyspectral Rayleigh unavailable")
def test_assemble_land_relax_sparse_vegetation_partial_debrown():
    """do_rayleigh=True over SPARSE vegetation (mid NDVI, real NIR): the relax
    still fires (blue preserved -> less brown), end-to-end through assemble."""
    import datetime as dt
    H, W = 1, 1
    red = np.array([[0.06]], np.float32)     # mid NDVI: (0.12-0.06)/0.18 = 0.33
    blue = np.array([[0.11]], np.float32)
    veggie = np.array([[0.12]], np.float32)  # NIR above the floor -> guard open
    when = dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
    lons, lats = np.meshgrid(np.linspace(0, 0, W), np.linspace(0, 0, H))
    lats = lats.astype(np.float32); lons = lons.astype(np.float32)
    common = dict(when=when, sub_sat_lon=0.0, platform_name="GOES-19",
                  sensor="abi", do_rayleigh=True)
    on, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    saved = tc.LAND_RAYLEIGH_RELAX
    try:
        tc.LAND_RAYLEIGH_RELAX = 0.0
        off, _ = tc.assemble_truecolor(red, None, blue, veggie, lats, lons, **common)
    finally:
        tc.LAND_RAYLEIGH_RELAX = saved
    assert on[0, 0, 2] > off[0, 0, 2]   # sparse veg also keeps some blue (de-browns)


# --- Terminator highlight rolloff (cold-cloud-top exposure) ----------------

def test_highlight_rolloff_field_zero_in_day_full_at_terminator():
    cos = np.array([[1.0, tc.HIGHLIGHT_DAY_COS, 0.25, tc.HIGHLIGHT_PEAK_COS,
                     0.0, -0.2]], np.float64)
    s = tc.highlight_rolloff_field(cos)
    # EXACTLY 0 at/above HIGHLIGHT_DAY_COS -> the daytime knee is a no-op.
    assert s[0, 0] == 0.0
    assert s[0, 1] == 0.0
    # full strength at/below the terminator threshold (and below the horizon).
    assert abs(s[0, 3] - tc.HIGHLIGHT_STRENGTH) < 1e-12   # at PEAK_COS
    assert abs(s[0, 4] - tc.HIGHLIGHT_STRENGTH) < 1e-12   # horizon
    assert abs(s[0, 5] - tc.HIGHLIGHT_STRENGTH) < 1e-12   # below horizon
    # partial in between, and non-increasing as the sun rises (cos increases).
    assert 0.0 < s[0, 2] < tc.HIGHLIGHT_STRENGTH
    assert np.all(np.diff(s[0]) >= -1e-12)                # cos descends -> s ascends
    # the knee gate lives below full daylight (never fires at noon).
    assert tc.HIGHLIGHT_DAY_COS <= tc.COS_SZA_FULL_DAY or tc.HIGHLIGHT_DAY_COS == 0.30


def test_highlight_rolloff_identity_in_full_day():
    # In full day (cos >= HIGHLIGHT_DAY_COS) the strength is exactly 0, so the
    # rolloff is an EXACT identity on the sun-corrected band -- INCLUDING >1
    # values (which must pass through unchanged so Rayleigh/synth-green/the clip
    # reproduce midday byte-for-byte).
    band = np.array([[0.3, 0.7, 1.0, 1.4, 2.5, 9.0]], np.float32)
    cos = np.full((1, 6), 0.5, np.float32)   # solidly day -> strength 0
    out = tc.highlight_rolloff(band, cos)
    assert np.array_equal(out, band)


def test_highlight_rolloff_rescues_terminator_highlights():
    # At the terminator (full strength) the knee soft-compresses >knee band values
    # BELOW 1 so distinct cloud tops keep distinct (sub-1) values = texture, while
    # below-knee values pass through untouched. Runs on a single sun-corrected band.
    knee = tc.HIGHLIGHT_KNEE
    band = np.array([[0.5, knee - 0.05, 1.5, 2.0, 4.0, 9.0]], np.float32)
    cos = np.zeros((1, 6), np.float32)       # horizon -> full strength
    out = tc.highlight_rolloff(band, cos)
    # below the knee: untouched
    assert out[0, 0] == np.float32(0.5)
    assert out[0, 1] == np.float32(knee - 0.05)
    # above the knee: compressed strictly into (knee, 1), monotonically ordered
    over = out[0, 2:]
    assert np.all(over > knee) and np.all(over < 1.0)
    assert np.all(np.diff(over) > 0)         # 1.5<2<4<9 -> distinct outputs (texture)


def test_highlight_rolloff_monotonic_no_inversion():
    # Across the full input range at full strength the rolloff is strictly
    # increasing -> tonal ordering is never inverted (no false edges in cloud).
    band = np.linspace(0.0, 12.0, 200).astype(np.float32)[None, :]
    cos = np.zeros((1, 200), np.float32)
    out = tc.highlight_rolloff(band, cos)
    assert np.all(np.diff(out[0]) > 0)


@pytest.mark.skipif(not _have_pyspectral(), reason="pyspectral Rayleigh unavailable")
def test_highlight_rolloff_recovers_texture_through_full_pipeline():
    """The regression the rolloff exists for: in the REAL do_rayleigh=True path,
    Rayleigh clips every band to [0,1], so without the rolloff a range of DIFFERENT
    bright cloud tops at low sun all clip to flat white (indistinguishable). The
    rolloff runs on the sun-corrected bands BEFORE that clip, so the tops keep
    distinct sub-white values. A purely-post-clip rolloff (the earlier buggy
    design) could not do this -- it would only uniformly darken the merged white.
    The pixels share one lat/lon (identical sun geometry) so the only variable is
    reflectance -> the recovered spread is pure texture, not a sun-angle gradient."""
    import datetime as dt
    # lon0/lat0 near sunrise on the solstice -> cos_sza ~0.12 (full-strength band).
    when = dt.datetime(2026, 6, 21, 6, 30, 0, tzinfo=dt.timezone.utc)
    lats = np.zeros((1, 4), np.float32); lons = np.zeros((1, 4), np.float32)
    cos, *_ = tc.solar_geometry(lats, lons, when)
    assert float(cos.max()) < tc.HIGHLIGHT_PEAK_COS, "scene must be deep terminator"
    # four cloud tops of increasing reflectance; all sun-correct well past 1.0
    refl = np.array([[0.35, 0.45, 0.55, 0.70]], np.float32)
    common = dict(when=when, sub_sat_lon=0.0, platform_name="GOES-19",
                  sensor="abi", do_rayleigh=True)
    on, _ = tc.assemble_truecolor(refl, None, refl.copy(), refl.copy(), lats, lons, **common)
    saved = tc.HIGHLIGHT_STRENGTH
    try:
        tc.HIGHLIGHT_STRENGTH = 0.0           # = old hard-clip behavior
        off, _ = tc.assemble_truecolor(refl, None, refl.copy(), refl.copy(), lats, lons, **common)
    finally:
        tc.HIGHLIGHT_STRENGTH = saved
    off_red, on_red = off[0, :, 0], on[0, :, 0]
    # WITHOUT the rolloff all four tops clip to ~identical white (texture lost)...
    assert off_red.max() - off_red.min() < 1 / 255 and off_red.min() > 0.97
    # ...WITH it they span a visible, monotonic range below white (texture).
    assert on_red.max() - on_red.min() > 5 / 255
    assert np.all(np.diff(on_red) > 0)
    assert on_red.max() < off_red.max()       # brightest top rescued downward from pure white


# --- AHI vegetation vibrance -----------------------------------------------

def test_ahi_vibrance_lifts_vegetation_protects_cloud_and_ocean():
    rgb = np.array([[[0.25, 0.34, 0.22],   # vegetation: green leads, blue low
                     [0.10, 0.20, 0.35],   # ocean: blue leads
                     [0.90, 0.90, 0.90]]], np.float32)  # cloud: neutral white
    out = tc.ahi_vegetation_vibrance(rgb)

    def sat(p):
        mx, mn = p.max(), p.min()
        return (mx - mn) / mx if mx > 0 else 0.0

    def luma(p):
        return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]

    veg_in, veg_out = rgb[0, 0], out[0, 0]
    # vegetation gains saturation and shifts greener (G up, R/B down)...
    assert sat(veg_out) > sat(veg_in)
    assert veg_out[1] > veg_in[1] and veg_out[0] < veg_in[0] and veg_out[2] < veg_in[2]
    # ...without changing brightness (luminance-preserving)
    assert abs(luma(veg_out) - luma(veg_in)) < 1e-3
    # ocean (blue-led) and cloud (neutral) are untouched
    assert np.allclose(out[0, 1], rgb[0, 1])
    assert np.allclose(out[0, 2], rgb[0, 2])
    # output stays in range and keeps shape/dtype
    assert out.shape == rgb.shape and out.dtype == rgb.dtype
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ahi_vibrance_water_guard_protects_teal_water():
    # Cyan/teal/turbid water reads green>=blue (so (g-b)>0) but absorbs red
    # (blue > red): the WATER GUARD must drive the lift to ~0 there so the water
    # is NOT pushed greener, while nearby vegetation (red >= blue) is still lifted.
    rgb = np.array([[[0.18, 0.50, 0.45],    # teal/turbid water: g>=b, blue >> red
                     [0.12, 0.40, 0.38],    # shallow shelf water: g>=b, blue > red
                     [0.10, 0.16, 0.09]]], np.float32)  # vegetation: red >= blue
    out = tc.ahi_vegetation_vibrance(rgb)
    # both water pixels are essentially untouched (guard killed the driver)
    assert np.allclose(out[0, 0], rgb[0, 0], atol=1e-3)
    assert np.allclose(out[0, 1], rgb[0, 1], atol=1e-3)
    # the land vegetation pixel is still lifted (green up)
    assert out[0, 2, 1] > rgb[0, 2, 1]


def _veg_scene(sensor, green_native):
    """A daytime scene whose pixel renders green-dominant, for the AHI-only
    vibrance gating test. green_native=None -> ABI (synthesized)."""
    import datetime as dt
    H = W = 3
    red = np.full((H, W), 0.10, np.float32)
    blue = np.full((H, W), 0.06, np.float32)
    veggie = np.full((H, W), 0.15, np.float32)
    green = np.full((H, W), 0.35, np.float32) if green_native else None
    when = dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
    lons, lats = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H))
    lats = lats.astype(np.float32); lons = lons.astype(np.float32)
    return dict(red=red, green=green, blue=blue, veggie=veggie, lats=lats, lons=lons,
                when=when, sub_sat_lon=0.0, do_rayleigh=False)


def test_assemble_vibrance_is_ahi_only():
    saved = tc.AHI_VIBRANCE_STRENGTH
    try:
        # AHI path: turning the vibrance on vs off MUST change the render.
        ahi = _veg_scene("ahi", green_native=True)
        tc.AHI_VIBRANCE_STRENGTH = 6.0
        on, _ = tc.assemble_truecolor(**ahi, platform_name="Himawari-9", sensor="ahi")
        tc.AHI_VIBRANCE_STRENGTH = 0.0
        off, _ = tc.assemble_truecolor(**ahi, platform_name="Himawari-9", sensor="ahi")
        assert not np.array_equal(on, off), "vibrance did not run on the AHI path"
        # ABI path: the AHI knob is irrelevant -- on vs off is byte-identical.
        abi = _veg_scene("abi", green_native=False)
        tc.AHI_VIBRANCE_STRENGTH = 6.0
        a_on, _ = tc.assemble_truecolor(**abi, platform_name="GOES-19", sensor="abi")
        tc.AHI_VIBRANCE_STRENGTH = 0.0
        a_off, _ = tc.assemble_truecolor(**abi, platform_name="GOES-19", sensor="abi")
        assert np.array_equal(a_on, a_off), "AHI vibrance leaked into the GOES/ABI path"
    finally:
        tc.AHI_VIBRANCE_STRENGTH = saved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
