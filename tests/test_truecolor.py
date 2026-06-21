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
    with both hooks neutralized to their pre-change form."""
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
