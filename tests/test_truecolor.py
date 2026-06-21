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


def test_synth_green_between_inputs():
    red = np.array([0.3], np.float32)
    veg = np.array([0.5], np.float32)
    blue = np.array([0.1], np.float32)
    g = tc.synth_green(red, veg, blue)
    fr, fv, fb = tc.GREEN_FRACTIONS
    assert np.isclose(g[0], fr * 0.3 + fv * 0.5 + fb * 0.1)


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

def test_rayleigh_scale_field_full_in_daytime_taper_at_terminator():
    cos = np.array([[1.0, 0.6, tc.RAYLEIGH_TAPER_DAY_COS, 0.15,
                     tc.RAYLEIGH_TAPER_MIN_COS, -0.1]], np.float64)
    s = tc.rayleigh_scale_field(cos)
    # EXACTLY full (==RAYLEIGH_SCALE) for cos_sza >= the daytime threshold:
    # this is what keeps midday byte-identical.
    assert s[0, 0] == tc.RAYLEIGH_SCALE
    assert s[0, 1] == tc.RAYLEIGH_SCALE
    assert s[0, 2] == tc.RAYLEIGH_SCALE
    # Tapers to ~0 toward/below the terminator, monotonically.
    assert s[0, 3] < tc.RAYLEIGH_SCALE
    assert s[0, 4] == 0.0
    assert s[0, 5] == 0.0
    mid = s[0, 1:5]
    assert np.all(np.diff(mid) <= 0)  # non-increasing as the sun sets


def test_warm_tint_is_exact_noop_in_daytime():
    # The midday byte-identical guarantee: for a high-cos_sza block the warm
    # tint returns the RGB bit-for-bit unchanged.
    rng = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3) / 20.0
    cos_day = np.full((2, 3), 0.8, np.float32)   # >= WARM_TINT_DAY_COS
    out = tc.warm_terminator_tint(rng, cos_day)
    assert np.array_equal(out, rng)
    # exactly at the daytime threshold is still a no-op (smoothstep saturates)
    cos_edge = np.full((2, 3), tc.WARM_TINT_DAY_COS, np.float32)
    assert np.array_equal(tc.warm_terminator_tint(rng, cos_edge), rng)


def test_warm_tint_warms_terminator_red_up_blue_green_down():
    rgb = np.full((1, 1, 3), 0.5, np.float32)
    cos_term = np.array([[tc.WARM_TINT_PEAK_COS]], np.float32)  # full tint
    out = tc.warm_terminator_tint(rgb, cos_term)
    assert out[0, 0, 0] > 0.5            # red boosted
    assert out[0, 0, 1] < 0.5            # green pulled down
    assert out[0, 0, 2] < 0.5            # blue pulled down
    assert out[0, 0, 2] < out[0, 0, 1]   # blue down MORE than green -> orange/red


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
