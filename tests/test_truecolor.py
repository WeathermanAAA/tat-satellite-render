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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
