"""M0 executable safety contract for the corrected reference pixels."""

from __future__ import annotations

import math

import numpy as np
import pytest

import truecolor
from validation.truecolor.reference_pixels import (
    DERIVED_INVARIANTS,
    DISPLAY_ATOL,
    EIGHT_BIT_DN_ATOL,
    PROTECTED_CONSTANTS,
    REFERENCE_PIXELS,
    REFLECTANCE_ATOL,
    ReferencePixel,
    expected_dn,
    quantize_display,
)


def _scalar(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float32)


def _run_production_pointwise_pipeline(pixel: ReferencePixel) -> np.ndarray:
    """Run the real spatially pointwise production functions for one fixture."""

    cos_sza = np.asarray([pixel.cos_sza], dtype=np.float64)
    for role in ("red", "blue", "green"):
        source = getattr(pixel.reflectance_factor, role)
        expected = getattr(pixel.sunz_corrected, role)
        if source is None:
            assert expected is None
            continue
        actual = truecolor.sunz_correct(_scalar(source), cos_sza)
        np.testing.assert_allclose(actual, [expected], rtol=0.0, atol=REFLECTANCE_ATOL)

    veggie = truecolor.sunz_correct_pathlength(
        _scalar(pixel.reflectance_factor.veggie), cos_sza
    )
    np.testing.assert_allclose(
        veggie,
        [pixel.sunz_corrected.veggie],
        rtol=0.0,
        atol=REFLECTANCE_ATOL,
    )

    corrected = pixel.rayleigh_corrected
    green = truecolor.make_green(
        pixel.sensor,
        _scalar(corrected.red),
        _scalar(corrected.blue),
        None if corrected.green is None else _scalar(corrected.green),
        _scalar(corrected.veggie),
    )
    np.testing.assert_allclose(
        green,
        [pixel.synthesized_green],
        rtol=0.0,
        atol=REFLECTANCE_ATOL,
    )

    linear_rgb = np.asarray(
        [[corrected.red, float(green[0]), corrected.blue]], dtype=np.float32
    )
    np.testing.assert_allclose(
        linear_rgb[0], pixel.linear_rgb, rtol=0.0, atol=REFLECTANCE_ATOL
    )

    display = truecolor.cira_stretch(linear_rgb)[0]
    np.testing.assert_allclose(
        display, pixel.display_rgb, rtol=0.0, atol=DISPLAY_ATOL
    )
    return display


def _assert_protected_constants() -> None:
    for attribute, expected in PROTECTED_CONSTANTS.items():
        assert getattr(truecolor, attribute) == expected, attribute


def test_all_five_reference_pixels_exercise_the_production_recipe():
    assert len(REFERENCE_PIXELS) == 5
    for pixel in REFERENCE_PIXELS:
        _run_production_pointwise_pipeline(pixel)


def test_protected_constants_are_exact():
    _assert_protected_constants()


def test_closed_form_derived_invariants_use_production_functions():
    inv = DERIVED_INVARIANTS
    black = inv["cira_black_clip_reflectance"]
    white = inv["cira_white_clip_reflectance"]

    display = truecolor.cira_stretch(
        np.asarray([black, 1.0, white], dtype=np.float64)
    )
    np.testing.assert_allclose(
        display,
        [0.0, inv["cira_at_unit_reflectance"], 1.0],
        rtol=0.0,
        atol=DISPLAY_ATOL,
    )

    octave = 255.0 * math.log10(2.0) / truecolor._CIRA_DENOM
    assert octave == pytest.approx(inv["cira_dn_per_octave"], abs=1e-12)

    cos_88 = np.asarray([math.cos(math.radians(88.0))], dtype=np.float64)
    geometric = truecolor.sunz_correct(np.ones(1, dtype=np.float32), cos_88)
    pathlength = truecolor.sunz_correct_pathlength(
        np.ones(1, dtype=np.float32), cos_88
    )
    nadir_pathlength = truecolor.sunz_correct_pathlength(
        np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float64)
    )
    np.testing.assert_allclose(
        geometric,
        [inv["sunz_multiplier_at_88deg"]],
        rtol=0.0,
        atol=REFLECTANCE_ATOL,
    )
    np.testing.assert_allclose(
        pathlength,
        [inv["li_shibata_at_88deg"]],
        rtol=0.0,
        atol=REFLECTANCE_ATOL,
    )
    np.testing.assert_allclose(
        nadir_pathlength,
        [inv["li_shibata_at_nadir"]],
        rtol=0.0,
        atol=REFLECTANCE_ATOL,
    )

    unit = truecolor.cira_stretch(np.ones(1, dtype=np.float32))
    floor_dn = int(quantize_display(unit, "floor_production")[0])
    round_half_up_dn = int(quantize_display(unit, "round_half_up")[0])
    assert floor_dn == inv["cira_dn_at_unit_reflectance_floor_production"]
    assert round_half_up_dn == inv["cira_dn_at_unit_reflectance_round_half_up"]
    assert 255 - floor_dn == inv[
        "cira_headroom_dn_above_unit_reflectance_floor_production"
    ]


def test_floor_quantizer_compares_to_floor_production_column():
    assert EIGHT_BIT_DN_ATOL == 0
    for pixel in REFERENCE_PIXELS:
        display = _run_production_pointwise_pipeline(pixel)
        actual = quantize_display(display, "floor_production")
        np.testing.assert_array_equal(actual, expected_dn(pixel, "floor_production"))


def test_round_half_up_quantizer_compares_to_round_half_up_column():
    assert EIGHT_BIT_DN_ATOL == 0
    for pixel in REFERENCE_PIXELS:
        display = _run_production_pointwise_pipeline(pixel)
        actual = quantize_display(display, "round_half_up")
        np.testing.assert_array_equal(actual, expected_dn(pixel, "round_half_up"))


def test_wrong_quantizer_column_is_detected_for_four_corrected_fixtures():
    floor_vs_round_half_up = []
    round_half_up_vs_floor = []
    for pixel in REFERENCE_PIXELS:
        display = _run_production_pointwise_pipeline(pixel)
        floor_vs_round_half_up.append(
            not np.array_equal(
                quantize_display(display, "floor_production"),
                expected_dn(pixel, "round_half_up"),
            )
        )
        round_half_up_vs_floor.append(
            not np.array_equal(
                quantize_display(display, "round_half_up"),
                expected_dn(pixel, "floor_production"),
            )
        )

    assert sum(floor_vs_round_half_up) == 4
    assert sum(round_half_up_vs_floor) == 4


@pytest.mark.parametrize(
    ("attribute", "mutated"),
    (
        ("SUNZ_LIMIT_DEG", 87.5),
        ("ABI_GREEN_FRACTIONS", (0.465, 0.465, 0.07)),
        ("_CIRA_LOG_ROOT", -1.64),
    ),
)
def test_protected_constant_mutations_are_detected(monkeypatch, attribute, mutated):
    monkeypatch.setattr(truecolor, attribute, mutated)
    with pytest.raises(AssertionError):
        _assert_protected_constants()


@pytest.mark.parametrize(
    ("attribute", "mutated", "fixture_id"),
    (
        ("SUNZ_LIMIT_DEG", 70.0, "P4_thin_cirrus_highSZA"),
        ("ABI_GREEN_FRACTIONS", (0.465, 0.465, 0.07), "P3_vegetated_land"),
        ("_CIRA_LOG_ROOT", -1.64, "P1_deep_convective_cloud_top"),
    ),
)
def test_reference_pipeline_discriminates_formula_mutations(
    monkeypatch, attribute, mutated, fixture_id
):
    pixel = next(pixel for pixel in REFERENCE_PIXELS if pixel.id == fixture_id)
    monkeypatch.setattr(truecolor, attribute, mutated)
    with pytest.raises(AssertionError):
        _run_production_pointwise_pipeline(pixel)
