"""M2 physical-state, inverse-CIRA, and pointwise-finalizer contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import truecolor as tc


def _cos_for_day_weight(weight: float) -> np.float32:
    if weight >= 1.0:
        return np.float32(1.0)
    if weight <= 0.0:
        return np.float32(math.cos(math.radians(96.0)))
    # Invert the unchanged Satpy 88->95 log taper.
    normalized_angle = 2.0 ** (1.0 - weight) - 1.0
    sza = tc.SUNZ_LIMIT_DEG + (
        tc.SUNZ_MAX_DEG - tc.SUNZ_LIMIT_DEG
    ) * normalized_angle
    return np.float32(math.cos(math.radians(sza)))


def _state(
    day_value: float | np.ndarray,
    *,
    day_weight: float = 1.0,
    day_coverage: float | np.ndarray = 1.0,
    ir_bt: float | np.ndarray | None = None,
    ir_coverage: float | np.ndarray = 1.0,
    red: float = 0.4,
    veggie: float | None = 0.4,
    glint: float = 90.0,
    geom_valid: bool | np.ndarray = True,
    sensor: str = "abi",
) -> tc.TrueColorPhysicalState:
    value = np.asarray(day_value, dtype=np.float32)
    if value.ndim == 0:
        rgb = np.full((1, 1, 3), value, dtype=np.float32)
    elif value.ndim == 1 and value.shape == (3,):
        rgb = value.reshape(1, 1, 3).copy()
    elif value.ndim == 3 and value.shape[-1] == 3:
        rgb = value.copy()
    else:
        raise AssertionError(f"unsupported test day shape {value.shape}")
    shape = rgb.shape[:2]

    def field(v, dtype=np.float32):
        a = np.asarray(v, dtype=dtype)
        return np.full(shape, a, dtype=dtype) if a.ndim == 0 else a.copy()

    day_cov = field(day_coverage)
    geom = field(geom_valid, np.bool_)
    day_cov = np.where(geom, day_cov, np.float32(0.0)).astype(np.float32)
    bt = None if ir_bt is None else field(ir_bt)
    ir_cov = None if bt is None else field(ir_coverage)
    if ir_cov is not None:
        ir_cov = np.where(geom, ir_cov, np.float32(0.0)).astype(np.float32)
    weight = np.where(
        geom, field(day_weight), np.float32(0.0)
    ).astype(np.float32)
    return tc.TrueColorPhysicalState(
        day_rgb_reflectance=rgb,
        day_coverage=day_cov,
        ir_bt_kelvin=bt,
        ir_coverage=ir_cov,
        cos_sza=field(_cos_for_day_weight(day_weight)),
        day_weight=weight,
        geom_valid=geom,
        red_corrected=field(red),
        veggie_corrected=None if veggie is None else field(veggie),
        glint_deg=field(glint),
        sensor=sensor,
    )


def test_inverse_cira_is_derived_continuous_and_round_trips_10001_float32_values():
    gray = np.linspace(0.0, 1.0, 10_001, dtype=np.float32)
    preimage = tc.cira_preimage_compatibility(gray)
    display = tc.cira_stretch(preimage)

    q0 = 10.0 ** tc._CIRA_LOG_ROOT
    q1 = 10.0 ** (tc._CIRA_LOG_ROOT + tc._CIRA_DENOM)
    assert preimage.dtype == np.float32
    assert preimage[0] == pytest.approx(q0, abs=2e-8)
    assert preimage[-1] == pytest.approx(q1, abs=2e-7)
    assert q0 == pytest.approx(0.0223, abs=1e-15)
    assert q1 == pytest.approx(2.1730825067739694, abs=1e-14)
    assert np.all(np.diff(preimage) > 0.0)
    assert float(np.max(np.abs(display - gray))) <= 1.0e-6


def test_inverse_cira_opposite_direction_is_only_identity_inside_principal_interval():
    q0 = np.float32(10.0 ** tc._CIRA_LOG_ROOT)
    q1 = np.float32(10.0 ** (tc._CIRA_LOG_ROOT + tc._CIRA_DENOM))
    reflectance = np.linspace(q0, q1, 1001, dtype=np.float32)
    recovered = tc.cira_preimage_compatibility(tc.cira_stretch(reflectance))
    np.testing.assert_allclose(recovered, reflectance, rtol=0.0, atol=8e-7)

    saturated = np.array([q0 / 2.0, q1 * 2.0], dtype=np.float32)
    recovered_saturated = tc.cira_preimage_compatibility(tc.cira_stretch(saturated))
    np.testing.assert_allclose(recovered_saturated, [q0, q1], rtol=0.0, atol=3e-7)
    assert not np.array_equal(recovered_saturated, saturated)


def test_inverse_cira_preserves_valid_black_but_never_manufactures_it_from_missing():
    gray = np.array([0.0, 0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    coverage = np.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    result = tc.cira_preimage_compatibility(gray, coverage=coverage)
    assert result[0] == pytest.approx(0.0223, abs=2e-8)
    assert np.isnan(result[1:]).all()
    assert tc.cira_stretch(result[:1])[0] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("masked_name", ["gray", "coverage", "bt_kelvin"])
def test_inverse_cira_helpers_reject_masked_array_lookalikes(masked_name):
    values = np.ma.array(
        np.array([0.5], dtype=np.float32),
        mask=np.array([True], dtype=np.bool_),
    )
    coverage = np.ones(1, dtype=np.float32)
    with pytest.raises(TypeError, match="plain numpy.ndarray"):
        if masked_name == "gray":
            tc.cira_preimage_compatibility(values, coverage=coverage)
        elif masked_name == "coverage":
            tc.cira_preimage_compatibility(
                np.array([0.5], dtype=np.float32), coverage=values
            )
        else:
            tc.clean_ir_cira_preimage(values, coverage=coverage)


@pytest.mark.parametrize(
    ("bt", "gray", "preimage"),
    (
        (330.0, 0.0, 0.0223),
        (303.0, 0.0, 0.0223),
        (300.0, 0.025, 0.025004843852977306),
        (270.0, 0.275, 0.07856280571886973),
        (243.0, 0.5, 0.22013573063239758),
        (230.0, 0.6083333333333333, 0.36152702658077823),
        (195.0, 0.9, 1.3746719799239653),
        (183.0, 1.0, 2.1730825067739694),
        (175.0, 1.0, 2.1730825067739694),
    ),
)
def test_clean_ir_preimage_closed_form_fixtures(bt, gray, preimage):
    actual = tc.clean_ir_cira_preimage(np.array([bt], dtype=np.float32))
    assert actual[0] == pytest.approx(preimage, rel=2e-7, abs=2e-8)
    assert tc.cira_stretch(actual)[0] == pytest.approx(gray, abs=2e-7)


def test_ir_spatial_operation_must_precede_nonlinear_night_mapping():
    # Equal physical support at 303 K and 183 K delivers 243 K, then gray .5.
    delivered_bt = np.array([(303.0 + 183.0) / 2.0], dtype=np.float32)
    approved = tc.cira_stretch(tc.clean_ir_cira_preimage(delivered_bt))[0]

    endpoints = tc.clean_ir_cira_preimage(
        np.array([303.0, 183.0], dtype=np.float32)
    )
    preimage_first_mutation = tc.cira_stretch(
        np.array([endpoints.mean()], dtype=np.float32)
    )[0]
    assert approved == pytest.approx(0.5, abs=2e-7)
    assert preimage_first_mutation == pytest.approx(0.8508647001182855, abs=3e-7)
    assert abs(float(approved - preimage_first_mutation)) > 0.3


@pytest.mark.parametrize("sensor", ("ami", "fci"))
def test_ami_and_fci_use_the_same_typed_physical_contract_without_e2e_claims(
    monkeypatch, sensor
):
    shape = (3, 4)

    def solar(lats, lons, when):
        return (
            np.ones(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
        )

    def satellite(lats, lons, sub_sat_lon, when):
        return (
            np.zeros(shape, dtype=np.float64),
            np.full(shape, 180.0, dtype=np.float64),
        )

    monkeypatch.setattr(tc, "solar_geometry", solar)
    monkeypatch.setattr(tc, "satellite_geometry", satellite)
    band = np.full(shape, 0.2, dtype=np.float32)
    lats = np.zeros(shape, dtype=np.float32)
    lons = np.zeros(shape, dtype=np.float32)
    geom = np.ones(shape, dtype=np.bool_)
    geom[0, 0] = False
    state = tc.assemble_truecolor_physical(
        band,
        band,
        band,
        band,
        lats,
        lons,
        when=SimpleNamespace(),
        sub_sat_lon=0.0,
        platform_name="TEST",
        sensor=sensor,
        ir_bt=np.full(shape, 243.0, dtype=np.float32),
        geom_valid=geom,
        do_rayleigh=False,
    )

    assert type(state) is tc.TrueColorPhysicalState
    assert state.sensor == sensor
    assert state.array_domain is tc.TrueColorArrayDomain.CORRECTED_TOA_REFLECTANCE
    assert state.ir_temperature_unit is tc.TrueColorTemperatureUnit.KELVIN
    for field in (
        state.day_rgb_reflectance,
        state.day_coverage,
        state.ir_bt_kelvin,
        state.ir_coverage,
        state.cos_sza,
        state.day_weight,
        state.red_corrected,
        state.veggie_corrected,
        state.glint_deg,
    ):
        assert field.dtype == np.float32
    assert state.day_rgb_reflectance.shape == (*shape, 3)
    assert state.day_coverage[0, 0] == 0.0
    assert state.ir_coverage[0, 0] == 0.0
    assert np.allclose(state.day_rgb_reflectance[1:, :, :], 0.2)


@pytest.mark.parametrize(
    ("field_name", "dtype"),
    (
        ("red", np.uint8),
        ("green", np.float64),
        ("blue", np.uint8),
        ("veggie", np.float64),
        ("lats", np.float64),
        ("lons", np.float64),
        ("ir_bt", np.uint8),
    ),
)
def test_physical_assembler_rejects_non_float32_domain_lookalikes(
    field_name, dtype
):
    shape = (2, 3)
    fields = {
        name: np.ones(shape, dtype=np.float32)
        for name in ("red", "green", "blue", "veggie", "lats", "lons", "ir_bt")
    }
    fields[field_name] = fields[field_name].astype(dtype)
    with pytest.raises(TypeError, match=f"{field_name} must be float32 physical"):
        tc.assemble_truecolor_physical(
            fields["red"],
            fields["green"],
            fields["blue"],
            fields["veggie"],
            fields["lats"],
            fields["lons"],
            when=SimpleNamespace(),
            sub_sat_lon=0.0,
            platform_name="TEST",
            sensor="abi",
            ir_bt=fields["ir_bt"],
            do_rayleigh=False,
        )


def test_physical_assembler_rejects_masked_raw_band_before_recipe():
    shape = (2, 3)
    band = np.ones(shape, dtype=np.float32)
    masked = np.ma.array(band, mask=np.zeros(shape, dtype=np.bool_))
    with pytest.raises(ValueError, match="blue must have shape"):
        tc.assemble_truecolor_physical(
            band,
            band,
            masked,
            band,
            band,
            band,
            when=SimpleNamespace(),
            sub_sat_lon=0.0,
            platform_name="TEST",
            sensor="abi",
            do_rayleigh=False,
        )


def test_physical_state_rejects_encoded_wrong_domain_shape_and_mask_broadcasting():
    state = _state(0.3)
    kwargs = dict(state.__dict__)

    encoded = dict(kwargs)
    encoded["day_rgb_reflectance"] = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="encoded RGB"):
        tc.TrueColorPhysicalState(**encoded)

    wrong_domain = dict(kwargs)
    wrong_domain["array_domain"] = "corrected_toa_reflectance"
    with pytest.raises(ValueError, match="encoded/display RGB"):
        tc.TrueColorPhysicalState(**wrong_domain)

    wrong_shape = dict(kwargs)
    wrong_shape["geom_valid"] = np.ones((1, 2), dtype=np.bool_)
    with pytest.raises(ValueError, match="geom_valid"):
        tc.TrueColorPhysicalState(**wrong_shape)

    outside_geometry = dict(kwargs)
    outside_geometry["geom_valid"] = np.zeros((1, 1), dtype=np.bool_)
    with pytest.raises(ValueError, match="outside geom_valid"):
        tc.TrueColorPhysicalState(**outside_geometry)

    masked_rgb = dict(kwargs)
    masked_rgb["day_rgb_reflectance"] = np.ma.array(
        kwargs["day_rgb_reflectance"],
        mask=np.zeros_like(kwargs["day_rgb_reflectance"], dtype=np.bool_),
    )
    with pytest.raises(TypeError, match="plain numpy.ndarray"):
        tc.TrueColorPhysicalState(**masked_rgb)

    masked_coverage = dict(kwargs)
    masked_coverage["day_coverage"] = np.ma.array(
        kwargs["day_coverage"],
        mask=np.zeros_like(kwargs["day_coverage"], dtype=np.bool_),
    )
    with pytest.raises(TypeError, match="plain numpy.ndarray"):
        tc.TrueColorPhysicalState(**masked_coverage)

    masked_weight = dict(kwargs)
    masked_weight["day_weight"] = np.ma.array(
        kwargs["day_weight"],
        mask=np.zeros_like(kwargs["day_weight"], dtype=np.bool_),
    )
    with pytest.raises(TypeError, match="plain numpy.ndarray"):
        tc.TrueColorPhysicalState(**masked_weight)

    bad_weight = dict(kwargs)
    bad_weight["day_weight"] = np.full((1, 1), 1.01, dtype=np.float32)
    with pytest.raises(ValueError, match=r"day_weight.*\[0, 1\]"):
        tc.TrueColorPhysicalState(**bad_weight)


def test_physical_state_rejects_missing_ir_pair_and_nonfinite_covered_values():
    state = _state(0.3, ir_bt=250.0)
    kwargs = dict(state.__dict__)
    kwargs["ir_coverage"] = None
    with pytest.raises(ValueError, match="both be present"):
        tc.TrueColorPhysicalState(**kwargs)

    state = _state(0.3)
    kwargs = dict(state.__dict__)
    bad = kwargs["day_rgb_reflectance"].copy()
    bad[0, 0, 1] = np.nan
    kwargs["day_rgb_reflectance"] = bad
    with pytest.raises(ValueError, match="finite where day coverage"):
        tc.TrueColorPhysicalState(**kwargs)


def test_legacy_finalizer_is_the_exact_existing_operation_chain():
    state = _state([0.25, 0.35, 0.45], day_weight=0.5, ir_bt=250.0)
    expected = tc.cira_stretch(state.day_rgb_reflectance)
    expected = tc.glint_tame(
        expected, state.glint_deg, state.red_corrected, state.veggie_corrected
    )
    expected = tc.night_blend(expected, state.ir_bt_kelvin, state.cos_sza)
    actual, actual_cos = tc.finalize_truecolor_legacy(
        state.day_rgb_reflectance,
        state.cos_sza,
        state.glint_deg,
        state.red_corrected,
        state.veggie_corrected,
        state.ir_bt_kelvin,
    )
    assert np.array_equal(actual, expected.astype(np.float32))
    assert np.array_equal(actual_cos, state.cos_sza)


def test_m2_pure_day_matches_existing_cira_and_is_independent_of_ir():
    state = _state([0.03, 0.3, 1.2], day_weight=1.0, ir_bt=np.nan,
                   ir_coverage=0.0)
    delivery = tc.finalize_truecolor_m2(state)
    np.testing.assert_array_equal(delivery.rgb, tc.cira_stretch(state.day_rgb_reflectance))
    np.testing.assert_array_equal(delivery.coverage, np.ones((1, 1), np.float32))
    np.testing.assert_array_equal(
        delivery.surviving_day_fraction, np.ones((1, 1), np.float32)
    )


def test_m2_pure_night_preserves_clean_ir_and_glint_is_exact_noop(monkeypatch):
    state = _state(0.87, day_weight=0.0, ir_bt=230.0, glint=0.0,
                   red=0.05, veggie=0.03)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure night must not call legacy glint_tame")

    monkeypatch.setattr(tc, "glint_tame", forbidden)
    delivery = tc.finalize_truecolor_m2(state)
    expected_gray = (tc.IR_T_WARM - 230.0) / (tc.IR_T_WARM - tc.IR_T_COLD)
    np.testing.assert_allclose(delivery.rgb, expected_gray, rtol=0.0, atol=2e-7)
    assert delivery.surviving_day_fraction[0, 0] == 0.0


@pytest.mark.parametrize(("bt_kelvin", "expected_dn"), ((255.0, 102), (287.0, 34)))
def test_m2_pure_night_preserves_legacy_clean_ir_floor_byte(
    bt_kelvin, expected_dn
):
    state = _state(0.7, day_weight=0.0, ir_bt=bt_kelvin, glint=0.0)
    delivery = tc.finalize_truecolor_m2(state)
    expected_gray = np.float32(
        (tc.IR_T_WARM - bt_kelvin) / (tc.IR_T_WARM - tc.IR_T_COLD)
    )
    np.testing.assert_array_equal(
        delivery.rgb[0, 0], np.full(3, expected_gray, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.floor(delivery.rgb[0, 0] * np.float32(255.0)).astype(np.uint8),
        np.full(3, expected_dn, dtype=np.uint8),
    )


def test_m2_mixed_fixture_rejects_direct_gray_and_old_display_blend_mutations():
    state = _state(0.03, day_weight=0.75, ir_bt=183.0, glint=90.0)
    approved = float(tc.finalize_truecolor_m2(state).rgb[0, 0, 0])
    actual_weight = float(tc.terminator_day_weight(state.cos_sza)[0, 0])
    assert actual_weight == pytest.approx(0.75, abs=2e-7)

    direct_gray_mutation = float(tc.cira_stretch(np.array(
        [actual_weight * 0.03 + (1.0 - actual_weight) * 1.0], dtype=np.float32
    ))[0])
    old_display_mutation = (
        actual_weight * float(tc.cira_stretch(np.array([0.03], np.float32))[0])
        + (1.0 - actual_weight)
    )
    assert approved == pytest.approx(0.7061322214477085, abs=4e-7)
    assert direct_gray_mutation == pytest.approx(0.5465996088897274, abs=4e-7)
    assert old_display_mutation == pytest.approx(0.29857888445637004, abs=4e-7)
    assert approved > direct_gray_mutation > old_display_mutation


def test_m2_valid_zero_night_endpoint_rejects_discontinuous_zero_preimage():
    state = _state(0.3, day_weight=0.5, ir_bt=303.0, glint=90.0)
    approved = float(tc.finalize_truecolor_m2(state).rgb[0, 0, 0])
    actual_weight = float(tc.terminator_day_weight(state.cos_sza)[0, 0])
    zero_preimage_mutation = float(tc.cira_stretch(np.array(
        [actual_weight * 0.3], dtype=np.float32
    ))[0])
    assert approved == pytest.approx(0.43188747948732364, abs=4e-7)
    assert zero_preimage_mutation == pytest.approx(0.41623004820434606, abs=4e-7)
    assert approved - zero_preimage_mutation > 0.015


@pytest.mark.parametrize("weight", (0.0, 0.2, 0.5, 0.8, 1.0))
def test_m2_constant_preimage_coordinate_is_invariant_across_modal_weights(weight):
    gray = np.float32(0.4)
    day_coordinate = tc.cira_preimage_compatibility(np.array([gray]))[0]
    ir_bt = tc.IR_T_WARM - float(gray) * (tc.IR_T_WARM - tc.IR_T_COLD)
    state = _state(day_coordinate, day_weight=weight, ir_bt=ir_bt, glint=90.0)
    delivery = tc.finalize_truecolor_m2(state)
    np.testing.assert_allclose(delivery.rgb, gray, rtol=0.0, atol=3e-7)


def test_m2_coverage_normalization_preserves_value_and_retains_missing_fraction():
    state = _state(0.4, day_weight=1.0, day_coverage=0.5, glint=90.0)
    delivery = tc.finalize_truecolor_m2(state)
    expected = tc.cira_stretch(np.array([0.4], dtype=np.float32))[0]
    black_fill_mutation = tc.cira_stretch(np.array([0.2], dtype=np.float32))[0]
    assert delivery.coverage[0, 0] == pytest.approx(0.5)
    assert delivery.rgb[0, 0, 0] == pytest.approx(expected, abs=1e-7)
    assert expected == pytest.approx(0.6304169288082948, abs=3e-7)
    assert black_fill_mutation == pytest.approx(0.4790521199512105, abs=3e-7)
    assert delivery.rgb[0, 0, 0] - black_fill_mutation > 0.15


def test_m2_zero_coefficient_excludes_missing_modality_without_nan_propagation():
    pure_day = _state(0.4, day_weight=1.0, ir_bt=np.nan, ir_coverage=0.0)
    assert np.isfinite(tc.finalize_truecolor_m2(pure_day).rgb).all()

    pure_night = _state(np.nan, day_weight=0.0, day_coverage=0.0,
                        ir_bt=243.0, ir_coverage=1.0)
    delivery = tc.finalize_truecolor_m2(pure_night)
    np.testing.assert_allclose(delivery.rgb, 0.5, rtol=0.0, atol=2e-7)


def test_m2_subthreshold_modality_weight_is_removed_from_coverage_and_numerator():
    state = _state(
        0.4,
        day_weight=1.0,
        day_coverage=float(tc.M2_COVERAGE_EPSILON) / 2.0,
        ir_bt=None,
        geom_valid=True,
    )
    delivery = tc.finalize_truecolor_m2(state)
    assert delivery.coverage[0, 0] == 0.0
    assert delivery.surviving_day_fraction[0, 0] == 0.0
    assert np.isnan(delivery.rgb[0, 0]).all()


def test_m2_alpha_nan_and_opaque_pre_cira_background_contracts_are_explicit():
    uncovered = _state(np.nan, day_weight=1.0, day_coverage=0.0,
                       ir_bt=None, geom_valid=True)
    alpha = tc.finalize_truecolor_m2(uncovered)
    assert alpha.background_contract is tc.TrueColorBackgroundContract.ALPHA_NAN
    assert alpha.coverage[0, 0] == 0.0
    assert np.isnan(alpha.rgb).all()

    background = (0.07, 0.11, 0.19)
    opaque = tc.finalize_truecolor_m2(
        uncovered,
        background_display_rgb=background,
    )
    assert opaque.background_contract is tc.TrueColorBackgroundContract.OPAQUE_PRE_CIRA
    np.testing.assert_allclose(opaque.rgb[0, 0], background, rtol=0.0, atol=2e-7)
    assert opaque.coverage[0, 0] == 0.0

    partial = _state(0.4, day_weight=1.0, day_coverage=0.5, glint=90.0)
    partial_delivery = tc.finalize_truecolor_m2(
        partial,
        background_display_rgb=background,
    )
    background_preimage = tc.cira_preimage_compatibility(
        np.asarray(background, dtype=np.float32)
    )
    expected = tc.cira_stretch(
        np.float32(0.5) * partial.day_rgb_reflectance[0, 0]
        + np.float32(0.5) * background_preimage
    )
    np.testing.assert_allclose(partial_delivery.rgb[0, 0], expected,
                               rtol=0.0, atol=2e-7)
    assert partial_delivery.coverage[0, 0] == pytest.approx(0.5)

    partial_night = _state(
        0.8,
        day_weight=0.0,
        day_coverage=0.0,
        ir_bt=243.0,
        ir_coverage=0.5,
    )
    partial_night_delivery = tc.finalize_truecolor_m2(
        partial_night,
        background_display_rgb=background,
    )
    night_preimage = tc.clean_ir_cira_preimage(
        np.asarray([243.0], dtype=np.float32)
    )[0]
    expected_partial_night = tc.cira_stretch(
        np.float32(0.5) * np.full(3, night_preimage, dtype=np.float32)
        + np.float32(0.5) * background_preimage
    )
    np.testing.assert_allclose(
        partial_night_delivery.rgb[0, 0],
        expected_partial_night,
        rtol=0.0,
        atol=2e-7,
    )
    assert not np.array_equal(
        partial_night_delivery.rgb[0, 0],
        np.full(3, np.float32(0.5), dtype=np.float32),
    )

    dark_bg = np.asarray([10, 13, 18], dtype=np.float32) / np.float32(255.0)
    exact_dark = tc.finalize_truecolor_m2(
        uncovered,
        background_display_rgb=tuple(float(v) for v in dark_bg),
    )
    np.testing.assert_array_equal(exact_dark.rgb[0, 0], dark_bg)
    np.testing.assert_array_equal(
        np.floor(exact_dark.rgb[0, 0] * np.float32(255.0)).astype(np.uint8),
        np.asarray([10, 13, 18], dtype=np.uint8),
    )


def test_m2_glint_compatibility_is_full_day_exact_and_scales_mixed_delta():
    full_state = _state([0.7, 0.75, 0.8], day_weight=1.0, red=0.05,
                        veggie=0.03, glint=0.0)
    full = tc.finalize_truecolor_m2(full_state)
    base = tc.cira_stretch(full_state.day_rgb_reflectance)
    legacy_glint = tc.glint_tame(
        base,
        full_state.glint_deg,
        full_state.red_corrected,
        full_state.veggie_corrected,
    )
    np.testing.assert_array_equal(full.rgb, legacy_glint)

    mixed_state = _state([0.7, 0.75, 0.8], day_weight=0.5, ir_bt=243.0,
                         red=0.05, veggie=0.03, glint=0.0)
    mixed = tc.finalize_truecolor_m2(mixed_state)
    weight = mixed.surviving_day_fraction[0, 0]
    night_j = tc.clean_ir_cira_preimage(np.array([243.0], np.float32))[0]
    coordinate = (
        np.float32(0.5) * mixed_state.day_rgb_reflectance[0, 0]
        + np.float32(0.5) * night_j
    )
    mixed_base = tc.cira_stretch(coordinate)
    mixed_full_glint = tc.glint_tame(
        mixed_base.reshape(1, 1, 3),
        mixed_state.glint_deg,
        mixed_state.red_corrected,
        mixed_state.veggie_corrected,
    )[0, 0]
    expected = mixed_base + (mixed_full_glint - mixed_base) * weight
    np.testing.assert_allclose(mixed.rgb[0, 0], expected, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize(
    "field_name",
    ("rgb", "glint_deg", "red_corrected", "veggie_corrected", "fraction"),
)
def test_m2_glint_compatibility_rejects_masked_array_lookalikes(field_name):
    rgb = np.full((1, 2, 3), 0.4, dtype=np.float32)
    field = np.full((1, 2), 0.5, dtype=np.float32)
    values = {
        "rgb": rgb,
        "glint_deg": field.copy(),
        "red_corrected": field.copy(),
        "veggie_corrected": field.copy(),
        "fraction": field.copy(),
    }
    values[field_name] = np.ma.array(
        values[field_name],
        mask=np.zeros_like(values[field_name], dtype=np.bool_),
    )
    with pytest.raises(TypeError, match="plain numpy.ndarray"):
        tc.apply_display_glint_compat(
            values["rgb"],
            values["glint_deg"],
            values["red_corrected"],
            values["veggie_corrected"],
            values["fraction"],
        )


@pytest.mark.parametrize(
    "field_name",
    ("rgb", "glint_deg", "red_corrected", "veggie_corrected", "fraction"),
)
def test_m2_glint_compatibility_rejects_non_float32_lookalikes(field_name):
    rgb = np.full((1, 2, 3), 0.4, dtype=np.float32)
    field = np.full((1, 2), 0.5, dtype=np.float32)
    values = {
        "rgb": rgb,
        "glint_deg": field.copy(),
        "red_corrected": field.copy(),
        "veggie_corrected": field.copy(),
        "fraction": field.copy(),
    }
    values[field_name] = values[field_name].astype(np.float64)
    with pytest.raises(TypeError, match="must be float32"):
        tc.apply_display_glint_compat(
            values["rgb"],
            values["glint_deg"],
            values["red_corrected"],
            values["veggie_corrected"],
            values["fraction"],
        )


def test_m2_glint_compatibility_requires_finite_values_only_on_active_day_support():
    rgb = np.array([[[np.nan] * 3, [0.4] * 3]], dtype=np.float32)
    glint = np.array([[np.nan, 0.0]], dtype=np.float32)
    red = np.array([[np.nan, 0.05]], dtype=np.float32)
    veggie = np.array([[np.nan, 0.03]], dtype=np.float32)
    fraction = np.array([[0.0, 0.5]], dtype=np.float32)

    result = tc.apply_display_glint_compat(rgb, glint, red, veggie, fraction)
    assert np.isnan(result[0, 0]).all()
    assert np.isfinite(result[0, 1]).all()

    active_nan = rgb.copy()
    active_nan[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="active display-glint inputs"):
        tc.apply_display_glint_compat(
            active_nan, glint, red, veggie, fraction
        )


def test_m2_uses_one_cira_call_and_retains_legacy_88_to_95_without_m6_gate(
    monkeypatch,
):
    assert tc.M2_DAY_WEIGHT_POLICY_ID == "legacy_88_to_95_no_survival_gate"
    cos = np.array(
        [math.cos(math.radians(v)) for v in (88.0, 90.0, 94.0, 95.0)],
        dtype=np.float32,
    )
    weight = tc.terminator_day_weight(cos)
    assert weight[0] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < weight[2] < weight[1] < 1.0
    assert weight[3] == pytest.approx(0.0, abs=1e-6)

    calls = 0
    original = tc.cira_stretch

    def counted(values):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(tc, "cira_stretch", counted)
    tc.finalize_truecolor_m2(_state(0.3, day_weight=0.5, ir_bt=250.0))
    assert calls == 1


def test_m2_finalizer_rejects_unregistered_lookalike_state():
    genuine = _state(0.3)
    lookalike = SimpleNamespace(**genuine.__dict__)
    with pytest.raises(TypeError, match="registered TrueColorPhysicalState"):
        tc.finalize_truecolor_m2(lookalike)
