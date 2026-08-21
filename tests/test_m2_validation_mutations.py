"""Mutation tests for the independent, non-product M2 scientific oracle."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from validation.truecolor.m2 import (
    CIRA_BLACK_PREIMAGE_EXPECTED,
    CIRA_DENOM_EXPECTED,
    CIRA_LOG_ROOT_EXPECTED,
    CIRA_WHITE_PREIMAGE_EXPECTED,
    COVERAGE_TAU,
    FORBIDDEN_M2_CLOSURE_GATE_IDS,
    M2_ALL_GATE_IDS,
    M2_CANONICAL_MUTATION_IDS,
    M2_CAPTURE_METHOD,
    M2_PRODUCTION_MUTATION_IDS,
    M2_REQUIRED_GATE_IDS,
    M2_SYNTHETIC_MUTATION_IDS,
    M2_TRACE_SCHEMA,
    PHYSICAL_COVERAGE_TRANSFORM,
    PHYSICAL_DELIVERY_KERNEL,
    PHYSICAL_DELIVERY_OPERATION_KIND,
    CapturedGeometry,
    MutationResult,
    SpatialOperation,
    StationaryFlickerMetrics,
    canonical_radial_power_profile,
    captured_geometry_issues,
    cira_stretch_reference,
    claim_boundary_issues,
    clean_ir_gray_reference,
    day_night_composite_reference,
    exact_area_resample,
    exact_area_response,
    geometry_matches_runtime_capture,
    inverse_cira_compatibility,
    legacy_m2_day_weight_from_sza,
    local_edge_ringing,
    local_neighbourhood_out_of_range_fraction,
    nearest_response,
    premultiplied_coverage_resample,
    primary_finalizer_provenance_issues,
    radial_high_frequency_power,
    rec709_proxy_high_frequency_power,
    repeatability_issues,
    run_m2_contract_oracle,
    scalar_field_before_palette_fixture,
    separable_resample,
    signed_bias_closed_form,
    source_coverage_from_responses,
    stationary_flicker_issues,
    unwindowed_proxy_high_frequency_power,
    weak_global_ringing,
)


_CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tat_truecolor_m2_validate.py"
_CLI_SPEC = importlib.util.spec_from_file_location("tat_truecolor_m2_validate_test", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
cli = importlib.util.module_from_spec(_CLI_SPEC)
sys.modules[_CLI_SPEC.name] = cli
_CLI_SPEC.loader.exec_module(cli)


def _captured_geometry() -> CapturedGeometry:
    return CapturedGeometry(
        schema=M2_TRACE_SCHEMA,
        capture_method=M2_CAPTURE_METHOD,
        captured_at_runtime=True,
        source_shape=(101, 137),
        internal_shape=(64, 79),
        delivered_internal_bounds=(1, 64, 0, 79),
        affine=(
            (0.5766423357664234, 0.0, -0.173),
            (0.0, -0.6237623762376238, 63.731),
            (0.0, 0.0, 1.0),
        ),
        sampling_phase=(0.43, 0.61),
        source_extent=(-84.1, -72.1, 10.3, 22.3),
        view_extent=(-83.9, -72.3, 10.5, 22.1),
    )


def _approved_operations(
    geometry: CapturedGeometry | None = None,
) -> tuple[SpatialOperation, ...]:
    captured = _captured_geometry() if geometry is None else geometry
    spatial = {
        "operation_kind": PHYSICAL_DELIVERY_OPERATION_KIND,
        "kernel": PHYSICAL_DELIVERY_KERNEL,
        "sampling_phase": captured.sampling_phase,
        "source_extent": captured.source_extent,
        "destination_extent": captured.view_extent,
        "coverage_transform": PHYSICAL_COVERAGE_TRANSFORM,
        "coverage_threshold": COVERAGE_TAU,
    }
    return (
        SpatialOperation(
            "day_delivery",
            "reflectance",
            True,
            input_semantics="corrected_reflectance",
            output_semantics="delivered_corrected_reflectance",
            **spatial,
        ),
        SpatialOperation(
            "ir_delivery",
            "kelvin",
            True,
            input_semantics="ir_bt_kelvin",
            output_semantics="delivered_ir_bt_kelvin",
            **spatial,
        ),
        SpatialOperation(
            "coverage_delivery",
            "linear_coverage",
            True,
            input_semantics="linear_coverage",
            output_semantics="delivered_linear_coverage",
            **spatial,
        ),
        SpatialOperation(
            "terminator_weight_delivery",
            "scalar_physical",
            True,
            input_semantics="source_day_weight_legacy_88_to_95",
            output_semantics="delivered_day_weight_legacy_88_to_95",
            **spatial,
        ),
        SpatialOperation(
            "clean_ir_gray",
            "kelvin",
            False,
            "pointwise",
            input_semantics="delivered_ir_bt_kelvin",
            output_semantics="clean_ir_gray",
        ),
        SpatialOperation(
            "inverse_cira_compatibility",
            "scalar_physical",
            False,
            "pointwise",
            input_semantics="clean_ir_gray",
            output_semantics="cira_preimage_compatibility_coordinate",
        ),
        SpatialOperation(
            "cira_stretch",
            "display_encoding",
            False,
            "pointwise",
            input_semantics="composite_cira_preimage",
            output_semantics="display_rgb",
        ),
        SpatialOperation(
            "display_glint_compat",
            "display",
            False,
            "pointwise",
            input_semantics="display_rgb",
            output_semantics="display_rgb",
        ),
    )


def _stationary_metrics(**overrides: object) -> StationaryFlickerMetrics:
    values: dict[str, object] = {
        "full_daylight": True,
        "stationary_patch": True,
        "patch_height": 96,
        "patch_width": 112,
        "luminance_mean": 0.4,
        "luminance_median": 0.5,
        "luminance_p1": 0.3,
        "luminance_p95": 1.2,
        "luminance_p999": 3.4,
        "laplacian_rms": 0.8,
        "red_green_ratio": 0.002,
        "blue_green_ratio": 0.008,
        "mean_saturation": 0.004,
        "black_clipped_fraction": 0.001,
        "white_clipped_fraction": 0.001,
    }
    values.update(overrides)
    return StationaryFlickerMetrics(**values)  # type: ignore[arg-type]


def _minimal_report() -> dict[str, object]:
    return {
        "schema": "tat_truecolor_m2_validation/v2",
        "summary": {
            "stage_advancement_passed": False,
            "canonical_blockers": ["V01"],
        },
        "repository": {
            "repository_root": "/repo/example",
            "branch": "fix/example",
            "head": "a" * 40,
            "git_directory": "/repo-meta/worktrees/example",
            "git_common_directory": "/repo-meta",
        },
        "source": {"source_sha256": {"render.py": "e" * 64}},
        "research_inputs": {"validation_gates_csv": {"sha256": "b" * 64}},
        "harness_source_sha256": {"validation/truecolor/m2.py": "c" * 64},
        "claim_boundary": {
            "claim_scope": "primary_imagery_layer_only_until_m5",
            "global_one_raster_claim": False,
        },
    }


def test_inverse_cira_sweep_and_continuous_endpoints() -> None:
    assert CIRA_LOG_ROOT_EXPECTED == -1.6516951369518393
    assert CIRA_DENOM_EXPECTED == 1.9887713527138795
    gray = np.linspace(0.0, 1.0, 10_001, dtype=np.float32)
    preimage = inverse_cira_compatibility(gray)

    assert preimage[0] == pytest.approx(CIRA_BLACK_PREIMAGE_EXPECTED, abs=1e-7)
    assert preimage[-1] == pytest.approx(CIRA_WHITE_PREIMAGE_EXPECTED, rel=1e-7)
    assert np.all(np.diff(preimage) > 0.0)
    assert np.max(np.abs(cira_stretch_reference(preimage) - gray)) <= 1e-6


def test_valid_zero_is_not_missing_or_numeric_zero() -> None:
    gray = np.asarray([0.0, 0.0, np.nan], dtype=np.float32)
    coverage = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    preimage = inverse_cira_compatibility(gray, coverage)

    assert preimage[0] == pytest.approx(0.0223, abs=1e-7)
    assert preimage[0] != 0.0
    assert np.isnan(preimage[1:]).all()
    assert cira_stretch_reference(preimage[:1])[0] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    ("bt_kelvin", "gray", "preimage"),
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
def test_clean_ir_canonical_fixtures(
    bt_kelvin: float, gray: float, preimage: float
) -> None:
    actual_gray = clean_ir_gray_reference(np.asarray([bt_kelvin]))
    actual_preimage = inverse_cira_compatibility(actual_gray)
    assert actual_gray[0] == pytest.approx(gray, abs=1e-7)
    assert actual_preimage[0] == pytest.approx(preimage, rel=2e-7)
    assert cira_stretch_reference(actual_preimage)[0] == pytest.approx(gray, abs=2e-7)


def test_ir_is_resampled_in_kelvin_before_gray_and_inverse() -> None:
    bt = np.asarray([[303.0, 183.0]], dtype=np.float32)
    delivered_bt = separable_resample(
        bt, np.asarray([[1.0]]), np.asarray([[0.5, 0.5]])
    )[0, 0]
    approved = cira_stretch_reference(
        inverse_cira_compatibility(clean_ir_gray_reference([delivered_bt]))
    )[0]
    inverse_first = inverse_cira_compatibility(np.asarray([0.0, 1.0]))
    mutant = cira_stretch_reference(np.asarray([np.mean(inverse_first)]))[0]

    assert delivered_bt == 243.0
    assert approved == pytest.approx(0.5, abs=2e-7)
    assert mutant == pytest.approx(0.8508647001182855, rel=2e-7)
    assert abs(float(mutant - approved)) > 0.3


def test_mixed_day_night_rejects_direct_gray_old_blend_and_j0_mutants() -> None:
    approved = day_night_composite_reference(
        np.full((1, 1, 3), 0.03, dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        np.asarray([[183.0]], dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        np.asarray([[0.75]], dtype=np.float32),
    ).display_rgb[0, 0, 0]
    direct_gray = cira_stretch_reference(np.asarray([0.75 * 0.03 + 0.25]))[0]
    old_display = 0.75 * cira_stretch_reference(np.asarray([0.03]))[0] + 0.25
    approved_zero = cira_stretch_reference(
        np.asarray([0.5 * 0.3 + 0.5 * 0.0223])
    )[0]
    zero_mutant = cira_stretch_reference(np.asarray([0.15]))[0]

    assert approved == pytest.approx(0.7061322214477085, rel=2e-7)
    assert direct_gray == pytest.approx(0.5465996088897274, rel=2e-6)
    assert old_display == pytest.approx(0.29857888445637004, rel=2e-6)
    assert approved_zero == pytest.approx(0.43188747948732364, rel=2e-7)
    assert zero_mutant == pytest.approx(0.41623004820434606, rel=2e-6)


def test_pure_day_and_pure_night_are_exact_appearance_endpoints() -> None:
    day = np.asarray([[[0.3, 0.4, 0.5]]], dtype=np.float32)
    ones = np.ones((1, 1), dtype=np.float32)
    pure_day = day_night_composite_reference(
        day, ones, np.asarray([[np.nan]]), np.zeros((1, 1)), ones
    )
    pure_night = day_night_composite_reference(
        np.full_like(day, np.nan),
        np.zeros((1, 1)),
        np.asarray([[243.0]]),
        ones,
        np.zeros((1, 1)),
    )

    assert pure_day.display_rgb == pytest.approx(cira_stretch_reference(day))
    assert pure_night.display_rgb == pytest.approx(np.full((1, 1, 3), 0.5), abs=2e-7)
    assert pure_day.effective_day_fraction[0, 0] == 1.0
    assert pure_night.effective_day_fraction[0, 0] == 0.0


def test_premultiplied_coverage_rejects_black_fill_nan_multiply_and_nearest_mask() -> None:
    values = np.asarray([[0.4, np.nan]], dtype=np.float32)
    coverage = np.asarray([[1.0, 0.0]], dtype=np.float32)
    y_response = np.asarray([[1.0]])
    x_response = np.asarray([[0.5, 0.5]])
    approved = premultiplied_coverage_resample(
        values, coverage, y_response, x_response
    )
    black_fill = separable_resample(
        np.nan_to_num(values, nan=0.0), y_response, x_response
    )[0, 0]
    nearest_mask = separable_resample(
        coverage, y_response, nearest_response(2, 1)
    )[0, 0]

    assert approved.values[0, 0] == pytest.approx(0.4, abs=1e-6)
    assert approved.coverage[0, 0] == pytest.approx(0.5, abs=1e-7)
    assert black_fill == pytest.approx(0.2)
    assert nearest_mask in {0.0, 1.0}
    assert nearest_mask != approved.coverage[0, 0]
    assert np.isnan(np.sum(values * coverage))


def test_exact_area_anisotropic_resample_preserves_dc_and_all_source_support() -> None:
    source = np.full((37, 53, 3), 0.271, dtype=np.float32)
    result = exact_area_resample(
        source, (23, 31), sampling_phase=(0.37, 0.63)
    )
    response_y = exact_area_response(37, 23, sampling_phase=0.37)
    response_x = exact_area_response(53, 31, sampling_phase=0.63)
    coverage = source_coverage_from_responses(response_y, response_x)

    assert result.shape == (23, 31, 3)
    assert result == pytest.approx(np.full((23, 31, 3), 0.271), abs=2e-8)
    assert coverage["discarded_rows"] == 0
    assert coverage["discarded_columns"] == 0


def test_nearest_response_discards_sources_while_convex_reference_does_not() -> None:
    approved = source_coverage_from_responses(
        exact_area_response(101, 63), exact_area_response(137, 79)
    )
    nearest = source_coverage_from_responses(
        nearest_response(101, 63), nearest_response(137, 79)
    )

    assert approved["discarded_rows"] == approved["discarded_columns"] == 0
    assert nearest["discarded_rows"] == 38
    assert nearest["discarded_columns"] == 58


def _spectral_fixture() -> np.ndarray:
    y, x = np.mgrid[:80, :96]
    red = 80 + 0.7 * x + 5 * np.sin(y / 6) + 9 * ((x + y) % 2)
    green = 60 + 0.2 * y + 13 * np.cos(x / 8) + 3 * ((2 * x + y) % 3)
    blue = 40 + 0.4 * x + 0.3 * y + 17 * (x % 5 == 0) - 11 * (y % 7 == 0)
    return np.stack([red, green, blue], axis=-1)


def test_v03_uses_canonical_windowed_rec601_radial_procedure() -> None:
    frequencies, power = canonical_radial_power_profile(_spectral_fixture())
    canonical = radial_high_frequency_power(_spectral_fixture())
    mutant = unwindowed_proxy_high_frequency_power(_spectral_fixture())
    rec709_mutant = rec709_proxy_high_frequency_power(_spectral_fixture())
    bin_mutant = radial_high_frequency_power(_spectral_fixture(), bins=63)

    assert frequencies.shape == power.shape == (64,)
    assert frequencies[0] == pytest.approx(0.00390625)
    assert frequencies[-1] == pytest.approx(0.49609375)
    assert power[0] == pytest.approx(5_577_898.367435859, rel=1e-6)
    assert power[38] == pytest.approx(348.86252836361695, rel=1e-6)
    assert canonical == pytest.approx(98_447.8327440082, rel=1e-6)
    assert rec709_mutant == pytest.approx(128_794.98847865824, rel=1e-6)
    assert bin_mutant == pytest.approx(91_844.7153028278, rel=1e-6)
    assert rec709_mutant != pytest.approx(canonical, rel=0.05)
    assert bin_mutant != pytest.approx(canonical, rel=0.05)
    assert mutant > canonical * 100.0


@pytest.mark.parametrize(
    ("contrast", "expected"),
    (
        (2.0, -0.05719095841793653),
        (5.0, -0.2546440075000701),
        (20.0, -0.57408229000004),
        (50.0, -0.7227032230640991),
    ),
)
def test_v14_closed_form_rejects_log_domain_bias(
    contrast: float, expected: float
) -> None:
    assert signed_bias_closed_form(contrast) == pytest.approx(expected, abs=1e-15)
    assert abs(signed_bias_closed_form(contrast)) > 0.005


def test_v15_uses_actual_local_support_not_global_extrema() -> None:
    source = np.full((7, 15), 100.0)
    source[:, 8:] = 146.1
    source[0, 0] = 0.0
    source[-1, -1] = 255.0
    candidate = source.copy()
    candidate[3, 11] = 150.0
    identity_y = np.eye(source.shape[0])
    identity_x = np.eye(source.shape[1])
    local = local_neighbourhood_out_of_range_fraction(
        source, candidate, identity_y, identity_x
    )
    global_proxy = np.mean(
        (candidate < source.min() - 1e-6) | (candidate > source.max() + 1e-6)
    )

    assert local > 0.0
    assert global_proxy == 0.0


def test_v16_internal_step_rejects_weak_global_ringing_proxy() -> None:
    edge = 32
    profile = np.full(64, 100.0)
    profile[edge:] = 146.1
    profile[0] = 0.0
    profile[-1] = 255.0
    profile[edge - 4] = 96.0
    profile[edge + 4] = 150.0

    local = local_edge_ringing(profile, edge)
    global_proxy = weak_global_ringing(profile, 0.0, 255.0)
    assert local.ringing_dn == pytest.approx(4.0)
    assert global_proxy == 0.0


def test_geometry_must_match_runtime_capture_including_phase_and_both_axes() -> None:
    actual = _captured_geometry()
    assert captured_geometry_issues(actual) == ()
    assert actual.delivered_shape == (63, 79)
    assert actual.anisotropic_ratios[0] != actual.anisotropic_ratios[1]
    assert geometry_matches_runtime_capture(actual, actual)

    hard_coded = dataclasses.replace(
        actual,
        capture_method="hard_coded_matplotlib_3_9_offset",
        captured_at_runtime=False,
    )
    translation_mutant = dataclasses.replace(
        actual,
        affine=(
            (actual.affine[0][0], 0.0, 0.0),
            actual.affine[1],
            actual.affine[2],
        ),
    )
    one_axis_ratio = dataclasses.replace(
        actual,
        affine=(
            actual.affine[0],
            (0.0, -actual.affine[0][0], actual.affine[1][2]),
            actual.affine[2],
        ),
    )
    assert captured_geometry_issues(hard_coded)
    assert not geometry_matches_runtime_capture(translation_mutant, actual)
    assert not geometry_matches_runtime_capture(one_axis_ratio, actual)


def test_geometry_parser_rejects_out_of_bounds_delivered_mapping() -> None:
    payload = _captured_geometry().as_dict()
    payload["delivered_internal_bounds"] = [1, 65, 0, 79]
    parsed = CapturedGeometry.from_mapping(payload)
    assert any("row mapping" in issue for issue in captured_geometry_issues(parsed))


def test_primary_provenance_binds_transform_metadata_and_ir_ordering() -> None:
    geometry = _captured_geometry()
    approved = _approved_operations(geometry)
    assert primary_finalizer_provenance_issues(
        approved,
        input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
        geometry=geometry,
    ) == ()
    assert primary_finalizer_provenance_issues(
        approved,
        input_semantics="display_encoded_rgb",
        geometry=geometry,
    )

    display_resize = (
        *approved[:-1],
        dataclasses.replace(
            approved[0],
            name="encoded_resize",
            domain="display",
            input_semantics="display_rgb",
            output_semantics="display_rgb",
        ),
        approved[-1],
    )
    glint_first = (approved[-1], *approved[:-1])
    assert primary_finalizer_provenance_issues(
        display_resize,
        input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
        geometry=geometry,
    )
    assert primary_finalizer_provenance_issues(
        glint_first,
        input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
        geometry=geometry,
    )

    mutants = (
        dataclasses.replace(approved[0], operation_kind="resize"),
        dataclasses.replace(approved[0], kernel="nearest"),
        dataclasses.replace(approved[0], sampling_phase=(0.5, 0.5)),
        dataclasses.replace(
            approved[0], destination_extent=(-80.0, -70.0, 12.0, 20.0)
        ),
        dataclasses.replace(approved[0], coverage_transform="nearest_mask"),
        dataclasses.replace(approved[0], coverage_threshold=1e-7),
    )
    for mutant in mutants:
        trace = (mutant, *approved[1:])
        assert primary_finalizer_provenance_issues(
            trace,
            input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
            geometry=geometry,
        )

    missing_weight_delivery = (*approved[:3], *approved[4:])
    mislabeled_weight_delivery = (
        *approved[:3],
        dataclasses.replace(
            approved[3], input_semantics="resampled_cos_sza_then_tapered"
        ),
        *approved[4:],
    )
    for trace in (missing_weight_delivery, mislabeled_weight_delivery):
        assert primary_finalizer_provenance_issues(
            trace,
            input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
            geometry=geometry,
        )

    inverse_before_kelvin = (
        approved[0],
        approved[5],
        approved[1],
        approved[2],
        approved[3],
        approved[4],
        approved[6],
        approved[7],
    )
    assert any(
        "physical ordering" in issue
        for issue in primary_finalizer_provenance_issues(
            inverse_before_kelvin,
            input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
            geometry=geometry,
        )
    )

    weight_after_gray = (
        *approved[:3],
        approved[4],
        approved[3],
        *approved[5:],
    )
    delivered_cosine_mutant = (
        *approved[:3],
        dataclasses.replace(
            approved[3],
            name="terminator_day_weight",
            input_semantics="delivered_cos_sza",
            output_semantics="delivered_day_weight_legacy_88_to_95",
        ),
        *approved[4:],
    )
    for trace in (weight_after_gray, delivered_cosine_mutant):
        assert primary_finalizer_provenance_issues(
            trace,
            input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
            geometry=geometry,
        )


def test_primary_provenance_rejects_label_only_trace_and_missing_geometry() -> None:
    approved = _approved_operations()
    label_only = tuple(
        SpatialOperation(op.name, op.domain, op.spatial, "label_only")
        for op in approved
    )
    assert primary_finalizer_provenance_issues(
        label_only,
        input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
        geometry=_captured_geometry(),
    )
    assert any(
        "runtime-captured geometry" in issue
        for issue in primary_finalizer_provenance_issues(
            approved,
            input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
        )
    )


def test_scalar_field_before_palette_is_discriminating() -> None:
    fixture = scalar_field_before_palette_fixture()
    assert fixture["delivered_kelvin"] == 243.0
    assert fixture["approved_rgb"] == pytest.approx([0.5, 0.25, np.sqrt(0.5)])
    assert fixture["palette_before_resample_rgb"] == pytest.approx([0.5, 0.5, 0.5])
    assert fixture["mutation_error"] > 0.2


def test_m2_retains_88_to_95_taper_without_m6_survival_gate() -> None:
    weights = legacy_m2_day_weight_from_sza([87.0, 88.0, 92.0, 95.0, np.nan])
    assert weights[:2] == pytest.approx([1.0, 1.0])
    assert 0.0 < weights[2] < 1.0
    assert weights[3:] == pytest.approx([0.0, 0.0])


def test_day_weight_is_tapered_on_source_then_spatially_delivered() -> None:
    source_sza = np.asarray([0.0, 96.0], dtype=np.float64)
    approved = float(legacy_m2_day_weight_from_sza(source_sza).mean())
    delivered_cosine = float(np.cos(np.deg2rad(source_sza)).mean())
    nonlinear_after_cosine_mutant = float(
        legacy_m2_day_weight_from_sza(
            [np.rad2deg(np.arccos(np.clip(delivered_cosine, -1.0, 1.0)))]
        )[0]
    )

    assert approved == pytest.approx(0.5)
    assert nonlinear_after_cosine_mutant == pytest.approx(1.0)
    assert nonlinear_after_cosine_mutant - approved > 0.4


def test_repeatability_hash_diagnostic_does_not_create_a_v25_binding() -> None:
    digest = "a" * 64
    assert repeatability_issues((digest, digest, digest)) == ()
    assert repeatability_issues((digest, digest))
    assert repeatability_issues((digest, "b" * 64, digest))
    assert repeatability_issues(("not-a-hash",) * 3)


def test_v30_rejects_wrong_scope_and_every_threshold_violation() -> None:
    assert stationary_flicker_issues(_stationary_metrics()) == ()
    assert stationary_flicker_issues(_stationary_metrics(stationary_patch=False))
    assert stationary_flicker_issues(_stationary_metrics(full_daylight=False))
    assert stationary_flicker_issues(_stationary_metrics(patch_width=95))
    assert stationary_flicker_issues(_stationary_metrics(luminance_median=1.01))
    assert stationary_flicker_issues(_stationary_metrics(blue_green_ratio=0.016))


def test_claim_boundary_rejects_m5_leakage_and_forbidden_coverage_passes() -> None:
    assert claim_boundary_issues(
        residuals=(),
        closed_gate_ids={"V22", "V23", "V44", "V45", "V46"},
        claims_global_one_raster=True,
    )


def test_contract_oracle_kills_every_registered_mutant_without_closing_m2() -> None:
    run = run_m2_contract_oracle()
    gates = {gate.gate_id: gate for gate in run.gates}

    assert not run.contract_passed
    assert not run.stage_advancement_passed
    assert set(run.canonical_blockers) <= M2_REQUIRED_GATE_IDS
    assert {mutation.mutation_id for mutation in run.mutations} == (
        M2_SYNTHETIC_MUTATION_IDS
    )
    assert len(run.mutations) == len(M2_SYNTHETIC_MUTATION_IDS)
    assert all(mutation.killed for mutation in run.mutations)
    assert {gate.gate_id for gate in run.gates} == M2_ALL_GATE_IDS
    assert not run.gate_registry_issues
    assert not run.mutation_registry_issues
    assert gates["V07"].outcome == gates["V08"].outcome == "PASS"
    assert gates["V07"].canonical_closure
    assert not gates["V08"].canonical_closure
    assert gates["V16"].outcome == "UNMEASURED"
    assert "3.66" not in gates["V16"].threshold
    assert gates["V16"].measurements["m1_runtime_baseline"] is None
    assert gates["V25"].outcome == gates["V30"].outcome == "UNMEASURED"
    for gate_id in FORBIDDEN_M2_CLOSURE_GATE_IDS:
        assert gates[gate_id].outcome == "UNMEASURED"
        assert not gates[gate_id].canonical_closure


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "unknown"))
def test_oracle_rejects_mutation_registry_deletion_duplicate_and_substitution(
    mutation: str,
) -> None:
    run = run_m2_contract_oracle()
    rows = list(run.mutations)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1] = dataclasses.replace(rows[-1], mutation_id="unknown_proxy")
    mutant = dataclasses.replace(run, mutations=tuple(rows))
    assert mutant.mutation_registry_issues
    assert not mutant.contract_passed
    assert not mutant.stage_advancement_passed


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "unknown"))
def test_oracle_rejects_gate_registry_deletion_duplicate_and_substitution(
    mutation: str,
) -> None:
    run = run_m2_contract_oracle()
    rows = list(run.gates)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1] = dataclasses.replace(rows[-1], gate_id="V99")
    mutant = dataclasses.replace(run, gates=tuple(rows))
    assert mutant.gate_registry_issues
    assert not mutant.contract_passed
    assert not mutant.stage_advancement_passed


def test_canonical_mutation_registry_is_exact_synthetic_plus_production() -> None:
    run = run_m2_contract_oracle()
    additions = tuple(
        MutationResult(identifier, True, "fixture", {})
        for identifier in sorted(M2_PRODUCTION_MUTATION_IDS)
    )
    canonical = dataclasses.replace(run, mutations=run.mutations + additions)
    assert {row.mutation_id for row in canonical.mutations} == (
        M2_CANONICAL_MUTATION_IDS
    )
    assert not canonical.mutation_registry_issues


def test_caller_supplied_v25_hashes_and_v30_metric_summaries_cannot_close() -> None:
    digest = "d" * 64
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_m2_contract_oracle(  # type: ignore[call-arg]
            renderer_repeatability_hashes=(digest, digest, digest)
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_m2_contract_oracle(  # type: ignore[call-arg]
            stationary_flicker=_stationary_metrics()
        )


def test_transactional_publisher_binds_and_hashes_all_payloads(tmp_path: Path) -> None:
    target = tmp_path / "scratch-m2-publisher"
    published = cli.publish_report_transactionally(_minimal_report(), target)

    assert published.directory == target
    assert set(path.name for path in target.iterdir()) == {
        "m2-validation.json",
        "m2-validation.json.sha256",
        "manifest.json",
        "manifest.json.sha256",
        "SHA256SUMS",
        "SHA256SUMS.sha256",
    }
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["evidence_class"] == "non_closing_contract_oracle"
    assert not manifest["stage_advancement_passed"]
    assert manifest["bound_source_sha256"] == {"render.py": "e" * 64}
    for line in (target / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert cli.sha256_file(target / name) == expected
    checksum_hash, checksum_name = (target / "SHA256SUMS.sha256").read_text().split()
    assert checksum_name == "SHA256SUMS"
    assert cli.sha256_file(target / checksum_name) == checksum_hash


def test_transactional_publisher_binds_native_gallery_payloads(tmp_path: Path) -> None:
    target = tmp_path / "m2-gallery-publisher"
    payload = b"exact native raster bytes"
    metadata = {
        "01_candidate.png": {
            "filename": "01_candidate.png",
            "sha256": cli.sha256_bytes(payload),
            "bytes": len(payload),
            "media_type": "image/png",
            "width": 1188,
            "height": 1188,
            "description": "fixture",
        }
    }
    report = _minimal_report()
    report["gallery_artifacts"] = metadata
    published = cli.publish_report_transactionally(
        report,
        target,
        additional_payloads={"01_candidate.png": payload},
        artifact_metadata=metadata,
    )
    assert published.directory == target
    assert (target / "01_candidate.png").read_bytes() == payload
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["artifacts"] == metadata
    indexed = {
        line.split("  ", 1)[1]
        for line in (target / "SHA256SUMS").read_text().splitlines()
    }
    assert "01_candidate.png" in indexed

    with pytest.raises(RuntimeError, match="metadata differs"):
        cli.publish_report_transactionally(
            report,
            tmp_path / "m2-gallery-tampered",
            additional_payloads={"01_candidate.png": payload + b"tampered"},
            artifact_metadata=metadata,
        )


def test_transactional_publisher_removes_private_stage_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "scratch-m2-failure"
    original = cli._write_new_file
    call_count = 0

    def fail_third_write(path: Path, payload: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise OSError("injected interrupted publication")
        original(path, payload)

    monkeypatch.setattr(cli, "_write_new_file", fail_third_write)
    with pytest.raises(OSError, match="injected"):
        cli.publish_report_transactionally(_minimal_report(), target)

    assert not os.path.lexists(target)
    assert not list(tmp_path.glob(".scratch-m2-failure.staging-*"))


def test_transactional_publisher_rechecks_bindings_before_atomic_rename(
    tmp_path: Path,
) -> None:
    target = tmp_path / "scratch-m2-binding-race"

    def changed_binding() -> None:
        raise RuntimeError("injected source identity change")

    with pytest.raises(RuntimeError, match="identity change"):
        cli.publish_report_transactionally(
            _minimal_report(), target, pre_publish_check=changed_binding
        )
    assert not os.path.lexists(target)
    assert not list(tmp_path.glob(".scratch-m2-binding-race.staging-*"))


@pytest.mark.parametrize("racing_leaf_kind", ("directory", "symlink"))
def test_transactional_publisher_atomically_refuses_concurrent_target_leaf(
    tmp_path: Path,
    racing_leaf_kind: str,
) -> None:
    target = tmp_path / f"scratch-m2-race-{racing_leaf_kind}"

    def create_racing_leaf() -> None:
        if racing_leaf_kind == "directory":
            target.mkdir()
        else:
            target.symlink_to(tmp_path / "attacker-destination")

    with pytest.raises(FileExistsError):
        cli.publish_report_transactionally(
            _minimal_report(),
            target,
            pre_publish_check=create_racing_leaf,
        )
    assert os.path.lexists(target)
    assert target.is_symlink() if racing_leaf_kind == "symlink" else target.is_dir()
    assert not list(tmp_path.glob(f".{target.name}.staging-*"))


def test_cli_preflight_refuses_existing_symlink_and_volatile_root(
    tmp_path: Path,
) -> None:
    symlink = tmp_path / "existing"
    symlink.symlink_to(tmp_path / "missing")
    arguments = argparse.Namespace(
        artifact_dir=symlink,
        research_root=tmp_path,
    )
    with pytest.raises(FileExistsError):
        cli.preflight_paths(arguments)

    volatile = argparse.Namespace(
        artifact_dir=tmp_path / "new-artifact",
        research_root=tmp_path,
    )
    with pytest.raises(cli.UnsafeArtifactPath):
        cli.preflight_paths(volatile)


def test_repository_identity_resolves_linked_worktree_pointer_portably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    linked = tmp_path / "linked"
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    (repository / "tracked.txt").write_text("identity fixture\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "add", "tracked.txt"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=M2 Validator Test",
            "-c",
            "user.email=m2-validator@example.invalid",
            "commit",
            "-m",
            "identity fixture",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "-b",
            "validator-linked",
            str(linked),
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(cli, "REPO_ROOT", linked)
    identity = cli.collect_repository_identity()
    assert identity["repository_root"] == str(linked)
    assert len(identity["head"]) == 40
    assert Path(identity["git_directory"]).is_dir()
    assert Path(identity["git_common_directory"]).is_dir()
    assert identity["git_pointer"]["dot_git_kind"] == "linked_worktree_pointer"


def test_m2_source_metadata_includes_new_resampler_and_all_authority_files() -> None:
    metadata = cli.collect_m2_source_metadata()
    hashes = metadata["source_sha256"]
    assert set(cli.M2_PRODUCT_SOURCE_FILES) <= hashes.keys()
    assert len(hashes["render_resample.py"]) == 64
    assert len(hashes["s2_pyramid.py"]) == 64
    assert len(hashes["s2_webmerc.py"]) == 64
    harness = cli.m2_harness_source_hashes()
    assert "validation/truecolor/report.py" in harness
    assert "validation/truecolor/m1.py" in harness
    assert "validation/truecolor/corpus.py" in harness
    assert "validation/truecolor/contracts.py" in harness
    assert "validation/truecolor/metrics.py" in harness
    assert "validation/truecolor/synthetics.py" in harness
    assert "tests/test_m1_hanning_policy.py" in harness
    assert "tests/test_m1_validation_policy_contract.py" in harness
    assert "tests/test_render_policy.py" in harness
    assert "tests/test_render_quality.py" in harness
    assert "tests/test_truecolor_gate_runner.py" in harness
    assert "tests/test_webp_frames.py" in harness
