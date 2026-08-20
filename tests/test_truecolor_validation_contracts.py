from __future__ import annotations

from collections import Counter

import pytest

from validation.truecolor.contracts import (
    BaselineStatus,
    C1_GEOMETRY,
    C1_RETIRED_VALUES,
    C3_DECLARED_QUANTIZERS,
    C3_EIGHT_BIT_TOLERANCE_DN,
    C3_FLOAT_TOLERANCE,
    C3_RECOMMENDED_QUANTIZER,
    C13_FLICKER_SCOPE,
    C13_LOOSER_DEFINITION_RESIDUAL_DN,
    C13_TERMINATOR_DAY_DEAD_MAX,
    C13_TERMINATOR_DEFINITION,
    C18_PER_FRAME_STATUS_LOG_REQUIRED,
    C18_RAYLEIGH_STATUSES,
    C18_STARTUP_CALLS_EVERY_CONFIGURED_SENSOR,
    C19_REGISTRATION_LIMIT_PX,
    C19_SELF_SHARPEN_DISABLE_PREREQUISITES,
    C20_REFERENCE_CONVENTION,
    GATES,
    GateOutcome,
    GateSeverity,
    GateSpec,
    gate_spec,
    research_baseline_result,
    result_for_measurement,
    validate_gate_catalog,
)


def test_gate_catalog_is_exactly_v01_through_v46() -> None:
    expected = [f"V{number:02d}" for number in range(1, 47)]
    actual = [spec.gate_id for spec in GATES]
    assert len(actual) == 46
    assert len(set(actual)) == 46
    assert actual == expected
    validate_gate_catalog()


def test_duplicate_or_missing_gate_ids_are_rejected() -> None:
    duplicate = (*GATES[:-1], GATES[-2])
    with pytest.raises(ValueError, match="unique"):
        validate_gate_catalog(duplicate)

    with pytest.raises(ValueError, match="exactly V01 through V46"):
        validate_gate_catalog(GATES[:-1])


def test_v43_is_the_only_soft_gate_and_v46_is_unmeasured_but_blocking() -> None:
    assert [gate.gate_id for gate in GATES if gate.severity is GateSeverity.SOFT] == ["V43"]
    assert gate_spec("V43").baseline_status is BaselineStatus.FAIL
    assert research_baseline_result("V43").outcome is GateOutcome.SOFT_WARNING
    assert not research_baseline_result("V43").blocking

    assert gate_spec("V46").severity is GateSeverity.HARD
    assert gate_spec("V46").baseline_status is BaselineStatus.UNMEASURED
    assert research_baseline_result("V46").outcome is GateOutcome.UNMEASURED
    assert research_baseline_result("V46").blocking


def test_skipped_and_unmeasured_hard_gates_never_become_passes() -> None:
    result = result_for_measurement("V01", None, "corpus absent")
    assert result.outcome is GateOutcome.UNMEASURED
    assert result.blocking

    skipped = type(result)(spec=gate_spec("V01"), outcome=GateOutcome.SKIPPED)
    assert skipped.blocking


def test_canonical_baseline_status_counts_follow_the_csv_semantics() -> None:
    counts = Counter(spec.baseline_status for spec in GATES)
    assert counts == {
        BaselineStatus.PASS: 12,
        BaselineStatus.FAIL: 33,
        BaselineStatus.UNMEASURED: 1,
    }
    assert sum(research_baseline_result(spec.gate_id).blocking for spec in GATES) == 33


def test_c1_uses_corrected_geometry_and_discards_exactly_412_columns() -> None:
    assert (C1_GEOMETRY.low_px, C1_GEOMETRY.default_px, C1_GEOMETRY.high_px) == (
        756,
        1188,
        2160,
    )
    assert C1_GEOMETRY.capped_source_columns == 1600
    assert C1_GEOMETRY.default_discarded_columns_under_nearest == 412
    assert C1_GEOMETRY.default_discarded_fraction == pytest.approx(0.2575)
    assert not ({756, 1188, 2160} & C1_RETIRED_VALUES)
    assert {757, 1189, 1214.4, 2162} <= C1_RETIRED_VALUES


def test_c3_keeps_floor_and_round_half_up_reference_columns_separate() -> None:
    assert C3_RECOMMENDED_QUANTIZER == "round_half_up"
    assert C3_FLOAT_TOLERANCE == 1e-6
    assert C3_EIGHT_BIT_TOLERANCE_DN == 0
    assert C3_DECLARED_QUANTIZERS["production_floor"] == {
        "P1": "#D0CFCE",
        "P2": "#000006",
        "P3": "#193F10",
        "P4": "#838689",
        "P5": "#195710",
    }
    assert C3_DECLARED_QUANTIZERS["round_half_up"] == {
        "P1": "#D0D0CF",
        "P2": "#000006",
        "P3": "#193F11",
        "P4": "#84868A",
        "P5": "#195711",
    }
    assert sum(
        C3_DECLARED_QUANTIZERS["production_floor"][pixel]
        != C3_DECLARED_QUANTIZERS["round_half_up"][pixel]
        for pixel in C3_DECLARED_QUANTIZERS["production_floor"]
    ) == 4


def test_c13_pins_flicker_scope_and_terminator_definition() -> None:
    assert C13_FLICKER_SCOPE.full_daylight_only
    assert C13_FLICKER_SCOPE.stationary_patch_only
    assert (
        C13_FLICKER_SCOPE.minimum_patch_width_px,
        C13_FLICKER_SCOPE.minimum_patch_height_px,
    ) == (96, 96)
    assert C13_TERMINATOR_DEFINITION == "day_rgb_black_clipped_to_zero"
    assert C13_TERMINATOR_DAY_DEAD_MAX == pytest.approx(1 / 255)
    assert C13_LOOSER_DEFINITION_RESIDUAL_DN == pytest.approx(12.81)


def test_c18_requires_every_sensor_startup_check_and_per_frame_status() -> None:
    assert C18_STARTUP_CALLS_EVERY_CONFIGURED_SENSOR
    assert C18_PER_FRAME_STATUS_LOG_REQUIRED
    assert C18_RAYLEIGH_STATUSES == {
        "applied",
        "not_required",
        "SKIPPED_LUT_UNAVAILABLE",
    }


def test_c19_makes_v46_a_precondition_for_m4_self_sharpen_disable() -> None:
    assert C19_SELF_SHARPEN_DISABLE_PREREQUISITES == {"V46"}
    assert C19_REGISTRATION_LIMIT_PX == pytest.approx(0.25)
    assert research_baseline_result("V46").blocking


def test_c20_keeps_every_alias_percentage_attached_to_its_reference() -> None:
    convention = C20_REFERENCE_CONVENTION
    assert convention.name == "same_1600_pixel_intermediate"
    assert convention.source_columns == 1600
    assert convention.cannot_score_stages == ("S02", "S03", "S04")
    assert convention.quoted_unweighted_reduction_pct == pytest.approx(81.9)
    assert convention.native_crop_kernel_reduction_pct == pytest.approx((47.7, 52.0))


def test_catalog_validator_rejects_a_second_soft_gate() -> None:
    replacement = GateSpec(
        gate_id="V42",
        name=gate_spec("V42").name,
        threshold=gate_spec("V42").threshold,
        baseline_status=gate_spec("V42").baseline_status,
        severity=GateSeverity.SOFT,
    )
    mutated = tuple(replacement if gate.gate_id == "V42" else gate for gate in GATES)
    with pytest.raises(ValueError, match="sole soft"):
        validate_gate_catalog(mutated)
