from __future__ import annotations

import numpy as np
import pytest

from validation.truecolor import production_adapter
from validation.truecolor.metrics import (
    ObservationCandidate,
    RayleighContractError,
    assert_rayleigh_startup_ready,
    cache_identity_digest,
    edge_ringing,
    evaluate_changed_observation,
    explicit_kernel_policy_passes,
    forbidden_sharpening_operators,
    fractional_area_resample_1d,
    high_frequency_power_ratio,
    linear_domain_pair_relative_bias,
    log_domain_pair_relative_bias,
    missing_cache_identity_fields,
    nearest_resample_1d,
    noise_variance_ratio,
    out_of_range_fraction,
    production_floor_quantize,
    rayleigh_frame_status_passes,
    rms_error,
    round_half_up_quantize,
    source_column_coverage,
)
from validation.truecolor.synthetics import (
    alias_stress_columns,
    deterministic_flat_noise,
    ringing_edge_mutant,
)


def test_nearest_1600_to_1188_discards_exactly_the_canonical_columns() -> None:
    coverage = source_column_coverage(1600, 1188)
    assert coverage.contributing_columns == 1188
    assert coverage.discarded_columns == 412
    assert coverage.discarded_fraction == pytest.approx(0.2575)
    assert coverage.contributor_steps == (1, 2)


def test_nearest_mutant_fails_alias_and_spectral_discriminators() -> None:
    source = alias_stress_columns()
    nearest = nearest_resample_1d(source, 1188)
    reference = fractional_area_resample_1d(source, 1188)

    # Convert the normalized synthetic to delivered DN before applying V01.
    assert rms_error(nearest * 255.0, reference * 255.0) > 1.5
    assert high_frequency_power_ratio(nearest, reference, cutoff=0.3) > 1.10


@pytest.mark.parametrize("automatic_name", ["auto", "antialiased", " AUTO "])
def test_automatic_interpolation_names_can_never_pass_v04(automatic_name: str) -> None:
    assert not explicit_kernel_policy_passes(automatic_name)
    assert not explicit_kernel_policy_passes(
        automatic_name,
        spatial_resample_completed_upstream=True,
        draw_is_one_to_one=True,
    )


def test_only_literal_hanning_or_explicit_upstream_one_to_one_passes_v04() -> None:
    assert explicit_kernel_policy_passes("hanning")
    assert not explicit_kernel_policy_passes("nearest")
    assert explicit_kernel_policy_passes(
        "nearest", spatial_resample_completed_upstream=True, draw_is_one_to_one=True
    )
    assert not explicit_kernel_policy_passes(
        "nearest", spatial_resample_completed_upstream=True, draw_is_one_to_one=False
    )


@pytest.mark.parametrize(
    ("contrast", "expected_pct"),
    [(2, -5.7191), (5, -25.4644), (20, -57.4081), (50, -72.2703)],
)
def test_log_domain_average_exposes_the_closed_form_dark_bias(
    contrast: float, expected_pct: float
) -> None:
    measured_pct = 100.0 * log_domain_pair_relative_bias(contrast)
    assert measured_pct == pytest.approx(expected_pct, abs=0.001)
    assert abs(measured_pct) > 0.5
    assert linear_domain_pair_relative_bias(contrast) == 0.0


def test_floor_and_round_half_up_quantizers_are_not_interchanged() -> None:
    values = np.asarray([0.8305103222137298, 0.5, 6.49 / 255.0])
    assert production_floor_quantize(values).tolist() == [211, 127, 6]
    assert round_half_up_quantize(values).tolist() == [212, 128, 6]


@pytest.mark.parametrize(
    "mutant_source",
    [
        "result = unsharp_mask(image)",
        "result = cv2.createCLAHE().apply(image)",
        "result = ImageFilter.UnsharpMask(radius=2)",
        "result = exposure.equalize_adapthist(image)",
        "result = local_tone_mapping(image)",
        "result = EdgeEnhance(image)",
        "result = sharpen(image)",
    ],
)
def test_forbidden_sharpening_mutants_are_detected(
    mutant_source: str, tmp_path
) -> None:
    assert forbidden_sharpening_operators(mutant_source)
    mutant_path = tmp_path / "render_mutant.py"
    mutant_path.write_text(mutant_source + "\n", encoding="utf-8")
    assert production_adapter.forbidden_operator_findings((mutant_path,))


def test_benign_band_limited_code_is_not_misclassified_as_sharpening() -> None:
    source = "result = fractional_area_resample_1d(reflectance, delivered_px)"
    assert forbidden_sharpening_operators(source) == ()


def test_overshooting_edge_mutant_fails_out_of_range_and_ringing_contracts() -> None:
    mutant = ringing_edge_mutant()
    assert out_of_range_fraction(mutant, 0.0, 1.0) == pytest.approx(2 / mutant.size)
    ringing = edge_ringing(mutant, 0.0, 1.0)
    assert ringing.undershoot == pytest.approx(0.08)
    assert ringing.overshoot == pytest.approx(0.11)
    assert ringing.amplitude == pytest.approx(0.11)


def test_nearest_does_not_overshoot_but_unsharp_noise_gain_is_rejected() -> None:
    source = alias_stress_columns()
    nearest = nearest_resample_1d(source, 1188)
    assert out_of_range_fraction(nearest, np.min(source), np.max(source)) == 0.0
    noise = deterministic_flat_noise()
    sharpened_mutant = 0.5 + 1.2 * (noise - 0.5)
    assert noise_variance_ratio(sharpened_mutant, noise) > 1.0


def _observation(**overrides: object) -> ObservationCandidate:
    values: dict[str, object] = {
        "required_bands": frozenset({"B01", "B02", "B03"}),
        "present_bands": frozenset({"B01", "B02", "B03"}),
        "fill_fraction": 0.8,
        "fill_floor": 0.05,
        "change_token": "scan-new",
        "previous_change_token": "scan-old",
        "http_status": 200,
        "all_black": False,
    }
    values.update(overrides)
    return ObservationCandidate(**values)  # type: ignore[arg-type]


def test_only_changed_complete_usable_observation_advances_liveness() -> None:
    assert evaluate_changed_observation(_observation()).qualifies

    black_200 = evaluate_changed_observation(
        _observation(fill_fraction=0.0, all_black=True)
    )
    assert not black_200.qualifies
    assert "insufficient_usable_fill" in black_200.reasons

    missing_himawari_b03 = evaluate_changed_observation(
        _observation(present_bands=frozenset({"B01", "B02"}))
    )
    assert not missing_himawari_b03.qualifies
    assert "required_band_missing" in missing_himawari_b03.reasons

    unchanged = evaluate_changed_observation(
        _observation(change_token="scan-old", previous_change_token="scan-old")
    )
    assert not unchanged.qualifies
    assert "observation_unchanged" in unchanged.reasons


@pytest.mark.parametrize("status", [401, 403, 500])
def test_unsuccessful_http_response_never_advances_liveness(status: int) -> None:
    decision = evaluate_changed_observation(_observation(http_status=status))
    assert not decision.qualifies
    assert "http_not_successful" in decision.reasons


def _valid_cache_identity() -> dict[str, object]:
    return {
        "bbox": [-84.1, 10.3, -72.1, 22.3],
        "ALGO_VERSION": "truecolor-m0-baseline",
        "recipe_sha256": "a" * 64,
        "delivered_px": 1188,
    }


@pytest.mark.parametrize("omitted", ["ALGO_VERSION", "recipe_sha256", "delivered_px"])
def test_each_required_cache_identity_omission_is_rejected(omitted: str) -> None:
    identity = _valid_cache_identity()
    identity.pop(omitted)
    assert missing_cache_identity_fields(identity) == {omitted}
    with pytest.raises(ValueError, match="missing required fields"):
        cache_identity_digest(identity)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("ALGO_VERSION", "truecolor-m1-hanning"),
        ("recipe_sha256", "b" * 64),
        ("delivered_px", 756),
    ],
)
def test_every_required_cache_identity_field_changes_the_digest(
    field: str, replacement: object
) -> None:
    baseline = _valid_cache_identity()
    changed = dict(baseline)
    changed[field] = replacement
    assert cache_identity_digest(changed) != cache_identity_digest(baseline)


def test_rayleigh_startup_calls_every_configured_sensor() -> None:
    configured = (
        ("GOES-19", "abi"),
        ("Himawari-9", "ahi"),
        ("GEO-KOMPSAT-2A", "ami"),
        ("Meteosat-12", "fci"),
    )
    calls: list[tuple[str, str]] = []

    def factory(platform: str, sensor: str) -> object:
        calls.append((platform, sensor))
        return object()

    correctors = assert_rayleigh_startup_ready(
        configured, factory, data_directory_writable=True
    )
    assert tuple(correctors) == configured
    assert tuple(calls) == configured


def test_rayleigh_none_is_aggregated_after_every_sensor_was_checked() -> None:
    configured = (("GOES-19", "abi"), ("Himawari-9", "ahi"))
    calls: list[tuple[str, str]] = []

    def factory(platform: str, sensor: str) -> object | None:
        calls.append((platform, sensor))
        return None if sensor == "abi" else object()

    with pytest.raises(RayleighContractError, match="GOES-19/abi: returned None"):
        assert_rayleigh_startup_ready(configured, factory, data_directory_writable=True)
    assert tuple(calls) == configured


def test_unwritable_rayleigh_directory_refuses_startup() -> None:
    with pytest.raises(RayleighContractError, match="not writable"):
        assert_rayleigh_startup_ready(
            (("GOES-19", "abi"),),
            lambda _platform, _sensor: object(),
            data_directory_writable=False,
        )


def test_required_rayleigh_frame_can_never_silently_skip() -> None:
    assert rayleigh_frame_status_passes(required=True, status="applied")
    assert not rayleigh_frame_status_passes(
        required=True, status="SKIPPED_LUT_UNAVAILABLE"
    )
    assert not rayleigh_frame_status_passes(required=True, status="not_required")
    assert rayleigh_frame_status_passes(required=False, status="not_required")
    assert not rayleigh_frame_status_passes(required=False, status="invented_status")
