"""Scientific metric and mutation contracts for the M1 stopgap."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from validation.truecolor.m1 import (
    AggResampleTrace,
    KernelCoverage,
    M1GateEvidence,
    M1Run,
    M1_REQUIRED_GATE_IDS,
    StaticPolicyAudit,
    V16EdgeSamplingGeometry,
    V16_CAPTURED_GEOMETRY_PROVENANCE,
    canonical_radial_power_profile,
    evaluate_m1_gates,
    find_internal_canvas_mapping,
    fractional_area_resample_axis,
    local_edge_overshoot,
    local_neighbourhood_out_of_range_fraction,
    internal_to_canvas_orientation,
    kernel_coverage_from_trace,
    matplotlib_axis_response,
    matplotlib_kernel_coverage,
    m1_harness_source_hashes,
    radial_high_frequency_power,
    run_synthetic_m1_discriminator,
    v16_edge_discriminator_passes,
    v16_edge_geometry_matches_canonical,
)


def _result_gate(
    gate_id: str,
    *,
    outcome: str = "PASS",
    blocking: bool = False,
) -> M1GateEvidence:
    return M1GateEvidence(
        gate_id=gate_id,
        name=gate_id,
        threshold="fixture",
        outcome=outcome,
        blocking=blocking,
        evidence_kind="fixture",
        detail="fixture",
        measurements={},
    )


def _result_run(gates: tuple[M1GateEvidence, ...]) -> M1Run:
    return M1Run(
        research=None,  # type: ignore[arg-type]
        static_policy=None,  # type: ignore[arg-type]
        runtime_traces=(),
        horizontal_coverage=None,  # type: ignore[arg-type]
        vertical_coverage=None,  # type: ignore[arg-type]
        synthetic=None,  # type: ignore[arg-type]
        canonical=None,
        rayleigh_startup=None,  # type: ignore[arg-type]
        gates=gates,
    )


def _spectral_fixture() -> np.ndarray:
    y, x = np.mgrid[:80, :96]
    red = 80 + 0.7 * x + 5 * np.sin(y / 6) + 9 * ((x + y) % 2)
    green = 60 + 0.2 * y + 13 * np.cos(x / 8) + 3 * ((2 * x + y) % 3)
    blue = 40 + 0.4 * x + 0.3 * y + 17 * (x % 5 == 0) - 11 * (y % 7 == 0)
    return np.stack([red, green, blue], axis=-1)


def _v16_canonical_stub(geometry: V16EdgeSamplingGeometry) -> SimpleNamespace:
    return SimpleNamespace(
        evidence_kind="canonical_real_pinned_scene",
        source_shape=(1527, geometry.source_pixels, 3),
        agg_internal_shape=(1189, geometry.internal_destination_pixels, 4),
        delivered_shape=(1188, geometry.delivered_pixels, 3),
        delivered_internal_pixel_bounds=(
            1,
            1189,
            geometry.delivered_internal_start,
            geometry.delivered_internal_stop,
        ),
        agg_transform_matrix=(
            (geometry.scale, 0.0, geometry.translation),
            (0.0, -0.7786510114559513, 1189.00007559459),
            (0.0, 0.0, 1.0),
        ),
        runtime_interpolations={
            "production_nearest": ("nearest",),
            "explicit_hanning": ("hanning",),
        },
    )


def test_canonical_v03_profile_is_pinned_to_research_conventions() -> None:
    fixture = _spectral_fixture()
    frequencies, power = canonical_radial_power_profile(fixture)

    assert frequencies.shape == power.shape == (64,)
    assert frequencies[0] == pytest.approx(0.00390625)
    assert frequencies[-1] == pytest.approx(0.49609375)
    # NumPy/FFT backends vary at roughly 2e-7 relative across the pinned
    # production and Codespace stacks.  1e-6 remains orders of magnitude
    # tighter than every convention mutant below.
    assert power[0] == pytest.approx(5_577_898.367435859, rel=1e-6)
    assert power[17] == pytest.approx(3_671.6539396587686, rel=1e-6)
    assert power[38] == pytest.approx(348.86252836361695, rel=1e-6)
    assert power[63] == pytest.approx(255.0203479808613, rel=1e-6)
    assert radial_high_frequency_power(fixture) == pytest.approx(
        98_447.8327440082,
        rel=1e-6,
    )


def test_v03_fixture_rejects_luma_window_radius_and_bin_mutants() -> None:
    fixture = _spectral_fixture()
    canonical = radial_high_frequency_power(fixture)
    size = min(fixture.shape[:2])

    rec709 = (
        0.2126 * fixture[..., 0]
        + 0.7152 * fixture[..., 1]
        + 0.0722 * fixture[..., 2]
    )
    rec709 -= rec709.mean()
    rec709 = rec709[:size, :size]
    window = np.hanning(size)
    rec709 *= window[:, None] * window[None, :]
    rec709_power = np.fft.fftshift(np.abs(np.fft.fft2(rec709)) ** 2)
    frequency = np.fft.fftshift(np.fft.fftfreq(size))
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    edges = np.linspace(0.0, 0.5, 65)
    indices = np.digitize(radius.ravel(), edges) - 1
    rec709_profile = np.asarray(
        [
            rec709_power.ravel()[indices == index].mean()
            for index in range(64)
        ]
    )
    centres = 0.5 * (edges[1:] + edges[:-1])
    rec709_high = float(rec709_profile[centres > 0.3].sum())

    assert rec709_high == pytest.approx(128_794.98847865824, rel=1e-6)
    assert rec709_high != pytest.approx(canonical, rel=0.05)
    assert radial_high_frequency_power(fixture, bins=63) == pytest.approx(
        91_844.7153028278,
        rel=1e-6,
    )
    assert radial_high_frequency_power(fixture, bins=63) != pytest.approx(
        canonical,
        rel=0.05,
    )

    # Removing the 2-D Hann window or including the 0.5..sqrt(.5) FFT corners
    # changes the measured high-frequency power by far more than numerical
    # tolerance; either mutation must be visible to this fixture.
    rec601 = (
        0.299 * fixture[..., 0]
        + 0.587 * fixture[..., 1]
        + 0.114 * fixture[..., 2]
    )
    rec601 -= rec601.mean()
    rec601 = rec601[:size, :size]
    unwindowed = np.fft.fftshift(np.abs(np.fft.fft2(rec601)) ** 2)
    raw_corner_sum = float(unwindowed[radius > 0.3].sum())
    assert raw_corner_sum > canonical * 100


def test_exact_area_reference_preserves_dc_at_fractional_device_phases() -> None:
    source = np.full(17, 7.25, dtype=np.float64)
    for phase in (0.3, 0.5, 0.8):
        result = fractional_area_resample_axis(
            source,
            11,
            axis=0,
            sampling_phase=phase,
        )
        assert result == pytest.approx(np.full(11, 7.25), abs=1e-12)


def test_canonical_edge_aligned_fractional_area_weights_are_exact() -> None:
    result = fractional_area_resample_axis(
        np.asarray([1.0, 3.0, 5.0, 7.0]),
        3,
        axis=0,
        sampling_phase=0.5,
    )
    assert result == pytest.approx([1.5, 4.0, 6.5], abs=1e-12)


def test_hanning_covers_all_1600_columns_while_nearest_discards_412() -> None:
    nearest = matplotlib_kernel_coverage(1600, 1188, "nearest")
    hanning = matplotlib_kernel_coverage(1600, 1188, "hanning")
    assert nearest.discarded_source_pixels == 412
    assert nearest.discarded_fraction == pytest.approx(0.2575)
    assert hanning.discarded_source_pixels == 0
    assert hanning.is_convex_full_coverage


def test_actual_phase_hanning_response_is_convex_and_normalized() -> None:
    response = matplotlib_axis_response(
        41,
        29,
        "hanning",
        sampling_phase=0.3,
    )
    assert response.shape == (29, 41)
    assert response.min() >= -2e-6
    assert np.max(np.abs(response.sum(axis=1) - 1.0)) <= 2e-6


def test_internal_canvas_mapping_requires_unique_all_channel_identity() -> None:
    y, x = np.mgrid[:7, :5]
    internal = np.stack(
        [17 * y + x, 31 * y + 2 * x, 47 * y + 3 * x],
        axis=-1,
    ).astype(np.uint8)
    delivered = internal[1:7].copy()
    mapping = find_internal_canvas_mapping(
        internal,
        delivered,
        boundary_exclusion_px=1,
    )
    assert mapping.pixel_bounds == (1, 7, 0, 5)
    assert mapping.full_identity == 1.0
    assert mapping.scored_identity == 1.0
    assert len(mapping.candidate_scores) == 2


def test_internal_orientation_follows_captured_affine_signs() -> None:
    trace = AggResampleTrace(
        interpolation="hanning",
        input_shape=(3, 2, 4),
        output_shape=(3, 2),
        transform_matrix=((1.0, 0.0, 0.0), (0.0, -1.0, 3.0), (0.0, 0.0, 1.0)),
    )
    internal = np.arange(6).reshape(3, 2)
    assert np.array_equal(internal_to_canvas_orientation(internal, trace), internal[::-1])


def test_trace_coverage_orients_negative_y_before_mapped_row_slice() -> None:
    # For this captured-style affine, raw internal rows [1:4] omit source row
    # 4, while top-down-oriented rows [1:4] cover all five source rows. This
    # assertion therefore fails if coverage slices before reversing y.
    trace = AggResampleTrace(
        interpolation="hanning",
        input_shape=(5, 5, 4),
        output_shape=(4, 4),
        transform_matrix=((0.8, 0.0, 0.0), (0.0, -0.8, 3.8), (0.0, 0.0, 1.0)),
    )
    horizontal, vertical = kernel_coverage_from_trace(
        trace,
        (5, 5),
        (1, 4, 0, 4),
    )
    assert horizontal.contributing_source_pixels == 5
    assert vertical.contributing_source_pixels == 5
    assert vertical.discarded_source_pixels == 0
    assert vertical.internal_destination_pixels == 4
    assert vertical.delivered_internal_start == 1
    assert vertical.delivered_internal_stop == 4


def test_v15_local_range_catches_negative_lobe_mutant_inside_global_range() -> None:
    source = np.zeros((2, 3, 3), dtype=np.float64)
    source[0, 0, :] = 0.4
    source[0, 1, :] = 0.6
    source[1, 2, :] = np.asarray([0.0, 0.5, 1.0])  # unrelated global range
    # A [-0.2, 1.2] negative-lobe response produces 0.64: globally legal,
    # locally above the only contributing neighbourhood's maximum 0.60.
    mutant = np.full((1, 1, 3), -0.2 * 0.4 + 1.2 * 0.6)
    assert np.all((mutant >= source.min()) & (mutant <= source.max()))
    fraction, comparisons = local_neighbourhood_out_of_range_fraction(
        source,
        mutant,
        [np.asarray([0])],
        [np.asarray([0, 1])],
        tolerance=2e-6,
    )
    assert comparisons == 3
    assert fraction == 1.0


def test_v16_uses_local_plateaus_and_fresh_deterministic_dn_fixture() -> None:
    evidence = run_synthetic_m1_discriminator()
    assert evidence.edge_sampling_geometry.provenance.endswith("_unbound")
    assert evidence.edge_fixture_nominal_step_dn == 46.1
    assert evidence.edge_fixture_noise_seed == 20251027
    assert evidence.edge_fixture_left_plateau_sigma_dn == pytest.approx(1.76)
    assert evidence.edge_fixture_right_plateau_sigma_dn == pytest.approx(1.76)
    assert evidence.nearest_edge_ringing_dn > 0.0
    assert evidence.hanning_edge_ringing_dn <= evidence.nearest_edge_ringing_dn
    # These are fresh measurements, not reverse-fit copies of the published
    # research profile's 3.66 and 2.60 DN values.
    assert evidence.nearest_edge_ringing_dn != pytest.approx(3.66, abs=0.01)
    assert evidence.hanning_edge_ringing_dn != pytest.approx(2.60, abs=0.01)


def test_v16_replays_nonzero_translation_and_delivered_column_offset() -> None:
    geometry = V16EdgeSamplingGeometry(
        provenance=V16_CAPTURED_GEOMETRY_PROVENANCE,
        axis="x",
        source_pixels=1600,
        internal_destination_pixels=1189,
        delivered_internal_start=1,
        delivered_internal_stop=1189,
        scale=1189 / 1600,
        translation=-0.125,
        baseline_interpolation="nearest",
        candidate_interpolation="hanning",
    )
    captured = run_synthetic_m1_discriminator(edge_sampling_geometry=geometry)
    ideal = run_synthetic_m1_discriminator()

    assert captured.edge_sampling_geometry == geometry
    assert captured.edge_fixture_edge_index == 593
    assert captured.edge_fixture_edge_index != ideal.edge_fixture_edge_index
    assert captured.hanning_edge_ringing_dn != pytest.approx(
        ideal.hanning_edge_ringing_dn,
        abs=1e-6,
    )


def test_v16_binding_rejects_ideal_affine_slice_and_cross_axis_mutants() -> None:
    geometry = V16EdgeSamplingGeometry(
        provenance=V16_CAPTURED_GEOMETRY_PROVENANCE,
        axis="x",
        source_pixels=1600,
        internal_destination_pixels=1189,
        delivered_internal_start=1,
        delivered_internal_stop=1189,
        scale=1189 / 1600,
        translation=-0.125,
        baseline_interpolation="nearest",
        candidate_interpolation="hanning",
    )
    canonical = _v16_canonical_stub(geometry)
    captured = run_synthetic_m1_discriminator(edge_sampling_geometry=geometry)
    ideal = run_synthetic_m1_discriminator()
    assert v16_edge_geometry_matches_canonical(captured, canonical)
    assert not v16_edge_geometry_matches_canonical(ideal, canonical)

    slice_mutant = dataclasses.replace(
        captured,
        edge_sampling_geometry=dataclasses.replace(
            geometry,
            delivered_internal_start=0,
            delivered_internal_stop=1188,
        ),
    )
    assert not v16_edge_geometry_matches_canonical(slice_mutant, canonical)

    translation_mutant = dataclasses.replace(
        captured,
        edge_sampling_geometry=dataclasses.replace(geometry, translation=0.0),
    )
    assert not v16_edge_geometry_matches_canonical(translation_mutant, canonical)

    scale_mutant = dataclasses.replace(
        captured,
        edge_sampling_geometry=dataclasses.replace(
            geometry,
            scale=geometry.delivered_pixels / geometry.source_pixels,
        ),
    )
    assert not v16_edge_geometry_matches_canonical(scale_mutant, canonical)

    cross_axis_mutant = _v16_canonical_stub(geometry)
    cross_axis_mutant.agg_transform_matrix = (
        (geometry.scale, 1e-12, geometry.translation),
        (0.0, -0.7786510114559513, 1189.00007559459),
        (0.0, 0.0, 1.0),
    )
    assert not v16_edge_geometry_matches_canonical(captured, cross_axis_mutant)


def test_v16_detects_local_ringing_that_stays_inside_unrelated_global_range() -> None:
    profile = np.full(80, 100.0)
    profile[40:] = 146.1
    profile[0] = 0.0
    profile[-1] = 255.0
    profile[45] = 148.0
    assert profile.min() == 0.0 and profile.max() == 255.0
    measured = local_edge_overshoot(profile, 40)
    assert measured.step_dn == pytest.approx(46.1)
    assert measured.overshoot_dn == pytest.approx(1.9)
    assert measured.ringing_dn == pytest.approx(1.9)


def test_v16_discriminator_rejects_candidate_ringing_above_measured_m0() -> None:
    geometry = V16EdgeSamplingGeometry(
        provenance=V16_CAPTURED_GEOMETRY_PROVENANCE,
        axis="x",
        source_pixels=1600,
        internal_destination_pixels=1188,
        delivered_internal_start=0,
        delivered_internal_stop=1188,
        scale=0.742500094413769,
        translation=-0.00015106202963011128,
        baseline_interpolation="nearest",
        candidate_interpolation="hanning",
    )
    synthetic = run_synthetic_m1_discriminator(edge_sampling_geometry=geometry)
    canonical = _v16_canonical_stub(geometry)
    assert v16_edge_discriminator_passes(synthetic, canonical)
    mutant = dataclasses.replace(
        synthetic,
        hanning_edge_ringing_dn=synthetic.nearest_edge_ringing_dn + 0.01,
    )
    assert not v16_edge_discriminator_passes(mutant, canonical)


def test_v16_gate_fails_closed_without_canonical_geometry() -> None:
    synthetic = run_synthetic_m1_discriminator()
    assert synthetic.hanning_edge_ringing_dn <= synthetic.nearest_edge_ringing_dn
    coverage = KernelCoverage(
        kernel="hanning",
        source_pixels=1600,
        destination_pixels=1188,
        contributing_source_pixels=1600,
        discarded_source_pixels=0,
        discarded_fraction=0.0,
        minimum_weight=0.0,
        maximum_weight=1.0,
        maximum_weight_sum_error=0.0,
    )
    gates = evaluate_m1_gates(
        static_policy=StaticPolicyAudit((), 2, True, True, True, ()),
        runtime_traces=(),
        horizontal_coverage=coverage,
        vertical_coverage=coverage,
        synthetic=synthetic,
        canonical=None,
    )
    v16 = next(gate for gate in gates if gate.gate_id == "V16")
    assert v16.outcome == "FAIL"
    assert v16.blocking
    assert not v16.measurements["captured_binding_matches_canonical"]


def test_synthetic_discriminator_cannot_bless_canonical_v01_v02_v03() -> None:
    synthetic = run_synthetic_m1_discriminator()
    coverage = KernelCoverage(
        kernel="hanning",
        source_pixels=1600,
        destination_pixels=1188,
        contributing_source_pixels=1600,
        discarded_source_pixels=0,
        discarded_fraction=0.0,
        minimum_weight=0.0,
        maximum_weight=1.0,
        maximum_weight_sum_error=0.0,
    )
    static = StaticPolicyAudit((), 2, True, True, True, ())
    gates = evaluate_m1_gates(
        static_policy=static,
        runtime_traces=(),
        horizontal_coverage=coverage,
        vertical_coverage=coverage,
        synthetic=synthetic,
        canonical=None,
    )
    outcomes = {gate.gate_id: gate.outcome for gate in gates}
    assert outcomes["V01"] == "UNMEASURED"
    assert outcomes["V02"] == "UNMEASURED"
    assert outcomes["V03"] == "UNMEASURED"


def test_m1_stage_and_threshold_truth_require_the_exact_gate_id_set() -> None:
    gates = tuple(
        _result_gate(
            gate_id,
            outcome="FAIL" if gate_id == "V01" else "PASS",
            blocking=False,
        )
        for gate_id in sorted(M1_REQUIRED_GATE_IDS)
    )
    partial = _result_run(gates)
    assert partial.gate_id_contract_valid
    assert partial.stage_advancement_passed
    assert not partial.all_thresholds_passed
    assert partial.threshold_fail_gate_ids == ("V01",)
    assert partial.unmeasured_gate_ids == ()

    missing = _result_run(gates[:-1])
    assert not missing.gate_id_contract_valid
    assert not missing.stage_advancement_passed
    assert not missing.all_thresholds_passed

    duplicated = _result_run(gates[:-1] + (gates[0],))
    assert not duplicated.gate_id_contract_valid
    assert not duplicated.stage_advancement_passed


def test_m1_harness_hashes_cover_every_declared_local_generator() -> None:
    hashes = m1_harness_source_hashes()
    required = {
        "scripts/tat_truecolor_m1_validate.py",
        "validation/truecolor/m1.py",
        "validation/truecolor/m1_gallery.py",
        "validation/truecolor/report.py",
        "validation/truecolor/production_adapter.py",
        "validation/truecolor/gates.py",
        "validation/truecolor/contracts.py",
        "validation/truecolor/metrics.py",
        "validation/truecolor/synthetics.py",
        "validation/truecolor/corpus.py",
        "validation/truecolor/reference_pixels.py",
    }
    assert required <= hashes.keys()
    assert all(len(digest) == 64 for digest in hashes.values())
