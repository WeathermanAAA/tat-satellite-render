"""Production-bound, mutation-resistant M2 canonical adapter tests."""

from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import json
import os
import shutil
import types
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import render_resample
import truecolor
from satellites import FetchResult
from validation.truecolor import m2_production as adapter


def _physical_fetch(height: int = 96, width: int = 104) -> tuple[FetchResult, tuple[float, ...]]:
    bbox = (-80.0, 14.0, -68.0, 26.0)
    lon_axis = (
        (np.linspace(bbox[0], bbox[2], width) + 180.0) % 360.0 - 180.0
    ).astype(np.float32)
    lat_axis = np.linspace(bbox[3], bbox[1], height).astype(np.float32)
    lons = np.broadcast_to(lon_axis[None, :], (height, width)).copy()
    lats = np.broadcast_to(lat_axis[:, None], (height, width)).copy()
    y, x = np.mgrid[:height, :width]
    base = (0.20 + 0.20 * x / (width - 1) + 0.10 * y / (height - 1)).astype(
        np.float32
    )
    day = np.stack((base, 0.8 * base, 0.6 * base), axis=-1).astype(np.float32)
    cos_sza = np.full((height, width), np.float32(0.8), dtype=np.float32)
    coverage = np.ones((height, width), dtype=np.float32)
    geom = np.ones((height, width), dtype=np.bool_)
    state = truecolor.TrueColorPhysicalState(
        day_rgb_reflectance=day,
        day_coverage=coverage,
        ir_bt_kelvin=np.full((height, width), 250.0, dtype=np.float32),
        ir_coverage=coverage.copy(),
        cos_sza=cos_sza,
        day_weight=truecolor.terminator_day_weight(cos_sza).astype(np.float32),
        geom_valid=geom,
        red_corrected=base.copy(),
        veggie_corrected=np.full((height, width), 0.4, dtype=np.float32),
        glint_deg=np.full((height, width), 90.0, dtype=np.float32),
        sensor="abi",
        grid_metadata={
            "grid": "regular_latlon",
            "row_order": "north_to_south",
            "sensor": "abi",
            "point_stride_applied": False,
        },
        extent=bbox,
        projection="PlateCarree",
    )
    return (
        FetchResult(
            cmi=state.day_rgb_reflectance,
            lats=lats,
            lons=lons,
            channel=2,
            generic_channel="true_color",
            scan_start=dt.datetime(2025, 10, 27, 15, tzinfo=dt.timezone.utc),
            product="CMIPF",
            bucket="noaa-goes19",
            sat_name="GOES-19",
            sub_sat_lon=-75.2,
            units="truecolor_physical",
            cos_sza=state.cos_sza,
            geom_valid=state.geom_valid,
            truecolor_physical=state,
        ),
        bbox,
    )


@pytest.fixture(scope="module")
def production_capture():
    data, bbox = _physical_fetch()
    return adapter.capture_production_m2_render(data, bbox=bbox, dpi=30)


def test_affine_exact_area_uses_captured_translation_and_axis_sign() -> None:
    positive = adapter.exact_area_response_from_affine(
        4, 2, scale=0.5, translation=0.0
    ).response
    np.testing.assert_array_equal(
        positive,
        np.asarray([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]),
    )
    negative = adapter.exact_area_response_from_affine(
        4, 2, scale=-0.5, translation=2.0
    ).response
    np.testing.assert_array_equal(negative, positive[::-1])
    shifted = adapter.exact_area_response_from_affine(
        4, 2, scale=0.5, translation=0.25
    ).response
    assert not np.array_equal(shifted, positive)
    np.testing.assert_allclose(shifted.sum(axis=1), 1.0, atol=1e-12)


def test_sparse_reference_is_exact_and_not_nearest_proxy() -> None:
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    response = adapter.exact_area_response_from_affine(
        4, 2, scale=0.5, translation=0.0
    ).response
    observed = adapter._sparse_separable(values, response, response)
    np.testing.assert_allclose(observed, [[2.5, 4.5], [10.5, 12.5]])
    nearest = values[[1, 3]][:, [1, 3]]
    assert not np.array_equal(observed, nearest)


def test_live_capture_binds_real_plan_resamplers_finalizer_and_floor(
    production_capture,
) -> None:
    capture = production_capture
    assert capture.trace_issues == ()
    assert capture.policy_version == adapter.M2_REQUIRED_POLICY_VERSION
    assert [row.call_kind for row in capture.resample_calls] == [
        "day_bundle",
        "ir",
        "solar_geometry",
        "geom_coverage",
    ]
    assert {
        row.physical_domain_transport for row in capture.resample_calls
    } == {adapter.M2_REQUIRED_PHYSICAL_DOMAIN_TRANSPORT}
    solar = capture.resample_calls[2]
    assert [row["name"] for row in solar.field_records] == ["cos_sza", "day_weight"]
    assert [row.name for row in capture.pointwise_calls] == [
        "clean_ir_gray",
        "inverse_cira_compatibility",
        "background_inverse_cira_precomposition",
        "cira_stretch",
        "display_glint_compat",
    ]
    assert sum(
        row.name == "terminator_weight_delivery" for row in capture.operations
    ) == 1
    assert capture.endpoint_binding["passed"] is True
    assert capture.endpoint_binding["clean_ir_gray_call_count"] == 1
    assert capture.endpoint_binding["cira_stretch_call_count"] == 1
    assert capture.plan.source_extent != capture.bbox
    assert capture.plan.view_extent == pytest.approx(
        (capture.bbox[0], capture.bbox[2], capture.bbox[1], capture.bbox[3])
    )
    assert capture.plan.destination_shape == capture.plan.delivered_shape
    assert capture.plan.delivered_internal_bounds == (
        0,
        capture.plan.destination_height,
        0,
        capture.plan.destination_width,
    )
    assert capture.geometry.affine == capture.plan.affine_matrix
    assert capture.geometry.delivered_internal_bounds == (
        capture.plan.delivered_internal_bounds
    )
    np.testing.assert_array_equal(
        capture.figure_image_rgb, capture.delivered_floor_rgb
    )


def test_trace_rejects_raw_physical_values_at_the_agg_image_boundary(
    production_capture,
) -> None:
    capture = production_capture
    mutated_calls = list(capture.resample_calls)
    mutated_calls[0] = dataclasses.replace(
        mutated_calls[0], physical_domain_transport="raw_physical_float32"
    )
    _, issues = adapter._derive_operations(
        capture.source_state,
        capture.geometry,
        mutated_calls,
        capture.pointwise_calls,
    )
    assert "physical resampler used an unregistered domain transport" in issues


def test_capture_passes_validated_dynamic_title_to_real_renderer(monkeypatch) -> None:
    import render

    data, bbox = _physical_fetch(48, 52)
    original = render.render_png
    observed: list[str] = []

    def recording_renderer(*args, **kwargs):
        observed.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(render, "render_png", recording_renderer)
    title = "2025-10-28 14:00:54.9 UTC"
    capture = adapter.capture_production_m2_render(
        data, bbox=bbox, dpi=20, title_text=title
    )
    assert observed == [title]
    assert capture.title_text == title


@pytest.mark.parametrize(
    ("title", "error"),
    (
        (None, TypeError),
        ("", ValueError),
        ("   ", ValueError),
        ("2025-10-28\nforged", ValueError),
    ),
)
def test_capture_rejects_unbound_or_control_bearing_title(title, error) -> None:
    data, bbox = _physical_fetch(8, 8)
    with pytest.raises(error):
        adapter.capture_production_m2_render(
            data, bbox=bbox, dpi=20, title_text=title
        )


def test_live_responses_cover_every_source_and_kill_nearest(
    production_capture,
) -> None:
    hanning_y, hanning_x = adapter.measured_kernel_responses_for_plan(
        production_capture.plan
    )
    approved = adapter._source_coverage_summary(hanning_y, hanning_x)
    assert approved["discarded_rows"] == approved["discarded_columns"] == 0
    # This small live fixture is an upsample.  Exercise the canonical
    # downsample discriminator separately; nearest must discard source cells.
    nearest_y = adapter.m2.nearest_response(160, 119)
    nearest_x = adapter.m2.nearest_response(160, 119)
    mutant = adapter._source_coverage_summary(nearest_y, nearest_x)
    assert mutant["discarded_rows"] + mutant["discarded_columns"] > 0
    assert adapter._terminator_mutation_measurement(hanning_x)["killed"] is True


@pytest.mark.parametrize("axis", (0, 1))
def test_measured_axis_response_isolates_every_impulse_from_other_axis(axis) -> None:
    """Kill both square-diagonal and guarded-diagonal response proxies."""

    source_pixels = 9
    destination_pixels = 7
    if axis == 0:
        # Match the captured production buffer orientation: Agg receives a
        # negative vertical scale and a destination-edge translation.
        scale = -0.7785059348940109
        translation = 7.189312133408
    else:
        scale = 0.7429644471932647
        translation = -0.3714822235961037
    measured = adapter._agg_axis_response_from_transform(
        source_pixels,
        destination_pixels,
        "hanning",
        scale=scale,
        translation=translation,
        axis=axis,
    )

    agg_image, affine_type = render_resample._ensure_hanning_backend()
    reference = np.empty_like(measured)
    if axis == 0:
        impulse = np.zeros((source_pixels, 3), dtype=np.float32)
        result = np.zeros((destination_pixels, 3), dtype=np.float32)
        transform = affine_type.from_values(
            1.0, 0.0, 0.0, scale, 0.0, translation
        )
        for source_index in range(source_pixels):
            impulse[source_index, :] = np.float32(1.0)
            result.fill(np.float32(0.0))
            agg_image.resample(
                impulse,
                result,
                transform,
                agg_image.HANNING,
                True,
                1.0,
                True,
                4.0,
            )
            reference[:, source_index] = result[:, 1]
            impulse[source_index, :] = np.float32(0.0)

        guarded_source = np.zeros(
            (source_pixels, source_pixels + 2), dtype=np.float32
        )
        tagged = np.arange(source_pixels)
        guarded_source[tagged, tagged + 1] = np.float32(1.0)
        guarded_result = np.zeros(
            (destination_pixels, source_pixels + 2), dtype=np.float32
        )
        guarded_transform = affine_type.from_values(
            1.0, 0.0, 0.0, scale, 0.0, translation
        )
        guarded_view = guarded_result[:, 1:-1]
    else:
        impulse = np.zeros((3, source_pixels), dtype=np.float32)
        result = np.zeros((3, destination_pixels), dtype=np.float32)
        transform = affine_type.from_values(
            scale, 0.0, 0.0, 1.0, translation, 0.0
        )
        for source_index in range(source_pixels):
            impulse[:, source_index] = np.float32(1.0)
            result.fill(np.float32(0.0))
            agg_image.resample(
                impulse,
                result,
                transform,
                agg_image.HANNING,
                True,
                1.0,
                True,
                4.0,
            )
            reference[:, source_index] = result[1, :]
            impulse[:, source_index] = np.float32(0.0)

        guarded_source = np.zeros(
            (source_pixels + 2, source_pixels), dtype=np.float32
        )
        tagged = np.arange(source_pixels)
        guarded_source[tagged + 1, tagged] = np.float32(1.0)
        guarded_result = np.zeros(
            (source_pixels + 2, destination_pixels), dtype=np.float32
        )
        guarded_transform = affine_type.from_values(
            scale, 0.0, 0.0, 1.0, translation, 0.0
        )
        guarded_view = guarded_result[1:-1, :].T

    np.testing.assert_array_equal(measured, reference)

    agg_image.resample(
        guarded_source,
        guarded_result,
        guarded_transform,
        agg_image.HANNING,
        True,
        1.0,
        True,
        4.0,
    )
    guarded_error = float(np.max(np.abs(guarded_view - reference)))

    square_diagonal = adapter.m1.matplotlib_axis_response_from_transform(
        source_pixels,
        destination_pixels,
        "hanning",
        scale=scale,
        translation=translation,
        axis=axis,
    )
    square_error = float(np.max(np.abs(square_diagonal - reference)))
    # This phase-sensitive fixture reproduces the exact production dependency
    # ABI's diagonal coupling on both the negative-y and positive-x axes. Newer
    # Agg builds can happen to make those proxies numerically equal, so the
    # structural isolation test below remains load-bearing on every ABI.
    if matplotlib.__version__ == "3.9.3":
        assert guarded_error > 1.0e-6
        assert square_error > 1.0e-6


@pytest.mark.parametrize("axis", (0, 1))
def test_axis_response_invokes_one_isolated_stripe_per_source(
    monkeypatch,
    axis,
) -> None:
    class FakeAgg:
        HANNING = 5
        NEAREST = 0

        def __init__(self) -> None:
            self.sources = []

        def resample(self, source, result, *args) -> None:
            self.sources.append(np.asarray(source).copy())
            result.fill(np.float32(0.0))

    class FakeAffine:
        @staticmethod
        def from_values(*values):
            return values

    fake = FakeAgg()
    monkeypatch.setattr(
        render_resample,
        "_ensure_hanning_backend",
        lambda: (fake, FakeAffine),
    )
    source_pixels = 7
    measured = adapter._agg_axis_response_from_transform(
        source_pixels,
        5,
        "hanning",
        scale=5 / source_pixels,
        translation=0.19,
        axis=axis,
    )
    assert measured.shape == (5, source_pixels)
    assert len(fake.sources) == source_pixels
    for source_index, source in enumerate(fake.sources):
        assert np.count_nonzero(source) == 3
        if axis == 0:
            np.testing.assert_array_equal(
                np.flatnonzero(np.any(source != 0.0, axis=1)),
                np.asarray([source_index]),
            )
            np.testing.assert_array_equal(
                source[source_index], np.ones(3, dtype=np.float32)
            )
        else:
            np.testing.assert_array_equal(
                np.flatnonzero(np.any(source != 0.0, axis=0)),
                np.asarray([source_index]),
            )
            np.testing.assert_array_equal(
                source[:, source_index], np.ones(3, dtype=np.float32)
            )


def test_canonical_private_make_image_crop_is_a_discarded_row_mutant() -> None:
    """Preserve the exact pinned failure as a geometry discriminator."""

    source_shape = (1527, 1600)
    old_private_plan = render_resample.RasterPlan.from_affine(
        source_shape=source_shape,
        destination_shape=(1189, 1188),
        transform=(
            (0.7429644471932646, 0.0, -0.371482223596189),
            (0.0, -0.7791612057667102, 1189.3895806028831),
            (0.0, 0.0, 1.0),
        ),
        delivered_internal_bounds=(1, 1189, 0, 1188),
        internal_to_canvas_flip_y=True,
        internal_to_canvas_flip_x=False,
    )
    old_y, old_x = adapter.measured_kernel_responses_for_plan(old_private_plan)
    old_summary = adapter._source_coverage_summary(old_y, old_x)
    missing_rows = np.flatnonzero(
        ~np.any(np.abs(old_y) > adapter._RESPONSE_EPS, axis=0)
    )
    assert old_summary["discarded_rows"] == 1
    np.testing.assert_array_equal(missing_rows, np.asarray([0]))
    assert np.all(
        np.isfinite(
            old_private_plan.delivered_values(
                np.ones(old_private_plan.destination_shape, dtype=np.float32)
            )
        )
    )

    delivered_plan = render_resample.RasterPlan.from_affine(
        source_shape=source_shape,
        destination_shape=(1188, 1188),
        transform=(
            (0.7429644117659566, 0.0, -0.37145388175304106),
            (0.0, -0.778505897771953, 1188.1892529488864),
            (0.0, 0.0, 1.0),
        ),
        delivered_internal_bounds=(0, 1188, 0, 1188),
        internal_to_canvas_flip_y=True,
        internal_to_canvas_flip_x=False,
    )
    delivered_y, delivered_x = adapter.measured_kernel_responses_for_plan(
        delivered_plan
    )
    delivered_summary = adapter._source_coverage_summary(
        delivered_y, delivered_x
    )
    assert delivered_summary["discarded_rows"] == 0
    assert delivered_summary["discarded_columns"] == 0


def test_exact_area_reference_preserves_day_weight_as_a_spatial_scalar(
    production_capture,
) -> None:
    response_y, response_x = adapter.exact_area_responses_for_plan(
        production_capture.plan
    )
    state = adapter._reference_state_from_responses(
        production_capture.source_state, response_y, response_x
    )
    assert state.day_weight.shape == production_capture.plan.delivered_shape
    assert state.day_weight.dtype == np.float32
    assert np.all((state.day_weight >= 0.0) & (state.day_weight <= 1.0))
    assert state.grid_metadata["validation_reference"].startswith(
        "captured_affine_exact_area"
    )


def test_capture_rejects_nonlinear_terminator_reapplication(monkeypatch) -> None:
    data, bbox = _physical_fetch(48, 52)
    real_finalize = truecolor.finalize_truecolor_m2

    def mutant(state, **kwargs):
        truecolor.terminator_day_weight(state.cos_sza)
        return real_finalize(state, **kwargs)

    monkeypatch.setattr(truecolor, "finalize_truecolor_m2", mutant)
    with pytest.raises(RuntimeError, match="reapplied nonlinear terminator"):
        adapter.capture_production_m2_render(data, bbox=bbox, dpi=20)


def test_local_v15_rejects_mutant_hidden_inside_global_extrema() -> None:
    source = np.ones((3, 5), dtype=np.float32)
    source[0, 0] = 0.0
    source[-1, -1] = 2.0
    candidate = source.copy()
    candidate[1, 2] = 1.5
    identity_y = np.eye(3, dtype=np.float64)
    identity_x = np.eye(5, dtype=np.float64)
    fraction, _ = adapter._local_range_fraction(
        source,
        candidate,
        np.ones_like(source),
        identity_y,
        identity_x,
    )
    assert fraction > 0.0
    assert float(candidate.min()) >= float(source.min())
    assert float(candidate.max()) <= float(source.max())


def test_reference_pixels_execute_registered_m2_finalizer() -> None:
    result = adapter._reference_pixel_finalizer_measurements()
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["binding"].endswith("finalize_truecolor_m2")


def test_endpoint_identity_restoration_is_narrow_and_kills_no_restore() -> None:
    result = adapter._endpoint_restoration_measurements()
    assert result["passed"] is True
    assert result["pure_night_float_and_floor_identity"] is True
    assert result["zero_coverage_float_and_floor_identity"] is True
    assert result["fractional_night_retained_pre_cira_composite"] is True
    assert result["mixed_day_night_retained_pre_cira_composite"] is True
    assert result["clean_ir_gray_call_count"] == 1
    assert result["cira_stretch_call_count"] == 1
    assert result["no_endpoint_restore_mutant_killed"] is True


def test_endpoint_binding_masks_unsupported_nan_before_floor() -> None:
    shape = (1, 2)
    zero = np.zeros(shape, dtype=np.float32)
    state = truecolor.TrueColorPhysicalState(
        day_rgb_reflectance=np.full((*shape, 3), 0.2, dtype=np.float32),
        day_coverage=zero.copy(),
        ir_bt_kelvin=np.asarray([[255.0, np.nan]], dtype=np.float32),
        ir_coverage=np.asarray([[1.0, 0.0]], dtype=np.float32),
        cos_sza=zero.copy(),
        day_weight=zero.copy(),
        geom_valid=np.ones(shape, dtype=np.bool_),
        red_corrected=np.full(shape, 0.2, dtype=np.float32),
        veggie_corrected=np.full(shape, 0.2, dtype=np.float32),
        glint_deg=np.full(shape, 180.0, dtype=np.float32),
        sensor="abi",
        grid_metadata={"validation": "masked_nan_endpoint_floor_v1"},
    )
    background = (10.0 / 255.0, 13.0 / 255.0, 18.0 / 255.0)
    delivery = truecolor.finalize_truecolor_m2(
        state, background_display_rgb=background
    )
    records = (
        adapter.PointwiseCallRecord("clean_ir_gray", (), {"name": "clean_ir_gray"}),
        adapter.PointwiseCallRecord("cira_stretch", (), {"name": "cira_stretch"}),
    )

    with np.errstate(invalid="raise"):
        binding = adapter._captured_endpoint_binding(
            state,
            delivery,
            background_display_rgb=background,
            pointwise_calls=records,
        )

    assert np.all(np.isfinite(delivery.rgb))
    assert binding["pure_night_count"] == 1
    assert binding["zero_coverage_count"] == 1
    assert binding["pure_night_float_and_floor_identity"] is True
    assert binding["zero_coverage_float_and_floor_identity"] is True
    assert binding["passed"] is True


def _v41_fixture(monkeypatch):
    response_y = np.eye(1, dtype=np.float64)
    response_x = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.25, 0.5, 0.25, 0.0),
            (0.0, 0.25, 0.5, 0.25),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    nearest_y = np.eye(1, dtype=np.float64)
    nearest_x = np.eye(4, dtype=np.float64)
    monkeypatch.setattr(
        adapter,
        "measured_kernel_responses_for_plan",
        lambda plan, *, kernel="hanning": (nearest_y, nearest_x),
    )
    source_coverage = np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    replay = adapter._sparse_separable(
        source_coverage, response_y, response_x, epsilon=adapter._RESPONSE_EPS
    ).astype(np.float32)
    delivered_coverage = replay.copy()
    delivered_coverage[0, 1:3] += np.float32(5.0e-5)
    weight = np.full((1, 4), np.float32(0.5), dtype=np.float32)
    scientific = (weight * delivered_coverage).astype(np.float32)
    scientific += ((np.float32(1.0) - weight) * delivered_coverage).astype(
        np.float32
    )
    plan = types.SimpleNamespace(delivered_values=lambda values: np.asarray(values))
    source = types.SimpleNamespace(
        day_coverage=source_coverage,
        ir_coverage=source_coverage.copy(),
        geom_valid=np.asarray([[True, True, False, False]], dtype=np.bool_),
    )
    delivered = types.SimpleNamespace(
        day_coverage=delivered_coverage,
        ir_coverage=delivered_coverage.copy(),
        day_weight=weight,
    )
    delivery = types.SimpleNamespace(
        coverage=scientific,
        rgb=np.full((1, 4, 3), np.float32(0.1), dtype=np.float32),
        background_contract=truecolor.TrueColorBackgroundContract.OPAQUE_PRE_CIRA,
    )
    capture = types.SimpleNamespace(
        source_state=source,
        delivered_state=delivered,
        delivery=delivery,
        plan=plan,
        bbox=(-159.0, 38.8, -147.0, 50.8),
    )
    return capture, response_y, response_x


def test_v41_support_uses_exact_captured_formula_and_numeric_crosscheck(
    monkeypatch,
) -> None:
    capture, response_y, response_x = _v41_fixture(monkeypatch)
    result = adapter._v41_measurements(capture, response_y, response_x)
    assert result["passed"] is True
    assert result["exact_formula_array_identity"] is True
    assert result["exact_formula_max_abs_residual"] == 0.0
    assert result["delivery_support_escape_pixels"] == 0
    assert result["numeric_crosscheck"]["passed"] is True
    assert result["numeric_crosscheck"]["absolute_tolerance"] == 1.0e-4
    assert result["numeric_crosscheck"]["day_coverage_max_abs_error"] > 2.0e-6
    assert result["all_mutations_killed"] is True


@pytest.mark.parametrize(
    "mutation",
    ("output_rgb_proxy", "nearest_mask", "missing_visibility", "support_escape"),
)
def test_v41_rejects_support_and_visibility_mutants(monkeypatch, mutation) -> None:
    capture, response_y, response_x = _v41_fixture(monkeypatch)
    if mutation in {"output_rgb_proxy", "support_escape"}:
        # Opaque output is finite everywhere, so an RGB-finiteness proxy invents
        # support in the final source-unsupported pixel.
        capture.delivery.coverage = np.all(
            np.isfinite(capture.delivery.rgb), axis=-1
        ).astype(np.float32)
    elif mutation == "nearest_mask":
        nearest = capture.source_state.day_coverage.copy()
        capture.delivered_state.day_coverage = nearest
        capture.delivered_state.ir_coverage = nearest.copy()
        capture.delivery.coverage = nearest.copy()
    elif mutation == "missing_visibility":
        capture.source_state.geom_valid = np.ones_like(
            capture.source_state.geom_valid
        )
    result = adapter._v41_measurements(capture, response_y, response_x)
    assert result["passed"] is False
    if mutation in {"output_rgb_proxy", "support_escape"}:
        assert result["delivery_support_escape_pixels"] == 1
    elif mutation == "nearest_mask":
        assert (
            result["mutation_discriminators"]["nearest_source_mask"]["killed"]
            is False
        )
    else:
        assert result["source_has_valid_invalid_boundary"] is False


def test_verified_m1_baseline_binds_report_manifest_and_v16() -> None:
    root = Path(
        "/workspaces/tat-truecolor-review-artifacts/m1/renderer-1ef77fd-production-393"
    )
    if not root.is_dir():
        pytest.skip("durable M1 artifact is not mounted")
    baseline = adapter.verify_m1_artifact_baseline(root)
    assert baseline.scene_id == adapter.M2_SCENE_ID
    assert baseline.v15_out_of_range_fraction == 0.0
    assert baseline.v16_nearest_ringing_dn > baseline.v16_hanning_ringing_dn
    assert len(baseline.report_sha256) == 64
    assert baseline.source_shape == adapter.M1_TRUSTED_SOURCE_SHAPE
    assert baseline.delivered_shape == adapter.M1_TRUSTED_DELIVERED_SHAPE
    assert (
        baseline.source_display_sha256
        == adapter.M1_TRUSTED_SOURCE_DISPLAY_SHA256
    )
    assert (
        baseline.m0_nearest_raster_sha256
        == adapter.M1_TRUSTED_M0_NEAREST_RASTER_SHA256
    )
    assert baseline.m0_nearest_png_sha256 == adapter.M1_TRUSTED_M0_NEAREST_PNG_SHA256
    assert (
        baseline.m1_hanning_raster_sha256
        == adapter.M1_TRUSTED_M1_HANNING_RASTER_SHA256
    )
    assert baseline.m1_hanning_png_sha256 == adapter.M1_TRUSTED_M1_HANNING_PNG_SHA256
    assert tuple(row["role"] for row in baseline.object_identities) == (
        "red",
        "blue",
        "veggie",
        "ir",
    )


def _rollback_replay_fixture(monkeypatch):
    import render_policy

    source = np.linspace(0.0, 1.0, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    m0_raster = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    m1_raster = np.arange(18, 36, dtype=np.uint8).reshape(2, 3, 3)
    m0_png = b"current-m0-nearest-png"
    m1_png = b"current-m1-hanning-png"
    identity_objects = tuple(
        adapter.m1.CanonicalObjectIdentity(
            role=role,
            file_identifier=f"canonical-{role}.nc",
            bytes=100 + index,
            sha256=adapter._sha256_bytes(role.encode("ascii")),
        )
        for index, role in enumerate(adapter.m1.M1_REQUIRED_ROLES)
    )
    identity_rows = tuple(dataclasses.asdict(item) for item in identity_objects)
    baseline = adapter.M1ArtifactBaseline(
        directory="/immutable-m1-fixture",
        report_sha256="a" * 64,
        checksum_index_sha256="b" * 64,
        gallery_manifest_sha256="c" * 64,
        git_head="d" * 40,
        scene_id=adapter.M2_SCENE_ID,
        object_identities=identity_rows,
        source_shape=tuple(source.shape),
        source_display_sha256=adapter._sha256_bytes(source.tobytes()),
        delivered_shape=tuple(m0_raster.shape),
        m0_nearest_raster_sha256=adapter._sha256_bytes(m0_raster.tobytes()),
        m0_nearest_png_sha256=adapter._sha256_bytes(m0_png),
        m1_hanning_raster_sha256=adapter._sha256_bytes(m1_raster.tobytes()),
        m1_hanning_png_sha256=adapter._sha256_bytes(m1_png),
        m0_runtime_interpolations=("nearest",),
        m1_runtime_interpolations=("hanning",),
        agg_internal_shape=(2, 3, 4),
        agg_affine=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        delivered_internal_bounds=(0, 2, 0, 3),
        v15_out_of_range_fraction=0.0,
        v16_nearest_ringing_dn=4.0,
        v16_hanning_ringing_dn=3.0,
        v16_measurements={},
    )
    source_fetch = types.SimpleNamespace(cmi=source)
    mutate_m0 = {"enabled": False}

    def fake_load(research, corpus_root):
        return source_fetch, identity_objects

    def fake_render(data, policy):
        assert data is source_fetch
        if policy is render_policy.M0_POLICY:
            raster = m0_raster.copy()
            if mutate_m0["enabled"]:
                raster.flat[0] ^= np.uint8(1)
            return adapter.m1.RenderedRaster(
                png=m0_png,
                raster=raster,
                interpolations=("nearest",),
                agg_traces=(),
                make_image_traces=(),
            )
        assert policy is render_policy.M1_POLICY
        return adapter.m1.RenderedRaster(
            png=m1_png,
            raster=m1_raster.copy(),
            interpolations=("hanning",),
            agg_traces=(),
            make_image_traces=(),
        )

    monkeypatch.setattr(adapter.m1, "load_canonical_melissa_display_array", fake_load)
    monkeypatch.setattr(adapter.m1, "_render_canonical_raster", fake_render)
    return baseline, identity_rows, mutate_m0


def test_current_m0_m1_rollback_replays_bytes_and_real_changed_byte(
    monkeypatch,
) -> None:
    baseline, identity_rows, mutate_m0 = _rollback_replay_fixture(monkeypatch)
    result = adapter._measure_current_m0_m1_rollback(
        object(), "/unused-no-network-corpus", baseline
    )
    assert result["passed"] is True
    assert all(result["hash_matches"].values())
    changed = result["changed_byte_discriminator"]
    assert changed["mutation"] == "xor_0x01_at_one_retained_payload_byte"
    assert changed["byte_after"] == (changed["byte_before"] ^ 0x01)
    assert changed["approved_sha256"] == baseline.m1_hanning_raster_sha256
    assert changed["mutant_sha256"] != changed["approved_sha256"]
    assert changed["mutant_matches_expected"] is False
    assert changed["killed"] is True
    assert adapter._m0_m1_rollback_binding_issues(
        result, baseline, identity_rows
    ) == ()

    mutate_m0["enabled"] = True
    changed_current_output = adapter._measure_current_m0_m1_rollback(
        object(), "/unused-no-network-corpus", baseline
    )
    assert changed_current_output["passed"] is False
    assert (
        changed_current_output["hash_matches"]["m0_nearest_delivered_raster"]
        is False
    )
    assert changed_current_output["changed_byte_discriminator"]["killed"] is False


def test_current_m0_m1_rollback_binding_rejects_hash_and_boolean_forgery(
    monkeypatch,
) -> None:
    baseline, identity_rows, _ = _rollback_replay_fixture(monkeypatch)
    result = adapter._measure_current_m0_m1_rollback(
        object(), "/unused-no-network-corpus", baseline
    )
    forged = json.loads(json.dumps(result))
    forged["observed_sha256"]["m0_nearest_png"] = "f" * 64
    # Leave the copied positive booleans untouched: the binding checker must
    # derive the mismatch from hashes instead of accepting those assertions.
    issues = adapter._m0_m1_rollback_binding_issues(
        forged, baseline, identity_rows
    )
    assert any("hash-match verdicts were not re-derived" in issue for issue in issues)
    assert any("bytes differ from trusted M1" in issue for issue in issues)

    forged_changed_byte = json.loads(json.dumps(result))
    forged_changed_byte["changed_byte_discriminator"]["byte_after"] = (
        forged_changed_byte["changed_byte_discriminator"]["byte_before"]
    )
    assert any(
        "changed-byte verdicts were not re-derived" in issue
        or "replay did not pass" in issue
        for issue in adapter._m0_m1_rollback_binding_issues(
            forged_changed_byte, baseline, identity_rows
        )
    )


def test_canonical_measurement_stops_before_m2_when_rollback_replay_fails(
    monkeypatch,
) -> None:
    baseline, _, _ = _rollback_replay_fixture(monkeypatch)
    monkeypatch.setattr(
        adapter, "verify_m1_artifact_baseline", lambda directory: baseline
    )
    monkeypatch.setattr(adapter, "_require_m1_commit_available", lambda value: None)
    monkeypatch.setattr(
        adapter,
        "_measure_current_m0_m1_rollback",
        lambda *args, **kwargs: {"passed": False},
    )

    def forbidden_physical_load(*args, **kwargs):
        raise AssertionError("M2 physical measurement ran after rollback failure")

    monkeypatch.setattr(
        adapter, "load_canonical_melissa_physical", forbidden_physical_load
    )
    with pytest.raises(RuntimeError, match="M0/M1 real-corpus rollback replay"):
        adapter.measure_canonical_m2(
            object(),
            corpus_root="/unused-no-network-corpus",
            m1_artifact_directory="/unused-immutable-m1",
        )


def test_verified_m1_baseline_rejects_self_consistent_forgery(tmp_path) -> None:
    source = Path(
        "/workspaces/tat-truecolor-review-artifacts/m1/renderer-1ef77fd-production-393"
    )
    if not source.is_dir():
        pytest.skip("durable M1 artifact is not mounted")
    forged = tmp_path / "forged-m1"
    shutil.copytree(source, forged)

    report_path = forged / "m1-validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    v16 = next(row for row in report["gates"] if row["gate_id"] == "V16")
    v16["measurements"]["nearest_ringing_dn"] = 999.0
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report_sha = adapter._sha256_file(report_path)
    (forged / "m1-validation.json.sha256").write_text(
        f"{report_sha}  m1-validation.json\n", encoding="utf-8"
    )

    gallery_path = forged / "gallery_manifest.json"
    gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
    gallery["report"]["sha256"] = report_sha
    gallery_path.write_text(
        json.dumps(gallery, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    index_path = forged / "SHA256SUMS"
    index = adapter._parse_sha256sums(index_path)
    for name in (
        "m1-validation.json",
        "m1-validation.json.sha256",
        "gallery_manifest.json",
    ):
        index[name] = adapter._sha256_file(forged / name)
    index_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in index.items()),
        encoding="utf-8",
    )
    index_sha = adapter._sha256_file(index_path)
    (forged / "SHA256SUMS.sha256").write_text(
        f"{index_sha}  SHA256SUMS\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="trusted M1 closure identity"):
        adapter.verify_m1_artifact_baseline(forged)


def _stable_sequence() -> adapter.TimestampedSequenceBinding:
    daylight = np.ones((100, 100), dtype=np.bool_)
    base = dt.datetime(2025, 10, 27, 15, tzinfo=dt.timezone.utc)
    frames = []
    for index in range(5):
        raster = np.full((100, 100, 3), 90, dtype=np.uint8)
        raster[..., 1] = 100
        # Keep the measured 96x96 patch stationary while proving that each
        # complete renderer output has distinct bytes.
        raster[99, 99, 0] = np.uint8(index)
        frames.append(
            adapter.TimestampedRendererFrame(
                timestamp_utc=(base + dt.timedelta(minutes=10 * index))
                .isoformat()
                .replace("+00:00", "Z"),
                raster_rgb_u8=raster,
                full_daylight_mask=daylight.copy(),
                request_identity_sha256="1" * 64,
                grid_identity_sha256="2" * 64,
                source_object_sha256=tuple(
                    adapter._sha256_bytes(f"frame-{index}-role-{role}".encode())
                    for role in range(4)
                ),
                renderer_png_sha256=adapter._sha256_bytes(
                    f"png-{index}".encode()
                ),
                renderer_raster_sha256=adapter._array_sha256(raster),
            )
        )
    return adapter.TimestampedSequenceBinding(
        schema=adapter.M2_SEQUENCE_SCHEMA,
        frames=tuple(frames),
        patch_bounds=(0, 96, 0, 96),
        stationary_world_bounds=(-79.0, 15.0, -78.0, 16.0),
        stationary_class="clear_land",
        stationarity_method=adapter.M2_STATIONARITY_METHOD,
        daylight_mask_producer_binding=(
            adapter.M2_DAYLIGHT_MASK_PRODUCER_BINDING
        ),
    )


def test_v30_is_unmeasured_without_arrays_and_recomputed_when_bound() -> None:
    absent = adapter.evaluate_timestamped_sequence(None)
    assert absent.outcome == "UNMEASURED"
    bound = adapter.evaluate_timestamped_sequence(_stable_sequence())
    assert bound.outcome == "PASS"
    assert bound.metrics is not None
    assert bound.metrics.luminance_mean == 0.0


def test_v30_rejects_caller_hash_that_does_not_match_frame_bytes() -> None:
    binding = _stable_sequence()
    bad_first = dataclasses.replace(
        binding.frames[0], renderer_raster_sha256="f" * 64
    )
    bad = dataclasses.replace(binding, frames=(bad_first, *binding.frames[1:]))
    result = adapter.evaluate_timestamped_sequence(bad)
    assert result.outcome == "FAIL"
    assert any("does not match array bytes" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "request",
        "grid",
        "source_count",
        "source_reuse",
        "frozen_raster",
        "cadence",
        "stationary_class",
        "world_bounds",
        "daylight_producer",
    ),
)
def test_v30_rejects_unbound_sequence_mutations(mutation: str) -> None:
    binding = _stable_sequence()
    frames = list(binding.frames)
    if mutation == "schema":
        binding = dataclasses.replace(binding, schema="lookalike/v1")
    elif mutation == "request":
        frames[1] = dataclasses.replace(frames[1], request_identity_sha256="3" * 64)
        binding = dataclasses.replace(binding, frames=tuple(frames))
    elif mutation == "grid":
        frames[1] = dataclasses.replace(frames[1], grid_identity_sha256="4" * 64)
        binding = dataclasses.replace(binding, frames=tuple(frames))
    elif mutation == "source_count":
        frames[1] = dataclasses.replace(
            frames[1], source_object_sha256=frames[1].source_object_sha256[:3]
        )
        binding = dataclasses.replace(binding, frames=tuple(frames))
    elif mutation == "source_reuse":
        frames[1] = dataclasses.replace(
            frames[1], source_object_sha256=frames[0].source_object_sha256
        )
        binding = dataclasses.replace(binding, frames=tuple(frames))
    elif mutation == "frozen_raster":
        frozen = frames[0].raster_rgb_u8.copy()
        frozen_hash = adapter._array_sha256(frozen)
        frames = [
            dataclasses.replace(
                frame,
                raster_rgb_u8=frozen.copy(),
                renderer_raster_sha256=frozen_hash,
            )
            for frame in frames
        ]
        binding = dataclasses.replace(binding, frames=tuple(frames))
    elif mutation == "cadence":
        frames[1] = dataclasses.replace(
            frames[1], timestamp_utc="2025-10-27T15:10:01Z"
        )
        binding = dataclasses.replace(binding, frames=tuple(frames))
    elif mutation == "stationary_class":
        binding = dataclasses.replace(binding, stationary_class="clear_ocean")
    elif mutation == "world_bounds":
        binding = dataclasses.replace(
            binding, stationary_world_bounds=(-78.0, 15.0, -79.0, 16.0)
        )
    elif mutation == "daylight_producer":
        # The arrays are intentionally all ones; only the registered producer
        # binding distinguishes them from a caller assertion.
        binding = dataclasses.replace(
            binding, daylight_mask_producer_binding="caller_supplied_all_ones"
        )
    result = adapter.evaluate_timestamped_sequence(binding)
    assert result.outcome == "FAIL"
    assert result.issues


def _manifest(pid: int, ticks: int, child_hash: str) -> adapter.ProcessRenderManifest:
    return adapter.ProcessRenderManifest(
        pid=pid,
        parent_pid=os.getpid(),
        process_start_ticks=ticks,
        start_method="fork",
        python="3.11",
        platform="linux",
        numpy="1.26",
        matplotlib="3.9.3",
        pillow="11",
        policy_version=adapter.M2_REQUIRED_POLICY_VERSION,
        request_identity_sha256="a" * 64,
        output_bytes=3,
        child_output_sha256=child_hash,
        parent_recomputed_output_sha256=child_hash,
    )


def test_v25_hashes_are_recomputed_from_bytes_not_accepted_from_caller() -> None:
    payload = b"png"
    digest = adapter._sha256_bytes(payload)
    result = adapter.RepeatabilityCapture(
        output_bytes=(payload, payload, payload),
        manifests=(
            _manifest(11, 101, digest),
            _manifest(12, 102, digest),
            _manifest(13, 103, digest),
        ),
    )
    assert result.passed
    assert result.matches_primary_png(payload)
    assert not result.matches_primary_png(b"distinct-but-repeatable")
    with pytest.raises(TypeError, match="exact bytes"):
        result.matches_primary_png(bytearray(payload))  # type: ignore[arg-type]
    mutant = dataclasses.replace(
        result, output_bytes=(payload, b"changed", payload)
    )
    assert not mutant.passed
    assert "hashes" not in inspect.signature(
        adapter.capture_three_process_repeatability
    ).parameters
    discriminator = adapter._repeatability_mutation_discriminator(result)
    assert discriminator["baseline_passed"] is True
    assert discriminator["changed_byte_mutant_passed"] is False
    assert discriminator["caller_hash_parameter_present"] is False
    assert discriminator["killed"] is True


def test_endpoint_fixture_kills_no_restore_under_production_floor() -> None:
    measured = adapter._endpoint_restoration_measurements()
    assert measured["passed"] is True
    assert measured["background_floor_dn"] == [10, 13, 18]
    assert measured["no_restore_background_floor_mismatch_count"] == 3
    assert measured["no_endpoint_restore_mutant_killed"] is True


def test_reported_output_proxy_mutant_is_derived_from_v41_measurement(
    monkeypatch,
) -> None:
    capture, response_y, response_x = _v41_fixture(monkeypatch)
    v41 = adapter._v41_measurements(capture, response_y, response_x)
    measured = v41["mutation_discriminators"]["output_rgb_finite_proxy"]
    assert measured["difference_pixels"] > 0
    assert adapter._output_proxy_mutation_discriminator(v41) == measured

    unmeasured = adapter._output_proxy_mutation_discriminator({"passed": False})
    assert unmeasured["killed"] is False
    assert "unmeasured_reason" in unmeasured
