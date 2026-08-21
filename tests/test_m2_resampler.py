"""Mutation-resistant contracts for M2 physical Hanning delivery."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
from matplotlib import _image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D

import render_resample as rr


def _plan(
    source_shape=(9, 13),
    destination_shape=(6, 8),
    *,
    translation=(0.13, -0.12),
) -> rr.RasterPlan:
    source_height, source_width = source_shape
    destination_height, destination_width = destination_shape
    transform = (
        Affine2D()
        .scale(
            destination_width / source_width,
            destination_height / source_height,
        )
        .translate(*translation)
    )
    return rr.RasterPlan.from_affine(
        source_shape=source_shape,
        destination_shape=destination_shape,
        transform=transform,
        source_extent=(-84.1, -72.1, 10.3, 22.3),
        view_extent=(-83.7, -72.4, 10.8, 21.9),
    )


def test_planning_image_uses_literal_hanning_and_rejects_override() -> None:
    class AxesSpy:
        def __init__(self) -> None:
            self.calls = []

        def imshow(self, values, **kwargs):
            self.calls.append((values, kwargs))
            return "planned-image"

    axes = AxesSpy()
    values = np.ones((3, 4), dtype=np.float32)
    assert rr.make_hanning_planning_image(axes, values, origin="upper") == (
        "planned-image"
    )
    assert axes.calls == [
        (values, {"interpolation": "hanning", "origin": "upper"})
    ]
    with pytest.raises(TypeError, match="fixed to literal hanning"):
        rr.make_hanning_planning_image(axes, values, interpolation="nearest")


def test_raster_plan_is_frozen_and_keeps_exact_anisotropic_geometry() -> None:
    source_shape = (1499, 1600)
    destination_shape = (913, 1188)
    transform = Affine2D.from_values(
        0.742500094413769,
        0.0,
        0.0,
        -0.609072715143429,
        -0.00015106202963011128,
        913.00007559459,
    )
    plan = rr.RasterPlan.from_affine(
        source_shape=source_shape,
        destination_shape=destination_shape,
        transform=transform,
        source_extent=(-84.1, -72.1, 10.3, 22.3),
        view_extent=(-83.8, -72.4, 10.7, 21.8),
        device_bounds=(66.0, 53.0, 1254.0, 966.0),
    )

    assert plan.source_shape == source_shape
    assert plan.destination_shape == destination_shape
    assert plan.x_transform == pytest.approx(
        (0.742500094413769, -0.00015106202963011128)
    )
    assert plan.y_transform == pytest.approx((-0.609072715143429, 913.00007559459))
    assert plan.x_source_pixels_per_destination == pytest.approx(1600 / 1188)
    assert plan.y_source_pixels_per_destination == pytest.approx(1499 / 913)
    assert plan.x_source_pixels_per_destination != pytest.approx(
        plan.y_source_pixels_per_destination
    )
    assert plan.source_extent != plan.view_extent
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.destination_width = 913


def test_equal_dimensions_are_not_mistaken_for_identity_when_phase_differs() -> None:
    identity = rr.RasterPlan.from_affine(
        source_shape=(7, 11),
        destination_shape=(7, 11),
        transform=Affine2D(),
    )
    shifted = rr.RasterPlan.from_affine(
        source_shape=(7, 11),
        destination_shape=(7, 11),
        transform=Affine2D().translate(0.51, 0.0),
    )
    assert identity.is_identity_mapping
    assert not shifted.is_identity_mapping
    assert identity.sampling_phase_yx == pytest.approx((0.5, 0.5))
    assert shifted.sampling_phase_yx != identity.sampling_phase_yx


def test_float32_hanning_preserves_anisotropic_constant_dc_and_headroom() -> None:
    # Keep this headroom/DC contract independent of the separately tested
    # fractional-edge coverage behavior, which legitimately differs across
    # supported Agg ABIs at a translated canvas boundary.
    plan = _plan(translation=(0.0, 0.0))
    source = np.full(plan.source_shape, np.float32(2.5), dtype=np.float32)
    coverage = np.ones(plan.source_shape, dtype=np.float32)
    observed, observed_coverage = rr.resample_field(source, coverage, plan)
    assert observed.dtype == np.float32
    assert observed_coverage.dtype == np.float32
    assert np.allclose(observed, 2.5, rtol=0.0, atol=2.0e-6)
    assert np.allclose(observed_coverage, 1.0, rtol=0.0, atol=2.0e-6)
    assert float(observed.max()) > 1.0  # physical values were not display-clipped


def test_physical_domain_transport_kills_unit_interval_clipping_backend(
    monkeypatch,
) -> None:
    """The Agg boundary sees coordinates, never raw Kelvin/angles/signs."""

    shape = (3, 5)
    plan = rr.RasterPlan.from_affine(
        source_shape=shape,
        destination_shape=shape,
        transform=Affine2D().translate(0.25, -0.25),
    )
    coverage = np.asarray(
        (
            (1.0, 0.75, 0.5, 0.25, 1.0),
            (0.4, 1.0, 0.8, 0.6, 0.2),
            (1.0, 0.3, 0.7, 0.9, 1.0),
        ),
        dtype=np.float32,
    )
    physical = np.linspace(-40.0, 250.0, np.prod(shape), dtype=np.float32).reshape(
        shape
    )
    backend_inputs: list[np.ndarray] = []

    def clipping_identity(values, selected_plan):
        assert selected_plan is plan
        array = np.asarray(values, dtype=np.float32)
        backend_inputs.append(array.copy())
        # Reproduce the old pinned ABI's destructive image-domain clamp. The
        # approved transport makes this mutation observationally irrelevant.
        return np.clip(array, 0.0, 1.0).astype(np.float32)

    monkeypatch.setattr(rr, "_hanning_resample", clipping_identity)
    observed, observed_coverage = rr.resample_field(
        physical, coverage, plan
    )

    np.testing.assert_array_equal(observed_coverage, coverage)
    np.testing.assert_allclose(observed, physical, rtol=0.0, atol=2.0e-5)
    assert len(backend_inputs) == 2
    assert all(
        np.all(np.isfinite(values))
        and float(values.min()) >= 0.0
        and float(values.max()) <= 1.0
        for values in backend_inputs
    )
    assert rr.PHYSICAL_DOMAIN_TRANSPORT == (
        "coverage_bound_affine_unit_interval_v1"
    )


def test_physical_domain_transport_is_affine_equivariant() -> None:
    plan = _plan()
    coverage = np.linspace(
        0.1, 1.0, np.prod(plan.source_shape), dtype=np.float32
    ).reshape(plan.source_shape)
    base = np.linspace(
        0.2, 0.8, np.prod(plan.source_shape), dtype=np.float32
    ).reshape(plan.source_shape)
    physical = (-64.0 + 256.0 * base).astype(np.float32)

    delivered_base, base_coverage = rr.resample_field(base, coverage, plan)
    delivered_physical, physical_coverage = rr.resample_field(
        physical, coverage, plan
    )

    np.testing.assert_array_equal(physical_coverage, base_coverage)
    np.testing.assert_allclose(
        delivered_physical,
        -64.0 + 256.0 * delivered_base,
        rtol=0.0,
        atol=2.0e-5,
    )


def test_literal_hanning_kills_nearest_checkerboard_mutant() -> None:
    plan = _plan(translation=(0.0, 0.0))
    source = (np.indices(plan.source_shape).sum(axis=0) % 2).astype(np.float32)
    observed, _ = rr.resample_field(
        source, np.ones(plan.source_shape, dtype=np.float32), plan
    )
    assert np.all(observed > 0.25)
    assert np.all(observed < 0.75)
    assert np.unique(observed).size > 2

    bilinear = np.zeros(plan.destination_shape, dtype=np.float32)
    _image.resample(
        np.ascontiguousarray(source),
        bilinear,
        Affine2D(np.asarray(plan.affine_matrix, dtype=np.float64)),
        _image.BILINEAR,
        True,
        1.0,
        True,
        4.0,
    )
    # Shape, phase and DC assertions alone cannot distinguish the smoother
    # literal-Hanning policy from a scale-unaware bilinear substitution.
    assert float(np.max(np.abs(observed - bilinear))) > 1.0e-2


def test_private_backend_mapping_drift_fails_closed(monkeypatch) -> None:
    from matplotlib import _image
    import matplotlib.image as mpl_image

    plan = _plan(translation=(0.0, 0.0))
    monkeypatch.setitem(mpl_image._interpd_, "hanning", _image.NEAREST)
    with pytest.raises(rr.HanningBackendError, match="mapping has changed"):
        rr.resample_field(
            np.ones(plan.source_shape, dtype=np.float32),
            np.ones(plan.source_shape, dtype=np.float32),
            plan,
        )


def test_phase_and_independent_y_transform_are_load_bearing() -> None:
    source = np.linspace(0.0, 1.0, 9 * 13, dtype=np.float32).reshape(9, 13)
    coverage = np.ones_like(source)
    actual = _plan(translation=(0.13, -0.12))
    phase_mutant = _plan(translation=(0.37, 0.24))
    isotropic_mutant = rr.RasterPlan.from_affine(
        source_shape=actual.source_shape,
        destination_shape=actual.destination_shape,
        transform=Affine2D().scale(
            actual.x_transform[0], actual.x_transform[0]
        ),
        source_extent=actual.source_extent,
        view_extent=actual.view_extent,
    )
    expected, _ = rr.resample_field(source, coverage, actual)
    shifted, _ = rr.resample_field(source, coverage, phase_mutant)
    wrong_axis, _ = rr.resample_field(source, coverage, isotropic_mutant)
    assert not np.allclose(expected, shifted, rtol=0.0, atol=1.0e-7)
    assert not np.allclose(expected, wrong_axis, rtol=0.0, atol=1.0e-7)


def test_premultiplied_coverage_prevents_nan_or_black_halo() -> None:
    plan = _plan()
    coverage = np.zeros(plan.source_shape, dtype=np.float32)
    coverage[:, 4:10] = 1.0
    source = np.full(plan.source_shape, np.nan, dtype=np.float32)
    source[coverage > 0.0] = 7.0

    observed, observed_coverage = rr.resample_field(source, coverage, plan)
    supported = observed_coverage > rr.DEFAULT_COVERAGE_THRESHOLD
    assert np.any((observed_coverage > 0.0) & (observed_coverage < 1.0))
    assert np.allclose(observed[supported], 7.0, rtol=0.0, atol=1.0e-6)
    assert np.all(np.isnan(observed[~supported]))

    # The caught black-fill mutant filters an opaque zero outside support.
    black_filled = np.where(coverage > 0.0, np.float32(7.0), np.float32(0.0))
    mutant, _ = rr.resample_field(
        black_filled, np.ones_like(coverage), plan
    )
    fractional = (observed_coverage > 0.0) & (observed_coverage < 1.0)
    assert np.any(mutant[fractional] < 6.0)


def test_rejected_sparse_hanning_tail_is_canonical_exact_zero() -> None:
    """A real fractional phase must not leave coverage without a value."""

    source_shape = (101, 137)
    destination_shape = (6, 8)
    plan = rr.RasterPlan.from_affine(
        source_shape=source_shape,
        destination_shape=destination_shape,
        transform=(
            Affine2D()
            .scale(
                destination_shape[1] / source_shape[1],
                destination_shape[0] / source_shape[0],
            )
            .translate(-0.85, -0.65)
        ),
    )
    coverage = np.zeros(source_shape, dtype=np.float32)
    coverage[50, 68] = 1.0
    field = np.full(source_shape, np.nan, dtype=np.float32)
    field[50, 68] = 7.0

    # A deliberately much smaller threshold exposes the positive numerical
    # tail in this fixture and proves the default assertion is load-bearing.
    _, exposed = rr.resample_field(
        field,
        coverage,
        plan,
        coverage_threshold=1.0e-12,
    )
    weak = (exposed > 0.0) & (exposed <= rr.DEFAULT_COVERAGE_THRESHOLD)
    assert np.any(weak)

    observed, effective = rr.resample_field(field, coverage, plan)
    assert np.all(effective[weak] == 0.0)
    assert np.all(np.isnan(observed[weak]))
    assert np.all(
        (effective == 0.0) | (effective > rr.DEFAULT_COVERAGE_THRESHOLD)
    )


def test_valid_zero_is_data_and_coverage_is_not_nearest_resized() -> None:
    plan = _plan()
    coverage = np.zeros(plan.source_shape, dtype=np.float32)
    coverage[:, :7] = 1.0
    source = np.zeros(plan.source_shape, dtype=np.float32)
    observed, observed_coverage = rr.resample_field(source, coverage, plan)
    supported = observed_coverage > rr.DEFAULT_COVERAGE_THRESHOLD
    assert np.all(observed[supported] == 0.0)
    assert np.any((observed_coverage > 0.0) & (observed_coverage < 1.0))


def test_rgb_uses_one_shared_coverage_transform_without_clipping() -> None:
    # Fractional boundary coverage is exercised separately; this contract is
    # specifically about retaining signed RGB headroom through the physical
    # transport on every supported Agg ABI.
    plan = _plan(translation=(0.0, 0.0))
    coverage = np.ones(plan.source_shape, dtype=np.float32)
    base = np.linspace(-0.2, 1.4, np.prod(plan.source_shape), dtype=np.float32)
    base = base.reshape(plan.source_shape)
    rgb = np.stack((base, base + 1.0, base - 1.0), axis=-1).astype(np.float32)
    observed, observed_coverage = rr.resample_rgb(rgb, coverage, plan)
    assert observed.shape == (*plan.destination_shape, 3)
    assert observed.dtype == np.float32
    assert np.allclose(observed[..., 1] - observed[..., 0], 1.0, atol=2.0e-6)
    assert np.allclose(observed[..., 0] - observed[..., 2], 1.0, atol=2.0e-6)
    assert float(observed.min()) < 0.0
    assert float(observed.max()) > 1.0
    assert np.allclose(observed_coverage, 1.0, atol=2.0e-6)


def test_rgb_rejects_masked_array_lookalike_before_resampling() -> None:
    plan = _plan()
    coverage = np.ones(plan.source_shape, dtype=np.float32)
    backing = np.ones((*plan.source_shape, 3), dtype=np.float32)
    masked = np.ma.array(backing, mask=np.zeros(backing.shape, dtype=bool))

    with pytest.raises(TypeError, match="exact plain numpy.ndarray"):
        rr.resample_rgb(masked, coverage, plan)


def test_bundle_computes_one_load_bearing_coverage_transform(monkeypatch) -> None:
    plan = _plan()
    coverage = np.ones(plan.source_shape, dtype=np.float32)
    coverage[:, :2] = 0.0
    scalar = np.full(plan.source_shape, np.float32(7.0), dtype=np.float32)
    rgb = np.stack((scalar, scalar + 1.0, scalar + 2.0), axis=-1)
    calls = []
    original = rr._hanning_resample

    def traced(values, selected_plan):
        calls.append(values)
        return original(values, selected_plan)

    monkeypatch.setattr(rr, "_hanning_resample", traced)
    (observed_rgb, observed_scalar), observed_coverage = (
        rr.resample_fields_shared_coverage((rgb, scalar), coverage, plan)
    )

    # One coverage plus four numerators.  A proxy implementation that invokes
    # the scalar/RGB wrappers separately performs an extra coverage transform.
    assert len(calls) == 5
    np.testing.assert_allclose(
        observed_rgb[..., 0], observed_scalar, rtol=0.0, atol=2.0e-6
    )
    supported = observed_coverage > 0.0
    assert np.all(observed_scalar[supported] == pytest.approx(7.0, abs=2e-6))


@pytest.mark.parametrize("bad_fields", [[], [np.ones((9, 13), np.float32)]])
def test_bundle_rejects_non_tuple_field_lookalikes(bad_fields) -> None:
    with pytest.raises(TypeError, match="exact tuple"):
        rr.resample_fields_shared_coverage(
            bad_fields,
            np.ones((9, 13), np.float32),
            _plan(),
        )


def test_float32_shape_and_validity_contracts_fail_closed() -> None:
    plan = _plan()
    values = np.ones(plan.source_shape, dtype=np.float32)
    coverage = np.ones(plan.source_shape, dtype=np.float32)
    with pytest.raises(TypeError, match="float32 physical"):
        rr.resample_field(values.astype(np.float64), coverage, plan)
    with pytest.raises(TypeError, match="coverage"):
        rr.resample_field(values, coverage.astype(np.float64), plan)
    with pytest.raises(ValueError, match="coverage shape"):
        rr.resample_field(values, coverage[:-1], plan)
    bad_coverage = coverage.copy()
    bad_coverage[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rr.resample_field(values, bad_coverage, plan)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite wherever coverage is positive"):
        rr.resample_field(values, coverage, plan)


@pytest.mark.parametrize("masked_name", ["field", "coverage"])
def test_resampler_rejects_masked_array_lookalikes(masked_name) -> None:
    plan = _plan()
    values = np.ones(plan.source_shape, dtype=np.float32)
    coverage = np.ones(plan.source_shape, dtype=np.float32)
    masked = np.ma.array(
        values if masked_name == "field" else coverage,
        mask=np.zeros(plan.source_shape, dtype=np.bool_),
    )
    if masked_name == "field":
        values = masked
    else:
        coverage = masked
    with pytest.raises(TypeError, match="plain numpy.ndarray"):
        rr.resample_field(values, coverage, plan)


def test_axes_image_plan_uses_real_make_image_affine_and_dynamic_slice() -> None:
    figure = Figure(figsize=(3.0, 3.0), dpi=20)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.101, 0.103, 0.799, 0.797])
    source = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    image = axes.imshow(
        source,
        origin="upper",
        interpolation="hanning",
        extent=(-4.0, 3.0, 10.0, 15.0),
    )
    axes.set_xlim(-3.5, 2.5)
    axes.set_ylim(10.4, 14.6)
    canvas.draw()
    renderer = canvas.get_renderer()

    unsliced = rr.RasterPlan.from_axes_image(image, renderer)
    left, bottom, right, top = unsliced.device_bounds
    delivered_device_bounds = (
        float(round(left) + 1),
        float(round(bottom) + 1),
        float(round(right) - 1),
        float(round(top) - 1),
    )
    plan = rr.RasterPlan.from_axes_image(
        image,
        renderer,
        delivered_device_bounds=delivered_device_bounds,
    )
    assert plan.source_shape == source.shape
    assert plan.source_extent == pytest.approx((-4.0, 3.0, 10.0, 15.0))
    assert plan.view_extent == pytest.approx((-3.5, 2.5, 10.4, 14.6))
    assert plan.destination_shape == unsliced.destination_shape
    assert np.allclose(
        plan.affine_matrix, unsliced.affine_matrix, rtol=0.0, atol=1.0e-12
    )
    assert plan.delivered_internal_bounds is not None
    assert plan.delivered_shape == (
        int(delivered_device_bounds[3] - delivered_device_bounds[1]),
        int(delivered_device_bounds[2] - delivered_device_bounds[0]),
    )


def test_delivered_device_plan_ignores_private_make_image_row_drift(
    monkeypatch,
) -> None:
    figure = Figure(figsize=(3.0, 3.0), dpi=20)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
    image = axes.imshow(
        np.arange(13 * 17, dtype=np.float32).reshape(13, 17),
        origin="upper",
        interpolation="hanning",
        extent=(-4.0, 3.0, 10.0, 15.0),
    )
    axes.set_xlim(-3.5, 2.5)
    axes.set_ylim(10.4, 14.6)
    canvas.draw()
    renderer = canvas.get_renderer()
    bbox = axes.get_window_extent(renderer)
    delivered_bounds = tuple(
        float(value)
        for value in (
            np.ceil(bbox.x0 - 0.5),
            np.ceil(bbox.y0 - 0.5),
            np.ceil(bbox.x1 - 0.5),
            np.ceil(bbox.y1 - 0.5),
        )
    )
    baseline = rr.RasterPlan.from_axes_image_delivered(
        image, renderer, delivered_device_bounds=delivered_bounds
    )
    original_make_image = image.make_image

    def oversized_private_raster(*args, **kwargs):
        made, origin_x, origin_y, transform = original_make_image(*args, **kwargs)
        assert made is not None
        oversized = np.pad(made, ((1, 0), (0, 0), (0, 0)), mode="edge")
        return oversized, origin_x, origin_y - 1.0, transform

    monkeypatch.setattr(image, "make_image", oversized_private_raster)
    drifted = rr.RasterPlan.from_axes_image_delivered(
        image, renderer, delivered_device_bounds=delivered_bounds
    )

    assert drifted.destination_shape == baseline.destination_shape
    assert drifted.delivered_internal_bounds == baseline.delivered_internal_bounds
    np.testing.assert_allclose(
        drifted.affine_matrix, baseline.affine_matrix, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("origin", ["upper", "lower"])
def test_canvas_orientation_precedes_asymmetric_delivered_slice(origin) -> None:
    source_height, source_width = 13, 17
    rows = np.linspace(0.2, 0.8, source_height, dtype=np.float32)[:, None]
    columns = np.linspace(0.1, 0.9, source_width, dtype=np.float32)[None, :]
    rgb = np.empty((source_height, source_width, 3), dtype=np.float32)
    rgb[..., 0] = rows
    rgb[..., 1] = columns
    rgb[..., 2] = 0.35 + 0.2 * rows

    figure = Figure(figsize=(3.0, 3.0), dpi=30, facecolor="black")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.103, 0.107, 0.791, 0.783])
    axes.set_axis_off()
    axes.patch.set_alpha(0.0)
    image = axes.imshow(
        rgb,
        origin=origin,
        interpolation="hanning",
        extent=(-4.0, 3.0, 10.0, 15.0),
    )
    axes.set_xlim(-3.5, 2.5)
    axes.set_ylim(10.4, 14.6)
    canvas.draw()
    rendered = np.asarray(canvas.buffer_rgba())
    image_pixels = np.any(rendered[..., :3] != 0, axis=-1)
    canvas_rows, canvas_columns = np.where(image_pixels)
    canvas_height = rendered.shape[0]
    full_device_bounds = (
        float(canvas_columns.min()),
        float(canvas_height - (canvas_rows.max() + 1)),
        float(canvas_columns.max() + 1),
        float(canvas_height - canvas_rows.min()),
    )
    # Remove only the top canvas row.  This kills the tempting mutation that
    # slices raw Agg rows first and flips the shortened result afterward.
    delivered_device_bounds = (
        full_device_bounds[0],
        full_device_bounds[1],
        full_device_bounds[2],
        full_device_bounds[3] - 1.0,
    )
    plan = rr.RasterPlan.from_axes_image(
        image,
        canvas.get_renderer(),
        delivered_device_bounds=delivered_device_bounds,
    )
    assert (plan.y_transform[0] < 0.0) is (origin == "upper")
    assert plan.internal_to_canvas_flip_y is True
    assert plan.internal_to_canvas_flip_x is False
    assert plan.delivered_mapping_scored_identity == 1.0

    internal, _ = rr.resample_rgb(
        rgb,
        np.ones((source_height, source_width), dtype=np.float32),
        plan,
    )
    delivered = plan.delivered_values(internal)
    quantized = np.floor(delivered * 255.0).astype(np.uint8)
    left, bottom, right, top = map(int, delivered_device_bounds)
    actual_canvas = rendered[
        canvas_height - top : canvas_height - bottom,
        left:right,
        :3,
    ]
    assert np.array_equal(quantized, actual_canvas)

    row_start, row_stop, column_start, column_stop = (
        plan.delivered_internal_bounds
    )
    slice_then_flip_mutant = internal[
        row_start:row_stop, column_start:column_stop
    ][::-1]
    assert not np.array_equal(
        np.floor(slice_then_flip_mutant * 255.0).astype(np.uint8),
        actual_canvas,
    )


def test_delivered_values_slice_and_identity_figure_image_are_exact() -> None:
    plan = rr.RasterPlan.from_affine(
        source_shape=(7, 11),
        destination_shape=(5, 8),
        transform=Affine2D.from_values(8 / 11, 0.0, 0.0, -5 / 7, 0.0, 5.0),
        delivered_internal_bounds=(1, 5, 2, 8),
        delivered_device_bounds=(5.0, 3.0, 11.0, 7.0),
        internal_to_canvas_flip_y=True,
        internal_to_canvas_flip_x=False,
    )
    full = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    delivered = plan.delivered_values(full)
    assert np.array_equal(delivered, full[::-1][1:5, 2:8])

    figure = Figure(figsize=(2.0, 2.0), dpi=10)
    canvas = FigureCanvasAgg(figure)
    image = figure.figimage(
        delivered,
        xo=5,
        yo=3,
        origin="upper",
        interpolation="nearest",
    )
    canvas.draw()
    trace = rr.verify_identity_draw(image, canvas.get_renderer(), plan)
    assert trace.source_shape == delivered.shape
    assert trace.output_shape == delivered.shape
    assert trace.device_bounds == (5.0, 3.0, 11.0, 7.0)
    assert trace.affine_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 4.0),
        (0.0, 0.0, 1.0),
    )


def test_identity_draw_rejects_same_shape_with_nonidentity_geometry() -> None:
    plan = rr.RasterPlan.from_affine(
        source_shape=(5, 6),
        destination_shape=(5, 6),
        transform=Affine2D(),
        delivered_internal_bounds=(0, 5, 0, 6),
        delivered_device_bounds=(4.0, 2.0, 10.0, 7.0),
        internal_to_canvas_flip_y=False,
        internal_to_canvas_flip_x=False,
    )
    figure = Figure(figsize=(2.0, 2.0), dpi=10)
    canvas = FigureCanvasAgg(figure)
    misplaced = figure.figimage(
        np.zeros((5, 6), dtype=np.float32),
        xo=5,
        yo=2,
        origin="upper",
        interpolation="nearest",
    )
    canvas.draw()
    with pytest.raises(RuntimeError, match="placement differs"):
        rr.verify_identity_draw(misplaced, canvas.get_renderer(), plan)


def test_product_resampler_does_not_import_validation_or_an_oracle() -> None:
    source = inspect.getsource(rr)
    assert "validation" not in source
    assert "fractional_area" not in source
