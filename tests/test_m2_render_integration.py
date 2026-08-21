"""M2 primary true-color renderer integration and ordering contracts."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
from unittest import mock

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from PIL import Image

import render
import render_resample
import truecolor
from render_policy import M1_POLICY, M2_POLICY
from satellites import FetchResult


def _physical_fetch(height=30, width=46):
    y, x = np.mgrid[:height, :width]
    red = (0.05 + 0.32 * x / max(width - 1, 1)).astype(np.float32)
    green = (0.04 + 0.25 * y / max(height - 1, 1)).astype(np.float32)
    blue = (0.03 + 0.18 * (x + y) / max(width + height - 2, 1)).astype(
        np.float32
    )
    day = np.dstack((red, green, blue)).astype(np.float32)
    geom = np.ones((height, width), dtype=bool)
    day_coverage = np.ones((height, width), dtype=np.float32)
    day_coverage[:3, :4] = 0.0
    ir_coverage = np.ones((height, width), dtype=np.float32)
    ir = (303.0 - 110.0 * y / max(height - 1, 1)).astype(np.float32)
    state = truecolor.TrueColorPhysicalState(
        day_rgb_reflectance=day,
        day_coverage=day_coverage,
        ir_bt_kelvin=ir,
        ir_coverage=ir_coverage,
        cos_sza=np.full((height, width), 0.45, np.float32),
        day_weight=np.ones((height, width), dtype=np.float32),
        geom_valid=geom,
        red_corrected=red.copy(),
        veggie_corrected=np.full((height, width), 0.035, np.float32),
        glint_deg=np.full((height, width), 90.0, np.float32),
        sensor="abi",
        grid_metadata={
            "grid": "regular_latlon",
            "row_order": "north_to_south",
            "sensor": "abi",
            "point_stride_applied": False,
        },
        extent=(-80.0, 14.0, -68.0, 26.0),
        projection="PlateCarree",
    )
    lon_axis = (
        (np.linspace(-80.0, -68.0, width) + 180.0) % 360.0 - 180.0
    ).astype(np.float32)
    lat_axis = np.linspace(26.0, 14.0, height).astype(np.float32)
    lons, lats = np.meshgrid(lon_axis, lat_axis)
    return FetchResult(
        cmi=state.day_rgb_reflectance,
        lats=lats,
        lons=lons,
        channel=2,
        generic_channel="true_color",
        scan_start=dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc),
        product="M2-TRUECOLOR-TEST",
        bucket="test",
        sat_name="TEST-SAT",
        sub_sat_lon=-75.2,
        units="truecolor_physical",
        cos_sza=state.cos_sza,
        geom_valid=geom,
        truecolor_physical=state,
    )


def _render(data, policy=M2_POLICY, *, dpi=20, downsample=1):
    return render.render_png(
        data,
        [-80.0, 14.0, -68.0, 26.0],
        2,
        "2026-08-20 00:00 UTC",
        "true_color",
        downsample,
        coastlines=False,
        gridlines=False,
        dpi=dpi,
        policy=policy,
    )


def test_all_physical_resampling_precedes_the_single_final_cira_call():
    data = _physical_fetch()
    events = []
    original_rgb = render_resample.resample_rgb
    original_field = render_resample.resample_field
    original_bundle = render_resample.resample_fields_shared_coverage
    original_cira = truecolor.cira_stretch
    original_place = render._place_device_raster

    def resample_rgb(*args, **kwargs):
        events.append("physical_resample")
        return original_rgb(*args, **kwargs)

    def resample_field(*args, **kwargs):
        events.append("physical_resample")
        return original_field(*args, **kwargs)

    def resample_bundle(*args, **kwargs):
        events.append("physical_resample")
        return original_bundle(*args, **kwargs)

    def cira(*args, **kwargs):
        events.append("cira")
        return original_cira(*args, **kwargs)

    def place(fig, ax, raster, bounds):
        events.append("device_place")
        return original_place(fig, ax, raster, bounds)

    with (
        mock.patch.object(render_resample, "resample_rgb", side_effect=resample_rgb),
        mock.patch.object(render_resample, "resample_field", side_effect=resample_field),
        mock.patch.object(
            render_resample,
            "resample_fields_shared_coverage",
            side_effect=resample_bundle,
        ),
        mock.patch.object(truecolor, "cira_stretch", side_effect=cira),
        mock.patch.object(render, "_place_device_raster", side_effect=place),
    ):
        png = _render(data)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert events.count("cira") == 1
    cira_index = events.index("cira")
    assert all(
        index < cira_index
        for index, event in enumerate(events)
        if event == "physical_resample"
    )
    assert events[-1] == "device_place"


def test_dynamic_plan_trace_is_bound_to_every_delivered_field():
    data = _physical_fetch(height=37, width=59)
    traces = []
    original = render._resample_truecolor_physical_state

    def traced(state, plan):
        delivered = original(state, plan)
        traces.append(delivered.grid_metadata)
        return delivered

    with mock.patch.object(
        render, "_resample_truecolor_physical_state", side_effect=traced
    ):
        _render(data)

    assert len(traces) == 1
    trace = traces[0]
    assert trace["source_shape"] == (37, 59)
    assert trace["internal_destination_shape"] != trace["source_shape"]
    assert trace["delivered_shape"][0] > 0 and trace["delivered_shape"][1] > 0
    assert trace["resample_kernel"] == "hanning"
    assert trace["resample_domain"] == "physical_float32"
    assert trace["physical_domain_transport"] == (
        render_resample.PHYSICAL_DOMAIN_TRANSPORT
    )
    assert trace["source_geometry_binding"] == (
        "request_regular_latlon_exact_float32_linspace_v1"
    )
    assert trace["primary_sensor"] == "abi"
    assert trace["source_projection"] == "PlateCarree"
    assert trace["source_declared_extent"] == (-80.0, 14.0, -68.0, 26.0)
    assert trace["shared_day_fields"] == [
        "day_rgb_reflectance",
        "red_corrected",
        "glint_deg",
        "veggie_corrected",
    ]
    assert trace["day_coverage_binding"].startswith(
        "one_premultiplied_hanning_replay"
    )
    assert trace["terminator_weight_binding"] == (
        "source_88_to_95_day_weight_resampled_with_modality_union_v1"
    )
    assert trace["terminator_weight_policy"] == (
        truecolor.M2_DAY_WEIGHT_POLICY_ID
    )
    assert trace["delivered_internal_bounds"] is not None
    assert trace["sampling_phase_yx"][0] != trace["sampling_phase_yx"][1]


def test_terminator_weight_is_resampled_before_nonlinear_blending():
    source_shape = (8, 40)
    destination_shape = (8, 9)
    original = _physical_fetch(*source_shape).truecolor_physical
    assert original is not None
    cos_sza = np.empty(source_shape, dtype=np.float32)
    cos_sza[:, :20] = np.float32(1.0)
    cos_sza[:, 20:] = np.float32(np.cos(np.deg2rad(96.0)))
    source_weight = np.empty(source_shape, dtype=np.float32)
    source_weight[:, :20] = np.float32(1.0)
    source_weight[:, 20:] = np.float32(0.0)
    state = dataclasses_replace_fetch(
        original,
        cos_sza=cos_sza,
        day_weight=source_weight,
    )
    plan = render_resample.RasterPlan.from_affine(
        source_shape=source_shape,
        destination_shape=destination_shape,
        transform=Affine2D().scale(
            destination_shape[1] / source_shape[1],
            destination_shape[0] / source_shape[0],
        ),
    )

    delivered = render._resample_truecolor_physical_state(state, plan)
    post_cosine_mutant = truecolor.terminator_day_weight(
        delivered.cos_sza
    ).astype(np.float32)
    transition = (3, 4)
    assert delivered.day_weight[transition] == pytest.approx(0.5078, abs=0.01)
    assert post_cosine_mutant[transition] == pytest.approx(1.0, abs=1e-6)
    assert abs(
        float(post_cosine_mutant[transition] - delivered.day_weight[transition])
    ) > 0.4

    approved = truecolor.finalize_truecolor_m2(delivered)
    mutant = truecolor.finalize_truecolor_m2(
        dataclasses_replace_fetch(delivered, day_weight=post_cosine_mutant)
    )
    assert np.max(np.abs(approved.rgb - mutant.rgb)) > 0.05


def test_sparse_fractional_support_uses_one_bound_threshold_for_all_fields():
    source_shape = (101, 137)
    destination_shape = (6, 8)
    original = _physical_fetch(*source_shape).truecolor_physical
    assert original is not None
    geom = np.zeros(source_shape, dtype=bool)
    geom[50, 68] = True
    coverage = geom.astype(np.float32)
    state = dataclasses_replace_fetch(
        original,
        day_coverage=coverage,
        ir_coverage=coverage.copy(),
        day_weight=np.where(
            geom, original.day_weight, np.float32(0.0)
        ).astype(np.float32),
        geom_valid=geom,
    )
    plan = render_resample.RasterPlan.from_affine(
        source_shape=source_shape,
        destination_shape=destination_shape,
        transform=(
            Affine2D()
            .scale(
                destination_shape[1] / source_shape[1],
                destination_shape[0] / source_shape[0],
            )
            .translate(-0.86625, -0.495)
        ),
    )

    delivered = render._resample_truecolor_physical_state(state, plan)
    assert delivered.grid_metadata["coverage_threshold"] == (
        render_resample.DEFAULT_COVERAGE_THRESHOLD
    )
    assert np.all(
        (delivered.day_coverage == 0.0)
        | (delivered.day_coverage > render_resample.DEFAULT_COVERAGE_THRESHOLD)
    )
    assert np.all(delivered.day_coverage[~delivered.geom_valid] == 0.0)
    assert np.all(delivered.ir_coverage[~delivered.geom_valid] == 0.0)


def test_m2_truecolor_bypasses_public_render_point_stride():
    data = _physical_fetch(height=37, width=59)
    source_shapes = []
    original = render._resample_truecolor_physical_state

    def traced(state, plan):
        source_shapes.append((state.day_coverage.shape, plan.source_shape))
        return original(state, plan)

    with mock.patch.object(
        render, "_resample_truecolor_physical_state", side_effect=traced
    ):
        _render(data, downsample=3)

    assert source_shapes == [((37, 59), (37, 59))]


def test_policy_and_payload_domains_fail_closed():
    physical = _physical_fetch()
    with pytest.raises(RuntimeError, match="requires the registered M2 policy"):
        _render(physical, M1_POLICY)

    encoded = dataclasses_replace_fetch(
        physical,
        units="rgb",
        truecolor_physical=None,
        cmi=truecolor.cira_stretch(physical.cmi),
    )
    with pytest.raises(RuntimeError, match="rejects legacy encoded"):
        _render(encoded, M2_POLICY)


@pytest.mark.parametrize(
    "mutation",
    (
        "foreign_extent",
        "foreign_projection",
        "metadata_value",
        "metadata_lookalike",
        "latitude_not_separable",
        "longitude_interior_warp",
        "longitude_endpoint",
        "masked_latitudes",
        "masked_longitudes",
        "geometry_alias_copy",
        "source_day_weight_lookalike",
        "unsupported_primary_sensor",
    ),
)
def test_m2_source_geometry_is_bound_before_raster_planning(mutation):
    data = _physical_fetch(height=19, width=27)
    state = data.truecolor_physical
    assert state is not None

    if mutation == "foreign_extent":
        state = dataclasses_replace_fetch(
            state, extent=(20.0, 30.0, 40.0, 50.0)
        )
    elif mutation == "foreign_projection":
        state = dataclasses_replace_fetch(state, projection="lookalike")
    elif mutation == "metadata_value":
        state = dataclasses_replace_fetch(
            state, grid_metadata={**state.grid_metadata, "sensor": "ahi"}
        )
    elif mutation == "metadata_lookalike":
        class DictLookalike(dict):
            pass

        state = dataclasses_replace_fetch(
            state, grid_metadata=DictLookalike(state.grid_metadata)
        )
    elif mutation == "unsupported_primary_sensor":
        state = dataclasses_replace_fetch(
            state,
            sensor="fci",
            grid_metadata={**state.grid_metadata, "sensor": "fci"},
        )
    elif mutation == "source_day_weight_lookalike":
        state = dataclasses_replace_fetch(
            state,
            day_weight=np.full(
                state.day_weight.shape, np.float32(0.5), dtype=np.float32
            ),
        )

    if state is not data.truecolor_physical:
        data = dataclasses_replace_fetch(
            data,
            truecolor_physical=state,
            cmi=state.day_rgb_reflectance,
            cos_sza=state.cos_sza,
            geom_valid=state.geom_valid,
        )
    if mutation == "latitude_not_separable":
        lats = data.lats.copy()
        lats[3, 4] += np.float32(0.125)
        data = dataclasses_replace_fetch(data, lats=lats)
    elif mutation == "longitude_interior_warp":
        lons = data.lons.copy()
        lons[:, 8] += np.float32(0.05)
        data = dataclasses_replace_fetch(data, lons=lons)
    elif mutation == "longitude_endpoint":
        lons = data.lons.copy()
        lons[:, -1] -= np.float32(0.25)
        data = dataclasses_replace_fetch(data, lons=lons)
    elif mutation == "masked_latitudes":
        data = dataclasses_replace_fetch(
            data,
            lats=np.ma.masked_array(data.lats, mask=False),
        )
    elif mutation == "masked_longitudes":
        data = dataclasses_replace_fetch(
            data,
            lons=np.ma.masked_array(data.lons, mask=False),
        )
    elif mutation == "geometry_alias_copy":
        data = dataclasses_replace_fetch(data, cos_sza=data.cos_sza.copy())

    with pytest.raises(RuntimeError, match="M2"):
        _render(data)


def test_m2_positive_180_endpoint_passes_source_binder_and_full_render():
    data = _physical_fetch(height=17, width=31)
    state = data.truecolor_physical
    assert state is not None
    bbox = [170.0, 14.0, 180.0, 26.0]
    state = dataclasses_replace_fetch(state, extent=tuple(bbox))
    lon_axis = np.linspace(bbox[0], bbox[2], data.lons.shape[1]).astype(
        np.float32
    )
    lat_axis = np.linspace(bbox[3], bbox[1], data.lats.shape[0]).astype(
        np.float32
    )
    lons, lats = np.meshgrid(lon_axis, lat_axis)
    data = dataclasses_replace_fetch(
        data,
        lons=lons,
        lats=lats,
        truecolor_physical=state,
        cmi=state.day_rgb_reflectance,
        cos_sza=state.cos_sza,
        geom_valid=state.geom_valid,
    )

    render._validate_m2_truecolor_source_geometry(data, bbox)
    png = render.render_png(
        data,
        bbox,
        2,
        "2026-08-20 00:00 UTC",
        "true_color",
        1,
        coastlines=False,
        gridlines=False,
        dpi=20,
        policy=M2_POLICY,
    )
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_uncovered_m2_pixels_match_figure_dark_background_at_native_bytes():
    state = _physical_fetch(height=4, width=5).truecolor_physical
    assert state is not None
    zero = np.zeros((4, 5), dtype=np.float32)
    uncovered = dataclasses_replace_fetch(
        state,
        day_coverage=zero,
        ir_coverage=zero.copy(),
    )
    background = np.asarray(to_rgb(render.DARK_BG), dtype=np.float32)
    delivery = truecolor.finalize_truecolor_m2(
        uncovered,
        background_display_rgb=tuple(float(value) for value in background),
    )
    raster = np.tile(delivery.rgb, (10, 10, 1))

    fig = Figure(figsize=(5.0, 4.0), dpi=10, facecolor=render.DARK_BG)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    render._place_device_raster(fig, ax, raster, (0, 40, 0, 50))
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=fig.dpi, facecolor=render.DARK_BG)
    pixels = np.asarray(Image.open(io.BytesIO(output.getvalue())).convert("RGB"))
    expected = np.asarray([10, 13, 18], dtype=np.uint8)
    np.testing.assert_array_equal(pixels[20, 25], expected)
    np.testing.assert_array_equal(pixels[0, 0], expected)


def test_truecolor_plan_uses_center_derived_edges_not_bbox_as_image_edges():
    data = _physical_fetch(height=17, width=31)
    captures = []
    original = render._m2_plan_axes_raster

    def capture(*args, **kwargs):
        captures.append(tuple(float(value) for value in kwargs["extent"]))
        return original(*args, **kwargs)

    with mock.patch.object(render, "_m2_plan_axes_raster", side_effect=capture):
        _render(data)

    assert len(captures) == 1
    left, right, bottom, top = captures[0]
    lon_centres = left + (np.arange(data.lons.shape[1]) + 0.5) * (
        (right - left) / data.lons.shape[1]
    )
    lat_centres = top - (np.arange(data.lats.shape[0]) + 0.5) * (
        (top - bottom) / data.lats.shape[0]
    )
    assert render._float32_axes_equal_within_ulps(
        lon_centres.astype(np.float32), data.lons[0]
    )
    assert render._float32_axes_equal_within_ulps(
        lat_centres.astype(np.float32), data.lats[:, 0]
    )

    bbox_as_edges = (-80.0, -68.0, 14.0, 26.0)
    mutant_lon_centres = bbox_as_edges[0] + (
        np.arange(data.lons.shape[1]) + 0.5
    ) * ((bbox_as_edges[1] - bbox_as_edges[0]) / data.lons.shape[1])
    mutant_lat_centres = bbox_as_edges[3] - (
        np.arange(data.lats.shape[0]) + 0.5
    ) * ((bbox_as_edges[3] - bbox_as_edges[2]) / data.lats.shape[0])
    assert not np.allclose(mutant_lon_centres, data.lons[0], rtol=0.0, atol=2.0e-6)
    assert not np.allclose(
        mutant_lat_centres,
        data.lats[:, 0],
        rtol=0.0,
        atol=2.0e-6,
    )


def dataclasses_replace_fetch(value, **changes):
    import dataclasses

    return dataclasses.replace(value, **changes)


def test_three_independent_m2_renders_are_byte_identical():
    data = _physical_fetch(height=24, width=36)
    hashes = [hashlib.sha256(_render(data)).hexdigest() for _ in range(3)]
    assert len(set(hashes)) == 1


@pytest.mark.parametrize("requested_dpi", [20, 137])
def test_authoritative_raster_is_planned_at_the_delivered_png_dpi(requested_dpi):
    data = _physical_fetch(height=31, width=47)
    captures = []
    original = render._m2_place_verified

    def traced(fig, ax, raster, plan, bounds):
        captures.append(
            {
                "figure_dpi": float(fig.dpi),
                "canvas_shape": (
                    int(round(float(fig.bbox.height))),
                    int(round(float(fig.bbox.width))),
                ),
                "bounds": tuple(int(value) for value in bounds),
                "raster_shape": tuple(raster.shape[:2]),
                "delivered_shape": plan.delivered_shape,
            }
        )
        return original(fig, ax, raster, plan, bounds)

    with mock.patch.object(render, "_m2_place_verified", side_effect=traced):
        png = _render(data, dpi=requested_dpi)

    assert len(captures) == 1
    capture = captures[0]
    decoded = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    assert capture["figure_dpi"] == float(requested_dpi)
    assert decoded.shape[:2] == capture["canvas_shape"]
    top, bottom, left, right = capture["bounds"]
    assert capture["raster_shape"] == (bottom - top, right - left)
    assert capture["delivered_shape"] == capture["raster_shape"]
    assert 0 <= top < bottom <= decoded.shape[0]
    assert 0 <= left < right <= decoded.shape[1]
    # The centre of the bound imagery region must survive into the final PNG;
    # a fixed 100-DPI figimage clipped by savefig would leave dark background.
    centre = decoded[(top + bottom) // 2, (left + right) // 2]
    background = np.rint(np.asarray(to_rgb(render.DARK_BG)) * 255.0).astype(
        np.uint8
    )
    assert not np.array_equal(centre, background)
