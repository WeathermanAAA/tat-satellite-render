"""M2 regular-grid scalar delivery stays in field space until colour mapping."""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
from unittest import mock

import numpy as np
import pytest
from PIL import Image

import render
import render_resample
from render_policy import M1_POLICY, M2_POLICY
from satellites import FetchResult


def _regular_scalar(
    *,
    units="K",
    product="GridSat-GOES",
    generic_channel="clean_ir",
    channel=4,
):
    height, width = 44, 68
    y, x = np.mgrid[:height, :width]
    lons, lats = np.meshgrid(
        np.linspace(-82.0, -66.0, width, dtype=np.float32),
        np.linspace(22.0, 10.0, height, dtype=np.float32),
    )
    kelvin = (205.0 + 90.0 * x / (width - 1)).astype(np.float32)
    kelvin[:, width // 2 :] += np.float32(8.0) * ((y[:, width // 2 :] % 2) == 0)
    kelvin[:4, :5] = np.nan
    field = kelvin if units == "K" else (kelvin - np.float32(273.15))
    return FetchResult(
        cmi=field,
        lats=lats,
        lons=lons,
        channel=channel,
        generic_channel=generic_channel,
        scan_start=dt.datetime(2005, 7, 1, 18, tzinfo=dt.timezone.utc),
        product=product,
        bucket="test",
        sat_name="TEST-SAT",
        sub_sat_lon=-75.0,
        units=units,
    )


def _regular_visible():
    data = _regular_scalar(units="K")
    reflectance = np.linspace(
        0.0, 1.0, data.cmi.size, dtype=np.float32
    ).reshape(data.cmi.shape)
    return dataclasses.replace(
        data,
        cmi=reflectance,
        channel=1,
        generic_channel="visible_red",
        units="1",
    )


def _render(data, policy, *, downsample=1):
    return render.render_png(
        data,
        [-82.0, 10.0, -66.0, 22.0],
        data.channel,
        "2005-07-01 18:00 UTC",
        "rainbow_ir",
        downsample,
        coastlines=False,
        gridlines=False,
        dpi=30,
        policy=policy,
    )


def test_regular_scalar_resamples_kelvin_before_norm_and_palette():
    data = _regular_scalar(units="K")
    seen = []
    original = render_resample.resample_field

    def traced(field, coverage, plan, **kwargs):
        seen.append(np.asarray(field).copy())
        return original(field, coverage, plan, **kwargs)

    with mock.patch.object(render_resample, "resample_field", side_effect=traced):
        png = _render(data, M2_POLICY)

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], data.cmi)
    finite = seen[0][np.isfinite(seen[0])]
    assert float(finite.min()) > 180.0 and float(finite.max()) < 330.0
    image = Image.open(io.BytesIO(png))
    assert image.size == (360, 270)


def test_celsius_input_is_converted_to_kelvin_before_spatial_operation():
    data = _regular_scalar(units="C")
    seen = []
    original = render_resample.resample_field

    def traced(field, coverage, plan, **kwargs):
        seen.append(np.asarray(field).copy())
        return original(field, coverage, plan, **kwargs)

    with mock.patch.object(render_resample, "resample_field", side_effect=traced):
        _render(data, M2_POLICY)

    finite = seen[0][np.isfinite(seen[0])]
    assert float(finite.min()) > 180.0
    np.testing.assert_allclose(
        seen[0][np.isfinite(data.cmi)],
        data.cmi[np.isfinite(data.cmi)] + np.float32(273.15),
        atol=2.0e-5,
    )


def test_visible_display_mapping_occurs_only_after_physical_resampling():
    data = _regular_visible()
    events = []
    original_resample = render_resample.resample_field
    original_normalize = render.normalize_visible

    def resampled(*args, **kwargs):
        events.append("resample")
        return original_resample(*args, **kwargs)

    def normalized(*args, **kwargs):
        events.append("display_map")
        return original_normalize(*args, **kwargs)

    with (
        mock.patch.object(render_resample, "resample_field", side_effect=resampled),
        mock.patch.object(render, "normalize_visible", side_effect=normalized),
    ):
        _render(data, M2_POLICY)

    assert events == ["resample", "display_map"]


def test_m2_regular_scalar_bypasses_point_stride_before_delivery_resample():
    data = _regular_scalar(units="K")
    seen = []
    original = render_resample.resample_field

    def traced(field, coverage, plan, **kwargs):
        seen.append((np.asarray(field).shape, plan.source_shape))
        return original(field, coverage, plan, **kwargs)

    with mock.patch.object(render_resample, "resample_field", side_effect=traced):
        _render(data, M2_POLICY, downsample=2)

    assert seen == [(data.cmi.shape, data.cmi.shape)]


def test_m2_uses_verified_figure_image_while_m1_keeps_axes_image():
    data = _regular_scalar()
    m2_placements = []
    original_place = render._place_device_raster

    def placed(fig, ax, raster, bounds):
        m2_placements.append((np.asarray(raster).shape, tuple(bounds)))
        return original_place(fig, ax, raster, bounds)

    with mock.patch.object(render, "_place_device_raster", side_effect=placed):
        _render(data, M2_POLICY)
    assert len(m2_placements) == 1
    shape, (top, bottom, left, right) = m2_placements[0]
    assert shape[:2] == (bottom - top, right - left)

    m1_placements = []
    with mock.patch.object(
        render,
        "_place_device_raster",
        side_effect=lambda *args: m1_placements.append(args),
    ):
        _render(data, M1_POLICY)
    assert m1_placements == []


def test_m2_rejects_curvilinear_scalar_instead_of_silent_display_fallback():
    data = _regular_scalar()
    curved_lons = data.lons.copy()
    curved_lons += np.linspace(0.0, 0.1, curved_lons.shape[0], dtype=np.float32)[:, None]
    curved = dataclasses.replace(data, lons=curved_lons)
    with pytest.raises(RuntimeError, match="rejects curvilinear"):
        _render(curved, M2_POLICY)
    assert _render(curved, M1_POLICY).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("product", "generic_channel", "channel", "units"),
    (
        ("GRIDSAT-B1", "clean_ir", 1, "K"),
        ("GRIDSAT-B1", "wv_upper", 2, "C"),
        ("GridSat-GOES", "visible_red", 1, "1"),
        ("GridSat-GOES", "shortwave_ir", 2, "degC"),
        ("GridSat-GOES", "wv_upper", 3, "celsius"),
        ("GridSat-GOES", "clean_ir", 4, "K"),
    ),
)
def test_registered_archive_family_channel_and_unit_conventions_are_explicit(
    product,
    generic_channel,
    channel,
    units,
):
    if generic_channel == "visible_red":
        data = dataclasses.replace(
            _regular_visible(),
            product=product,
            channel=channel,
            units=units,
        )
    else:
        data = _regular_scalar(
            units=units,
            product=product,
            generic_channel=generic_channel,
            channel=channel,
        )
    render._validate_m2_scalar_source(data, channel)
    axes = render._m2_regular_grid_axes(data.lons, data.lats)
    assert axes is not None
    np.testing.assert_array_equal(axes[0], data.lons[0])
    np.testing.assert_array_equal(axes[1], data.lats[:, 0])


def test_m2_strict_grid_rejects_gradual_warp_accepted_by_legacy_detector():
    data = _regular_scalar()
    steps = np.linspace(
        0.004,
        0.004999,
        data.lons.shape[1] - 1,
        dtype=np.float64,
    )
    warped_axis = np.concatenate(
        ([float(data.lons[0, 0])], float(data.lons[0, 0]) + np.cumsum(steps))
    ).astype(np.float32)
    warped_lons = np.broadcast_to(warped_axis, data.lons.shape).copy()
    warped = dataclasses.replace(data, lons=warped_lons)

    assert render._regular_grid_axes(warped.lons, warped.lats) is not None
    assert render._m2_regular_grid_axes(warped.lons, warped.lats) is None
    with pytest.raises(RuntimeError, match="warped"):
        _render(warped, M2_POLICY)


@pytest.mark.parametrize(
    ("start", "step", "count"),
    (
        (-179.97, 0.07, 5142),
        (-135.98, 0.04, 4200),
        (-72.018, 0.036, 3100),
    ),
)
def test_m2_strict_grid_accepts_float64_archive_axes_cast_to_float32(
    start, step, count
):
    # Mirrors the archive loaders: the netCDF coordinate vector is float64,
    # then the cropped meshgrid is cast to float32 in the FetchResult.
    lon_axis = (start + np.arange(count, dtype=np.float64) * step).astype(
        np.float32
    )
    lat_axis = (22.0 - np.arange(3, dtype=np.float64) * step).astype(np.float32)
    lons, lats = np.meshgrid(lon_axis, lat_axis)
    axes = render._m2_regular_grid_axes(lons, lats)
    assert axes is not None
    np.testing.assert_array_equal(axes[0], lon_axis)
    np.testing.assert_array_equal(axes[1], lat_axis)


@pytest.mark.parametrize("coord_name", ("lats", "lons"))
def test_m2_strict_grid_rejects_masked_coordinate_lookalikes(coord_name):
    data = _regular_scalar()
    masked = np.ma.masked_array(getattr(data, coord_name), mask=False)
    mutation = dataclasses.replace(data, **{coord_name: masked})
    with pytest.raises(RuntimeError, match="exact plain ndarrays"):
        _render(mutation, M2_POLICY)


@pytest.mark.parametrize(
    ("generic_channel", "channel", "units"),
    (
        ("visible_red", 1, "K"),
        ("visible_red", 1, "C"),
        ("clean_ir", 4, "1"),
        ("clean_ir", 4, "kelvin"),
        ("clean_ir", 4, ""),
        ("clean_ir", 4, None),
        ("wv_lower", 10, "K"),
    ),
)
def test_m2_rejects_missing_unknown_or_mismatched_channel_units(
    generic_channel,
    channel,
    units,
):
    data = _regular_visible() if generic_channel == "visible_red" else _regular_scalar()
    mutation = dataclasses.replace(
        data,
        generic_channel=generic_channel,
        channel=channel,
        units=units,
    )
    with pytest.raises(RuntimeError, match="M2 scalar"):
        _render(mutation, M2_POLICY)


@pytest.mark.parametrize("dtype", (np.float64, np.uint8))
def test_m2_rejects_non_float32_calibrated_fields(dtype):
    data = _regular_scalar()
    values = np.nan_to_num(data.cmi, nan=0.0).astype(dtype)
    mutation = dataclasses.replace(data, cmi=values)
    with pytest.raises(RuntimeError, match="exact float32 ndarray"):
        _render(mutation, M2_POLICY)


def test_m2_rejects_masked_float32_calibrated_field_lookalike():
    data = _regular_scalar()
    mutation = dataclasses.replace(
        data,
        cmi=np.ma.masked_array(data.cmi, mask=False),
    )
    with pytest.raises(RuntimeError, match="exact float32 ndarray"):
        _render(mutation, M2_POLICY)


def test_m2_rejects_unregistered_archive_family_and_native_band_mismatch():
    data = _regular_scalar()
    with pytest.raises(RuntimeError, match="outside archive authority"):
        _render(dataclasses.replace(data, product="GridSat-lookalike"), M2_POLICY)
    with pytest.raises(RuntimeError, match="outside archive authority"):
        _render(dataclasses.replace(data, product="MERGIR"), M2_POLICY)
    with pytest.raises(RuntimeError, match="native channel"):
        _render(dataclasses.replace(data, channel=3), M2_POLICY)


def test_m2_rebinds_the_same_strict_grid_at_final_planning_boundary():
    data = _regular_scalar()
    original = render._m2_regular_grid_axes
    calls = 0

    def drifting_binder(*args):
        nonlocal calls
        calls += 1
        axes = original(*args)
        assert axes is not None
        if calls == 2:
            changed_lon = axes[0].copy()
            changed_lon[1] = np.nextafter(changed_lon[1], np.float32(np.inf))
            return changed_lon, axes[1]
        return axes

    with (
        mock.patch.object(render, "_m2_regular_grid_axes", side_effect=drifting_binder),
        pytest.raises(RuntimeError, match="binding changed"),
    ):
        _render(data, M2_POLICY)
    assert calls == 2


def test_m1_never_enters_strict_m2_scalar_boundary():
    data = dataclasses.replace(_regular_scalar(), cmi=_regular_scalar().cmi.astype(np.float64))
    with mock.patch.object(
        render,
        "_m2_regular_grid_axes",
        side_effect=AssertionError("M1 entered M2 strict binder"),
    ):
        png = _render(data, M1_POLICY)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
