"""M2 native ABI crop and upstream physical-cap mutation contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import satellites


def _abi_dataset(*, height: int, width: int) -> xr.Dataset:
    x = np.linspace(-0.1, 0.1, width, dtype=np.float32)
    y = np.linspace(0.05, -0.05, height, dtype=np.float32)
    values = np.broadcast_to(
        np.arange(width, dtype=np.float32)[None, :],
        (height, width),
    ).copy()
    projection = xr.DataArray(
        np.int8(0),
        attrs={
            "perspective_point_height": 35_786_023.0,
            "semi_major_axis": 6_378_137.0,
            "semi_minor_axis": 6_356_752.31414,
            "longitude_of_projection_origin": -75.2,
        },
    )
    return xr.Dataset(
        data_vars={
            "CMI": (("y", "x"), values),
            "goes_imager_projection": projection,
        },
        coords={"x": x, "y": y},
    )


@pytest.fixture
def full_window_projection(monkeypatch):
    def project(lat, lon, lon_origin, height, r_eq, r_pol):
        shape = np.asarray(lat).shape
        x = np.linspace(-0.2, 0.2, np.prod(shape), dtype=np.float64).reshape(
            shape
        )
        y = np.linspace(-0.1, 0.1, np.prod(shape), dtype=np.float64).reshape(
            shape
        )
        return x, y

    monkeypatch.setattr(satellites, "_latlon_to_xy", project)
    monkeypatch.setattr(satellites, "_latlon_to_xy_unchecked", project)
    monkeypatch.setattr(satellites, "_latlon_to_xy_visible", project)


def test_abi_m2_native_crop_kills_legacy_point_stride_mutant(
    monkeypatch,
    full_window_projection,
):
    """A >2x-cap fixture distinguishes stride=1 from the legacy point cap."""

    monkeypatch.setenv("SAT_MAX_PX_PER_AXIS", "2400")
    dataset = _abi_dataset(height=7, width=4802)
    bbox = [-90.0, 5.0, -60.0, 25.0]

    legacy = satellites.GOES_EAST._crop_to_bbox(dataset, bbox)
    physical = satellites.GOES_EAST._crop_to_bbox(
        dataset,
        bbox,
        preserve_native_for_m2=True,
        return_m2_preload_guard=True,
    )

    legacy_cmi, legacy_x, legacy_y = legacy[:3]
    physical_cmi, physical_x, physical_y = physical[:3]
    assert physical_cmi.shape == (7, 4802)
    assert np.array_equal(physical_cmi, dataset["CMI"].values)
    assert np.array_equal(physical_x, dataset["x"].values)
    assert np.array_equal(physical_y, dataset["y"].values)
    assert physical[7]["basis"] == "abi_native_crop_before_xarray_isel_load"
    assert physical[7]["native_pixels"] == 7 * 4802

    # The old/S2 path is intentionally unchanged and still obeys its mutable
    # point-stride limit. This is the exact mutant forbidden on the M2 path.
    assert legacy_cmi.shape == (7, 2401)
    assert np.array_equal(legacy_cmi, dataset["CMI"].values[:, ::2])
    assert np.array_equal(legacy_x, dataset["x"].values[::2])
    assert np.array_equal(legacy_y, dataset["y"].values)


def test_abi_m2_exact_crop_guard_runs_before_xarray_isel_or_load(
    monkeypatch,
    full_window_projection,
):
    class Variable:
        def __init__(self, values=None, attrs=None):
            self.values = values
            self.attrs = attrs or {}

    class DatasetSpy:
        def __init__(self):
            self.variables = {
                "goes_imager_projection": Variable(
                    attrs={
                        "perspective_point_height": 35_786_023.0,
                        "semi_major_axis": 6_378_137.0,
                        "semi_minor_axis": 6_356_752.31414,
                        "longitude_of_projection_origin": -75.2,
                    }
                ),
                "x": Variable(np.linspace(-0.1, 0.1, 4000, dtype=np.float32)),
                "y": Variable(np.linspace(0.05, -0.05, 2000, dtype=np.float32)),
            }
            self.isel_calls = []

        def __getitem__(self, key):
            return self.variables[key]

        def isel(self, **kwargs):
            self.isel_calls.append(kwargs)
            raise AssertionError("M2 guard ran after xarray isel/load")

    dataset = DatasetSpy()
    with pytest.raises(RuntimeError, match="safety envelope exceeded"):
        satellites.GOES_EAST._crop_to_bbox(
            dataset,
            [-90.0, 5.0, -60.0, 25.0],
            preserve_native_for_m2=True,
        )
    assert dataset.isel_calls == []


def test_goes_physical_compose_requests_native_crop_mode_for_band_reads(
    monkeypatch,
):
    class StopAfterCrop(RuntimeError):
        pass

    calls = []

    class Dataset:
        def close(self):
            return None

    monkeypatch.setattr(
        satellites.GOES_EAST,
        "open",
        lambda resolved: (Dataset(), "/tmp/nonexistent-m2-native-crop-test"),
    )

    def crop(
        dataset,
        bbox,
        *,
        preserve_native_for_m2=False,
        return_m2_preload_guard=False,
    ):
        calls.append(preserve_native_for_m2)
        assert return_m2_preload_guard is preserve_native_for_m2
        raise StopAfterCrop

    monkeypatch.setattr(satellites.GOES_EAST, "_crop_to_bbox", crop)
    resolved = SimpleNamespace()
    with pytest.raises(StopAfterCrop):
        satellites.GOES_EAST._compose_true_color_sync(
            {"red": resolved},
            [-80.0, 10.0, -79.0, 11.0],
            resolved,
            physical=True,
        )
    assert calls == [True]

    calls.clear()
    with pytest.raises(StopAfterCrop):
        satellites.GOES_EAST._compose_true_color_sync(
            {"red": resolved},
            [-80.0, 10.0, -79.0, 11.0],
            resolved,
            physical=False,
        )
    assert calls == [False]


def test_ahi_day_bundle_uses_one_shared_coverage_replay(monkeypatch):
    shape = (31, 47)
    y, x = np.mgrid[: shape[0], : shape[1]]
    checker = ((x + y) % 2).astype(np.float32)
    lon_axis = np.linspace(130.0, 142.0, shape[1], dtype=np.float32)
    lat_axis = np.linspace(22.0, 10.0, shape[0], dtype=np.float32)
    lons, lats = np.meshgrid(lon_axis, lat_axis)
    fields = {
        "red": 0.2 + 0.1 * checker,
        "green": 0.25 + 0.1 * checker,
        "blue": 0.3 + 0.1 * checker,
        "veggie": 0.4 + 0.1 * checker,
        "ir_bt": 245.0 + 20.0 * checker,
        "lats": lats,
        "lons": lons,
        "geom_valid": np.ones(shape, dtype=bool),
    }
    for name in ("red", "green", "blue", "veggie", "ir_bt"):
        fields[name] = np.asarray(fields[name], dtype=np.float32)

    import render_resample

    original = render_resample.resample_fields_shared_coverage
    calls = []

    def spy(bundle, coverage, plan, **kwargs):
        calls.append(tuple(array.shape for array in bundle))
        return original(bundle, coverage, plan, **kwargs)

    monkeypatch.setattr(render_resample, "resample_fields_shared_coverage", spy)
    satellites._bandlimit_truecolor_physical_cap(
        **fields,
        bbox=[130.0, 10.0, 142.0, 22.0],
        sensor="ahi",
        max_px=11,
    )

    # First call is the complete AHI day bundle: RGB plus veggie share one
    # literal coverage transform. IR and geometry are separate support modes.
    assert calls[0] == ((31, 47, 3), (31, 47))
    assert calls.count(((31, 47, 3), (31, 47))) == 1


@pytest.mark.parametrize("field_name", ["red", "blue", "lats", "geom_valid"])
def test_m2_cap_rejects_masked_array_lookalikes(field_name):
    shape = (7, 9)
    y, x = np.mgrid[: shape[0], : shape[1]]
    fields = {
        "red": np.full(shape, np.float32(0.2), dtype=np.float32),
        "green": None,
        "blue": np.full(shape, np.float32(0.3), dtype=np.float32),
        "veggie": np.full(shape, np.float32(0.4), dtype=np.float32),
        "ir_bt": np.full(shape, np.float32(250.0), dtype=np.float32),
        "lats": (20.0 - y).astype(np.float32),
        "lons": (-80.0 + x).astype(np.float32),
        "geom_valid": np.ones(shape, dtype=bool),
    }
    fields[field_name] = np.ma.array(
        fields[field_name],
        mask=np.zeros(shape, dtype=bool),
    )
    with pytest.raises(TypeError, match="exact ndarray"):
        satellites._bandlimit_truecolor_physical_cap(
            **fields,
            bbox=[-80.0, 14.0, -72.0, 20.0],
            sensor="abi",
            max_px=5,
        )
