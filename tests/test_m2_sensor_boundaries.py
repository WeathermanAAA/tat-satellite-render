"""M2 sensor-boundary tests: no stride, no mask mismatch, no fake RGB."""

from __future__ import annotations

import asyncio
import ast
import dataclasses
import datetime as dt
import inspect
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import satellites
import truecolor
from render_policy import M1_POLICY, M2_POLICY, RenderPolicy


def _grids(shape=(6, 8)):
    y, x = np.mgrid[: shape[0], : shape[1]]
    base = (0.1 + y * 0.01 + x * 0.001).astype(np.float32)
    return dict(
        red=base,
        green=(base + 0.01).astype(np.float32),
        blue=(base + 0.02).astype(np.float32),
        veggie=(base + 0.03).astype(np.float32),
        ir_bt=(250.0 + base).astype(np.float32),
        lats=(20.0 - y).astype(np.float32),
        lons=(-80.0 + x).astype(np.float32),
        geom_valid=np.ones(shape, dtype=bool),
    )


_GOES_LON_ORIGIN = -75.2
_GOES_H = 42_164_160.0
_GOES_R_EQ = 6_378_137.0
_GOES_R_POL = 6_356_752.31414


def _abi_key(channel: int, scan_token: str) -> str:
    return (
        "noaa-goes19/ABI-L2-CMIPF/2026/163/19/"
        f"OR_ABI-L2-CMIPF-M6C{channel:02d}_G19_s{scan_token}_e1_c1.nc"
    )


def test_m2_abi_sibling_lookup_rejects_nearest_scan_fallback(monkeypatch):
    requested = satellites._parse_scan_start(
        _abi_key(2, "20261631947266")
    )
    adjacent = _abi_key(1, "20261631947296")
    monkeypatch.setattr(
        satellites,
        "_list_hour",
        lambda bucket, product, band, when: [adjacent],
    )

    with pytest.raises(RuntimeError, match="requires exact same-scan ABI band 1"):
        asyncio.run(
            satellites.GOES_EAST._find_band_at(
                "noaa-goes19",
                "CMIPF",
                1,
                requested,
                require_exact=True,
            )
        )


def test_m2_abi_exact_duplicate_selection_is_deterministic(monkeypatch):
    requested = satellites._parse_scan_start(
        _abi_key(2, "20261631947266")
    )
    first = _abi_key(1, "20261631947266").replace("_c1.nc", "_c1-a.nc")
    second = _abi_key(1, "20261631947266").replace("_c1.nc", "_c1-b.nc")
    monkeypatch.setattr(
        satellites,
        "_list_hour",
        lambda bucket, product, band, when: [second, first],
    )

    resolved = asyncio.run(
        satellites.GOES_EAST._find_band_at(
            "noaa-goes19",
            "CMIPF",
            1,
            requested,
            require_exact=True,
        )
    )
    assert resolved.s3_key == first


def test_legacy_abi_sibling_lookup_retains_nearest_fallback(monkeypatch):
    requested = satellites._parse_scan_start(
        _abi_key(2, "20261631947266")
    )
    adjacent = _abi_key(1, "20261631947296")
    monkeypatch.setattr(
        satellites,
        "_list_hour",
        lambda bucket, product, band, when: [adjacent],
    )

    resolved = asyncio.run(
        satellites.GOES_EAST._find_band_at(
            "noaa-goes19",
            "CMIPF",
            1,
            requested,
        )
    )
    assert resolved.s3_key == adjacent
    assert resolved.scan_start == requested


def test_legacy_abi_duplicate_and_nearest_ties_preserve_listing_order(monkeypatch):
    requested = satellites._parse_scan_start(
        _abi_key(2, "20261631947266")
    )
    lexical_first = _abi_key(1, "20261631947266").replace(
        "_c1.nc", "_c1-a.nc"
    )
    listed_first = _abi_key(1, "20261631947266").replace(
        "_c1.nc", "_c1-b.nc"
    )
    monkeypatch.setattr(
        satellites,
        "_list_hour",
        lambda bucket, product, band, when: [listed_first, lexical_first],
    )
    exact = asyncio.run(
        satellites.GOES_EAST._find_band_at(
            "noaa-goes19", "CMIPF", 1, requested
        )
    )
    assert exact.s3_key == listed_first

    after = _abi_key(1, "20261631947296")
    before = _abi_key(1, "20261631947236")
    monkeypatch.setattr(
        satellites,
        "_list_hour",
        lambda bucket, product, band, when: [after, before],
    )
    nearest = asyncio.run(
        satellites.GOES_EAST._find_band_at(
            "noaa-goes19", "CMIPF", 1, requested
        )
    )
    assert nearest.s3_key == after


def test_physical_abi_fetch_requests_exact_siblings(monkeypatch):
    requested = satellites._parse_scan_start(
        _abi_key(2, "20261631947266")
    )
    red = satellites.GOES_EAST._make_resolved(
        "noaa-goes19",
        _abi_key(2, "20261631947266"),
        "CMIPF",
        requested,
    )
    exact_flags = []

    async def reject_lookup(bucket, product, band, scan_start, *, require_exact=False):
        exact_flags.append((band, require_exact, scan_start))
        raise RuntimeError("stop before composition")

    monkeypatch.setattr(satellites.GOES_EAST, "_find_band_at", reject_lookup)
    with pytest.raises(RuntimeError, match="stop before composition"):
        asyncio.run(
            satellites.GOES_EAST.fetch_true_color_physical(
                [-84.1, 10.3, -72.1, 22.3], red
            )
        )
    assert exact_flags
    assert all(flag is True and scan == requested for _, flag, scan in exact_flags)


def test_physical_abi_supplied_manifest_reuses_only_its_exact_objects(
    monkeypatch,
):
    requested = satellites._parse_scan_start(
        _abi_key(2, "20261631947266")
    )
    red = satellites.GOES_EAST._make_resolved(
        "noaa-goes19",
        _abi_key(2, "20261631947266"),
        "CMIPF",
        requested,
    )
    bbox = [-84.1, 10.3, -72.1, 22.3]
    role_bands = satellites.GOES_EAST._physical_truecolor_role_bands()
    keys = {
        role: red.s3_key
        if role == "red"
        else _abi_key(band, "20261631947266")
        for role, band in role_bands
    }
    manifest = satellites.TrueColorPhysicalSourceManifest(
        sensor="abi",
        bucket=red.bucket,
        product=red.product,
        scan_start=red.scan_start,
        sat_name=red.sat_name,
        resolved_anchor_s3_key=red.s3_key,
        bbox=tuple(bbox),
        sources=tuple(
            satellites.TrueColorPhysicalSource(
                role, band, (keys[role],), (keys[role],)
            )
            for role, band in role_bands
        ),
    )
    monkeypatch.setattr(
        satellites.GOES_EAST,
        "_find_band_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("supplied M2 manifest relisted ABI objects")
        ),
    )
    sentinel = object()
    compose = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(
        satellites.GOES_EAST, "_compose_true_color_sync", compose
    )
    result = asyncio.run(
        satellites.GOES_EAST.fetch_true_color_physical(
            bbox, red, source_manifest=manifest
        )
    )
    assert result is sentinel
    band_files = compose.call_args.args[0]
    assert {
        role: resolved.s3_key for role, resolved in band_files.items()
    } == keys
    assert compose.call_args.kwargs == {
        "physical": True,
        "physical_source_manifest": manifest,
    }

    unused = dataclasses.replace(
        manifest.sources[1],
        complete_scan_object_keys=(
            manifest.sources[1].object_keys[0],
            "unused-lookalike.nc",
        ),
    )
    with pytest.raises(RuntimeError, match="unconsumed scan objects"):
        dataclasses.replace(
            manifest,
            sources=(manifest.sources[0], unused, *manifest.sources[2:]),
        ).as_dict()

    blue_index = next(
        index
        for index, source in enumerate(manifest.sources)
        if source.role == "blue"
    )
    for lookalike_key in (
        _abi_key(1, "20261631948266"),
        _abi_key(2, "20261631947266"),
        _abi_key(1, "20261631947266").replace("CMIPF", "CMIPC"),
        _abi_key(2, "20261631947266").replace(
            ".nc", "_C01_.nc"
        ),
    ):
        lookalike_source = dataclasses.replace(
            manifest.sources[blue_index],
            object_keys=(lookalike_key,),
            complete_scan_object_keys=(lookalike_key,),
        )
        lookalike_sources = list(manifest.sources)
        lookalike_sources[blue_index] = lookalike_source
        with pytest.raises(RuntimeError, match="bucket/product/band/platform/scan"):
            asyncio.run(
                satellites.GOES_EAST.fetch_true_color_physical(
                    bbox,
                    red,
                    source_manifest=dataclasses.replace(
                        manifest, sources=tuple(lookalike_sources)
                    ),
                )
            )


def test_ahi_physical_manifest_binds_exact_segments_and_load_reuses_them(
    monkeypatch,
):
    from vendor import ahi_loader

    when = dt.datetime(2026, 6, 12, 12, tzinfo=dt.timezone.utc)
    resolved = satellites.ResolvedFile(
        bucket="noaa-himawari9",
        s3_key="noaa-himawari9/AHI-L1b-FLDK/2026/06/12/1200/",
        product="FLDK",
        scan_start=when,
        sat_name="Himawari-9",
        sub_sat_lon=140.7,
    )
    monkeypatch.setattr(satellites, "_get_fs", lambda: object())

    def resolve_paths(_fs, bucket, scan_start, band, bbox):
        assert bucket == resolved.bucket
        assert scan_start == when
        assert bbox == (120.0, 10.0, 132.0, 22.0)
        path = (
            f"{bucket}/AHI-L1b-FLDK/2026/06/12/1200/"
            f"HS_H09_20260612_1200_B{band:02d}_FLDK_"
            f"{ahi_loader.BAND_RES_SUFFIX[band]}_S0101.DAT.bz2"
        )
        return (path,), (path,)

    monkeypatch.setattr(ahi_loader, "resolve_band_paths_sync", resolve_paths)
    manifest = satellites.HIMAWARI_PACIFIC._resolve_true_color_physical_sources_sync(
        [120.0, 10.0, 132.0, 22.0], resolved
    )
    assert manifest.resolved_anchor_s3_key == resolved.s3_key
    assert tuple((row.role, row.band) for row in manifest.sources) == (
        ("red", 3),
        ("green", 2),
        ("blue", 1),
        ("veggie", 4),
        ("ir", 13),
    )

    red_paths = manifest.sources[0].object_keys
    monkeypatch.setattr(
        ahi_loader,
        "_list_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pre-resolved M2 load relisted AHI objects")
        ),
    )
    observed = {}
    monkeypatch.setattr(
        ahi_loader,
        "_download_all",
        lambda fs, paths: observed.setdefault("paths", tuple(paths)) or [object()],
    )
    sentinel = object()
    monkeypatch.setattr(ahi_loader, "_assemble_disk", lambda *args, **kwargs: sentinel)
    loaded = ahi_loader.load_band_sync(
        object(),
        resolved.bucket,
        when,
        3,
        bbox=(120.0, 10.0, 132.0, 22.0),
        resolved_paths=red_paths,
        complete_resolved_paths=manifest.sources[0].complete_scan_object_keys,
    )
    assert loaded is sentinel
    assert observed["paths"] == red_paths


def _ahi_fldk_path(
    *, band: int, sequence: int, total: int = 10, stamp: str = "20260612_1200"
) -> str:
    from vendor.ahi_loader import BAND_RES_SUFFIX

    return (
        "noaa-himawari9/AHI-L1b-FLDK/2026/06/12/1200/"
        f"HS_H09_{stamp}_B{band:02d}_FLDK_{BAND_RES_SUFFIX[band]}_"
        f"S{sequence:02d}{total:02d}.DAT.bz2"
    )


def test_ahi_m2_segment_binding_rejects_wrong_or_incomplete_object_sets(
    monkeypatch,
):
    from vendor import ahi_loader

    when = dt.datetime(2026, 6, 12, 12, tzinfo=dt.timezone.utc)
    bbox = (120.0, 10.0, 132.0, 22.0)
    complete = tuple(
        _ahi_fldk_path(band=3, sequence=sequence)
        for sequence in range(1, 11)
    )
    selected = tuple(
        ahi_loader._filter_segments_for_bbox(
            list(complete), bbox, 3, strict=True
        )
    )
    assert selected
    assert ahi_loader._require_resolved_band_paths(
        selected,
        complete,
        "noaa-himawari9",
        when,
        3,
        bbox,
    ) == selected

    with pytest.raises(RuntimeError, match="missing segments"):
        ahi_loader._require_resolved_band_paths(
            selected,
            complete[:-1],
            "noaa-himawari9",
            when,
            3,
            bbox,
        )
    malformed = (complete[0].replace("_S0110", "_S9910"),) + complete[1:]
    with pytest.raises(RuntimeError, match="bound slot/band|sequence layout"):
        ahi_loader._require_resolved_band_paths(
            selected, malformed, "noaa-himawari9", when, 3, bbox
        )
    wrong_stamp = (
        complete[0].replace(
            "HS_H09_20260612_1200", "HS_H09_19990101_0000"
        ),
    ) + complete[1:]
    with pytest.raises(RuntimeError, match="platform/date/slot/band"):
        ahi_loader._require_resolved_band_paths(
            selected, wrong_stamp, "noaa-himawari9", when, 3, bbox
        )
    junk = (complete[0].replace("HS_H09_", "junk_HS_H09_"),) + complete[1:]
    with pytest.raises(
        RuntimeError, match="bound slot/band|platform/date/slot/band"
    ):
        ahi_loader._require_resolved_band_paths(
            selected, junk, "noaa-himawari9", when, 3, bbox
        )
    nested = (
        complete[0].replace("/1200/", "/1200/lookalike/"),
    ) + complete[1:]
    with pytest.raises(RuntimeError, match="bound slot/band"):
        ahi_loader._require_resolved_band_paths(
            selected, nested, "noaa-himawari9", when, 3, bbox
        )
    unregistered_two_segment = tuple(
        _ahi_fldk_path(band=3, sequence=sequence, total=2)
        for sequence in (1, 2)
    )
    with pytest.raises(RuntimeError, match="registered NOAA layout"):
        ahi_loader._require_ahi_segment_layout(
            unregistered_two_segment,
            "noaa-himawari9",
            when,
            3,
            require_complete=True,
        )
    with pytest.raises(RuntimeError, match="bound bbox"):
        ahi_loader._require_resolved_band_paths(
            selected[:-1], complete, "noaa-himawari9", when, 3, bbox
        )
    other_bbox = (120.0, -45.0, 132.0, -35.0)
    assert tuple(
        ahi_loader._filter_segments_for_bbox(
            list(complete), other_bbox, 3, strict=True
        )
    ) != selected
    with pytest.raises(RuntimeError, match="bound bbox"):
        ahi_loader._require_resolved_band_paths(
            selected, complete, "noaa-himawari9", when, 3, other_bbox
        )

    monkeypatch.setattr(ahi_loader, "_list_segments", lambda *args: list(complete[:-1]))
    with pytest.raises(RuntimeError, match="missing segments"):
        ahi_loader.resolve_band_paths_sync(
            object(), "noaa-himawari9", when, 3, bbox
        )


def test_ahi_physical_compose_passes_every_bound_object_set_without_relisting(
    monkeypatch,
):
    from vendor import ahi_loader

    when = dt.datetime(2026, 6, 12, 12, tzinfo=dt.timezone.utc)
    bbox = [120.0, 10.0, 132.0, 22.0]
    resolved = satellites.ResolvedFile(
        bucket="noaa-himawari9",
        s3_key="noaa-himawari9/AHI-L1b-FLDK/2026/06/12/1200/",
        product="FLDK",
        scan_start=when,
        sat_name="Himawari-9",
        sub_sat_lon=140.7,
    )
    role_bands = satellites.HIMAWARI_PACIFIC._physical_truecolor_role_bands()
    manifest = satellites.TrueColorPhysicalSourceManifest(
        sensor="ahi",
        bucket=resolved.bucket,
        product=resolved.product,
        scan_start=resolved.scan_start,
        sat_name=resolved.sat_name,
        resolved_anchor_s3_key=resolved.s3_key,
        bbox=tuple(bbox),
        sources=tuple(
            satellites.TrueColorPhysicalSource(
                role,
                band,
                (_ahi_fldk_path(band=band, sequence=1, total=1),),
                (_ahi_fldk_path(band=band, sequence=1, total=1),),
            )
            for role, band in role_bands
        ),
    )
    observed = {}

    def fake_load(
        fs,
        bucket,
        scan_start,
        band,
        *,
        bbox,
        resolved_paths,
        complete_resolved_paths,
    ):
        observed[band] = (resolved_paths, complete_resolved_paths)
        return object()

    class StopAfterLoads(RuntimeError):
        pass

    monkeypatch.setattr(satellites, "_get_fs", lambda: object())
    monkeypatch.setattr(ahi_loader, "load_band_sync", fake_load)
    monkeypatch.setattr(
        satellites, "_truecolor_native_target_dims", lambda *_args: (4, 4)
    )
    monkeypatch.setattr(
        satellites,
        "_guard_m2_truecolor_native_grid",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        satellites,
        "_truecolor_target_longitudes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterLoads()),
    )
    with pytest.raises(StopAfterLoads):
        satellites.HIMAWARI_PACIFIC._compose_true_color_sync(
            bbox,
            resolved,
            physical=True,
            physical_source_manifest=manifest,
        )
    assert set(observed) == {band for _, band in role_bands}
    for source in manifest.sources:
        assert observed[source.band] == (
            source.object_keys,
            source.complete_scan_object_keys,
        )


def test_abi_forward_projection_masks_backside_but_preserves_visible_bits():
    """The PUG scan equations need a separate ellipsoid occlusion test."""

    lats = np.array([[0.0, 15.0, -25.0]], dtype=np.float64)
    lons = np.array(
        [[_GOES_LON_ORIGIN, -60.0, -95.0]], dtype=np.float64
    )
    raw_x, raw_y = satellites._latlon_to_xy_unchecked(
        lats,
        lons,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    visible_x, visible_y = satellites._latlon_to_xy_visible(
        lats,
        lons,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    assert np.array_equal(visible_x, raw_x)
    assert np.array_equal(visible_y, raw_y)

    backside_lon = np.array([_GOES_LON_ORIGIN + 180.0])
    mutant_x, mutant_y = satellites._latlon_to_xy_unchecked(
        np.array([0.0]),
        backside_lon,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    corrected_x, corrected_y = satellites._latlon_to_xy_visible(
        np.array([0.0]),
        backside_lon,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    # The caught mutant omits occlusion and aliases the far-side equator near
    # the centre of the front-side scan window.
    assert np.isfinite(mutant_x[0]) and np.isfinite(mutant_y[0])
    assert abs(mutant_x[0]) < 1.0e-12
    assert np.isnan(corrected_x[0]) and np.isnan(corrected_y[0])


def test_abi_historical_helper_stays_unchecked_for_legacy_and_s2_callers():
    latitudes = np.array([0.0, 12.5, 0.0], dtype=np.float64)
    longitudes = np.array(
        [_GOES_LON_ORIGIN, -52.0, _GOES_LON_ORIGIN + 180.0],
        dtype=np.float64,
    )
    expected = satellites._latlon_to_xy_unchecked(
        latitudes,
        longitudes,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    historical = satellites._latlon_to_xy(
        latitudes,
        longitudes,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    assert np.array_equal(historical[0], expected[0], equal_nan=True)
    assert np.array_equal(historical[1], expected[1], equal_nan=True)
    assert np.isfinite(historical[0][-1]) and np.isfinite(historical[1][-1])

    physical = satellites._latlon_to_xy_visible(
        latitudes,
        longitudes,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    assert np.array_equal(physical[0][:-1], historical[0][:-1])
    assert np.array_equal(physical[1][:-1], historical[1][:-1])
    assert np.isnan(physical[0][-1]) and np.isnan(physical[1][-1])

    compose_source = inspect.getsource(
        satellites.GOESBaseSatellite._compose_true_color_sync
    )
    crop_source = inspect.getsource(satellites.GOESBaseSatellite._crop_to_bbox)
    assert "_latlon_to_xy_visible if physical else _latlon_to_xy" in compose_source
    assert "_latlon_to_xy_visible" in crop_source
    assert "else _latlon_to_xy" in crop_source

    import s2_imagery

    for s2_producer in (
        s2_imagery.sample_bands_to_grid,
        s2_imagery.produce_global_composite,
    ):
        s2_source = inspect.getsource(s2_producer)
        assert "from satellites import _latlon_to_xy" in s2_source
        assert "_latlon_to_xy_visible" not in s2_source


def test_abi_visibility_discriminant_front_tangent_and_behind_limb():
    limb_delta_deg = np.rad2deg(np.arccos(_GOES_R_EQ / _GOES_H))
    delta = np.array(
        [limb_delta_deg - 1.0e-5, limb_delta_deg, limb_delta_deg + 1.0e-5]
    )
    lat = np.zeros(3, dtype=np.float64)
    lon = _GOES_LON_ORIGIN + delta
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    phi_c = np.arctan(
        ((_GOES_R_POL**2) / (_GOES_R_EQ**2)) * np.tan(lat_rad)
    )
    eccentricity = np.sqrt(
        1.0 - (_GOES_R_POL**2) / (_GOES_R_EQ**2)
    )
    rc = _GOES_R_POL / np.sqrt(
        1.0 - eccentricity**2 * np.cos(phi_c) ** 2
    )
    sx = _GOES_H - rc * np.cos(phi_c) * np.cos(
        lon_rad - np.deg2rad(_GOES_LON_ORIGIN)
    )
    sy = -rc * np.cos(phi_c) * np.sin(
        lon_rad - np.deg2rad(_GOES_LON_ORIGIN)
    )
    sz = rc * np.sin(phi_c)
    discriminant = satellites._goes_line_of_sight_discriminant(
        sx, sy, sz, _GOES_H, _GOES_R_EQ, _GOES_R_POL
    )
    assert discriminant[0] > 0.0
    assert discriminant[1] >= 0.0
    assert discriminant[2] < 0.0

    x, y = satellites._latlon_to_xy_visible(
        lat,
        lon,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    assert np.all(np.isfinite(x[:2])) and np.all(np.isfinite(y[:2]))
    assert np.isnan(x[2]) and np.isnan(y[2])


def _ahi_forward_components_at_equator(delta_deg: np.ndarray):
    distance = satellites._AHI_SATELLITE_DISTANCE_KM
    radius = satellites._AHI_EQUATORIAL_RADIUS_KM
    delta = np.deg2rad(np.asarray(delta_deg, dtype=np.float64))
    r1 = distance - radius * np.cos(delta)
    r2 = -radius * np.sin(delta)
    r3 = np.zeros(delta.shape, dtype=np.float64)
    return r1, r2, r3


def test_ahi_visibility_discriminant_front_tangent_and_behind_limb():
    limb_delta_deg = np.rad2deg(
        np.arccos(
            satellites._AHI_EQUATORIAL_RADIUS_KM
            / satellites._AHI_SATELLITE_DISTANCE_KM
        )
    )
    delta = np.array(
        [limb_delta_deg - 1.0e-5, limb_delta_deg, limb_delta_deg + 1.0e-5]
    )
    r1, r2, r3 = _ahi_forward_components_at_equator(delta)
    discriminant = satellites._ahi_line_of_sight_discriminant(r1, r2, r3)
    visible = satellites._ahi_visible_from_discriminant(
        discriminant, r1, r2, r3
    )
    assert discriminant[0] > 0.0
    assert abs(discriminant[1]) < 1.0e-6
    assert discriminant[2] < 0.0
    assert np.array_equal(visible, np.array([True, True, False]))

    lats = np.zeros(3, dtype=np.float64)
    lons = 140.7 + delta
    x, y = satellites._ahi_latlon_to_xy_deg_visible(lats, lons, 140.7)
    assert np.all(np.isfinite(x[:2])) and np.all(np.isfinite(y[:2]))
    assert np.isnan(x[2]) and np.isnan(y[2])


def test_ahi_physical_visibility_kills_farside_alias_and_preserves_legacy_bits():
    lats = np.array([0.0, 20.0, 0.0], dtype=np.float64)
    lons = np.array([140.7, 155.0, 140.7 + 180.0], dtype=np.float64)
    historical = satellites._ahi_latlon_to_xy_deg(lats, lons, 140.7)
    physical = satellites._ahi_latlon_to_xy_deg_visible(lats, lons, 140.7)

    assert np.array_equal(physical[0][:-1], historical[0][:-1])
    assert np.array_equal(physical[1][:-1], historical[1][:-1])
    # The unchecked far-side equator aliases almost exactly to nadir.
    assert np.isfinite(historical[0][-1]) and abs(historical[0][-1]) < 0.1
    assert np.isfinite(historical[1][-1]) and abs(historical[1][-1]) < 0.1
    assert np.isnan(physical[0][-1]) and np.isnan(physical[1][-1])

    import s2_imagery

    for s2_producer in (
        s2_imagery._ahi_target_mesh,
        s2_imagery.produce_global_composite,
    ):
        s2_source = inspect.getsource(s2_producer)
        assert "_ahi_latlon_to_xy_deg" in s2_source
        assert "_ahi_latlon_to_xy_deg_visible" not in s2_source


def test_ahi_can_see_approximate_boundary_cannot_create_physical_support():
    when = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    # The historical picker deliberately admits centres just inside 85 degrees,
    # beyond the WGS84 equatorial tangent at roughly 81.30 degrees. M2 must
    # retain that selection behavior but reject the occluded pixels downstream.
    admitted_center = ((140.7 + 84.9 + 180.0) % 360.0) - 180.0
    admitted_bbox = [
        admitted_center - 0.05,
        -0.05,
        admitted_center + 0.05,
        0.05,
    ]
    assert satellites.HIMAWARI_PACIFIC.can_see(admitted_bbox, when)
    admitted_x, admitted_y = satellites._ahi_latlon_to_xy_deg_visible(
        np.array([0.0]), np.array([admitted_center]), 140.7
    )
    assert np.isnan(admitted_x[0]) and np.isnan(admitted_y[0])

    rejected_center = ((140.7 + 85.1 + 180.0) % 360.0) - 180.0
    rejected_bbox = [
        rejected_center - 0.05,
        -0.05,
        rejected_center + 0.05,
        0.05,
    ]
    assert not satellites.HIMAWARI_PACIFIC.can_see(rejected_bbox, when)

    compose_source = inspect.getsource(
        satellites.HimawariPacificSatellite._compose_true_color_sync
    )
    assert "_ahi_latlon_to_xy_deg_visible" in compose_source
    assert "if physical" in compose_source
    assert "else _ahi_latlon_to_xy_deg" in compose_source
    assert "himawari_ellipsoid_line_of_sight_and_red_crop_v1" in compose_source


def test_melissa_limb_bbox_geom_support_follows_actual_goes_visibility():
    """Exercise the real V41 bbox without loading its four large ABI files."""

    west, south, east, north = (-159.084, 38.8, -147.084, 50.8)
    lons = np.linspace(west, east, 241)
    lats = np.linspace(north, south, 241)
    tlon, tlat = np.meshgrid(lons, lats)
    x, y = satellites._latlon_to_xy_visible(
        tlat,
        tlon,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    raw_x, raw_y = satellites._latlon_to_xy_unchecked(
        tlat,
        tlon,
        _GOES_LON_ORIGIN,
        _GOES_H,
        _GOES_R_EQ,
        _GOES_R_POL,
    )
    # Full-disk ABI scan limits are wider than every valid query in this box;
    # this isolates line-of-sight support from crop-window support.
    scan_limit = 0.152
    full_disk_x = np.array([-scan_limit, scan_limit], dtype=np.float64)
    full_disk_y = np.array([scan_limit, -scan_limit], dtype=np.float64)
    geom_valid = satellites._abi_regular_grid_geometric_support(
        x, y, full_disk_x, full_disk_y
    )
    mutant_geom = satellites._abi_regular_grid_geometric_support(
        raw_x, raw_y, full_disk_x, full_disk_y
    )
    assert np.any(geom_valid)
    assert np.any(~geom_valid)
    assert np.count_nonzero(geom_valid[1:] != geom_valid[:-1]) > 0
    assert np.count_nonzero(mutant_geom & ~geom_valid) > 10_000


def test_legacy_stride_keeps_geom_valid_coregistered():
    grids = _grids()
    expected_shape = (3, 4)

    def legacy(red, green, blue, veggie, lats, lons, **kwargs):
        assert red.shape == green.shape == blue.shape == veggie.shape
        assert red.shape == lats.shape == lons.shape == expected_shape
        assert kwargs["ir_bt"].shape == expected_shape
        return np.dstack((red, green, blue)), np.ones(expected_shape, np.float32)

    with mock.patch.object(truecolor, "assemble_truecolor", side_effect=legacy):
        rgb, lats, lons, cos_sza, geom_valid, state = (
            satellites._finish_truecolor_grid(
                **grids,
                downsample=2,
                physical=False,
                assembly_kwargs={},
            )
        )

    assert rgb.shape == (*expected_shape, 3)
    assert lats.shape == lons.shape == cos_sza.shape == expected_shape
    assert geom_valid.shape == expected_shape
    assert state is None


def test_m2_physical_boundary_bypasses_point_stride():
    grids = _grids()
    shape = grids["red"].shape
    day_rgb = np.dstack((grids["red"], grids["green"], grids["blue"]))
    state = SimpleNamespace(
        day_rgb_reflectance=day_rgb,
        day_coverage=np.ones(shape, np.float32),
        ir_bt_kelvin=grids["ir_bt"],
        ir_coverage=np.ones(shape, np.float32),
        cos_sza=np.ones(shape, np.float32),
        geom_valid=grids["geom_valid"],
        red_corrected=grids["red"],
        veggie_corrected=grids["veggie"],
        glint_deg=np.zeros(shape, np.float32),
    )

    def physical(red, green, blue, veggie, lats, lons, **kwargs):
        assert red.shape == green.shape == blue.shape == veggie.shape == shape
        assert lats.shape == lons.shape == kwargs["ir_bt"].shape == shape
        assert kwargs["geom_valid"].shape == shape
        return state

    with mock.patch.object(
        truecolor, "assemble_truecolor_physical", side_effect=physical, create=True
    ):
        rgb, lats, lons, cos_sza, geom_valid, returned = (
            satellites._finish_truecolor_grid(
                **grids,
                downsample=3,
                physical=True,
                assembly_kwargs={},
            )
        )

    assert returned is state
    assert rgb.shape == (*shape, 3)
    assert lats.shape == lons.shape == cos_sza.shape == geom_valid.shape == shape


def test_shape_mismatch_fails_before_recipe_or_render():
    grids = _grids()
    grids["geom_valid"] = np.ones((12, 16), dtype=bool)
    with mock.patch.object(truecolor, "assemble_truecolor") as legacy:
        with pytest.raises(ValueError, match="exactly co-registered"):
            satellites._finish_truecolor_grid(
                **grids,
                downsample=2,
                physical=False,
                assembly_kwargs={},
            )
    legacy.assert_not_called()


def test_physical_fetch_result_is_explicitly_tagged():
    state = object()
    result = satellites.FetchResult(
        cmi=np.zeros((2, 3, 3), np.float32),
        lats=np.zeros((2, 3), np.float32),
        lons=np.zeros((2, 3), np.float32),
        channel=2,
        generic_channel="true_color",
        scan_start=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        product="TEST",
        bucket="test",
        sat_name="TEST",
        sub_sat_lon=0.0,
        units="truecolor_physical",
        truecolor_physical=state,
    )
    assert result.units == "truecolor_physical"
    assert result.truecolor_physical is state


def test_app_policy_selects_physical_fetch_without_legacy_stride():
    import app

    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    source_manifest = satellites.TrueColorPhysicalSourceManifest(
        sensor="abi",
        bucket="test",
        product="TEST",
        scan_start=now,
        sat_name="TEST",
        resolved_anchor_s3_key="red.nc",
        bbox=(-80.0, 10.0, -70.0, 20.0),
        sources=(
            satellites.TrueColorPhysicalSource(
                "red", 2, ("red.nc",), ("red.nc",)
            ),
        ),
    )

    class FakeSatellite:
        family = "TEST"

        def __init__(self):
            self.calls = []

        async def fetch_true_color(self, bbox, resolved, downsample):
            self.calls.append(("legacy", downsample))
            return "legacy"

        async def fetch_true_color_physical(
            self, bbox, resolved, *, source_manifest=None
        ):
            self.calls.append(("physical", source_manifest))
            return "physical"

    satellite = FakeSatellite()
    got = asyncio.run(
        app._fetch_true_color_for_policy(
            satellite,
            [-80.0, 10.0, -70.0, 20.0],
            object(),
            4,
            policy=M2_POLICY,
            source_manifest=source_manifest,
        )
    )
    assert got == "physical"
    assert satellite.calls == [("physical", source_manifest)]

    got = asyncio.run(
        app._fetch_true_color_for_policy(
            satellite, [-80.0, 10.0, -70.0, 20.0], object(), 4, policy=M1_POLICY
        )
    )
    assert got == "legacy"
    assert satellite.calls[-1] == ("legacy", 4)


def test_response_point_stride_does_not_mislabel_m2_delivery_reduction():
    import app

    assert app._reported_point_stride(
        4, is_true_color=True, policy=M2_POLICY
    ) == 1
    assert app._reported_point_stride(
        4, is_true_color=True, policy=M1_POLICY
    ) == 4
    assert app._reported_point_stride(
        4, is_true_color=False, policy=M2_POLICY
    ) == 1


def test_app_binds_physical_sensor_to_independent_satellite_authority():
    import app

    shape = (3, 4)
    scan_start = dt.datetime(2026, 6, 12, 12, tzinfo=dt.timezone.utc)
    resolved = satellites.ResolvedFile(
        bucket="noaa-goes19",
        s3_key="red.nc",
        product="CMIPF",
        scan_start=scan_start,
        sat_name="GOES-19",
        sub_sat_lon=-75.2,
    )
    source_manifest = satellites.TrueColorPhysicalSourceManifest(
        sensor="abi",
        bucket=resolved.bucket,
        product=resolved.product,
        scan_start=resolved.scan_start,
        sat_name=resolved.sat_name,
        resolved_anchor_s3_key=resolved.s3_key,
        bbox=(-80.0, 10.0, -70.0, 20.0),
        sources=tuple(
            satellites.TrueColorPhysicalSource(role, band, (key,), (key,))
            for role, band, key in (
                ("red", 2, "red.nc"),
                ("blue", 1, "blue.nc"),
                ("veggie", 3, "veggie.nc"),
                ("ir", 13, "ir.nc"),
            )
        ),
    )
    scalar = np.full(shape, np.float32(0.2), dtype=np.float32)
    state = truecolor.TrueColorPhysicalState(
        day_rgb_reflectance=np.repeat(scalar[..., None], 3, axis=-1),
        day_coverage=np.ones(shape, dtype=np.float32),
        ir_bt_kelvin=None,
        ir_coverage=None,
        cos_sza=np.ones(shape, dtype=np.float32),
        day_weight=np.ones(shape, dtype=np.float32),
        geom_valid=np.ones(shape, dtype=bool),
        red_corrected=scalar.copy(),
        veggie_corrected=scalar.copy(),
        glint_deg=np.full(shape, np.float32(90.0), dtype=np.float32),
        sensor="abi",
        grid_metadata={
            "grid": "regular_latlon",
            "row_order": "north_to_south",
            "sensor": "abi",
            "point_stride_applied": False,
            "physical_source_manifest": source_manifest.as_dict(),
        },
        extent=(-80.0, 10.0, -70.0, 20.0),
        projection="PlateCarree",
    )
    producer = satellites.GOES_EAST
    assert producer.sensor == "ABI"
    data = satellites.FetchResult(
        cmi=state.day_rgb_reflectance,
        lats=np.zeros(shape, dtype=np.float32),
        lons=np.zeros(shape, dtype=np.float32),
        channel=2,
        generic_channel="true_color",
        scan_start=scan_start,
        product=resolved.product,
        bucket=resolved.bucket,
        sat_name=resolved.sat_name,
        sub_sat_lon=resolved.sub_sat_lon,
        units="truecolor_physical",
        cos_sza=state.cos_sza,
        geom_valid=state.geom_valid,
        truecolor_physical=state,
    )
    app._require_m2_fetch_sensor_binding(
        data,
        producer,
        policy=M2_POLICY,
        source_manifest=source_manifest,
        resolved=resolved,
    )

    paired_relabel = dataclasses.replace(
        state,
        sensor="ahi",
        grid_metadata={**state.grid_metadata, "sensor": "ahi"},
    )
    with pytest.raises(RuntimeError, match="selected satellite"):
        app._require_m2_fetch_sensor_binding(
            dataclasses.replace(data, truecolor_physical=paired_relabel),
            producer,
            policy=M2_POLICY,
            source_manifest=source_manifest,
            resolved=resolved,
        )
    with pytest.raises(RuntimeError, match="exact ABI/AHI"):
        app._require_m2_fetch_sensor_binding(
            data,
            SimpleNamespace(sensor="abi"),
            policy=M2_POLICY,
            source_manifest=source_manifest,
            resolved=resolved,
        )
    missing_manifest = dataclasses.replace(
        state,
        grid_metadata={
            key: value
            for key, value in state.grid_metadata.items()
            if key != "physical_source_manifest"
        },
    )
    with pytest.raises(RuntimeError, match="selected satellite"):
        app._require_m2_fetch_sensor_binding(
            dataclasses.replace(data, truecolor_physical=missing_manifest),
            producer,
            policy=M2_POLICY,
            source_manifest=source_manifest,
            resolved=resolved,
        )
    with pytest.raises(RuntimeError, match="FetchResult provenance"):
        app._require_m2_fetch_sensor_binding(
            dataclasses.replace(data, product="CMIPC"),
            producer,
            policy=M2_POLICY,
            source_manifest=source_manifest,
            resolved=resolved,
        )
    # The rollback path has no physical payload and remains untouched.
    app._require_m2_fetch_sensor_binding(
        SimpleNamespace(), producer, policy=M1_POLICY
    )


def test_effective_policy_scopes_m2_without_silent_curvilinear_fallback():
    import app

    common = dict(
        process_policy=M2_POLICY,
        crosses_antimeridian=False,
        is_backdrop=False,
        output_format="png",
        m2_truecolor_supported=True,
        m2_archive_scalar_supported=True,
    )
    assert app._effective_render_policy(
        **common, is_true_color=True, use_archive=False
    ) is M2_POLICY
    assert app._effective_render_policy(
        **common, is_true_color=False, use_archive=True
    ) is M2_POLICY
    assert app._effective_render_policy(
        **common, is_true_color=False, use_archive=False
    ) is M1_POLICY
    assert app._effective_render_policy(
        **(common | {"crosses_antimeridian": True}),
        is_true_color=False,
        use_archive=True,
    ) is M1_POLICY
    assert app._effective_render_policy(
        **(common | {"is_backdrop": True}),
        is_true_color=False,
        use_archive=True,
    ) is M1_POLICY
    assert app._effective_render_policy(
        **(common | {"crosses_antimeridian": True}),
        is_true_color=True,
        use_archive=False,
    ) is M1_POLICY
    assert app._effective_render_policy(
        **(common | {"m2_truecolor_supported": False}),
        is_true_color=True,
        use_archive=False,
    ) is M1_POLICY
    assert app._effective_render_policy(
        **(common | {"is_backdrop": True}),
        is_true_color=True,
        use_archive=False,
    ) is M1_POLICY


def test_m2_archive_scalar_scope_excludes_mergir_post_resolution_fallback():
    import app
    import gridsat
    import gridsat_goes
    import mergir

    when = dt.datetime(2005, 8, 28, 18, tzinfo=dt.timezone.utc)
    rows = (
        (
            gridsat.GRIDSAT,
            satellites.ResolvedFile(
                "noaa-cdr-gridsat-b1",
                "GRIDSAT-B1.2005.08.28.18.v02r01.nc",
                "GRIDSAT-B1",
                when,
                "GridSat-B1",
                0.0,
            ),
            True,
        ),
        (
            gridsat_goes.GRIDSAT_GOES,
            satellites.ResolvedFile(
                "noaa-gridsat-goes",
                "goes/2005/example.nc",
                "GridSat-GOES",
                when,
                "GridSat-GOES",
                -75.0,
            ),
            True,
        ),
        (
            mergir.MERGIR,
            satellites.ResolvedFile(
                "gesdisc-mergir",
                "https://example.invalid/merg_2005082818_4km-pixel.nc4",
                "MERGIR",
                when,
                "MergIR",
                0.0,
            ),
            False,
        ),
    )
    for producer, resolved, expected in rows:
        assert app._m2_archive_scalar_source_supported(producer, resolved) is expected
    assert app._m2_archive_scalar_source_supported(
        SimpleNamespace(family="MergIR"), rows[-1][1]
    ) is False


def test_mergir_alternate_slot_is_load_bearing_reason_for_m1_scope(
    monkeypatch,
):
    import app
    import mergir

    requested = dt.datetime(2005, 8, 28, 18, tzinfo=dt.timezone.utc)
    nominal_url, _ = mergir.url_for(requested)
    resolved = satellites.ResolvedFile(
        "gesdisc-mergir",
        nominal_url,
        "MERGIR",
        requested,
        "MergIR",
        0.0,
    )
    alternate = requested - dt.timedelta(minutes=30)
    alternate_url, _ = mergir.url_for(alternate)

    def download(url):
        if url == alternate_url:
            return b"alternate-slot"
        raise FileNotFoundError(url)

    monkeypatch.setattr(mergir, "download_granule", download)
    monkeypatch.setattr(
        mergir,
        "crop_from_bytes",
        lambda *_args, **_kwargs: (
            np.ones((2, 3), dtype=np.float32),
            np.array((10.0, 11.0)),
            np.array((-80.0, -79.0, -78.0)),
        ),
    )
    fetched = mergir.MERGIR._fetch_sync(
        resolved, [-80.0, 10.0, -78.0, 11.0], "clean_ir"
    )
    assert fetched.scan_start == alternate
    assert fetched.scan_start != resolved.scan_start
    supported = app._m2_archive_scalar_source_supported(
        mergir.MERGIR, resolved
    )
    assert supported is False
    assert app._effective_render_policy(
        process_policy=M2_POLICY,
        is_true_color=False,
        use_archive=True,
        crosses_antimeridian=False,
        is_backdrop=False,
        output_format="png",
        m2_truecolor_supported=True,
        m2_archive_scalar_supported=supported,
    ) is M1_POLICY


def test_m2_scalar_postfetch_binding_rejects_provenance_drift():
    import app
    import gridsat

    when = dt.datetime(2005, 8, 28, 18, tzinfo=dt.timezone.utc)
    resolved = satellites.ResolvedFile(
        gridsat.GRIDSAT_BUCKET,
        "gridsat/2005/GRIDSAT-B1.2005.08.28.18.v02r01.nc",
        "GRIDSAT-B1",
        when,
        "GridSat-B1",
        0.0,
    )
    values = np.ones((2, 3), dtype=np.float32)
    data = satellites.FetchResult(
        cmi=values,
        lats=values.copy(),
        lons=values.copy(),
        channel=1,
        generic_channel="clean_ir",
        scan_start=when,
        product=resolved.product,
        bucket=resolved.bucket,
        sat_name=resolved.sat_name,
        sub_sat_lon=resolved.sub_sat_lon,
        units="K",
        source_s3_key=resolved.s3_key,
    )
    app._require_m2_scalar_fetch_binding(
        data,
        gridsat.GRIDSAT,
        resolved,
        "clean_ir",
        policy=M2_POLICY,
    )
    mutations = (
        {"product": "GridSat-GOES"},
        {"bucket": "lookalike-bucket"},
        {"scan_start": when + dt.timedelta(hours=3)},
        {"sat_name": "GridSat-lookalike"},
        {"sub_sat_lon": 1.0},
        {"generic_channel": "wv_upper"},
        {"channel": 2},
        {"source_s3_key": resolved.s3_key + ".reprocessed"},
    )
    for mutation in mutations:
        with pytest.raises(RuntimeError, match="pre-cache object"):
            app._require_m2_scalar_fetch_binding(
                dataclasses.replace(data, **mutation),
                gridsat.GRIDSAT,
                resolved,
                "clean_ir",
                policy=M2_POLICY,
            )
    app._require_m2_scalar_fetch_binding(
        SimpleNamespace(),
        gridsat.GRIDSAT,
        resolved,
        "clean_ir",
        policy=M1_POLICY,
    )


def test_m2_bbox_preflight_binds_policy_before_cache_identity():
    import app

    tree = ast.parse(inspect.getsource(app))
    calls = []
    request_policy_call = None
    source_resolution_call = None
    request_key_call = None
    physical_fetch_call = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id in {
                "m2_truecolor_bbox_preflight",
                "_m2_archive_scalar_source_supported",
                "_effective_render_policy",
                "_request_key",
                "_fetch_true_color_for_policy",
            }:
                calls.append((node.func.id, node.lineno, node))
            if node.func.id == "_effective_render_policy":
                request_policy_call = node
            elif node.func.id == "_request_key":
                request_key_call = node
            elif node.func.id == "_fetch_true_color_for_policy":
                physical_fetch_call = node
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "resolve_true_color_physical_sources"
        ):
            source_resolution_call = node
            calls.append((node.func.attr, node.lineno, node))
    lines = {name: line for name, line, _ in calls}
    assert lines["m2_truecolor_bbox_preflight"] < lines["_effective_render_policy"]
    assert (
        lines["_m2_archive_scalar_source_supported"]
        < lines["_effective_render_policy"]
    )
    assert (
        lines["_effective_render_policy"]
        < lines["resolve_true_color_physical_sources"]
        < lines["_request_key"]
        < lines["_fetch_true_color_for_policy"]
    )
    assert request_policy_call is not None
    support = next(
        keyword.value
        for keyword in request_policy_call.keywords
        if keyword.arg == "m2_truecolor_supported"
    )
    assert isinstance(support, ast.Name)
    assert support.id == "m2_truecolor_supported"
    scalar_support = next(
        keyword.value
        for keyword in request_policy_call.keywords
        if keyword.arg == "m2_archive_scalar_supported"
    )
    assert isinstance(scalar_support, ast.Name)
    assert scalar_support.id == "m2_archive_scalar_supported"
    assert source_resolution_call is not None
    assert request_key_call is not None
    assert physical_fetch_call is not None
    cache_manifest = next(
        keyword.value
        for keyword in request_key_call.keywords
        if keyword.arg == "resolved_source_manifest"
    )
    fetch_manifest = next(
        keyword.value
        for keyword in physical_fetch_call.keywords
        if keyword.arg == "source_manifest"
    )
    assert isinstance(cache_manifest, ast.Name)
    assert isinstance(fetch_manifest, ast.Name)
    assert cache_manifest.id == fetch_manifest.id == "physical_source_manifest"


def test_truecolor_backdrop_is_rejected_before_resolution_or_fetch(monkeypatch):
    import app

    monkeypatch.setattr(app, "is_internal_request", lambda request: True)
    body = app.RenderRequest(
        bbox=[-80.0, 10.0, -70.0, 20.0],
        channel="true_color",
        backdrop=True,
    )
    with pytest.raises(app.HTTPException) as caught:
        asyncio.run(app.render(SimpleNamespace(), body))
    assert caught.value.status_code == 422
    assert "single-band Clean-IR" in caught.value.detail


def test_app_rejects_m2_policy_lookalike_before_fetch():
    import app

    lookalike = RenderPolicy(**M2_POLICY.canonical_dict())
    satellite = SimpleNamespace(family="TEST")
    with pytest.raises(RuntimeError, match="unregistered render policy"):
        asyncio.run(
            app._fetch_true_color_for_policy(
                satellite, [-80.0, 10.0, -70.0, 20.0], object(), 1,
                policy=lookalike,
            )
        )


def _physical_cap_grids(shape=(31, 47), *, native_green=False):
    y, x = np.mgrid[: shape[0], : shape[1]]
    checker = ((x + y) % 2).astype(np.float32)
    lat_axis = np.linspace(22.0, 10.0, shape[0], dtype=np.float32)
    lon_axis = np.linspace(-84.0, -72.0, shape[1], dtype=np.float32)
    lons, lats = np.meshgrid(lon_axis, lat_axis)
    return dict(
        red=(np.float32(0.20) + np.float32(0.10) * checker).astype(np.float32),
        green=(
            (np.float32(0.25) + np.float32(0.10) * checker).astype(np.float32)
            if native_green
            else None
        ),
        blue=(np.float32(0.30) + np.float32(0.10) * checker).astype(np.float32),
        veggie=(np.float32(0.40) + np.float32(0.10) * checker).astype(np.float32),
        ir_bt=(np.float32(245.0) + np.float32(20.0) * checker).astype(np.float32),
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
        geom_valid=np.ones(shape, dtype=bool),
    )


def _physical_cap(grids, *, max_px, sensor="abi", bbox=None):
    return satellites._bandlimit_truecolor_physical_cap(
        **grids,
        bbox=bbox or [-84.0, 10.0, -72.0, 22.0],
        sensor=sensor,
        max_px=max_px,
    )


def test_m2_non_crossing_positive_180_endpoint_stays_monotonic_for_abi_and_ahi():
    physical = satellites._truecolor_target_longitudes(
        170.0, 180.0, 7, physical=True
    )
    legacy = satellites._truecolor_target_longitudes(
        170.0, 180.0, 7, physical=False
    )

    assert physical[0] == 170.0
    assert physical[-1] == 180.0
    assert np.all(np.diff(physical) > 0.0)
    assert legacy[-1] == -180.0  # historical [-180, 180) representation

    for producer in (
        satellites.GOESBaseSatellite._compose_true_color_sync,
        satellites.HimawariPacificSatellite._compose_true_color_sync,
    ):
        source = inspect.getsource(producer)
        assert source.count("_truecolor_target_longitudes(") == 1
        assert "physical=physical" in source

    capped = _physical_cap(
        _physical_cap_grids(),
        max_px=11,
        bbox=[170.0, 0.0, 180.0, 10.0],
    )
    axis = capped.lons[0]
    assert axis[-1] == np.float32(180.0)
    assert np.all(np.diff(axis) > 0.0)


def test_m2_physical_source_cap_is_exact_identity_below_limit(monkeypatch):
    grids = _physical_cap_grids(shape=(17, 23))
    with monkeypatch.context() as patch:
        # Identity must not invoke even a nominal Hanning backend operation.
        patch.setattr(
            "render_resample.resample_fields_shared_coverage",
            mock.Mock(side_effect=AssertionError("identity resampled")),
        )
        result = _physical_cap(grids, max_px=64)

    assert result.red is grids["red"]
    assert result.blue is grids["blue"]
    assert result.veggie is grids["veggie"]
    assert result.ir_bt is grids["ir_bt"]
    assert result.lats is grids["lats"]
    assert result.lons is grids["lons"]
    assert result.geom_valid is grids["geom_valid"]
    assert np.array_equal(result.day_coverage, np.ones((17, 23), np.float32))
    assert np.array_equal(result.ir_coverage, np.ones((17, 23), np.float32))
    assert result.metadata["applied"] is False
    assert result.metadata["completed_source_boundary_stage_count"] == 1
    assert result.metadata["completed_source_cap_hanning_operation_count"] == 0
    assert result.metadata["expected_hanning_count_if_delivery_completes"] == 1
    assert result.metadata["kernel"] == "identity"
    assert result.metadata["native_shape"] == [17, 23]
    assert result.metadata["capped_shape"] == [17, 23]
    assert result.metadata["source_to_cap_pixel_center_affine"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_m2_source_cap_is_independent_of_mutable_legacy_s2_limit(monkeypatch):
    grids = _physical_cap_grids(shape=(17, 23))
    monkeypatch.setattr(satellites, "TRUECOLOR_MAX_PX", 4)
    result = satellites._bandlimit_truecolor_physical_cap(
        **grids,
        bbox=[-84.0, 10.0, -72.0, 22.0],
        sensor="abi",
    )
    assert result.red.shape == (17, 23)
    assert result.metadata["max_long_axis_px"] == 1600
    assert result.metadata["applied"] is False


@pytest.mark.parametrize("native_green,sensor", [(False, "abi"), (True, "ahi")])
def test_m2_physical_source_cap_preserves_dc_and_kelvin(native_green, sensor):
    grids = _physical_cap_grids(native_green=native_green)
    grids["red"].fill(np.float32(0.25))
    grids["blue"].fill(np.float32(0.50))
    grids["veggie"].fill(np.float32(0.75))
    if grids["green"] is not None:
        grids["green"].fill(np.float32(0.625))
    grids["ir_bt"].fill(np.float32(257.25))

    result = _physical_cap(grids, max_px=11, sensor=sensor)
    assert result.red.shape == (7, 11)
    assert np.allclose(result.red, 0.25, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.blue, 0.50, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.veggie, 0.75, rtol=0.0, atol=2.0e-6)
    if native_green:
        assert np.allclose(result.green, 0.625, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.ir_bt, 257.25, rtol=0.0, atol=2.0e-5)
    assert float(np.nanmin(result.ir_bt)) > 200.0  # kelvin, never gray/display
    assert np.allclose(result.day_coverage, 1.0, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.ir_coverage, 1.0, rtol=0.0, atol=2.0e-6)
    assert result.metadata["domain"] == "raw_calibrated_reflectance_and_kelvin"
    assert result.metadata["completed_source_boundary_stage_count"] == 2
    assert result.metadata["completed_source_cap_hanning_operation_count"] == 1
    assert result.metadata["expected_hanning_count_if_delivery_completes"] == 2


def test_m2_hanning_source_cap_kills_point_bilinear_checkerboard_mutant():
    grids = _physical_cap_grids()
    result = _physical_cap(grids, max_px=11)

    source = grids["red"]
    row_indices = np.rint(np.linspace(0, source.shape[0] - 1, 7)).astype(int)
    column_indices = np.rint(np.linspace(0, source.shape[1] - 1, 11)).astype(int)
    point_cap_mutant = source[np.ix_(row_indices, column_indices)]

    # A point target cap aliases this Nyquist fixture back to its two source
    # levels. Literal Hanning integrates it near its 0.25 DC level.
    assert set(np.unique(point_cap_mutant)).issubset({np.float32(0.2), np.float32(0.3)})
    assert np.unique(result.red).size > 2
    assert np.all(result.red > 0.23)
    assert np.all(result.red < 0.27)
    assert not np.allclose(result.red, point_cap_mutant, rtol=0.0, atol=1.0e-4)

    from matplotlib import _image
    from matplotlib.transforms import Affine2D

    bilinear_cap_mutant = np.zeros(result.red.shape, dtype=np.float32)
    _image.resample(
        np.ascontiguousarray(source),
        bilinear_cap_mutant,
        Affine2D(
            np.asarray(
                result.metadata["source_to_cap_pixel_center_affine"],
                dtype=np.float64,
            )
        ),
        _image.BILINEAR,
        True,
        1.0,
        True,
        4.0,
    )
    # Bilinear also preserves dimensions, phase and approximate DC. This
    # load-bearing discriminator proves the configured kernel is Hanning.
    assert float(np.max(np.abs(result.red - bilinear_cap_mutant))) > 2.0e-4

    impulse_grids = _physical_cap_grids()
    for name in ("red", "blue", "veggie"):
        impulse_grids[name].fill(np.float32(0.0))
        impulse_grids[name][15, 23] = np.float32(1.0)
    impulse = _physical_cap(impulse_grids, max_px=11)
    point_impulse = impulse_grids["red"][np.ix_(row_indices, column_indices)]
    assert not np.array_equal(impulse.red, point_impulse)
    assert np.count_nonzero(impulse.red > 0.0) >= 1


def test_m2_cap_affine_phase_extents_and_reversed_latitude_are_exact():
    grids = _physical_cap_grids()
    result = _physical_cap(grids, max_px=11)
    metadata = result.metadata
    source_height, source_width = grids["red"].shape
    destination_height, destination_width = result.red.shape
    expected_scale_x = (destination_width - 1) / (source_width - 1)
    expected_scale_y = (destination_height - 1) / (source_height - 1)
    expected = np.array(
        [
            [expected_scale_x, 0.0, 0.5 * (1.0 - expected_scale_x)],
            [0.0, expected_scale_y, 0.5 * (1.0 - expected_scale_y)],
            [0.0, 0.0, 1.0],
        ]
    )
    affine = np.asarray(metadata["source_to_cap_pixel_center_affine"])
    assert np.array_equal(affine, expected)
    assert affine[0, 0] * 0.5 + affine[0, 2] == pytest.approx(0.5)
    assert affine[0, 0] * (source_width - 0.5) + affine[0, 2] == pytest.approx(
        destination_width - 0.5
    )
    assert affine[1, 1] * 0.5 + affine[1, 2] == pytest.approx(0.5)
    assert affine[1, 1] * (source_height - 0.5) + affine[1, 2] == pytest.approx(
        destination_height - 0.5
    )
    assert metadata["sampling_phase_yx"] == pytest.approx(
        [0.5 * expected_scale_y, 0.5 * expected_scale_x]
    )
    assert metadata["source_pixel_center_extent_unwrapped"] == [
        -84.0,
        -72.0,
        10.0,
        22.0,
    ]
    assert metadata["capped_pixel_center_extent_unwrapped"] == [
        -84.0,
        -72.0,
        10.0,
        22.0,
    ]
    assert result.lats[0, 0] == np.float32(22.0)
    assert result.lats[-1, 0] == np.float32(10.0)
    assert np.all(np.diff(result.lats[:, 0]) < 0.0)
    assert result.lons[0, 0] == np.float32(-84.0)
    assert result.lons[0, -1] == np.float32(-72.0)
    assert metadata["latitude_axis_reversed"] is True


def test_m2_cap_joint_day_coverage_and_independent_ir_coverage():
    grids = _physical_cap_grids(native_green=True)
    grids["red"].fill(np.float32(0.4))
    grids["green"].fill(np.float32(0.5))
    grids["blue"].fill(np.float32(0.6))
    grids["veggie"].fill(np.float32(0.7))
    grids["ir_bt"].fill(np.float32(251.0))

    # A hole in any required day band removes support for every day band, but
    # clean IR remains independently supported and in calibrated kelvin.
    grids["blue"][10:21, 17:31] = np.nan
    result = _physical_cap(grids, max_px=17, sensor="ahi")
    fractional = (result.day_coverage > 0.0) & (result.day_coverage < 1.0)
    assert np.any(fractional)
    assert np.any(result.day_coverage == 0.0)
    supported = result.day_coverage > 0.0
    assert np.allclose(result.red[supported], 0.4, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.green[supported], 0.5, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.blue[supported], 0.6, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.veggie[supported], 0.7, rtol=0.0, atol=2.0e-6)
    assert np.all(np.isnan(result.red[~supported]))
    assert np.all(np.isnan(result.green[~supported]))
    assert np.all(np.isnan(result.blue[~supported]))
    assert np.all(np.isnan(result.veggie[~supported]))
    assert np.allclose(result.ir_coverage, 1.0, rtol=0.0, atol=2.0e-6)
    assert np.allclose(result.ir_bt, 251.0, rtol=0.0, atol=2.0e-5)


def test_m2_fractional_cap_coverages_reach_physical_assembly_unchanged():
    grids = _physical_cap_grids(native_green=True)
    grids["blue"][10:21, 17:31] = np.nan
    grids["ir_bt"][2:9, 3:15] = np.nan
    cap = _physical_cap(grids, max_px=17, sensor="ahi")
    observed = {}

    def physical(red, green, blue, veggie, lats, lons, **kwargs):
        observed.update(kwargs)
        shape = red.shape
        return SimpleNamespace(
            day_rgb_reflectance=np.dstack((red, green, blue)),
            day_coverage=kwargs["day_coverage"],
            ir_bt_kelvin=kwargs["ir_bt"],
            ir_coverage=kwargs["ir_coverage"],
            cos_sza=np.ones(shape, np.float32),
            geom_valid=kwargs["geom_valid"],
            red_corrected=red,
            veggie_corrected=veggie,
            glint_deg=np.zeros(shape, np.float32),
        )

    metadata = {"grid_metadata": {"physical_source_cap": cap.metadata}}
    with mock.patch.object(
        truecolor, "assemble_truecolor_physical", side_effect=physical
    ):
        satellites._finish_truecolor_grid(
            cap.red,
            cap.green,
            cap.blue,
            cap.veggie,
            cap.ir_bt,
            cap.lats,
            cap.lons,
            cap.geom_valid,
            downsample=7,
            physical=True,
            assembly_kwargs={},
            physical_metadata=metadata,
            day_coverage=cap.day_coverage,
            ir_coverage=cap.ir_coverage,
        )

    assert observed["day_coverage"] is cap.day_coverage
    assert observed["ir_coverage"] is cap.ir_coverage
    assert observed["grid_metadata"] is metadata["grid_metadata"]
    assert np.any(
        (observed["day_coverage"] > 0.0)
        & (observed["day_coverage"] < 1.0)
    )
    assert np.any(
        (observed["ir_coverage"] > 0.0)
        & (observed["ir_coverage"] < 1.0)
    )


def test_m2_native_guard_fails_before_ahi_load_or_target_allocation(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("guard ran after expensive work")

    monkeypatch.setattr(satellites, "_get_fs", forbidden)
    monkeypatch.setattr("vendor.ahi_loader.load_band_sync", forbidden)
    monkeypatch.setattr(satellites.np, "linspace", forbidden)
    resolved = SimpleNamespace(
        bucket="test",
        scan_start=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        sat_name="Himawari-9",
        product="FLDK",
        sub_sat_lon=140.7,
    )
    with pytest.raises(RuntimeError, match="safety envelope exceeded"):
        satellites.HIMAWARI_PACIFIC._compose_true_color_sync(
            [100.0, -20.0, 140.0, 20.0],
            resolved,
            physical=True,
        )
    assert calls == []


def test_m2_native_guard_fails_before_goes_band_open_or_target_allocation(
    monkeypatch,
):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("guard ran after expensive work")

    monkeypatch.setattr(satellites.GOES_EAST, "open", forbidden)
    monkeypatch.setattr(satellites.np, "linspace", forbidden)
    with pytest.raises(RuntimeError, match="safety envelope exceeded"):
        satellites.GOES_EAST._compose_true_color_sync(
            {"red": object()},
            [-100.0, -20.0, -60.0, 20.0],
            SimpleNamespace(),
            physical=True,
        )
    assert calls == []


def test_m2_guard_records_exact_pixel_and_byte_envelope():
    accepted = satellites._guard_m2_truecolor_native_grid(
        (2_500, 3_000), sensor="abi", basis="test"
    )
    assert accepted == {
        "basis": "test",
        "native_pixels": 7_500_000,
        "estimated_target_workspace_bytes": 960_000_000,
        "bytes_per_pixel_estimate": 128,
        "max_native_pixels": 7_500_000,
        "max_estimated_target_workspace_bytes": 960_000_000,
        "target_workspace_scope": "regular_grid_fetch_and_source_cap",
        "excluded_memory_scopes": [
            "sensor_loader_native_buffers",
            "downstream_high_dpi_delivery_workspace",
        ],
        "excluded_memory_scopes_may_overlap": True,
    }
    with pytest.raises(RuntimeError, match="refusing silent point-cap fallback"):
        satellites._guard_m2_truecolor_native_grid(
            (2_501, 3_000), sensor="abi", basis="test"
        )


def test_m2_bbox_preflight_is_allocation_free_and_keeps_canonical_scene(monkeypatch):
    monkeypatch.setattr(
        satellites.np,
        "linspace",
        mock.Mock(side_effect=AssertionError("bbox preflight allocated a grid")),
    )
    canonical = satellites.m2_truecolor_bbox_preflight(
        [-84.1, 10.3, -72.1, 22.3], sensor="GOES-East"
    )
    assert canonical["supported"] is True
    assert canonical["sensor"] == "abi"
    assert canonical["estimated_native_shape"] == [2664, 2664]
    assert canonical["native_pixels"] == 7_096_896
    assert canonical["estimated_target_workspace_bytes"] == 908_402_688
    assert canonical["fallback_required"] is None
    assert canonical["excluded_memory_scopes"] == [
        "sensor_loader_native_buffers",
        "downstream_high_dpi_delivery_workspace",
    ]
    assert canonical["excluded_memory_scopes_may_overlap"] is True

    oversized = satellites.m2_truecolor_bbox_preflight(
        [100.0, -6.5, 113.0, 6.5], sensor="ahi"
    )
    assert oversized["supported"] is False
    assert oversized["fallback_required"] == "registered_m1_policy"
    assert oversized["estimated_target_workspace_bytes"] > 960_000_000


def test_m2_cap_provenance_binds_residual_and_processing_order():
    result = _physical_cap(_physical_cap_grids(), max_px=11)
    metadata = result.metadata
    assert metadata["pre_cap_coregistration"] == {
        "method": "bilinear",
        "status": "retained_s03_m4_residual",
        "target": "uncapped_regular_red_resolution_grid",
    }
    assert metadata["declared_ordering_residual"] == (
        "source_cap_precedes_rayleigh_until_m3"
    )
    assert metadata["completed_operations_at_source_boundary"] == [
        "retained_bilinear_band_coregistration",
        "upstream_physical_hanning_cap",
    ]
    assert metadata["declared_downstream_contract"] == [
        "rayleigh_after_source_cap",
        "current_ratio_sharpen_after_rayleigh",
        "delivery_hanning_after_physical_assembly",
        "cira_once_after_delivery_resample",
    ]
    assert metadata["completed_source_boundary_stage_count"] == 2
    assert metadata["completed_source_cap_hanning_operation_count"] == 1
    assert metadata["expected_hanning_count_if_delivery_completes"] == 2
    assert metadata["coverage"]["mode"] == "premultiplied_fractional"
