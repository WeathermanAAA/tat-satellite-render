from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from validation.truecolor import m2_sequence


def _nominal_plan(**overrides):
    values = {
        "delivered_shape": (1188, 1188),
        "view_extent": (-91.0, -87.0, 31.2, 35.2),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_predeclared_world_patch_maps_to_exact_live_plan_pixels():
    assert m2_sequence._patch_bounds_from_plan(_nominal_plan()) == (
        1054,
        1150,
        250,
        346,
    )


@pytest.mark.parametrize(
    "plan",
    [
        _nominal_plan(delivered_shape=(1187, 1188)),
        _nominal_plan(view_extent=(-91.01, -87.0, 31.2, 35.2)),
        _nominal_plan(view_extent=(-89.0, -88.0, 32.0, 33.0)),
    ],
)
def test_patch_binding_rejects_shape_extent_and_post_metric_proxy_mutants(plan):
    with pytest.raises(RuntimeError):
        m2_sequence._patch_bounds_from_plan(plan)


def test_nearest_axis_binding_is_orientation_independent_and_fails_outside():
    target = np.asarray([-1.7, -0.2, 1.6], dtype=np.float64)
    ascending = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.array_equal(
        m2_sequence._nearest_axis_indices(ascending, target),
        np.asarray([0, 2, 4]),
    )
    descending = ascending[::-1]
    assert np.array_equal(
        m2_sequence._nearest_axis_indices(descending, target),
        np.asarray([4, 2, 0]),
    )
    with pytest.raises(RuntimeError, match="outside"):
        m2_sequence._nearest_axis_indices(ascending, np.asarray([2.6]))


def test_request_identity_is_static_and_registered_policy_bound(monkeypatch):
    first = m2_sequence._request_identity((1188, 1188))
    second = m2_sequence._request_identity((1188, 1188))
    assert first == second and len(first) == 64
    assert first != m2_sequence._request_identity((1187, 1188))

    import render_policy

    mutant = SimpleNamespace(algo_version="unregistered-lookalike")
    monkeypatch.setattr(render_policy, "M2_POLICY", mutant)
    assert first != m2_sequence._request_identity((1188, 1188))


def test_checksum_parser_rejects_traversal_duplicate_and_malformed_rows(tmp_path):
    path = tmp_path / "SHA256SUMS"
    path.write_text("0" * 64 + "  ../escape\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="unsafe"):
        m2_sequence._parse_checksum_index(path)
    path.write_text(
        "0" * 64 + "  same\n" + "1" * 64 + "  same\n",
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="unsafe"):
        m2_sequence._parse_checksum_index(path)
    path.write_text("not-a-checksum\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="invalid"):
        m2_sequence._parse_checksum_index(path)


def test_v30_producer_has_no_network_or_nearest_scan_resolution_path():
    source = Path(m2_sequence.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    capture = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "capture_v30_sequence"
    )
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(capture)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "fetch_true_color_physical" not in calls
    assert "_find_band_at" not in calls
    assert "find_file" not in calls
    assert "_compose_true_color_sync" in calls
    assert "capture_production_m2_render" in calls


def test_canonical_sequence_constants_are_exact_and_noncrossing():
    assert m2_sequence.V30_ROLE_BY_CHANNEL == {
        "C01": "blue",
        "C02": "red",
        "C03": "veggie",
        "C13": "ir",
    }
    west, south, east, north = m2_sequence.V30_BBOX
    assert west < east and south < north
    patch_west, patch_south, patch_east, patch_north = (
        m2_sequence.V30_PATCH_WORLD_BOUNDS
    )
    assert west < patch_west < patch_east < east
    assert south < patch_south < patch_north < north
    assert m2_sequence.V30_DPI == 110


def test_manifest_key_is_normalized_once_for_local_and_resolved_lookup():
    key = (
        "ABI-L2-CMIPM/2025/301/14/"
        "OR_ABI-L2-CMIPM2-M6C02_G19_"
        "s20253011400549_e20253011401007_c20253011401058.nc"
    )
    assert m2_sequence._v30_logical_s3_key(key) == f"noaa-goes19/{key}"
    for lookalike in (
        f"noaa-goes19/{key}",
        "ABI-L2-CMIPM/../escape.nc",
        "ABI-L2-CMIPF/2025/not-meso.nc",
    ):
        with pytest.raises(RuntimeError, match="bound NOAA namespace"):
            m2_sequence._v30_logical_s3_key(lookalike)

    source = Path(m2_sequence.__file__).read_text(encoding="utf-8")
    capture = ast.parse(source)
    function = next(
        node
        for node in capture.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "capture_v30_sequence"
    )
    helper_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_v30_logical_s3_key"
    ]
    assert len(helper_calls) == 1
