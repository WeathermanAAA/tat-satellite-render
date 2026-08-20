"""M0 port-fidelity and repeatability checks against the real render path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from validation.truecolor import production_adapter as adapter


def test_adapter_imports_only_the_target_production_checkout():
    modules = adapter.production_modules()
    assert set(modules) == set(adapter.PRODUCTION_MODULE_NAMES)
    for module in modules.values():
        path = Path(module.__file__).resolve()
        assert adapter.REPO_ROOT in path.parents


def test_adapter_observes_both_current_nearest_imshow_sites():
    calls = adapter.imshow_calls()
    assert [(call.branch, call.interpolation) for call in calls] == [
        ("rgb", "nearest"),
        ("scalar", "nearest"),
    ]


@pytest.mark.parametrize(
    ("case_id", "bbox", "figure_shape", "delivered_shape", "boundary"),
    (
        ("bbox_1x1", (4.0, 4.0), (1320, 1320, 3), (1188, 1188, 3), 1),
        ("bbox_2x1", (4.0, 2.0), (660, 1320, 3), (594, 1188, 3), 2),
    ),
)
def test_cartopy_port_fidelity_meets_canonical_geographic_aspect_contract(
    case_id, bbox, figure_shape, delivered_shape, boundary
):
    result = adapter.port_fidelity_probe(case_id)
    assert result.bbox_degrees == bbox
    assert result.geographic_aspect_ratio == pytest.approx(bbox[0] / bbox[1])
    assert result.figure_shape == figure_shape
    assert result.delivered_shape == delivered_shape
    assert result.matplotlib_version
    assert result.sampling_model == adapter.matplotlib_image_sampling_model()
    assert result.boundary_exclusion_px == boundary
    # The unmasked score remains evidence. Only the documented Cartopy clip
    # boundary is excluded from the canonical port-fidelity score.
    assert result.full_pixel_identity < result.scored_pixel_identity
    assert result.scored_pixel_identity >= result.minimum_scored_identity
    if result.exact_identity_required:
        assert result.scored_pixel_identity == 1.0
    assert result.png_mode == "RGBA"
    assert not result.png_icc_profile_present
    assert result.quantization_model == "float_nearest_then_floor_u8"
    assert result.floor_quantization_scored_identity == result.scored_pixel_identity
    assert result.round_half_up_mutant_scored_identity < 0.001


def test_analytic_port_pins_both_primary_matplotlib_bbox_algorithms():
    square_anchor = adapter.analytic_figure_geometry(
        4.0, 4.0, sampling_model="agg_anchor_round_half"
    )
    wide_anchor = adapter.analytic_figure_geometry(
        4.0, 2.0, sampling_model="agg_anchor_round_half"
    )
    square_legacy = adapter.analytic_figure_geometry(
        4.0, 4.0, sampling_model="legacy_fractional_clip_ceil"
    )
    wide_legacy = adapter.analytic_figure_geometry(
        4.0, 2.0, sampling_model="legacy_fractional_clip_ceil"
    )

    for square in (square_anchor, square_legacy):
        assert square.device_origin == (79, 66)
        assert square.delivered_shape == (1188, 1188)
    for wide in (wide_anchor, wide_legacy):
        assert wide.device_origin == (40, 66)
        assert wide.delivered_shape == (594, 1188)

    assert square_anchor.sampling_phase == pytest.approx((0.3, 0.5), abs=1e-12)
    assert wide_anchor.sampling_phase == pytest.approx((0.9, 0.5), abs=1e-12)
    assert square_legacy.sampling_phase == (0.5, 0.5)
    assert wide_legacy.sampling_phase == (0.5, 0.5)


def test_default_port_uses_the_installed_matplotlib_bbox_algorithm():
    square = adapter.analytic_figure_geometry(4.0, 4.0)
    assert square.sampling_model == adapter.matplotlib_image_sampling_model()


def test_port_fidelity_case_registry_is_immutable():
    with pytest.raises(TypeError):
        adapter.PORT_FIDELITY_CASES["bbox_1x1"] = adapter.PORT_FIDELITY_CASES[
            "bbox_1x1"
        ]


def test_same_production_request_is_identical_in_three_processes(tmp_path):
    worker = (
        "from validation.truecolor.production_adapter import "
        "port_probe_png_sha256; print(port_probe_png_sha256('bbox_1x1'))"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    # Give all workers the same writable Matplotlib configuration directory;
    # this changes neither pixels nor metadata and avoids a home-dir fallback.
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    hashes = []
    for _ in range(3):
        completed = subprocess.run(
            [sys.executable, "-c", worker],
            cwd=adapter.REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        hashes.append(completed.stdout.strip().splitlines()[-1])
    assert len(set(hashes)) == 1, json.dumps(hashes)
    assert len(hashes[0]) == 64
