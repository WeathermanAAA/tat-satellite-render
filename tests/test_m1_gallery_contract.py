"""Original-scale and canonical-spectrum contracts for the M1 gallery."""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.tat_truecolor_m1_validate import build_parser
from validation.truecolor.m1 import canonical_radial_power_profile
from validation.truecolor.m1_gallery import (
    GALLERY_SCHEMA,
    _error_comparison_sheet_bytes,
    _full_scale_sheet_bytes,
    _signed_error_map,
    _signed_error_summary_csv_bytes,
    _signed_luma_error,
    _validate_rendered_raster_bindings,
    radial_power_profile,
    write_m1_gallery_manifest,
)


def _sha256_raster(raster: np.ndarray) -> str:
    return hashlib.sha256(raster.tobytes(order="C")).hexdigest()


def _bound_gallery_run() -> SimpleNamespace:
    nearest = np.zeros((1188, 1188, 3), dtype=np.uint8)
    hanning = np.full_like(nearest, 17)
    reference = np.full_like(nearest, 23)
    canonical = SimpleNamespace(
        production_nearest_raster_sha256=_sha256_raster(nearest),
        explicit_hanning_raster_sha256=_sha256_raster(hanning),
        bandlimited_reference_raster_sha256=_sha256_raster(reference),
        scene_id="fixture",
        reference_convention="fixture convention",
        source_shape=(1527, 1600, 3),
        figure_shape=(1320, 1320, 3),
        map_pixel_bounds=(66, 53, 1254, 1241),
        delivered_shape=(1188, 1188, 3),
        sampling_model="captured",
        sampling_phase=(0.5, 0.5),
        agg_internal_shape=(1188, 1188, 4),
        agg_transform_matrix=((1.0, 0.0, 0.0),) * 3,
        agg_make_image_geometry={},
        delivered_internal_pixel_bounds=(0, 1188, 0, 1188),
        internal_mapping_candidate_scores=(((0, 1188, 0, 1188), 0.9983),),
        reference_identity_scored=0.9983,
    )
    rendered = SimpleNamespace(
        production_nearest=nearest,
        explicit_hanning=hanning,
        bandlimited_reference=reference,
    )
    gates = (
        SimpleNamespace(gate_id="V01", outcome="FAIL"),
        SimpleNamespace(gate_id="V02", outcome="PASS"),
    )
    return SimpleNamespace(
        canonical=canonical,
        rendered=rendered,
        gates=gates,
        stage_advancement_passed=True,
        all_thresholds_passed=False,
        threshold_fail_gate_ids=("V01",),
        unmeasured_gate_ids=(),
    )


def _write_manifest_inputs(directory) -> tuple[dict[str, dict[str, object]], str]:
    artifact_payload = b"gallery-raster-payload"
    artifact_name = "01_fixture.bin"
    (directory / artifact_name).write_bytes(artifact_payload)
    artifacts: dict[str, dict[str, object]] = {
        artifact_name: {
            "filename": artifact_name,
            "sha256": hashlib.sha256(artifact_payload).hexdigest(),
            "bytes": len(artifact_payload),
            "media_type": "application/octet-stream",
            "width": None,
            "height": None,
            "description": "fixture",
        }
    }
    report_payload = (
        json.dumps({"gallery_artifacts": artifacts}, sort_keys=True) + "\n"
    ).encode("utf-8")
    (directory / "m1-validation.json").write_bytes(report_payload)
    return artifacts, hashlib.sha256(report_payload).hexdigest()


def test_full_scale_sheet_embeds_each_1188_panel_without_resampling() -> None:
    y, x = np.mgrid[:1188, :1188]
    nearest = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    hanning = np.roll(nearest, 1, axis=1)
    reference = np.roll(nearest, 1, axis=0)

    payload, dimensions = _full_scale_sheet_bytes(nearest, hanning, reference)
    assert dimensions == (3564, 1248)
    with Image.open(io.BytesIO(payload)) as image:
        sheet = np.asarray(image.convert("RGB"))
    assert np.array_equal(sheet[60:1248, 0:1188], nearest)
    assert np.array_equal(sheet[60:1248, 1188:2376], hanning)
    assert np.array_equal(sheet[60:1248, 2376:3564], reference)


def test_gallery_spectrum_is_exactly_the_gate_spectrum() -> None:
    generator = np.random.default_rng(43)
    raster = generator.integers(0, 256, size=(72, 80, 3), dtype=np.uint8)
    gallery_frequency, gallery_power = radial_power_profile(raster)
    gate_frequency, gate_power = canonical_radial_power_profile(raster, bins=64)
    assert np.array_equal(gallery_frequency, gate_frequency)
    np.testing.assert_allclose(gallery_power, gate_power, equal_nan=True)
    assert gallery_frequency[-1] < 0.5
    assert gallery_frequency.shape == (64,)


def test_signed_error_map_uses_rec601_luma_sign() -> None:
    reference = np.full((4, 4, 3), 100, dtype=np.uint8)
    candidate = reference.copy()
    candidate[..., 0] += 10
    luma_error = _signed_luma_error(candidate, reference)
    assert luma_error == pytest.approx(np.full((4, 4), 2.99))
    error = _signed_error_map(luma_error, scale=2.99)
    assert np.all(error[..., 0] > error[..., 2])


def test_error_comparison_uses_one_scale_and_embeds_both_maps_exactly() -> None:
    nearest_values = np.linspace(-12.0, 12.0, 1188 * 1188).reshape(1188, 1188)
    hanning_values = np.linspace(-3.0, 3.0, 1188 * 1188).reshape(1188, 1188)
    shared_scale = max(
        np.percentile(np.abs(nearest_values), 99.9),
        np.percentile(np.abs(hanning_values), 99.9),
    )
    nearest_map = _signed_error_map(nearest_values, scale=shared_scale)
    hanning_map = _signed_error_map(hanning_values, scale=shared_scale)
    payload, dimensions = _error_comparison_sheet_bytes(
        nearest_map,
        hanning_map,
        shared_scale=shared_scale,
    )
    assert dimensions == (2376, 1268)
    with Image.open(io.BytesIO(payload)) as image:
        sheet = np.asarray(image.convert("RGB"))
    assert np.array_equal(sheet[80:1268, 0:1188], nearest_map)
    assert np.array_equal(sheet[80:1268, 1188:2376], hanning_map)

    summary = _signed_error_summary_csv_bytes(
        nearest_values,
        hanning_values,
        shared_scale=shared_scale,
    ).decode("ascii")
    rows = summary.splitlines()
    assert len(rows) == 3
    assert rows[1].split(",")[1] == rows[2].split(",")[1]


def test_gallery_rasters_are_bound_to_canonical_raw_byte_hashes() -> None:
    run = _bound_gallery_run()
    _validate_rendered_raster_bindings(run)

    mutated = run.rendered.explicit_hanning.copy()
    mutated[713, 404, 1] ^= np.uint8(1)
    mutant_run = SimpleNamespace(
        canonical=run.canonical,
        rendered=SimpleNamespace(
            production_nearest=run.rendered.production_nearest,
            explicit_hanning=mutated,
            bandlimited_reference=run.rendered.bandlimited_reference,
        ),
    )
    with pytest.raises(RuntimeError, match="does not match canonical evidence"):
        _validate_rendered_raster_bindings(mutant_run)


def test_gallery_v2_manifest_declares_shared_error_encoding_and_immutable_review(
    tmp_path,
) -> None:
    run = _bound_gallery_run()
    directory = tmp_path / "gallery"
    directory.mkdir()
    artifacts, report_sha256 = _write_manifest_inputs(directory)
    artifact = write_m1_gallery_manifest(
        run,
        directory,
        artifacts,
        report_filename="m1-validation.json",
        report_sha256=report_sha256,
    )
    manifest = json.loads((directory / artifact.filename).read_text())
    assert GALLERY_SCHEMA == "tat_truecolor_m1_gallery/v2"
    assert manifest["schema"] == GALLERY_SCHEMA
    encoding = manifest["error_map_encoding"]
    assert encoding["rec601_coefficients_rgb"] == [0.299, 0.587, 0.114]
    assert encoding["shared_scale_percentile"] == 99.9
    assert encoding["shared_symmetric_limit_dn"] == pytest.approx(23.0)
    assert encoding["comparison_sheet"]["nearest_raster_bounds_xyxy"] == [
        0,
        80,
        1188,
        1268,
    ]
    assert manifest["andrew_review"]["verdict"] == "PENDING"
    assert manifest["andrew_review"]["manifest_is_immutable"] is True
    assert manifest["threshold_fail_gate_ids"] == ["V01"]


def test_manifest_fails_closed_on_missing_or_wrong_report_and_artifact_mutation(
    tmp_path,
) -> None:
    run = _bound_gallery_run()
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="report is absent"):
        write_m1_gallery_manifest(
            run,
            missing,
            {"x.bin": {"filename": "x.bin", "sha256": "0" * 64, "bytes": 1}},
            report_filename="m1-validation.json",
            report_sha256="0" * 64,
        )

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    artifacts, report_sha256 = _write_manifest_inputs(wrong)
    with pytest.raises(RuntimeError, match="report SHA-256"):
        write_m1_gallery_manifest(
            run,
            wrong,
            artifacts,
            report_filename="m1-validation.json",
            report_sha256="f" * 64,
        )

    mutated = tmp_path / "mutated"
    mutated.mkdir()
    artifacts, report_sha256 = _write_manifest_inputs(mutated)
    payload = bytearray((mutated / "01_fixture.bin").read_bytes())
    payload[0] ^= 1
    (mutated / "01_fixture.bin").write_bytes(payload)
    with pytest.raises(RuntimeError, match="SHA-256"):
        write_m1_gallery_manifest(
            run,
            mutated,
            artifacts,
            report_filename="m1-validation.json",
            report_sha256=report_sha256,
        )


def test_cli_requires_explicit_m1_enforcement_stage() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--research-root",
                "/research",
                "--corpus-root",
                "/corpus",
                "--artifact-dir",
                "/artifacts",
                "--rayleigh-data-dir",
                "/rayleigh",
                "--pyspectral-config",
                "/config",
            ]
        )
