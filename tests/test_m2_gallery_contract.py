"""Independent contract checks for the in-memory M2 review gallery."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from validation.truecolor import m1, m2, m2_gallery, m2_production


def _png(array: np.ndarray) -> bytes:
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _decode(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"))


def _display_float(array: np.ndarray) -> np.ndarray:
    # Stay strictly inside each floor-DN bin, including an exact upper endpoint.
    result = (array.astype(np.float32) + np.float32(0.25)) / np.float32(255.0)
    return np.clip(result, np.float32(0.0), np.float32(1.0)).astype(np.float32)


def _baseline() -> m2_production.M1ArtifactBaseline:
    return m2_production.M1ArtifactBaseline(
        directory="/durable/read-only/m1",
        report_sha256="a" * 64,
        checksum_index_sha256="b" * 64,
        gallery_manifest_sha256="c" * 64,
        git_head="d" * 40,
        scene_id=m2_production.M2_SCENE_ID,
        source_shape=(1600, 1600, 3),
        agg_internal_shape=(129, 128, 4),
        agg_affine=((0.8, 0.0, 0.25), (0.0, 0.8, 0.5), (0.0, 0.0, 1.0)),
        delivered_internal_bounds=(1, 129, 0, 128),
        v15_out_of_range_fraction=0.0,
        v16_nearest_ringing_dn=3.0,
        v16_hanning_ringing_dn=0.2,
        v16_measurements={"fixture": "bound"},
    )


def _evidence(height: int = 128, width: int = 136) -> m2_production.CanonicalM2Evidence:
    y, x = np.mgrid[:height, :width]
    reference = np.stack(
        (
            (31 + 3 * x + 5 * y + 41 * ((x + y) % 2)) % 256,
            (17 + 7 * x + 2 * y + 29 * ((x // 3) % 2)) % 256,
            (9 + 2 * x + 9 * y + 37 * ((y // 2) % 2)) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    candidate = reference.astype(np.int16)
    candidate[::7, ::5, 0] += 2
    candidate[3::11, 2::9, 2] -= 1
    candidate = np.clip(candidate, 0, 255).astype(np.uint8)
    nearest = np.roll(reference, 1, axis=1)
    nearest[::2, ::2] = np.uint8(255) - nearest[::2, ::2]
    display_first = reference.astype(np.int16)
    display_first[..., 0] += 8
    display_first[..., 1] -= 5
    display_first[..., 2] += ((x + y) % 5).astype(np.int16)
    display_first = np.clip(display_first, 0, 255).astype(np.uint8)

    reference_float = np.ascontiguousarray(_display_float(reference))
    candidate_float = np.ascontiguousarray(_display_float(candidate))
    display_float = np.ascontiguousarray(_display_float(display_first))
    reference = np.ascontiguousarray(reference)
    candidate = np.ascontiguousarray(candidate)
    nearest = np.ascontiguousarray(nearest)
    display_first = np.ascontiguousarray(display_first)
    primary_png = _png(candidate)

    rec601 = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
    signed = float(
        np.mean((candidate_float.astype(np.float64) - reference_float.astype(np.float64)) * 255.0)
    )
    signed_luma = float(
        np.mean(
            np.tensordot(
                candidate_float.astype(np.float64) - reference_float.astype(np.float64),
                rec601,
                axes=([-1], [0]),
            )
            * 255.0
        )
    )
    display_signed = float(
        np.mean((display_float.astype(np.float64) - reference_float.astype(np.float64)) * 255.0)
    )
    metrics = {
        "V01": {
            "candidate_rms_dn": m2.rms_dn(candidate, reference),
            "old_nearest_rms_dn": m2.rms_dn(nearest, reference),
            "reference_quantizer": "production_floor_u8_no_dither",
        },
        "V02": {"candidate": {"discarded_rows": 0, "discarded_columns": 0}},
        "V03": {
            "candidate_ratio": m2.radial_high_frequency_power_ratio(
                candidate, reference, cutoff=0.3, bins=64
            ),
            "old_nearest_ratio": m2.radial_high_frequency_power_ratio(
                nearest, reference, cutoff=0.3, bins=64
            ),
            "luma": "Rec.601",
            "spatial_window": "separable_2D_Hann",
            "radial_bins": 64,
            "cutoff_cycles_per_pixel": 0.3,
        },
        "V07": {},
        "V08": {"passed": True},
        "V13": {
            "mean_signed_all_channel_dn": signed,
            "mean_signed_rec601_luma_dn": signed_luma,
            "display_first_mutant_mean_signed_dn": display_signed,
            "trace_issues": [],
        },
        "V14": {},
        "V15": {},
        "V16": {},
        "V41": {},
        "mutation_discriminators": {},
        "limitations": [],
    }
    capture = {
        "schema": m2_production.M2_PRODUCTION_SCHEMA,
        "bbox": list(m2_production.M2_CANONICAL_BBOX),
        "dpi": 110,
        "policy_version": m2_production.M2_REQUIRED_POLICY_VERSION,
        "png_bytes": len(primary_png),
        "png_sha256": hashlib.sha256(primary_png).hexdigest(),
        "delivered_float_rgb": m2_production._array_record(  # noqa: SLF001
            "delivered_float_rgb", candidate_float
        ),
        "delivered_floor_rgb": m2_production._array_record(  # noqa: SLF001
            "delivered_floor_rgb", candidate
        ),
        "figure_image_rgb": m2_production._array_record(  # noqa: SLF001
            "figure_image_rgb", candidate
        ),
        "trace_issues": [],
    }
    objects = tuple(
        {"role": role, "sha256": f"{index + 1:x}" * 64}
        for index, role in enumerate(m1.M1_REQUIRED_ROLES)
    )
    return m2_production.CanonicalM2Evidence(
        scene_id=m2_production.M2_SCENE_ID,
        bbox=m2_production.M2_CANONICAL_BBOX,
        object_identities=objects,
        primary_capture=capture,
        limb_capture=None,
        primary_png=primary_png,
        candidate_float=candidate_float,
        candidate_floor=candidate,
        exact_area_reference_float=reference_float,
        exact_area_reference_floor=reference,
        old_nearest_mutant_floor=nearest,
        display_first_mutant_float=display_float,
        display_first_mutant_floor=display_first,
        metrics=metrics,
        m1_baseline=_baseline(),
    )


@pytest.fixture(scope="module")
def evidence() -> m2_production.CanonicalM2Evidence:
    return _evidence()


@pytest.fixture(scope="module")
def bundle(evidence) -> m2_gallery.M2GalleryBundle:
    return m2_gallery.build_m2_gallery(
        evidence,
        repeatability_summary={
            "passed": True,
            "hashes": ["e" * 64, "e" * 64, "e" * 64],
            "output_bytes": [100, 100, 100],
            "manifests": [],
        },
        sequence_summary={"outcome": "UNMEASURED", "issues": ["not supplied"]},
    )


def test_bundle_is_deterministic_immutable_safe_and_manifest_ready(evidence, bundle) -> None:
    second = m2_gallery.build_m2_gallery(
        evidence,
        repeatability_summary={
            "passed": True,
            "hashes": ["e" * 64, "e" * 64, "e" * 64],
            "output_bytes": [100, 100, 100],
            "manifests": [],
        },
        sequence_summary={"outcome": "UNMEASURED", "issues": ["not supplied"]},
    )
    assert dict(bundle.payloads) == dict(second.payloads)
    assert bundle.manifest_dict() == second.manifest_dict()
    with pytest.raises(TypeError):
        bundle.payloads["new"] = b"no"  # type: ignore[index]
    with pytest.raises(TypeError):
        bundle.artifacts["README.md"]["sha256"] = "0" * 64  # type: ignore[index]

    manifest = bundle.manifest_dict()
    assert manifest["schema"] == m2_gallery.GALLERY_SCHEMA
    assert manifest["andrew_review"]["verdict"] == "PENDING"
    assert manifest["claim_boundary"]["stage_advancement_claimed_by_gallery"] is False
    limitations = " ".join(manifest["claim_boundary"]["limitations"])
    for required in (
        "primary imagery-layer",
        ">1600 source cap",
        "before Rayleigh",
        "M3 ordering residual",
        "ratio self-sharpen",
        "S03",
        "M4",
        "M5",
        "V44",
        "V45",
        "V46",
    ):
        assert required in limitations
    for filename, payload in bundle.payloads.items():
        assert Path(filename).name == filename
        assert "/" not in filename and "\\" not in filename
        record = manifest["artifacts"][filename]
        immutable_record = dict(bundle.artifacts[filename])
        immutable_record["source_bindings"] = list(
            immutable_record["source_bindings"]
        )
        immutable_record["source_binding_sha256"] = dict(
            immutable_record["source_binding_sha256"]
        )
        assert record == immutable_record
        for binding, digest in record["source_binding_sha256"].items():
            if binding in manifest["source_evidence"]["array_records"]:
                assert digest == manifest["source_evidence"]["array_records"][binding]["sha256"]
            elif binding == "primary_png":
                assert digest == manifest["source_evidence"]["primary_png"]["sha256"]
            else:
                assert binding == "evidence_summary"
                assert digest == manifest["source_evidence"]["summary_sha256"]
        assert record["bytes"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert b"Andrew verdict: PENDING" in bundle.payloads["README.md"]
    assert b"Andrew verdict: PENDING" in bundle.payloads["REVIEW_INSTRUCTIONS.md"]


def test_native_pngs_and_100_percent_sheet_panels_are_pixel_identical(evidence, bundle) -> None:
    expected = {
        "01_m2_candidate_native.png": evidence.candidate_floor,
        "02_exact_area_reference_native.png": evidence.exact_area_reference_floor,
        "03_old_nearest_mutant_native.png": evidence.old_nearest_mutant_floor,
        "04_display_first_mutant_native.png": evidence.display_first_mutant_floor,
    }
    for name, array in expected.items():
        np.testing.assert_array_equal(_decode(bundle.payloads[name]), array)
    assert bundle.payloads["14_production_renderer.png"] == evidence.primary_png

    manifest = bundle.manifest_dict()
    sheet = _decode(bundle.payloads["08_native_comparison_sheet_100pct.png"])
    for panel, array in zip(
        manifest["native_comparison_sheet"]["panel_bounds"],
        (
            evidence.candidate_floor,
            evidence.exact_area_reference_floor,
            evidence.old_nearest_mutant_floor,
            evidence.display_first_mutant_floor,
        ),
        strict=True,
    ):
        left, top, right, bottom = panel["bounds_xyxy"]
        np.testing.assert_array_equal(sheet[top:bottom, left:right], array)


def test_error_npy_maps_and_error_sheet_recompute_from_bound_arrays(evidence, bundle) -> None:
    manifest = bundle.manifest_dict()
    raw = np.load(
        io.BytesIO(bundle.payloads["10_signed_rec601_error_values_float32.npy"]),
        allow_pickle=False,
    )
    rec601 = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
    expected = []
    for candidate in (
        evidence.candidate_floor,
        evidence.old_nearest_mutant_floor,
        evidence.display_first_mutant_floor,
    ):
        expected.append(
            np.tensordot(
                candidate.astype(np.float64)
                - evidence.exact_area_reference_floor.astype(np.float64),
                rec601,
                axes=([-1], [0]),
            ).astype(np.float32)
        )
    np.testing.assert_array_equal(raw, np.stack(expected))
    canonical_scale = max(
        1.0, *(float(np.percentile(np.abs(value), 99.9)) for value in expected)
    )
    assert manifest["signed_error"]["shared_symmetric_limit_dn"] == pytest.approx(
        canonical_scale, abs=1e-12
    )

    def encode(error: np.ndarray) -> np.ndarray:
        normalized = np.clip(error.astype(np.float64) / canonical_scale, -1.0, 1.0)
        magnitude = np.abs(normalized)
        neutral = 245.0 * (1.0 - magnitude)
        rgb = np.stack(
            (
                neutral + np.clip(normalized, 0.0, 1.0) * 255.0,
                neutral,
                neutral + np.clip(-normalized, 0.0, 1.0) * 255.0,
            ),
            axis=-1,
        )
        return np.clip(np.floor(rgb + 0.5), 0.0, 255.0).astype(np.uint8)

    maps = [encode(value) for value in expected]
    for index, (name, encoded) in enumerate(
        zip(m2_gallery.ERROR_ORDER, maps, strict=True), start=5
    ):
        np.testing.assert_array_equal(
            _decode(bundle.payloads[f"{index:02d}_{name}_signed_rec601_error_native.png"]),
            encoded,
        )
    sheet = _decode(bundle.payloads["09_signed_error_sheet_shared_scale_100pct.png"])
    for panel, encoded in zip(
        manifest["signed_error"]["comparison_sheet"]["panel_bounds"],
        maps,
        strict=True,
    ):
        left, top, right, bottom = panel["bounds_xyxy"]
        np.testing.assert_array_equal(sheet[top:bottom, left:right], encoded)


def test_spectral_csv_is_the_canonical_rec601_hann_radial_procedure(evidence, bundle) -> None:
    rows = list(
        csv.DictReader(
            io.StringIO(bundle.payloads["12_canonical_spectral_comparison.csv"].decode("ascii"))
        )
    )
    assert len(rows) == 64
    rasters = {
        "candidate_luma_power": evidence.candidate_floor,
        "exact_area_reference_luma_power": evidence.exact_area_reference_floor,
        "old_nearest_mutant_luma_power": evidence.old_nearest_mutant_floor,
        "display_first_mutant_luma_power": evidence.display_first_mutant_floor,
    }
    canonical_frequency = None
    for column, raster in rasters.items():
        frequency, power = m2.canonical_radial_power_profile(raster, bins=64)
        if canonical_frequency is None:
            canonical_frequency = frequency
        observed = np.asarray([float(row[column]) for row in rows])
        np.testing.assert_allclose(observed, power, rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(
        np.asarray([float(row["radial_frequency_cycles_per_delivered_pixel"]) for row in rows]),
        canonical_frequency,
        rtol=0.0,
        atol=0.0,
    )
    spectral_plot = _decode(bundle.payloads["13_canonical_spectral_comparison.png"])
    assert spectral_plot.shape == (720, 1200, 3)


def test_tampered_arrays_capture_metrics_and_proxy_evidence_are_rejected(evidence) -> None:
    bad_floor = evidence.candidate_floor.copy()
    bad_floor[0, 0, 0] ^= np.uint8(1)
    with pytest.raises(RuntimeError, match="candidate floor"):
        m2_gallery.build_m2_gallery(dataclasses.replace(evidence, candidate_floor=bad_floor))

    bad_capture = dict(evidence.primary_capture)
    bad_capture["delivered_floor_rgb"] = {
        **bad_capture["delivered_floor_rgb"],
        "sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="production capture delivered_floor_rgb"):
        m2_gallery.build_m2_gallery(dataclasses.replace(evidence, primary_capture=bad_capture))

    bad_metrics = dict(evidence.metrics)
    bad_metrics["V03"] = {**bad_metrics["V03"], "candidate_ratio": 0.0}
    with pytest.raises(RuntimeError, match="V03 candidate ratio"):
        m2_gallery.build_m2_gallery(dataclasses.replace(evidence, metrics=bad_metrics))

    with pytest.raises(TypeError, match="exact CanonicalM2Evidence"):
        m2_gallery.build_m2_gallery(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="old_nearest_mutant_floor"):
        m2_gallery.build_m2_gallery(
            dataclasses.replace(
                evidence,
                old_nearest_mutant_floor=evidence.old_nearest_mutant_floor.astype(np.float32),
            )
        )


def test_noncanonical_scale_summary_hash_and_unsafe_summary_proxies_are_rejected(evidence) -> None:
    canonical = m2_gallery.build_m2_gallery(evidence).manifest_dict()["signed_error"][
        "shared_symmetric_limit_dn"
    ]
    with pytest.raises(ValueError, match="noncanonical signed-error scale"):
        m2_gallery.build_m2_gallery(
            evidence, expected_shared_error_scale_dn=float(canonical) + 1.0
        )
    with pytest.raises(RuntimeError, match="summary SHA-256"):
        m2_gallery.build_m2_gallery(
            evidence, expected_evidence_summary_sha256="0" * 64
        )

    class DictProxy(dict):
        pass

    with pytest.raises(TypeError, match="exact dict"):
        m2_gallery.build_m2_gallery(
            evidence,
            repeatability_summary=DictProxy(
                passed=True, hashes=["e" * 64] * 3
            ),
        )
    with pytest.raises(ValueError, match="finite canonical JSON"):
        m2_gallery.build_m2_gallery(
            evidence,
            sequence_summary={"outcome": "PASS", "metric": math.nan},
        )
    for unsafe in ("../bad.png", "nested/bad.png", "bad\\name.png", "", "."):
        with pytest.raises(ValueError, match="unsafe"):
            m2_gallery._safe_filename(unsafe)  # noqa: SLF001
