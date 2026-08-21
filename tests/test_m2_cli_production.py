"""Production-closing and immutable-payload contracts for the M2 CLI."""

from __future__ import annotations

import dataclasses
import datetime as dt
import copy
import functools
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import tat_truecolor_m2_validate as cli
from validation.truecolor import m2_production, production_adapter
from validation.truecolor.m2_sequence import (
    V30_CANONICAL_ACQUISITION_INDEX_SHA256,
    V30_CANONICAL_MANIFEST_SHA256,
    V30_CAPTURE_SCHEMA,
    V30_NATURAL_EARTH_ZIP_SHA256,
    V30_PATCH_EDGE_BINDING_TOLERANCE_PX,
    V30Corpus,
    V30Object,
    V30SequenceCapture,
)


@pytest.fixture(autouse=True)
def _reverify_canonical_publication_fixture_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cli.verify_research_root
    fixture_value = {
        "root": "/fixture/research",
        "validation_gates_csv": {
            "path": "/fixture/research/tables/validation_gates.csv",
            "sha256": "a" * 64,
        },
        "reference_pixels_json": {
            "path": "/fixture/research/sources/reference_pixels.json",
            "sha256": "b" * 64,
        },
    }

    def verify(root):
        if str(root) == "/fixture/research":
            return SimpleNamespace(
                as_dict=lambda: json.loads(json.dumps(fixture_value)),
                # Mirror the authoritative research loader's exact V01--V46
                # registry. M2 consumes a strict subset; equating the live
                # registry itself to that subset rejects valid research input.
                gates={
                    gate_id: SimpleNamespace(
                        threshold="fixture threshold",
                        procedure="fixture procedure",
                        source_document="fixture source",
                    )
                    for gate_id in (gate.gate_id for gate in cli.GATES)
                },
            )
        return original(root)

    monkeypatch.setattr(cli, "verify_research_root", verify)


@pytest.fixture
def external_tree():
    base = Path(
        tempfile.mkdtemp(
            prefix=".m2-cli-production-",
            dir=production_adapter.REPO_ROOT.resolve().parent,
        )
    )
    try:
        for name in (
            "research",
            "corpus",
            "m1",
            "rayleigh",
            "config",
            "acquisition",
            "raw",
            "land",
            "artifacts",
        ):
            (base / name).mkdir()
        (base / "config" / "pyspectral.yaml").write_text(
            "rsr_dir: fixture\nrayleigh_dir: fixture\n", encoding="utf-8"
        )
        (base / "land" / "natural-earth.zip").write_bytes(b"fixture zip")
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _production_argv(base: Path, target: Path) -> list[str]:
    return [
        "--enforce-through",
        "M2",
        "--research-root",
        str(base / "research"),
        "--artifact-dir",
        str(target),
        "--canonical-corpus",
        str(base / "corpus"),
        "--verified-m1-artifact",
        str(base / "m1"),
        "--rayleigh-dir",
        str(base / "rayleigh"),
        "--rayleigh-config",
        str(base / "config" / "pyspectral.yaml"),
        "--v30-acquisition",
        str(base / "acquisition"),
        "--v30-raw",
        str(base / "raw"),
        "--v30-land-zip",
        str(base / "land" / "natural-earth.zip"),
        "--runner-image-id",
        cli.M2_RUNNER_IMAGE_ID,
    ]


def _production_namespace(base: Path, target: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_dir=target,
        research_root=base / "research",
        corpus_root=base / "corpus",
        m1_artifact_dir=base / "m1",
        rayleigh_data_dir=base / "rayleigh",
        pyspectral_config=base / "config" / "pyspectral.yaml",
        v30_acquisition_dir=base / "acquisition",
        v30_raw_dir=base / "raw",
        v30_land_zip=base / "land" / "natural-earth.zip",
        runner_image_id=cli.M2_RUNNER_IMAGE_ID,
        contract_only=False,
    )


def test_preflight_requires_complete_disjoint_production_inputs_and_scratch_contract(
    external_tree: Path,
) -> None:
    target = external_tree / "artifacts" / "production"
    complete = _production_namespace(external_tree, target)
    paths = cli.preflight_paths(complete)
    assert paths.production

    for missing in cli.PRODUCTION_ARGUMENTS:
        values = vars(_production_namespace(external_tree, target)).copy()
        values[missing] = None
        with pytest.raises(ValueError, match="missing"):
            cli.preflight_paths(SimpleNamespace(**values))

    for runner_image_id in (None, "sha256:" + "0" * 64):
        values = vars(_production_namespace(external_tree, target)).copy()
        values["runner_image_id"] = runner_image_id
        with pytest.raises(ValueError, match="runner-image-id"):
            cli.preflight_paths(SimpleNamespace(**values))

    aliased = vars(_production_namespace(external_tree, target)).copy()
    aliased["v30_raw_dir"] = aliased["corpus_root"]
    with pytest.raises(cli.UnsafeArtifactPath, match="alias or nest"):
        cli.preflight_paths(SimpleNamespace(**aliased))

    contract = SimpleNamespace(
        artifact_dir=external_tree / "artifacts" / "scratch-contract",
        research_root=external_tree / "research",
        contract_only=True,
        runner_image_id=None,
        **{name: None for name in cli.PRODUCTION_ARGUMENTS},
    )
    assert not cli.preflight_paths(contract).production
    contract.artifact_dir = external_tree / "artifacts" / "not-scratch"
    with pytest.raises(cli.UnsafeArtifactPath, match="scratch"):
        cli.preflight_paths(contract)
    contract.artifact_dir = external_tree / "artifacts" / "scratch-with-product"
    contract.corpus_root = external_tree / "corpus"
    with pytest.raises(ValueError, match="cannot accept"):
        cli.preflight_paths(contract)


def test_production_dependency_contract_binds_repository_exact_pins(
    monkeypatch,
) -> None:
    metadata = cli.collect_m2_dependency_metadata(
        {
            "python": "fixture",
            "distributions": {},
            "direct_urls": {},
            "module_origins": {},
        }
    )
    contract = metadata["production_dependency_contract"]
    assert contract["exact_requirement_pins"]["matplotlib"] == "3.9.3"
    assert contract["exact_requirement_pins"]["numpy"] == "2.1.3"
    assert contract["exact_requirement_pins"]["Pillow"] == "11.0.0"
    assert contract["runtime_declaration"] == "python-3.11.10"
    assert contract["expected_python_series"] == "3.11"
    assert len(contract["requirements_sha256"]) == 64
    assert len(contract["runtime_sha256"]) == 64
    if contract["passed"]:
        cli.require_production_dependency_identity(metadata)
    else:
        with pytest.raises(RuntimeError, match="exact requirements pins"):
            cli.require_production_dependency_identity(metadata)

    mutated = json.loads(json.dumps(metadata))
    mutated["production_dependency_contract"]["passed"] = False
    mutated["production_dependency_contract"]["issues"] = ["injected drift"]
    with pytest.raises(RuntimeError, match="injected drift"):
        cli.require_production_dependency_identity(mutated)

    monkeypatch.setattr(cli.platform, "python_version", lambda: "3.11.16")
    compatible = cli.collect_m2_dependency_metadata()
    compatible_contract = compatible["production_dependency_contract"]
    assert compatible_contract["runtime_series_matches_actual"] is True
    assert compatible_contract["runtime_declaration_matches_actual"] is False
    assert compatible_contract["actual_python"] == "3.11.16"

    monkeypatch.setattr(cli.platform, "python_version", lambda: "3.12.1")
    incompatible = cli.collect_m2_dependency_metadata()
    incompatible_contract = incompatible["production_dependency_contract"]
    assert incompatible_contract["runtime_series_matches_actual"] is False
    assert any(
        "runtime series mismatch" in issue
        for issue in incompatible_contract["issues"]
    )
    with pytest.raises(RuntimeError, match="runtime series mismatch"):
        cli.require_production_dependency_identity(incompatible)


def _canonical_tat_palettes_metadata(tmp_path: Path) -> dict[str, object]:
    module = tmp_path / "site-packages" / "tat_palettes" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("# immutable fixture\n", encoding="ascii")
    exact_pins = cli._declared_exact_requirement_pins()
    distributions = dict(exact_pins)
    distributions["tat-palettes"] = cli.TAT_PALETTES_VERSION
    runtime_declaration = (cli.REPO_ROOT / "runtime.txt").read_text().strip()
    expected_python = runtime_declaration.removeprefix("python-")
    metadata: dict[str, object] = {
        "python": "3.11.16",
        "implementation": "CPython",
        "distributions": distributions,
        "direct_urls": {
            "tat-palettes": json.loads(json.dumps(cli.TAT_PALETTES_DIRECT_URL))
        },
        "module_origins": {"tat_palettes": str(module)},
    }
    identity = cli._tat_palettes_identity(metadata)
    assert identity["passed"] is True
    metadata["production_dependency_contract"] = {
        "passed": True,
        "issues": [],
        "requirements_sha256": cli.sha256_file(cli.REPO_ROOT / "requirements.txt"),
        "runtime_sha256": cli.sha256_file(cli.REPO_ROOT / "runtime.txt"),
        "runtime_declaration": runtime_declaration,
        "expected_python": expected_python,
        "actual_python": "3.11.16",
        "expected_python_series": "3.11",
        "actual_python_series": "3.11",
        "implementation": "CPython",
        "runtime_declaration_matches_actual": False,
        "runtime_series_matches_actual": True,
        "exact_requirement_pins": exact_pins,
        "installed_exact_pin_versions": exact_pins,
        "tat_palettes_identity": identity,
    }
    return metadata


def test_tat_palettes_dependency_contract_requires_exact_sanitized_vcs_identity(
    tmp_path: Path,
) -> None:
    metadata = _canonical_tat_palettes_metadata(tmp_path)
    identity = metadata["production_dependency_contract"][  # type: ignore[index]
        "tat_palettes_identity"
    ]
    assert identity["expected_version"] == "0.2.2"
    assert identity["expected_direct_url"] == {
        "url": "https://github.com/WeathermanAAA/Triple-A-Tropics.git",
        "subdirectory": "palette",
        "vcs_info": {
            "vcs": "git",
            "requested_revision": "tat-palettes-v0.2.2",
            "commit_id": "569450d82ee2b7b8a54c11df9976b32f7a8f10af",
        },
    }
    cli.require_production_dependency_identity(metadata)


@pytest.mark.parametrize(
    "mutation",
    (
        "version",
        "editable_local",
        "url",
        "revision",
        "commit",
        "vcs",
        "subdirectory",
        "module_origin",
    ),
)
def test_tat_palettes_dependency_contract_kills_supply_chain_lookalikes(
    tmp_path: Path,
    mutation: str,
) -> None:
    metadata = _canonical_tat_palettes_metadata(tmp_path)
    distributions = metadata["distributions"]  # type: ignore[assignment]
    direct_url = metadata["direct_urls"]["tat-palettes"]  # type: ignore[index]
    if mutation == "version":
        distributions["tat-palettes"] = "0.2.2+lookalike"
    elif mutation == "editable_local":
        metadata["direct_urls"]["tat-palettes"] = {  # type: ignore[index]
            "url": "file:///tmp/palette",
            "dir_info": {"editable": True},
        }
    elif mutation == "url":
        direct_url["url"] = "https://example.invalid/lookalike.git"
    elif mutation == "revision":
        direct_url["vcs_info"]["requested_revision"] = "main"
    elif mutation == "commit":
        direct_url["vcs_info"]["commit_id"] = "0" * 40
    elif mutation == "vcs":
        direct_url["vcs_info"]["vcs"] = "hg"
    elif mutation == "subdirectory":
        direct_url["subdirectory"] = "."
    else:
        local_module = tmp_path / "checkout" / "tat_palettes" / "__init__.py"
        local_module.parent.mkdir(parents=True)
        local_module.write_text("# local lookalike\n", encoding="ascii")
        metadata["module_origins"]["tat_palettes"] = str(local_module)  # type: ignore[index]

    with pytest.raises(RuntimeError, match="tat-palettes VCS identity"):
        cli.require_production_dependency_identity(metadata)


def test_dependency_metadata_names_every_direct_m2_science_input() -> None:
    distributions = cli.collect_dependency_metadata()["distributions"]
    exact_pins = cli._declared_exact_requirement_pins()
    required = {
        "numpy",
        "scipy",
        "matplotlib",
        "cartopy",
        "Pillow",
        "pyspectral",
        "pyorbital",
        "shapely",
        "PyYAML",
        "xarray",
        "h5netcdf",
        "h5py",
        "tat-palettes",
    }
    assert required <= distributions.keys()
    assert exact_pins.keys() <= distributions.keys()
    assert all(distributions[name] is not None for name in required)


def test_m2_source_metadata_binds_both_executed_ahi_authorities() -> None:
    metadata = cli.collect_m2_source_metadata()
    hashes = metadata["source_sha256"]
    for name in ("vendor/ahi_loader.py", "vendor/ahi_hsd.py", "colormaps.py"):
        assert name in metadata["m2_bound_product_and_residual_source_files"]
        assert hashes[name] == cli.sha256_file(cli.REPO_ROOT / name)
    assert metadata["m2_bound_local_render_data_files"] == [
        "cyclolab_ne_10m_states.geojson"
    ]
    assert metadata["local_render_data_sha256"] == {
        "cyclolab_ne_10m_states.geojson": cli.sha256_file(
            cli.REPO_ROOT / "cyclolab_ne_10m_states.geojson"
        )
    }


def test_m2_source_metadata_preserves_both_git_porcelain_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = cli._git

    def exact_status_git(*arguments: str, check: bool = True):
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return SimpleNamespace(stdout=" M app.py\n?? render_resample.py\n")
        return original_git(*arguments, check=check)

    monkeypatch.setattr(cli, "_git", exact_status_git)
    metadata = cli.collect_m2_source_metadata()
    assert metadata["git_status_porcelain"] == [
        " M app.py",
        "?? render_resample.py",
    ]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _v30_capture_fixture(tmp_path: Path):
    roles = (("red", "C02", 2), ("blue", "C01", 1), ("veggie", "C03", 3), ("ir", "C13", 13))
    objects: list[V30Object] = []
    frames: list[m2_production.TimestampedRendererFrame] = []
    provenance: list[dict[str, object]] = []
    raster_hashes: list[str] = []
    request_hash = _digest("request")
    grid_hash = _digest("grid")
    start = dt.datetime(2025, 10, 28, 14, tzinfo=dt.timezone.utc)
    for frame_index in range(19):
        nominal = start + dt.timedelta(minutes=10 * frame_index)
        exact = nominal + dt.timedelta(seconds=54.9)
        group: list[V30Object] = []
        for role, channel, band in roles:
            filename = f"frame-{frame_index:02d}-{channel}.nc"
            item = V30Object(
                nominal_slot_utc=nominal.isoformat(),
                exact_scan_start_utc=exact.isoformat(),
                exact_scan_end_utc=(exact + dt.timedelta(seconds=30)).isoformat(),
                date_created_utc=(exact + dt.timedelta(minutes=1)).isoformat(),
                channel=channel,
                role=role,
                key=f"fixture/{filename}",
                filename=filename,
                bytes=1024 + frame_index,
                sha256=_digest(f"source-{frame_index}-{role}"),
                band_id=band,
            )
            group.append(item)
            objects.append(item)
        raster = np.full((96, 96, 3), 40 + frame_index, dtype=np.uint8)
        daylight = np.ones((96, 96), dtype=np.bool_)
        raster_sha = cli._array_sha256(raster)
        raster_hashes.append(raster_sha)
        frame = m2_production.TimestampedRendererFrame(
            timestamp_utc=exact.isoformat(),
            raster_rgb_u8=raster,
            full_daylight_mask=daylight,
            request_identity_sha256=request_hash,
            grid_identity_sha256=grid_hash,
            source_object_sha256=tuple(item.sha256 for item in group),
            renderer_png_sha256=_digest(f"png-{frame_index}"),
            renderer_raster_sha256=raster_sha,
        )
        frames.append(frame)
        provenance.append(
            {
                "nominal_slot_utc": nominal.isoformat(),
                "exact_noaa_scan_start_utc": exact.isoformat(),
                "production_scan_start_utc": exact.replace(microsecond=0).isoformat(),
                "source_objects": [dataclasses.asdict(item) for item in group],
                "request_identity_sha256": request_hash,
                "grid_identity_sha256": grid_hash,
                "renderer_png_sha256": frame.renderer_png_sha256,
                "renderer_raster_sha256": raster_sha,
                "trace_identity_sha256": _digest(f"trace-{frame_index}"),
                "delivered_shape": [96, 96],
                "patch_bounds": [0, 96, 0, 96],
                "world_patch_max_pixel_edge_residual": 0.0,
                "world_patch_pixel_edge_tolerance": (
                    V30_PATCH_EDGE_BINDING_TOLERANCE_PX
                ),
            }
        )

    binding = m2_production.TimestampedSequenceBinding(
        schema=m2_production.M2_SEQUENCE_SCHEMA,
        frames=tuple(frames),
        patch_bounds=(0, 96, 0, 96),
        stationary_world_bounds=(-90.158249, 31.327946, -89.835017, 31.651178),
        stationary_class="clear_land",
        stationarity_method=m2_production.M2_STATIONARITY_METHOD,
        daylight_mask_producer_binding=(
            m2_production.M2_DAYLIGHT_MASK_PRODUCER_BINDING
        ),
    )
    corpus = V30Corpus(
        acquisition_directory=tmp_path / "acquisition",
        raw_directory=tmp_path / "raw",
        manifest_path=tmp_path / "acquisition" / "alternative_m2_objects.csv",
        land_zip_path=tmp_path / "land.zip",
        objects=tuple(objects),
        acquisition_index_sha256=V30_CANONICAL_ACQUISITION_INDEX_SHA256,
        manifest_sha256=V30_CANONICAL_MANIFEST_SHA256,
        land_zip_sha256=V30_NATURAL_EARTH_ZIP_SHA256,
        land_binding={
            "dataset": "Natural Earth 1:10m land",
            "zip_sha256": V30_NATURAL_EARTH_ZIP_SHA256,
            "patch_fully_covered": True,
            "distance_to_land_boundary_deg": 0.93765049,
            "minimum_required_distance_deg": 0.5,
        },
    )
    capture = V30SequenceCapture(
        schema=V30_CAPTURE_SCHEMA,
        binding=binding,
        corpus=corpus,
        frame_provenance=tuple(provenance),
        screening=tuple(
            {
                "nominal_slot_utc": (start + dt.timedelta(minutes=10 * index)).isoformat(),
                "dqf_zero_fraction_by_role": {
                    role: 1.0 for role in ("red", "blue", "veggie", "ir")
                },
                "cloud_proxy_fraction": 0.0,
                "c13_above_280k_fraction": 1.0,
                "cloud_proxy_definition": (
                    "C02>0.30 or C13<270K; acquisition screen only"
                ),
                "selection_policy": "predeclared_before_renderer_metric_evaluation",
            }
            for index in range(19)
        ),
        request_identity_sha256=request_hash,
        grid_identity_sha256=grid_hash,
        raster_sha256=tuple(raster_hashes),
        frozen_output_mutant_killed=True,
    )
    evaluation = m2_production.evaluate_timestamped_sequence(binding)
    assert evaluation.passed, evaluation.as_dict()
    return capture, evaluation


def test_v30_payloads_decode_to_all_exact_bound_uint8_patches(tmp_path: Path) -> None:
    capture, evaluation = _v30_capture_fixture(tmp_path)
    payloads, metadata = cli.build_v30_patch_payloads(capture, evaluation)
    png_names = sorted(name for name in payloads if name.endswith("_96px.png"))
    assert len(png_names) == 19
    stack = np.load(io.BytesIO(payloads["v30_patches_19x96x96_rgb_uint8.npy"]), allow_pickle=False)
    assert stack.shape == (19, 96, 96, 3)
    assert stack.dtype == np.uint8
    for index, name in enumerate(png_names):
        with Image.open(io.BytesIO(payloads[name])) as image:
            decoded = np.asarray(image)
        assert decoded.shape == (96, 96, 3)
        assert np.array_equal(decoded, stack[index])
        assert metadata[name]["sha256"] == cli.sha256_bytes(payloads[name])

    evidence = json.loads(payloads["v30_patch_evidence.json"])
    assert evidence["patch_count"] == 19
    assert evidence["patch_shape"] == [96, 96, 3]
    assert evidence["evaluation"]["outcome"] == "PASS"
    assert len(evidence["patches"]) == 19
    for index, row in enumerate(evidence["patches"]):
        assert row["array"]["sha256"] == cli._array_sha256(stack[index])
        assert row["png_sha256"] == cli.sha256_bytes(payloads[row["filename"]])
        assert row["metrics"] == cli._patch_metrics(stack[index])

    capture.binding.frames[0].raster_rgb_u8[0, 0, 0] ^= np.uint8(1)
    with pytest.raises(RuntimeError, match="differs|changed"):
        cli.build_v30_patch_payloads(capture, evaluation)


def test_v30_publication_metric_arithmetic_matches_registered_producer() -> None:
    yy, xx = np.indices((96, 96), dtype=np.uint16)
    patch = np.stack(
        ((xx + 1) % 256, (2 * yy + 1) % 256, (xx + yy + 3) % 256),
        axis=-1,
    ).astype(np.uint8)

    assert cli._patch_metrics(patch) == m2_production._frame_patch_metrics(patch)


def test_v30_payloads_reject_nominal_time_and_png_provenance_mutants(
    tmp_path: Path,
) -> None:
    capture, _ = _v30_capture_fixture(tmp_path)
    nominal_frames = tuple(
        dataclasses.replace(
            frame,
            timestamp_utc=capture.corpus.groups[index][0].nominal_slot_utc,
        )
        for index, frame in enumerate(capture.binding.frames)
    )
    nominal_binding = dataclasses.replace(capture.binding, frames=nominal_frames)
    nominal_provenance = tuple(
        {
            **row,
            "exact_noaa_scan_start_utc": nominal_frames[index].timestamp_utc,
        }
        for index, row in enumerate(capture.frame_provenance)
    )
    nominal_capture = dataclasses.replace(
        capture,
        binding=nominal_binding,
        frame_provenance=nominal_provenance,
    )
    nominal_evaluation = m2_production.evaluate_timestamped_sequence(nominal_binding)
    assert nominal_evaluation.passed
    with pytest.raises(RuntimeError, match="provenance differs"):
        cli.build_v30_patch_payloads(nominal_capture, nominal_evaluation)

    capture, evaluation = _v30_capture_fixture(tmp_path)
    provenance = list(capture.frame_provenance)
    provenance[0] = {
        **provenance[0],
        "renderer_png_sha256": _digest("substituted-png"),
    }
    png_mutant = dataclasses.replace(capture, frame_provenance=tuple(provenance))
    with pytest.raises(RuntimeError, match="provenance differs"):
        cli.build_v30_patch_payloads(png_mutant, evaluation)

    screens = list(capture.screening)
    screens[0] = {**screens[0], "cloud_proxy_fraction": 0.01}
    screen_mutant = dataclasses.replace(capture, screening=tuple(screens))
    with pytest.raises(RuntimeError, match="raw clear-land screen"):
        cli.build_v30_patch_payloads(screen_mutant, evaluation)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "world_patch_max_pixel_edge_residual",
            V30_PATCH_EDGE_BINDING_TOLERANCE_PX * 2.0,
        ),
        ("world_patch_pixel_edge_tolerance", 0.5),
        ("production_scan_start_utc", "2025-10-28T14:00:53+00:00"),
    ),
)
def test_v30_payloads_reject_world_geometry_and_production_time_mutants(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    capture, evaluation = _v30_capture_fixture(tmp_path)
    provenance = list(capture.frame_provenance)
    provenance[0] = {**provenance[0], field: replacement}
    mutant = dataclasses.replace(capture, frame_provenance=tuple(provenance))
    with pytest.raises(RuntimeError, match="provenance differs"):
        cli.build_v30_patch_payloads(mutant, evaluation)


def test_v30_payloads_reject_full_source_object_identity_mutant(
    tmp_path: Path,
) -> None:
    capture, evaluation = _v30_capture_fixture(tmp_path)
    provenance = list(capture.frame_provenance)
    source_objects = [dict(row) for row in provenance[0]["source_objects"]]
    source_objects[0]["role"] = "blue"
    provenance[0] = {**provenance[0], "source_objects": source_objects}
    mutant = dataclasses.replace(capture, frame_provenance=tuple(provenance))
    with pytest.raises(RuntimeError, match="provenance differs"):
        cli.build_v30_patch_payloads(mutant, evaluation)


def _minimal_report(metadata: dict[str, dict[str, object]]) -> dict[str, object]:
    """Deliberately insufficient stage-flag proxy used by rejection tests."""

    return {
        "schema": cli.M2_REPORT_SCHEMA,
        "command": "canonical-production",
        "summary": {
            "contract_passed": True,
            "stage_advancement_passed": True,
            "canonical_blockers": [],
            "mutation_count": 0,
            "mutation_kill_count": 0,
        },
        "repository": {
            "repository_root": "/fixture/repository",
            "branch": "fixture",
            "head": "a" * 40,
            "git_directory": "/fixture/gitdir",
            "git_common_directory": "/fixture/common",
        },
        "research_inputs": {"fixture": True},
        "harness_source_sha256": {"harness.py": "b" * 64},
        "source": {"source_sha256": {"render.py": "c" * 64}},
        "claim_boundary": {"scope": "M2 primary imagery layer"},
        "gallery_artifacts": metadata,
    }


def _artifact_metadata(payloads: dict[str, bytes]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "filename": name,
            "sha256": cli.sha256_bytes(payload),
            "bytes": len(payload),
            "media_type": "application/octet-stream",
            "width": None,
            "height": None,
            "description": f"fixture artifact {name}",
            "source_bindings": [],
            "source_binding_sha256": {},
        }
        for name, payload in payloads.items()
    }


@functools.lru_cache(maxsize=1)
def _canonical_publication_fixture_template() -> tuple[
    dict[str, object], dict[str, bytes], dict[str, dict[str, object]]
]:
    fixture_research = {
        "root": "/fixture/research",
        "validation_gates_csv": {
            "path": "/fixture/research/tables/validation_gates.csv",
            "sha256": "a" * 64,
        },
        "reference_pixels_json": {
            "path": "/fixture/research/sources/reference_pixels.json",
            "sha256": "b" * 64,
        },
    }
    claim_boundary = {
        "claim_scope": "primary_imagery_layer_only_until_m5",
        "allowed_claim": (
            "no post-CIRA spatial operation inside the M2 primary "
            "imagery-layer finalizer"
        ),
        "global_one_raster_claim": False,
        "downstream_encoded_resizes": list(cli.M5_SURVIVING_ENCODED_RESIZES),
        "night_mapping": {
            "id": "inverse_cira_continuous_preimage_v1",
            "input": "clean_ir_gray_from_delivered_bt_kelvin",
            "output_semantics": "cira_preimage_compatibility_coordinate",
            "physical_reflectance_claim": False,
            "black_endpoint": 0.0223,
            "white_endpoint": 2.1730825067739694,
        },
        "day_weight_policy": "legacy_88_to_95_no_survival_gate",
        "glint": "post_CIRA_pointwise_M8_compatibility_exception",
        "forbidden_pass_gate_ids": sorted(cli.FORBIDDEN_M2_CLOSURE_GATE_IDS),
    }
    v41_mutations = {
        "output_rgb_finite_proxy": {"difference_pixels": 7, "killed": True},
        "nearest_source_mask": {"difference_pixels": 5, "killed": True},
        "missing_geom_visibility_boundary": {
            "source_boundary_edges": 11,
            "killed": True,
        },
        "delivery_outside_captured_modality_union": {
            "injectable_unsupported_pixels": 13,
            "observed_escape_pixels": 0,
            "killed": True,
        },
        "over_tight_float64_proxy_tolerance": {
            "rejected_tolerance": 2.0e-6,
            "observed_max_abs_residual": 6.0e-5,
            "registered_tolerance": 1.0e-4,
            "killed": True,
        },
    }
    v41 = {
        "scene_bbox": [-105.0, -45.0, -25.0, 35.0],
        "source_geometric_boundary_edges": 11,
        "source_has_valid_invalid_boundary": True,
        "captured_modality_union_unsupported_pixels": 13,
        "delivery_support_escape_pixels": 0,
        "exact_formula_max_abs_residual": 0.0,
        "exact_formula_array_identity": True,
        # Mirror the exact float32 product threshold recorded by real V41
        # evidence.  Python's binary64 spelling of 1e-6 is not equal to it.
        "registered_coverage_tau": float(
            np.float32(cli.m2_validation.COVERAGE_TAU)
        ),
        "registered_tau_matches_oracle": True,
        "support_proof": (
            "captured_delivered_day_ir_coverage_union_plus_exact_"
            "day_weight_finalizer_formula"
        ),
        "numeric_crosscheck": {
            "procedure": (
                "float64_sparse_measured_Agg_response_replay_only;_"
                "never_used_to_invent_support"
            ),
            "day_coverage_max_abs_error": 6.0e-5,
            "ir_coverage_max_abs_error": 5.0e-5,
            "absolute_tolerance": 1.0e-4,
            "tolerance_justification": "fixture measured sparse replay residual",
            "passed": True,
        },
        "composite_domain": "linear_premultiplied",
        "background_contract": "opaque_pre_cira",
        "background_contract_passed": True,
        "mutation_discriminators": v41_mutations,
        "all_mutations_killed": True,
        "passed": True,
    }
    gates = [
        {
            "gate_id": gate_id,
            "outcome": "PASS",
            "canonical_closure": True,
            "evidence_scope": "canonical_bound_production_renderer",
            "threshold": "fixture threshold",
            "detail": "fixture measured pass",
            "measurements": v41 if gate_id == "V41" else {"fixture": True},
            "canonical_procedure": "fixture procedure",
            "canonical_source_document": "fixture source",
        }
        for gate_id in sorted(cli.M2_REQUIRED_GATE_IDS)
    ] + [
        {
            "gate_id": gate_id,
            "outcome": "DEFERRED",
            "canonical_closure": False,
            "evidence_scope": "explicit_M5_residual",
            "threshold": "fixture threshold",
            "detail": "Surviving encoded-pixel resizes prevent M2 closure.",
            "measurements": {
                "surviving_resizes": list(cli.M5_SURVIVING_ENCODED_RESIZES)
            },
            "canonical_procedure": "fixture procedure",
            "canonical_source_document": "fixture source",
        }
        for gate_id in sorted(cli.M5_DEFERRED_GATE_IDS)
    ] + [
        {
            "gate_id": gate_id,
            "outcome": "UNMEASURED",
            "canonical_closure": False,
            "evidence_scope": "explicitly_unclosed",
            "threshold": "fixture threshold",
            "detail": "M2 has no evidence that can close this coverage gate.",
            "measurements": {},
            "canonical_procedure": "fixture procedure",
            "canonical_source_document": "fixture source",
        }
        for gate_id in sorted(cli.FORBIDDEN_M2_CLOSURE_GATE_IDS)
    ]
    synthetic_oracle = cli.m2_validation.run_m2_contract_oracle(
        cli.verify_research_root("/fixture/research")
    )
    mutations = [
        json.loads(json.dumps(row.as_dict())) for row in synthetic_oracle.mutations
    ]
    oracle = {
        "schema": cli.M2_ORACLE_SCHEMA,
        "contract_passed": True,
        "stage_advancement_passed": True,
        "canonical_blockers": [],
        "gate_registry_issues": [],
        "mutation_registry_issues": [],
        "gates": gates,
        "mutations": mutations,
        "array_identities": [
            {
                "name": "fixture_source",
                "shape": [2, 2],
                "dtype": "<f4",
                "sha256": "c" * 64,
            }
        ],
        "product_color_protection": {
            "v07_passed": True,
            "v07_mismatches": {},
            "v08_passed": True,
            "v08_failures": [],
            "v08_declared_quantizer": "production_floor",
            "v08_delivered_hex": {},
            "cira_stretch_source_sha256": "d" * 64,
            "policy_rollback_binding": json.loads(
                json.dumps(cli.m2_validation.m2_policy_rollback_binding())
            ),
        },
        "claim_boundary": claim_boundary,
    }
    corpus_objects = [
        {
            "role": role,
            "file_identifier": f"{role}.nc",
            "bytes": index,
            "sha256": f"{index:064x}",
        }
        for index, role in enumerate(("red", "blue", "veggie", "ir"), start=1)
    ]
    canonical_corpus = {
        "schema": "tat_truecolor_m2_canonical_corpus_binding/v1",
        "root": "/fixture/corpus",
        "objects": corpus_objects,
    }
    canonical_corpus["inventory_sha256"] = cli.sha256_bytes(
        cli._strict_json_bytes(canonical_corpus)
    )
    rayleigh_files = [
        {"relative_path": "fixture.h5", "bytes": 4, "sha256": "2" * 64}
    ]
    rayleigh_file_set = cli.sha256_bytes(
        cli._strict_json_bytes(
            {
                "schema": cli.RAYLEIGH_DATA_INVENTORY_SCHEMA,
                "relative_paths": ["fixture.h5"],
            }
        )
    )
    rayleigh_inventory = cli.sha256_bytes(
        cli._strict_json_bytes(
            {
                "schema": cli.RAYLEIGH_DATA_INVENTORY_SCHEMA,
                "file_count": 1,
                "total_bytes": 4,
                "file_set_sha256": rayleigh_file_set,
                "files": rayleigh_files,
            }
        )
    )
    rayleigh = {
        "passed": True,
        "factory_binding": "production_fresh_cache_session",
        "production_cache_session_started_empty": True,
        "production_cache_exact_expected_keys": True,
        "production_cache_returned_identities_match": True,
        "data_directory": "/fixture/rayleigh",
        "config_file": "/fixture/pyspectral.yaml",
        "configured_rsr_directory": "/fixture/rayleigh",
        "configured_rayleigh_directory": "/fixture/rayleigh",
        "config_sha256": "3" * 64,
        "writable_probe_passed": True,
        "sensors": [
            {
                "platform": platform,
                "sensor": sensor,
                "status": "valid_corrector",
                "corrector_class": "fixture.Rayleigh",
                "detail": None,
            }
            for platform, sensor in cli.RAYLEIGH_STARTUP_SENSORS
        ],
        "data_inventory_schema": cli.RAYLEIGH_DATA_INVENTORY_SCHEMA,
        "data_inventory_file_count": 1,
        "data_inventory_total_bytes": 4,
        "data_inventory_file_set_sha256": rayleigh_file_set,
        "data_inventory_sha256": rayleigh_inventory,
        "data_inventory_files": rayleigh_files,
    }
    native_y, native_x = np.indices((32, 32), dtype=np.uint16)
    reference_floor = np.stack(
        (
            35 + (3 * native_x + native_y) % 170,
            45 + (native_x + 2 * native_y) % 160,
            30 + (2 * native_x + 3 * native_y) % 175,
        ),
        axis=-1,
    ).astype(np.uint8)
    candidate_floor = reference_floor.copy()
    old_nearest_floor = np.clip(
        reference_floor.astype(np.int16)
        + np.where((native_x + native_y) % 2 == 0, 18, -18)[..., None],
        0,
        255,
    ).astype(np.uint8)
    display_first_floor = np.clip(
        reference_floor.astype(np.int16) + np.asarray([5, -3, 7]),
        0,
        255,
    ).astype(np.uint8)
    candidate_float = candidate_floor.astype(np.float32) / np.float32(255.0)
    reference_float = reference_floor.astype(np.float32) / np.float32(255.0)
    display_first_float = (
        display_first_floor.astype(np.float32) / np.float32(255.0)
    )
    canonical_arrays = {
        "candidate_float": cli._array_record("candidate_float", candidate_float),
        "candidate_floor": cli._array_record("candidate_floor", candidate_floor),
        "exact_area_reference_float": cli._array_record(
            "exact_area_reference_float", reference_float
        ),
        "exact_area_reference_floor": cli._array_record(
            "exact_area_reference_floor", reference_floor
        ),
        "old_nearest_mutant_floor": cli._array_record(
            "old_nearest_mutant_floor", old_nearest_floor
        ),
        "display_first_mutant_float": cli._array_record(
            "display_first_mutant_float", display_first_float
        ),
        "display_first_mutant_floor": cli._array_record(
            "display_first_mutant_floor", display_first_floor
        ),
    }
    primary_rgba = np.concatenate(
        (
            candidate_floor,
            np.full((*candidate_floor.shape[:2], 1), 255, dtype=np.uint8),
        ),
        axis=-1,
    )
    primary_png_buffer = io.BytesIO()
    Image.fromarray(primary_rgba, mode="RGBA").save(
        primary_png_buffer, format="PNG", optimize=False, compress_level=9
    )
    primary_png_payload = primary_png_buffer.getvalue()

    def capture_summary(bbox: list[float]) -> dict[str, object]:
        return {
            "schema": cli.M2_PRODUCTION_SCHEMA,
            "bbox": bbox,
            "dpi": 100,
            "title_text": "fixture",
            "policy_version": cli.M2_REQUIRED_POLICY_VERSION,
            "png_bytes": len(primary_png_payload),
            "png_sha256": cli.sha256_bytes(primary_png_payload),
            "geometry": {
                "schema": "fixture-geometry/v1",
                "identity_sha256": "9" * 64,
            },
            "delivered_float_rgb": {
                **canonical_arrays["candidate_float"],
                "name": "delivered_float_rgb",
            },
            "delivered_floor_rgb": {
                **canonical_arrays["candidate_floor"],
                "name": "delivered_floor_rgb",
            },
            "figure_image_rgb": {
                **canonical_arrays["candidate_floor"],
                "name": "figure_image_rgb",
            },
            "identity_draw": {"floor_identity": True},
            "resample_calls": [],
            "pointwise_calls": [],
            "operations": [{"name": "fixture_physical_first"}],
            "endpoint_binding": {"passed": True, "fixture": "captured"},
            "trace_issues": [],
            "trace_identity_sha256": "e" * 64,
        }

    m1_baseline = {
        "directory": "/fixture/m1",
        "report_sha256": cli.M1_TRUSTED_REPORT_SHA256,
        "checksum_index_sha256": cli.M1_TRUSTED_CHECKSUM_INDEX_SHA256,
        "gallery_manifest_sha256": cli.M1_TRUSTED_GALLERY_MANIFEST_SHA256,
        "git_head": cli.M1_TRUSTED_GIT_HEAD,
        "scene_id": "fixture-scene",
        "source_shape": [64, 64, 3],
        "agg_internal_shape": [32, 32, 3],
        "agg_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "delivered_internal_bounds": [0, 32, 0, 32],
        "v15_out_of_range_fraction": 0.0,
        "v16_nearest_ringing_dn": 1.0,
        "v16_hanning_ringing_dn": 0.0,
        "v16_measurements": {"fixture": True},
        "object_identities": corpus_objects,
        "source_display_sha256": "f" * 64,
        "delivered_shape": [32, 32, 3],
        "m0_nearest_raster_sha256": "1" * 64,
        "m0_nearest_png_sha256": "2" * 64,
        "m1_hanning_raster_sha256": "3" * 64,
        "m1_hanning_png_sha256": "4" * 64,
        "m0_runtime_interpolations": ["nearest"],
        "m1_runtime_interpolations": ["hanning"],
    }
    terminator_discriminator = {
        "fixture": "alternating_source_sza_88_and_95_degrees",
        "approved_source_weight_then_resample_sha256": "5" * 64,
        "mutant_resample_cos_then_weight_sha256": "6" * 64,
        "maximum_abs_weight_difference": 0.25,
        "mean_abs_weight_difference": 0.125,
        "killed": True,
    }
    rollback_changed = {
        "artifact": "current_m1_hanning_delivered_raster",
        "mutation": "xor_0x01_at_one_retained_payload_byte",
        "payload_bytes": 32 * 32 * 3,
        "byte_offset": (32 * 32 * 3) // 2,
        "byte_before": 10,
        "byte_after": 11,
        "expected_sha256": "3" * 64,
        "approved_sha256": "3" * 64,
        "mutant_sha256": "7" * 64,
        "approved_matches_expected": True,
        "mutant_matches_expected": False,
        "target_hash_discriminator_killed": True,
        "full_replay_baseline_passed": True,
        "killed": True,
    }
    rollback_replay = {
        "schema": "tat_truecolor_m0_m1_current_rollback_replay/v1",
        "passed": True,
        "changed_byte_discriminator": rollback_changed,
    }
    production_discriminators = {
        "old_nearest": {
            "rms_dn": 18.0,
            "spectral_ratio": 1.5,
            "killed": True,
        },
        "display_first": {"mean_signed_dn": 3.0, "killed": True},
        "resample_cos_then_nonlinear_day_weight": terminator_discriminator,
        "output_proxy_canonical_coverage": v41_mutations[
            "output_rgb_finite_proxy"
        ],
        "current_m0_m1_rollback_bytes": rollback_changed,
    }
    v01_metrics = {
        "candidate_rms_dn": cli.m2_validation.rms_dn(
            candidate_floor, reference_floor
        ),
        "old_nearest_rms_dn": cli.m2_validation.rms_dn(
            old_nearest_floor, reference_floor
        ),
        "reference_quantizer": "production_floor_u8_no_dither",
        "reference": (
            "exact detector-cell area response derived from captured affine; "
            "physical reflectance/K/day-weight and coverage first, one CIRA last"
        ),
    }
    v03_metrics = {
        "candidate_ratio": cli.m2_validation.radial_high_frequency_power_ratio(
            candidate_floor, reference_floor, cutoff=0.3, bins=64
        ),
        "old_nearest_ratio": cli.m2_validation.radial_high_frequency_power_ratio(
            old_nearest_floor, reference_floor, cutoff=0.3, bins=64
        ),
        "luma": "Rec.601",
        "spatial_window": "separable_2D_Hann",
        "radial_bins": 64,
        "cutoff_cycles_per_pixel": 0.3,
    }
    coverage_row = {
        "source_rows": 64,
        "source_columns": 64,
        "contributing_rows": 64,
        "contributing_columns": 64,
        "discarded_rows": 0,
        "discarded_columns": 0,
        "discarded_row_fraction": 0.0,
        "discarded_column_fraction": 0.0,
        "minimum_weight": 0.0,
        "maximum_row_sum_error": 0.0,
        "procedure": "measured_Agg_impulse_response_source_support",
    }
    v02_metrics = {
        "candidate": dict(coverage_row),
        "old_nearest_mutant": dict(coverage_row),
    }
    primary_capture_fixture = capture_summary([-90.1, -89.8, 31.3, 31.6])
    limb_capture_fixture = capture_summary(v41["scene_bbox"])
    signed_all = float(
        np.mean(
            (candidate_float.astype(np.float64) - reference_float.astype(np.float64))
            * 255.0
        )
    )
    signed_luma = float(
        np.mean(
            np.tensordot(
                candidate_float.astype(np.float64)
                - reference_float.astype(np.float64),
                np.asarray((0.299, 0.587, 0.114), dtype=np.float64),
                axes=([-1], [0]),
            )
            * 255.0
        )
    )
    display_signed = float(
        np.mean(
            (
                display_first_float.astype(np.float64)
                - reference_float.astype(np.float64)
            )
            * 255.0
        )
    )
    v13_metrics = {
        "mean_signed_all_channel_dn": signed_all,
        "mean_signed_rec601_luma_dn": signed_luma,
        "display_first_mutant_mean_signed_dn": display_signed,
        "operation_trace": primary_capture_fixture["operations"],
        "trace_issues": [],
        "trace_identity_sha256": primary_capture_fixture["trace_identity_sha256"],
        "geometry_identity_sha256": primary_capture_fixture["geometry"][
            "identity_sha256"
        ],
        "terminator_order_mutant": terminator_discriminator,
        "captured_endpoint_binding": primary_capture_fixture["endpoint_binding"],
        "endpoint_identity_fixture": json.loads(
            json.dumps(
                cli.m2_production_module._endpoint_restoration_measurements()
            )
        ),
    }
    v15_metrics = {
        "maximum_out_of_range_fraction": 0.0,
        "tolerance": 2.0e-6,
        "fields": {
            name: {"out_of_range_fraction": 0.0, "comparisons": 10}
            for name in (
                "shared_day_fields", "ir_bt_kelvin",
                "solar_geometry_and_day_weight",
            )
        },
        "mapping": "captured_Agg_impulse_support_and_positive_source_coverage",
        "m1_report_sha256": m1_baseline["report_sha256"],
        "m1_baseline_out_of_range_fraction": 0.0,
    }
    v16_metrics = {
        "m1_report_sha256": m1_baseline["report_sha256"],
        "m1_git_head": m1_baseline["git_head"],
        "m1_nearest_ringing_dn": m1_baseline["v16_nearest_ringing_dn"],
        "m1_hanning_ringing_dn": m1_baseline["v16_hanning_ringing_dn"],
        "candidate": {
            "step_dn": 46.1,
            "overshoot_dn": 0.0,
            "undershoot_dn": 0.0,
            "ringing_dn": 0.0,
        },
        "candidate_edge_index": 16,
        "candidate_geometry_identity_sha256": "9" * 64,
        "candidate_source_pixels": 64,
        "candidate_delivered_pixels": 32,
        "candidate_x_scale": 0.5,
        "candidate_x_translation": 0.0,
        "procedure": (
            "fixed_seed_46.1_DN_plateau_fixture_inverse_CIRA_to_physical_"
            "then_measured_Hanning_response_then_CIRA_and_local_plateaus"
        ),
        "passed": True,
    }
    canonical = {
        "schema": cli.M2_PRODUCTION_SCHEMA,
        "scene_id": "fixture-scene",
        "bbox": [-90.1, -89.8, 31.3, 31.6],
        "objects": corpus_objects,
        "primary_capture": primary_capture_fixture,
        "limb_capture": limb_capture_fixture,
        "primary_png": {
            "sha256": cli.sha256_bytes(primary_png_payload),
            "bytes": len(primary_png_payload),
        },
        "arrays": canonical_arrays,
        "metrics": {
            "V01": v01_metrics,
            "V02": v02_metrics,
            "V03": v03_metrics,
            "V07": {
                "m1_report_sha256": m1_baseline["report_sha256"],
                "m1_git_head": m1_baseline["git_head"],
                "current_cira_source_sha256": cli.production_adapter.cira_source_sha256(),
            },
            "V08": json.loads(
                json.dumps(
                    cli.m2_production_module._reference_pixel_finalizer_measurements()
                )
            ),
            "V13": v13_metrics,
            "V14": json.loads(
                json.dumps(cli.m2_production_module._contrast_measurements())
            ),
            "V15": v15_metrics,
            "V16": v16_metrics,
            "V41": v41,
            "M0_M1_ROLLBACK": rollback_replay,
            "mutation_discriminators": production_discriminators,
            "limitations": ["fixture bounded production metrics"],
        },
        "m1_baseline": m1_baseline,
    }
    repeatability_hash = cli.sha256_bytes(primary_png_payload)
    repeatability_bytes = len(primary_png_payload)
    dependency_versions = {
        "numpy": np.__version__,
        "matplotlib": cli.importlib.metadata.version("matplotlib"),
        "Pillow": cli.importlib.metadata.version("Pillow"),
    }
    repeatability = {
        "passed": True,
        "hashes": [repeatability_hash] * 3,
        "output_bytes": [repeatability_bytes] * 3,
        "hash_provenance": "parent_recomputed_from_child_renderer_bytes",
        "manifests": [
            {
                "pid": os.getpid() + 100 + index,
                "parent_pid": os.getpid(),
                "process_start_ticks": 1000 + index,
                "start_method": "fork",
                "python": cli.platform.python_version(),
                "platform": cli.platform.platform(),
                "numpy": dependency_versions["numpy"],
                "matplotlib": dependency_versions["matplotlib"],
                "pillow": dependency_versions["Pillow"],
                "policy_version": cli.M2_REQUIRED_POLICY_VERSION,
                "request_identity_sha256": "5" * 64,
                "output_bytes": repeatability_bytes,
                "child_output_sha256": repeatability_hash,
                "parent_recomputed_output_sha256": repeatability_hash,
            }
            for index in range(3)
        ],
    }
    mutations.extend(
        (
            {
                "mutation_id": "canonical_old_nearest_behavior",
                "killed": True,
                "discriminator": "real-scene exact-area V01/V03",
                "measurements": production_discriminators["old_nearest"],
            },
            {
                "mutation_id": "canonical_display_first_behavior",
                "killed": True,
                "discriminator": "real-scene signed physical-domain V13",
                "measurements": production_discriminators["display_first"],
            },
            {
                "mutation_id": "resample_cos_then_nonlinear_day_weight",
                "killed": True,
                "discriminator": (
                    "source-evaluated terminator taper through captured response"
                ),
                "measurements": terminator_discriminator,
            },
            {
                "mutation_id": "output_proxy_canonical_coverage",
                "killed": True,
                "discriminator": (
                    "V02/V41 require measured input support and source coverage"
                ),
                "measurements": production_discriminators[
                    "output_proxy_canonical_coverage"
                ],
            },
            {
                "mutation_id": "caller_supplied_repeatability_hashes",
                "killed": True,
                "discriminator": "V25 hashes are recomputed from child renderer bytes",
                "measurements": {
                    "baseline_passed": True,
                    "changed_byte_index": repeatability_bytes // 2,
                    "changed_byte_mutant_passed": False,
                    "caller_hash_parameter_present": False,
                    "killed": True,
                },
            },
            {
                "mutation_id": "current_m0_m1_rollback_bytes",
                "killed": True,
                "discriminator": (
                    "current real-corpus legacy compositor and registered M0/M1 "
                    "renderer bytes versus immutable M1 hashes"
                ),
                "measurements": rollback_changed,
            },
        )
    )
    patch_bounds = [0, 96, 0, 96]
    stationary_world_bounds = [-90.1, 31.3, -89.8, 31.6]
    yy, xx = np.indices((96, 96), dtype=np.uint16)
    patch_arrays = [
        np.stack(
            (
                (xx + index) % 256,
                (2 * yy + index) % 256,
                (xx + yy + 3 * index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        for index in range(1, 20)
    ]
    sequence_frames = []
    for index, patch in enumerate(patch_arrays):
        timestamp = (
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
            + dt.timedelta(minutes=10 * index)
        ).isoformat()
        raster_record = cli._array_record(f"frame_{index}_rgb", patch)
        daylight_record = cli._array_record(
            f"frame_{index}_full_daylight", np.ones((96, 96), dtype=np.bool_)
        )
        sequence_frames.append(
            {
                "timestamp_utc": timestamp,
                "raster": raster_record,
                "daylight_mask": daylight_record,
                "request_identity_sha256": "5" * 64,
                "grid_identity_sha256": "6" * 64,
                "source_object_sha256": [
                    f"{100 + 4 * index + role_index:064x}"
                    for role_index in range(4)
                ],
                "renderer_png_sha256": cli.sha256_bytes(cli._png_bytes(patch)),
                "renderer_raster_sha256": raster_record["sha256"],
                "daylight_mask_producer_binding": "fixture-delivered-daylight/v1",
            }
        )
    # Model the real report producer, not the publication replay.  This keeps
    # the positive fixture capable of killing arithmetic drift between them.
    patch_metric_rows = [
        m2_production._frame_patch_metrics(patch) for patch in patch_arrays
    ]
    flicker_metrics = {}
    for metric_name in patch_metric_rows[0]:
        values = np.asarray(
            [row[metric_name] for row in patch_metric_rows], dtype=np.float64
        )
        second = values[2:] - 2.0 * values[1:-1] + values[:-2]
        median = float(np.median(second))
        flicker_metrics[metric_name] = (
            1.4826 * float(np.median(np.abs(second - median))) / np.sqrt(6.0)
        )
    sequence = {
        "outcome": "PASS",
        "metrics": {
            "full_daylight": True,
            "stationary_patch": True,
            "patch_height": 96,
            "patch_width": 96,
            **flicker_metrics,
        },
        "issues": [],
        "frames": sequence_frames,
        "patch_bounds": patch_bounds,
        "binding": {
            "schema": cli.V30_BINDING_SCHEMA,
            "frame_count": 19,
            "stationary_world_bounds": stationary_world_bounds,
            "stationary_class": "clear_land",
            "stationarity_method": "fixture-fixed-world-patch/v1",
            "daylight_mask_producer_binding": "fixture-delivered-daylight/v1",
        },
        "procedure": (
            "1.4826*MAD(per-frame metric second difference)/sqrt(6), "
            "recomputed from bound uint8 renderer arrays"
        ),
    }
    gate_measurements = {
        gate_id: canonical["metrics"][gate_id]
        for gate_id in (
            "V01", "V02", "V03", "V07", "V08", "V13", "V14", "V15",
            "V16", "V41",
        )
    }
    gate_measurements.update(
        {
            "V25": {
                **repeatability,
                "canonical_primary_png": canonical["primary_png"],
                "matches_canonical_primary_png": True,
            },
            "V30": sequence,
        }
    )
    gate_details = {
        "V01": "Real pinned ABI scene versus captured-affine exact-area physical reference.",
        "V02": "Source support enumerated from measured Agg impulse responses.",
        "V03": "Canonical Rec.601, 2-D Hann, 64-bin radial spectrum on delivered bytes.",
        "V07": "Protected constants plus immutable M1 artifact/repository binding.",
        "V08": "Five canonical pixels executed through the registered M2 finalizer and floor quantizer.",
        "V13": "Runtime operation trace binds reflectance/K/coverage/source day-weight delivery before the sole CIRA call.",
        "V14": "Closed-form contrast fixtures are bound to the captured physical-first trace.",
        "V15": "All delivered physical fields use actual local Agg support, not global extrema.",
        "V16": "Physical-first edge fixture compared with the verified M1 nearest baseline.",
        "V25": "Parent recomputed SHA-256 from bytes returned by three distinct renderer processes.",
        "V30": "No metric summary is accepted; flicker is recomputed from bound timestamped arrays.",
        "V41": "Real pinned full-disk objects rendered across the GOES limb with source-mask replay.",
    }
    for gate in gates:
        gate_id = gate["gate_id"]
        if gate_id in gate_measurements:
            gate["measurements"] = gate_measurements[gate_id]
            gate["detail"] = gate_details[gate_id]
    v30_corpus = {
        "acquisition_directory": "/fixture/v30/acquisition",
        "raw_directory": "/fixture/v30/raw",
        "manifest": {
            "path": "/fixture/v30/acquisition/manifest.csv",
            "sha256": cli.V30_CANONICAL_MANIFEST_SHA256,
        },
        "acquisition_index_sha256": cli.V30_CANONICAL_ACQUISITION_INDEX_SHA256,
        "land_zip": {
            "path": "/fixture/v30/land.zip",
            "sha256": cli.V30_NATURAL_EARTH_ZIP_SHA256,
        },
        "land_binding": {
            "dataset": "Natural Earth 1:10m land",
            "zip_sha256": cli.V30_NATURAL_EARTH_ZIP_SHA256,
            "patch_fully_covered": True,
            "distance_to_land_boundary_deg": 1.0,
            "minimum_required_distance_deg": 0.5,
        },
        "object_count": 76,
        "group_count": 19,
        "total_bytes": 7600,
        "inventory_sha256": "8" * 64,
    }
    provenance = []
    screening = []
    roles = ("red", "blue", "veggie", "ir")
    channels = ("C02", "C01", "C03", "C13")
    for index, frame in enumerate(sequence_frames):
        nominal = frame["timestamp_utc"]
        objects = []
        for role_index, (role, channel) in enumerate(zip(roles, channels, strict=True)):
            digest = frame["source_object_sha256"][role_index]
            objects.append(
                {
                    "nominal_slot_utc": nominal,
                    "exact_scan_start_utc": nominal,
                    "exact_scan_end_utc": nominal,
                    "date_created_utc": nominal,
                    "channel": channel,
                    "role": role,
                    "key": f"fixture/{index:02d}_{channel}.nc",
                    "filename": f"{index:02d}_{channel}.nc",
                    "bytes": 100,
                    "sha256": digest,
                    "band_id": int(channel[1:]),
                }
            )
        provenance.append(
            {
                "nominal_slot_utc": nominal,
                "exact_noaa_scan_start_utc": nominal,
                "production_scan_start_utc": nominal,
                "source_objects": objects,
                "request_identity_sha256": frame["request_identity_sha256"],
                "grid_identity_sha256": frame["grid_identity_sha256"],
                "renderer_png_sha256": frame["renderer_png_sha256"],
                "renderer_raster_sha256": frame["renderer_raster_sha256"],
                "trace_identity_sha256": f"{3000 + index:064x}",
                "delivered_shape": [96, 96],
                "patch_bounds": patch_bounds,
                "world_patch_max_pixel_edge_residual": 0.0,
                "world_patch_pixel_edge_tolerance": (
                    cli.V30_PATCH_EDGE_BINDING_TOLERANCE_PX
                ),
            }
        )
        screening.append(
            {
                "nominal_slot_utc": nominal,
                "dqf_zero_fraction_by_role": {role: 1.0 for role in roles},
                "cloud_proxy_fraction": 0.0,
                "c13_above_280k_fraction": 1.0,
                "cloud_proxy_definition": (
                    "C02>0.30 or C13<270K; acquisition screen only"
                ),
                "selection_policy": (
                    "predeclared_before_renderer_metric_evaluation"
                ),
            }
        )
    provenance_objects = [
        source
        for frame_provenance in provenance
        for source in frame_provenance["source_objects"]
    ]
    v30_corpus["inventory_sha256"] = cli.sha256_bytes(
        "".join(
            f"{row['sha256']}  {row['filename']}\n"
            for row in sorted(provenance_objects, key=lambda value: value["filename"])
        ).encode("ascii")
    )
    v30_capture = {
        "schema": cli.V30_CAPTURE_SCHEMA,
        "corpus": v30_corpus,
        "request_identity_sha256": "5" * 64,
        "grid_identity_sha256": "6" * 64,
        "raster_sha256": [
            frame["renderer_raster_sha256"] for frame in sequence_frames
        ],
        "distinct_raster_count": 19,
        "frozen_output_mutant_killed": True,
        "patch_bounds": patch_bounds,
        "stationary_world_bounds": stationary_world_bounds,
        "stationary_class": "clear_land",
        "stationarity_method": "fixture-fixed-world-patch/v1",
        "frames": provenance,
        "screening": screening,
    }
    input_bindings = {
        "canonical_corpus": canonical_corpus,
        "verified_m1_artifact": m1_baseline,
        "rayleigh_startup": rayleigh,
        "runner_image": json.loads(json.dumps(cli.M2_RUNNER_IMAGE_PROVENANCE)),
        "v30_corpus": v30_corpus,
    }
    production = {
        "schema": cli.M2_PRODUCTION_SCHEMA,
        "stage_advancement_passed": True,
        "binding_issues": [],
        "canonical": canonical,
        "repeatability": repeatability,
        "sequence": sequence,
        "oracle": oracle,
        "input_bindings": input_bindings,
        "rayleigh_startup": rayleigh,
    }

    gallery_payloads = {
        "01_m2_candidate_native.png": cli._png_bytes(candidate_floor),
        "02_exact_area_reference_native.png": cli._png_bytes(reference_floor),
        "03_old_nearest_mutant_native.png": cli._png_bytes(old_nearest_floor),
        "04_display_first_mutant_native.png": cli._png_bytes(display_first_floor),
    }
    errors = {
        "candidate": cli.m2_gallery_module._signed_rec601_error(
            candidate_floor, reference_floor
        ),
        "old_nearest_mutant": cli.m2_gallery_module._signed_rec601_error(
            old_nearest_floor, reference_floor
        ),
        "display_first_mutant": cli.m2_gallery_module._signed_rec601_error(
            display_first_floor, reference_floor
        ),
    }
    shared_scale = cli.m2_gallery_module._canonical_error_scale(tuple(errors.values()))
    error_maps = {
        name: cli.m2_gallery_module._signed_error_map(values, shared_scale)
        for name, values in errors.items()
    }
    for index, name in enumerate(cli.m2_gallery_module.ERROR_ORDER, start=5):
        gallery_payloads[f"{index:02d}_{name}_signed_rec601_error_native.png"] = (
            cli._png_bytes(error_maps[name])
        )
    main_sheet, main_size, main_bounds = cli.m2_gallery_module._comparison_sheet(
        (
            ("M2 candidate", candidate_floor),
            ("Exact-area reference", reference_floor),
            ("Old nearest mutant", old_nearest_floor),
            ("Display-first mutant", display_first_floor),
        )
    )
    error_sheet, error_size, error_bounds = cli.m2_gallery_module._comparison_sheet(
        tuple(
            (name.replace("_", " "), error_maps[name])
            for name in cli.m2_gallery_module.ERROR_ORDER
        )
    )
    gallery_payloads["08_native_comparison_sheet_100pct.png"] = main_sheet
    gallery_payloads["09_signed_error_sheet_shared_scale_100pct.png"] = error_sheet
    error_stack = np.stack(
        [errors[name] for name in cli.m2_gallery_module.ERROR_ORDER]
    ).astype(np.float32)
    gallery_payloads["10_signed_rec601_error_values_float32.npy"] = cli._npy_bytes(
        error_stack
    )
    gallery_payloads["11_signed_rec601_error_summary.csv"] = (
        cli.m2_gallery_module._error_summary_csv(errors, shared_scale)
    )
    spectral_rasters = {
        "candidate": candidate_floor,
        "exact_area_reference": reference_floor,
        "old_nearest_mutant": old_nearest_floor,
        "display_first_mutant": display_first_floor,
    }
    frequency, profiles = cli.m2_gallery_module._spectral_profiles(spectral_rasters)
    spectral_ratios = {
        name: cli.m2_validation.radial_high_frequency_power_ratio(
            raster,
            reference_floor,
            cutoff=cli.m2_gallery_module.SPECTRAL_CUTOFF,
            bins=cli.m2_gallery_module.SPECTRAL_BINS,
        )
        for name, raster in spectral_rasters.items()
        if name != "exact_area_reference"
    }
    gallery_payloads["12_canonical_spectral_comparison.csv"] = (
        cli.m2_gallery_module._spectral_csv(frequency, profiles)
    )
    gallery_payloads["13_canonical_spectral_comparison.png"] = (
        cli.m2_gallery_module._spectral_plot(frequency, profiles, spectral_ratios)
    )
    gallery_payloads["14_production_renderer.png"] = primary_png_payload
    canonical_summary_sha = cli.sha256_bytes(cli._strict_json_bytes(canonical))
    repeatability_sha = cli.sha256_bytes(cli._strict_json_bytes(repeatability))
    sequence_sha = cli.sha256_bytes(cli._strict_json_bytes(sequence))
    limitation_text = "\n".join(
        f"- {item}" for item in cli.M2_GALLERY_REQUIRED_LIMITATIONS
    )
    gallery_payloads["REVIEW_INSTRUCTIONS.md"] = (
        "# M2 immutable visual-review instructions\n\n"
        "Andrew verdict: PENDING. No visual review has been accepted. Visual review "
        "cannot close or relabel a scientific gate.\n\n"
        "Review the native candidate, exact-area reference, and both rejected mutants "
        "at 100% scale. The raster panels inside both comparison sheets are exact "
        "pixel-for-pixel embeddings; inspect them without browser zoom or smoothing.\n\n"
        "The signed Rec.601 maps use one shared symmetric scale declared in the future "
        "gallery manifest. Recompute their source values from the float32 NPY rather "
        "than sampling colors from the PNG. Recompute V03 from the spectral CSV with "
        "the canonical Rec.601 / square crop / 2-D Hann / 64-bin procedure.\n\n"
        "This file and every returned payload are immutable inputs to the transactional "
        "publisher. After publication, record any verdict in a separate append-only "
        "`ANDREW_VERDICT_<UTC_TIMESTAMP>.md` that binds the final gallery-manifest "
        "SHA-256. Never edit a published payload or checksum index.\n\n"
        "Exact limitations:\n\n"
        f"{limitation_text}\n"
    ).encode("utf-8")
    gallery_payloads["README.md"] = (
        "# M2 primary-imagery-layer review payloads\n\n"
        "Andrew verdict: PENDING. This gallery is not itself a stage-advancement "
        "decision and cannot substitute for the canonical gate report.\n\n"
        f"Scene: `{canonical['scene_id']}`\n\n"
        "Native delivered raster: `32 x 32` pixels\n\n"
        f"Canonical evidence summary SHA-256: `{canonical_summary_sha}`\n\n"
        "Repeatability summary: supplied; "
        f"passed=True; summary SHA-256={repeatability_sha}.\n\n"
        "Timestamped sequence summary: supplied; "
        f"outcome=PASS; summary SHA-256={sequence_sha}.\n\n"
        "Every standalone scientific raster is lossless and native scale. The two "
        "comparison sheets add labels only outside the embedded raster bounds. The "
        "production PNG is copied byte-for-byte rather than decoded and re-encoded.\n\n"
        "Exact limitations:\n\n"
        f"{limitation_text}\n"
    ).encode("utf-8")

    binding_hashes = {
        name: record["sha256"] for name, record in canonical_arrays.items()
    }
    binding_hashes.update(
        {
            "primary_png": canonical["primary_png"]["sha256"],
            "evidence_summary": canonical_summary_sha,
        }
    )
    semantic_rows = {
        "01_m2_candidate_native.png": ("image/png", 32, 32, ("candidate_floor",)),
        "02_exact_area_reference_native.png": (
            "image/png", 32, 32, ("exact_area_reference_floor",)
        ),
        "03_old_nearest_mutant_native.png": (
            "image/png", 32, 32, ("old_nearest_mutant_floor",)
        ),
        "04_display_first_mutant_native.png": (
            "image/png", 32, 32, ("display_first_mutant_floor",)
        ),
        "05_candidate_signed_rec601_error_native.png": (
            "image/png", 32, 32, ("candidate_floor", "exact_area_reference_floor")
        ),
        "06_old_nearest_mutant_signed_rec601_error_native.png": (
            "image/png", 32, 32,
            ("old_nearest_mutant_floor", "exact_area_reference_floor")
        ),
        "07_display_first_mutant_signed_rec601_error_native.png": (
            "image/png", 32, 32,
            ("display_first_mutant_floor", "exact_area_reference_floor")
        ),
        "08_native_comparison_sheet_100pct.png": (
            "image/png", main_size[0], main_size[1],
            ("candidate_floor", "exact_area_reference_floor", "old_nearest_mutant_floor", "display_first_mutant_floor")
        ),
        "09_signed_error_sheet_shared_scale_100pct.png": (
            "image/png", error_size[0], error_size[1],
            ("candidate_floor", "old_nearest_mutant_floor", "display_first_mutant_floor", "exact_area_reference_floor")
        ),
        "10_signed_rec601_error_values_float32.npy": (
            "application/x-npy", 32, 32,
            ("candidate_floor", "old_nearest_mutant_floor", "display_first_mutant_floor", "exact_area_reference_floor")
        ),
        "11_signed_rec601_error_summary.csv": (
            "text/csv", None, None,
            ("candidate_floor", "old_nearest_mutant_floor", "display_first_mutant_floor", "exact_area_reference_floor")
        ),
        "12_canonical_spectral_comparison.csv": (
            "text/csv", None, None,
            ("candidate_floor", "exact_area_reference_floor", "old_nearest_mutant_floor", "display_first_mutant_floor")
        ),
        "13_canonical_spectral_comparison.png": (
            "image/png", 1200, 720,
            ("candidate_floor", "exact_area_reference_floor", "old_nearest_mutant_floor", "display_first_mutant_floor")
        ),
        "14_production_renderer.png": (
            "image/png", 32, 32, ("primary_png", "candidate_floor")
        ),
        "README.md": ("text/markdown", None, None, ("evidence_summary",)),
        "REVIEW_INSTRUCTIONS.md": (
            "text/markdown", None, None, ("evidence_summary",)
        ),
    }
    semantic_descriptions = {
        "01_m2_candidate_native.png": (
            "Registered M2 delivered map raster at exact native inspection scale."
        ),
        "02_exact_area_reference_native.png": (
            "Captured-affine exact-area physical reference at exact native inspection scale."
        ),
        "03_old_nearest_mutant_native.png": (
            "Rejected old nearest-neighbour behavior at exact native inspection scale."
        ),
        "04_display_first_mutant_native.png": (
            "Rejected display-domain-first behavior at exact native inspection scale."
        ),
        "05_candidate_signed_rec601_error_native.png": (
            "Signed Rec.601 luma error (candidate minus exact-area reference); "
            f"red positive, blue negative, shared symmetric scale {shared_scale:.6g} DN."
        ),
        "06_old_nearest_mutant_signed_rec601_error_native.png": (
            "Signed Rec.601 luma error (old_nearest_mutant minus exact-area reference); "
            f"red positive, blue negative, shared symmetric scale {shared_scale:.6g} DN."
        ),
        "07_display_first_mutant_signed_rec601_error_native.png": (
            "Signed Rec.601 luma error (display_first_mutant minus exact-area reference); "
            f"red positive, blue negative, shared symmetric scale {shared_scale:.6g} DN."
        ),
        "08_native_comparison_sheet_100pct.png": (
            "Four labeled native rasters embedded pixel-for-pixel at 100% scale."
        ),
        "09_signed_error_sheet_shared_scale_100pct.png": (
            "Three signed-error maps embedded pixel-for-pixel with one shared scale."
        ),
        "10_signed_rec601_error_values_float32.npy": (
            "Raw float32 signed Rec.601 DN errors with order candidate, old-nearest, "
            "display-first and shape [3,H,W]."
        ),
        "11_signed_rec601_error_summary.csv": (
            "Machine-readable signed Rec.601 error statistics and canonical shared scale."
        ),
        "12_canonical_spectral_comparison.csv": (
            "Canonical Rec.601, square-crop, 2-D-Hann, 64-bin radial spectra."
        ),
        "13_canonical_spectral_comparison.png": (
            "Plot of the exact canonical spectral profiles stored in the companion CSV."
        ),
        "14_production_renderer.png": (
            "Exact PNG bytes returned by the bound primary M2 production renderer."
        ),
        "README.md": (
            "Gallery index, source identity, optional evidence status, and exact limits."
        ),
        "REVIEW_INSTRUCTIONS.md": (
            "Immutable append-only review procedure; Andrew remains PENDING."
        ),
    }
    gallery_metadata = _artifact_metadata(gallery_payloads)
    for name, (media_type, row_width, row_height, bindings) in semantic_rows.items():
        gallery_metadata[name].update(
            {
                "media_type": media_type,
                "width": row_width,
                "height": row_height,
                "description": semantic_descriptions[name],
                "source_bindings": list(bindings),
                "source_binding_sha256": {
                    binding: binding_hashes[binding] for binding in bindings
                },
            }
        )
    gallery_manifest = {
        "schema": cli.GALLERY_SCHEMA,
        "scene_id": canonical["scene_id"],
        "bbox": canonical["bbox"],
        "native_shape": [32, 32, 3],
        "source_evidence": {
            "schema": cli.M2_PRODUCTION_SCHEMA,
            "summary_sha256": cli.sha256_bytes(cli._strict_json_bytes(canonical)),
            "array_records": canonical["arrays"],
            "primary_png": canonical["primary_png"],
            "production_policy": canonical["primary_capture"]["policy_version"],
            "source_objects": canonical["objects"],
        },
        "artifacts": gallery_metadata,
        "payload_index_sha256": cli.sha256_bytes(
            cli._strict_json_bytes(
                {
                    name: {"sha256": row["sha256"], "bytes": row["bytes"]}
                    for name, row in sorted(gallery_metadata.items())
                }
            )
        ),
        "native_comparison_sheet": {
            "artifact": "08_native_comparison_sheet_100pct.png",
            "scale_percent": 100,
            "panel_bounds": main_bounds,
            "panels_are_pixel_identical": True,
        },
        "signed_error": {
            "quantity": "candidate minus exact-area reference, signed Rec.601 luma DN",
            "rec601_coefficients_rgb": [0.299, 0.587, 0.114],
            "raw_values_artifact": "10_signed_rec601_error_values_float32.npy",
            "raw_values_dtype": "float32",
            "raw_values_shape": [3, 32, 32],
            "raw_values_order": list(cli.m2_gallery_module.ERROR_ORDER),
            "shared_scale_percentile": cli.m2_gallery_module.ERROR_SCALE_PERCENTILE,
            "shared_symmetric_limit_dn": shared_scale,
            "shared_scale_definition": (
                "max(1 DN, p99.9 absolute candidate error, p99.9 absolute old-nearest "
                "error, p99.9 absolute display-first error)"
            ),
            "comparison_sheet": {
                "artifact": "09_signed_error_sheet_shared_scale_100pct.png",
                "scale_percent": 100,
                "panel_bounds": error_bounds,
                "panels_are_pixel_identical": True,
            },
        },
        "spectral": {
            "csv": "12_canonical_spectral_comparison.csv",
            "plot": "13_canonical_spectral_comparison.png",
            "procedure": "Rec.601_square_crop_2D_Hann_64_radial_bins_v1",
            "bins": cli.m2_gallery_module.SPECTRAL_BINS,
            "cutoff_cycles_per_delivered_pixel": cli.m2_gallery_module.SPECTRAL_CUTOFF,
            "ratios_to_exact_area_reference": spectral_ratios,
        },
        "repeatability": {
            "present": True,
            "summary": repeatability,
            "summary_sha256": cli.sha256_bytes(
                cli._strict_json_bytes(repeatability)
            ),
        },
        "sequence": {
            "present": True,
            "summary": sequence,
            "summary_sha256": cli.sha256_bytes(cli._strict_json_bytes(sequence)),
        },
        "claim_boundary": {
            "stage_advancement_claimed_by_gallery": False,
            "scope": "M2 primary imagery-layer finalizer only",
            "limitations": list(cli.M2_GALLERY_REQUIRED_LIMITATIONS),
        },
        "andrew_review": {
            "verdict": "PENDING",
            "instructions": "REVIEW_INSTRUCTIONS.md",
            "supplement_pattern": "ANDREW_VERDICT_<UTC_TIMESTAMP>.md",
            "supplement_must_bind_final_gallery_manifest_sha256": True,
            "published_payloads_are_immutable": True,
        },
    }
    payloads = dict(gallery_payloads)
    payloads["m2_gallery_manifest.json"] = cli._strict_json_bytes(gallery_manifest)
    for index, patch in enumerate(patch_arrays, start=1):
        name = f"v30_patch_{index:02d}_96px.png"
        payloads[name] = cli._png_bytes(patch)
    patch_stack = np.stack(patch_arrays, axis=0)
    stack_name = "v30_patches_19x96x96_rgb_uint8.npy"
    payloads[stack_name] = cli._npy_bytes(patch_stack)
    patches = []
    for index, (patch, frame) in enumerate(
        zip(patch_arrays, sequence_frames, strict=True)
    ):
        name = f"v30_patch_{index + 1:02d}_96px.png"
        patches.append(
            {
                "index": index,
                "timestamp_utc": frame["timestamp_utc"],
                "filename": name,
                "png_sha256": cli.sha256_bytes(payloads[name]),
                "png_bytes": len(payloads[name]),
                "array": cli._array_record(f"v30_patch_{index + 1:02d}", patch),
                "metrics": cli._patch_metrics(patch),
                "parent_renderer_raster_sha256": frame[
                    "renderer_raster_sha256"
                ],
                "parent_renderer_png_sha256": frame["renderer_png_sha256"],
                "request_identity_sha256": frame["request_identity_sha256"],
                "grid_identity_sha256": frame["grid_identity_sha256"],
                "source_object_sha256": frame["source_object_sha256"],
            }
        )
    v30_evidence = {
        "schema": "tat_truecolor_m2_v30_patch_evidence/v1",
        "patch_count": 19,
        "patch_shape": [96, 96, 3],
        "patch_bounds": patch_bounds,
        "stationary_world_bounds": stationary_world_bounds,
        "capture_summary": v30_capture,
        "capture_summary_sha256": cli.sha256_bytes(
            cli._strict_json_bytes(v30_capture)
        ),
        "evaluation": sequence,
        "evaluation_summary_sha256": cli.sha256_bytes(
            cli._strict_json_bytes(sequence)
        ),
        "stack": {
            "filename": stack_name,
            "payload_sha256": cli.sha256_bytes(payloads[stack_name]),
            "payload_bytes": len(payloads[stack_name]),
            **cli._array_record("v30_patch_stack", patch_stack),
        },
        "patches": patches,
        "limitations": list(cli.PRODUCTION_LIMITATIONS),
    }
    payloads["v30_patch_evidence.json"] = cli._strict_json_bytes(v30_evidence)
    metadata = _artifact_metadata(payloads)
    metadata.update(gallery_metadata)
    metadata["m2_gallery_manifest.json"] = cli._artifact_row(
        "m2_gallery_manifest.json",
        payloads["m2_gallery_manifest.json"],
        media_type="application/json",
        description="Immutable manifest for all primary-layer M2 review payloads.",
        source_binding_sha256={
            "canonical_evidence_summary": canonical_summary_sha
        },
    )
    for index, patch_row in enumerate(patches, start=1):
        name = patch_row["filename"]
        metadata[name] = cli._artifact_row(
            name,
            payloads[name],
            media_type="image/png",
            width=96,
            height=96,
            description=(
                f"V30 frame {index:02d} predeclared clear-land full-daylight "
                "patch at exact delivered 100% scale."
            ),
            source_binding_sha256={
                "patch_array": patch_row["array"]["sha256"],
                "renderer_raster": patch_row["parent_renderer_raster_sha256"],
            },
        )
    metadata[stack_name] = cli._artifact_row(
        stack_name,
        payloads[stack_name],
        media_type="application/x-npy",
        width=96,
        height=96,
        description="Raw uint8 stack for all 19 bound V30 patches in timestamp order.",
        source_binding_sha256={
            "v30_patch_stack": v30_evidence["stack"]["sha256"]
        },
    )
    metadata["v30_patch_evidence.json"] = cli._artifact_row(
        "v30_patch_evidence.json",
        payloads["v30_patch_evidence.json"],
        media_type="application/json",
        description=(
            "Machine-readable V30 patch identities and independently recomputed metrics."
        ),
        source_binding_sha256={
            "v30_capture_summary": v30_evidence["capture_summary_sha256"],
            "v30_evaluation_summary": v30_evidence["evaluation_summary_sha256"],
            "v30_patch_stack": v30_evidence["stack"]["sha256"],
        },
    )
    report: dict[str, object] = {
        "schema": cli.M2_REPORT_SCHEMA,
        "command": "canonical-production",
        "summary": {
            "contract_passed": True,
            "stage_advancement_passed": True,
            "canonical_blockers": [],
            "mutation_count": len(mutations),
            "mutation_kill_count": len(mutations),
        },
        "repository": {
            "repository_root": "/fixture/repository",
            "branch": "fixture",
            "head": "a" * 40,
            "git_directory": "/fixture/gitdir",
            "git_common_directory": "/fixture/common",
        },
        "research_inputs": fixture_research,
        "harness_source_sha256": {"harness.py": "b" * 64},
        "source": {"source_sha256": {"render.py": "c" * 64}},
        "claim_boundary": claim_boundary,
        "limitations": list(cli.PRODUCTION_LIMITATIONS),
        "runner_image": json.loads(json.dumps(cli.M2_RUNNER_IMAGE_PROVENANCE)),
        "invocation": {
            "command": "canonical-production",
            "runner_image_id": cli.M2_RUNNER_IMAGE_ID,
            "runner_image_identity_source": "external_operator_declared",
            "container_self_attested": False,
        },
        "dependencies": {
            "python": cli.platform.python_version(),
            "implementation": "CPython",
            "platform": cli.platform.platform(),
            "distributions": dependency_versions,
            "production_dependency_contract": {"passed": True},
        },
        "oracle": oracle,
        "production_evaluation": production,
        "v30_capture": v30_capture,
        "gallery_artifacts": metadata,
    }
    report["evidence_payload_sha256"] = cli.sha256_bytes(
        cli.stable_json_bytes(report)
    )
    return report, payloads, metadata


def _canonical_publication_fixture() -> tuple[
    dict[str, object], dict[str, bytes], dict[str, dict[str, object]]
]:
    report, payloads, metadata = _canonical_publication_fixture_template()
    return copy.deepcopy(report), dict(payloads), copy.deepcopy(metadata)


def test_canonical_publication_accepts_exact_float32_product_coverage_tau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    recorded_tau = report["production_evaluation"]["canonical"]["metrics"][  # type: ignore[index]
        "V41"
    ]["registered_coverage_tau"]
    assert recorded_tau == float(np.float32(cli.m2_validation.COVERAGE_TAU))
    assert recorded_tau != cli.m2_validation.COVERAGE_TAU
    monkeypatch.setattr(
        cli,
        "_repository_publication_issues",
        lambda value: ((), ("fixture repository is precommit",)),
    )
    monkeypatch.setattr(
        cli, "_require_publication_dependency_identity", lambda value: None
    )
    assert (
        cli._classify_report_for_publication(
            report,
            payloads,
            metadata,
            artifact_leaf="scratch-float32-coverage-tau",
        )
        == "noncanonical_precommit_candidate"
    )


def test_bound_production_inputs_accept_exact_valid_rayleigh_sensor_schema() -> None:
    report, _, _ = _canonical_publication_fixture()
    production = report["production_evaluation"]  # type: ignore[index]
    rayleigh = production["rayleigh_startup"]
    assert all(
        set(sensor)
        == {"platform", "sensor", "status", "corrector_class", "detail"}
        and sensor["detail"] is None
        for sensor in rayleigh["sensors"]
    )
    cli._validate_bound_production_inputs(production["input_bindings"], rayleigh)

    rayleigh["sensors"][0]["detail"] = "unexpected diagnostic text"
    with pytest.raises(RuntimeError, match="Rayleigh corrector binding is invalid"):
        cli._validate_bound_production_inputs(
            production["input_bindings"], rayleigh
        )


def test_production_renderer_payload_requires_exact_opaque_rgba_png() -> None:
    _, payloads, _ = _canonical_publication_fixture()
    payload = payloads["14_production_renderer.png"]
    decoded = cli._decode_publication_png(
        payload, label="production renderer fixture", opaque_rgba=True
    )
    with Image.open(io.BytesIO(payload)) as image:
        assert image.mode == "RGBA"
        rgba = np.asarray(image).copy()
    assert np.all(rgba[..., 3] == 255)
    np.testing.assert_array_equal(decoded, rgba[..., :3])

    rgba[0, 0, 3] = 254
    mutant = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(
        mutant, format="PNG", optimize=False, compress_level=9
    )
    with pytest.raises(RuntimeError, match="exact opaque RGBA pixels"):
        cli._decode_publication_png(
            mutant.getvalue(),
            label="production renderer alpha mutant",
            opaque_rgba=True,
        )


def _reseal_report(report: dict[str, object]) -> None:
    report.pop("evidence_payload_sha256", None)
    report["evidence_payload_sha256"] = cli.sha256_bytes(
        cli.stable_json_bytes(report)
    )


def _replace_json_payload(
    report: dict[str, object],
    payloads: dict[str, bytes],
    metadata: dict[str, dict[str, object]],
    name: str,
    value: dict[str, object],
) -> None:
    payloads[name] = cli._strict_json_bytes(value)
    metadata[name]["sha256"] = cli.sha256_bytes(payloads[name])
    metadata[name]["bytes"] = len(payloads[name])
    report["gallery_artifacts"] = metadata


def _reseal_gallery_payload(
    report: dict[str, object],
    payloads: dict[str, bytes],
    metadata: dict[str, dict[str, object]],
    name: str,
    payload: bytes,
) -> None:
    payloads[name] = payload
    metadata[name]["sha256"] = cli.sha256_bytes(payload)
    metadata[name]["bytes"] = len(payload)
    manifest_name = "m2_gallery_manifest.json"
    manifest = json.loads(payloads[manifest_name])
    if name in manifest["artifacts"]:
        manifest["artifacts"][name] = json.loads(json.dumps(metadata[name]))
        manifest["payload_index_sha256"] = cli.sha256_bytes(
            cli._strict_json_bytes(
                {
                    artifact_name: {
                        "sha256": row["sha256"],
                        "bytes": row["bytes"],
                    }
                    for artifact_name, row in sorted(manifest["artifacts"].items())
                }
            )
        )
        payloads[manifest_name] = cli._strict_json_bytes(manifest)
        metadata[manifest_name]["sha256"] = cli.sha256_bytes(
            payloads[manifest_name]
        )
        metadata[manifest_name]["bytes"] = len(payloads[manifest_name])
    report["gallery_artifacts"] = metadata
    _reseal_report(report)


def test_transaction_contains_every_extra_in_report_manifest_and_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_value, payloads, metadata = _canonical_publication_fixture()
    monkeypatch.setattr(cli, "_repository_publication_issues", lambda report: ((), ()))
    monkeypatch.setattr(
        cli, "_require_publication_dependency_identity", lambda metadata: None
    )
    target = tmp_path / "canonical-m2"
    cli.publish_report_transactionally(
        report_value,
        target,
        additional_payloads=payloads,
        artifact_metadata=metadata,
        pre_publish_check=lambda: None,
    )
    report = json.loads((target / "m2-validation.json").read_text())
    manifest = json.loads((target / "manifest.json").read_text())
    assert report["gallery_artifacts"] == manifest["artifacts"] == metadata
    indexed = {
        line.split("  ", 1)[1]
        for line in (target / "SHA256SUMS").read_text().splitlines()
    }
    assert set(payloads) <= indexed
    for name, payload in payloads.items():
        assert (target / name).read_bytes() == payload
        assert cli.sha256_file(target / name) == metadata[name]["sha256"]


def test_stage_flag_proxy_cannot_be_published_as_canonical(tmp_path: Path) -> None:
    target = tmp_path / "proxy-canonical"
    with pytest.raises(RuntimeError, match="top-level schema"):
        cli.publish_report_transactionally(_minimal_report({}), target)
    assert not target.exists()
    assert not list(tmp_path.glob(".proxy-canonical.staging-*"))


def test_canonical_publisher_requires_final_recheck_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    monkeypatch.setattr(cli, "_repository_publication_issues", lambda value: ((), ()))
    monkeypatch.setattr(
        cli, "_require_publication_dependency_identity", lambda value: None
    )
    target = tmp_path / "canonical-without-recheck"
    with pytest.raises(RuntimeError, match="pre_publish_check"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_canonical_publisher_rechecks_staged_bytes_after_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    monkeypatch.setattr(cli, "_repository_publication_issues", lambda value: ((), ()))
    monkeypatch.setattr(
        cli, "_require_publication_dependency_identity", lambda value: None
    )
    target = tmp_path / "callback-byte-drift"

    def mutate_staging() -> None:
        staging = list(tmp_path.glob(".callback-byte-drift.staging-*"))
        assert len(staging) == 1
        (staging[0] / "README.md").write_bytes(b"mutated after final check\n")

    with pytest.raises(RuntimeError, match="bytes changed after pre-publish check"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
            pre_publish_check=mutate_staging,
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".callback-byte-drift.staging-*"))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("schema",), "tat_truecolor_m2_validation/proxy", "schema"),
        (("m5_stage_advancement_passed",), True, "top-level schema"),
        (("v44_passed",), True, "top-level schema"),
        (
            ("summary", "m5_stage_advancement_passed"),
            True,
            "unknown nested stage/closure claim",
        ),
        (
            ("oracle", "gates", 0, "measurements", "m8_closed"),
            True,
            "unknown nested stage/closure claim",
        ),
        (
            ("oracle", "mutations", 0, "measurements", "historical_true_color"),
            True,
            "historical true-color claim",
        ),
        (("command",), "contract-oracle", "canonical-production summary"),
        (("summary", "canonical_blockers"), ["V30"], "summary"),
        (("limitations",), list(cli.PRODUCTION_LIMITATIONS[:-1]), "bounded authority"),
        (("runner_image", "image_id"), "sha256:" + "0" * 64, "runner image"),
        (("invocation", "runner_image_id"), None, "runner image"),
        (("oracle", "stage_advancement_passed"), False, "oracle"),
        (("oracle", "claim_boundary", "global_one_raster_claim"), True, "claim boundary"),
        (("oracle", "claim_boundary", "glint"), "removed", "claim boundary"),
        (("oracle", "claim_boundary", "downstream_encoded_resizes"), [], "claim boundary"),
        (
            ("oracle", "gates", 0, "canonical_closure"),
            False,
            "required gates",
        ),
        (
            ("oracle", "gates", -1, "outcome"),
            "PASS",
            "V44, V45, and V46",
        ),
        (
            ("oracle", "gates", len(cli.M2_REQUIRED_GATE_IDS), "outcome"),
            "PASS",
            "V21 through V24",
        ),
        (
            ("oracle", "gates", 0, "gate_id"),
            "proxy-gate",
            "gate registry",
        ),
        (
            ("oracle", "mutations", 0, "mutation_id"),
            "proxy-mutation",
            "mutation",
        ),
        (
            ("oracle", "mutations", 0, "discriminator"),
            "proxy discriminator",
            "mutation evidence",
        ),
        (
            ("oracle", "gates", 0, "measurements"),
            {"nearest_aliasing_rms_dn": 0.0},
            "required gate evidence",
        ),
        (
            ("oracle", "gates", 0, "detail"),
            "M8 and historical products are closed",
            "required gate evidence",
        ),
        (
            ("production_evaluation", "canonical", "metrics", "V01"),
            {},
            "required gate evidence",
        ),
        (
            (
                "production_evaluation", "canonical", "metrics", "V03",
                "candidate_ratio",
                ),
                0.0,
                "gallery source binding",
        ),
        (
            ("production_evaluation", "stage_advancement_passed"),
            False,
            "production evaluation",
        ),
        (
            ("production_evaluation", "binding_issues"),
            ["injected"],
            "production evaluation",
        ),
        (("v30_capture", "schema"), "proxy", "V30 capture"),
        (
            ("v30_capture", "frames", 0, "renderer_raster_sha256"),
            "0" * 64,
            "V30 provenance/screening",
        ),
        (
            ("v30_capture", "screening", 0, "cloud_proxy_fraction"),
            0.5,
            "V30 provenance/screening",
        ),
        (
            ("v30_capture", "frames", 0, "source_objects", 0, "band_id"),
            1,
            "V30 source-object identity",
        ),
        (
            ("v30_capture", "corpus", "inventory_sha256"),
            "0" * 64,
            "V30 corpus aggregate",
        ),
        (
            ("production_evaluation", "canonical", "objects"),
            [],
            "production objects",
        ),
        (
            ("production_evaluation", "canonical", "m1_baseline"),
            {"report_sha256": cli.M1_TRUSTED_REPORT_SHA256},
            "production M1 baseline",
        ),
        (
            ("production_evaluation", "input_bindings", "v30_corpus"),
            {"group_count": 18},
            "V30",
        ),
        (
            (
                "production_evaluation",
                "input_bindings",
                "verified_m1_artifact",
                "report_sha256",
            ),
            "0" * 64,
            "trusted identities",
        ),
        (
            (
                "production_evaluation",
                "input_bindings",
                "canonical_corpus",
                "objects",
                1,
                "sha256",
            ),
            "1" * 64,
            "corpus inventory",
        ),
        (
            (
                "production_evaluation",
                "input_bindings",
                "rayleigh_startup",
                "data_inventory_files",
            ),
            [],
            "Rayleigh",
        ),
        (
            (
                "production_evaluation",
                "input_bindings",
                "rayleigh_startup",
                "factory_binding",
            ),
            "caller_supplied_factory",
            "Rayleigh",
        ),
        (
            (
                "production_evaluation",
                "input_bindings",
                "runner_image",
                "image_id",
            ),
            "sha256:" + "f" * 64,
            "runner-image",
        ),
        (
            ("dependencies", "production_dependency_contract", "passed"),
            False,
            "dependency contract",
        ),
        (("gallery_artifacts",), {}, "gallery artifact map"),
    ),
)
def test_canonical_publisher_rejects_semantic_report_mutants_before_staging(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
    message: str,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    cursor: object = report
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    target = tmp_path / "semantic-mutant"
    with pytest.raises(RuntimeError, match=message):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".semantic-mutant.staging-*"))


def test_canonical_publisher_rejects_circular_empty_v01_metric_proxy(
    tmp_path: Path,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    canonical = report["production_evaluation"]["canonical"]  # type: ignore[index]
    canonical["metrics"]["V01"] = {}  # type: ignore[index]
    oracle = report["oracle"]  # type: ignore[index]
    gate = next(row for row in oracle["gates"] if row["gate_id"] == "V01")
    gate["measurements"] = {}
    _reseal_report(report)
    target = tmp_path / "circular-empty-v01"
    with pytest.raises(RuntimeError, match="gallery source binding|V01 metrics"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_canonical_publisher_binds_v25_repeatability_to_primary_png(
    tmp_path: Path,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    production = report["production_evaluation"]  # type: ignore[index]
    repeatability = production["repeatability"]
    alternate_hash = "f" * 64
    alternate_bytes = production["canonical"]["primary_png"]["bytes"] + 1
    repeatability["hashes"] = [alternate_hash] * 3
    repeatability["output_bytes"] = [alternate_bytes] * 3
    for manifest in repeatability["manifests"]:
        manifest["output_bytes"] = alternate_bytes
        manifest["child_output_sha256"] = alternate_hash
        manifest["parent_recomputed_output_sha256"] = alternate_hash
    gate = next(
        row for row in report["oracle"]["gates"] if row["gate_id"] == "V25"  # type: ignore[index]
    )
    gate["measurements"] = {
        **repeatability,
        "canonical_primary_png": production["canonical"]["primary_png"],
        "matches_canonical_primary_png": True,
    }
    _reseal_report(report)
    target = tmp_path / "alternate-v25-bytes"
    with pytest.raises(RuntimeError, match="V25 repeatability"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


@pytest.mark.parametrize("payload_name", ("m2_gallery_manifest.json", "v30_patch_evidence.json"))
def test_canonical_publisher_rejects_byte_consistent_proxy_payload_semantics(
    tmp_path: Path,
    payload_name: str,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    parsed = json.loads(payloads[payload_name])
    if payload_name == "m2_gallery_manifest.json":
        parsed["native_comparison_sheet"]["scale_percent"] = 50
    else:
        parsed["capture_summary"] = {"schema": cli.V30_CAPTURE_SCHEMA}
        parsed["capture_summary_sha256"] = cli.sha256_bytes(
            cli._strict_json_bytes(parsed["capture_summary"])
        )
    payloads[payload_name] = cli._strict_json_bytes(parsed)
    metadata[payload_name]["sha256"] = cli.sha256_bytes(payloads[payload_name])
    metadata[payload_name]["bytes"] = len(payloads[payload_name])
    report["gallery_artifacts"] = metadata
    report.pop("evidence_payload_sha256")
    report["evidence_payload_sha256"] = cli.sha256_bytes(
        cli.stable_json_bytes(report)
    )

    target = tmp_path / f"proxy-{payload_name}"
    with pytest.raises(
        RuntimeError,
        match=(
            "gallery-manifest metadata|V30 stack/evidence metadata|"
            "native-scale|V30 patch evidence"
        ),
    ):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("payload_index", "payload-index"),
        ("andrew_approved", "gallery evaluation bindings"),
        ("andrew_acceptance_extra", "Andrew review state.*schema"),
        ("source_arrays", "source binding"),
        ("top_level_overclaim", "gallery manifest coverage"),
        ("claim_overclaim", "gallery claim boundary.*schema"),
    ),
)
def test_canonical_publisher_rejects_gallery_identity_and_review_mutants(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    name = "m2_gallery_manifest.json"
    manifest = json.loads(payloads[name])
    if mutation == "payload_index":
        manifest["payload_index_sha256"] = "0" * 64
    elif mutation == "andrew_approved":
        manifest["andrew_review"]["verdict"] = "APPROVED"
    elif mutation == "andrew_acceptance_extra":
        manifest["andrew_review"]["accepted_by"] = "Andrew"
    elif mutation == "source_arrays":
        manifest["source_evidence"]["array_records"] = {}
    elif mutation == "top_level_overclaim":
        manifest["stage_advancement_passed"] = True
    elif mutation == "claim_overclaim":
        manifest["claim_boundary"]["m5_closed"] = True
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    _replace_json_payload(report, payloads, metadata, name, manifest)
    _reseal_report(report)

    target = tmp_path / f"gallery-mutant-{mutation}"
    with pytest.raises(RuntimeError, match=message):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_canonical_publisher_rejects_unknown_artifact_metadata_claim(
    tmp_path: Path,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    metadata["01_m2_candidate_native.png"]["stage_advancement_passed"] = True
    report["gallery_artifacts"] = metadata
    _reseal_report(report)
    target = tmp_path / "artifact-metadata-overclaim"
    with pytest.raises(RuntimeError, match="artifact metadata schema"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


@pytest.mark.parametrize(
    ("name", "message"),
    (
        ("01_m2_candidate_native.png", "gallery artifact semantics differ"),
        ("v30_patch_01_96px.png", "V30 patch metadata differs"),
    ),
)
def test_canonical_publisher_rejects_resealed_artifact_description_overclaim(
    tmp_path: Path,
    name: str,
    message: str,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    metadata[name]["description"] = "M5/M8/V46 closed and approved."
    if name.startswith("01_"):
        manifest_name = "m2_gallery_manifest.json"
        manifest = json.loads(payloads[manifest_name])
        manifest["artifacts"][name] = json.loads(json.dumps(metadata[name]))
        _replace_json_payload(report, payloads, metadata, manifest_name, manifest)
    report["gallery_artifacts"] = metadata
    _reseal_report(report)
    target = tmp_path / f"description-overclaim-{name}"
    with pytest.raises(RuntimeError, match=message):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("native_png", "native gallery pixels differ"),
        ("signed_npy", "signed-error NPY differs"),
        ("signed_png", "signed-error PNG differs"),
        ("spectral_csv", "spectral CSV differs"),
        ("spectral_plot", "spectral plot differs"),
        ("readme", "gallery README is not reproducible"),
        ("review", "review instructions are not reproducible"),
    ),
)
def test_canonical_publisher_rejects_resealed_gallery_payload_proxies(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    if mutation == "native_png":
        name = "01_m2_candidate_native.png"
        with Image.open(io.BytesIO(payloads[name])) as image:
            values = np.asarray(image).copy()
        values[0, 0, 0] ^= np.uint8(1)
        payload = cli._png_bytes(values)
    elif mutation == "signed_npy":
        name = "10_signed_rec601_error_values_float32.npy"
        values = np.load(io.BytesIO(payloads[name]), allow_pickle=False).copy()
        values[0, 0, 0] += np.float32(1.0)
        payload = cli._npy_bytes(values)
    elif mutation == "signed_png":
        name = "05_candidate_signed_rec601_error_native.png"
        with Image.open(io.BytesIO(payloads[name])) as image:
            values = np.asarray(image).copy()
        values[0, 0, 0] ^= np.uint8(1)
        payload = cli._png_bytes(values)
    elif mutation == "spectral_csv":
        name = "12_canonical_spectral_comparison.csv"
        payload = payloads[name] + b"proxy,0,0,0,0\n"
    elif mutation == "spectral_plot":
        name = "13_canonical_spectral_comparison.png"
        with Image.open(io.BytesIO(payloads[name])) as image:
            values = np.asarray(image).copy()
        values[0, 0, 0] ^= np.uint8(1)
        payload = cli._png_bytes(values)
    elif mutation == "readme":
        name = "README.md"
        payload = payloads[name] + b"\nM8 closed.\n"
    elif mutation == "review":
        name = "REVIEW_INSTRUCTIONS.md"
        payload = payloads[name] + b"\nAndrew verdict: APPROVED.\n"
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    _reseal_gallery_payload(report, payloads, metadata, name, payload)

    target = tmp_path / f"gallery-payload-proxy-{mutation}"
    with pytest.raises(RuntimeError, match=message):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_canonical_publisher_rejects_byte_consistent_v30_stack_slice_mutant(
    tmp_path: Path,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    stack_name = "v30_patches_19x96x96_rgb_uint8.npy"
    evidence_name = "v30_patch_evidence.json"
    stack = np.load(io.BytesIO(payloads[stack_name]), allow_pickle=False).copy()
    stack[0, 0, 0, 0] ^= np.uint8(1)
    payloads[stack_name] = cli._npy_bytes(stack)
    metadata[stack_name]["sha256"] = cli.sha256_bytes(payloads[stack_name])
    metadata[stack_name]["bytes"] = len(payloads[stack_name])
    evidence = json.loads(payloads[evidence_name])
    evidence["stack"] = {
        "filename": stack_name,
        "payload_sha256": cli.sha256_bytes(payloads[stack_name]),
        "payload_bytes": len(payloads[stack_name]),
        **cli._array_record("v30_patch_stack", stack),
    }
    new_stack_sha = evidence["stack"]["sha256"]
    metadata[stack_name]["source_binding_sha256"]["v30_patch_stack"] = (
        new_stack_sha
    )
    metadata[evidence_name]["source_binding_sha256"]["v30_patch_stack"] = (
        new_stack_sha
    )
    _replace_json_payload(report, payloads, metadata, evidence_name, evidence)
    _reseal_report(report)

    target = tmp_path / "v30-stack-slice-mutant"
    with pytest.raises(RuntimeError, match="V30 patch identity"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_canonical_publisher_rejects_byte_consistent_v30_png_pixel_mutant(
    tmp_path: Path,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    patch_name = "v30_patch_01_96px.png"
    evidence_name = "v30_patch_evidence.json"
    with Image.open(io.BytesIO(payloads[patch_name])) as image:
        patch = np.asarray(image).copy()
    patch[0, 0, 0] ^= np.uint8(1)
    payloads[patch_name] = cli._png_bytes(patch)
    metadata[patch_name]["sha256"] = cli.sha256_bytes(payloads[patch_name])
    metadata[patch_name]["bytes"] = len(payloads[patch_name])
    evidence = json.loads(payloads[evidence_name])
    evidence["patches"][0]["png_sha256"] = cli.sha256_bytes(
        payloads[patch_name]
    )
    evidence["patches"][0]["png_bytes"] = len(payloads[patch_name])
    _replace_json_payload(report, payloads, metadata, evidence_name, evidence)
    _reseal_report(report)

    target = tmp_path / "v30-png-pixel-mutant"
    with pytest.raises(RuntimeError, match="PNG pixels differ"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_publisher_rejects_nonfinite_report_before_staging(tmp_path: Path) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    report["research_inputs"]["nonfinite_mutant"] = float("nan")  # type: ignore[index]
    target = tmp_path / "nonfinite-report"
    with pytest.raises(RuntimeError, match="strict JSON"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


@pytest.mark.parametrize(
    "payload",
    (
        b'{"schema":"tat_truecolor_m2_v30_patch_evidence/v1","x":NaN}\n',
        b'{"schema":"tat_truecolor_m2_v30_patch_evidence/v1","schema":"duplicate"}\n',
    ),
)
def test_publisher_rejects_non_strict_or_duplicate_evidence_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    name = "v30_patch_evidence.json"
    payloads[name] = payload
    metadata[name] = _artifact_metadata({name: payload})[name]
    report["gallery_artifacts"] = metadata
    report.pop("evidence_payload_sha256")
    report["evidence_payload_sha256"] = cli.sha256_bytes(
        cli.stable_json_bytes(report)
    )
    target = tmp_path / "strict-json-mutant"
    with pytest.raises(RuntimeError, match="strict JSON"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_dirty_passing_report_is_only_a_noncanonical_scratch_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    monkeypatch.setattr(
        cli,
        "_repository_publication_issues",
        lambda value: ((), ("repository status_porcelain_v1 is not empty",)),
    )
    monkeypatch.setattr(
        cli, "_require_publication_dependency_identity", lambda value: None
    )
    target = tmp_path / "scratch-precommit-candidate"
    cli.publish_report_transactionally(
        report,
        target,
        additional_payloads=payloads,
        artifact_metadata=metadata,
        pre_publish_check=lambda: None,
    )
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["evidence_class"] == "noncanonical_precommit_candidate"
    assert manifest["evaluation_stage_advancement_passed"] is True
    assert manifest["stage_advancement_passed"] is False
    assert manifest["canonical_blockers"] == ["REPOSITORY_PRECOMMIT_STATE"]

    report, payloads, metadata = _canonical_publication_fixture()
    with pytest.raises(RuntimeError, match=r"scratch-\*"):
        cli.publish_report_transactionally(
            report,
            tmp_path / "not-scratch",
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )


def test_repository_fatal_identity_never_publishes_even_to_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, payloads, metadata = _canonical_publication_fixture()
    monkeypatch.setattr(
        cli,
        "_repository_publication_issues",
        lambda value: (("repository is behind",), ()),
    )
    monkeypatch.setattr(
        cli, "_require_publication_dependency_identity", lambda value: None
    )
    target = tmp_path / "scratch-fatal"
    with pytest.raises(RuntimeError, match="repository is behind"):
        cli.publish_report_transactionally(
            report,
            target,
            additional_payloads=payloads,
            artifact_metadata=metadata,
        )
    assert not target.exists()


def test_repository_closure_requires_clean_status_and_head_tree_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "a" * 40
    tree = "b" * 40
    upstream_head = "c" * 40
    source_name = "render.py"
    harness_name = "tests/test_m2_cli_production.py"
    source_hashes = {
        name: cli.sha256_file(cli.REPO_ROOT / name)
        for name in cli.M2_PRODUCT_SOURCE_FILES
    }
    local_data_hashes = {
        name: cli.sha256_file(cli.REPO_ROOT / name)
        for name in cli.M2_LOCAL_RENDER_DATA_FILES
    }
    source_hash = source_hashes[source_name]
    harness_hash = cli.sha256_file(cli.REPO_ROOT / harness_name)
    report = {
        "repository": {
            "repository_root": str(cli.REPO_ROOT.resolve()),
            "branch": "fix/fixture",
            "head": head,
            "head_tree": tree,
            "upstream": "origin/fix/fixture",
            "upstream_head": upstream_head,
            "divergence": {"ahead": 0, "behind": 0},
            "status_porcelain_v1": [],
            "git_directory": "/fixture/worktree-gitdir",
            "git_common_directory": "/fixture/common-gitdir",
            "git_pointer": {
                "dot_git_kind": "linked_worktree_pointer",
                "dot_git_path": str(cli.REPO_ROOT / ".git"),
                "pointer_text": "gitdir: /fixture/worktree-gitdir",
                "resolved_git_directory": "/fixture/worktree-gitdir",
            },
        },
        "source": {
            "repository_root": str(cli.REPO_ROOT.resolve()),
            "git_head": head,
            "git_branch": "fix/fixture",
            "git_status_porcelain": [],
            "source_sha256": source_hashes,
            "m2_bound_product_and_residual_source_files": list(
                cli.M2_PRODUCT_SOURCE_FILES
            ),
            "local_render_data_sha256": local_data_hashes,
            "m2_bound_local_render_data_files": list(
                cli.M2_LOCAL_RENDER_DATA_FILES
            ),
            "cira_stretch_source_sha256": (
                cli.production_adapter.cira_source_sha256()
            ),
        },
        "harness_source_sha256": {harness_name: harness_hash},
    }

    def fake_git(*arguments: str, check: bool = True) -> SimpleNamespace:
        values = {
            ("rev-parse", "--show-toplevel"): str(cli.REPO_ROOT.resolve()),
            ("rev-parse", "HEAD"): head,
            ("rev-parse", "HEAD^{tree}"): tree,
            ("branch", "--show-current"): "fix/fixture",
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/fix/fixture",
            ("rev-parse", "@{upstream}"): upstream_head,
            ("rev-list", "--left-right", "--count", "HEAD...@{upstream}"): "0 0",
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }
        value = values[arguments]
        return SimpleNamespace(stdout=value + ("\n" if value else ""))

    monkeypatch.setattr(cli, "_git", fake_git)
    fresh_repository = json.loads(json.dumps(report["repository"]))
    monkeypatch.setattr(
        cli,
        "collect_repository_identity",
        lambda: json.loads(json.dumps(fresh_repository)),
    )
    monkeypatch.setattr(
        cli,
        "m2_harness_source_hashes",
        lambda root: {harness_name: harness_hash},
    )
    monkeypatch.setattr(
        cli,
        "_head_blob_sha256",
        lambda current_head, name: {
            **source_hashes,
            **local_data_hashes,
            harness_name: harness_hash,
        }[name],
    )
    assert cli._repository_publication_issues(report) == ((), ())

    pointer_mutant = json.loads(json.dumps(report))
    pointer_mutant["repository"]["git_pointer"]["pointer_text"] = (
        "gitdir: /fixture/lookalike-gitdir"
    )
    fatal, _ = cli._repository_publication_issues(pointer_mutant)
    assert any("linked-worktree pointer" in issue for issue in fatal)
    object_key_mutant = json.loads(json.dumps(report))
    object_key_mutant["repository"]["schema"] = "lookalike-repository/v1"
    fatal, _ = cli._repository_publication_issues(object_key_mutant)
    assert any("object keys" in issue for issue in fatal)

    missing_source = json.loads(json.dumps(report))
    missing_source["source"]["source_sha256"].pop("colormaps.py")
    fatal, _ = cli._repository_publication_issues(missing_source)
    assert any("product registry" in issue for issue in fatal)
    missing_local_data = json.loads(json.dumps(report))
    missing_local_data["source"]["local_render_data_sha256"].pop(
        "cyclolab_ne_10m_states.geojson"
    )
    fatal, _ = cli._repository_publication_issues(missing_local_data)
    assert any("local render-data" in issue for issue in fatal)

    report["repository"]["status_porcelain_v1"] = [" M render.py"]
    report["source"]["git_status_porcelain"] = [" M render.py"]
    fresh_repository["status_porcelain_v1"] = [" M render.py"]

    def dirty_git(*arguments: str, check: bool = True) -> SimpleNamespace:
        result = fake_git(*arguments, check=check)
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return SimpleNamespace(stdout=" M render.py\n")
        return result

    monkeypatch.setattr(cli, "_git", dirty_git)
    monkeypatch.setattr(
        cli,
        "_head_blob_sha256",
        lambda current_head, name: (
            "0" * 64 if name == source_name else harness_hash
        ),
    )
    fatal, precommit = cli._repository_publication_issues(report)
    assert fatal == ()
    assert any("status_porcelain_v1" in issue for issue in precommit)
    assert any("HEAD-tree blob" in issue for issue in precommit)
    report["source"]["git_head"] = "0" * 40
    fatal, _ = cli._repository_publication_issues(report)
    assert any("source Git identity" in issue for issue in fatal)


def test_explicit_contract_oracle_stays_nonclosing(tmp_path: Path) -> None:
    blockers = ["V01"]
    report = {
        "schema": cli.M2_REPORT_SCHEMA,
        "command": "contract-oracle",
        "summary": {
            "stage_advancement_passed": False,
            "canonical_blockers": blockers,
        },
        "repository": {
            "repository_root": "/fixture/repository",
            "branch": "fixture",
            "head": "a" * 40,
            "git_directory": "/fixture/gitdir",
            "git_common_directory": "/fixture/common",
        },
        "research_inputs": {"fixture": True},
        "harness_source_sha256": {"harness.py": "b" * 64},
        "source": {"source_sha256": {"render.py": "c" * 64}},
        "claim_boundary": {"scope": "contract only"},
        "oracle": {
            "schema": cli.M2_ORACLE_SCHEMA,
            "stage_advancement_passed": False,
            "canonical_blockers": blockers,
        },
    }
    target = tmp_path / "scratch-contract"
    cli.publish_report_transactionally(report, target)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["evidence_class"] == "non_closing_contract_oracle"
    assert manifest["stage_advancement_passed"] is False


def test_publisher_rejects_leaf_symlink_redirect_before_resolution(
    tmp_path: Path,
) -> None:
    redirect_parent = tmp_path / "redirect-parent"
    redirect_parent.mkdir()
    redirected = redirect_parent / "unexpected-publication"
    requested = tmp_path / "requested-artifact"
    requested.symlink_to(redirected)
    with pytest.raises(FileExistsError, match="requested artifact leaf"):
        cli.publish_report_transactionally(_minimal_report({}), requested)
    assert requested.is_symlink()
    assert not redirected.exists()


def _install_production_main_fakes(
    monkeypatch: pytest.MonkeyPatch,
    paths: cli.PreflightPaths,
    *,
    stage_passed: bool = True,
    source_drift: bool = False,
    rayleigh_inventory_drift: bool = False,
) -> tuple[list[str], dict[str, object]]:
    calls: list[str] = []
    captured: dict[str, object] = {}

    class FakeResearch:
        def as_dict(self):
            return {"research": "bound"}

    class FakeBaseline:
        def as_dict(self):
            return {"m1": "bound"}

    baseline = FakeBaseline()

    class FakeEvidence:
        bbox = (-84.1, 10.3, -72.1, 22.3)
        m1_baseline = baseline

        def summary_dict(self):
            return {"canonical": "bound"}

    evidence = FakeEvidence()

    class FakeRepeatability:
        passed = True

        def summary_dict(self):
            return {"passed": True, "hashes": ["r"] * 3}

    repeatability = FakeRepeatability()

    class FakeSequence:
        passed = True
        outcome = "PASS"

        def as_dict(self):
            return {"outcome": "PASS", "metrics": {"fixture": 0.0}}

    sequence = FakeSequence()

    class FakeV30Corpus:
        acquisition_directory = paths.v30_acquisition_directory
        raw_directory = paths.v30_raw_directory
        land_zip_path = paths.v30_land_zip

        def summary_dict(self):
            return {"v30_corpus": "bound"}

    v30_corpus = FakeV30Corpus()

    class FakeV30Capture:
        binding = object()

        def summary_dict(self):
            return {"v30_capture": "bound"}

    v30_capture = FakeV30Capture()

    class FakeRun:
        contract_passed = stage_passed
        stage_advancement_passed = stage_passed
        canonical_blockers = () if stage_passed else ("V30",)

    synthetic = FakeRun()
    evaluated_run = FakeRun()

    class FakeEvaluation:
        def __init__(self):
            self.oracle = evaluated_run
            self.binding_issues = ()
            self.stage_advancement_passed = stage_passed
            self.sequence = sequence

        def as_dict(self):
            return {
                "stage_advancement_passed": stage_passed,
                "binding_issues": [],
                "sequence": sequence.as_dict(),
            }

    evaluation = FakeEvaluation()

    class FakeRayleigh:
        passed = True

        def __init__(self, inventory_sha256: str):
            self.inventory_sha256 = inventory_sha256

        def as_dict(self):
            return {
                "passed": True,
                "config_sha256": "d" * 64,
                "data_inventory_schema": (
                    "tat_truecolor_rayleigh_data_inventory/v1"
                ),
                "data_inventory_file_count": 1,
                "data_inventory_total_bytes": 4,
                "data_inventory_file_set_sha256": "f" * 64,
                "data_inventory_sha256": self.inventory_sha256,
                "data_inventory_files": [
                    {
                        "relative_path": "fixture.h5",
                        "bytes": 4,
                        "sha256": self.inventory_sha256,
                    }
                ],
            }

    rayleigh = FakeRayleigh("a" * 64)
    changed_rayleigh = FakeRayleigh("b" * 64)
    repository = {
        "repository_root": str(cli.REPO_ROOT),
        "branch": "fixture",
        "head": "e" * 40,
        "git_directory": "/gitdir",
        "git_common_directory": "/common",
    }
    source_calls = 0

    monkeypatch.setattr(cli, "CanonicalM2Evidence", FakeEvidence)
    monkeypatch.setattr(cli, "RepeatabilityCapture", FakeRepeatability)
    monkeypatch.setattr(cli, "V30SequenceCapture", FakeV30Capture)
    monkeypatch.setattr(cli, "M2ProductionEvaluation", FakeEvaluation)
    monkeypatch.setattr(cli, "preflight_paths", lambda arguments: calls.append("preflight") or paths)
    monkeypatch.setattr(cli, "verify_research_root", lambda path: calls.append("research") or FakeResearch())
    monkeypatch.setattr(cli, "collect_repository_identity", lambda: calls.append("repository") or repository)

    def source_metadata():
        nonlocal source_calls
        source_calls += 1
        calls.append("source")
        return {"source_sha256": {"render.py": ("changed" if source_drift and source_calls > 1 else "stable")}}

    monkeypatch.setattr(cli, "collect_m2_source_metadata", source_metadata)
    monkeypatch.setattr(cli, "collect_dependency_metadata", lambda: calls.append("dependencies") or {"python": "fixture"})
    monkeypatch.setattr(
        cli,
        "collect_m2_dependency_metadata",
        lambda base=None: {"python": "fixture"} if base is None else dict(base),
    )
    monkeypatch.setattr(cli, "require_production_dependency_identity", lambda value: None)
    monkeypatch.setattr(cli, "m2_harness_source_hashes", lambda root: calls.append("harness") or {"harness": "stable"})
    rayleigh_calls = 0

    def startup_check(*_args, **_kwargs):
        nonlocal rayleigh_calls
        rayleigh_calls += 1
        calls.append("rayleigh")
        if rayleigh_calls == 1:
            assert "expected_production_evidence" not in _kwargs
        else:
            assert _kwargs["expected_production_evidence"] is rayleigh
        if rayleigh_inventory_drift and rayleigh_calls > 1:
            return changed_rayleigh
        return rayleigh

    monkeypatch.setattr(cli, "run_rayleigh_startup_check", startup_check)
    monkeypatch.setattr(cli, "run_m2_contract_oracle", lambda research: calls.append("synthetic") or synthetic)
    monkeypatch.setattr(cli, "verify_m1_artifact_baseline", lambda path: calls.append("m1") or baseline)
    monkeypatch.setattr(cli, "measure_canonical_m2", lambda *args, **kwargs: calls.append("measure") or evidence)
    corpus_binding = {
        "root": str(paths.corpus_root),
        "objects": [
            {"role": role, "file_identifier": f"{role}.nc", "bytes": 1, "sha256": _digest(role)}
            for role in ("red", "blue", "veggie", "ir")
        ],
    }
    monkeypatch.setattr(cli, "canonical_corpus_binding_from_evidence", lambda *args: calls.append("corpus-bind") or corpus_binding)
    monkeypatch.setattr(cli, "collect_canonical_corpus_binding", lambda *args: calls.append("corpus-recheck") or corpus_binding)
    reload_data = object()
    reload_rows = tuple(dict(row) for row in corpus_binding["objects"])
    monkeypatch.setattr(cli, "load_canonical_melissa_physical", lambda *args: calls.append("reload") or (reload_data, reload_rows))

    def repeat(data, **kwargs):
        assert data is reload_data
        calls.append("repeatability")
        return repeatability

    monkeypatch.setattr(cli, "capture_three_process_repeatability", repeat)
    monkeypatch.setattr(cli, "verify_v30_corpus", lambda *args: calls.append("v30-verify") or v30_corpus)
    monkeypatch.setattr(cli, "capture_v30_sequence", lambda corpus: calls.append("v30-capture") or v30_capture)
    monkeypatch.setattr(cli, "evaluate_canonical_m2", lambda *args, **kwargs: calls.append("evaluate") or evaluation)
    monkeypatch.setattr(cli, "build_m2_gallery_payloads", lambda *args, **kwargs: calls.append("gallery") or object())
    monkeypatch.setattr(cli, "build_v30_patch_payloads", lambda *args: calls.append("v30-payloads") or ({"v30.json": b"v30"}, {"v30.json": {"filename": "v30.json", "sha256": cli.sha256_bytes(b"v30"), "bytes": 3}}))
    extras = {"gallery.bin": b"gallery", "v30.json": b"v30"}
    extra_meta = {
        name: {"filename": name, "sha256": cli.sha256_bytes(payload), "bytes": len(payload)}
        for name, payload in extras.items()
    }
    monkeypatch.setattr(cli, "assemble_production_payloads", lambda *args: calls.append("assemble") or (extras, extra_meta))

    def report(run, **kwargs):
        calls.append("report")
        captured["report_run"] = run
        captured["report_kwargs"] = kwargs
        return {"fixture": True}

    monkeypatch.setattr(cli, "build_m2_report", report)

    def publish(report_value, directory, **kwargs):
        calls.append("publish")
        captured["published_report"] = report_value
        captured["payloads"] = kwargs["additional_payloads"]
        captured["metadata"] = kwargs["artifact_metadata"]
        kwargs["pre_publish_check"]()
        return cli.PublishedArtifacts(directory, "1" * 64, "2" * 64, "3" * 64)

    monkeypatch.setattr(cli, "publish_report_transactionally", publish)
    return calls, captured


def test_mocked_production_main_calls_every_bound_stage_in_order(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = cli.PreflightPaths(
        artifact_directory=external_tree / "artifacts" / "canonical",
        research_root=external_tree / "research",
        corpus_root=external_tree / "corpus",
        m1_artifact_directory=external_tree / "m1",
        rayleigh_data_directory=external_tree / "rayleigh",
        pyspectral_config=external_tree / "config" / "pyspectral.yaml",
        v30_acquisition_directory=external_tree / "acquisition",
        v30_raw_directory=external_tree / "raw",
        v30_land_zip=external_tree / "land" / "natural-earth.zip",
    )
    calls, captured = _install_production_main_fakes(monkeypatch, paths)
    old_config = os.environ.get("PSP_CONFIG_FILE")
    assert cli.main(_production_argv(external_tree, paths.artifact_directory)) == 0
    assert os.environ.get("PSP_CONFIG_FILE") == old_config
    assert calls[:19] == [
        "preflight",
        "research",
        "repository",
        "source",
        "dependencies",
        "harness",
        "rayleigh",
        "synthetic",
        "m1",
        "measure",
        "corpus-bind",
        "reload",
        "repeatability",
        "v30-verify",
        "v30-capture",
        "evaluate",
        "gallery",
        "v30-payloads",
        "assemble",
    ]
    assert "report" in calls and "publish" in calls
    kwargs = captured["report_kwargs"]
    assert captured["report_run"] is not None
    assert kwargs["command"] == "canonical-production"
    assert kwargs["limitations"] == cli.PRODUCTION_LIMITATIONS
    assert kwargs["production_evaluation"]["input_bindings"]
    assert kwargs["production_evaluation"]["input_bindings"]["runner_image"] == cli.M2_RUNNER_IMAGE_PROVENANCE
    assert captured["published_report"]["runner_image"] == cli.M2_RUNNER_IMAGE_PROVENANCE
    assert captured["published_report"]["invocation"] == {
        "command": "canonical-production",
        "runner_image_id": cli.M2_RUNNER_IMAGE_ID,
        "runner_image_identity_source": "external_operator_declared",
        "container_self_attested": False,
    }
    assert captured["payloads"] == {"gallery.bin": b"gallery", "v30.json": b"v30"}
    assert set(captured["metadata"]) == set(captured["payloads"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage_advancement_passed"] is True
    assert summary["command"] == "canonical-production"


def test_unmeasured_production_evaluation_publishes_nothing(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = external_tree / "artifacts" / "unmeasured"
    paths = cli.PreflightPaths(
        target,
        external_tree / "research",
        external_tree / "corpus",
        external_tree / "m1",
        external_tree / "rayleigh",
        external_tree / "config" / "pyspectral.yaml",
        external_tree / "acquisition",
        external_tree / "raw",
        external_tree / "land" / "natural-earth.zip",
    )
    calls, _ = _install_production_main_fakes(
        monkeypatch, paths, stage_passed=False
    )
    assert cli.main(_production_argv(external_tree, target)) == 2
    assert "gallery" not in calls
    assert "report" not in calls
    assert "publish" not in calls
    assert not target.exists()


def test_final_source_race_fails_cleanly_before_publish_rename(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = external_tree / "artifacts" / "source-race"
    paths = cli.PreflightPaths(
        target,
        external_tree / "research",
        external_tree / "corpus",
        external_tree / "m1",
        external_tree / "rayleigh",
        external_tree / "config" / "pyspectral.yaml",
        external_tree / "acquisition",
        external_tree / "raw",
        external_tree / "land" / "natural-earth.zip",
    )
    calls, _ = _install_production_main_fakes(
        monkeypatch, paths, source_drift=True
    )
    assert cli.main(_production_argv(external_tree, target)) == 2
    assert calls.count("source") == 2
    assert "publish" in calls
    assert not target.exists()


def test_final_rayleigh_inventory_race_fails_cleanly_before_publish_rename(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = external_tree / "artifacts" / "rayleigh-race"
    paths = cli.PreflightPaths(
        target,
        external_tree / "research",
        external_tree / "corpus",
        external_tree / "m1",
        external_tree / "rayleigh",
        external_tree / "config" / "pyspectral.yaml",
        external_tree / "acquisition",
        external_tree / "raw",
        external_tree / "land" / "natural-earth.zip",
    )
    calls, _ = _install_production_main_fakes(
        monkeypatch,
        paths,
        rayleigh_inventory_drift=True,
    )
    assert cli.main(_production_argv(external_tree, target)) == 2
    assert calls.count("rayleigh") == 2
    assert "publish" in calls
    assert not target.exists()
