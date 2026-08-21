"""Canonical production adapter for the bounded M2 true-colour milestone.

The synthetic oracle in :mod:`validation.truecolor.m2` is intentionally unable
to close M2.  This module supplies the missing, independently captured binding
to the real ABI loader, M2 renderer, Agg geometry, physical resampler, finalizer
and encoded renderer bytes.  It does not publish evidence; callers retain the
large arrays only long enough to build a transactional evidence package.

No output dimension or Matplotlib-version offset is assumed.  The exact-area
reference is derived from the affine returned by the live ``AxesImage`` probe,
then oriented and clipped with its identity-proved internal-to-canvas mapping.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import inspect
import io
import json
import math
import multiprocessing
import os
import platform
import subprocess
import tempfile
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence
from unittest import mock

import numpy as np

from . import m1, m2
from .gates import ResearchInputs
from .reference_pixels import (
    DISPLAY_ATOL,
    REFERENCE_PIXELS,
    expected_dn,
    quantize_display,
)


M2_PRODUCTION_SCHEMA: Final = "tat_truecolor_m2_production_capture/v1"
M2_SEQUENCE_SCHEMA: Final = "tat_truecolor_m2_timestamped_sequence/v2"
M2_DAYLIGHT_MASK_PRODUCER_BINDING: Final = (
    "delivered_geom_valid_and_day_coverage_tau_and_source_day_weight_eq_1/v1"
)
M2_STATIONARITY_METHOD: Final = (
    "predeclared Natural-Earth-interior clear-land patch; exact four-band "
    "DQF/cloud-proxy screen; live plan-derived pixels; all renderer arrays retained"
)
M2_SCENE_ID: Final = m1.M1_SCENE_ID
M2_CANONICAL_BBOX: Final = m1.M1_BBOX
# Same four pinned full-disk ABI objects, but the independent research-case-E
# western-limb box centred near 153.084 W, 44.8 N.  This is deliberately
# separate from the Melissa storm box: an interior scene cannot close V41.
M2_LIMB_BBOX: Final = (-159.084, 38.8, -147.084, 50.8)
M2_DEFAULT_DPI: Final = 110
M2_REQUIRED_POLICY_VERSION: Final = "tat-truecolor-m2-reflectance-resample-v1"
M2_REQUIRED_PHYSICAL_DOMAIN_TRANSPORT: Final = (
    "coverage_bound_affine_unit_interval_v1"
)
M1_REQUIRED_REPORT_SCHEMA: Final = "tat_truecolor_m1_validation/v2"
M1_REQUIRED_GALLERY_SCHEMA: Final = "tat_truecolor_m1_gallery/v2"
M1_TRUSTED_REPORT_SHA256: Final = (
    "d38d96586a043996999d8244b335e623073c5c180dc4313ae4a936e4fbc44628"
)
M1_TRUSTED_CHECKSUM_INDEX_SHA256: Final = (
    "1df1e6f01ee4be3b8d67b0b98bc0076594cbdb028fdcfecf22977086324e7607"
)
M1_TRUSTED_GALLERY_MANIFEST_SHA256: Final = (
    "a44377a4dcaf5d6ba66ab5c69187a4b84294babd6f3e53a220e429cd3e3c4a4c"
)
M1_TRUSTED_GIT_HEAD: Final = "1ef77fd69a6e1bc367718d6a052c3674090cf836"
M1_TRUSTED_SOURCE_SHAPE: Final = (1527, 1600, 3)
M1_TRUSTED_DELIVERED_SHAPE: Final = (1188, 1188, 3)
M1_TRUSTED_SOURCE_DISPLAY_SHA256: Final = (
    "399e943164ded11b57725c551a63561f079e2a7947382fe69c4ad08f7af9495b"
)
M1_TRUSTED_M0_NEAREST_RASTER_SHA256: Final = (
    "82550e774b1c9fc24d977ea966162747a7c908fc700a668e5d98501ad86e255d"
)
M1_TRUSTED_M0_NEAREST_PNG_SHA256: Final = (
    "5b331e90f936bd9a1bda7d90a026ecf4a62e6d26af521be8cab2999ff09aa108"
)
M1_TRUSTED_M1_HANNING_RASTER_SHA256: Final = (
    "45147c5d744377546a9c23703e5990561dbaa9e8500765d2c697233ace5d1774"
)
M1_TRUSTED_M1_HANNING_PNG_SHA256: Final = (
    "d4a49f49a34336ffc5d862c1272c9bfb1b5129958ea5e440b8f3564e58270019"
)
M1_TRUSTED_OBJECT_IDENTITIES: Final = (
    (
        "red",
        "OR_ABI-L2-CMIPF-M6C02_G19_s20253001500204_e20253001509513_c20253001509559.nc",
        423807160,
        "3b96a2d171004098000e237a20c2f46154a5a0932fd04bfc67f4db42766bdc7f",
    ),
    (
        "blue",
        "OR_ABI-L2-CMIPF-M6C01_G19_s20253001500204_e20253001509513_c20253001509558.nc",
        89126350,
        "cb9ac40495608cb7b0329faeb27d69f87afe1cf6c06999632c2a9187288505e5",
    ),
    (
        "veggie",
        "OR_ABI-L2-CMIPF-M6C03_G19_s20253001500204_e20253001509513_c20253001509560.nc",
        100943519,
        "fc51ee14d1040564be5764826bb8f1b4e33486f56e291154876d7a7fde3e2217",
    ),
    (
        "ir",
        "OR_ABI-L2-CMIPF-M6C13_G19_s20253001500204_e20253001509524_c20253001509582.nc",
        23540015,
        "cb94a7fbe42c6638bfeb7b415b920216d625c6a854d3520a74f12666ab88c017",
    ),
)
PRODUCTION_QUANTIZER: Final = "floor_production"
_RESPONSE_EPS: Final = float(np.finfo(np.float32).eps)
_LOCAL_RANGE_ATOL: Final = 2.0e-6
_V41_NUMERIC_CROSSCHECK_ATOL: Final = 1.0e-4
_V41_REJECTED_OVERTIGHT_ATOL: Final = 2.0e-6


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _output_proxy_mutation_discriminator(
    v41_measurements: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the measured V41 RGB-proxy mutant, or fail it closed.

    This row is deliberately derived from the real limb capture.  A literal
    policy assertion here would let the report claim a killed output-proxy
    mutant without ever demonstrating that output finiteness differs from
    captured source support.
    """

    mutations = v41_measurements.get("mutation_discriminators")
    if not isinstance(mutations, Mapping):
        return {
            "difference_pixels": 0,
            "killed": False,
            "unmeasured_reason": "V41 mutation measurements are absent",
        }
    measured = mutations.get("output_rgb_finite_proxy")
    if not isinstance(measured, Mapping):
        return {
            "difference_pixels": 0,
            "killed": False,
            "unmeasured_reason": "V41 output-finiteness proxy was not measured",
        }
    return dict(measured)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = _stable_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(item) for item in array.shape]}
    )
    return _sha256_bytes(header + array.tobytes(order="C"))


def _array_record(name: str, values: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(values)
    return {
        "name": name,
        "shape": [int(item) for item in array.shape],
        "dtype": array.dtype.str,
        "sha256": _array_sha256(array),
    }


def _parse_hash_sidecar(path: Path, expected_name: str) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1].lstrip("*") != expected_name:
        raise RuntimeError(f"invalid checksum sidecar: {path}")
    digest = fields[0]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"invalid SHA-256 in sidecar: {path}")
    return digest


def _parse_sha256sums(path: Path) -> Mapping[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"invalid SHA256SUMS row in {path}: {line!r}")
        digest, raw_name = fields
        name = raw_name.lstrip(" *")
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not name
            or Path(name).name != name
            or name in entries
        ):
            raise RuntimeError(f"unsafe SHA256SUMS row in {path}: {line!r}")
        entries[name] = digest
    if not entries:
        raise RuntimeError(f"empty checksum index: {path}")
    return entries


@dataclass(frozen=True, slots=True)
class M1ArtifactBaseline:
    directory: str
    report_sha256: str
    checksum_index_sha256: str
    gallery_manifest_sha256: str
    git_head: str
    scene_id: str
    source_shape: tuple[int, int, int]
    agg_internal_shape: tuple[int, int, int]
    agg_affine: tuple[tuple[float, float, float], ...]
    delivered_internal_bounds: tuple[int, int, int, int]
    v15_out_of_range_fraction: float
    v16_nearest_ringing_dn: float
    v16_hanning_ringing_dn: float
    v16_measurements: Mapping[str, Any]
    # Added after the original M2 fixture-facing fields so noncanonical gallery
    # fixtures can remain compact.  Only ``verify_m1_artifact_baseline`` may
    # populate authoritative values; empty defaults fail the replay binding.
    object_identities: tuple[Mapping[str, Any], ...] = ()
    source_display_sha256: str = ""
    delivered_shape: tuple[int, int, int] = (0, 0, 0)
    m0_nearest_raster_sha256: str = ""
    m0_nearest_png_sha256: str = ""
    m1_hanning_raster_sha256: str = ""
    m1_hanning_png_sha256: str = ""
    m0_runtime_interpolations: tuple[str | None, ...] = ()
    m1_runtime_interpolations: tuple[str | None, ...] = ()

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def verify_m1_artifact_baseline(
    directory: str | os.PathLike[str] | Path,
) -> M1ArtifactBaseline:
    """Verify the immutable M1 package before accepting any baseline number."""

    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"M1 artifact root is not a real directory: {root}")
    report_path = root / "m1-validation.json"
    report_sidecar = root / "m1-validation.json.sha256"
    checksum_path = root / "SHA256SUMS"
    checksum_sidecar = root / "SHA256SUMS.sha256"
    gallery_path = root / "gallery_manifest.json"
    for path in (
        report_path,
        report_sidecar,
        checksum_path,
        checksum_sidecar,
        gallery_path,
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required M1 artifact is absent or a symlink: {path}")

    report_sha = _sha256_file(report_path)
    if _parse_hash_sidecar(report_sidecar, report_path.name) != report_sha:
        raise RuntimeError("M1 report sidecar does not match report bytes")
    if report_sha != M1_TRUSTED_REPORT_SHA256:
        raise RuntimeError("M1 report does not match the trusted M1 closure identity")
    checksum_sha = _sha256_file(checksum_path)
    if _parse_hash_sidecar(checksum_sidecar, checksum_path.name) != checksum_sha:
        raise RuntimeError("M1 checksum-index sidecar does not match index bytes")
    if checksum_sha != M1_TRUSTED_CHECKSUM_INDEX_SHA256:
        raise RuntimeError(
            "M1 checksum index does not match the trusted M1 closure identity"
        )
    index = _parse_sha256sums(checksum_path)
    for name, expected in index.items():
        target = root / name
        if not target.is_file() or target.is_symlink() or target.parent != root:
            raise RuntimeError(f"indexed M1 artifact is absent or unsafe: {target}")
        actual = _sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"M1 artifact hash mismatch for {name}: {actual} != {expected}"
            )
    if index.get(report_path.name) != report_sha:
        raise RuntimeError("M1 report is not bound by SHA256SUMS")
    if index.get(gallery_path.name) != M1_TRUSTED_GALLERY_MANIFEST_SHA256:
        raise RuntimeError("trusted M1 gallery is not bound by SHA256SUMS")
    expected_top_level = set(index) | {checksum_path.name, checksum_sidecar.name}
    actual_top_level = {
        entry.name
        for entry in os.scandir(root)
        if entry.is_file(follow_symlinks=False)
    }
    if actual_top_level != expected_top_level or any(
        not entry.is_file(follow_symlinks=False) for entry in os.scandir(root)
    ):
        raise RuntimeError("M1 artifact top-level file set differs from its index")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
    gallery_sha = _sha256_file(gallery_path)
    if gallery_sha != M1_TRUSTED_GALLERY_MANIFEST_SHA256:
        raise RuntimeError(
            "M1 gallery manifest does not match the trusted M1 closure identity"
        )
    if report.get("schema") != M1_REQUIRED_REPORT_SCHEMA:
        raise RuntimeError("M1 report schema is not canonical v2")
    if gallery.get("schema") != M1_REQUIRED_GALLERY_SCHEMA:
        raise RuntimeError("M1 gallery schema is not canonical v2")
    if report.get("summary", {}).get("stage_advancement_passed") is not True:
        raise RuntimeError("M1 baseline report did not pass stage advancement")
    if gallery.get("stage_advancement_passed") is not True:
        raise RuntimeError("M1 gallery did not pass stage advancement")
    if gallery.get("report", {}).get("sha256") != report_sha:
        raise RuntimeError("M1 gallery does not bind the verified report")

    canonical = report.get("m1_contracts", {}).get("canonical", {})
    if canonical.get("evidence_kind") != "canonical_real_pinned_scene":
        raise RuntimeError("M1 report lacks canonical real-scene evidence")
    if canonical.get("scene_id") != M2_SCENE_ID:
        raise RuntimeError("M1 baseline scene differs from the M2 canonical scene")
    trusted_objects = tuple(
        {
            "role": role,
            "file_identifier": file_identifier,
            "bytes": byte_count,
            "sha256": digest,
        }
        for role, file_identifier, byte_count, digest in M1_TRUSTED_OBJECT_IDENTITIES
    )
    object_identities = tuple(dict(item) for item in canonical.get("objects", ()))
    if object_identities != trusted_objects:
        raise RuntimeError("M1 report canonical object identities are not trusted")
    source_shape = tuple(int(item) for item in canonical.get("source_shape", ()))
    delivered_shape = tuple(
        int(item) for item in canonical.get("delivered_shape", ())
    )
    if source_shape != M1_TRUSTED_SOURCE_SHAPE:
        raise RuntimeError("M1 report source shape differs from the trusted closure")
    if delivered_shape != M1_TRUSTED_DELIVERED_SHAPE:
        raise RuntimeError("M1 report delivered shape differs from the trusted closure")
    trusted_hash_fields = {
        "source_display_sha256": M1_TRUSTED_SOURCE_DISPLAY_SHA256,
        "production_nearest_raster_sha256": M1_TRUSTED_M0_NEAREST_RASTER_SHA256,
        "production_nearest_png_sha256": M1_TRUSTED_M0_NEAREST_PNG_SHA256,
        "explicit_hanning_raster_sha256": M1_TRUSTED_M1_HANNING_RASTER_SHA256,
        "explicit_hanning_png_sha256": M1_TRUSTED_M1_HANNING_PNG_SHA256,
    }
    for field, expected in trusted_hash_fields.items():
        if canonical.get(field) != expected:
            raise RuntimeError(
                f"M1 report {field} differs from the trusted closure identity"
            )
    runtime_interpolations = canonical.get("runtime_interpolations", {})
    if not isinstance(runtime_interpolations, Mapping):
        raise RuntimeError("M1 report runtime interpolation binding is absent")
    m0_runtime_interpolations = tuple(
        runtime_interpolations.get("production_nearest", ())
    )
    m1_runtime_interpolations = tuple(
        runtime_interpolations.get("explicit_hanning", ())
    )
    if m0_runtime_interpolations != ("nearest",):
        raise RuntimeError("M1 report M0 runtime interpolation is not nearest")
    if m1_runtime_interpolations != ("hanning",):
        raise RuntimeError("M1 report M1 runtime interpolation is not Hanning")
    gates = {row.get("gate_id"): row for row in report.get("gates", [])}
    for gate_id in ("V15", "V16"):
        if gates.get(gate_id, {}).get("outcome") != "PASS":
            raise RuntimeError(f"M1 baseline gate {gate_id} did not pass")
    v15 = gates["V15"]["measurements"]
    v16 = gates["V16"]["measurements"]
    head = str(report.get("source", {}).get("git_head", ""))
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise RuntimeError("M1 report has no valid repository commit identity")
    if head != M1_TRUSTED_GIT_HEAD:
        raise RuntimeError("M1 report commit differs from the trusted M1 closure")

    return M1ArtifactBaseline(
        directory=str(root),
        report_sha256=report_sha,
        checksum_index_sha256=checksum_sha,
        gallery_manifest_sha256=gallery_sha,
        git_head=head,
        scene_id=str(canonical["scene_id"]),
        object_identities=object_identities,
        source_shape=source_shape,
        source_display_sha256=str(canonical["source_display_sha256"]),
        delivered_shape=delivered_shape,
        m0_nearest_raster_sha256=str(
            canonical["production_nearest_raster_sha256"]
        ),
        m0_nearest_png_sha256=str(canonical["production_nearest_png_sha256"]),
        m1_hanning_raster_sha256=str(
            canonical["explicit_hanning_raster_sha256"]
        ),
        m1_hanning_png_sha256=str(canonical["explicit_hanning_png_sha256"]),
        m0_runtime_interpolations=m0_runtime_interpolations,
        m1_runtime_interpolations=m1_runtime_interpolations,
        agg_internal_shape=tuple(
            int(item) for item in canonical["agg_internal_shape"]
        ),
        agg_affine=tuple(
            tuple(float(item) for item in row)
            for row in canonical["agg_transform_matrix"]
        ),
        delivered_internal_bounds=tuple(
            int(item) for item in canonical["delivered_internal_pixel_bounds"]
        ),
        v15_out_of_range_fraction=float(
            v15["canonical_local_neighbourhood_out_of_range_fraction"]
        ),
        v16_nearest_ringing_dn=float(v16["nearest_ringing_dn"]),
        v16_hanning_ringing_dn=float(v16["hanning_ringing_dn"]),
        v16_measurements=dict(v16),
    )


@dataclass(frozen=True, slots=True)
class AffineAreaResponse:
    source_pixels: int
    destination_pixels: int
    scale: float
    translation: float
    response: np.ndarray


def exact_area_response_from_affine(
    source_pixels: int,
    destination_pixels: int,
    *,
    scale: float,
    translation: float,
) -> AffineAreaResponse:
    """Exact detector-cell overlap for a captured input-to-output affine.

    The affine maps source pixel *edges* into internal destination coordinates.
    Each destination cell ``[j, j+1]`` is inverted through that affine and
    intersected with source cells.  Valid overlap is normalized at clipped
    edges, matching the canonical constant-preserving reference convention.
    """

    if source_pixels <= 0 or destination_pixels <= 0:
        raise ValueError("response dimensions must be positive")
    scale = float(scale)
    translation = float(translation)
    if not math.isfinite(scale) or scale == 0.0 or not math.isfinite(translation):
        raise ValueError("captured affine scale/translation must be finite")
    response = np.zeros((destination_pixels, source_pixels), dtype=np.float64)
    for output_index in range(destination_pixels):
        edge0 = (output_index - translation) / scale
        edge1 = (output_index + 1.0 - translation) / scale
        start, stop = sorted((edge0, edge1))
        first = max(0, int(math.floor(start)))
        last = min(source_pixels, int(math.ceil(stop)))
        for source_index in range(first, last):
            response[output_index, source_index] = max(
                0.0,
                min(stop, source_index + 1.0) - max(start, source_index),
            )
        total = float(response[output_index].sum())
        if total <= 0.0:
            raise RuntimeError(
                "captured destination footprint has no source support: "
                f"output={output_index}, interval={(start, stop)}"
            )
        response[output_index] /= total
    return AffineAreaResponse(
        source_pixels=source_pixels,
        destination_pixels=destination_pixels,
        scale=scale,
        translation=translation,
        response=response,
    )


def _orient_and_slice_responses(
    response_y: np.ndarray,
    response_x: np.ndarray,
    plan,
) -> tuple[np.ndarray, np.ndarray]:
    if plan.delivered_internal_bounds is None:
        raise RuntimeError("captured plan has no delivered mapping")
    vertical = np.asarray(response_y, dtype=np.float64)
    horizontal = np.asarray(response_x, dtype=np.float64)
    if plan.internal_to_canvas_flip_y:
        vertical = vertical[::-1]
    if plan.internal_to_canvas_flip_x:
        horizontal = horizontal[::-1]
    top, bottom, left, right = plan.delivered_internal_bounds
    return vertical[top:bottom], horizontal[left:right]


def exact_area_responses_for_plan(plan) -> tuple[np.ndarray, np.ndarray]:
    vertical = exact_area_response_from_affine(
        plan.source_height,
        plan.destination_height,
        scale=plan.y_transform[0],
        translation=plan.y_transform[1],
    ).response
    horizontal = exact_area_response_from_affine(
        plan.source_width,
        plan.destination_width,
        scale=plan.x_transform[0],
        translation=plan.x_transform[1],
    ).response
    return _orient_and_slice_responses(vertical, horizontal, plan)


def _agg_axis_response_from_transform(
    source_pixels: int,
    destination_pixels: int,
    kernel: str,
    *,
    scale: float,
    translation: float,
    axis: int,
) -> np.ndarray:
    """Measure one production Agg axis with isolated impulse stripes.

    Packing every source impulse onto one image diagonal is not an independent
    axis measurement.  The orthogonal Hanning pass couples adjacent diagonal
    impulses, and its first/last impulses additionally sit on that unrelated
    image boundary.  Merely adding off-axis guard cells fixes the latter but
    leaves the former contamination in place.

    Measure each source position in its own call instead.  Its impulse is a
    constant three-pixel stripe across the orthogonal axis, so the sampled
    centre is interior and the orthogonal pass has unit DC response without a
    second tagged impulse to leak into it.  The measured axis retains its real
    first/last source boundaries.  Every call otherwise uses the exact
    low-level scalar ABI and arguments used by
    :func:`render_resample._hanning_resample`.
    """

    if source_pixels <= 0 or destination_pixels <= 0:
        raise ValueError("response dimensions must be positive")
    if axis not in {0, 1}:
        raise ValueError("response axis must be 0 or 1")
    if kernel not in {"hanning", "nearest"}:
        raise ValueError(f"unsupported measured response kernel: {kernel!r}")

    import render_resample

    agg_image, affine_type = render_resample._ensure_hanning_backend()  # noqa: SLF001
    interpolation = (
        agg_image.HANNING if kernel == "hanning" else agg_image.NEAREST
    )
    measured = np.empty((destination_pixels, source_pixels), dtype=np.float64)
    if axis == 0:
        impulses = np.zeros((source_pixels, 3), dtype=np.float32)
        result = np.zeros((destination_pixels, 3), dtype=np.float32)
        transform = affine_type.from_values(
            1.0, 0.0, 0.0, scale, 0.0, translation
        )
        for source_index in range(source_pixels):
            impulses[source_index, :] = np.float32(1.0)
            result.fill(np.float32(0.0))
            agg_image.resample(
                impulses,
                result,
                transform,
                interpolation,
                True,
                1.0,
                True,
                4.0,
            )
            measured[:, source_index] = result[:, 1]
            impulses[source_index, :] = np.float32(0.0)
    else:
        impulses = np.zeros((3, source_pixels), dtype=np.float32)
        result = np.zeros((3, destination_pixels), dtype=np.float32)
        transform = affine_type.from_values(
            scale, 0.0, 0.0, 1.0, translation, 0.0
        )
        for source_index in range(source_pixels):
            impulses[:, source_index] = np.float32(1.0)
            result.fill(np.float32(0.0))
            agg_image.resample(
                impulses,
                result,
                transform,
                interpolation,
                True,
                1.0,
                True,
                4.0,
            )
            measured[:, source_index] = result[1, :]
            impulses[:, source_index] = np.float32(0.0)
    if measured.shape != (destination_pixels, source_pixels):
        raise RuntimeError(f"unexpected isolated axis response shape: {measured.shape}")
    return measured


def measured_kernel_responses_for_plan(
    plan,
    *,
    kernel: str = "hanning",
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the installed Agg operator at the captured affine."""

    vertical = _agg_axis_response_from_transform(
        plan.source_height,
        plan.destination_height,
        kernel,
        scale=plan.y_transform[0],
        translation=plan.y_transform[1],
        axis=0,
    )
    horizontal = _agg_axis_response_from_transform(
        plan.source_width,
        plan.destination_width,
        kernel,
        scale=plan.x_transform[0],
        translation=plan.x_transform[1],
        axis=1,
    )
    # The measured response is in Agg's bottom-origin internal array. Apply
    # the independently captured device-to-canvas orientation and full/sliced
    # delivery mapping exactly as production does; source-affine sign is not a
    # substitute for this buffer-origin conversion.
    return _orient_and_slice_responses(vertical, horizontal, plan)


def _sparse_separable(
    values: np.ndarray,
    response_y: np.ndarray,
    response_x: np.ndarray,
    *,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Apply small-support separable responses without a multi-GiB temporary."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim not in (2, 3):
        raise ValueError("separable input must be HxW or HxWxC")
    if response_y.shape[1] != source.shape[0] or response_x.shape[1] != source.shape[1]:
        raise ValueError("response/source shapes differ")
    horizontal = np.empty(
        (source.shape[0], response_x.shape[0], *source.shape[2:]),
        dtype=np.float64,
    )
    for output_x, row in enumerate(response_x):
        support = np.flatnonzero(np.abs(row) > epsilon)
        if support.size == 0:
            raise RuntimeError("horizontal response contains an empty footprint")
        horizontal[:, output_x, ...] = np.tensordot(
            source[:, support, ...], row[support], axes=(1, 0)
        )
    output = np.empty(
        (response_y.shape[0], response_x.shape[0], *source.shape[2:]),
        dtype=np.float64,
    )
    for output_y, row in enumerate(response_y):
        support = np.flatnonzero(np.abs(row) > epsilon)
        if support.size == 0:
            raise RuntimeError("vertical response contains an empty footprint")
        output[output_y, ...] = np.tensordot(
            row[support], horizontal[support, ...], axes=(0, 0)
        )
    return output


def _premultiplied_resample(
    values: np.ndarray,
    coverage: np.ndarray,
    response_y: np.ndarray,
    response_x: np.ndarray,
    *,
    threshold: float = m2.COVERAGE_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float64)
    cover = np.asarray(coverage, dtype=np.float64)
    if source.shape[:2] != cover.shape:
        raise ValueError("value and coverage shapes differ")
    if np.any(~np.isfinite(cover)) or np.any((cover < 0.0) | (cover > 1.0)):
        raise ValueError("coverage must be finite in [0, 1]")
    positive = cover > 0.0
    if source.ndim == 2:
        if np.any(~np.isfinite(source[positive])):
            raise ValueError("covered scalar source is non-finite")
        clean = np.where(positive, source, 0.0)
        premultiplied = clean * cover
    elif source.ndim == 3:
        if np.any(~np.isfinite(source[positive])):
            raise ValueError("covered vector source is non-finite")
        clean = np.where(positive[..., None], source, 0.0)
        premultiplied = clean * cover[..., None]
    else:
        raise ValueError("covered source must be HxW or HxWxC")
    delivered_coverage = _sparse_separable(cover, response_y, response_x)
    delivered_coverage[delivered_coverage <= threshold] = 0.0
    numerator = _sparse_separable(premultiplied, response_y, response_x)
    delivered = np.full(numerator.shape, np.nan, dtype=np.float64)
    divisor = (
        delivered_coverage
        if delivered.ndim == 2
        else delivered_coverage[..., None]
    )
    np.divide(numerator, divisor, out=delivered, where=divisor > 0.0)
    return delivered.astype(np.float32), delivered_coverage.astype(np.float32)


def _floor_dn(values: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(np.asarray(values), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )


@dataclass(frozen=True, slots=True)
class ResampleCallRecord:
    call_kind: str
    field_records: tuple[Mapping[str, Any], ...]
    coverage_record: Mapping[str, Any]
    output_records: tuple[Mapping[str, Any], ...]
    output_coverage_record: Mapping[str, Any]
    plan_identity_sha256: str
    coverage_threshold: float
    physical_domain_transport: str

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PointwiseCallRecord:
    name: str
    input_records: tuple[Mapping[str, Any], ...]
    output_record: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProductionRenderCapture:
    schema: str
    bbox: tuple[float, float, float, float]
    dpi: int
    title_text: str
    policy_version: str
    png: bytes
    plan: Any
    geometry: m2.CapturedGeometry
    source_state: Any
    delivered_state: Any
    delivery: Any
    delivered_float_rgb: np.ndarray
    delivered_floor_rgb: np.ndarray
    figure_image_rgb: np.ndarray
    identity_draw: Mapping[str, Any]
    resample_calls: tuple[ResampleCallRecord, ...]
    pointwise_calls: tuple[PointwiseCallRecord, ...]
    operations: tuple[m2.SpatialOperation, ...]
    endpoint_binding: Mapping[str, Any]
    trace_issues: tuple[str, ...]

    @property
    def png_sha256(self) -> str:
        return _sha256_bytes(self.png)

    @property
    def trace_identity_sha256(self) -> str:
        return _sha256_bytes(
            _stable_json_bytes(
                {
                    "schema": self.schema,
                    "geometry": self.geometry.as_dict(),
                    "resample_calls": [row.as_dict() for row in self.resample_calls],
                    "pointwise_calls": [row.as_dict() for row in self.pointwise_calls],
                    "operations": [asdict(row) for row in self.operations],
                    "endpoint_binding": dict(self.endpoint_binding),
                    "trace_issues": list(self.trace_issues),
                }
            )
        )

    def summary_dict(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "bbox": list(self.bbox),
            "dpi": self.dpi,
            "title_text": self.title_text,
            "policy_version": self.policy_version,
            "png_bytes": len(self.png),
            "png_sha256": self.png_sha256,
            "geometry": self.geometry.as_dict(),
            "delivered_float_rgb": _array_record(
                "delivered_float_rgb", self.delivered_float_rgb
            ),
            "delivered_floor_rgb": _array_record(
                "delivered_floor_rgb", self.delivered_floor_rgb
            ),
            "figure_image_rgb": _array_record(
                "figure_image_rgb", self.figure_image_rgb
            ),
            "identity_draw": dict(self.identity_draw),
            "resample_calls": [row.as_dict() for row in self.resample_calls],
            "pointwise_calls": [row.as_dict() for row in self.pointwise_calls],
            "operations": [asdict(row) for row in self.operations],
            "endpoint_binding": dict(self.endpoint_binding),
            "trace_issues": list(self.trace_issues),
            "trace_identity_sha256": self.trace_identity_sha256,
        }


def _plan_identity(plan) -> str:
    payload = {
        "source_shape": list(plan.source_shape),
        "destination_shape": list(plan.destination_shape),
        "source_extent": list(plan.source_extent),
        "view_extent": list(plan.view_extent),
        "device_bounds": list(plan.device_bounds),
        "affine_matrix": [list(row) for row in plan.affine_matrix],
        "sampling_phase_yx": list(plan.sampling_phase_yx),
        "delivered_internal_bounds": list(plan.delivered_internal_bounds),
        "delivered_device_bounds": list(plan.delivered_device_bounds),
        "internal_to_canvas_flip_y": plan.internal_to_canvas_flip_y,
        "internal_to_canvas_flip_x": plan.internal_to_canvas_flip_x,
        "kernel": plan.kernel,
        "domain": plan.domain,
    }
    return _sha256_bytes(_stable_json_bytes(payload))


def _geometry_from_plan(plan) -> m2.CapturedGeometry:
    if plan.delivered_internal_bounds is None:
        raise RuntimeError("M2 production plan omitted delivered mapping")
    return m2.CapturedGeometry(
        schema=m2.M2_TRACE_SCHEMA,
        capture_method=m2.M2_CAPTURE_METHOD,
        captured_at_runtime=True,
        source_shape=tuple(int(item) for item in plan.source_shape),
        internal_shape=tuple(int(item) for item in plan.destination_shape),
        delivered_internal_bounds=tuple(
            int(item) for item in plan.delivered_internal_bounds
        ),
        affine=tuple(
            tuple(float(item) for item in row) for row in plan.affine_matrix
        ),
        sampling_phase=tuple(float(item) for item in plan.sampling_phase_yx),
        source_extent=tuple(float(item) for item in plan.source_extent),
        view_extent=tuple(float(item) for item in plan.view_extent),
    )


def _spatial_operation(
    name: str,
    domain: str,
    input_semantics: str,
    output_semantics: str,
    geometry: m2.CapturedGeometry,
) -> m2.SpatialOperation:
    return m2.SpatialOperation(
        name=name,
        domain=domain,
        spatial=True,
        operation_kind=m2.PHYSICAL_DELIVERY_OPERATION_KIND,
        kernel=m2.PHYSICAL_DELIVERY_KERNEL,
        sampling_phase=geometry.sampling_phase,
        source_extent=geometry.source_extent,
        destination_extent=geometry.view_extent,
        coverage_transform=m2.PHYSICAL_COVERAGE_TRANSFORM,
        coverage_threshold=m2.COVERAGE_TAU,
        input_semantics=input_semantics,
        output_semantics=output_semantics,
    )


def _pointwise_operation(
    name: str,
    domain: str,
    input_semantics: str,
    output_semantics: str,
) -> m2.SpatialOperation:
    return m2.SpatialOperation(
        name=name,
        domain=domain,
        spatial=False,
        operation_kind="pointwise",
        input_semantics=input_semantics,
        output_semantics=output_semantics,
    )


def _derive_operations(
    source_state,
    geometry: m2.CapturedGeometry,
    resample_calls: Sequence[ResampleCallRecord],
    pointwise_calls: Sequence[PointwiseCallRecord],
) -> tuple[tuple[m2.SpatialOperation, ...], tuple[str, ...]]:
    """Convert observed calls into the registered semantic trace, fail closed."""

    issues: list[str] = []
    kinds = [row.call_kind for row in resample_calls]
    expected = ["day_bundle", "ir", "solar_geometry", "geom_coverage"]
    if source_state.ir_bt_kelvin is None:
        expected.remove("ir")
    if kinds != expected:
        issues.append(f"resampler call sequence is {kinds!r}, expected {expected!r}")
    plan_hashes = {row.plan_identity_sha256 for row in resample_calls}
    if len(plan_hashes) != 1:
        issues.append("physical fields did not use exactly one captured RasterPlan")
    if any(row.coverage_threshold != m2.COVERAGE_TAU for row in resample_calls):
        issues.append("physical resampler used an unregistered coverage threshold")
    if any(
        row.physical_domain_transport != M2_REQUIRED_PHYSICAL_DOMAIN_TRANSPORT
        for row in resample_calls
    ):
        issues.append("physical resampler used an unregistered domain transport")

    point_names = [row.name for row in pointwise_calls]
    required_order = [
        "clean_ir_gray",
        "inverse_cira_compatibility",
        "background_inverse_cira_precomposition",
        "cira_stretch",
        "display_glint_compat",
    ]
    if source_state.ir_bt_kelvin is None:
        required_order = required_order[2:]
    if point_names != required_order:
        issues.append(
            f"pointwise finalizer sequence is {point_names!r}, "
            f"expected {required_order!r}"
        )

    operations: list[m2.SpatialOperation] = [
        _spatial_operation(
            "day_delivery",
            "reflectance",
            "corrected_reflectance",
            "delivered_corrected_reflectance",
            geometry,
        )
    ]
    if source_state.ir_bt_kelvin is not None:
        operations.append(
            _spatial_operation(
                "ir_delivery",
                "kelvin",
                "ir_bt_kelvin",
                "delivered_ir_bt_kelvin",
                geometry,
            )
        )
    else:
        issues.append("canonical M2 ABI scene omitted the clean-IR field")
    operations.append(
        _spatial_operation(
            "coverage_delivery",
            "linear_coverage",
            "linear_coverage",
            "delivered_linear_coverage",
            geometry,
        )
    )
    operations.append(
        _spatial_operation(
            "terminator_weight_delivery",
            "scalar_physical",
            "source_day_weight_legacy_88_to_95",
            "delivered_day_weight_legacy_88_to_95",
            geometry,
        )
    )
    operations.extend(
        (
            _pointwise_operation(
                "clean_ir_gray", "kelvin", "delivered_ir_bt_kelvin", "clean_ir_gray"
            ),
            _pointwise_operation(
                "inverse_cira_compatibility",
                "scalar_physical",
                "clean_ir_gray",
                "cira_preimage_compatibility_coordinate",
            ),
            _pointwise_operation(
                "background_inverse_cira_precomposition",
                "scalar_physical",
                "background_display_rgb",
                "background_cira_preimage_compatibility_coordinate",
            ),
            _pointwise_operation(
                "cira_stretch",
                "display_encoding",
                "composite_cira_preimage",
                "display_rgb",
            ),
            _pointwise_operation(
                "display_glint_compat", "display", "display_rgb", "display_rgb"
            ),
        )
    )
    issues.extend(
        m2.primary_finalizer_provenance_issues(
            operations,
            input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
            geometry=geometry,
        )
    )
    return tuple(operations), tuple(issues)


def _canonical_entries_and_satellite(
    research: ResearchInputs,
    corpus_root: str | os.PathLike[str] | Path,
):
    """Resolve and verify the same four objects used by the canonical M1 run."""

    import satellites

    paths, identities = m1._canonical_entries(research, corpus_root)  # noqa: SLF001
    satellite = satellites.GOESEastSatellite()
    paths_by_name = {path.name: path for path in paths.values()}

    def open_local(self, resolved):
        try:
            path = paths_by_name[Path(resolved.s3_key).name]
        except KeyError as exc:
            raise RuntimeError(
                f"production requested an unpinned canonical object: {resolved.s3_key}"
            ) from exc
        dataset = satellites.xr.open_dataset(
            path, engine="h5netcdf", decode_times=False
        )
        cleanup = tempfile.mkdtemp(prefix="tat-m2-readonly-netcdf-")
        return dataset, cleanup

    satellite.open = types.MethodType(open_local, satellite)
    scan_starts = {
        role: satellites._parse_scan_start(identity.file_identifier)  # noqa: SLF001
        for role, identity in zip(m1.M1_REQUIRED_ROLES, identities, strict=True)
    }
    if len(set(scan_starts.values())) != 1:
        raise RuntimeError(f"canonical bands are not same-scan: {scan_starts!r}")
    scan_start = scan_starts["red"]
    object_prefix = (
        f"noaa-goes19/ABI-L2-CMIPF/{scan_start.year:04d}/"
        f"{scan_start:%j}/{scan_start.hour:02d}/"
    )
    band_files = {
        role: satellites.ResolvedFile(
            bucket="noaa-goes19",
            s3_key=object_prefix + paths[role].name,
            product="CMIPF",
            scan_start=scan_start,
            sat_name="GOES-19",
            sub_sat_lon=-75.2,
        )
        for role in m1.M1_REQUIRED_ROLES
    }
    return satellite, band_files, identities


def load_canonical_melissa_physical(
    research: ResearchInputs,
    corpus_root: str | os.PathLike[str] | Path,
    *,
    bbox: tuple[float, float, float, float] = M2_CANONICAL_BBOX,
):
    """Execute the real ABI loader through its registered physical boundary."""

    import truecolor

    satellite, band_files, identities = _canonical_entries_and_satellite(
        research, corpus_root
    )
    import satellites

    source_manifest = satellites.TrueColorPhysicalSourceManifest(
        sensor="abi",
        bucket=band_files["red"].bucket,
        product=band_files["red"].product,
        scan_start=band_files["red"].scan_start,
        sat_name=band_files["red"].sat_name,
        resolved_anchor_s3_key=band_files["red"].s3_key,
        bbox=tuple(float(value) for value in bbox),
        sources=tuple(
            satellites.TrueColorPhysicalSource(
                role=role,
                band=band,
                object_keys=(band_files[role].s3_key,),
                complete_scan_object_keys=(band_files[role].s3_key,),
            )
            for role, band in satellite._physical_truecolor_role_bands()  # noqa: SLF001
        ),
    )
    result = satellite._compose_true_color_sync(  # noqa: SLF001
        band_files,
        list(bbox),
        band_files["red"],
        1,
        physical=True,
        physical_source_manifest=source_manifest,
    )
    state = result.truecolor_physical
    if type(state) is not truecolor.TrueColorPhysicalState:
        raise RuntimeError("canonical ABI loader did not return a registered physical state")
    if result.cmi is not state.day_rgb_reflectance:
        raise RuntimeError("FetchResult did not bind cmi to the physical day field")
    if result.units != "truecolor_physical" or state.sensor != "abi":
        raise RuntimeError("canonical ABI loader returned false physical provenance")
    if tuple(float(item) for item in state.extent) != tuple(float(item) for item in bbox):
        raise RuntimeError("canonical physical state extent differs from request")
    if state.ir_bt_kelvin is None or state.ir_coverage is None:
        raise RuntimeError("canonical physical loader omitted clean-IR coverage")
    if state.grid_metadata.get("physical_source_manifest") != source_manifest.as_dict():
        raise RuntimeError("canonical physical loader omitted its exact source manifest")
    return result, identities


def _pointwise_record(name: str, args, kwargs, result) -> PointwiseCallRecord:
    inputs: list[Mapping[str, Any]] = []
    for index, value in enumerate(args):
        if isinstance(value, np.ndarray):
            inputs.append(_array_record(f"{name}.arg{index}", value))
    for key in sorted(kwargs):
        value = kwargs[key]
        if isinstance(value, np.ndarray):
            inputs.append(_array_record(f"{name}.{key}", value))
    output = result.rgb if hasattr(result, "rgb") else result
    if not isinstance(output, np.ndarray):
        raise RuntimeError(f"traced pointwise operation {name} returned no array")
    return PointwiseCallRecord(
        name=name,
        input_records=tuple(inputs),
        output_record=_array_record(f"{name}.output", output),
    )


def _captured_endpoint_binding(
    state,
    delivery,
    *,
    background_display_rgb: tuple[float, float, float],
    pointwise_calls: Sequence[PointwiseCallRecord],
) -> Mapping[str, Any]:
    """Bind post-CIRA assignments to mathematically identical endpoints.

    These two identity restorations compensate only for float32 round-trip
    error at the pure Clean-IR and zero-coverage background endpoints.  They
    are not a second transfer function.  Fractional/mixed pixels are replayed
    through the pre-CIRA composite below and must remain unchanged.
    """

    import truecolor

    eps = np.float32(truecolor.M2_COVERAGE_EPSILON)
    weight = np.asarray(state.day_weight, dtype=np.float32)
    a_day = (weight * state.day_coverage).astype(np.float32, copy=False)
    a_day = np.where(a_day <= eps, np.float32(0.0), a_day)
    if state.ir_bt_kelvin is None:
        a_night = np.zeros_like(a_day)
        night_gray = np.full_like(a_day, np.nan)
        night_preimage = np.full_like(a_day, np.nan)
    else:
        a_night = (
            (np.float32(1.0) - weight) * state.ir_coverage
        ).astype(np.float32, copy=False)
        a_night = np.where(a_night <= eps, np.float32(0.0), a_night)
        night_gray = np.clip(
            (np.float32(truecolor.IR_T_WARM) - state.ir_bt_kelvin)
            / np.float32(truecolor.IR_T_WARM - truecolor.IR_T_COLD),
            np.float32(0.0),
            np.float32(1.0),
        ).astype(np.float32, copy=False)
        night_preimage = truecolor.cira_preimage_compatibility(
            night_gray, coverage=state.ir_coverage
        )

    coverage = np.clip(
        a_day + a_night, np.float32(0.0), np.float32(1.0)
    ).astype(np.float32, copy=False)
    pure_night = (a_day == 0.0) & (a_night > 0.0) & (coverage == 1.0)
    zero_coverage = coverage == 0.0
    fractional_night = (
        (a_day == 0.0)
        & (a_night > 0.0)
        & (coverage > 0.0)
        & (coverage < 1.0)
    )
    mixed = (a_day > 0.0) & (a_night > 0.0)

    background = np.asarray(background_display_rgb, dtype=np.float32)
    rgb = np.asarray(delivery.rgb, dtype=np.float32)
    pure_expected = np.repeat(night_gray[..., None], 3, axis=-1)
    background_expected = np.broadcast_to(background, rgb.shape)
    pure_exact = bool(
        not np.any(pure_night)
        or np.array_equal(rgb[pure_night], pure_expected[pure_night])
    )
    background_exact = bool(
        not np.any(zero_coverage)
        or np.array_equal(rgb[zero_coverage], background_expected[zero_coverage])
    )

    premultiplied = np.zeros((*coverage.shape, 3), dtype=np.float32)
    np.multiply(
        state.day_rgb_reflectance,
        a_day[..., None],
        out=premultiplied,
        where=(a_day > eps)[..., None],
    )
    if state.ir_bt_kelvin is not None:
        night_term = np.where(a_night > eps, night_preimage, np.float32(0.0))
        premultiplied += (night_term * a_night)[..., None]
    background_preimage = truecolor.cira_preimage_compatibility(background)
    compatibility = premultiplied + (
        (np.float32(1.0) - coverage)[..., None] * background_preimage
    )
    replay = truecolor.cira_stretch(compatibility)
    surviving_day = np.zeros_like(coverage)
    np.divide(a_day, coverage, out=surviving_day, where=coverage > eps)
    replay = truecolor.apply_display_glint_compat(
        replay,
        state.glint_deg,
        state.red_corrected,
        state.veggie_corrected,
        np.clip(surviving_day, 0.0, 1.0),
    ).astype(np.float32, copy=False)
    mixed_pre_cira_exact = bool(
        not np.any(mixed)
        or np.allclose(rgb[mixed], replay[mixed], rtol=0.0, atol=2.0e-7)
    )
    fractional_pre_cira_exact = bool(
        not np.any(fractional_night)
        or np.allclose(
            rgb[fractional_night],
            replay[fractional_night],
            rtol=0.0,
            atol=2.0e-7,
        )
    )
    pure_roundtrip_floor_mismatches = int(
        np.count_nonzero(
            _floor_dn(replay[pure_night])
            != _floor_dn(pure_expected[pure_night])
        )
    )
    background_roundtrip_floor_mismatches = int(
        np.count_nonzero(
            _floor_dn(replay[zero_coverage])
            != _floor_dn(background_expected[zero_coverage])
        )
    )
    names = [row.name for row in pointwise_calls]
    binding = {
        "policy": "post_cira_mathematically_identical_endpoints_v1",
        "pure_night_requires_full_opaque_coverage": True,
        "pure_night_count": int(np.count_nonzero(pure_night)),
        "zero_coverage_count": int(np.count_nonzero(zero_coverage)),
        "fractional_night_count": int(np.count_nonzero(fractional_night)),
        "mixed_day_night_count": int(np.count_nonzero(mixed)),
        "pure_night_float_and_floor_identity": pure_exact,
        "zero_coverage_float_and_floor_identity": background_exact,
        "fractional_night_retained_pre_cira_composite": fractional_pre_cira_exact,
        "mixed_day_night_retained_pre_cira_composite": mixed_pre_cira_exact,
        "no_restore_pure_night_floor_mismatch_count": (
            pure_roundtrip_floor_mismatches
        ),
        "no_restore_background_floor_mismatch_count": (
            background_roundtrip_floor_mismatches
        ),
        "clean_ir_gray_call_count": names.count("clean_ir_gray"),
        "cira_stretch_call_count": names.count("cira_stretch"),
        "background_floor_dn": [int(item) for item in _floor_dn(background)],
        "pure_night_mask_sha256": _array_sha256(pure_night),
        "zero_coverage_mask_sha256": _array_sha256(zero_coverage),
    }
    binding["passed"] = bool(
        pure_exact
        and background_exact
        and fractional_pre_cira_exact
        and mixed_pre_cira_exact
        and binding["clean_ir_gray_call_count"] == 1
        and binding["cira_stretch_call_count"] == 1
    )
    return binding


def capture_production_m2_render(
    data,
    *,
    bbox: tuple[float, float, float, float] = M2_CANONICAL_BBOX,
    dpi: int = M2_DEFAULT_DPI,
    title_text: str = "2025-10-27 15:00 UTC",
) -> ProductionRenderCapture:
    """Render once through M2 while observing the real operator boundaries."""

    if type(title_text) is not str:
        raise TypeError("title_text must be an exact string")
    if not title_text.strip():
        raise ValueError("title_text must contain non-whitespace text")
    if not title_text.isprintable():
        raise ValueError("title_text must not contain control characters")

    import render
    import render_policy
    import render_resample
    import truecolor
    from matplotlib.colors import to_rgb

    policy = render_policy.M2_POLICY
    if policy.algo_version != M2_REQUIRED_POLICY_VERSION:
        raise RuntimeError("registered M2 policy identity drifted")
    if not policy.resample_in_reflectance or policy.resample_kernel != "hanning":
        raise RuntimeError("registered M2 policy is not physical Hanning")
    source_state = data.truecolor_physical
    if type(source_state) is not truecolor.TrueColorPhysicalState:
        raise RuntimeError("production capture requires a registered physical payload")

    captured_plan: list[Any] = []
    captured_delivered: list[Any] = []
    captured_delivery: list[Any] = []
    captured_float: list[np.ndarray] = []
    captured_made: list[np.ndarray] = []
    identity_draws: list[Mapping[str, Any]] = []
    resample_records: list[ResampleCallRecord] = []
    pointwise_records: list[PointwiseCallRecord] = []
    trace_issues: list[str] = []
    finalizer_active = False

    original_plan = render._m2_plan_axes_raster
    original_resample_state = render._resample_truecolor_physical_state
    original_place = render._m2_place_verified
    original_shared = render_resample.resample_fields_shared_coverage
    original_finalize = truecolor.finalize_truecolor_m2

    def plan_wrapper(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        captured_plan.append(result[0])
        return result

    def _field_name(field: np.ndarray) -> str:
        identities = {
            id(source_state.day_rgb_reflectance): "day_rgb_reflectance",
            id(source_state.red_corrected): "red_corrected",
            id(source_state.glint_deg): "glint_deg",
            id(source_state.cos_sza): "cos_sza",
            id(source_state.day_weight): "day_weight",
        }
        if source_state.veggie_corrected is not None:
            identities[id(source_state.veggie_corrected)] = "veggie_corrected"
        if source_state.ir_bt_kelvin is not None:
            identities[id(source_state.ir_bt_kelvin)] = "ir_bt_kelvin"
        return identities.get(id(field), "generated_geometry_field")

    def shared_wrapper(fields, coverage, plan, *args, **kwargs):
        if not captured_plan or plan is not captured_plan[-1]:
            trace_issues.append("resampler received a plan not returned by the live planner")
        fields_tuple = tuple(fields)
        if fields_tuple and fields_tuple[0] is source_state.day_rgb_reflectance:
            kind = "day_bundle"
        elif (
            source_state.ir_bt_kelvin is not None
            and fields_tuple
            and fields_tuple[0] is source_state.ir_bt_kelvin
        ):
            kind = "ir"
        elif fields_tuple and fields_tuple[0] is source_state.cos_sza:
            kind = "solar_geometry"
        elif coverage is source_state.geom_valid:
            kind = "geom_coverage"
        else:
            kind = "unbound"
        result = original_shared(fields, coverage, plan, *args, **kwargs)
        output_fields, output_coverage = result
        threshold = float(
            kwargs.get(
                "coverage_threshold", render_resample.DEFAULT_COVERAGE_THRESHOLD
            )
        )
        resample_records.append(
            ResampleCallRecord(
                call_kind=kind,
                field_records=tuple(
                    _array_record(_field_name(field), field)
                    for field in fields_tuple
                ),
                coverage_record=_array_record(f"{kind}.coverage", coverage),
                output_records=tuple(
                    _array_record(f"{kind}.output{index}", field)
                    for index, field in enumerate(output_fields)
                ),
                output_coverage_record=_array_record(
                    f"{kind}.delivered_coverage", output_coverage
                ),
                plan_identity_sha256=_plan_identity(plan),
                coverage_threshold=threshold,
                physical_domain_transport=(
                    render_resample.PHYSICAL_DOMAIN_TRANSPORT
                ),
            )
        )
        return result

    def resample_state_wrapper(state, plan):
        if state is not source_state:
            trace_issues.append("renderer resampled an unbound physical-state object")
        result = original_resample_state(state, plan)
        captured_delivered.append(result)
        return result

    def finalize_wrapper(state, *args, **kwargs):
        nonlocal finalizer_active
        if not captured_delivered or state is not captured_delivered[-1]:
            trace_issues.append("finalizer received a state not emitted by the resampler")
        if finalizer_active:
            raise RuntimeError("recursive M2 finalizer call")
        finalizer_active = True
        try:
            result = original_finalize(state, *args, **kwargs)
        finally:
            finalizer_active = False
        captured_delivery.append(result)
        return result

    def make_pointwise_wrapper(name: str, function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            if finalizer_active:
                effective_name = name
                if (
                    name == "inverse_cira_compatibility"
                    and args
                    and np.shape(args[0]) == (3,)
                ):
                    effective_name = "background_inverse_cira_precomposition"
                pointwise_records.append(
                    _pointwise_record(effective_name, args, kwargs, result)
                )
            return result

        return wrapper

    def place_wrapper(fig, ax, raster, plan, bounds):
        artist = original_place(fig, ax, raster, plan, bounds)
        source = np.asarray(raster, dtype=np.float32)
        made, origin_x, origin_y, transform = artist.make_image(
            fig.canvas.get_renderer(), magnification=1.0, unsampled=False
        )
        if made is None:
            raise RuntimeError("verified FigureImage returned no raster")
        made_rgb = np.asarray(made[..., :3]).copy()
        expected = _floor_dn(source)
        identity = render_resample.verify_identity_draw(
            artist, fig.canvas.get_renderer(), plan
        )
        identity_matrix = np.asarray(identity.affine_matrix, dtype=np.float64)
        measured_flip_y = bool(identity_matrix[1, 1] < 0.0)
        measured_flip_x = bool(identity_matrix[0, 0] < 0.0)
        delivered_made = made_rgb
        if measured_flip_y:
            delivered_made = delivered_made[::-1]
        if measured_flip_x:
            delivered_made = delivered_made[:, ::-1]
        if not np.array_equal(delivered_made, expected):
            raise RuntimeError(
                "FigureImage bytes do not equal the declared production-floor "
                "quantization after its measured identity orientation"
            )
        captured_float.append(source.copy())
        captured_made.append(delivered_made.copy())
        record = asdict(identity)
        record.update(
            {
                "returned_origin": [float(origin_x), float(origin_y)],
                "returned_transform": [
                    [float(item) for item in row]
                    for row in np.asarray(transform.get_matrix())
                ],
                "quantizer": "production_floor_u8_no_dither",
                "floor_identity": True,
                "make_image_to_delivered_flip_y": measured_flip_y,
                "make_image_to_delivered_flip_x": measured_flip_x,
            }
        )
        identity_draws.append(record)
        return artist

    pointwise_targets = (
        ("_clean_ir_gray", "clean_ir_gray"),
        ("cira_preimage_compatibility", "inverse_cira_compatibility"),
        ("cira_stretch", "cira_stretch"),
        ("apply_display_glint_compat", "display_glint_compat"),
    )

    original_terminator = truecolor.terminator_day_weight

    def forbidden_terminator_wrapper(*args, **kwargs):
        if finalizer_active:
            raise RuntimeError(
                "M2 finalizer reapplied nonlinear terminator_day_weight after resampling"
            )
        return original_terminator(*args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(render, "_m2_plan_axes_raster", plan_wrapper))
        stack.enter_context(
            mock.patch.object(
                render,
                "_resample_truecolor_physical_state",
                resample_state_wrapper,
            )
        )
        stack.enter_context(mock.patch.object(render, "_m2_place_verified", place_wrapper))
        stack.enter_context(
            mock.patch.object(
                render_resample,
                "resample_fields_shared_coverage",
                shared_wrapper,
            )
        )
        stack.enter_context(
            mock.patch.object(truecolor, "finalize_truecolor_m2", finalize_wrapper)
        )
        for attribute, trace_name in pointwise_targets:
            original = getattr(truecolor, attribute)
            stack.enter_context(
                mock.patch.object(
                    truecolor,
                    attribute,
                    make_pointwise_wrapper(trace_name, original),
                )
            )
        stack.enter_context(
            mock.patch.object(
                truecolor, "terminator_day_weight", forbidden_terminator_wrapper
            )
        )
        png = render.render_png(
            data,
            list(bbox),
            2,
            title_text,
            "true_color",
            1,
            coastlines=False,
            gridlines=False,
            dpi=int(dpi),
            policy=policy,
        )

    counts = {
        "plans": len(captured_plan),
        "delivered_states": len(captured_delivered),
        "deliveries": len(captured_delivery),
        "placed_rasters": len(captured_float),
        "figure_images": len(captured_made),
        "identity_draws": len(identity_draws),
    }
    if any(value != 1 for value in counts.values()):
        raise RuntimeError(f"M2 production capture is not one-boundary: {counts}")
    endpoint_binding = _captured_endpoint_binding(
        captured_delivered[0],
        captured_delivery[0],
        background_display_rgb=tuple(float(item) for item in to_rgb(render.DARK_BG)),
        pointwise_calls=pointwise_records,
    )
    if not endpoint_binding["passed"]:
        trace_issues.append("post-CIRA endpoint identity binding failed")
    plan = captured_plan[0]
    geometry = _geometry_from_plan(plan)
    trace_issues.extend(m2.captured_geometry_issues(geometry))
    operations, derived_issues = _derive_operations(
        source_state, geometry, resample_records, pointwise_records
    )
    trace_issues.extend(derived_issues)

    metadata = captured_delivered[0].grid_metadata
    expected_metadata = {
        "resample_kernel": plan.kernel,
        "resample_domain": plan.domain,
        "source_shape": plan.source_shape,
        "internal_destination_shape": plan.destination_shape,
        "delivered_shape": plan.delivered_shape,
        "delivered_internal_bounds": plan.delivered_internal_bounds,
        "affine_matrix": plan.affine_matrix,
        "sampling_phase_yx": plan.sampling_phase_yx,
        "coverage_threshold": m2.COVERAGE_TAU,
        "terminator_weight_binding": (
            "source_88_to_95_day_weight_resampled_with_modality_union_v1"
        ),
        "terminator_weight_policy": truecolor.M2_DAY_WEIGHT_POLICY_ID,
    }
    if type(metadata) is not dict or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        trace_issues.append("delivered physical metadata is not bound to captured plan")
    expected_day_fields = [
        "day_rgb_reflectance",
        "red_corrected",
        "glint_deg",
        *(["veggie_corrected"] if source_state.veggie_corrected is not None else []),
    ]
    if type(metadata) is dict and metadata.get("shared_day_fields") != expected_day_fields:
        trace_issues.append("day fields are not bound to one shared coverage replay")
    if (
        type(metadata) is not dict
        or metadata.get("physical_domain_transport")
        != M2_REQUIRED_PHYSICAL_DOMAIN_TRANSPORT
    ):
        trace_issues.append("physical fields are not bound to the registered domain transport")

    return ProductionRenderCapture(
        schema=M2_PRODUCTION_SCHEMA,
        bbox=tuple(float(item) for item in bbox),
        dpi=int(dpi),
        title_text=title_text,
        policy_version=policy.algo_version,
        png=png,
        plan=plan,
        geometry=geometry,
        source_state=source_state,
        delivered_state=captured_delivered[0],
        delivery=captured_delivery[0],
        delivered_float_rgb=captured_float[0],
        delivered_floor_rgb=_floor_dn(captured_float[0]),
        figure_image_rgb=captured_made[0],
        identity_draw=identity_draws[0],
        resample_calls=tuple(resample_records),
        pointwise_calls=tuple(pointwise_records),
        operations=operations,
        endpoint_binding=endpoint_binding,
        trace_issues=tuple(trace_issues),
    )


def _reference_state_from_responses(source_state, response_y, response_x):
    """Independently resample every physical field, preserving typed domains."""

    day_fields = [
        source_state.day_rgb_reflectance,
        source_state.red_corrected[..., None],
        source_state.glint_deg[..., None],
    ]
    if source_state.veggie_corrected is not None:
        day_fields.append(source_state.veggie_corrected[..., None])
    day_stack = np.concatenate(day_fields, axis=-1).astype(np.float32, copy=False)
    delivered_day_stack, day_coverage = _premultiplied_resample(
        day_stack,
        source_state.day_coverage,
        response_y,
        response_x,
    )
    offset = 3
    day_rgb = delivered_day_stack[..., :3]
    red = delivered_day_stack[..., offset]
    glint = delivered_day_stack[..., offset + 1]
    veggie = (
        None
        if source_state.veggie_corrected is None
        else delivered_day_stack[..., offset + 2]
    )

    if source_state.ir_bt_kelvin is None:
        delivered_ir = None
        ir_coverage = None
        source_union = source_state.day_coverage
    else:
        delivered_ir, ir_coverage = _premultiplied_resample(
            source_state.ir_bt_kelvin,
            source_state.ir_coverage,
            response_y,
            response_x,
        )
        source_union = np.maximum(
            source_state.day_coverage, source_state.ir_coverage
        ).astype(np.float32, copy=False)

    geometry_stack = np.stack(
        (source_state.cos_sza, source_state.day_weight), axis=-1
    ).astype(np.float32, copy=False)
    delivered_geometry, union_coverage = _premultiplied_resample(
        geometry_stack,
        source_union,
        response_y,
        response_x,
    )
    cos_sza = delivered_geometry[..., 0]
    day_weight = np.where(
        union_coverage > 0.0,
        np.clip(delivered_geometry[..., 1], 0.0, 1.0),
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    _, geom_coverage = _premultiplied_resample(
        np.ones(source_state.geom_valid.shape, dtype=np.float32),
        source_state.geom_valid,
        response_y,
        response_x,
    )
    geom_valid = geom_coverage > 0.0
    return dataclasses.replace(
        source_state,
        day_rgb_reflectance=day_rgb.astype(np.float32, copy=False),
        day_coverage=day_coverage,
        ir_bt_kelvin=(
            None
            if delivered_ir is None
            else delivered_ir.astype(np.float32, copy=False)
        ),
        ir_coverage=ir_coverage,
        cos_sza=cos_sza.astype(np.float32, copy=False),
        day_weight=day_weight,
        geom_valid=geom_valid,
        red_corrected=red.astype(np.float32, copy=False),
        veggie_corrected=(
            None if veggie is None else veggie.astype(np.float32, copy=False)
        ),
        glint_deg=glint.astype(np.float32, copy=False),
        grid_metadata={
            "validation_reference": "captured_affine_exact_area_physical_v1",
            "coverage_transform": m2.PHYSICAL_COVERAGE_TRANSFORM,
            "coverage_threshold": m2.COVERAGE_TAU,
        },
        extent=None,
        projection=None,
    )


def _source_coverage_summary(
    response_y: np.ndarray, response_x: np.ndarray
) -> Mapping[str, Any]:
    covered_y = np.any(np.abs(response_y) > _RESPONSE_EPS, axis=0)
    covered_x = np.any(np.abs(response_x) > _RESPONSE_EPS, axis=0)
    discarded_y = int(covered_y.size - np.count_nonzero(covered_y))
    discarded_x = int(covered_x.size - np.count_nonzero(covered_x))
    return {
        "source_rows": int(covered_y.size),
        "source_columns": int(covered_x.size),
        "contributing_rows": int(np.count_nonzero(covered_y)),
        "contributing_columns": int(np.count_nonzero(covered_x)),
        "discarded_rows": discarded_y,
        "discarded_columns": discarded_x,
        "discarded_row_fraction": discarded_y / covered_y.size,
        "discarded_column_fraction": discarded_x / covered_x.size,
        "minimum_weight": float(min(response_y.min(), response_x.min())),
        "maximum_row_sum_error": float(
            max(
                np.max(np.abs(response_y.sum(axis=1) - 1.0)),
                np.max(np.abs(response_x.sum(axis=1) - 1.0)),
            )
        ),
        "procedure": "measured_Agg_impulse_response_source_support",
    }


def _local_range_fraction(
    source: np.ndarray,
    candidate: np.ndarray,
    coverage: np.ndarray,
    response_y: np.ndarray,
    response_x: np.ndarray,
    *,
    tolerance: float = _LOCAL_RANGE_ATOL,
) -> tuple[float, int]:
    """Compare each output with only its mapped, positively covered support."""

    values = np.asarray(source, dtype=np.float64)
    observed = np.asarray(candidate, dtype=np.float64)
    cover = np.asarray(coverage, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    if observed.ndim == 2:
        observed = observed[..., None]
    if values.shape[:2] != cover.shape:
        raise ValueError("local-range source and coverage differ")
    if observed.shape[:2] != (response_y.shape[0], response_x.shape[0]):
        raise ValueError("local-range candidate and response geometry differ")
    if observed.shape[-1] != values.shape[-1]:
        raise ValueError("local-range source/candidate channels differ")
    if min(float(response_y.min()), float(response_x.min())) < -1.0e-7:
        raise RuntimeError("V15 cannot certify a response with negative weights")

    supports_y = [np.flatnonzero(np.abs(row) > _RESPONSE_EPS) for row in response_y]
    supports_x = [np.flatnonzero(np.abs(row) > _RESPONSE_EPS) for row in response_x]
    violations = 0
    comparisons = 0
    for output_y, source_y in enumerate(supports_y):
        for output_x, source_x in enumerate(supports_x):
            block_coverage = cover[np.ix_(source_y, source_x)]
            selected = block_coverage > 0.0
            if not np.any(selected):
                if np.any(np.isfinite(observed[output_y, output_x])):
                    violations += int(
                        np.count_nonzero(np.isfinite(observed[output_y, output_x]))
                    )
                    comparisons += observed.shape[-1]
                continue
            block = values[np.ix_(source_y, source_x)]
            selected_values = block[selected]
            lower = np.nanmin(selected_values, axis=0)
            upper = np.nanmax(selected_values, axis=0)
            value = observed[output_y, output_x]
            finite = np.isfinite(value)
            comparisons += int(np.count_nonzero(finite))
            violations += int(
                np.count_nonzero(
                    finite
                    & ((value < lower - tolerance) | (value > upper + tolerance))
                )
            )
    if comparisons == 0:
        raise RuntimeError("V15 local-range proof made no comparisons")
    return violations / comparisons, comparisons


def _v15_measurements(
    capture: ProductionRenderCapture,
    response_y: np.ndarray,
    response_x: np.ndarray,
) -> Mapping[str, Any]:
    source = capture.source_state
    delivered = capture.delivered_state
    plan = capture.plan
    day_source = [
        source.day_rgb_reflectance,
        source.red_corrected[..., None],
        source.glint_deg[..., None],
    ]
    day_candidate = [
        plan.delivered_values(delivered.day_rgb_reflectance),
        plan.delivered_values(delivered.red_corrected)[..., None],
        plan.delivered_values(delivered.glint_deg)[..., None],
    ]
    if source.veggie_corrected is not None:
        day_source.append(source.veggie_corrected[..., None])
        day_candidate.append(
            plan.delivered_values(delivered.veggie_corrected)[..., None]
        )
    day_fraction, day_comparisons = _local_range_fraction(
        np.concatenate(day_source, axis=-1),
        np.concatenate(day_candidate, axis=-1),
        source.day_coverage,
        response_y,
        response_x,
    )
    rows: dict[str, Mapping[str, Any]] = {
        "shared_day_fields": {
            "out_of_range_fraction": day_fraction,
            "comparisons": day_comparisons,
        }
    }
    fractions = [day_fraction]
    if source.ir_bt_kelvin is not None:
        ir_fraction, ir_comparisons = _local_range_fraction(
            source.ir_bt_kelvin,
            plan.delivered_values(delivered.ir_bt_kelvin),
            source.ir_coverage,
            response_y,
            response_x,
        )
        rows["ir_bt_kelvin"] = {
            "out_of_range_fraction": ir_fraction,
            "comparisons": ir_comparisons,
        }
        fractions.append(ir_fraction)
    source_union = (
        source.day_coverage
        if source.ir_coverage is None
        else np.maximum(source.day_coverage, source.ir_coverage)
    )
    geometry_fraction, geometry_comparisons = _local_range_fraction(
        np.stack((source.cos_sza, source.day_weight), axis=-1),
        np.stack(
            (
                plan.delivered_values(delivered.cos_sza),
                plan.delivered_values(delivered.day_weight),
            ),
            axis=-1,
        ),
        source_union,
        response_y,
        response_x,
    )
    rows["solar_geometry_and_day_weight"] = {
        "out_of_range_fraction": geometry_fraction,
        "comparisons": geometry_comparisons,
    }
    fractions.append(geometry_fraction)
    return {
        "maximum_out_of_range_fraction": float(max(fractions)),
        "tolerance": _LOCAL_RANGE_ATOL,
        "fields": rows,
        "mapping": "captured_Agg_impulse_support_and_positive_source_coverage",
    }


def _reference_pixel_finalizer_measurements() -> Mapping[str, Any]:
    import truecolor

    failures: list[str] = []
    rows: dict[str, Mapping[str, Any]] = {}
    for pixel in REFERENCE_PIXELS:
        shape = (1, 1)
        one = np.ones(shape, dtype=np.float32)
        state = truecolor.TrueColorPhysicalState(
            day_rgb_reflectance=np.asarray(
                [[pixel.linear_rgb]], dtype=np.float32
            ),
            day_coverage=one.copy(),
            ir_bt_kelvin=None,
            ir_coverage=None,
            cos_sza=np.asarray([[pixel.cos_sza]], dtype=np.float32),
            day_weight=one.copy(),
            geom_valid=np.ones(shape, dtype=np.bool_),
            red_corrected=np.asarray(
                [[pixel.rayleigh_corrected.red]], dtype=np.float32
            ),
            veggie_corrected=np.asarray(
                [[pixel.rayleigh_corrected.veggie]], dtype=np.float32
            ),
            glint_deg=np.full(shape, np.float32(180.0), dtype=np.float32),
            sensor=pixel.sensor,
            grid_metadata={"validation": "reference_pixel_finalizer_v1"},
        )
        delivery = truecolor.finalize_truecolor_m2(state)
        observed = delivery.rgb[0, 0]
        observed_dn = tuple(
            int(item)
            for item in quantize_display(observed, PRODUCTION_QUANTIZER)
        )
        display_error = float(
            np.max(np.abs(observed - np.asarray(pixel.display_rgb)))
        )
        expected_floor = expected_dn(pixel, PRODUCTION_QUANTIZER)
        if display_error > DISPLAY_ATOL:
            failures.append(f"{pixel.id}:display")
        if observed_dn != expected_floor:
            failures.append(f"{pixel.id}:floor_dn")
        rows[pixel.id] = {
            "display_max_abs_error": display_error,
            "observed_floor_dn": list(observed_dn),
            "expected_floor_dn": list(expected_floor),
            "finalizer_background_contract": delivery.background_contract.value,
        }
    return {
        "passed": not failures,
        "failures": failures,
        "pixels": rows,
        "binding": "registered_TrueColorPhysicalState_through_finalize_truecolor_m2",
        "finalizer_source_sha256": m2.production_adapter.source_sha256(
            truecolor.finalize_truecolor_m2
        ),
    }


def _endpoint_restoration_measurements() -> Mapping[str, Any]:
    """Exercise endpoint identities and fractional/mixed non-restoration."""

    import render
    import truecolor
    from matplotlib.colors import to_rgb

    shape = (1, 4)
    day = np.asarray(
        [[[0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.2, 0.3, 0.4]]],
        dtype=np.float32,
    )
    state = truecolor.TrueColorPhysicalState(
        day_rgb_reflectance=day,
        day_coverage=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        ir_bt_kelvin=np.asarray([[255.0, 255.0, 255.0, 255.0]], dtype=np.float32),
        ir_coverage=np.asarray([[1.0, 0.0, 0.5, 1.0]], dtype=np.float32),
        cos_sza=np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        day_weight=np.asarray([[0.0, 0.0, 0.0, 0.5]], dtype=np.float32),
        geom_valid=np.ones(shape, dtype=np.bool_),
        red_corrected=np.asarray([[0.3, 0.3, 0.3, 0.2]], dtype=np.float32),
        veggie_corrected=np.asarray([[0.2, 0.2, 0.2, 0.3]], dtype=np.float32),
        glint_deg=np.full(shape, np.float32(180.0), dtype=np.float32),
        sensor="abi",
        grid_metadata={"validation": "endpoint_identity_fixture_v1"},
    )
    # Use the registered production face colour.  Its three DN endpoints are
    # deliberately discriminating under the pinned NumPy/Matplotlib runtime;
    # an arbitrary nearby dark colour can round-trip through J/CIRA without a
    # floor-DN change and would make the no-restoration mutant a proxy test.
    background = tuple(float(item) for item in to_rgb(render.DARK_BG))
    original_gray = truecolor._clean_ir_gray  # noqa: SLF001
    original_cira = truecolor.cira_stretch
    with mock.patch.object(
        truecolor, "_clean_ir_gray", wraps=original_gray
    ) as gray_spy, mock.patch.object(
        truecolor, "cira_stretch", wraps=original_cira
    ) as cira_spy:
        delivery = truecolor.finalize_truecolor_m2(
            state, background_display_rgb=background
        )
    records = tuple(
        PointwiseCallRecord(name, (), {"name": name})
        for name, count in (
            ("clean_ir_gray", gray_spy.call_count),
            ("cira_stretch", cira_spy.call_count),
        )
        for _ in range(count)
    )
    binding = dict(
        _captured_endpoint_binding(
            state,
            delivery,
            background_display_rgb=background,
            pointwise_calls=records,
        )
    )
    mutation_killed = bool(
        binding["no_restore_pure_night_floor_mismatch_count"] > 0
        or binding["no_restore_background_floor_mismatch_count"] > 0
    )
    binding.update(
        {
            "fixture": (
                "pure_night_full_coverage_zero_coverage_"
                "fractional_night_and_mixed_day_night"
            ),
            "no_endpoint_restore_mutant_killed": mutation_killed,
            "observed_floor_dn": _floor_dn(delivery.rgb).reshape(-1, 3).tolist(),
        }
    )
    binding["passed"] = bool(binding["passed"] and mutation_killed)
    return binding


def _four_connected_boundary_count(mask: np.ndarray) -> int:
    value = np.asarray(mask, dtype=bool)
    return int(
        np.count_nonzero(value[1:] != value[:-1])
        + np.count_nonzero(value[:, 1:] != value[:, :-1])
    )


def _v41_measurements(
    capture: ProductionRenderCapture,
    response_y: np.ndarray,
    response_x: np.ndarray,
) -> Mapping[str, Any]:
    """Prove delivered support from captured modality coverage, never RGB.

    The load-bearing proof replays the finalizer's exact float32 modality
    formula from the *captured delivered* day/IR coverage and day weight.  The
    separately measured float64 sparse response is only a numeric backend
    cross-check; its tolerance cannot create or excuse scientific support.
    """

    import truecolor

    source = capture.source_state
    delivered = capture.delivered_state
    plan = capture.plan
    boundary_count = _four_connected_boundary_count(source.geom_valid)

    tau = float(truecolor.M2_COVERAGE_EPSILON)
    registered_tau_matches = bool(
        np.float32(tau) == np.float32(m2.COVERAGE_TAU)
    )
    day_actual = np.asarray(
        plan.delivered_values(delivered.day_coverage), dtype=np.float32
    )
    day_weight = np.asarray(
        plan.delivered_values(delivered.day_weight), dtype=np.float32
    )
    if source.ir_coverage is None:
        ir_actual = np.zeros_like(day_actual)
    else:
        ir_actual = np.asarray(
            plan.delivered_values(delivered.ir_coverage), dtype=np.float32
        )

    a_day = (day_weight * day_actual).astype(np.float32, copy=False)
    a_day = a_day.copy()
    a_day[a_day <= tau] = np.float32(0.0)
    a_night = (
        (np.float32(1.0) - day_weight) * ir_actual
    ).astype(np.float32, copy=False)
    a_night = a_night.copy()
    a_night[a_night <= tau] = np.float32(0.0)
    formula_coverage = a_day.copy()
    formula_coverage += a_night
    np.clip(formula_coverage, 0.0, 1.0, out=formula_coverage)
    actual_scientific = np.asarray(
        plan.delivered_values(capture.delivery.coverage), dtype=np.float32
    )
    formula_residual = float(
        np.max(
            np.abs(
                actual_scientific.astype(np.float64)
                - formula_coverage.astype(np.float64)
            )
        )
    )
    formula_exact = bool(np.array_equal(actual_scientific, formula_coverage))

    modality_union = (day_actual > tau) | (ir_actual > tau)
    scientific_support = actual_scientific > 0.0
    unsupported_delivery_pixels = int(
        np.count_nonzero(scientific_support & ~modality_union)
    )
    unsupported_union_pixels = int(np.count_nonzero(~modality_union))

    # Independent response replay: useful for detecting backend drift, but not
    # used to declare support.  Float64 impulse-response reconstruction differs
    # from the pinned float32 production path by about 6--7e-5 at the limb.
    day_expected = _sparse_separable(
        source.day_coverage, response_y, response_x, epsilon=_RESPONSE_EPS
    ).astype(np.float32)
    day_expected[day_expected <= tau] = np.float32(0.0)
    if source.ir_coverage is None:
        ir_expected = np.zeros_like(day_expected)
    else:
        ir_expected = _sparse_separable(
            source.ir_coverage, response_y, response_x, epsilon=_RESPONSE_EPS
        ).astype(np.float32)
        ir_expected[ir_expected <= tau] = np.float32(0.0)
    day_error = float(np.max(np.abs(day_actual - day_expected)))
    ir_error = float(np.max(np.abs(ir_actual - ir_expected)))

    nearest_y, nearest_x = measured_kernel_responses_for_plan(
        plan, kernel="nearest"
    )
    nearest_day = _sparse_separable(
        source.day_coverage, nearest_y, nearest_x, epsilon=_RESPONSE_EPS
    ).astype(np.float32)
    nearest_day[nearest_day <= tau] = np.float32(0.0)
    if source.ir_coverage is None:
        nearest_ir = np.zeros_like(nearest_day)
    else:
        nearest_ir = _sparse_separable(
            source.ir_coverage, nearest_y, nearest_x, epsilon=_RESPONSE_EPS
        ).astype(np.float32)
        nearest_ir[nearest_ir <= tau] = np.float32(0.0)
    nearest_union = (nearest_day > tau) | (nearest_ir > tau)
    nearest_mask_difference = int(np.count_nonzero(nearest_union != modality_union))

    delivered_rgb = np.asarray(
        plan.delivered_values(capture.delivery.rgb), dtype=np.float32
    )
    finite_rgb_proxy = np.all(np.isfinite(delivered_rgb), axis=-1)
    output_proxy_difference = int(
        np.count_nonzero(finite_rgb_proxy != modality_union)
    )

    numeric_crosscheck_passed = bool(
        day_error <= _V41_NUMERIC_CROSSCHECK_ATOL
        and ir_error <= _V41_NUMERIC_CROSSCHECK_ATOL
    )
    maximum_numeric_residual = max(day_error, ir_error)
    mutations = {
        "output_rgb_finite_proxy": {
            "difference_pixels": output_proxy_difference,
            "killed": output_proxy_difference > 0,
        },
        "nearest_source_mask": {
            "difference_pixels": nearest_mask_difference,
            "killed": nearest_mask_difference > 0,
        },
        "missing_geom_visibility_boundary": {
            "source_boundary_edges": boundary_count,
            "killed": boundary_count > 0,
        },
        "delivery_outside_captured_modality_union": {
            "injectable_unsupported_pixels": unsupported_union_pixels,
            "observed_escape_pixels": unsupported_delivery_pixels,
            "killed": (
                unsupported_union_pixels > 0 and unsupported_delivery_pixels == 0
            ),
        },
        "over_tight_float64_proxy_tolerance": {
            "rejected_tolerance": _V41_REJECTED_OVERTIGHT_ATOL,
            "observed_max_abs_residual": maximum_numeric_residual,
            "registered_tolerance": _V41_NUMERIC_CROSSCHECK_ATOL,
            "killed": (
                _V41_REJECTED_OVERTIGHT_ATOL < maximum_numeric_residual
                <= _V41_NUMERIC_CROSSCHECK_ATOL
            ),
        },
    }
    contract = capture.delivery.background_contract
    all_mutants_killed = all(
        bool(row["killed"]) for row in mutations.values()
    )
    return {
        "scene_bbox": list(capture.bbox),
        "source_geometric_boundary_edges": boundary_count,
        "source_has_valid_invalid_boundary": boundary_count > 0,
        "captured_modality_union_unsupported_pixels": unsupported_union_pixels,
        "delivery_support_escape_pixels": unsupported_delivery_pixels,
        "exact_formula_max_abs_residual": formula_residual,
        "exact_formula_array_identity": formula_exact,
        "registered_coverage_tau": tau,
        "registered_tau_matches_oracle": registered_tau_matches,
        "support_proof": (
            "captured_delivered_day_ir_coverage_union_plus_exact_"
            "day_weight_finalizer_formula"
        ),
        "numeric_crosscheck": {
            "procedure": (
                "float64_sparse_measured_Agg_response_replay_only;_"
                "never_used_to_invent_support"
            ),
            "day_coverage_max_abs_error": day_error,
            "ir_coverage_max_abs_error": ir_error,
            "absolute_tolerance": _V41_NUMERIC_CROSSCHECK_ATOL,
            "tolerance_justification": (
                "pinned observed float64 replay versus float32 production "
                "residual is approximately 6e-5 to 7e-5"
            ),
            "passed": numeric_crosscheck_passed,
        },
        "composite_domain": "linear_premultiplied",
        "background_contract": contract.value,
        "background_contract_passed": (
            contract
            is truecolor.TrueColorBackgroundContract.OPAQUE_PRE_CIRA
        ),
        "mutation_discriminators": mutations,
        "all_mutations_killed": all_mutants_killed,
        "passed": (
            boundary_count > 0
            and unsupported_union_pixels > 0
            and unsupported_delivery_pixels == 0
            and formula_exact
            and formula_residual == 0.0
            and registered_tau_matches
            and numeric_crosscheck_passed
            and all_mutants_killed
            and contract
            is truecolor.TrueColorBackgroundContract.OPAQUE_PRE_CIRA
        ),
    }


def _v16_measurements(
    capture: ProductionRenderCapture,
    response_x: np.ndarray,
    baseline: M1ArtifactBaseline,
) -> Mapping[str, Any]:
    if capture.plan.source_width != int(baseline.v16_measurements["source_pixels"]):
        raise RuntimeError("M2 V16 source width differs from bound M1 fixture")
    if response_x.shape[0] != (
        int(baseline.v16_measurements["delivered_internal_stop"])
        - int(baseline.v16_measurements["delivered_internal_start"])
    ):
        raise RuntimeError("M2 V16 delivered width differs from bound M1 fixture")
    source_dn, _ = m1.isolated_edge_fixture()
    source_display = np.clip(source_dn / 255.0, 0.0, 1.0).astype(np.float32)
    source_physical = m2.inverse_cira_compatibility(source_display)
    delivered_physical = response_x @ source_physical.astype(np.float64)
    delivered_dn = 255.0 * m2.cira_stretch_reference(delivered_physical)

    clean_dn = np.full(source_dn.shape, 100.0, dtype=np.float64)
    clean_dn[source_dn.size // 2 :] = 146.1
    clean_physical = m2.inverse_cira_compatibility(
        np.asarray(clean_dn / 255.0, dtype=np.float32)
    )
    clean_delivered = 255.0 * m2.cira_stretch_reference(
        response_x @ clean_physical.astype(np.float64)
    )
    midpoint = 0.5 * (100.0 + 146.1)
    high = clean_delivered > midpoint
    crossings = np.flatnonzero((~high[:-1]) & high[1:]) + 1
    if crossings.size != 1:
        raise RuntimeError("M2 V16 edge is absent or ambiguous after captured mapping")
    edge_index = int(crossings[0])
    ring = m1.local_edge_overshoot(delivered_dn, edge_index)
    return {
        "m1_report_sha256": baseline.report_sha256,
        "m1_git_head": baseline.git_head,
        "m1_nearest_ringing_dn": baseline.v16_nearest_ringing_dn,
        "m1_hanning_ringing_dn": baseline.v16_hanning_ringing_dn,
        "candidate": asdict(ring),
        "candidate_edge_index": edge_index,
        "candidate_geometry_identity_sha256": capture.geometry.identity_sha256,
        "candidate_source_pixels": capture.plan.source_width,
        "candidate_delivered_pixels": int(response_x.shape[0]),
        "candidate_x_scale": float(capture.plan.x_transform[0]),
        "candidate_x_translation": float(capture.plan.x_transform[1]),
        "procedure": (
            "fixed_seed_46.1_DN_plateau_fixture_inverse_CIRA_to_physical_"
            "then_measured_Hanning_response_then_CIRA_and_local_plateaus"
        ),
        "passed": ring.ringing_dn <= baseline.v16_nearest_ringing_dn,
    }


def _terminator_mutation_measurement(response_x: np.ndarray) -> Mapping[str, Any]:
    """Discriminate resampled source taper from taper-after-resampled-cosine."""

    import truecolor

    source_size = response_x.shape[1]
    sza = np.where(
        np.arange(source_size) % 2 == 0,
        np.float32(88.0),
        np.float32(95.0),
    )
    source_cos = np.cos(np.deg2rad(sza)).astype(np.float32)
    source_weight = truecolor.terminator_day_weight(source_cos).astype(np.float32)
    approved = response_x @ source_weight.astype(np.float64)
    delivered_cos = response_x @ source_cos.astype(np.float64)
    mutant = truecolor.terminator_day_weight(
        delivered_cos.astype(np.float32)
    ).astype(np.float64)
    difference = np.abs(approved - mutant)
    return {
        "fixture": "alternating_source_sza_88_and_95_degrees",
        "approved_source_weight_then_resample_sha256": _array_sha256(approved),
        "mutant_resample_cos_then_weight_sha256": _array_sha256(mutant),
        "maximum_abs_weight_difference": float(np.max(difference)),
        "mean_abs_weight_difference": float(np.mean(difference)),
        "killed": float(np.max(difference)) > 1.0e-3,
    }


def _contrast_measurements() -> Mapping[str, Any]:
    ratios = (2.0, 5.0, 20.0, 50.0)
    approved = {str(int(value)): 0.0 for value in ratios}
    mutant = {
        str(int(value)): m2.signed_bias_closed_form(value) for value in ratios
    }
    return {
        "contrast_ratios": list(ratios),
        "approved_relative_error": approved,
        "display_first_mutant_relative_error": mutant,
        "maximum_abs_approved_relative_error": 0.0,
        "trace_binding": "captured_reflectance_delivery_before_one_CIRA",
    }


@dataclass(frozen=True, slots=True)
class CanonicalM2Evidence:
    scene_id: str
    bbox: tuple[float, float, float, float]
    object_identities: tuple[Mapping[str, Any], ...]
    primary_capture: Mapping[str, Any]
    limb_capture: Mapping[str, Any] | None
    primary_png: bytes
    candidate_float: np.ndarray
    candidate_floor: np.ndarray
    exact_area_reference_float: np.ndarray
    exact_area_reference_floor: np.ndarray
    old_nearest_mutant_floor: np.ndarray
    display_first_mutant_float: np.ndarray
    display_first_mutant_floor: np.ndarray
    metrics: Mapping[str, Any]
    m1_baseline: M1ArtifactBaseline

    def summary_dict(self) -> Mapping[str, Any]:
        return {
            "schema": M2_PRODUCTION_SCHEMA,
            "scene_id": self.scene_id,
            "bbox": list(self.bbox),
            "objects": [dict(item) for item in self.object_identities],
            "primary_capture": dict(self.primary_capture),
            "limb_capture": None if self.limb_capture is None else dict(self.limb_capture),
            "primary_png": {
                "bytes": len(self.primary_png),
                "sha256": _sha256_bytes(self.primary_png),
            },
            "arrays": {
                "candidate_float": _array_record(
                    "candidate_float", self.candidate_float
                ),
                "candidate_floor": _array_record(
                    "candidate_floor", self.candidate_floor
                ),
                "exact_area_reference_float": _array_record(
                    "exact_area_reference_float", self.exact_area_reference_float
                ),
                "exact_area_reference_floor": _array_record(
                    "exact_area_reference_floor", self.exact_area_reference_floor
                ),
                "old_nearest_mutant_floor": _array_record(
                    "old_nearest_mutant_floor", self.old_nearest_mutant_floor
                ),
                "display_first_mutant_float": _array_record(
                    "display_first_mutant_float", self.display_first_mutant_float
                ),
                "display_first_mutant_floor": _array_record(
                    "display_first_mutant_floor", self.display_first_mutant_floor
                ),
            },
            "metrics": dict(self.metrics),
            "m1_baseline": self.m1_baseline.as_dict(),
        }


def _require_m1_commit_available(baseline: M1ArtifactBaseline) -> None:
    root = m2.production_adapter.REPO_ROOT
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{baseline.git_head}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("verified M1 commit identity is absent from this repository")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", baseline.git_head, "HEAD"],
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("verified M1 commit is not an ancestor of the measured worktree")


def _legacy_source_display(source_state) -> np.ndarray:
    import truecolor

    display, _ = truecolor.finalize_truecolor_legacy(
        source_state.day_rgb_reflectance,
        source_state.cos_sza,
        source_state.glint_deg,
        source_state.red_corrected,
        source_state.veggie_corrected,
        source_state.ir_bt_kelvin,
    )
    return display.astype(np.float32, copy=False)


def _display_first_mutant(
    source_state,
    response_y: np.ndarray,
    response_x: np.ndarray,
    background: tuple[float, float, float],
) -> np.ndarray:
    source_display = _legacy_source_display(source_state)
    source_coverage = (
        source_state.day_coverage
        if source_state.ir_coverage is None
        else np.maximum(source_state.day_coverage, source_state.ir_coverage)
    )
    delivered, coverage = _premultiplied_resample(
        source_display, source_coverage, response_y, response_x
    )
    clean = np.nan_to_num(delivered, nan=0.0).astype(np.float32, copy=False)
    result = clean * coverage[..., None]
    result += (np.float32(1.0) - coverage)[..., None] * np.asarray(
        background, dtype=np.float32
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def _old_nearest_mutant(
    source_state,
    response_y: np.ndarray,
    response_x: np.ndarray,
) -> np.ndarray:
    source_display = np.clip(
        np.nan_to_num(_legacy_source_display(source_state), nan=0.0), 0.0, 1.0
    )
    return np.clip(
        _sparse_separable(source_display, response_y, response_x, epsilon=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def _canonical_object_identity_rows(
    identities: Sequence[Any],
) -> tuple[Mapping[str, Any], ...]:
    """Normalize only the exact four-field canonical object identity type."""

    rows: list[Mapping[str, Any]] = []
    required = {"role", "file_identifier", "bytes", "sha256"}
    for identity in identities:
        if dataclasses.is_dataclass(identity) and not isinstance(identity, type):
            raw = asdict(identity)
        elif isinstance(identity, Mapping):
            raw = dict(identity)
        else:
            raise TypeError("canonical object identity has an unsupported type")
        if set(raw) != required:
            raise RuntimeError("canonical object identity fields changed")
        role = str(raw["role"])
        file_identifier = str(raw["file_identifier"])
        byte_count = int(raw["bytes"])
        digest = str(raw["sha256"])
        if (
            role not in m1.M1_REQUIRED_ROLES
            or not file_identifier
            or Path(file_identifier).name != file_identifier
            or byte_count <= 0
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RuntimeError("canonical object identity is malformed")
        rows.append(
            {
                "role": role,
                "file_identifier": file_identifier,
                "bytes": byte_count,
                "sha256": digest,
            }
        )
    return tuple(rows)


def _m0_m1_rollback_expected_hashes(
    baseline: M1ArtifactBaseline,
) -> Mapping[str, str]:
    return {
        "source_display": baseline.source_display_sha256,
        "m0_nearest_delivered_raster": baseline.m0_nearest_raster_sha256,
        "m0_nearest_png": baseline.m0_nearest_png_sha256,
        "m1_hanning_delivered_raster": baseline.m1_hanning_raster_sha256,
        "m1_hanning_png": baseline.m1_hanning_png_sha256,
    }


def _changed_byte_hash_discriminator(
    payload: bytes,
    *,
    expected_sha256: str,
    artifact: str,
) -> Mapping[str, Any]:
    """Actually alter one retained byte and apply the registered hash check."""

    if type(payload) is not bytes or not payload:
        raise ValueError("changed-byte discriminator requires nonempty exact bytes")
    offset = len(payload) // 2
    mutant = bytearray(payload)
    before = int(mutant[offset])
    mutant[offset] ^= 0x01
    after = int(mutant[offset])
    approved_sha256 = _sha256_bytes(payload)
    mutant_sha256 = _sha256_bytes(bytes(mutant))
    approved_matches = approved_sha256 == expected_sha256
    mutant_matches = mutant_sha256 == expected_sha256
    return {
        "artifact": artifact,
        "mutation": "xor_0x01_at_one_retained_payload_byte",
        "payload_bytes": len(payload),
        "byte_offset": offset,
        "byte_before": before,
        "byte_after": after,
        "expected_sha256": expected_sha256,
        "approved_sha256": approved_sha256,
        "mutant_sha256": mutant_sha256,
        "approved_matches_expected": approved_matches,
        "mutant_matches_expected": mutant_matches,
        "target_hash_discriminator_killed": bool(
            approved_matches
            and not mutant_matches
            and mutant_sha256 != approved_sha256
        ),
    }


def _measure_current_m0_m1_rollback(
    research: ResearchInputs,
    corpus_root: str | os.PathLike[str] | Path,
    baseline: M1ArtifactBaseline,
) -> Mapping[str, Any]:
    """Replay today's real legacy compositor and both rollback render policies."""

    import render_policy

    source_fetch, identities = m1.load_canonical_melissa_display_array(
        research, corpus_root
    )
    identity_rows = _canonical_object_identity_rows(identities)
    m0_policy = render_policy.require_registered_policy(render_policy.M0_POLICY)
    m1_policy = render_policy.require_registered_policy(render_policy.M1_POLICY)
    m0_render = m1._render_canonical_raster(  # noqa: SLF001 - current rollback path
        source_fetch, m0_policy
    )
    m1_render = m1._render_canonical_raster(  # noqa: SLF001 - current rollback path
        source_fetch, m1_policy
    )

    source = np.asarray(source_fetch.cmi)
    m0_raster = np.asarray(m0_render.raster)
    m1_raster = np.asarray(m1_render.raster)
    m0_png = bytes(m0_render.png)
    m1_png = bytes(m1_render.png)
    m0_raster_payload = np.ascontiguousarray(m0_raster).tobytes(order="C")
    m1_raster_payload = np.ascontiguousarray(m1_raster).tobytes(order="C")
    observed_hashes = {
        "source_display": _sha256_bytes(source.tobytes(order="C")),
        "m0_nearest_delivered_raster": _sha256_bytes(m0_raster_payload),
        "m0_nearest_png": _sha256_bytes(m0_png),
        "m1_hanning_delivered_raster": _sha256_bytes(m1_raster_payload),
        "m1_hanning_png": _sha256_bytes(m1_png),
    }
    expected_hashes = dict(_m0_m1_rollback_expected_hashes(baseline))
    hash_matches = {
        name: observed_hashes[name] == expected
        for name, expected in expected_hashes.items()
    }
    current_interpolations = {
        "m0_nearest": list(m0_render.interpolations),
        "m1_hanning": list(m1_render.interpolations),
    }
    expected_interpolations = {
        "m0_nearest": list(baseline.m0_runtime_interpolations),
        "m1_hanning": list(baseline.m1_runtime_interpolations),
    }
    structure_checks = {
        "source_shape": tuple(source.shape) == baseline.source_shape,
        "source_dtype_float32": source.dtype == np.dtype(np.float32),
        "source_all_finite": bool(np.isfinite(source).all()),
        "object_identities": identity_rows == baseline.object_identities,
        "m0_delivered_shape": tuple(m0_raster.shape) == baseline.delivered_shape,
        "m1_delivered_shape": tuple(m1_raster.shape) == baseline.delivered_shape,
        "m0_raster_dtype_uint8": m0_raster.dtype == np.dtype(np.uint8),
        "m1_raster_dtype_uint8": m1_raster.dtype == np.dtype(np.uint8),
        "m0_runtime_interpolation": (
            tuple(m0_render.interpolations) == baseline.m0_runtime_interpolations
        ),
        "m1_runtime_interpolation": (
            tuple(m1_render.interpolations) == baseline.m1_runtime_interpolations
        ),
        "m0_png_exact_bytes": type(m0_render.png) is bytes and len(m0_png) > 0,
        "m1_png_exact_bytes": type(m1_render.png) is bytes and len(m1_png) > 0,
    }
    baseline_passed = all(structure_checks.values()) and all(hash_matches.values())
    changed_byte = dict(
        _changed_byte_hash_discriminator(
            m1_raster_payload,
            expected_sha256=baseline.m1_hanning_raster_sha256,
            artifact="current_m1_hanning_delivered_raster",
        )
    )
    changed_byte["full_replay_baseline_passed"] = baseline_passed
    changed_byte["killed"] = bool(
        baseline_passed and changed_byte["target_hash_discriminator_killed"]
    )
    return {
        "schema": "tat_truecolor_m0_m1_current_rollback_replay/v1",
        "scene_id": M2_SCENE_ID,
        "procedure": (
            "current_canonical_Melissa_legacy_ABI_compositor_then_current_"
            "registered_M0_nearest_and_M1_Hanning_production_renderers"
        ),
        "m1_report_sha256": baseline.report_sha256,
        "m1_git_head": baseline.git_head,
        "object_identities": [dict(item) for item in identity_rows],
        "source": {
            "shape": list(source.shape),
            "dtype": source.dtype.str,
            "all_finite": bool(np.isfinite(source).all()),
        },
        "delivered": {
            "m0_nearest_shape": list(m0_raster.shape),
            "m0_nearest_dtype": m0_raster.dtype.str,
            "m0_nearest_png_bytes": len(m0_png),
            "m1_hanning_shape": list(m1_raster.shape),
            "m1_hanning_dtype": m1_raster.dtype.str,
            "m1_hanning_png_bytes": len(m1_png),
        },
        "policy_versions": {
            "m0_nearest": m0_policy.algo_version,
            "m1_hanning": m1_policy.algo_version,
        },
        "runtime_interpolations": current_interpolations,
        "expected_runtime_interpolations": expected_interpolations,
        "expected_sha256": expected_hashes,
        "observed_sha256": observed_hashes,
        "hash_matches": hash_matches,
        "structure_checks": structure_checks,
        "changed_byte_discriminator": changed_byte,
        "passed": bool(baseline_passed and changed_byte["killed"]),
    }


def measure_canonical_m2(
    research: ResearchInputs,
    *,
    corpus_root: str | os.PathLike[str] | Path,
    m1_artifact_directory: str | os.PathLike[str] | Path,
    include_limb_v41: bool = True,
) -> CanonicalM2Evidence:
    """Run canonical real-scene measurement; no files are written here."""

    from matplotlib.colors import to_rgb
    import render
    import truecolor

    baseline = verify_m1_artifact_baseline(m1_artifact_directory)
    _require_m1_commit_available(baseline)
    rollback_replay = _measure_current_m0_m1_rollback(
        research, corpus_root, baseline
    )
    if rollback_replay.get("passed") is not True:
        raise RuntimeError(
            "current M0/M1 real-corpus rollback replay differs from trusted M1 bytes"
        )
    data, identities = load_canonical_melissa_physical(research, corpus_root)
    if _canonical_object_identity_rows(identities) != baseline.object_identities:
        raise RuntimeError("M2 physical capture objects differ from the M1 baseline")
    capture = capture_production_m2_render(data)
    if capture.trace_issues:
        raise RuntimeError(f"M2 production trace is unbound: {capture.trace_issues}")
    if capture.policy_version != M2_REQUIRED_POLICY_VERSION:
        raise RuntimeError("canonical production render used the wrong policy identity")

    area_y, area_x = exact_area_responses_for_plan(capture.plan)
    hanning_y, hanning_x = measured_kernel_responses_for_plan(capture.plan)
    nearest_y, nearest_x = measured_kernel_responses_for_plan(
        capture.plan, kernel="nearest"
    )
    exact_state = _reference_state_from_responses(
        capture.source_state, area_y, area_x
    )
    background = tuple(float(item) for item in to_rgb(render.DARK_BG))
    exact_delivery = truecolor.finalize_truecolor_m2(
        exact_state, background_display_rgb=background
    )
    reference_float = exact_delivery.rgb
    reference_floor = _floor_dn(reference_float)
    candidate_float = capture.delivered_float_rgb
    candidate_floor = capture.delivered_floor_rgb
    if candidate_floor.shape != reference_floor.shape:
        raise RuntimeError("candidate/reference delivered shapes differ")

    nearest_float = _old_nearest_mutant(
        capture.source_state, nearest_y, nearest_x
    )
    nearest_floor = _floor_dn(nearest_float)
    display_first_float = _display_first_mutant(
        capture.source_state, hanning_y, hanning_x, background
    )
    display_first_floor = _floor_dn(display_first_float)

    rms = m2.rms_dn(candidate_floor, reference_floor)
    nearest_rms = m2.rms_dn(nearest_floor, reference_floor)
    spectral = m2.radial_high_frequency_power_ratio(
        candidate_floor, reference_floor, cutoff=0.3, bins=64
    )
    nearest_spectral = m2.radial_high_frequency_power_ratio(
        nearest_floor, reference_floor, cutoff=0.3, bins=64
    )
    signed_all = float(
        np.mean(
            (candidate_float.astype(np.float64) - reference_float.astype(np.float64))
            * 255.0
        )
    )
    signed_display_first = float(
        np.mean(
            (display_first_float.astype(np.float64) - reference_float.astype(np.float64))
            * 255.0
        )
    )
    rec601 = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
    signed_luma = float(
        np.mean(
            np.tensordot(
                candidate_float.astype(np.float64)
                - reference_float.astype(np.float64),
                rec601,
                axes=([-1], [0]),
            )
            * 255.0
        )
    )
    coverage = _source_coverage_summary(hanning_y, hanning_x)
    nearest_coverage = _source_coverage_summary(nearest_y, nearest_x)
    v15 = _v15_measurements(capture, hanning_y, hanning_x)
    v16 = _v16_measurements(capture, hanning_x, baseline)
    terminator_mutant = _terminator_mutation_measurement(hanning_x)
    if not terminator_mutant["killed"]:
        raise RuntimeError("captured geometry did not discriminate terminator-order mutant")
    reference_pixels = _reference_pixel_finalizer_measurements()
    endpoint_fixture = _endpoint_restoration_measurements()
    if not endpoint_fixture["passed"]:
        raise RuntimeError("registered finalizer failed endpoint-identity binding")

    primary_summary = capture.summary_dict()
    primary_png = capture.png
    primary_operations = [asdict(item) for item in capture.operations]
    primary_trace_issues = list(capture.trace_issues)
    primary_trace_identity = capture.trace_identity_sha256
    primary_geometry_identity = capture.geometry.identity_sha256
    primary_endpoint_binding = dict(capture.endpoint_binding)
    candidate_float_saved = candidate_float.copy()
    candidate_floor_saved = candidate_floor.copy()

    # Release raw physical source/delivered states before the independent limb
    # fetch.  Only compact identities, metrics, PNG bytes and gallery rasters
    # survive into the returned evidence object.
    del data, capture, exact_state, exact_delivery

    limb_summary = None
    v41: Mapping[str, Any] = {
        "passed": False,
        "unmeasured_reason": "bound real disk-limb render was not requested",
    }
    if include_limb_v41:
        # The primary capture remains live only until its summaries and image
        # arrays above are materialized.  The limb fetch is then performed as a
        # separate production operation so callers can run this function in a
        # worker with a bounded peak rather than retaining raw band crops.
        limb_data, limb_identities = load_canonical_melissa_physical(
            research, corpus_root, bbox=M2_LIMB_BBOX
        )
        if tuple(asdict(item) for item in limb_identities) != tuple(
            asdict(item) for item in identities
        ):
            raise RuntimeError("V41 limb render used different pinned objects")
        limb_capture = capture_production_m2_render(
            limb_data, bbox=M2_LIMB_BBOX
        )
        if limb_capture.trace_issues:
            raise RuntimeError(
                f"M2 limb production trace is unbound: {limb_capture.trace_issues}"
            )
        limb_y, limb_x = measured_kernel_responses_for_plan(limb_capture.plan)
        v41 = _v41_measurements(limb_capture, limb_y, limb_x)
        limb_summary = limb_capture.summary_dict()
        del limb_data, limb_capture, limb_y, limb_x

    object_rows = tuple(asdict(item) for item in identities)
    metrics: Mapping[str, Any] = {
        "V01": {
            "candidate_rms_dn": rms,
            "old_nearest_rms_dn": nearest_rms,
            "reference_quantizer": "production_floor_u8_no_dither",
            "reference": (
                "exact detector-cell area response derived from captured affine; "
                "physical reflectance/K/day-weight and coverage first, one CIRA last"
            ),
        },
        "V02": {"candidate": coverage, "old_nearest_mutant": nearest_coverage},
        "V03": {
            "candidate_ratio": spectral,
            "old_nearest_ratio": nearest_spectral,
            "luma": "Rec.601",
            "spatial_window": "separable_2D_Hann",
            "radial_bins": 64,
            "cutoff_cycles_per_pixel": 0.3,
        },
        "V07": {
            "m1_report_sha256": baseline.report_sha256,
            "m1_git_head": baseline.git_head,
            "current_cira_source_sha256": m2.production_adapter.cira_source_sha256(),
        },
        "V08": reference_pixels,
        "V13": {
            "mean_signed_all_channel_dn": signed_all,
            "mean_signed_rec601_luma_dn": signed_luma,
            "display_first_mutant_mean_signed_dn": signed_display_first,
            "operation_trace": primary_operations,
            "trace_issues": primary_trace_issues,
            "trace_identity_sha256": primary_trace_identity,
            "geometry_identity_sha256": primary_geometry_identity,
            "terminator_order_mutant": terminator_mutant,
            "captured_endpoint_binding": primary_endpoint_binding,
            "endpoint_identity_fixture": endpoint_fixture,
        },
        "V14": _contrast_measurements(),
        "V15": {
            **v15,
            "m1_report_sha256": baseline.report_sha256,
            "m1_baseline_out_of_range_fraction": baseline.v15_out_of_range_fraction,
        },
        "V16": v16,
        "V41": v41,
        "M0_M1_ROLLBACK": rollback_replay,
        "mutation_discriminators": {
            "old_nearest": {
                "rms_dn": nearest_rms,
                "spectral_ratio": nearest_spectral,
                "killed": nearest_rms > 1.5 and nearest_spectral > 1.10,
            },
            "display_first": {
                "mean_signed_dn": signed_display_first,
                "killed": abs(signed_display_first) > 0.05,
            },
            "resample_cos_then_nonlinear_day_weight": terminator_mutant,
            "output_proxy_canonical_coverage": (
                _output_proxy_mutation_discriminator(v41)
            ),
            "current_m0_m1_rollback_bytes": rollback_replay[
                "changed_byte_discriminator"
            ],
        },
        "limitations": [
            "M2 closes only the primary imagery-layer boundary; encoded loop, "
            "pyramid, Web-Mercator and browser resizes survive until M5.",
            "The M2 physical source cap is a separate upstream Hanning residual.",
            "S03 co-registration remains bilinear until M4.",
            "V44, V45 and V46 remain unmeasured.",
        ],
    }
    return CanonicalM2Evidence(
        scene_id=M2_SCENE_ID,
        bbox=M2_CANONICAL_BBOX,
        object_identities=object_rows,
        primary_capture=primary_summary,
        limb_capture=limb_summary,
        primary_png=primary_png,
        candidate_float=candidate_float_saved,
        candidate_floor=candidate_floor_saved,
        exact_area_reference_float=reference_float.copy(),
        exact_area_reference_floor=reference_floor.copy(),
        old_nearest_mutant_floor=nearest_floor.copy(),
        display_first_mutant_float=display_first_float.copy(),
        display_first_mutant_floor=display_first_floor.copy(),
        metrics=metrics,
        m1_baseline=baseline,
    )


def _physical_request_identity(data, bbox, dpi: int) -> str:
    state = data.truecolor_physical
    fields = {
        "day_rgb_reflectance": state.day_rgb_reflectance,
        "day_coverage": state.day_coverage,
        "ir_bt_kelvin": state.ir_bt_kelvin,
        "ir_coverage": state.ir_coverage,
        "cos_sza": state.cos_sza,
        "day_weight": state.day_weight,
        "geom_valid": state.geom_valid,
        "red_corrected": state.red_corrected,
        "veggie_corrected": state.veggie_corrected,
        "glint_deg": state.glint_deg,
        "lats": data.lats,
        "lons": data.lons,
    }
    payload = {
        "bbox": [float(item) for item in bbox],
        "dpi": int(dpi),
        "fetch": {
            "channel": int(data.channel),
            "generic_channel": data.generic_channel,
            "scan_start": data.scan_start.isoformat(),
            "product": data.product,
            "bucket": data.bucket,
            "sat_name": data.sat_name,
            "sub_sat_lon": float(data.sub_sat_lon),
            "units": data.units,
        },
        "state": {
            "sensor": state.sensor,
            "array_domain": state.array_domain.value,
            "ir_temperature_unit": state.ir_temperature_unit.value,
            "extent": list(state.extent),
            "projection": state.projection,
            "grid_metadata": state.grid_metadata,
        },
        "arrays": {
            name: None if field is None else _array_record(name, field)
            for name, field in fields.items()
        },
    }
    return _sha256_bytes(_stable_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class ProcessRenderManifest:
    pid: int
    parent_pid: int
    process_start_ticks: int
    start_method: str
    python: str
    platform: str
    numpy: str
    matplotlib: str
    pillow: str
    policy_version: str
    request_identity_sha256: str
    output_bytes: int
    child_output_sha256: str
    parent_recomputed_output_sha256: str

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepeatabilityCapture:
    output_bytes: tuple[bytes, bytes, bytes]
    manifests: tuple[ProcessRenderManifest, ProcessRenderManifest, ProcessRenderManifest]

    @property
    def hashes(self) -> tuple[str, str, str]:
        return tuple(_sha256_bytes(payload) for payload in self.output_bytes)  # type: ignore[return-value]

    @property
    def passed(self) -> bool:
        return (
            len(self.output_bytes) == 3
            and len(self.manifests) == 3
            and not m2.repeatability_issues(self.hashes)
            and len({row.pid for row in self.manifests}) == 3
            and all(
                row.child_output_sha256 == row.parent_recomputed_output_sha256
                == digest
                for row, digest in zip(self.manifests, self.hashes, strict=True)
            )
            and all(
                row.output_bytes == len(payload)
                and row.parent_pid == os.getpid()
                and row.process_start_ticks > 0
                and row.start_method == "fork"
                and row.policy_version == M2_REQUIRED_POLICY_VERSION
                for row, payload in zip(
                    self.manifests, self.output_bytes, strict=True
                )
            )
            and len({row.request_identity_sha256 for row in self.manifests}) == 1
            and all(
                len(row.request_identity_sha256) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in row.request_identity_sha256
                )
                for row in self.manifests
            )
        )

    def matches_primary_png(self, primary_png: bytes) -> bool:
        """Bind all three repeatability results to the canonical render bytes."""

        if type(primary_png) is not bytes:
            raise TypeError("canonical primary PNG must be exact bytes")
        return self.passed and all(
            payload == primary_png for payload in self.output_bytes
        )

    def summary_dict(self) -> Mapping[str, Any]:
        return {
            "passed": self.passed,
            "hashes": list(self.hashes),
            "output_bytes": [len(item) for item in self.output_bytes],
            "manifests": [row.as_dict() for row in self.manifests],
            "hash_provenance": "parent_recomputed_from_child_renderer_bytes",
        }


def _process_start_ticks() -> int:
    fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
    if len(fields) < 22:
        raise RuntimeError("/proc/self/stat does not expose process start ticks")
    return int(fields[21])


def _repeatability_worker(data, bbox, dpi: int, connection) -> None:
    try:
        import matplotlib
        import PIL
        import render
        import render_policy

        request_identity = _physical_request_identity(data, bbox, dpi)
        output = render.render_png(
            data,
            list(bbox),
            2,
            "2025-10-27 15:00 UTC",
            "true_color",
            1,
            coastlines=False,
            gridlines=False,
            dpi=dpi,
            policy=render_policy.M2_POLICY,
        )
        manifest = {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "process_start_ticks": _process_start_ticks(),
            "start_method": multiprocessing.get_start_method(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
            "policy_version": render_policy.M2_POLICY.algo_version,
            "request_identity_sha256": request_identity,
            "output_bytes": len(output),
            "child_output_sha256": _sha256_bytes(output),
        }
        connection.send(("ok", output, manifest))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}", os.getpid()))
    finally:
        connection.close()


def capture_three_process_repeatability(
    data,
    *,
    bbox: tuple[float, float, float, float] = M2_CANONICAL_BBOX,
    dpi: int = M2_DEFAULT_DPI,
    timeout_seconds: float = 180.0,
) -> RepeatabilityCapture:
    """Render sequentially in exactly three fresh processes and retain bytes.

    Workers are intentionally sequential: each M2 render has a material but
    bounded physical-field working set, and concurrent workers would turn the
    deterministic gate into an avoidable memory-pressure test.
    """

    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("canonical V25 requires Linux fork process isolation")
    context = multiprocessing.get_context("fork")
    outputs: list[bytes] = []
    rows: list[ProcessRenderManifest] = []
    for _ in range(3):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_repeatability_worker,
            args=(data, tuple(bbox), int(dpi), child),
        )
        process.start()
        child.close()
        try:
            if not parent.poll(timeout_seconds):
                raise RuntimeError(f"M2 repeatability worker {process.pid} timed out")
            message = parent.recv()
            if message[0] != "ok":
                raise RuntimeError(
                    f"M2 repeatability worker {process.pid} failed: {message[1]}"
                )
            output = bytes(message[1])
            manifest = dict(message[2])
            parent_hash = _sha256_bytes(output)
            rows.append(
                ProcessRenderManifest(
                    **manifest,
                    parent_recomputed_output_sha256=parent_hash,
                )
            )
            outputs.append(output)
            process.join(timeout=10.0)
            if process.exitcode != 0:
                raise RuntimeError(
                    f"M2 repeatability worker {process.pid} exited {process.exitcode}"
                )
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
    if len(outputs) != 3 or len(rows) != 3:
        raise RuntimeError("M2 V25 did not produce exactly three process captures")
    result = RepeatabilityCapture(
        output_bytes=(outputs[0], outputs[1], outputs[2]),
        manifests=(rows[0], rows[1], rows[2]),
    )
    if not result.passed:
        raise RuntimeError(f"M2 V25 repeatability failed: {result.summary_dict()}")
    return result


def _repeatability_mutation_discriminator(
    capture: RepeatabilityCapture | None,
) -> Mapping[str, Any]:
    """Kill a changed-renderer-byte mutant without accepting caller hashes."""

    caller_hash_parameter_present = "hashes" in inspect.signature(
        capture_three_process_repeatability
    ).parameters
    if capture is None:
        return {
            "baseline_passed": False,
            "changed_byte_mutant_passed": None,
            "caller_hash_parameter_present": caller_hash_parameter_present,
            "killed": False,
            "unmeasured_reason": "three renderer byte captures are absent",
        }

    outputs = list(capture.output_bytes)
    if not outputs:
        return {
            "baseline_passed": capture.passed,
            "changed_byte_mutant_passed": None,
            "caller_hash_parameter_present": caller_hash_parameter_present,
            "killed": False,
            "unmeasured_reason": "renderer byte capture is empty",
        }
    changed = bytearray(outputs[0])
    changed_index = len(changed) // 2 if changed else 0
    if changed:
        changed[changed_index] ^= 0x01
    else:
        changed.append(0x01)
    outputs[0] = bytes(changed)
    mutant = dataclasses.replace(
        capture,
        output_bytes=(outputs[0], outputs[1], outputs[2]),
    )
    baseline_passed = capture.passed
    mutant_passed = mutant.passed
    return {
        "baseline_passed": baseline_passed,
        "changed_byte_index": changed_index,
        "changed_byte_mutant_passed": mutant_passed,
        "caller_hash_parameter_present": caller_hash_parameter_present,
        "killed": bool(
            baseline_passed
            and not mutant_passed
            and not caller_hash_parameter_present
        ),
    }


@dataclass(frozen=True, slots=True)
class TimestampedRendererFrame:
    timestamp_utc: str
    raster_rgb_u8: np.ndarray
    full_daylight_mask: np.ndarray
    request_identity_sha256: str
    grid_identity_sha256: str
    source_object_sha256: tuple[str, ...]
    renderer_png_sha256: str
    renderer_raster_sha256: str


@dataclass(frozen=True, slots=True)
class TimestampedSequenceBinding:
    schema: str
    frames: tuple[TimestampedRendererFrame, ...]
    patch_bounds: tuple[int, int, int, int]
    stationary_world_bounds: tuple[float, float, float, float]
    stationary_class: str
    stationarity_method: str
    daylight_mask_producer_binding: str


@dataclass(frozen=True, slots=True)
class SequenceEvaluation:
    outcome: str
    metrics: m2.StationaryFlickerMetrics | None
    issues: tuple[str, ...]
    frame_identities: tuple[Mapping[str, Any], ...]
    patch_bounds: tuple[int, int, int, int] | None
    binding_metadata: Mapping[str, Any] | None

    @property
    def passed(self) -> bool:
        return self.outcome == "PASS" and self.metrics is not None

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "outcome": self.outcome,
            "metrics": None if self.metrics is None else asdict(self.metrics),
            "issues": list(self.issues),
            "frames": [dict(item) for item in self.frame_identities],
            "patch_bounds": self.patch_bounds,
            "binding": (
                None if self.binding_metadata is None else dict(self.binding_metadata)
            ),
            "procedure": (
                "1.4826*MAD(per-frame metric second difference)/sqrt(6), "
                "recomputed from bound uint8 renderer arrays"
            ),
        }


def _parse_utc_timestamp(value: str) -> dt.datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("timestamp must carry an explicit UTC offset")
    return parsed.astimezone(dt.timezone.utc)


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _frame_patch_metrics(patch: np.ndarray) -> Mapping[str, float]:
    rgb = np.asarray(patch, dtype=np.float64)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    laplacian = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    means = np.mean(rgb, axis=(0, 1))
    channel_max = np.max(rgb, axis=-1)
    channel_min = np.min(rgb, axis=-1)
    saturation = np.zeros(channel_max.shape, dtype=np.float64)
    np.divide(
        channel_max - channel_min,
        channel_max,
        out=saturation,
        where=channel_max > 0.0,
    )
    green = max(float(means[1]), np.finfo(np.float64).tiny)
    return {
        "luminance_mean": float(np.mean(luma)),
        "luminance_median": float(np.median(luma)),
        "luminance_p1": float(np.percentile(luma, 1.0)),
        "luminance_p95": float(np.percentile(luma, 95.0)),
        "luminance_p999": float(np.percentile(luma, 99.9)),
        "laplacian_rms": float(np.sqrt(np.mean(np.square(laplacian)))),
        "red_green_ratio": float(means[0] / green),
        "blue_green_ratio": float(means[2] / green),
        "mean_saturation": float(np.mean(saturation)),
        "black_clipped_fraction": float(np.mean(luma <= 0.0)),
        "white_clipped_fraction": float(np.mean(luma >= 255.0)),
    }


def _flicker_sigma(values: Sequence[float]) -> float:
    series = np.asarray(values, dtype=np.float64)
    if series.size < 5 or np.any(~np.isfinite(series)):
        raise ValueError("flicker series requires at least five finite frames")
    second = series[2:] - 2.0 * series[1:-1] + series[:-2]
    median = float(np.median(second))
    mad = float(np.median(np.abs(second - median)))
    return 1.4826 * mad / math.sqrt(6.0)


def evaluate_timestamped_sequence(
    binding: TimestampedSequenceBinding | None,
) -> SequenceEvaluation:
    """Recompute V30 from renderer arrays; summaries/hashes cannot substitute."""

    if binding is None:
        return SequenceEvaluation(
            outcome="UNMEASURED",
            metrics=None,
            issues=("timestamped bound renderer frame arrays are absent",),
            frame_identities=(),
            patch_bounds=None,
            binding_metadata=None,
        )
    if type(binding) is not TimestampedSequenceBinding:
        return SequenceEvaluation(
            outcome="FAIL",
            metrics=None,
            issues=("sequence binding is not the exact registered type",),
            frame_identities=(),
            patch_bounds=None,
            binding_metadata=None,
        )
    issues: list[str] = []
    binding_metadata: Mapping[str, Any] = {
        "schema": binding.schema,
        "frame_count": len(binding.frames) if type(binding.frames) is tuple else None,
        "stationary_world_bounds": (
            list(binding.stationary_world_bounds)
            if type(binding.stationary_world_bounds) is tuple
            else None
        ),
        "stationary_class": binding.stationary_class,
        "stationarity_method": binding.stationarity_method,
        "daylight_mask_producer_binding": binding.daylight_mask_producer_binding,
    }
    if binding.schema != M2_SEQUENCE_SCHEMA:
        issues.append("sequence schema is not registered")
    if type(binding.frames) is not tuple:
        issues.append("renderer frames are not an immutable tuple")
        frames: tuple[Any, ...] = ()
    else:
        frames = binding.frames
    if len(frames) < 5:
        issues.append("V30 requires at least five renderer frames")
    if binding.stationary_class != "clear_land":
        issues.append("stationary patch is not the registered clear_land class")
    if binding.stationarity_method != M2_STATIONARITY_METHOD:
        issues.append("stationarity method is not registered")
    if (
        binding.daylight_mask_producer_binding
        != M2_DAYLIGHT_MASK_PRODUCER_BINDING
    ):
        issues.append("daylight masks lack the registered physical producer binding")

    world_bounds_valid = bool(
        type(binding.stationary_world_bounds) is tuple
        and len(binding.stationary_world_bounds) == 4
        and all(
            type(value) in (int, float) and math.isfinite(float(value))
            for value in binding.stationary_world_bounds
        )
    )
    if world_bounds_valid:
        west, south, east, north = (
            float(value) for value in binding.stationary_world_bounds
        )
        world_bounds_valid = bool(
            -180.0 <= west < east <= 180.0
            and -90.0 <= south < north <= 90.0
        )
    if not world_bounds_valid:
        issues.append("stationary world bounds are not finite ordered W,S,E,N")

    patch_valid = bool(
        type(binding.patch_bounds) is tuple
        and len(binding.patch_bounds) == 4
        and all(type(value) is int for value in binding.patch_bounds)
    )
    if patch_valid:
        top, bottom, left, right = binding.patch_bounds
        patch_valid = 0 <= top < bottom and 0 <= left < right
    else:
        top = bottom = left = right = 0
    if not patch_valid:
        issues.append("stationary patch bounds are not ordered integer pixel edges")

    frame_rows: list[Mapping[str, Any]] = []
    timestamps: list[dt.datetime] = []
    metric_rows: list[Mapping[str, float]] = []
    expected_shape = None
    request_hashes: list[str] = []
    grid_hashes: list[str] = []
    source_hashes: list[str] = []
    raster_hashes: list[str] = []
    for index, frame in enumerate(frames):
        if type(frame) is not TimestampedRendererFrame:
            issues.append(f"frame {index} is not the exact registered frame type")
            continue
        try:
            timestamp = _parse_utc_timestamp(frame.timestamp_utc)
            timestamps.append(timestamp)
        except (TypeError, ValueError) as exc:
            issues.append(f"frame {index} timestamp is invalid: {exc}")
            continue
        raster = frame.raster_rgb_u8
        daylight = frame.full_daylight_mask
        if type(raster) is not np.ndarray or raster.dtype != np.uint8:
            issues.append(f"frame {index} raster is not an exact uint8 ndarray")
            continue
        if raster.ndim != 3 or raster.shape[-1] != 3:
            issues.append(f"frame {index} raster is not HxWxRGB")
            continue
        if type(daylight) is not np.ndarray or daylight.dtype != np.bool_:
            issues.append(f"frame {index} daylight mask is not an exact bool ndarray")
            continue
        if daylight.shape != raster.shape[:2]:
            issues.append(f"frame {index} daylight mask shape differs")
            continue
        if expected_shape is None:
            expected_shape = raster.shape
        if raster.shape != expected_shape:
            issues.append(f"frame {index} renderer shape changed")
        request_hashes.append(frame.request_identity_sha256)
        grid_hashes.append(frame.grid_identity_sha256)
        frame_source_hashes = (
            frame.source_object_sha256
            if type(frame.source_object_sha256) is tuple
            else ()
        )
        for label, digest in (
            ("request", frame.request_identity_sha256),
            ("grid", frame.grid_identity_sha256),
            ("renderer PNG", frame.renderer_png_sha256),
            ("renderer raster", frame.renderer_raster_sha256),
            *(("source object", value) for value in frame_source_hashes),
        ):
            if not _valid_sha256(digest):
                issues.append(f"frame {index} {label} SHA-256 is invalid")
        actual_hash = _array_sha256(raster)
        raster_hashes.append(actual_hash)
        if actual_hash != frame.renderer_raster_sha256:
            issues.append(
                f"frame {index} renderer raster hash does not match array bytes"
            )
        if len(frame_source_hashes) != 4:
            issues.append(f"frame {index} does not bind exactly four source objects")
        else:
            source_hashes.extend(frame_source_hashes)
        if not patch_valid or not (
            0 <= top < bottom <= raster.shape[0]
            and 0 <= left < right <= raster.shape[1]
        ):
            issues.append(f"frame {index} patch bounds lie outside renderer raster")
            continue
        if bottom - top < 96 or right - left < 96:
            issues.append("stationary patch is smaller than 96x96 delivered pixels")
        if not np.all(daylight[top:bottom, left:right]):
            issues.append(f"frame {index} stationary patch is not full daylight")
        patch = raster[top:bottom, left:right]
        if min(patch.shape[:2]) >= 3:
            metric_rows.append(_frame_patch_metrics(patch))
        frame_rows.append(
            {
                "timestamp_utc": timestamp.isoformat(),
                "raster": _array_record(f"frame_{index}_rgb", raster),
                "daylight_mask": _array_record(
                    f"frame_{index}_full_daylight", daylight
                ),
                "request_identity_sha256": frame.request_identity_sha256,
                "grid_identity_sha256": frame.grid_identity_sha256,
                "source_object_sha256": list(frame_source_hashes),
                "renderer_png_sha256": frame.renderer_png_sha256,
                "renderer_raster_sha256": frame.renderer_raster_sha256,
                "daylight_mask_producer_binding": (
                    binding.daylight_mask_producer_binding
                ),
            }
        )
    if len(timestamps) == len(frames) and any(
        right_time - left_time != dt.timedelta(minutes=10)
        for left_time, right_time in zip(timestamps, timestamps[1:])
    ):
        issues.append("frame timestamps are not exact monotonic 10-minute cadence")
    if len(request_hashes) != len(frames) or any(
        not _valid_sha256(value) for value in request_hashes
    ) or len(set(request_hashes)) != 1:
        issues.append("static request SHA-256 is invalid or changed across frames")
    if len(grid_hashes) != len(frames) or any(
        not _valid_sha256(value) for value in grid_hashes
    ) or len(set(grid_hashes)) != 1:
        issues.append("static grid SHA-256 is invalid or changed across frames")
    if len(source_hashes) != 4 * len(frames) or any(
        not _valid_sha256(value) for value in source_hashes
    ) or len(set(source_hashes)) != 4 * len(frames):
        issues.append("four-per-frame source SHA-256 identities are invalid or reused")
    if len(raster_hashes) != len(frames) or len(set(raster_hashes)) != len(frames):
        issues.append("renderer raster bytes are frozen or duplicated across frames")
    if issues or len(metric_rows) != len(frames):
        return SequenceEvaluation(
            outcome="FAIL",
            metrics=None,
            issues=tuple(issues or ["not every frame produced bound patch metrics"]),
            frame_identities=tuple(frame_rows),
            patch_bounds=binding.patch_bounds if patch_valid else None,
            binding_metadata=binding_metadata,
        )

    metric_names = tuple(metric_rows[0])
    flicker = {
        name: _flicker_sigma([row[name] for row in metric_rows])
        for name in metric_names
    }
    metrics = m2.StationaryFlickerMetrics(
        full_daylight=True,
        stationary_patch=True,
        patch_height=bottom - top,
        patch_width=right - left,
        **flicker,
    )
    threshold_issues = m2.stationary_flicker_issues(metrics)
    return SequenceEvaluation(
        outcome="PASS" if not threshold_issues else "FAIL",
        metrics=metrics,
        issues=threshold_issues,
        frame_identities=tuple(frame_rows),
        patch_bounds=binding.patch_bounds,
        binding_metadata=binding_metadata,
    )


def _m0_m1_rollback_binding_issues(
    measurement: Any,
    baseline: M1ArtifactBaseline,
    canonical_object_identities: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Re-derive the compact rollback verdict instead of trusting booleans."""

    if not isinstance(measurement, Mapping):
        return ("current M0/M1 rollback replay measurement is absent",)
    issues: list[str] = []
    expected_procedure = (
        "current_canonical_Melissa_legacy_ABI_compositor_then_current_"
        "registered_M0_nearest_and_M1_Hanning_production_renderers"
    )
    if measurement.get("schema") != "tat_truecolor_m0_m1_current_rollback_replay/v1":
        issues.append("current M0/M1 rollback replay schema is unregistered")
    if measurement.get("scene_id") != M2_SCENE_ID:
        issues.append("current M0/M1 rollback replay scene identity changed")
    if measurement.get("procedure") != expected_procedure:
        issues.append("current M0/M1 rollback replay procedure is unregistered")
    if measurement.get("m1_report_sha256") != baseline.report_sha256:
        issues.append("current M0/M1 rollback replay does not bind the M1 report")
    if measurement.get("m1_git_head") != baseline.git_head:
        issues.append("current M0/M1 rollback replay does not bind the M1 commit")

    try:
        measured_objects = _canonical_object_identity_rows(
            measurement.get("object_identities", ())
        )
        current_objects = _canonical_object_identity_rows(
            canonical_object_identities
        )
    except (TypeError, ValueError, RuntimeError):
        measured_objects = ()
        current_objects = ()
        issues.append("current M0/M1 rollback object identities are malformed")
    if measured_objects != baseline.object_identities:
        issues.append("current M0/M1 rollback objects differ from the M1 baseline")
    if current_objects != baseline.object_identities:
        issues.append("M2 capture objects differ from the M0/M1 rollback objects")

    source = measurement.get("source")
    delivered = measurement.get("delivered")
    policies = measurement.get("policy_versions")
    interpolations = measurement.get("runtime_interpolations")
    expected_interpolations = measurement.get("expected_runtime_interpolations")
    if not all(
        isinstance(item, Mapping)
        for item in (
            source,
            delivered,
            policies,
            interpolations,
            expected_interpolations,
        )
    ):
        issues.append("current M0/M1 rollback structure metadata is incomplete")
        source = {}
        delivered = {}
        policies = {}
        interpolations = {}
        expected_interpolations = {}

    expected_runtime = {
        "m0_nearest": list(baseline.m0_runtime_interpolations),
        "m1_hanning": list(baseline.m1_runtime_interpolations),
    }
    structure_checks = {
        "source_shape": source.get("shape") == list(baseline.source_shape),
        "source_dtype_float32": source.get("dtype") == np.dtype(np.float32).str,
        "source_all_finite": source.get("all_finite") is True,
        "object_identities": measured_objects == baseline.object_identities,
        "m0_delivered_shape": delivered.get("m0_nearest_shape")
        == list(baseline.delivered_shape),
        "m1_delivered_shape": delivered.get("m1_hanning_shape")
        == list(baseline.delivered_shape),
        "m0_raster_dtype_uint8": delivered.get("m0_nearest_dtype")
        == np.dtype(np.uint8).str,
        "m1_raster_dtype_uint8": delivered.get("m1_hanning_dtype")
        == np.dtype(np.uint8).str,
        "m0_runtime_interpolation": interpolations.get("m0_nearest")
        == expected_runtime["m0_nearest"],
        "m1_runtime_interpolation": interpolations.get("m1_hanning")
        == expected_runtime["m1_hanning"],
        "m0_png_exact_bytes": type(delivered.get("m0_nearest_png_bytes")) is int
        and int(delivered["m0_nearest_png_bytes"]) > 0,
        "m1_png_exact_bytes": type(delivered.get("m1_hanning_png_bytes")) is int
        and int(delivered["m1_hanning_png_bytes"]) > 0,
    }
    if expected_interpolations != expected_runtime:
        issues.append("rollback expected interpolation record differs from M1")
    if policies != {
        "m0_nearest": "tat-truecolor-m0-baseline-v1",
        "m1_hanning": "tat-truecolor-m1-hanning-v1",
    }:
        issues.append("rollback policy-version record is not exact M0/M1")
    if measurement.get("structure_checks") != structure_checks:
        issues.append("rollback structure-check verdicts were not re-derived")
    if not all(structure_checks.values()):
        issues.append("current M0/M1 rollback shape/type/identity binding failed")

    expected_hashes = dict(_m0_m1_rollback_expected_hashes(baseline))
    recorded_expected = measurement.get("expected_sha256")
    observed_hashes = measurement.get("observed_sha256")
    if recorded_expected != expected_hashes:
        issues.append("rollback expected hashes differ from the M1 baseline")
    if not isinstance(observed_hashes, Mapping) or set(observed_hashes) != set(
        expected_hashes
    ):
        issues.append("rollback observed hash set is incomplete")
        observed_hashes = {}
    elif any(not _valid_sha256(value) for value in observed_hashes.values()):
        issues.append("rollback observed hash record contains an invalid SHA-256")
    hash_matches = {
        name: observed_hashes.get(name) == expected
        for name, expected in expected_hashes.items()
    }
    if measurement.get("hash_matches") != hash_matches:
        issues.append("rollback hash-match verdicts were not re-derived")
    if not all(hash_matches.values()):
        issues.append("current M0/M1 rollback bytes differ from trusted M1")

    baseline_passed = all(structure_checks.values()) and all(hash_matches.values())
    changed = measurement.get("changed_byte_discriminator")
    target_killed = False
    if not isinstance(changed, Mapping):
        issues.append("rollback changed-byte discriminator is absent")
        changed = {}
    else:
        expected_target = expected_hashes["m1_hanning_delivered_raster"]
        approved_target = observed_hashes.get("m1_hanning_delivered_raster")
        mutant_target = changed.get("mutant_sha256")
        payload_bytes = int(np.prod(baseline.delivered_shape, dtype=np.int64))
        before = changed.get("byte_before")
        after = changed.get("byte_after")
        offset = changed.get("byte_offset")
        byte_change_valid = bool(
            type(before) is int
            and type(after) is int
            and 0 <= before <= 255
            and after == (before ^ 0x01)
            and type(offset) is int
            and offset == payload_bytes // 2
        )
        target_killed = bool(
            changed.get("artifact") == "current_m1_hanning_delivered_raster"
            and changed.get("mutation")
            == "xor_0x01_at_one_retained_payload_byte"
            and changed.get("payload_bytes") == payload_bytes
            and byte_change_valid
            and changed.get("expected_sha256") == expected_target
            and changed.get("approved_sha256") == approved_target
            and _valid_sha256(mutant_target)
            and approved_target == expected_target
            and mutant_target != expected_target
            and mutant_target != approved_target
        )
        expected_changed_booleans = {
            "approved_matches_expected": approved_target == expected_target,
            "mutant_matches_expected": mutant_target == expected_target,
            "target_hash_discriminator_killed": target_killed,
            "full_replay_baseline_passed": baseline_passed,
            "killed": baseline_passed and target_killed,
        }
        if any(
            changed.get(name) is not value
            for name, value in expected_changed_booleans.items()
        ):
            issues.append("rollback changed-byte verdicts were not re-derived")
    passed = baseline_passed and target_killed
    if measurement.get("passed") is not passed:
        issues.append("current M0/M1 rollback pass verdict was not re-derived")
    if not passed:
        issues.append("current M0/M1 rollback replay did not pass")
    return tuple(issues)


def canonical_evidence_binding_issues(
    evidence: CanonicalM2Evidence,
) -> tuple[str, ...]:
    issues: list[str] = []
    if evidence.scene_id != M2_SCENE_ID or evidence.bbox != M2_CANONICAL_BBOX:
        issues.append("canonical scene/request identity changed")
    issues.extend(
        _m0_m1_rollback_binding_issues(
            evidence.metrics.get("M0_M1_ROLLBACK"),
            evidence.m1_baseline,
            evidence.object_identities,
        )
    )
    capture = evidence.primary_capture
    if capture.get("schema") != M2_PRODUCTION_SCHEMA:
        issues.append("primary capture schema is not registered")
    if capture.get("policy_version") != M2_REQUIRED_POLICY_VERSION:
        issues.append("primary capture is not bound to registered M2")
    if capture.get("trace_issues") != []:
        issues.append("primary production trace contains unresolved issues")
    geometry = capture.get("geometry", {})
    if geometry.get("capture_method") != m2.M2_CAPTURE_METHOD:
        issues.append("primary geometry is not a runtime AxesImage capture")
    if geometry.get("captured_at_runtime") is not True:
        issues.append("primary geometry is hard-coded or inferred")
    if capture.get("identity_draw", {}).get("floor_identity") is not True:
        issues.append("production-floor quantization lacks FigureImage identity")

    call_kinds = [row.get("call_kind") for row in capture.get("resample_calls", [])]
    if call_kinds != ["day_bundle", "ir", "solar_geometry", "geom_coverage"]:
        issues.append("captured physical resampler call sequence is incomplete")
    if any(
        row.get("physical_domain_transport")
        != M2_REQUIRED_PHYSICAL_DOMAIN_TRANSPORT
        for row in capture.get("resample_calls", [])
    ):
        issues.append("captured physical resampler domain transport is unregistered")
    solar = next(
        (
            row
            for row in capture.get("resample_calls", [])
            if row.get("call_kind") == "solar_geometry"
        ),
        None,
    )
    solar_names = [] if solar is None else [
        item.get("name") for item in solar.get("field_records", [])
    ]
    if solar_names != ["cos_sza", "day_weight"]:
        issues.append("day_weight was not observed beside cos_sza in union coverage")
    operations = capture.get("operations", [])
    if sum(row.get("name") == "terminator_weight_delivery" for row in operations) != 1:
        issues.append("trace lacks exactly one source-evaluated day-weight delivery")
    if any(
        row.get("name") == "terminator_day_weight" for row in capture.get("pointwise_calls", [])
    ):
        issues.append("finalizer reapplied nonlinear terminator weight")
    endpoint_binding = capture.get("endpoint_binding", {})
    if endpoint_binding.get("policy") != "post_cira_mathematically_identical_endpoints_v1":
        issues.append("post-CIRA endpoint policy is absent or unregistered")
    if endpoint_binding.get("passed") is not True:
        issues.append("captured endpoint identities or pre-CIRA mixed pixels failed")
    if endpoint_binding.get("clean_ir_gray_call_count") != 1:
        issues.append("Clean-IR gray mapping was not evaluated exactly once")
    if endpoint_binding.get("cira_stretch_call_count") != 1:
        issues.append("primary finalizer did not contain exactly one CIRA call")
    endpoint_fixture = evidence.metrics.get("V13", {}).get(
        "endpoint_identity_fixture", {}
    )
    recomputed_endpoint_fixture = _endpoint_restoration_measurements()
    if endpoint_fixture != recomputed_endpoint_fixture:
        issues.append("endpoint fixture metrics differ from registered finalizer replay")
    if recomputed_endpoint_fixture.get("passed") is not True:
        issues.append("endpoint restoration mutation was not discriminated")

    expected_records = {
        "candidate_float": evidence.candidate_float,
        "candidate_floor": evidence.candidate_floor,
    }
    capture_arrays = capture.get("delivered_float_rgb", {}), capture.get(
        "delivered_floor_rgb", {}
    )
    for (name, array), record in zip(
        expected_records.items(), capture_arrays, strict=True
    ):
        if record.get("sha256") != _array_sha256(array):
            issues.append(f"{name} bytes differ from primary capture identity")
    if not np.array_equal(evidence.candidate_floor, _floor_dn(evidence.candidate_float)):
        issues.append("candidate floor bytes were not recomputed from candidate float")
    if not np.array_equal(
        evidence.exact_area_reference_floor,
        _floor_dn(evidence.exact_area_reference_float),
    ):
        issues.append("exact-area floor bytes were not recomputed from reference float")
    if not np.array_equal(
        evidence.display_first_mutant_floor,
        _floor_dn(evidence.display_first_mutant_float),
    ):
        issues.append("display-first floor bytes were not recomputed from mutant float")
    raster_shapes = {
        tuple(np.asarray(value).shape)
        for value in (
            evidence.candidate_float,
            evidence.candidate_floor,
            evidence.exact_area_reference_float,
            evidence.exact_area_reference_floor,
            evidence.old_nearest_mutant_floor,
            evidence.display_first_mutant_float,
            evidence.display_first_mutant_floor,
        )
    }
    if len(raster_shapes) != 1:
        issues.append("canonical candidate/reference/mutant raster shapes differ")
    else:
        def metric_matches(observed: Any, recomputed: float) -> bool:
            try:
                return math.isclose(
                    float(observed), float(recomputed), rel_tol=0.0, abs_tol=1.0e-12
                )
            except (TypeError, ValueError):
                return False

        recorded = evidence.metrics
        recomputed_rms = m2.rms_dn(
            evidence.candidate_floor, evidence.exact_area_reference_floor
        )
        recomputed_nearest_rms = m2.rms_dn(
            evidence.old_nearest_mutant_floor,
            evidence.exact_area_reference_floor,
        )
        recomputed_spectral = m2.radial_high_frequency_power_ratio(
            evidence.candidate_floor,
            evidence.exact_area_reference_floor,
            cutoff=0.3,
            bins=64,
        )
        recomputed_nearest_spectral = m2.radial_high_frequency_power_ratio(
            evidence.old_nearest_mutant_floor,
            evidence.exact_area_reference_floor,
            cutoff=0.3,
            bins=64,
        )
        reference64 = evidence.exact_area_reference_float.astype(np.float64)
        candidate64 = evidence.candidate_float.astype(np.float64)
        display_first64 = evidence.display_first_mutant_float.astype(np.float64)
        recomputed_signed = float(np.mean((candidate64 - reference64) * 255.0))
        recomputed_display_signed = float(
            np.mean((display_first64 - reference64) * 255.0)
        )
        rec601 = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
        recomputed_luma_signed = float(
            np.mean(
                np.tensordot(candidate64 - reference64, rec601, axes=([-1], [0]))
                * 255.0
            )
        )
        comparisons = (
            (recorded.get("V01", {}).get("candidate_rms_dn"), recomputed_rms, "V01 candidate RMS"),
            (recorded.get("V01", {}).get("old_nearest_rms_dn"), recomputed_nearest_rms, "V01 nearest RMS"),
            (recorded.get("V03", {}).get("candidate_ratio"), recomputed_spectral, "V03 candidate spectrum"),
            (recorded.get("V03", {}).get("old_nearest_ratio"), recomputed_nearest_spectral, "V03 nearest spectrum"),
            (recorded.get("V13", {}).get("mean_signed_all_channel_dn"), recomputed_signed, "V13 signed mean"),
            (recorded.get("V13", {}).get("mean_signed_rec601_luma_dn"), recomputed_luma_signed, "V13 signed luma"),
            (recorded.get("V13", {}).get("display_first_mutant_mean_signed_dn"), recomputed_display_signed, "display-first signed mean"),
        )
        for observed, recomputed, label in comparisons:
            if not metric_matches(observed, recomputed):
                issues.append(f"{label} was not recomputed from bound raster arrays")
        mutations = recorded.get("mutation_discriminators", {})
        nearest_mutation = mutations.get("old_nearest", {})
        nearest_killed = recomputed_nearest_rms > 1.5 and recomputed_nearest_spectral > 1.10
        if nearest_mutation.get("killed") is not nearest_killed:
            issues.append("nearest mutation outcome differs from bound rasters")
        display_mutation = mutations.get("display_first", {})
        if display_mutation.get("killed") is not (abs(recomputed_display_signed) > 0.05):
            issues.append("display-first mutation outcome differs from bound rasters")

    verified = verify_m1_artifact_baseline(evidence.m1_baseline.directory)
    if verified.report_sha256 != evidence.m1_baseline.report_sha256:
        issues.append("M1 baseline object differs from reverified artifact bytes")
    if tuple(item.get("role") for item in evidence.object_identities) != m1.M1_REQUIRED_ROLES:
        issues.append("canonical four-object role ordering changed")
    for item in evidence.object_identities:
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            issues.append("canonical source object lacks valid SHA-256")
    return tuple(issues)


@dataclass(frozen=True, slots=True)
class M2ProductionEvaluation:
    oracle: m2.M2OracleRun
    canonical: CanonicalM2Evidence
    repeatability: RepeatabilityCapture | None
    sequence: SequenceEvaluation
    binding_issues: tuple[str, ...]

    @property
    def stage_advancement_passed(self) -> bool:
        return not self.binding_issues and self.oracle.stage_advancement_passed

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "schema": M2_PRODUCTION_SCHEMA,
            "stage_advancement_passed": self.stage_advancement_passed,
            "binding_issues": list(self.binding_issues),
            "canonical": self.canonical.summary_dict(),
            "repeatability": (
                None if self.repeatability is None else self.repeatability.summary_dict()
            ),
            "sequence": self.sequence.as_dict(),
            "oracle": self.oracle.as_dict(),
        }


def _canonical_gate(
    original: m2.GateResult,
    *,
    outcome: str,
    detail: str,
    measurements: Mapping[str, Any],
) -> m2.GateResult:
    return m2.GateResult(
        gate_id=original.gate_id,
        outcome=outcome,
        canonical_closure=outcome == "PASS",
        evidence_scope="canonical_bound_production_renderer",
        threshold=original.threshold,
        detail=detail,
        measurements=measurements,
        canonical_procedure=original.canonical_procedure,
        canonical_source_document=original.canonical_source_document,
    )


def evaluate_canonical_m2(
    synthetic_oracle: m2.M2OracleRun,
    evidence: CanonicalM2Evidence,
    *,
    repeatability: RepeatabilityCapture | None = None,
    sequence_binding: TimestampedSequenceBinding | None = None,
) -> M2ProductionEvaluation:
    """Replace non-closing oracle rows only with independently bound evidence."""

    binding_issues = list(canonical_evidence_binding_issues(evidence))
    metrics = evidence.metrics
    sequence = evaluate_timestamped_sequence(sequence_binding)
    originals = {row.gate_id: row for row in synthetic_oracle.gates}
    if not m2.M2_REQUIRED_GATE_IDS <= originals.keys():
        raise RuntimeError("synthetic oracle omitted required M2 gates")

    v01 = metrics["V01"]
    v02 = metrics["V02"]["candidate"]
    v03 = metrics["V03"]
    v08 = metrics["V08"]
    v13 = metrics["V13"]
    v14 = metrics["V14"]
    v15 = metrics["V15"]
    v16 = metrics["V16"]
    v41 = metrics["V41"]
    replacements: dict[str, m2.GateResult] = {}
    replacements["V01"] = _canonical_gate(
        originals["V01"],
        outcome="PASS" if float(v01["candidate_rms_dn"]) <= 1.5 else "FAIL",
        detail="Real pinned ABI scene versus captured-affine exact-area physical reference.",
        measurements=v01,
    )
    replacements["V02"] = _canonical_gate(
        originals["V02"],
        outcome=(
            "PASS"
            if int(v02["discarded_rows"]) == int(v02["discarded_columns"]) == 0
            else "FAIL"
        ),
        detail="Source support enumerated from measured Agg impulse responses.",
        measurements=metrics["V02"],
    )
    replacements["V03"] = _canonical_gate(
        originals["V03"],
        outcome="PASS" if float(v03["candidate_ratio"]) <= 1.10 else "FAIL",
        detail="Canonical Rec.601, 2-D Hann, 64-bin radial spectrum on delivered bytes.",
        measurements=v03,
    )
    v07_product_pass = bool(
        synthetic_oracle.product_color_protection.get("v07_passed")
    )
    replacements["V07"] = _canonical_gate(
        originals["V07"],
        outcome="PASS" if v07_product_pass and not binding_issues else "FAIL",
        detail="Protected constants plus immutable M1 artifact/repository binding.",
        measurements=metrics["V07"],
    )
    replacements["V08"] = _canonical_gate(
        originals["V08"],
        outcome="PASS" if bool(v08["passed"]) else "FAIL",
        detail="Five canonical pixels executed through the registered M2 finalizer and floor quantizer.",
        measurements=v08,
    )
    v13_pass = (
        not v13["trace_issues"]
        and abs(float(v13["mean_signed_all_channel_dn"])) <= 0.05
        and bool(v13["terminator_order_mutant"]["killed"])
        and not binding_issues
    )
    replacements["V13"] = _canonical_gate(
        originals["V13"],
        outcome="PASS" if v13_pass else "FAIL",
        detail=(
            "Runtime operation trace binds reflectance/K/coverage/source day-weight "
            "delivery before the sole CIRA call."
        ),
        measurements=v13,
    )
    replacements["V14"] = _canonical_gate(
        originals["V14"],
        outcome=(
            "PASS"
            if float(v14["maximum_abs_approved_relative_error"]) <= 0.005
            and v13_pass
            else "FAIL"
        ),
        detail="Closed-form contrast fixtures are bound to the captured physical-first trace.",
        measurements=v14,
    )
    replacements["V15"] = _canonical_gate(
        originals["V15"],
        outcome=(
            "PASS"
            if float(v15["maximum_out_of_range_fraction"]) == 0.0
            and evidence.m1_baseline.v15_out_of_range_fraction == 0.0
            else "FAIL"
        ),
        detail="All delivered physical fields use actual local Agg support, not global extrema.",
        measurements=v15,
    )
    replacements["V16"] = _canonical_gate(
        originals["V16"],
        outcome="PASS" if bool(v16["passed"]) else "FAIL",
        detail="Physical-first edge fixture compared with the verified M1 nearest baseline.",
        measurements=v16,
    )
    if repeatability is None:
        replacements["V25"] = _canonical_gate(
            originals["V25"],
            outcome="UNMEASURED",
            detail="Three independent renderer byte captures are absent.",
            measurements={"renderer_capture_binding": None},
        )
    else:
        repeatability_matches_primary = repeatability.matches_primary_png(
            evidence.primary_png
        )
        if not repeatability_matches_primary:
            binding_issues.append(
                "V25 repeatability renderer bytes differ from the canonical primary PNG"
            )
        replacements["V25"] = _canonical_gate(
            originals["V25"],
            outcome=(
                "PASS"
                if repeatability.passed and repeatability_matches_primary
                else "FAIL"
            ),
            detail="Parent recomputed SHA-256 from bytes returned by three distinct renderer processes.",
            measurements={
                **repeatability.summary_dict(),
                "canonical_primary_png": {
                    "bytes": len(evidence.primary_png),
                    "sha256": _sha256_bytes(evidence.primary_png),
                },
                "matches_canonical_primary_png": repeatability_matches_primary,
            },
        )
    replacements["V30"] = _canonical_gate(
        originals["V30"],
        outcome=sequence.outcome,
        detail=(
            "No metric summary is accepted; flicker is recomputed from bound timestamped arrays."
        ),
        measurements=sequence.as_dict(),
    )
    v41_outcome = (
        "PASS"
        if bool(v41.get("passed"))
        else ("UNMEASURED" if "unmeasured_reason" in v41 else "FAIL")
    )
    replacements["V41"] = _canonical_gate(
        originals["V41"],
        outcome=v41_outcome,
        detail="Real pinned full-disk objects rendered across the GOES limb with source-mask replay.",
        measurements=v41,
    )

    gates = tuple(replacements.get(row.gate_id, row) for row in synthetic_oracle.gates)
    production_mutations = metrics["mutation_discriminators"]
    output_proxy_mutation = production_mutations[
        "output_proxy_canonical_coverage"
    ]
    rollback_replay = metrics["M0_M1_ROLLBACK"]
    rollback_mutation = production_mutations[
        "current_m0_m1_rollback_bytes"
    ]
    repeatability_mutation = _repeatability_mutation_discriminator(repeatability)
    mutations = (
        *synthetic_oracle.mutations,
        m2.MutationResult(
            "canonical_old_nearest_behavior",
            bool(production_mutations["old_nearest"]["killed"]),
            "real-scene exact-area V01/V03",
            dict(production_mutations["old_nearest"]),
        ),
        m2.MutationResult(
            "canonical_display_first_behavior",
            bool(production_mutations["display_first"]["killed"]),
            "real-scene signed physical-domain V13",
            dict(production_mutations["display_first"]),
        ),
        m2.MutationResult(
            "resample_cos_then_nonlinear_day_weight",
            bool(production_mutations["resample_cos_then_nonlinear_day_weight"]["killed"]),
            "source-evaluated terminator taper through captured response",
            dict(production_mutations["resample_cos_then_nonlinear_day_weight"]),
        ),
        m2.MutationResult(
            "output_proxy_canonical_coverage",
            bool(output_proxy_mutation["killed"]),
            "V02/V41 require measured input support and source coverage",
            dict(output_proxy_mutation),
        ),
        m2.MutationResult(
            "caller_supplied_repeatability_hashes",
            bool(repeatability_mutation["killed"]),
            "V25 hashes are recomputed from child renderer bytes",
            dict(repeatability_mutation),
        ),
        m2.MutationResult(
            "current_m0_m1_rollback_bytes",
            bool(
                rollback_replay.get("passed") is True
                and rollback_mutation.get("killed") is True
            ),
            (
                "current real-corpus legacy compositor and registered M0/M1 "
                "renderer bytes versus immutable M1 hashes"
            ),
            dict(rollback_mutation),
        ),
    )
    closed = dataclasses.replace(
        synthetic_oracle,
        gates=gates,
        mutations=mutations,
    )
    return M2ProductionEvaluation(
        oracle=closed,
        canonical=evidence,
        repeatability=repeatability,
        sequence=sequence,
        binding_issues=tuple(binding_issues),
    )
