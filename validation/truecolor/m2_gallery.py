"""Deterministic, in-memory M2 scientific-review gallery payloads.

This module deliberately performs no filesystem writes.  It validates a
``CanonicalM2Evidence`` object, derives every review image and machine-readable
companion from its bound arrays, and returns immutable payload bytes plus
manifest-ready metadata.  The command-line publisher owns staging, fsync,
checksum-index creation, and atomic publication.

The gallery is a review aid, not an additional scientific oracle.  Recorded
gate metrics are independently recomputed here before their arrays are
accepted, and Andrew's visual verdict remains explicitly PENDING.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from . import m1, m2, m2_production


GALLERY_SCHEMA: Final = "tat_truecolor_m2_gallery_payloads/v1"
SPECTRAL_BINS: Final = 64
SPECTRAL_CUTOFF: Final = 0.3
ERROR_SCALE_PERCENTILE: Final = 99.9
ERROR_ORDER: Final = (
    "candidate",
    "old_nearest_mutant",
    "display_first_mutant",
)
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REC601 = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
_LIMITATIONS: Final = (
    "M2 primary imagery-layer authority is limited to ABI/AHI noncrossing, bounded, normal-DPI truecolor requests.",
    "M2 scalar authority is limited to noncrossing regular GridSat-B1 and GridSat-GOES archive scalar requests.",
    "Real canonical imagery, V41, and V30 evidence use ABI GOES-19; AHI authority is supported by typed product/mutation tests and Rayleigh preflight, not a bound real-AHI corpus render.",
    "GridSat-B1/GridSat-GOES scalar authority is supported by contract and integration fixtures, not a canonical external scalar object/render.",
    "MergIR remains on registered M1 because its legacy fetch may select an alternate slot after resolution; M2 does not claim its cache/source provenance.",
    "High-DPI truecolor remains on the registered M1 fallback and is not M2 evidence.",
    "Antimeridian/crossing truecolor is rejected by the M2 envelope; any separately registered crossing behavior remains outside M2 authority.",
    "Backdrop rendering remains on the registered M1 fallback.",
    "Non-archive or native-grid scalar rendering remains on the registered M1 fallback.",
    "btpng rendering remains on the registered M1 fallback.",
    "Unsupported bounding boxes or requests outside the M2 preflight envelope remain rejected or on explicitly registered M1 behavior.",
    "The M2 physical >1600 source cap is a separate upstream Hanning operation before Rayleigh and remains an M3 ordering residual.",
    "Accepted ratio self-sharpen remains an unchanged neighborhood spatial operator before CIRA and is parked to M4.",
    "S03 bilinear inter-band co-registration remains until M4.",
    "Encoded loop, S2 pyramid, Web-Mercator, and browser resizing remain until M5.",
    "V44, V45, and V46 are unmeasured and unclosed.",
)


@dataclass(frozen=True, slots=True)
class GalleryArtifact:
    """One payload declaration suitable for a later transactional manifest."""

    filename: str
    sha256: str
    bytes: int
    media_type: str
    width: int | None
    height: int | None
    description: str
    source_bindings: tuple[str, ...]
    source_binding_sha256: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "description": self.description,
            "source_bindings": list(self.source_bindings),
            "source_binding_sha256": dict(self.source_binding_sha256),
        }


@dataclass(frozen=True, slots=True)
class M2GalleryBundle:
    """Immutable payload map and JSON-ready manifest material.

    ``payloads`` maps safe sibling filenames to immutable ``bytes``.  The
    mapping and its nested metadata are read-only views.  ``manifest_dict``
    returns an ordinary deep JSON value for the external publisher.
    """

    payloads: Mapping[str, bytes]
    artifacts: Mapping[str, Mapping[str, Any]]
    manifest_metadata: Mapping[str, Any]

    def manifest_dict(self) -> dict[str, Any]:
        return _thaw(self.manifest_metadata)


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _safe_filename(filename: str) -> str:
    if (
        type(filename) is not str
        or filename in {"", ".", ".."}
        or not _SAFE_FILENAME.fullmatch(filename)
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise ValueError(f"unsafe M2 gallery filename: {filename!r}")
    return filename


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = _stable_json_bytes(
        {
            "dtype": contiguous.dtype.str,
            "shape": [int(item) for item in contiguous.shape],
        }
    )
    return _sha256(header + contiguous.tobytes(order="C"))


def _array_record(name: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(item) for item in array.shape],
        "dtype": array.dtype.str,
        "sha256": _array_sha256(array),
    }


def _record_payload(record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (record.get("shape"), record.get("dtype"), record.get("sha256"))


def _floor_dn(values: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def _metric_matches(recorded: Any, recomputed: float) -> bool:
    try:
        return math.isclose(
            float(recorded), float(recomputed), rel_tol=0.0, abs_tol=1.0e-12
        )
    except (TypeError, ValueError):
        return False


def _validate_optional_summary(
    value: Mapping[str, Any] | None,
    *,
    kind: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, None
    if type(value) is not dict:
        raise TypeError(f"{kind} summary must be an exact dict")
    try:
        payload = _stable_json_bytes(value)
        normalized = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{kind} summary is not finite canonical JSON") from exc
    if kind == "repeatability":
        if type(normalized.get("passed")) is not bool:
            raise ValueError("repeatability summary must contain an exact passed boolean")
        hashes = normalized.get("hashes")
        if not isinstance(hashes, list) or len(hashes) != 3:
            raise ValueError("repeatability summary must bind exactly three hashes")
        if any(
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in hashes
        ):
            raise ValueError("repeatability summary contains an invalid SHA-256")
    elif kind == "sequence":
        if normalized.get("outcome") not in {"PASS", "FAIL", "UNMEASURED"}:
            raise ValueError("sequence summary has no registered outcome")
    else:  # pragma: no cover - internal call-site guard
        raise ValueError(f"unknown optional M2 summary kind {kind!r}")
    return normalized, _sha256(payload)


def _validated_evidence(
    evidence: m2_production.CanonicalM2Evidence,
) -> tuple[dict[str, Any], dict[str, np.ndarray], bytes]:
    """Fail closed on proxy, dtype, shape, metric, and capture-hash drift."""

    if type(evidence) is not m2_production.CanonicalM2Evidence:
        raise TypeError("M2 gallery requires an exact CanonicalM2Evidence object")
    if type(evidence.m1_baseline) is not m2_production.M1ArtifactBaseline:
        raise TypeError("M2 gallery requires an exact verified M1 baseline record")
    if evidence.scene_id != m2_production.M2_SCENE_ID:
        raise RuntimeError("M2 gallery scene is not the canonical Melissa scene")
    if tuple(evidence.bbox) != tuple(m2_production.M2_CANONICAL_BBOX):
        raise RuntimeError("M2 gallery bbox is not the canonical primary bbox")
    if type(evidence.primary_png) is not bytes or not evidence.primary_png.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        raise TypeError("M2 production renderer payload must be exact PNG bytes")

    arrays = {
        "candidate_float": evidence.candidate_float,
        "candidate_floor": evidence.candidate_floor,
        "exact_area_reference_float": evidence.exact_area_reference_float,
        "exact_area_reference_floor": evidence.exact_area_reference_floor,
        "old_nearest_mutant_floor": evidence.old_nearest_mutant_floor,
        "display_first_mutant_float": evidence.display_first_mutant_float,
        "display_first_mutant_floor": evidence.display_first_mutant_floor,
    }
    float_names = {
        "candidate_float",
        "exact_area_reference_float",
        "display_first_mutant_float",
    }
    shape: tuple[int, int, int] | None = None
    for name, array in arrays.items():
        if type(array) is not np.ndarray:
            raise TypeError(f"{name} must be an exact plain ndarray")
        expected_dtype = np.dtype(np.float32 if name in float_names else np.uint8)
        if array.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
        if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) < 2:
            raise ValueError(f"{name} must be a nontrivial HxWx3 raster")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be a canonical C-contiguous raster")
        if shape is None:
            shape = tuple(int(item) for item in array.shape)
        elif tuple(array.shape) != shape:
            raise ValueError("M2 candidate/reference/mutant raster shapes differ")
        if name in float_names and (
            not np.all(np.isfinite(array))
            or np.any((array < 0.0) | (array > 1.0))
        ):
            raise ValueError(f"{name} must be finite display RGB in [0, 1]")

    if not np.array_equal(arrays["candidate_floor"], _floor_dn(arrays["candidate_float"])):
        raise RuntimeError("candidate floor is not derived from candidate float")
    if not np.array_equal(
        arrays["exact_area_reference_floor"],
        _floor_dn(arrays["exact_area_reference_float"]),
    ):
        raise RuntimeError("exact-area floor is not derived from reference float")
    if not np.array_equal(
        arrays["display_first_mutant_floor"],
        _floor_dn(arrays["display_first_mutant_float"]),
    ):
        raise RuntimeError("display-first floor is not derived from mutant float")

    summary = evidence.summary_dict()
    if type(summary) is not dict or summary.get("schema") != m2_production.M2_PRODUCTION_SCHEMA:
        raise RuntimeError("canonical evidence summary has an unregistered schema")
    summary_arrays = summary.get("arrays")
    if type(summary_arrays) is not dict:
        raise RuntimeError("canonical evidence summary omitted array identities")
    for name, array in arrays.items():
        expected = _array_record(name, array)
        if summary_arrays.get(name) != expected:
            raise RuntimeError(f"{name} does not match evidence.summary_dict()")

    primary = summary.get("primary_capture")
    if type(primary) is not dict:
        raise RuntimeError("canonical evidence omitted its production capture")
    if (
        primary.get("schema") != m2_production.M2_PRODUCTION_SCHEMA
        or primary.get("policy_version") != m2_production.M2_REQUIRED_POLICY_VERSION
        or primary.get("bbox") != list(evidence.bbox)
        or primary.get("trace_issues") != []
    ):
        raise RuntimeError("production capture identity/trace is not canonical M2")
    for capture_name, array_name in (
        ("delivered_float_rgb", "candidate_float"),
        ("delivered_floor_rgb", "candidate_floor"),
        ("figure_image_rgb", "candidate_floor"),
    ):
        record = primary.get(capture_name)
        if type(record) is not dict or _record_payload(record) != _record_payload(
            summary_arrays[array_name]
        ):
            raise RuntimeError(
                f"production capture {capture_name} is not bound to {array_name}"
            )
    png_record = summary.get("primary_png")
    if (
        type(png_record) is not dict
        or png_record.get("bytes") != len(evidence.primary_png)
        or png_record.get("sha256") != _sha256(evidence.primary_png)
        or primary.get("png_bytes") != len(evidence.primary_png)
        or primary.get("png_sha256") != _sha256(evidence.primary_png)
    ):
        raise RuntimeError("production PNG is not cross-bound by canonical evidence")

    object_rows = summary.get("objects")
    if not isinstance(object_rows, list) or tuple(
        row.get("role") if isinstance(row, dict) else None for row in object_rows
    ) != m1.M1_REQUIRED_ROLES:
        raise RuntimeError("canonical source-object role ordering is absent")
    for row in object_rows:
        digest = row.get("sha256")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("canonical source object has no valid SHA-256")

    metrics = summary.get("metrics")
    if type(metrics) is not dict:
        raise RuntimeError("canonical M2 metrics are absent")
    candidate_floor = arrays["candidate_floor"]
    reference_floor = arrays["exact_area_reference_floor"]
    nearest_floor = arrays["old_nearest_mutant_floor"]
    candidate_float = arrays["candidate_float"]
    reference_float = arrays["exact_area_reference_float"]
    display_float = arrays["display_first_mutant_float"]
    recomputed = {
        "candidate_rms": m2.rms_dn(candidate_floor, reference_floor),
        "nearest_rms": m2.rms_dn(nearest_floor, reference_floor),
        "candidate_spectral": m2.radial_high_frequency_power_ratio(
            candidate_floor,
            reference_floor,
            cutoff=SPECTRAL_CUTOFF,
            bins=SPECTRAL_BINS,
        ),
        "nearest_spectral": m2.radial_high_frequency_power_ratio(
            nearest_floor,
            reference_floor,
            cutoff=SPECTRAL_CUTOFF,
            bins=SPECTRAL_BINS,
        ),
        "candidate_signed": float(
            np.mean(
                (candidate_float.astype(np.float64) - reference_float.astype(np.float64))
                * 255.0
            )
        ),
        "candidate_luma_signed": float(
            np.mean(
                np.tensordot(
                    candidate_float.astype(np.float64)
                    - reference_float.astype(np.float64),
                    _REC601,
                    axes=([-1], [0]),
                )
                * 255.0
            )
        ),
        "display_signed": float(
            np.mean(
                (display_float.astype(np.float64) - reference_float.astype(np.float64))
                * 255.0
            )
        ),
    }
    v01 = metrics.get("V01", {})
    v03 = metrics.get("V03", {})
    v13 = metrics.get("V13", {})
    comparisons = (
        (v01.get("candidate_rms_dn"), recomputed["candidate_rms"], "V01 candidate RMS"),
        (v01.get("old_nearest_rms_dn"), recomputed["nearest_rms"], "V01 nearest RMS"),
        (v03.get("candidate_ratio"), recomputed["candidate_spectral"], "V03 candidate ratio"),
        (v03.get("old_nearest_ratio"), recomputed["nearest_spectral"], "V03 nearest ratio"),
        (v13.get("mean_signed_all_channel_dn"), recomputed["candidate_signed"], "V13 signed mean"),
        (v13.get("mean_signed_rec601_luma_dn"), recomputed["candidate_luma_signed"], "V13 signed luma"),
        (v13.get("display_first_mutant_mean_signed_dn"), recomputed["display_signed"], "display-first signed mean"),
    )
    for recorded, actual, label in comparisons:
        if not _metric_matches(recorded, actual):
            raise RuntimeError(f"{label} is not bound to canonical evidence arrays")
    if (
        v01.get("reference_quantizer") != "production_floor_u8_no_dither"
        or v03.get("luma") != "Rec.601"
        or v03.get("spatial_window") != "separable_2D_Hann"
        or v03.get("radial_bins") != SPECTRAL_BINS
        or not _metric_matches(v03.get("cutoff_cycles_per_pixel"), SPECTRAL_CUTOFF)
    ):
        raise RuntimeError("canonical V01/V03 procedure declaration drifted")

    summary_payload = _stable_json_bytes(summary)
    return summary, arrays, summary_payload


def _png_bytes(array: np.ndarray) -> bytes:
    from PIL import Image

    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            return (int(image.width), int(image.height))
    except Exception as exc:
        raise RuntimeError("production renderer payload is not a decodable PNG") from exc


def _comparison_sheet(
    panels: Sequence[tuple[str, np.ndarray]],
) -> tuple[bytes, tuple[int, int], list[dict[str, Any]]]:
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("comparison sheet requires at least one panel")
    height, width = panels[0][1].shape[:2]
    if any(panel.shape != (height, width, 3) for _, panel in panels):
        raise ValueError("comparison sheet panels must share one HxWx3 shape")
    header = 52
    sheet = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(sheet)
    bounds: list[dict[str, Any]] = []
    for index, (label, array) in enumerate(panels):
        left = index * width
        draw.text((left + 8, 18), label, fill="black")
        sheet.paste(Image.fromarray(array, mode="RGB"), (left, header))
        bounds.append(
            {
                "label": label,
                "bounds_xyxy": [left, header, left + width, header + height],
            }
        )
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue(), sheet.size, bounds


def _signed_rec601_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    # The machine-readable authority is the float32 NPY.  Canonicalise once
    # here so its percentile reproduces the declared shared scale exactly;
    # deriving the scale from a hidden float64 precursor would make the PNG
    # legend impossible to recompute bit-for-bit from the published values.
    return np.tensordot(difference, _REC601, axes=([-1], [0])).astype(np.float32)


def _canonical_error_scale(errors: Sequence[np.ndarray]) -> float:
    if not errors:
        raise ValueError("shared error scale requires at least one field")
    values = [float(np.percentile(np.abs(error), ERROR_SCALE_PERCENTILE)) for error in errors]
    scale = max(1.0, *values)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("canonical shared signed-error scale is invalid")
    return scale


def _signed_error_map(error: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(np.asarray(error, dtype=np.float64) / scale, -1.0, 1.0)
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


def _error_npy_bytes(errors: Sequence[np.ndarray]) -> bytes:
    stack = np.ascontiguousarray(np.stack(errors).astype(np.float32))
    buffer = io.BytesIO()
    np.save(buffer, stack, allow_pickle=False)
    return buffer.getvalue()


def _error_summary_csv(errors: Mapping[str, np.ndarray], scale: float) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "path",
            "shared_symmetric_scale_dn_p99_9_max",
            "mean_signed_dn",
            "rms_dn",
            "min_dn",
            "max_dn",
            "p99_abs_dn",
            "p99_9_abs_dn",
        )
    )
    for name in ERROR_ORDER:
        values = errors[name].astype(np.float64)
        absolute = np.abs(values)
        writer.writerow(
            (
                name,
                f"{scale:.17g}",
                f"{np.mean(values):.17g}",
                f"{np.sqrt(np.mean(np.square(values))):.17g}",
                f"{np.min(values):.17g}",
                f"{np.max(values):.17g}",
                f"{np.percentile(absolute, 99):.17g}",
                f"{np.percentile(absolute, ERROR_SCALE_PERCENTILE):.17g}",
            )
        )
    return buffer.getvalue().encode("ascii")


def _spectral_profiles(
    rasters: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frequency: np.ndarray | None = None
    profiles: dict[str, np.ndarray] = {}
    for name, raster in rasters.items():
        observed_frequency, power = m2.canonical_radial_power_profile(
            raster, bins=SPECTRAL_BINS
        )
        if frequency is None:
            frequency = observed_frequency
        elif not np.array_equal(frequency, observed_frequency):
            raise RuntimeError("canonical spectral frequencies changed between rasters")
        profiles[name] = power
    assert frequency is not None
    return frequency, profiles


def _spectral_csv(
    frequency: np.ndarray,
    profiles: Mapping[str, np.ndarray],
) -> bytes:
    order = ("candidate", "exact_area_reference", "old_nearest_mutant", "display_first_mutant")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "radial_frequency_cycles_per_delivered_pixel",
            "candidate_luma_power",
            "exact_area_reference_luma_power",
            "old_nearest_mutant_luma_power",
            "display_first_mutant_luma_power",
        )
    )
    for index, value in enumerate(frequency):
        writer.writerow(
            (f"{value:.17g}", *(f"{profiles[name][index]:.17g}" for name in order))
        )
    return buffer.getvalue().encode("ascii")


def _spectral_plot(
    frequency: np.ndarray,
    profiles: Mapping[str, np.ndarray],
    ratios: Mapping[str, float],
) -> bytes:
    """Draw a deterministic plot of already-canonical profile values."""

    from PIL import Image, ImageDraw

    width, height = 1200, 720
    left, top, right, bottom = 92, 86, 1160, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 18), "M2 canonical V03 spectrum", fill="black")
    draw.text(
        (left, 39),
        "Rec.601 luma; square crop; 2-D Hann; 64 radial bins; cutoff >0.3 cyc/pixel",
        fill="black",
    )
    draw.line((left, top, left, bottom), fill="black", width=1)
    draw.line((left, bottom, right, bottom), fill="black", width=1)

    finite_logs: list[np.ndarray] = []
    for power in profiles.values():
        positive = np.asarray(power, dtype=np.float64)
        finite = np.isfinite(positive) & (positive > 0.0)
        finite_logs.append(np.log10(positive[finite]))
    joined = np.concatenate([item for item in finite_logs if item.size])
    y_min = float(np.min(joined))
    y_max = float(np.max(joined))
    if not y_max > y_min:
        y_max = y_min + 1.0

    cutoff_x = left + int(round((right - left) * SPECTRAL_CUTOFF / 0.5))
    draw.line((cutoff_x, top, cutoff_x, bottom), fill=(80, 80, 80), width=1)
    draw.text((cutoff_x + 5, top + 4), ">0.3 integration", fill=(80, 80, 80))
    colors = {
        "candidate": (35, 100, 170),
        "exact_area_reference": (42, 157, 143),
        "old_nearest_mutant": (193, 18, 31),
        "display_first_mutant": (230, 126, 34),
    }
    labels = {
        "candidate": "M2 candidate",
        "exact_area_reference": "Exact-area reference",
        "old_nearest_mutant": "Old nearest mutant",
        "display_first_mutant": "Display-first mutant",
    }
    legend_y = 645
    for legend_index, name in enumerate(labels):
        power = profiles[name]
        points: list[tuple[int, int]] = []
        for x_value, y_value in zip(frequency, power, strict=True):
            if not np.isfinite(y_value) or y_value <= 0.0:
                continue
            x_pixel = left + int(round((right - left) * float(x_value) / 0.5))
            fraction = (math.log10(float(y_value)) - y_min) / (y_max - y_min)
            y_pixel = bottom - int(round((bottom - top) * fraction))
            points.append((x_pixel, y_pixel))
        if len(points) >= 2:
            draw.line(points, fill=colors[name], width=2)
        legend_x = left + legend_index * 265
        draw.line((legend_x, legend_y, legend_x + 28, legend_y), fill=colors[name], width=3)
        suffix = ""
        if name in ratios:
            suffix = f" ({ratios[name]:.6g}x)"
        draw.text((legend_x + 34, legend_y - 7), labels[name] + suffix, fill="black")
    draw.text((right - 180, bottom + 18), "cycles / delivered pixel", fill="black")
    draw.text((8, top), "log10 mean power", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def build_m2_gallery(
    evidence: m2_production.CanonicalM2Evidence,
    *,
    repeatability_summary: Mapping[str, Any] | None = None,
    sequence_summary: Mapping[str, Any] | None = None,
    expected_shared_error_scale_dn: float | None = None,
    expected_evidence_summary_sha256: str | None = None,
) -> M2GalleryBundle:
    """Build all M2 gallery files in memory after strict evidence validation."""

    summary, arrays, evidence_summary_payload = _validated_evidence(evidence)
    evidence_summary_sha256 = _sha256(evidence_summary_payload)
    if expected_evidence_summary_sha256 is not None:
        if (
            type(expected_evidence_summary_sha256) is not str
            or expected_evidence_summary_sha256 != evidence_summary_sha256
        ):
            raise RuntimeError("canonical evidence summary SHA-256 does not match caller binding")
    repeatability, repeatability_sha256 = _validate_optional_summary(
        repeatability_summary, kind="repeatability"
    )
    sequence, sequence_sha256 = _validate_optional_summary(
        sequence_summary, kind="sequence"
    )

    candidate = arrays["candidate_floor"]
    reference = arrays["exact_area_reference_floor"]
    nearest = arrays["old_nearest_mutant_floor"]
    display_first = arrays["display_first_mutant_floor"]
    height, width = candidate.shape[:2]

    errors = {
        "candidate": _signed_rec601_error(candidate, reference),
        "old_nearest_mutant": _signed_rec601_error(nearest, reference),
        "display_first_mutant": _signed_rec601_error(display_first, reference),
    }
    shared_scale = _canonical_error_scale(tuple(errors[name] for name in ERROR_ORDER))
    if expected_shared_error_scale_dn is not None:
        try:
            supplied_scale = float(expected_shared_error_scale_dn)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected shared error scale must be numeric") from exc
        if not math.isclose(supplied_scale, shared_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "noncanonical signed-error scale: expected "
                f"{shared_scale:.17g}, got {supplied_scale:.17g}"
            )
    error_maps = {
        name: _signed_error_map(errors[name], shared_scale) for name in ERROR_ORDER
    }

    payloads: dict[str, bytes] = {}
    declarations: dict[str, dict[str, Any]] = {}

    def add(
        filename: str,
        payload: bytes,
        *,
        media_type: str,
        artifact_width: int | None,
        artifact_height: int | None,
        description: str,
        source_bindings: Sequence[str],
    ) -> None:
        safe = _safe_filename(filename)
        if safe in payloads:
            raise RuntimeError(f"duplicate M2 gallery payload {safe}")
        if type(payload) is not bytes or not payload:
            raise TypeError(f"M2 gallery payload {safe} must be non-empty bytes")
        immutable = bytes(payload)
        binding_hashes: dict[str, str] = {}
        for binding in source_bindings:
            if binding in summary["arrays"]:
                binding_hashes[binding] = summary["arrays"][binding]["sha256"]
            elif binding == "primary_png":
                binding_hashes[binding] = summary["primary_png"]["sha256"]
            elif binding == "evidence_summary":
                binding_hashes[binding] = evidence_summary_sha256
            else:
                raise RuntimeError(f"unknown M2 gallery source binding {binding!r}")
        payloads[safe] = immutable
        declarations[safe] = GalleryArtifact(
            filename=safe,
            sha256=_sha256(immutable),
            bytes=len(immutable),
            media_type=media_type,
            width=artifact_width,
            height=artifact_height,
            description=description,
            source_bindings=tuple(source_bindings),
            source_binding_sha256=binding_hashes,
        ).as_dict()

    native_specs = (
        (
            "01_m2_candidate_native.png",
            candidate,
            "candidate_floor",
            "Registered M2 delivered map raster at exact native inspection scale.",
        ),
        (
            "02_exact_area_reference_native.png",
            reference,
            "exact_area_reference_floor",
            "Captured-affine exact-area physical reference at exact native inspection scale.",
        ),
        (
            "03_old_nearest_mutant_native.png",
            nearest,
            "old_nearest_mutant_floor",
            "Rejected old nearest-neighbour behavior at exact native inspection scale.",
        ),
        (
            "04_display_first_mutant_native.png",
            display_first,
            "display_first_mutant_floor",
            "Rejected display-domain-first behavior at exact native inspection scale.",
        ),
    )
    for filename, raster, binding, description in native_specs:
        add(
            filename,
            _png_bytes(raster),
            media_type="image/png",
            artifact_width=width,
            artifact_height=height,
            description=description,
            source_bindings=(binding,),
        )

    for index, name in enumerate(ERROR_ORDER, start=5):
        add(
            f"{index:02d}_{name}_signed_rec601_error_native.png",
            _png_bytes(error_maps[name]),
            media_type="image/png",
            artifact_width=width,
            artifact_height=height,
            description=(
                f"Signed Rec.601 luma error ({name} minus exact-area reference); "
                f"red positive, blue negative, shared symmetric scale {shared_scale:.6g} DN."
            ),
            source_bindings=(
                {
                    "candidate": "candidate_floor",
                    "old_nearest_mutant": "old_nearest_mutant_floor",
                    "display_first_mutant": "display_first_mutant_floor",
                }[name],
                "exact_area_reference_floor",
            ),
        )

    main_sheet, main_sheet_size, main_bounds = _comparison_sheet(
        (
            ("M2 candidate", candidate),
            ("Exact-area reference", reference),
            ("Old nearest mutant", nearest),
            ("Display-first mutant", display_first),
        )
    )
    add(
        "08_native_comparison_sheet_100pct.png",
        main_sheet,
        media_type="image/png",
        artifact_width=main_sheet_size[0],
        artifact_height=main_sheet_size[1],
        description="Four labeled native rasters embedded pixel-for-pixel at 100% scale.",
        source_bindings=(
            "candidate_floor",
            "exact_area_reference_floor",
            "old_nearest_mutant_floor",
            "display_first_mutant_floor",
        ),
    )
    error_sheet, error_sheet_size, error_bounds = _comparison_sheet(
        tuple((name.replace("_", " "), error_maps[name]) for name in ERROR_ORDER)
    )
    add(
        "09_signed_error_sheet_shared_scale_100pct.png",
        error_sheet,
        media_type="image/png",
        artifact_width=error_sheet_size[0],
        artifact_height=error_sheet_size[1],
        description="Three signed-error maps embedded pixel-for-pixel with one shared scale.",
        source_bindings=(
            "candidate_floor",
            "old_nearest_mutant_floor",
            "display_first_mutant_floor",
            "exact_area_reference_floor",
        ),
    )
    add(
        "10_signed_rec601_error_values_float32.npy",
        _error_npy_bytes(tuple(errors[name] for name in ERROR_ORDER)),
        media_type="application/x-npy",
        artifact_width=width,
        artifact_height=height,
        description=(
            "Raw float32 signed Rec.601 DN errors with order candidate, old-nearest, "
            "display-first and shape [3,H,W]."
        ),
        source_bindings=(
            "candidate_floor",
            "old_nearest_mutant_floor",
            "display_first_mutant_floor",
            "exact_area_reference_floor",
        ),
    )
    add(
        "11_signed_rec601_error_summary.csv",
        _error_summary_csv(errors, shared_scale),
        media_type="text/csv",
        artifact_width=None,
        artifact_height=None,
        description="Machine-readable signed Rec.601 error statistics and canonical shared scale.",
        source_bindings=(
            "candidate_floor",
            "old_nearest_mutant_floor",
            "display_first_mutant_floor",
            "exact_area_reference_floor",
        ),
    )

    spectral_rasters = {
        "candidate": candidate,
        "exact_area_reference": reference,
        "old_nearest_mutant": nearest,
        "display_first_mutant": display_first,
    }
    frequency, profiles = _spectral_profiles(spectral_rasters)
    ratios = {
        name: m2.radial_high_frequency_power_ratio(
            raster,
            reference,
            cutoff=SPECTRAL_CUTOFF,
            bins=SPECTRAL_BINS,
        )
        for name, raster in spectral_rasters.items()
        if name != "exact_area_reference"
    }
    add(
        "12_canonical_spectral_comparison.csv",
        _spectral_csv(frequency, profiles),
        media_type="text/csv",
        artifact_width=None,
        artifact_height=None,
        description="Canonical Rec.601, square-crop, 2-D-Hann, 64-bin radial spectra.",
        source_bindings=(
            "candidate_floor",
            "exact_area_reference_floor",
            "old_nearest_mutant_floor",
            "display_first_mutant_floor",
        ),
    )
    add(
        "13_canonical_spectral_comparison.png",
        _spectral_plot(frequency, profiles, ratios),
        media_type="image/png",
        artifact_width=1200,
        artifact_height=720,
        description="Plot of the exact canonical spectral profiles stored in the companion CSV.",
        source_bindings=(
            "candidate_floor",
            "exact_area_reference_floor",
            "old_nearest_mutant_floor",
            "display_first_mutant_floor",
        ),
    )

    production_width, production_height = _png_dimensions(evidence.primary_png)
    add(
        "14_production_renderer.png",
        evidence.primary_png,
        media_type="image/png",
        artifact_width=production_width,
        artifact_height=production_height,
        description="Exact PNG bytes returned by the bound primary M2 production renderer.",
        source_bindings=("primary_png", "candidate_floor"),
    )

    repeatability_text = (
        "not supplied to this gallery builder"
        if repeatability is None
        else f"supplied; passed={repeatability['passed']}; summary SHA-256={repeatability_sha256}"
    )
    sequence_text = (
        "not supplied to this gallery builder"
        if sequence is None
        else f"supplied; outcome={sequence['outcome']}; summary SHA-256={sequence_sha256}"
    )
    limitation_text = "\n".join(f"- {item}" for item in _LIMITATIONS)
    review = (
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
    add(
        "REVIEW_INSTRUCTIONS.md",
        review,
        media_type="text/markdown",
        artifact_width=None,
        artifact_height=None,
        description="Immutable append-only review procedure; Andrew remains PENDING.",
        source_bindings=("evidence_summary",),
    )
    readme = (
        "# M2 primary-imagery-layer review payloads\n\n"
        "Andrew verdict: PENDING. This gallery is not itself a stage-advancement "
        "decision and cannot substitute for the canonical gate report.\n\n"
        f"Scene: `{evidence.scene_id}`\n\n"
        f"Native delivered raster: `{width} x {height}` pixels\n\n"
        f"Canonical evidence summary SHA-256: `{evidence_summary_sha256}`\n\n"
        f"Repeatability summary: {repeatability_text}.\n\n"
        f"Timestamped sequence summary: {sequence_text}.\n\n"
        "Every standalone scientific raster is lossless and native scale. The two "
        "comparison sheets add labels only outside the embedded raster bounds. The "
        "production PNG is copied byte-for-byte rather than decoded and re-encoded.\n\n"
        "Exact limitations:\n\n"
        f"{limitation_text}\n"
    ).encode("utf-8")
    add(
        "README.md",
        readme,
        media_type="text/markdown",
        artifact_width=None,
        artifact_height=None,
        description="Gallery index, source identity, optional evidence status, and exact limits.",
        source_bindings=("evidence_summary",),
    )

    payload_index = {
        name: {"sha256": declarations[name]["sha256"], "bytes": declarations[name]["bytes"]}
        for name in sorted(declarations)
    }
    summary_arrays = summary["arrays"]
    manifest = {
        "schema": GALLERY_SCHEMA,
        "scene_id": evidence.scene_id,
        "bbox": list(evidence.bbox),
        "native_shape": [height, width, 3],
        "source_evidence": {
            "schema": summary["schema"],
            "summary_sha256": evidence_summary_sha256,
            "array_records": summary_arrays,
            "primary_png": summary["primary_png"],
            "production_policy": summary["primary_capture"]["policy_version"],
            "source_objects": summary["objects"],
        },
        "artifacts": {name: declarations[name] for name in sorted(declarations)},
        "payload_index_sha256": _sha256(_stable_json_bytes(payload_index)),
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
            "raw_values_shape": [len(ERROR_ORDER), height, width],
            "raw_values_order": list(ERROR_ORDER),
            "shared_scale_percentile": ERROR_SCALE_PERCENTILE,
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
            "plot": "13_canonical_spectral_comparison.png",
            "csv": "12_canonical_spectral_comparison.csv",
            "procedure": "Rec.601_square_crop_2D_Hann_64_radial_bins_v1",
            "bins": SPECTRAL_BINS,
            "cutoff_cycles_per_delivered_pixel": SPECTRAL_CUTOFF,
            "ratios_to_exact_area_reference": ratios,
        },
        "repeatability": {
            "present": repeatability is not None,
            "summary_sha256": repeatability_sha256,
            "summary": repeatability,
        },
        "sequence": {
            "present": sequence is not None,
            "summary_sha256": sequence_sha256,
            "summary": sequence,
        },
        "andrew_review": {
            "verdict": "PENDING",
            "instructions": "REVIEW_INSTRUCTIONS.md",
            "supplement_pattern": "ANDREW_VERDICT_<UTC_TIMESTAMP>.md",
            "supplement_must_bind_final_gallery_manifest_sha256": True,
            "published_payloads_are_immutable": True,
        },
        "claim_boundary": {
            "stage_advancement_claimed_by_gallery": False,
            "scope": "M2 primary imagery-layer finalizer only",
            "limitations": list(_LIMITATIONS),
        },
    }

    # Detect concurrent or accidental mutation after payload derivation.  The
    # future publisher can perform the same check immediately before rename.
    if _stable_json_bytes(evidence.summary_dict()) != evidence_summary_payload:
        raise RuntimeError("canonical M2 evidence changed while building the gallery")
    frozen_payloads = MappingProxyType({name: payloads[name] for name in sorted(payloads)})
    frozen_declarations = MappingProxyType(
        {name: _freeze(declarations[name]) for name in sorted(declarations)}
    )
    return M2GalleryBundle(
        payloads=frozen_payloads,
        artifacts=frozen_declarations,
        manifest_metadata=_freeze(manifest),
    )


def build_m2_gallery_payloads(
    evidence: m2_production.CanonicalM2Evidence,
    **kwargs: Any,
) -> M2GalleryBundle:
    """Explicitly named alias used by the future transactional publisher."""

    return build_m2_gallery(evidence, **kwargs)
