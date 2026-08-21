"""Independent scientific oracle for the bounded M2 delivery-domain change.

This module is deliberately not imported by product code.  It provides small,
closed-form reference operators and mutation discriminators for the M2 gates.
Synthetic contract evidence proves that a procedure can reject a known bad
algorithm; it never substitutes for a captured production-renderer binding.

The M2 claim is intentionally narrower than the eventual M5 claim: there is no
post-CIRA spatial operation *inside the primary imagery-layer finalizer*.
Encoded loop, explorer-pyramid, Web-Mercator, and browser resizes remain
enumerated M5 residuals.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

from . import production_adapter
from .gates import ResearchInputs
from .reference_pixels import (
    DISPLAY_ATOL,
    PROTECTED_CONSTANTS,
    REFERENCE_PIXELS,
    REFLECTANCE_ATOL,
    expected_dn,
    quantize_display,
)


M2_ORACLE_SCHEMA: Final = "tat_truecolor_m2_oracle/v2"
M2_REPORT_SCHEMA: Final = "tat_truecolor_m2_validation/v2"
M2_TRACE_SCHEMA: Final = "tat_truecolor_m2_renderer_trace/v2"
M2_CAPTURE_METHOD: Final = "axesimage_delivered_device_affine_runtime_capture_v2"

M2_ASSIGNED_GATE_IDS: Final[frozenset[str]] = frozenset(
    {"V01", "V02", "V03", "V13", "V14", "V41"}
)
M2_PROTECTION_GATE_IDS: Final[frozenset[str]] = frozenset(
    {"V07", "V08", "V15", "V16", "V25", "V30"}
)
M2_REQUIRED_GATE_IDS: Final[frozenset[str]] = (
    M2_ASSIGNED_GATE_IDS | M2_PROTECTION_GATE_IDS
)
M5_DEFERRED_GATE_IDS: Final[frozenset[str]] = frozenset(
    {"V21", "V22", "V23", "V24"}
)
FORBIDDEN_M2_CLOSURE_GATE_IDS: Final[frozenset[str]] = frozenset(
    {"V44", "V45", "V46"}
)
M2_ALL_GATE_IDS: Final[frozenset[str]] = (
    M2_REQUIRED_GATE_IDS | M5_DEFERRED_GATE_IDS | FORBIDDEN_M2_CLOSURE_GATE_IDS
)
M2_SYNTHETIC_MUTATION_IDS: Final[frozenset[str]] = frozenset(
    {
        "m1_rollback_policy_bypass",
        "nearest_neighbor",
        "auto_nearest_fallback",
        "cira_before_resample",
        "direct_gray_as_reflectance",
        "old_display_space_day_night_blend",
        "night_preimage_zero_endpoint",
        "inverse_before_bt_resample",
        "terminator_taper_after_cosine_resample",
        "nan_fill_black",
        "nan_times_zero",
        "nearest_mask",
        "proxy_or_unwindowed_spectrum",
        "rec709_spectrum_proxy",
        "wrong_spectrum_bin_count",
        "global_extrema_range_proxy",
        "weak_global_ringing",
        "hard_coded_geometry_offset",
        "single_isotropic_ratio",
        "encoded_rgb_physical_boundary",
        "display_domain_spatial_operation",
        "glint_before_authoritative_resample",
        "label_only_operation_trace",
        "unbound_delivery_kernel",
        "unbound_delivery_phase",
        "unbound_delivery_extent",
        "proxy_coverage_transform",
        "inverse_before_kelvin_delivery_trace",
        "palette_before_scalar_resample",
        "m6_terminator_leakage",
        "false_global_one_raster_claim",
    }
)
M2_PRODUCTION_MUTATION_IDS: Final[frozenset[str]] = frozenset(
    {
        "canonical_old_nearest_behavior",
        "canonical_display_first_behavior",
        "resample_cos_then_nonlinear_day_weight",
        "output_proxy_canonical_coverage",
        "caller_supplied_repeatability_hashes",
        "current_m0_m1_rollback_bytes",
    }
)
M2_CANONICAL_MUTATION_IDS: Final[frozenset[str]] = (
    M2_SYNTHETIC_MUTATION_IDS | M2_PRODUCTION_MUTATION_IDS
)
M5_SURVIVING_ENCODED_RESIZES: Final[tuple[str, ...]] = (
    "render.transcode_frame:Pillow-Lanczos encoded loop resize",
    "s2_pyramid.cut_pyramid:Pillow-Lanczos encoded RGBA pyramid resize",
    "s2_webmerc.reproject_tile:encoded RGBA Web-Mercator resample",
    "browser-facing CSS sizing may resample delivered encoded pixels",
)

CIRA_LOG_ROOT_EXPECTED: Final = -1.6516951369518393
CIRA_DENOM_EXPECTED: Final = 1.9887713527138795
CIRA_BLACK_PREIMAGE_EXPECTED: Final = 0.0223
CIRA_WHITE_PREIMAGE_EXPECTED: Final = 2.1730825067739694
IR_T_WARM_K: Final = 303.0
IR_T_COLD_K: Final = 183.0
COVERAGE_TAU: Final = 1e-6

PHYSICAL_DELIVERY_OPERATION_KIND: Final = "physical_delivery_resample"
PHYSICAL_DELIVERY_KERNEL: Final = "hanning"
PHYSICAL_COVERAGE_TRANSFORM: Final = (
    "premultiplied_values_and_linear_coverage_same_transform"
)

CANONICAL_SPATIAL_DOMAINS: Final[frozenset[str]] = frozenset(
    {"reflectance", "kelvin", "linear_coverage", "scalar_physical"}
)


def stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArrayIdentity:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def array_identity(name: str, values: np.ndarray) -> ArrayIdentity:
    array = np.ascontiguousarray(np.asarray(values))
    header = stable_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return ArrayIdentity(
        name=name,
        shape=tuple(int(value) for value in array.shape),
        dtype=array.dtype.str,
        sha256=sha256_bytes(header + array.tobytes(order="C")),
    )


def cira_stretch_reference(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """The unchanged Satpy/CIRA transfer function, used only as an oracle."""

    source = np.asarray(values)
    work = np.clip(source, np.finfo(np.float32).eps, None)
    with np.errstate(invalid="ignore"):
        result = (np.log10(work) - CIRA_LOG_ROOT_EXPECTED) / CIRA_DENOM_EXPECTED
    result = np.where(np.isnan(source), np.nan, result)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def inverse_cira_compatibility(
    gray: np.ndarray | Sequence[float],
    coverage: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Map clean-IR gray to the continuous CIRA preimage coordinate.

    The result is an appearance-compatibility coordinate, never infrared
    reflectance.  Finite gray zero maps to 0.0223.  Missing values remain NaN
    and are never confused with that valid black endpoint.
    """

    values = np.asarray(gray, dtype=np.float64)
    if coverage is None:
        valid = np.isfinite(values)
    else:
        cover = np.asarray(coverage, dtype=np.float64)
        if cover.shape != values.shape:
            raise ValueError("night-gray coverage shape differs from gray")
        if np.any(~np.isfinite(cover)) or np.any((cover < 0.0) | (cover > 1.0)):
            raise ValueError("coverage must be finite and lie in [0, 1]")
        valid = np.isfinite(values) & (cover > 0.0)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    clipped = np.clip(values, 0.0, 1.0)
    result[valid] = np.power(
        10.0,
        CIRA_LOG_ROOT_EXPECTED + CIRA_DENOM_EXPECTED * clipped[valid],
    )
    return result.astype(np.float32)


def clean_ir_gray_reference(
    bt_kelvin: np.ndarray | Sequence[float],
    coverage: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    bt = np.asarray(bt_kelvin, dtype=np.float64)
    if coverage is None:
        valid = np.isfinite(bt)
    else:
        cover = np.asarray(coverage, dtype=np.float64)
        if cover.shape != bt.shape:
            raise ValueError("IR coverage shape differs from brightness temperature")
        if np.any(~np.isfinite(cover)) or np.any((cover < 0.0) | (cover > 1.0)):
            raise ValueError("coverage must be finite and lie in [0, 1]")
        valid = np.isfinite(bt) & (cover > 0.0)
    result = np.full(bt.shape, np.nan, dtype=np.float64)
    result[valid] = np.clip(
        (IR_T_WARM_K - bt[valid]) / (IR_T_WARM_K - IR_T_COLD_K),
        0.0,
        1.0,
    )
    return result.astype(np.float32)


def legacy_m2_day_weight_from_sza(
    sza_degrees: np.ndarray | Sequence[float],
) -> np.ndarray:
    """M2's deliberately retained 88-to-95 degree, no-survival day weight."""

    sza = np.asarray(sza_degrees, dtype=np.float64)
    result = np.zeros(sza.shape, dtype=np.float64)
    finite = np.isfinite(sza)
    result[finite & (sza <= 88.0)] = 1.0
    transition = finite & (sza > 88.0) & (sza < 95.0)
    fraction = (sza[transition] - 88.0) / 7.0
    result[transition] = np.clip(
        1.0 - np.log1p(fraction) / np.log(2.0), 0.0, 1.0
    )
    return result.astype(np.float32)


def exact_area_response(
    source_size: int,
    destination_size: int,
    *,
    sampling_phase: float = 0.5,
) -> np.ndarray:
    """Exact, convex one-dimensional footprint matrix for the oracle."""

    if source_size <= 0 or destination_size <= 0:
        raise ValueError("source and destination sizes must be positive")
    if not 0.0 <= sampling_phase <= 1.0:
        raise ValueError("sampling phase must lie in [0, 1]")
    scale = source_size / destination_size
    response = np.zeros((destination_size, source_size), dtype=np.float64)
    for output_index in range(destination_size):
        start = (output_index + sampling_phase - 0.5) * scale
        stop = (output_index + sampling_phase + 0.5) * scale
        first = max(0, int(math.floor(start)))
        last = min(source_size, int(math.ceil(stop)))
        for source_index in range(first, last):
            response[output_index, source_index] = max(
                0.0,
                min(stop, source_index + 1.0) - max(start, source_index),
            )
        weight = float(response[output_index].sum())
        if weight <= 0.0:
            raise RuntimeError("exact-area destination footprint has no support")
        response[output_index] /= weight
    return response


def nearest_response(source_size: int, destination_size: int) -> np.ndarray:
    if source_size <= 0 or destination_size <= 0:
        raise ValueError("source and destination sizes must be positive")
    indices = np.floor(
        (np.arange(destination_size, dtype=np.float64) + 0.5)
        * source_size
        / destination_size
    ).astype(np.int64)
    indices = np.clip(indices, 0, source_size - 1)
    response = np.zeros((destination_size, source_size), dtype=np.float64)
    response[np.arange(destination_size), indices] = 1.0
    return response


def _validate_response(response: np.ndarray, source_size: int) -> np.ndarray:
    matrix = np.asarray(response, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != source_size:
        raise ValueError("resample response has the wrong source dimension")
    if matrix.shape[0] <= 0 or np.any(~np.isfinite(matrix)):
        raise ValueError("resample response is empty or non-finite")
    if float(matrix.min()) < -1e-7:
        raise ValueError("oracle accepts only convex response weights")
    if np.max(np.abs(matrix.sum(axis=1) - 1.0)) > 2e-6:
        raise ValueError("resample response rows must sum to one")
    return matrix


def separable_resample(
    values: np.ndarray,
    response_y: np.ndarray,
    response_x: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim not in {2, 3}:
        raise ValueError("separable resample expects HxW or HxWxC")
    vertical = _validate_response(response_y, source.shape[0])
    horizontal = _validate_response(response_x, source.shape[1])
    if source.ndim == 2:
        return vertical @ source @ horizontal.T
    return np.einsum("yi,ijc,xj->yxc", vertical, source, horizontal)


def exact_area_resample(
    values: np.ndarray,
    output_shape: tuple[int, int],
    *,
    sampling_phase: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim not in {2, 3}:
        raise ValueError("exact-area resample expects HxW or HxWxC")
    response_y = exact_area_response(
        source.shape[0], output_shape[0], sampling_phase=sampling_phase[0]
    )
    response_x = exact_area_response(
        source.shape[1], output_shape[1], sampling_phase=sampling_phase[1]
    )
    return separable_resample(source, response_y, response_x)


@dataclass(frozen=True, slots=True)
class CoverageResample:
    values: np.ndarray
    coverage: np.ndarray


def premultiplied_coverage_resample(
    values: np.ndarray,
    coverage: np.ndarray,
    response_y: np.ndarray,
    response_x: np.ndarray,
    *,
    tau: float = COVERAGE_TAU,
) -> CoverageResample:
    """Resample value*coverage and coverage with the same convex transform."""

    source = np.asarray(values, dtype=np.float64)
    cover = np.asarray(coverage, dtype=np.float64)
    if source.ndim not in {2, 3}:
        raise ValueError("covered values must be HxW or HxWxC")
    if cover.shape != source.shape[:2]:
        raise ValueError("coverage and value shapes differ")
    if tau < 0.0:
        raise ValueError("coverage threshold must be non-negative")
    if np.any(~np.isfinite(cover)) or np.any((cover < 0.0) | (cover > 1.0)):
        raise ValueError("coverage must be finite and lie in [0, 1]")
    positive = cover > 0.0
    if source.ndim == 2:
        invalid_covered = positive & ~np.isfinite(source)
        cleaned = np.where(positive, source, 0.0)
        premultiplied = cleaned * cover
    else:
        invalid_covered = positive[..., None] & ~np.isfinite(source)
        cleaned = np.where(positive[..., None], source, 0.0)
        premultiplied = cleaned * cover[..., None]
    if np.any(invalid_covered):
        raise ValueError("positive coverage cannot accompany a non-finite value")
    delivered_coverage = separable_resample(cover, response_y, response_x)
    delivered_premultiplied = separable_resample(
        premultiplied, response_y, response_x
    )
    if source.ndim == 2:
        delivered = np.full(delivered_coverage.shape, np.nan, dtype=np.float64)
        np.divide(
            delivered_premultiplied,
            delivered_coverage,
            out=delivered,
            where=delivered_coverage > tau,
        )
    else:
        delivered = np.full(
            (*delivered_coverage.shape, source.shape[-1]), np.nan, dtype=np.float64
        )
        np.divide(
            delivered_premultiplied,
            delivered_coverage[..., None],
            out=delivered,
            where=delivered_coverage[..., None] > tau,
        )
    return CoverageResample(
        values=delivered.astype(np.float32),
        coverage=delivered_coverage.astype(np.float32),
    )


@dataclass(frozen=True, slots=True)
class DayNightComposite:
    display_rgb: np.ndarray
    cira_preimage_compatibility: np.ndarray
    coverage: np.ndarray
    effective_day_fraction: np.ndarray


def day_night_composite_reference(
    day_reflectance: np.ndarray,
    day_coverage: np.ndarray,
    delivered_ir_bt_kelvin: np.ndarray,
    ir_coverage: np.ndarray,
    day_weight: np.ndarray,
    *,
    tau: float = COVERAGE_TAU,
) -> DayNightComposite:
    """Pointwise M2 compatibility blend after physical delivery resampling."""

    day = np.asarray(day_reflectance, dtype=np.float64)
    if day.ndim != 3 or day.shape[-1] != 3:
        raise ValueError("day reflectance must be HxWx3")
    shape = day.shape[:2]
    day_cover = np.asarray(day_coverage, dtype=np.float64)
    bt = np.asarray(delivered_ir_bt_kelvin, dtype=np.float64)
    ir_cover = np.asarray(ir_coverage, dtype=np.float64)
    weight = np.asarray(day_weight, dtype=np.float64)
    for label, field in (
        ("day coverage", day_cover),
        ("IR BT", bt),
        ("IR coverage", ir_cover),
        ("day weight", weight),
    ):
        if field.shape != shape:
            raise ValueError(f"{label} shape differs from day reflectance")
    for label, cover in (("day", day_cover), ("IR", ir_cover)):
        if np.any(~np.isfinite(cover)) or np.any((cover < 0.0) | (cover > 1.0)):
            raise ValueError(f"{label} coverage must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(weight)) or np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("day weight must be finite and lie in [0, 1]")
    if np.any((day_cover[..., None] > 0.0) & ~np.isfinite(day)):
        raise ValueError("positive day coverage cannot accompany non-finite RGB")
    if np.any((ir_cover > 0.0) & ~np.isfinite(bt)):
        raise ValueError("positive IR coverage cannot accompany non-finite BT")

    gray = clean_ir_gray_reference(bt, ir_cover)
    night = inverse_cira_compatibility(gray, ir_cover).astype(np.float64)
    alpha_day = weight * day_cover
    alpha_night = (1.0 - weight) * ir_cover
    coverage = alpha_day + alpha_night
    day_clean = np.where(alpha_day[..., None] > 0.0, day, 0.0)
    day_term = day_clean * alpha_day[..., None]
    night_clean = np.where(alpha_night > 0.0, night, 0.0)
    night_term = (night_clean * alpha_night)[..., None]
    preimage = np.full(day.shape, np.nan, dtype=np.float64)
    np.divide(
        day_term + night_term,
        coverage[..., None],
        out=preimage,
        where=coverage[..., None] > tau,
    )
    effective_day = np.zeros(shape, dtype=np.float64)
    np.divide(alpha_day, coverage, out=effective_day, where=coverage > tau)
    return DayNightComposite(
        display_rgb=cira_stretch_reference(preimage),
        cira_preimage_compatibility=preimage.astype(np.float32),
        coverage=coverage.astype(np.float32),
        effective_day_fraction=effective_day.astype(np.float32),
    )


def canonical_radial_power_profile(
    values: np.ndarray,
    *,
    bins: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical Rec.601, square-crop, 2-D Hann, radial mean-PSD procedure."""

    if bins <= 0:
        raise ValueError("radial bin count must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[-1] != 3:
            raise ValueError("RGB spectral input must have three channels")
        luma = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )
    elif array.ndim == 2:
        luma = array
    else:
        raise ValueError("spectral input must be HxW or HxWx3")
    if min(luma.shape) < 2:
        raise ValueError("spectral input is too small")
    luma = luma - np.mean(luma)
    size = min(luma.shape)
    luma = luma[:size, :size]
    window = np.hanning(size)
    windowed = luma * window[:, None] * window[None, :]
    power = np.fft.fftshift(np.abs(np.fft.fft2(windowed)) ** 2)
    frequency = np.fft.fftshift(np.fft.fftfreq(size))
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    edges = np.linspace(0.0, 0.5, bins + 1)
    indices = np.digitize(radius.ravel(), edges) - 1
    flattened = power.ravel()
    profile = np.asarray(
        [
            float(np.mean(flattened[indices == index]))
            if np.any(indices == index)
            else np.nan
            for index in range(bins)
        ],
        dtype=np.float64,
    )
    return 0.5 * (edges[:-1] + edges[1:]), profile


def radial_high_frequency_power(
    values: np.ndarray,
    *,
    cutoff: float = 0.3,
    bins: int = 64,
) -> float:
    if not 0.0 <= cutoff < 0.5:
        raise ValueError("radial cutoff must lie in [0, 0.5)")
    frequencies, profile = canonical_radial_power_profile(values, bins=bins)
    selected = np.isfinite(profile) & (frequencies > cutoff)
    return float(np.sum(profile[selected]))


def radial_high_frequency_power_ratio(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    cutoff: float = 0.3,
    bins: int = 64,
) -> float:
    denominator = radial_high_frequency_power(reference, cutoff=cutoff, bins=bins)
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("reference contains no measurable high-frequency power")
    return radial_high_frequency_power(candidate, cutoff=cutoff, bins=bins) / denominator


def unwindowed_proxy_high_frequency_power(
    values: np.ndarray,
    *,
    cutoff: float = 0.3,
) -> float:
    """Intentional V03 mutant: no Hann window and includes FFT corners."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        array = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    size = min(array.shape)
    square = array[:size, :size] - np.mean(array[:size, :size])
    power = np.fft.fftshift(np.abs(np.fft.fft2(square)) ** 2)
    frequency = np.fft.fftshift(np.fft.fftfreq(size))
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    return float(np.sum(power[radius > cutoff]))


def rec709_proxy_high_frequency_power(
    values: np.ndarray,
    *,
    cutoff: float = 0.3,
    bins: int = 64,
) -> float:
    """Intentional V03 mutant using Rec.709 instead of canonical Rec.601."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Rec.709 proxy expects HxWx3 RGB")
    rec709 = (
        0.2126 * array[..., 0]
        + 0.7152 * array[..., 1]
        + 0.0722 * array[..., 2]
    )
    return radial_high_frequency_power(rec709, cutoff=cutoff, bins=bins)


def rms_dn(candidate: np.ndarray, reference: np.ndarray) -> float:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("RMS arrays have different shapes")
    return float(np.sqrt(np.mean(np.square(left - right))))


def source_coverage_from_responses(
    response_y: np.ndarray,
    response_x: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> Mapping[str, int | float]:
    vertical = np.asarray(response_y, dtype=np.float64)
    horizontal = np.asarray(response_x, dtype=np.float64)
    if vertical.ndim != 2 or horizontal.ndim != 2:
        raise ValueError("coverage responses must be matrices")
    covered_y = np.any(np.abs(vertical) > epsilon, axis=0)
    covered_x = np.any(np.abs(horizontal) > epsilon, axis=0)
    discarded_y = int(covered_y.size - np.count_nonzero(covered_y))
    discarded_x = int(covered_x.size - np.count_nonzero(covered_x))
    return {
        "source_rows": int(covered_y.size),
        "source_columns": int(covered_x.size),
        "discarded_rows": discarded_y,
        "discarded_columns": discarded_x,
        "discarded_row_fraction": discarded_y / covered_y.size,
        "discarded_column_fraction": discarded_x / covered_x.size,
    }


def local_neighbourhood_out_of_range_fraction(
    source: np.ndarray,
    candidate: np.ndarray,
    response_y: np.ndarray,
    response_x: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> float:
    original = np.asarray(source, dtype=np.float64)
    delivered = np.asarray(candidate, dtype=np.float64)
    if original.ndim == 2:
        original = original[..., None]
    if delivered.ndim == 2:
        delivered = delivered[..., None]
    vertical = _validate_response(response_y, original.shape[0])
    horizontal = _validate_response(response_x, original.shape[1])
    if delivered.shape != (vertical.shape[0], horizontal.shape[0], original.shape[2]):
        raise ValueError("candidate shape differs from response geometry")
    violations = 0
    for output_y, weights_y in enumerate(vertical):
        support_y = np.flatnonzero(np.abs(weights_y) > 1e-12)
        for output_x, weights_x in enumerate(horizontal):
            support_x = np.flatnonzero(np.abs(weights_x) > 1e-12)
            neighbourhood = original[np.ix_(support_y, support_x)]
            lower = np.min(neighbourhood, axis=(0, 1))
            upper = np.max(neighbourhood, axis=(0, 1))
            value = delivered[output_y, output_x]
            violations += int(
                np.count_nonzero(
                    (value < lower - tolerance) | (value > upper + tolerance)
                )
            )
    return violations / delivered.size


@dataclass(frozen=True, slots=True)
class EdgeRinging:
    step_dn: float
    overshoot_dn: float
    undershoot_dn: float
    ringing_dn: float


def local_edge_ringing(
    profile: np.ndarray | Sequence[float],
    edge_index: int,
    *,
    window: int = 12,
) -> EdgeRinging:
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("edge profile must be one-dimensional")
    left_start = max(0, edge_index - window)
    left_stop = max(0, edge_index - 3)
    right_start = min(values.size, edge_index + 3)
    right_stop = min(values.size, edge_index + window)
    if left_stop <= left_start or right_stop <= right_start:
        raise ValueError("edge fixture lacks two plateau windows")
    left_values = values[left_start:left_stop]
    right_values = values[right_start:right_stop]
    left = float(np.median(left_values))
    right = float(np.median(right_values))
    if math.isclose(left, right, abs_tol=1e-9):
        raise ValueError("edge plateau step is zero")
    if right > left:
        undershoot = max(0.0, left - float(np.min(left_values)))
        overshoot = max(0.0, float(np.max(right_values)) - right)
    else:
        undershoot = max(0.0, right - float(np.min(right_values)))
        overshoot = max(0.0, float(np.max(left_values)) - left)
    return EdgeRinging(
        step_dn=abs(right - left),
        overshoot_dn=overshoot,
        undershoot_dn=undershoot,
        ringing_dn=max(overshoot, undershoot),
    )


def weak_global_ringing(
    profile: np.ndarray | Sequence[float], source_min: float, source_max: float
) -> float:
    values = np.asarray(profile, dtype=np.float64)
    return max(
        0.0,
        source_min - float(np.min(values)),
        float(np.max(values)) - source_max,
    )


@dataclass(frozen=True, slots=True)
class CapturedGeometry:
    schema: str
    capture_method: str
    captured_at_runtime: bool
    source_shape: tuple[int, int]
    internal_shape: tuple[int, int]
    delivered_internal_bounds: tuple[int, int, int, int]
    affine: tuple[tuple[float, float, float], ...]
    sampling_phase: tuple[float, float]
    source_extent: tuple[float, float, float, float]
    view_extent: tuple[float, float, float, float]

    @property
    def delivered_shape(self) -> tuple[int, int]:
        top, bottom, left, right = self.delivered_internal_bounds
        return bottom - top, right - left

    @property
    def anisotropic_ratios(self) -> tuple[float, float]:
        return (
            self.source_shape[0] / self.delivered_shape[0],
            self.source_shape[1] / self.delivered_shape[1],
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["delivered_shape"] = self.delivered_shape
        result["anisotropic_ratios"] = self.anisotropic_ratios
        result["identity_sha256"] = self.identity_sha256
        return result

    @property
    def identity_sha256(self) -> str:
        return sha256_bytes(stable_json_bytes(asdict(self)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapturedGeometry":
        return cls(
            schema=str(value["schema"]),
            capture_method=str(value["capture_method"]),
            captured_at_runtime=value["captured_at_runtime"] is True,
            source_shape=tuple(int(item) for item in value["source_shape"]),  # type: ignore[arg-type]
            internal_shape=tuple(int(item) for item in value["internal_shape"]),  # type: ignore[arg-type]
            delivered_internal_bounds=tuple(
                int(item) for item in value["delivered_internal_bounds"]
            ),  # type: ignore[arg-type]
            affine=tuple(
                tuple(float(item) for item in row) for row in value["affine"]
            ),
            sampling_phase=tuple(
                float(item) for item in value["sampling_phase"]
            ),  # type: ignore[arg-type]
            source_extent=tuple(float(item) for item in value["source_extent"]),  # type: ignore[arg-type]
            view_extent=tuple(float(item) for item in value["view_extent"]),  # type: ignore[arg-type]
        )


def captured_geometry_issues(value: CapturedGeometry) -> tuple[str, ...]:
    issues: list[str] = []
    if value.schema != M2_TRACE_SCHEMA:
        issues.append("geometry trace schema is not the registered M2 schema")
    if value.capture_method != M2_CAPTURE_METHOD:
        issues.append("geometry was not obtained from the runtime AxesImage capture")
    if not value.captured_at_runtime:
        issues.append("geometry declares a hard-coded or inferred origin")
    if len(value.source_shape) != 2 or len(value.internal_shape) != 2:
        issues.append("source/internal shapes must each have two axes")
    elif min(*value.source_shape, *value.internal_shape) <= 0:
        issues.append("source/internal shapes must be positive")
    if len(value.delivered_internal_bounds) != 4:
        issues.append("delivered internal bounds must contain four integers")
    else:
        top, bottom, left, right = value.delivered_internal_bounds
        if not (0 <= top < bottom <= value.internal_shape[0]):
            issues.append("delivered row mapping is outside the captured internal raster")
        if not (0 <= left < right <= value.internal_shape[1]):
            issues.append("delivered column mapping is outside the captured internal raster")
    matrix = np.asarray(value.affine, dtype=np.float64)
    if matrix.shape != (3, 3) or np.any(~np.isfinite(matrix)):
        issues.append("captured affine must be a finite 3x3 matrix")
    elif not np.array_equal(matrix[2], np.asarray([0.0, 0.0, 1.0])):
        issues.append("captured affine has an invalid homogeneous row")
    phase = np.asarray(value.sampling_phase, dtype=np.float64)
    if phase.shape != (2,) or np.any(~np.isfinite(phase)):
        issues.append("captured sampling phase must contain two finite values")
    elif np.any((phase < 0.0) | (phase > 1.0)):
        issues.append("captured sampling phase must lie in [0, 1] on both axes")
    for label, extent in (("source", value.source_extent), ("view", value.view_extent)):
        if len(extent) != 4 or not np.all(np.isfinite(extent)):
            issues.append(f"{label} extent must contain four finite coordinates")
    return tuple(issues)


def geometry_matches_runtime_capture(
    candidate: CapturedGeometry, actual_capture: CapturedGeometry
) -> bool:
    return (
        not captured_geometry_issues(candidate)
        and not captured_geometry_issues(actual_capture)
        and candidate.identity_sha256 == actual_capture.identity_sha256
    )


@dataclass(frozen=True, slots=True)
class SpatialOperation:
    name: str
    domain: str
    spatial: bool
    operation_kind: str
    kernel: str | None = None
    sampling_phase: tuple[float, float] | None = None
    source_extent: tuple[float, float, float, float] | None = None
    destination_extent: tuple[float, float, float, float] | None = None
    coverage_transform: str | None = None
    coverage_threshold: float | None = None
    input_semantics: str | None = None
    output_semantics: str | None = None


_REQUIRED_FINALIZER_SEMANTICS: Final[Mapping[str, tuple[str, str, str]]] = {
    "day_delivery": (
        "reflectance",
        "corrected_reflectance",
        "delivered_corrected_reflectance",
    ),
    "ir_delivery": (
        "kelvin",
        "ir_bt_kelvin",
        "delivered_ir_bt_kelvin",
    ),
    "coverage_delivery": (
        "linear_coverage",
        "linear_coverage",
        "delivered_linear_coverage",
    ),
    "terminator_weight_delivery": (
        "scalar_physical",
        "source_day_weight_legacy_88_to_95",
        "delivered_day_weight_legacy_88_to_95",
    ),
    "clean_ir_gray": (
        "kelvin",
        "delivered_ir_bt_kelvin",
        "clean_ir_gray",
    ),
    "inverse_cira_compatibility": (
        "scalar_physical",
        "clean_ir_gray",
        "cira_preimage_compatibility_coordinate",
    ),
    "cira_stretch": (
        "display_encoding",
        "composite_cira_preimage",
        "display_rgb",
    ),
    "display_glint_compat": (
        "display",
        "display_rgb",
        "display_rgb",
    ),
}


def _extent_issues(label: str, value: tuple[float, ...] | None) -> list[str]:
    if value is None:
        return [f"{label} is absent"]
    extent = np.asarray(value, dtype=np.float64)
    if extent.shape != (4,) or np.any(~np.isfinite(extent)):
        return [f"{label} must contain four finite coordinates"]
    if extent[0] == extent[1] or extent[2] == extent[3]:
        return [f"{label} must be non-degenerate"]
    return []


def primary_finalizer_provenance_issues(
    operations: Sequence[SpatialOperation],
    *,
    input_semantics: str,
    geometry: CapturedGeometry | None = None,
) -> tuple[str, ...]:
    issues: list[str] = []
    if input_semantics != "corrected_reflectance_plus_ir_bt_kelvin":
        issues.append("M2 physical boundary received encoded or untyped input")

    by_name: dict[str, list[int]] = {}
    for index, operation in enumerate(operations):
        by_name.setdefault(operation.name, []).append(index)
    for name, (domain, expected_input, expected_output) in (
        _REQUIRED_FINALIZER_SEMANTICS.items()
    ):
        indices = by_name.get(name, [])
        if len(indices) != 1:
            issues.append(f"primary finalizer must contain exactly one {name} operation")
            continue
        operation = operations[indices[0]]
        if operation.domain != domain:
            issues.append(f"{name} has domain {operation.domain!r}, expected {domain!r}")
        if operation.input_semantics != expected_input:
            issues.append(f"{name} input semantics are not bound")
        if operation.output_semantics != expected_output:
            issues.append(f"{name} output semantics are not bound")

    cira_indices = by_name.get("cira_stretch", [])
    cira_index = cira_indices[0] if len(cira_indices) == 1 else -1
    geometry_valid = geometry is not None and not captured_geometry_issues(geometry)
    if not geometry_valid:
        issues.append("V13 provenance is not bound to valid runtime-captured geometry")

    spatial_signatures: set[tuple[Any, ...]] = set()
    for index, operation in enumerate(operations):
        if operation.spatial and operation.domain not in CANONICAL_SPATIAL_DOMAINS:
            issues.append(
                f"spatial operation {operation.name!r} uses forbidden domain {operation.domain!r}"
            )
        if operation.spatial and cira_index >= 0 and index > cira_index:
            issues.append(f"spatial operation {operation.name!r} occurs after CIRA")
        if operation.name == "cira_stretch" and operation.spatial:
            issues.append("CIRA must be pointwise")
        if operation.spatial:
            if operation.operation_kind != PHYSICAL_DELIVERY_OPERATION_KIND:
                issues.append(
                    f"spatial operation {operation.name!r} has an unbound operation kind"
                )
            if operation.kernel != PHYSICAL_DELIVERY_KERNEL:
                issues.append(
                    f"spatial operation {operation.name!r} does not bind the Hanning kernel"
                )
            phase = (
                None
                if operation.sampling_phase is None
                else np.asarray(operation.sampling_phase, dtype=np.float64)
            )
            if (
                phase is None
                or phase.shape != (2,)
                or np.any(~np.isfinite(phase))
                or np.any((phase < 0.0) | (phase > 1.0))
            ):
                issues.append(
                    f"spatial operation {operation.name!r} has no valid sampling phase"
                )
            issues.extend(
                _extent_issues(
                    f"spatial operation {operation.name!r} source extent",
                    operation.source_extent,
                )
            )
            issues.extend(
                _extent_issues(
                    f"spatial operation {operation.name!r} destination extent",
                    operation.destination_extent,
                )
            )
            if operation.coverage_transform != PHYSICAL_COVERAGE_TRANSFORM:
                issues.append(
                    f"spatial operation {operation.name!r} has an unbound coverage transform"
                )
            if operation.coverage_threshold != COVERAGE_TAU:
                issues.append(
                    f"spatial operation {operation.name!r} has an unbound coverage threshold"
                )
            spatial_signatures.add(
                (
                    operation.kernel,
                    operation.sampling_phase,
                    operation.source_extent,
                    operation.destination_extent,
                    operation.coverage_transform,
                    operation.coverage_threshold,
                )
            )
            if geometry_valid and geometry is not None:
                if operation.sampling_phase != geometry.sampling_phase:
                    issues.append(
                        f"spatial operation {operation.name!r} phase differs from captured geometry"
                    )
                if operation.source_extent != geometry.source_extent:
                    issues.append(
                        f"spatial operation {operation.name!r} source extent differs from captured geometry"
                    )
                if operation.destination_extent != geometry.view_extent:
                    issues.append(
                        f"spatial operation {operation.name!r} destination extent differs from captured geometry"
                    )
        else:
            if operation.operation_kind != "pointwise":
                issues.append(
                    f"pointwise operation {operation.name!r} has an unbound operation kind"
                )
            if any(
                value is not None
                for value in (
                    operation.kernel,
                    operation.sampling_phase,
                    operation.source_extent,
                    operation.destination_extent,
                    operation.coverage_transform,
                    operation.coverage_threshold,
                )
            ):
                issues.append(
                    f"pointwise operation {operation.name!r} falsely declares spatial metadata"
                )
        if operation.name == "display_glint_compat":
            if operation.spatial:
                issues.append("display glint compatibility operation must be pointwise")
            if cira_index < 0 or index <= cira_index:
                issues.append("display glint compatibility operation must follow CIRA")
        if operation.name == "terminator_day_weight" or any(
            semantics is not None and "delivered_cos_sza" in semantics
            for semantics in (operation.input_semantics, operation.output_semantics)
        ):
            issues.append(
                "terminator taper must be evaluated on the source grid and "
                "delivered directly; nonlinear tapering of delivered cosine is forbidden"
            )
    if len(spatial_signatures) != 1:
        issues.append("physical values and coverage do not share one bound transform")
    if not any(op.spatial and op.domain == "reflectance" for op in operations):
        issues.append("no reflectance-domain delivery operation was recorded")
    if not any(op.spatial and op.domain == "kelvin" for op in operations):
        issues.append("no kelvin-domain delivery operation was recorded")
    if not any(
        op.spatial and op.name == "terminator_weight_delivery"
        for op in operations
    ):
        issues.append("source-evaluated terminator day weight was not delivered")

    ordered_names = (
        "ir_delivery",
        "terminator_weight_delivery",
        "clean_ir_gray",
        "inverse_cira_compatibility",
        "cira_stretch",
    )
    if all(len(by_name.get(name, [])) == 1 for name in ordered_names):
        ordered_indices = tuple(by_name[name][0] for name in ordered_names)
        if ordered_indices != tuple(sorted(ordered_indices)):
            issues.append(
                "physical ordering must deliver kelvin and source day weight "
                "before gray mapping, inverse CIRA, and CIRA"
            )
    return tuple(issues)


def claim_boundary_issues(
    *,
    residuals: Iterable[str],
    closed_gate_ids: Iterable[str],
    claims_global_one_raster: bool,
) -> tuple[str, ...]:
    issues: list[str] = []
    residual_set = frozenset(str(item) for item in residuals)
    missing = set(M5_SURVIVING_ENCODED_RESIZES) - residual_set
    if missing:
        issues.append(f"M5 encoded-resize residuals omitted: {sorted(missing)}")
    closed = frozenset(str(item) for item in closed_gate_ids)
    leaked = closed & (M5_DEFERRED_GATE_IDS | FORBIDDEN_M2_CLOSURE_GATE_IDS)
    if leaked:
        issues.append(f"M2 falsely closes out-of-scope gates: {sorted(leaked)}")
    if claims_global_one_raster:
        issues.append("M2 cannot claim a global one-raster pipeline before M5")
    return tuple(issues)


@dataclass(frozen=True, slots=True)
class StationaryFlickerMetrics:
    full_daylight: bool
    stationary_patch: bool
    patch_height: int
    patch_width: int
    luminance_mean: float
    luminance_median: float
    luminance_p1: float
    luminance_p95: float
    luminance_p999: float
    laplacian_rms: float
    red_green_ratio: float
    blue_green_ratio: float
    mean_saturation: float
    black_clipped_fraction: float
    white_clipped_fraction: float


def stationary_flicker_issues(metrics: StationaryFlickerMetrics) -> tuple[str, ...]:
    issues: list[str] = []
    if not metrics.full_daylight:
        issues.append("V30 evidence is not a full-daylight sequence")
    if not metrics.stationary_patch:
        issues.append("V30 evidence is not a stationary reference patch")
    if min(metrics.patch_height, metrics.patch_width) < 96:
        issues.append("V30 reference patch is smaller than 96x96 delivered pixels")
    thresholds = {
        "luminance_mean": 1.0,
        "luminance_median": 1.0,
        "luminance_p1": 1.0,
        "luminance_p95": 2.5,
        "luminance_p999": 5.0,
        "laplacian_rms": 2.0,
        "red_green_ratio": 0.005,
        "blue_green_ratio": 0.015,
        "mean_saturation": 0.010,
        "black_clipped_fraction": 0.005,
        "white_clipped_fraction": 0.005,
    }
    for name, threshold in thresholds.items():
        value = float(getattr(metrics, name))
        if not math.isfinite(value) or value < 0.0:
            issues.append(f"V30 metric {name} is invalid")
        elif value > threshold:
            issues.append(f"V30 metric {name}={value} exceeds {threshold}")
    return tuple(issues)


def repeatability_issues(hashes: Sequence[str]) -> tuple[str, ...]:
    if len(hashes) != 3:
        return ("V25 requires exactly three independent renderer hashes",)
    if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) for value in hashes):
        return ("V25 renderer hashes must be lowercase SHA-256 hex",)
    if len(set(hashes)) != 1:
        return ("V25 renderer hashes are not byte-identical",)
    return ()


def signed_bias_closed_form(contrast_ratio: float) -> float:
    if contrast_ratio <= 0.0:
        raise ValueError("contrast ratio must be positive")
    return 2.0 * math.sqrt(contrast_ratio) / (1.0 + contrast_ratio) - 1.0


def scalar_field_before_palette_fixture() -> Mapping[str, Any]:
    source_kelvin = np.asarray([[183.0, 303.0]], dtype=np.float64)
    response_y = np.asarray([[1.0]], dtype=np.float64)
    response_x = np.asarray([[0.5, 0.5]], dtype=np.float64)
    delivered_kelvin = float(separable_resample(source_kelvin, response_y, response_x)[0, 0])

    def nonlinear_palette(normalized: np.ndarray) -> np.ndarray:
        return np.stack(
            [normalized, np.square(normalized), np.sqrt(normalized)], axis=-1
        )

    approved = nonlinear_palette(
        np.asarray((delivered_kelvin - 183.0) / 120.0)
    )
    encoded_source = nonlinear_palette((source_kelvin - 183.0) / 120.0)
    mutant = separable_resample(encoded_source, response_y, response_x)[0, 0]
    return {
        "delivered_kelvin": delivered_kelvin,
        "approved_rgb": [float(value) for value in approved],
        "palette_before_resample_rgb": [float(value) for value in mutant],
        "mutation_error": float(np.max(np.abs(approved - mutant))),
    }


def _zone_plate_fixture() -> tuple[np.ndarray, tuple[int, int]]:
    height, width = 96, 104
    output_shape = (61, 67)
    y, x = np.mgrid[:height, :width]
    x_norm = (x - (width - 1) / 2.0) / (width / 2.0)
    y_norm = (y - (height - 1) / 2.0) / (height / 2.0)
    phase = 24.0 * np.pi * (np.square(x_norm) + np.square(y_norm))
    base = 0.03 + 0.85 * (0.5 + 0.5 * np.cos(phase))
    reflectance = np.stack(
        [base, 0.7 * base + 0.03, 0.45 * base + 0.06], axis=-1
    )
    return reflectance.astype(np.float32), output_shape


def _floor_dn(values: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(np.asarray(values), 0.0, 1.0) * 255.0).astype(np.uint8)


def m2_render_policy_forwarding_issues(
    source: str,
) -> tuple[int, tuple[str, ...]]:
    """Audit only live M2 render helpers reachable from ``render_png``."""

    tree = ast.parse(textwrap.dedent(source))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "render_png" not in functions:
        return 0, ("render_png definition is missing",)

    def scoped_calls(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[ast.Call, ...]:
        calls: list[ast.Call] = []
        pending: list[ast.AST] = list(function.body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Call):
                calls.append(node)
            pending.extend(ast.iter_child_nodes(node))
        return tuple(sorted(calls, key=lambda item: (item.lineno, item.col_offset)))

    calls_by_function = {
        name: scoped_calls(node) for name, node in functions.items()
    }
    local_edges = {
        name: {
            call.func.id
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id in functions
        }
        for name, calls in calls_by_function.items()
    }
    reaches_dispatcher = {
        name: any(
            isinstance(call.func, ast.Name)
            and call.func.id == "_imshow_with_render_policy"
            for call in calls
        )
        for name, calls in calls_by_function.items()
    }
    changed = True
    while changed:
        changed = False
        for name, edges in local_edges.items():
            if not reaches_dispatcher[name] and any(
                reaches_dispatcher[edge] for edge in edges
            ):
                reaches_dispatcher[name] = True
                changed = True

    live_dispatcher_calls: list[ast.Call] = []
    issues: list[str] = []

    def forwards_policy(call: ast.Call) -> bool:
        value = next(
            (item.value for item in call.keywords if item.arg == "policy"), None
        )
        return isinstance(value, ast.Name) and value.id == "policy"

    def visit(function_name: str, stack: tuple[str, ...]) -> None:
        for call in calls_by_function[function_name]:
            target = call.func.id if isinstance(call.func, ast.Name) else None
            if target == "_imshow_with_render_policy":
                live_dispatcher_calls.append(call)
                if not forwards_policy(call):
                    issues.append(
                        f"render dispatcher call at line {call.lineno} does not "
                        "forward policy=policy"
                    )
                continue
            if target not in functions or not reaches_dispatcher[target]:
                continue
            if not forwards_policy(call):
                issues.append(
                    f"render helper call {target} at line {call.lineno} does not "
                    "forward policy=policy"
                )
            if target in stack:
                issues.append(
                    f"render helper cycle through {target} prevents finite policy proof"
                )
                continue
            visit(target, stack + (function_name,))

    visit("render_png", ())
    if len(live_dispatcher_calls) != 2:
        issues.append(
            f"render_png live call graph has {len(live_dispatcher_calls)} "
            "dispatcher calls, expected 2"
        )
    return len(live_dispatcher_calls), tuple(issues)


def m2_policy_rollback_binding() -> Mapping[str, Any]:
    """Bind the M2 default and the live registered M1/M0 rollback paths."""

    import render_policy

    render_source = (
        production_adapter.REPO_ROOT / "render.py"
    ).read_text(encoding="utf-8")
    call_count, forwarding_issues = m2_render_policy_forwarding_issues(
        render_source
    )
    m0_registered = (
        render_policy.POLICIES.get(render_policy.M0_POLICY.algo_version)
        is render_policy.M0_POLICY
    )
    m1_registered = (
        render_policy.POLICIES.get(render_policy.M1_POLICY.algo_version)
        is render_policy.M1_POLICY
        and render_policy.resolve_render_policy(render_policy.M1_POLICY.algo_version)
        is render_policy.M1_POLICY
    )
    m2_default = (
        render_policy.resolve_render_policy() is render_policy.M2_POLICY
        and render_policy.ALGO_VERSION == render_policy.M2_POLICY.algo_version
    )
    m1_calls = production_adapter.imshow_calls(
        render_policy.M1_POLICY.algo_version
    )
    m1_hanning_sites = (
        len(m1_calls) == 2
        and {call.branch for call in m1_calls} == {"rgb", "scalar"}
        and all(call.interpolation == "hanning" for call in m1_calls)
    )
    issues = [*forwarding_issues]
    if not m0_registered:
        issues.append("registered M0 rollback singleton is absent")
    if not m1_registered:
        issues.append("registered M1 rollback singleton is absent or reconstructed")
    if not m1_hanning_sites:
        issues.append("M1 rollback does not retain two literal Hanning sites")
    if not m2_default:
        issues.append("registered M2 policy is not the active default")
    return {
        "passed": not issues,
        "issues": issues,
        "live_dispatcher_call_count": call_count,
        "live_dispatchers_forward_request_policy": not forwarding_issues,
        "m0_rollback_registered": m0_registered,
        "m1_rollback_registered": m1_registered,
        "m1_literal_hanning_sites": m1_hanning_sites,
        "m2_registered_default": m2_default,
    }


def _product_color_protection() -> Mapping[str, Any]:
    truecolor = production_adapter.production_modules()["truecolor"]
    mismatches: dict[str, Mapping[str, Any]] = {}
    for name, expected in PROTECTED_CONSTANTS.items():
        actual = getattr(truecolor, name, "<missing>")
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}

    failures: list[str] = []
    delivered: dict[str, str] = {}
    for pixel in REFERENCE_PIXELS:
        cosine = np.asarray([pixel.cos_sza], dtype=np.float64)
        for role in ("red", "blue", "green"):
            source = getattr(pixel.reflectance_factor, role)
            expected = getattr(pixel.sunz_corrected, role)
            if source is None:
                continue
            actual = truecolor.sunz_correct(
                np.asarray([source], dtype=np.float32), cosine
            )
            if not np.allclose(actual, [expected], rtol=0.0, atol=REFLECTANCE_ATOL):
                failures.append(f"{pixel.id}:sunz:{role}")
        veggie = truecolor.sunz_correct_pathlength(
            np.asarray([pixel.reflectance_factor.veggie], dtype=np.float32), cosine
        )
        if not np.allclose(
            veggie,
            [pixel.sunz_corrected.veggie],
            rtol=0.0,
            atol=REFLECTANCE_ATOL,
        ):
            failures.append(f"{pixel.id}:sunz:veggie")
        corrected = pixel.rayleigh_corrected
        green = truecolor.make_green(
            pixel.sensor,
            np.asarray([corrected.red], dtype=np.float32),
            np.asarray([corrected.blue], dtype=np.float32),
            None
            if corrected.green is None
            else np.asarray([corrected.green], dtype=np.float32),
            np.asarray([corrected.veggie], dtype=np.float32),
        )
        if not np.allclose(
            green,
            [pixel.synthesized_green],
            rtol=0.0,
            atol=REFLECTANCE_ATOL,
        ):
            failures.append(f"{pixel.id}:green")
        linear = np.asarray(
            [[corrected.red, float(green[0]), corrected.blue]], dtype=np.float32
        )
        display = truecolor.cira_stretch(linear)[0]
        if not np.allclose(
            display, pixel.display_rgb, rtol=0.0, atol=DISPLAY_ATOL
        ):
            failures.append(f"{pixel.id}:display")
        dn = tuple(
            int(value) for value in quantize_display(display, "floor_production")
        )
        if dn != expected_dn(pixel, "floor_production"):
            failures.append(f"{pixel.id}:floor_dn")
        delivered[pixel.id] = "#" + "".join(f"{value:02X}" for value in dn)
    return {
        "v07_passed": not mismatches,
        "v07_mismatches": mismatches,
        "v08_passed": not failures,
        "v08_failures": failures,
        "v08_declared_quantizer": "production_floor",
        "v08_delivered_hex": delivered,
        "cira_stretch_source_sha256": production_adapter.cira_source_sha256(),
        "policy_rollback_binding": m2_policy_rollback_binding(),
    }


@dataclass(frozen=True, slots=True)
class MutationResult:
    mutation_id: str
    killed: bool
    discriminator: str
    measurements: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    outcome: str
    canonical_closure: bool
    evidence_scope: str
    threshold: str
    detail: str
    measurements: Mapping[str, Any]
    canonical_procedure: str | None = None
    canonical_source_document: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M2OracleRun:
    gates: tuple[GateResult, ...]
    mutations: tuple[MutationResult, ...]
    array_identities: tuple[ArrayIdentity, ...]
    product_color_protection: Mapping[str, Any]
    claim_boundary: Mapping[str, Any]

    @property
    def gate_registry_issues(self) -> tuple[str, ...]:
        identifiers = [gate.gate_id for gate in self.gates]
        observed = set(identifiers)
        issues: list[str] = []
        if len(identifiers) != len(observed):
            issues.append("duplicate M2 gate identifiers")
        if observed != M2_ALL_GATE_IDS:
            issues.append(
                "M2 gate registry differs from the exact assigned/deferred/forbidden set"
            )
        by_id = {gate.gate_id: gate for gate in self.gates}
        for gate_id in M5_DEFERRED_GATE_IDS:
            gate = by_id.get(gate_id)
            if gate is None or gate.outcome != "DEFERRED" or gate.canonical_closure:
                issues.append(f"{gate_id} is not exactly DEFERRED/noncanonical")
        for gate_id in FORBIDDEN_M2_CLOSURE_GATE_IDS:
            gate = by_id.get(gate_id)
            if gate is None or gate.outcome != "UNMEASURED" or gate.canonical_closure:
                issues.append(f"{gate_id} is not exactly UNMEASURED/noncanonical")
        return tuple(issues)

    @property
    def mutation_registry_issues(self) -> tuple[str, ...]:
        identifiers = [mutation.mutation_id for mutation in self.mutations]
        observed = set(identifiers)
        issues: list[str] = []
        if len(identifiers) != len(observed):
            issues.append("duplicate M2 mutation identifiers")
        if observed not in {
            M2_SYNTHETIC_MUTATION_IDS,
            M2_CANONICAL_MUTATION_IDS,
        }:
            issues.append("M2 mutation registry is incomplete or contains unknown IDs")
        return tuple(issues)

    @property
    def contract_passed(self) -> bool:
        by_id = {gate.gate_id: gate for gate in self.gates}
        return (
            not self.gate_registry_issues
            and not self.mutation_registry_issues
            and all(result.killed for result in self.mutations)
            and all(
                by_id[gate_id].outcome == "PASS"
                for gate_id in M2_REQUIRED_GATE_IDS
            )
        )

    @property
    def stage_advancement_passed(self) -> bool:
        by_id = {gate.gate_id: gate for gate in self.gates}
        return (
            self.contract_passed
            and {mutation.mutation_id for mutation in self.mutations}
            == M2_CANONICAL_MUTATION_IDS
            and all(
                by_id[gate_id].outcome == "PASS"
                and by_id[gate_id].canonical_closure
                for gate_id in M2_REQUIRED_GATE_IDS
            )
        )

    @property
    def canonical_blockers(self) -> tuple[str, ...]:
        return tuple(
            gate.gate_id
            for gate in self.gates
            if gate.gate_id in M2_REQUIRED_GATE_IDS
            and not (gate.outcome == "PASS" and gate.canonical_closure)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": M2_ORACLE_SCHEMA,
            "contract_passed": self.contract_passed,
            "stage_advancement_passed": self.stage_advancement_passed,
            "canonical_blockers": list(self.canonical_blockers),
            "gate_registry_issues": list(self.gate_registry_issues),
            "mutation_registry_issues": list(self.mutation_registry_issues),
            "gates": [gate.as_dict() for gate in self.gates],
            "mutations": [mutation.as_dict() for mutation in self.mutations],
            "array_identities": [identity.as_dict() for identity in self.array_identities],
            "product_color_protection": dict(self.product_color_protection),
            "claim_boundary": dict(self.claim_boundary),
        }


def _gate(
    gate_id: str,
    *,
    outcome: str,
    canonical_closure: bool,
    evidence_scope: str,
    threshold: str,
    detail: str,
    measurements: Mapping[str, Any],
    research: ResearchInputs | None,
) -> GateResult:
    row = None if research is None else research.gates.get(gate_id)
    return GateResult(
        gate_id=gate_id,
        outcome=outcome,
        canonical_closure=canonical_closure,
        evidence_scope=evidence_scope,
        threshold=threshold if row is None else row.threshold,
        detail=detail,
        measurements=measurements,
        canonical_procedure=None if row is None else row.procedure,
        canonical_source_document=None if row is None else row.source_document,
    )


def _mutation(
    mutation_id: str,
    killed: bool,
    discriminator: str,
    **measurements: Any,
) -> MutationResult:
    return MutationResult(
        mutation_id=mutation_id,
        killed=bool(killed),
        discriminator=discriminator,
        measurements=measurements,
    )


def _synthetic_geometry() -> CapturedGeometry:
    return CapturedGeometry(
        schema=M2_TRACE_SCHEMA,
        capture_method=M2_CAPTURE_METHOD,
        captured_at_runtime=True,
        source_shape=(96, 104),
        internal_shape=(62, 67),
        delivered_internal_bounds=(1, 62, 0, 67),
        affine=(
            (0.6442307692307693, 0.0, -0.173),
            (0.0, -0.6354166666666666, 61.731),
            (0.0, 0.0, 1.0),
        ),
        sampling_phase=(0.5, 0.5),
        source_extent=(-84.1, -72.1, 10.3, 22.3),
        view_extent=(-83.9, -72.3, 10.5, 22.1),
    )


def run_m2_contract_oracle(
    research: ResearchInputs | None = None,
) -> M2OracleRun:
    """Run deterministic closed-form M2 contracts.

    V25 and V30 remain UNMEASURED because this oracle has no renderer-process
    capture or real timestamped sequence adapter.  In particular, this API
    deliberately accepts neither caller-supplied hashes nor metric summaries.
    Synthetic spatial gates may pass, but ``canonical_closure`` remains false,
    so this function cannot advance M2 alone.
    """

    source, output_shape = _zone_plate_fixture()
    response_y = exact_area_response(source.shape[0], output_shape[0])
    response_x = exact_area_response(source.shape[1], output_shape[1])
    nearest_y = nearest_response(source.shape[0], output_shape[0])
    nearest_x = nearest_response(source.shape[1], output_shape[1])

    reference_reflectance = separable_resample(source, response_y, response_x)
    reference_display = cira_stretch_reference(reference_reflectance)
    reference_dn = _floor_dn(reference_display)
    approved_display = cira_stretch_reference(
        separable_resample(source, response_y, response_x)
    )
    approved_dn = _floor_dn(approved_display)
    nearest_display = cira_stretch_reference(
        separable_resample(source, nearest_y, nearest_x)
    )
    nearest_dn = _floor_dn(nearest_display)
    display_first = separable_resample(
        cira_stretch_reference(source), response_y, response_x
    )

    approved_rms = rms_dn(approved_dn, reference_dn)
    nearest_rms = rms_dn(nearest_dn, reference_dn)
    approved_spectral_ratio = radial_high_frequency_power_ratio(
        approved_dn, reference_dn
    )
    nearest_spectral_ratio = radial_high_frequency_power_ratio(
        nearest_dn, reference_dn
    )
    canonical_reference_power = radial_high_frequency_power(reference_dn)
    rec709_reference_power = rec709_proxy_high_frequency_power(reference_dn)
    wrong_bin_reference_power = radial_high_frequency_power(reference_dn, bins=63)
    signed_display_first_dn = float(
        np.mean((display_first.astype(np.float64) - reference_display) * 255.0)
    )
    coverage = source_coverage_from_responses(response_y, response_x)
    nearest_coverage = source_coverage_from_responses(nearest_y, nearest_x)

    contrast_ratios = (2.0, 5.0, 20.0, 50.0)
    approved_bias = {str(int(value)): 0.0 for value in contrast_ratios}
    log_bias = {
        str(int(value)): signed_bias_closed_form(value) for value in contrast_ratios
    }

    inverse_sweep = np.linspace(0.0, 1.0, 10_001, dtype=np.float32)
    inverse_round_trip = cira_stretch_reference(
        inverse_cira_compatibility(inverse_sweep)
    )
    inverse_error = float(np.max(np.abs(inverse_round_trip - inverse_sweep)))

    day = np.full((1, 1, 3), 0.03, dtype=np.float32)
    night_white = np.asarray([[183.0]], dtype=np.float32)
    mixed = day_night_composite_reference(
        day,
        np.ones((1, 1), dtype=np.float32),
        night_white,
        np.ones((1, 1), dtype=np.float32),
        np.asarray([[0.75]], dtype=np.float32),
    )
    approved_mixed = float(mixed.display_rgb[0, 0, 0])
    direct_gray_mutant = float(cira_stretch_reference(np.asarray([0.75 * 0.03 + 0.25]))[0])
    old_display_mutant = float(
        0.75 * cira_stretch_reference(np.asarray([0.03]))[0] + 0.25
    )
    approved_zero_endpoint = float(
        cira_stretch_reference(
            np.asarray([0.5 * 0.3 + 0.5 * CIRA_BLACK_PREIMAGE_EXPECTED])
        )[0]
    )
    zero_endpoint_mutant = float(cira_stretch_reference(np.asarray([0.15]))[0])

    delivered_bt = float(separable_resample(
        np.asarray([[303.0, 183.0]]),
        np.asarray([[1.0]]),
        np.asarray([[0.5, 0.5]]),
    )[0, 0])
    approved_ir_order = float(
        cira_stretch_reference(
            inverse_cira_compatibility(
                clean_ir_gray_reference(np.asarray([delivered_bt]))
            )
        )[0]
    )
    inverse_first = inverse_cira_compatibility(np.asarray([0.0, 1.0]))
    inverse_first_mutant = float(
        cira_stretch_reference(np.asarray([np.mean(inverse_first)]))[0]
    )

    source_sza = np.asarray([0.0, 96.0], dtype=np.float64)
    delivered_source_day_weight = float(
        legacy_m2_day_weight_from_sza(source_sza).mean()
    )
    averaged_cosine = float(np.cos(np.deg2rad(source_sza)).mean())
    nonlinear_after_cosine_day_weight = float(
        legacy_m2_day_weight_from_sza(
            [np.rad2deg(np.arccos(np.clip(averaged_cosine, -1.0, 1.0)))]
        )[0]
    )

    values = np.asarray([[0.4, np.nan]], dtype=np.float32)
    source_coverage = np.asarray([[1.0, 0.0]], dtype=np.float32)
    half_response_x = np.asarray([[0.5, 0.5]], dtype=np.float64)
    covered = premultiplied_coverage_resample(
        values,
        source_coverage,
        np.asarray([[1.0]], dtype=np.float64),
        half_response_x,
    )
    black_fill_mutant = float(
        separable_resample(
            np.nan_to_num(values, nan=0.0),
            np.asarray([[1.0]], dtype=np.float64),
            half_response_x,
        )[0, 0]
    )
    naive_nan_product = float(np.sum(values * source_coverage))
    nearest_mask_coverage = float(
        separable_resample(
            source_coverage,
            np.asarray([[1.0]], dtype=np.float64),
            nearest_response(2, 1),
        )[0, 0]
    )

    range_source = np.full((9, 21), 100.0, dtype=np.float64)
    range_source[:, 11:] = 146.1
    range_source[0, 0] = 0.0
    range_source[-1, -1] = 255.0
    identity_y = np.eye(range_source.shape[0], dtype=np.float64)
    identity_x = np.eye(range_source.shape[1], dtype=np.float64)
    approved_range = local_neighbourhood_out_of_range_fraction(
        range_source, range_source, identity_y, identity_x
    )
    range_mutant = range_source.copy()
    range_mutant[4, 15] = 150.0
    local_range_mutant = local_neighbourhood_out_of_range_fraction(
        range_source, range_mutant, identity_y, identity_x
    )
    global_range_proxy = float(
        np.mean(
            (range_mutant < range_source.min() - 1e-6)
            | (range_mutant > range_source.max() + 1e-6)
        )
    )

    edge_index = 32
    clean_profile = np.full(64, 100.0, dtype=np.float64)
    clean_profile[edge_index:] = 146.1
    clean_profile[0] = 0.0
    clean_profile[-1] = 255.0
    approved_ringing = local_edge_ringing(clean_profile, edge_index)
    ringing_mutant = clean_profile.copy()
    ringing_mutant[edge_index - 4] = 96.0
    ringing_mutant[edge_index + 4] = 150.0
    strengthened_ringing = local_edge_ringing(ringing_mutant, edge_index)
    weak_ringing = weak_global_ringing(ringing_mutant, 0.0, 255.0)

    scalar = scalar_field_before_palette_fixture()
    geometry = _synthetic_geometry()
    hard_coded_geometry = dataclasses.replace(
        geometry,
        capture_method="hard_coded_matplotlib_3_9_plus_one_row",
        captured_at_runtime=False,
    )
    isotropic_geometry = dataclasses.replace(
        geometry,
        affine=(
            (geometry.affine[0][0], 0.0, geometry.affine[0][2]),
            (0.0, -geometry.affine[0][0], geometry.affine[1][2]),
            (0.0, 0.0, 1.0),
        ),
    )
    spatial_metadata = {
        "operation_kind": PHYSICAL_DELIVERY_OPERATION_KIND,
        "kernel": PHYSICAL_DELIVERY_KERNEL,
        "sampling_phase": geometry.sampling_phase,
        "source_extent": geometry.source_extent,
        "destination_extent": geometry.view_extent,
        "coverage_transform": PHYSICAL_COVERAGE_TRANSFORM,
        "coverage_threshold": COVERAGE_TAU,
    }
    operations = (
        SpatialOperation(
            "day_delivery",
            "reflectance",
            True,
            input_semantics="corrected_reflectance",
            output_semantics="delivered_corrected_reflectance",
            **spatial_metadata,
        ),
        SpatialOperation(
            "ir_delivery",
            "kelvin",
            True,
            input_semantics="ir_bt_kelvin",
            output_semantics="delivered_ir_bt_kelvin",
            **spatial_metadata,
        ),
        SpatialOperation(
            "coverage_delivery",
            "linear_coverage",
            True,
            input_semantics="linear_coverage",
            output_semantics="delivered_linear_coverage",
            **spatial_metadata,
        ),
        SpatialOperation(
            "terminator_weight_delivery",
            "scalar_physical",
            True,
            input_semantics="source_day_weight_legacy_88_to_95",
            output_semantics="delivered_day_weight_legacy_88_to_95",
            **spatial_metadata,
        ),
        SpatialOperation(
            "clean_ir_gray",
            "kelvin",
            False,
            "pointwise",
            input_semantics="delivered_ir_bt_kelvin",
            output_semantics="clean_ir_gray",
        ),
        SpatialOperation(
            "inverse_cira_compatibility",
            "scalar_physical",
            False,
            "pointwise",
            input_semantics="clean_ir_gray",
            output_semantics="cira_preimage_compatibility_coordinate",
        ),
        SpatialOperation(
            "cira_stretch",
            "display_encoding",
            False,
            "pointwise",
            input_semantics="composite_cira_preimage",
            output_semantics="display_rgb",
        ),
        SpatialOperation(
            "display_glint_compat",
            "display",
            False,
            "pointwise",
            input_semantics="display_rgb",
            output_semantics="display_rgb",
        ),
    )
    display_domain_ops = (
        dataclasses.replace(
            operations[0],
            name="legacy_rgb_resize",
            domain="display",
            input_semantics="display_rgb",
            output_semantics="display_rgb",
        ),
        *operations[1:],
    )
    glint_first_ops = (
        SpatialOperation(
            "display_glint_compat",
            "display",
            False,
            "pointwise",
            input_semantics="display_rgb",
            output_semantics="display_rgb",
        ),
        *operations[:-1],
    )
    label_only_ops = tuple(
        SpatialOperation(op.name, op.domain, op.spatial, "label_only")
        for op in operations
    )
    wrong_kernel_ops = (
        dataclasses.replace(operations[0], kernel="nearest"),
        *operations[1:],
    )
    wrong_phase_ops = (
        dataclasses.replace(operations[0], sampling_phase=(0.0, 0.0)),
        *operations[1:],
    )
    wrong_extent_ops = (
        dataclasses.replace(
            operations[0],
            destination_extent=(-84.0, -72.0, 10.0, 22.0),
        ),
        *operations[1:],
    )
    wrong_coverage_ops = (
        dataclasses.replace(operations[0], coverage_transform="nearest_mask"),
        *operations[1:],
    )
    inverse_before_kelvin_ops = (
        operations[0],
        operations[5],
        operations[1],
        operations[2],
        operations[3],
        operations[4],
        operations[6],
        operations[7],
    )

    provenance_issues = primary_finalizer_provenance_issues(
        operations,
        input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
        geometry=geometry,
    )
    product_protection = _product_color_protection()

    mutations = (
        _mutation(
            "m1_rollback_policy_bypass",
            bool(product_protection["policy_rollback_binding"]["passed"]),
            "M2 call-graph policy forwarding plus registered M1/M0 rollback",
            **dict(product_protection["policy_rollback_binding"]),
        ),
        _mutation(
            "nearest_neighbor",
            nearest_rms > 1.5 and nearest_spectral_ratio > 1.10,
            "V01/V03 exact-area physical reference",
            rms_dn=nearest_rms,
            spectral_ratio=nearest_spectral_ratio,
        ),
        _mutation(
            "auto_nearest_fallback",
            nearest_rms > 1.5,
            "automatic kernel alias is scored as its measured nearest fallback",
            rms_dn=nearest_rms,
        ),
        _mutation(
            "cira_before_resample",
            signed_display_first_dn < -0.05,
            "V13 signed physical-domain fixture",
            mean_signed_dn=signed_display_first_dn,
        ),
        _mutation(
            "direct_gray_as_reflectance",
            abs(direct_gray_mutant - approved_mixed) > 1e-3,
            "resolved inverse-CIRA mixed day/night fixture",
            approved=approved_mixed,
            mutant=direct_gray_mutant,
        ),
        _mutation(
            "old_display_space_day_night_blend",
            abs(old_display_mutant - approved_mixed) > 1e-3,
            "resolved one-CIRA mixed day/night fixture",
            approved=approved_mixed,
            mutant=old_display_mutant,
        ),
        _mutation(
            "night_preimage_zero_endpoint",
            abs(zero_endpoint_mutant - approved_zero_endpoint) > 1e-3,
            "valid gray zero must map continuously to J(0)=0.0223",
            approved=approved_zero_endpoint,
            mutant=zero_endpoint_mutant,
        ),
        _mutation(
            "inverse_before_bt_resample",
            abs(inverse_first_mutant - approved_ir_order) > 0.1,
            "IR must be sampled in kelvin before gray/inverse mapping",
            delivered_bt_kelvin=delivered_bt,
            approved=approved_ir_order,
            mutant=inverse_first_mutant,
        ),
        _mutation(
            "terminator_taper_after_cosine_resample",
            abs(
                nonlinear_after_cosine_day_weight
                - delivered_source_day_weight
            ) > 0.4,
            "source 88-to-95 day weight is delivered as a scalar field",
            approved_day_weight=delivered_source_day_weight,
            mutant_day_weight=nonlinear_after_cosine_day_weight,
        ),
        _mutation(
            "nan_fill_black",
            abs(float(covered.values[0, 0]) - black_fill_mutant) > 0.1,
            "V41 premultiplied value/coverage fixture",
            approved_value=float(covered.values[0, 0]),
            mutant_value=black_fill_mutant,
        ),
        _mutation(
            "nan_times_zero",
            math.isnan(naive_nan_product) and np.isfinite(covered.values[0, 0]),
            "where(mask>0, value*mask, 0) must precede multiplication",
            naive_is_nan=math.isnan(naive_nan_product),
        ),
        _mutation(
            "nearest_mask",
            abs(nearest_mask_coverage - float(covered.coverage[0, 0])) > 0.1,
            "coverage uses the identical convex transform as values",
            approved_coverage=float(covered.coverage[0, 0]),
            mutant_coverage=nearest_mask_coverage,
        ),
        _mutation(
            "proxy_or_unwindowed_spectrum",
            unwindowed_proxy_high_frequency_power(reference_dn)
            > canonical_reference_power * 10.0,
            "V03 canonical Rec.601/windowed/radial-bin procedure",
            canonical=canonical_reference_power,
            mutant=unwindowed_proxy_high_frequency_power(reference_dn),
        ),
        _mutation(
            "rec709_spectrum_proxy",
            abs(rec709_reference_power / canonical_reference_power - 1.0) > 0.005,
            "V03 luma coefficients are canonical Rec.601",
            canonical=canonical_reference_power,
            mutant=rec709_reference_power,
        ),
        _mutation(
            "wrong_spectrum_bin_count",
            abs(wrong_bin_reference_power / canonical_reference_power - 1.0) > 0.01,
            "V03 uses 64 equal radial bins through axial Nyquist",
            canonical=canonical_reference_power,
            mutant=wrong_bin_reference_power,
        ),
        _mutation(
            "global_extrema_range_proxy",
            local_range_mutant > 0.0 and global_range_proxy == 0.0,
            "V15 actual mapped local-neighbourhood range",
            local_fraction=local_range_mutant,
            global_proxy_fraction=global_range_proxy,
        ),
        _mutation(
            "weak_global_ringing",
            strengthened_ringing.ringing_dn > 0.0 and weak_ringing == 0.0,
            "V16 plateau-relative internal step fixture",
            local_ringing_dn=strengthened_ringing.ringing_dn,
            global_proxy_ringing_dn=weak_ringing,
        ),
        _mutation(
            "hard_coded_geometry_offset",
            bool(captured_geometry_issues(hard_coded_geometry)),
            "runtime AxesImage geometry provenance is mandatory",
            issues=list(captured_geometry_issues(hard_coded_geometry)),
        ),
        _mutation(
            "single_isotropic_ratio",
            not geometry_matches_runtime_capture(isotropic_geometry, geometry),
            "candidate geometry must exactly bind both captured axes",
            actual_ratios=geometry.anisotropic_ratios,
        ),
        _mutation(
            "encoded_rgb_physical_boundary",
            bool(
                primary_finalizer_provenance_issues(
                    operations,
                    input_semantics="display_encoded_rgb",
                    geometry=geometry,
                )
            ),
            "typed M2 input domain assertion",
        ),
        _mutation(
            "display_domain_spatial_operation",
            bool(
                primary_finalizer_provenance_issues(
                    display_domain_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "V13 structural operation-order assertion",
        ),
        _mutation(
            "glint_before_authoritative_resample",
            bool(
                primary_finalizer_provenance_issues(
                    glint_first_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "post-CIRA, pointwise M8-compatibility boundary",
        ),
        _mutation(
            "label_only_operation_trace",
            bool(
                primary_finalizer_provenance_issues(
                    label_only_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "V13 binds operation kinds and typed transitions, not labels",
        ),
        _mutation(
            "unbound_delivery_kernel",
            bool(
                primary_finalizer_provenance_issues(
                    wrong_kernel_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "V13 binds the delivery kernel",
        ),
        _mutation(
            "unbound_delivery_phase",
            bool(
                primary_finalizer_provenance_issues(
                    wrong_phase_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "V13 binds delivery phase to runtime geometry",
        ),
        _mutation(
            "unbound_delivery_extent",
            bool(
                primary_finalizer_provenance_issues(
                    wrong_extent_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "V13 binds source and destination extents to runtime geometry",
        ),
        _mutation(
            "proxy_coverage_transform",
            bool(
                primary_finalizer_provenance_issues(
                    wrong_coverage_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "V13 binds values and coverage to one premultiplied transform",
        ),
        _mutation(
            "inverse_before_kelvin_delivery_trace",
            bool(
                primary_finalizer_provenance_issues(
                    inverse_before_kelvin_ops,
                    input_semantics="corrected_reflectance_plus_ir_bt_kelvin",
                    geometry=geometry,
                )
            ),
            "V13 binds IR kelvin delivery before gray and inverse-CIRA mapping",
        ),
        _mutation(
            "palette_before_scalar_resample",
            float(scalar["mutation_error"]) > 0.1,
            "physical scalar field must precede nonlinear palette",
            mutation_error=scalar["mutation_error"],
        ),
        _mutation(
            "m6_terminator_leakage",
            float(legacy_m2_day_weight_from_sza([92.0])[0]) > 0.0,
            "M2 retains the 88-to-95 degree taper without survival gating",
            m2_weight_at_92_deg=float(legacy_m2_day_weight_from_sza([92.0])[0]),
        ),
        _mutation(
            "false_global_one_raster_claim",
            bool(
                claim_boundary_issues(
                    residuals=(),
                    closed_gate_ids={"V22", "V23", "V44"},
                    claims_global_one_raster=True,
                )
            ),
            "M5 residual/forbidden-gate claim-boundary guard",
        ),
    )

    gates: list[GateResult] = [
        _gate(
            "V01",
            outcome="PASS" if approved_rms <= 1.5 else "FAIL",
            canonical_closure=False,
            evidence_scope="synthetic_contract_only",
            threshold="<= 1.5 DN",
            detail="Exact-area physical zone-plate oracle; production binding still required.",
            measurements={"approved_rms_dn": approved_rms, "nearest_rms_dn": nearest_rms},
            research=research,
        ),
        _gate(
            "V02",
            outcome="PASS"
            if coverage["discarded_rows"] == coverage["discarded_columns"] == 0
            else "FAIL",
            canonical_closure=False,
            evidence_scope="synthetic_contract_only",
            threshold="0 % discarded",
            detail="Direct response support enumeration, not an output-image proxy.",
            measurements={"approved": coverage, "nearest_mutant": nearest_coverage},
            research=research,
        ),
        _gate(
            "V03",
            outcome="PASS" if approved_spectral_ratio <= 1.10 else "FAIL",
            canonical_closure=False,
            evidence_scope="synthetic_contract_only",
            threshold="<= 1.10x",
            detail="Canonical windowed Rec.601 radial-spectrum procedure.",
            measurements={
                "approved_ratio": approved_spectral_ratio,
                "nearest_ratio": nearest_spectral_ratio,
                "bins": 64,
                "cutoff_cycles_per_pixel": 0.3,
            },
            research=research,
        ),
        _gate(
            "V13",
            outcome="PASS" if not provenance_issues else "FAIL",
            canonical_closure=False,
            evidence_scope="synthetic_contract_only",
            threshold="0 display-domain ops; signed error within +/-0.05 DN",
            detail="Physical operation-order trace and signed-domain fixture.",
            measurements={
                "approved_mean_signed_dn": 0.0,
                "display_first_mean_signed_dn": signed_display_first_dn,
                "delivered_source_day_weight_fixture": (
                    delivered_source_day_weight
                ),
                "nonlinear_after_cosine_day_weight_mutant": (
                    nonlinear_after_cosine_day_weight
                ),
                "provenance_issues": list(provenance_issues),
                "operation_trace": [asdict(operation) for operation in operations],
                "geometry_identity_sha256": geometry.identity_sha256,
                "coverage_threshold": COVERAGE_TAU,
            },
            research=research,
        ),
        _gate(
            "V14",
            outcome="PASS"
            if max(abs(value) for value in approved_bias.values()) <= 0.005
            else "FAIL",
            canonical_closure=False,
            evidence_scope="synthetic_contract_only",
            threshold="|relative error| <= 0.5% at C=2,5,20,50",
            detail="Arithmetic-mean reflectance closed form plus resolved night mapping mutants.",
            measurements={
                "approved_relative_error": approved_bias,
                "log_domain_mutant_relative_error": log_bias,
                "approved_mixed_day_night": approved_mixed,
                "direct_gray_mutant": direct_gray_mutant,
                "old_display_blend_mutant": old_display_mutant,
                "approved_zero_endpoint": approved_zero_endpoint,
                "zero_endpoint_mutant": zero_endpoint_mutant,
                "inverse_roundtrip_max_abs": inverse_error,
                "IR_ordering_approved": approved_ir_order,
                "IR_inverse_first_mutant": inverse_first_mutant,
            },
            research=research,
        ),
        _gate(
            "V41",
            outcome="PASS"
            if math.isclose(float(covered.values[0, 0]), 0.4, abs_tol=1e-6)
            and math.isclose(float(covered.coverage[0, 0]), 0.5, abs_tol=1e-6)
            else "FAIL",
            canonical_closure=False,
            evidence_scope="synthetic_contract_only",
            threshold="0 interpolated no-data pixels; linear_premultiplied",
            detail="Value/coverage use one transform; numeric zero remains valid data.",
            measurements={
                "approved_value": float(covered.values[0, 0]),
                "approved_coverage": float(covered.coverage[0, 0]),
                "black_fill_mutant_value": black_fill_mutant,
                "nearest_mask_mutant_coverage": nearest_mask_coverage,
                "composite_domain": "linear_premultiplied",
            },
            research=research,
        ),
        _gate(
            "V07",
            outcome="PASS" if product_protection["v07_passed"] else "FAIL",
            canonical_closure=bool(product_protection["v07_passed"]),
            evidence_scope="current_product_pointwise_protection",
            threshold="exact equality",
            detail="Protected constants read from the imported product module.",
            measurements={
                "mismatches": product_protection["v07_mismatches"],
                "cira_stretch_source_sha256": product_protection[
                    "cira_stretch_source_sha256"
                ],
            },
            research=research,
        ),
        _gate(
            "V08",
            outcome="PASS" if product_protection["v08_passed"] else "FAIL",
            canonical_closure=False,
            evidence_scope="component_fixture_without_M2_finalizer_binding",
            threshold="display <=1e-6; exact declared floor DN",
            detail=(
                "Five reference pixels execute current pointwise components, but no "
                "captured M2 finalizer/output binding is present."
            ),
            measurements={
                "failures": product_protection["v08_failures"],
                "declared_quantizer": product_protection["v08_declared_quantizer"],
                "delivered_hex": product_protection["v08_delivered_hex"],
            },
            research=research,
        ),
        _gate(
            "V15",
            outcome="PASS" if approved_range == 0.0 else "FAIL",
            canonical_closure=False,
            evidence_scope="synthetic_contract_only",
            threshold="exactly 0.000000",
            detail="Actual response-support neighbourhoods; global extrema are rejected.",
            measurements={
                "approved_out_of_range_fraction": approved_range,
                "local_mutant_fraction": local_range_mutant,
                "global_proxy_mutant_fraction": global_range_proxy,
            },
            research=research,
        ),
        _gate(
            "V16",
            outcome="UNMEASURED",
            canonical_closure=False,
            evidence_scope="captured_M1_runtime_baseline_absent",
            threshold="<= bound captured M1 runtime baseline",
            detail=(
                "The strengthened fixture discriminates the weak proxy, but no "
                "captured M1 runtime baseline is bound to this run."
            ),
            measurements={
                "approved": asdict(approved_ringing),
                "strengthened_mutant": asdict(strengthened_ringing),
                "weak_global_mutant_ringing_dn": weak_ringing,
                "m1_runtime_baseline": None,
                "baseline_binding": "absent",
            },
            research=research,
        ),
    ]

    gates.append(
        _gate(
            "V25",
            outcome="UNMEASURED",
            canonical_closure=False,
            evidence_scope="renderer_bytes_and_process_manifests_absent",
            threshold="3 of 3 byte-identical independent renders",
            detail=(
                "This oracle accepts no caller-supplied hashes. Closure requires "
                "recomputed hashes of bound renderer bytes plus three independent "
                "process manifests in a production adapter."
            ),
            measurements={"renderer_capture_binding": None},
            research=research,
        )
    )

    gates.append(
        _gate(
            "V30",
            outcome="UNMEASURED",
            canonical_closure=False,
            evidence_scope="timestamped_renderer_sequence_frames_absent",
            threshold="canonical stationary full-daylight thresholds",
            detail=(
                "This oracle accepts no caller-supplied metric summary. Closure "
                "requires recomputation from bound real renderer frames, timestamps, "
                "and a stationary >=96x96 full-daylight patch."
            ),
            measurements={"sequence_frame_binding": None},
            research=research,
        )
    )

    for gate_id in sorted(M5_DEFERRED_GATE_IDS):
        gates.append(
            _gate(
                gate_id,
                outcome="DEFERRED",
                canonical_closure=False,
                evidence_scope="explicit_M5_residual",
                threshold="deferred to M5",
                detail="Surviving encoded-pixel resizes prevent M2 closure.",
                measurements={"surviving_resizes": list(M5_SURVIVING_ENCODED_RESIZES)},
                research=research,
            )
        )
    for gate_id in sorted(FORBIDDEN_M2_CLOSURE_GATE_IDS):
        gates.append(
            _gate(
                gate_id,
                outcome="UNMEASURED",
                canonical_closure=False,
                evidence_scope="explicitly_unclosed",
                threshold="not an M2 stage gate",
                detail="M2 has no evidence that can close this coverage gate.",
                measurements={},
                research=research,
            )
        )

    claim_boundary = {
        "claim_scope": "primary_imagery_layer_only_until_m5",
        "allowed_claim": (
            "no post-CIRA spatial operation inside the M2 primary "
            "imagery-layer finalizer"
        ),
        "global_one_raster_claim": False,
        "downstream_encoded_resizes": list(M5_SURVIVING_ENCODED_RESIZES),
        "night_mapping": {
            "id": "inverse_cira_continuous_preimage_v1",
            "input": "clean_ir_gray_from_delivered_bt_kelvin",
            "output_semantics": "cira_preimage_compatibility_coordinate",
            "physical_reflectance_claim": False,
            "black_endpoint": CIRA_BLACK_PREIMAGE_EXPECTED,
            "white_endpoint": CIRA_WHITE_PREIMAGE_EXPECTED,
        },
        "day_weight_policy": "legacy_88_to_95_no_survival_gate",
        "glint": "post_CIRA_pointwise_M8_compatibility_exception",
        "forbidden_pass_gate_ids": sorted(FORBIDDEN_M2_CLOSURE_GATE_IDS),
    }
    issues = claim_boundary_issues(
        residuals=claim_boundary["downstream_encoded_resizes"],
        closed_gate_ids={
            gate.gate_id
            for gate in gates
            if gate.outcome == "PASS" and gate.canonical_closure
        },
        claims_global_one_raster=False,
    )
    if issues:
        raise RuntimeError(f"internal claim-boundary error: {issues}")
    forbidden_passes = [
        gate.gate_id
        for gate in gates
        if gate.gate_id in FORBIDDEN_M2_CLOSURE_GATE_IDS
        and (gate.outcome == "PASS" or gate.canonical_closure)
    ]
    if forbidden_passes:
        raise RuntimeError(f"oracle attempted forbidden M2 closure: {forbidden_passes}")

    identities = (
        array_identity("zone_plate_source_reflectance", source),
        array_identity("exact_area_reference_display", reference_display),
        array_identity("exact_area_reference_floor_dn", reference_dn),
        array_identity("nearest_mutant_floor_dn", nearest_dn),
        array_identity("premultiplied_coverage_output", covered.values),
    )
    return M2OracleRun(
        gates=tuple(gates),
        mutations=mutations,
        array_identities=identities,
        product_color_protection=product_protection,
        claim_boundary=claim_boundary,
    )


M2_HARNESS_SOURCE_FILES: Final[tuple[str, ...]] = (
    "scripts/tat_truecolor_m2_validate.py",
    "validation/truecolor/m2.py",
    "validation/truecolor/m2_production.py",
    "validation/truecolor/m2_sequence.py",
    "validation/truecolor/m2_gallery.py",
    "validation/truecolor/m1.py",
    "validation/truecolor/corpus.py",
    "validation/truecolor/contracts.py",
    "validation/truecolor/metrics.py",
    "validation/truecolor/synthetics.py",
    "validation/truecolor/gates.py",
    "validation/truecolor/reference_pixels.py",
    "validation/truecolor/production_adapter.py",
    "validation/truecolor/report.py",
    # Explicit rollback-preservation authorities are not matched by the M2
    # test glob below.  Their exact bytes are load-bearing evidence that M2
    # still leaves the registered M1 path available and honest.
    "tests/test_m1_hanning_policy.py",
    "tests/test_m1_validation_policy_contract.py",
    "tests/test_render_policy.py",
    "tests/test_render_quality.py",
    "tests/test_truecolor_gate_runner.py",
    "tests/test_webp_frames.py",
)


def m2_harness_source_hashes(
    repo_root: Path = production_adapter.REPO_ROOT,
) -> Mapping[str, str]:
    root = repo_root.resolve()
    result: dict[str, str] = {}
    test_files = tuple(
        str(path.relative_to(root))
        for path in sorted((root / "tests").glob("test_m2*.py"))
    )
    if not test_files:
        raise FileNotFoundError("M2 mutation/product tests are absent")
    for name in (*M2_HARNESS_SOURCE_FILES, *test_files):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"M2 harness source is absent: {path}")
        result[name] = sha256_file(path)
    return result


def build_m2_report(
    run: M2OracleRun,
    *,
    research: ResearchInputs,
    repository_identity: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    dependency_metadata: Mapping[str, Any],
    harness_source_hashes: Mapping[str, str],
    command: str = "contract-oracle",
    production_evaluation: Mapping[str, Any] | None = None,
    v30_capture: Mapping[str, Any] | None = None,
    gallery_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    limitations: Sequence[str] | None = None,
) -> dict[str, Any]:
    if command not in {"contract-oracle", "canonical-production"}:
        raise ValueError(f"unregistered M2 report command: {command!r}")
    if command == "canonical-production" and production_evaluation is None:
        raise ValueError("canonical-production report requires bound evaluation")
    if command == "contract-oracle" and any(
        value is not None for value in (production_evaluation, v30_capture)
    ):
        raise ValueError("contract oracle cannot carry production evidence")
    default_limitations = [
        "Synthetic contracts do not close a canonical production-renderer gate.",
        "V08 needs reference pixels executed through the bound M2 finalizer/output.",
        "V16 needs a captured and identity-bound M1 runtime baseline.",
        "V25 needs bound output bytes and manifests from three independent renderer processes.",
        "V30 needs bound real frames and timestamps for a >=96x96 stationary full-daylight sequence.",
        "Loop, explorer pyramid, Web-Mercator, and browser encoded resizes survive until M5.",
        "V44, V45, and V46 remain unmeasured and unclosed.",
    ]
    report: dict[str, Any] = {
        "schema": M2_REPORT_SCHEMA,
        "command": command,
        "summary": {
            "contract_passed": run.contract_passed,
            "stage_advancement_passed": run.stage_advancement_passed,
            "canonical_blockers": list(run.canonical_blockers),
            "mutation_count": len(run.mutations),
            "mutation_kill_count": sum(item.killed for item in run.mutations),
        },
        "claim_boundary": dict(run.claim_boundary),
        "limitations": list(default_limitations if limitations is None else limitations),
        "research_inputs": research.as_dict(),
        "repository": dict(repository_identity),
        "source": dict(source_metadata),
        "dependencies": dict(dependency_metadata),
        "harness_source_sha256": dict(harness_source_hashes),
        "oracle": run.as_dict(),
    }
    if production_evaluation is not None:
        report["production_evaluation"] = dict(production_evaluation)
    if v30_capture is not None:
        report["v30_capture"] = dict(v30_capture)
    if gallery_artifacts is not None:
        report["gallery_artifacts"] = {
            name: dict(row) for name, row in sorted(gallery_artifacts.items())
        }
    payload = stable_json_bytes(report)
    report["evidence_payload_sha256"] = sha256_bytes(payload)
    return report
