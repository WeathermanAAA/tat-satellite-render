"""Deterministic M0 measurements and contract evaluators.

The functions here are deliberately small and renderer-agnostic.  Mutation
tests use them to prove that the validation layer rejects known bad behaviour
before the production path is changed.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .contracts import C18_RAYLEIGH_STATUSES


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    source_columns: int
    destination_columns: int
    contributing_columns: int
    discarded_columns: int
    discarded_fraction: float
    contributor_steps: tuple[int, ...]


def nearest_sample_indices(source_columns: int, destination_columns: int) -> np.ndarray:
    """Return centre-aligned point-sample indices for a regular 1-D grid."""

    if source_columns <= 0 or destination_columns <= 0:
        raise ValueError("source and destination sizes must be positive")
    scale = source_columns / destination_columns
    indices = np.floor((np.arange(destination_columns, dtype=np.float64) + 0.5) * scale)
    return np.clip(indices.astype(np.int64), 0, source_columns - 1)


def source_column_coverage(source_columns: int, destination_columns: int) -> SourceCoverage:
    selected = np.unique(nearest_sample_indices(source_columns, destination_columns))
    steps = tuple(int(value) for value in np.unique(np.diff(selected)))
    contributing = int(selected.size)
    discarded = source_columns - contributing
    return SourceCoverage(
        source_columns=source_columns,
        destination_columns=destination_columns,
        contributing_columns=contributing,
        discarded_columns=discarded,
        discarded_fraction=discarded / source_columns,
        contributor_steps=steps,
    )


def nearest_resample_1d(values: np.ndarray | Sequence[float], destination_size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("nearest_resample_1d expects a one-dimensional field")
    return array[nearest_sample_indices(array.size, destination_size)]


def fractional_area_resample_1d(
    values: np.ndarray | Sequence[float], destination_size: int
) -> np.ndarray:
    """Exact fractional-area average for a regular 1-D grid.

    This is a compact reference implementation for synthetic validation.  It
    is not the production resampler.
    """

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("fractional_area_resample_1d expects a one-dimensional field")
    if destination_size <= 0:
        raise ValueError("destination size must be positive")
    scale = source.size / destination_size
    result = np.empty(destination_size, dtype=np.float64)
    for output_index in range(destination_size):
        start = output_index * scale
        stop = (output_index + 1) * scale
        first = int(np.floor(start))
        last = int(np.ceil(stop))
        weighted_sum = 0.0
        for source_index in range(first, min(last, source.size)):
            overlap = max(0.0, min(stop, source_index + 1.0) - max(start, source_index))
            weighted_sum += float(source[source_index]) * overlap
        result[output_index] = weighted_sum / scale
    return result


def rms_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape:
        raise ValueError("candidate and reference shapes differ")
    return float(np.sqrt(np.mean(np.square(candidate_array - reference_array))))


def high_frequency_power(values: np.ndarray | Sequence[float], cutoff: float = 0.3) -> float:
    if not 0.0 <= cutoff < 0.5:
        raise ValueError("cutoff must lie in [0, 0.5)")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("high_frequency_power expects a one-dimensional field")
    centered = array - np.mean(array)
    power = np.square(np.abs(np.fft.rfft(centered)))
    frequencies = np.fft.rfftfreq(centered.size)
    return float(np.sum(power[frequencies > cutoff]))


def high_frequency_power_ratio(
    candidate: np.ndarray | Sequence[float],
    reference: np.ndarray | Sequence[float],
    cutoff: float = 0.3,
) -> float:
    denominator = high_frequency_power(reference, cutoff)
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("reference contains no measurable high-frequency power")
    return high_frequency_power(candidate, cutoff) / denominator


def explicit_kernel_policy_passes(
    interpolation: str | None,
    *,
    spatial_resample_completed_upstream: bool = False,
    draw_is_one_to_one: bool = False,
) -> bool:
    """V04 policy; automatic aliases are never accepted."""

    normalized = "none" if interpolation is None else interpolation.strip().lower()
    if normalized in {"auto", "antialiased"}:
        return False
    if normalized == "hanning":
        return True
    return (
        spatial_resample_completed_upstream
        and draw_is_one_to_one
        and normalized in {"none", "nearest"}
    )


def log_domain_pair_relative_bias(contrast_ratio: float) -> float:
    """AM-GM bias for an equal-weight pair with values 1 and C."""

    if contrast_ratio <= 0:
        raise ValueError("contrast ratio must be positive")
    return 2.0 * np.sqrt(contrast_ratio) / (1.0 + contrast_ratio) - 1.0


def linear_domain_pair_relative_bias(contrast_ratio: float) -> float:
    if contrast_ratio <= 0:
        raise ValueError("contrast ratio must be positive")
    arithmetic_mean = (1.0 + contrast_ratio) / 2.0
    return arithmetic_mean / arithmetic_mean - 1.0


def production_floor_quantize(values: np.ndarray | Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.floor(clipped * 255.0).astype(np.uint8)


def round_half_up_quantize(values: np.ndarray | Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.floor(clipped * 255.0 + 0.5).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class RingingMeasurement:
    undershoot: float
    overshoot: float

    @property
    def amplitude(self) -> float:
        return max(self.undershoot, self.overshoot)


def out_of_range_fraction(
    candidate: np.ndarray | Sequence[float],
    contributing_min: np.ndarray | Sequence[float] | float,
    contributing_max: np.ndarray | Sequence[float] | float,
    *,
    atol: float = 1e-7,
) -> float:
    values = np.asarray(candidate, dtype=np.float64)
    lower = np.asarray(contributing_min, dtype=np.float64)
    upper = np.asarray(contributing_max, dtype=np.float64)
    if np.any(lower > upper):
        raise ValueError("contributing minimum exceeds maximum")
    invalid = (values < lower - atol) | (values > upper + atol)
    return float(np.mean(invalid))


def edge_ringing(
    candidate: np.ndarray | Sequence[float], source_min: float, source_max: float
) -> RingingMeasurement:
    values = np.asarray(candidate, dtype=np.float64)
    if source_min > source_max:
        raise ValueError("source minimum exceeds source maximum")
    return RingingMeasurement(
        undershoot=max(0.0, float(source_min - np.min(values))),
        overshoot=max(0.0, float(np.max(values) - source_max)),
    )


def noise_variance_ratio(
    candidate: np.ndarray | Sequence[float], reference: np.ndarray | Sequence[float]
) -> float:
    candidate_variance = float(np.var(np.asarray(candidate, dtype=np.float64)))
    reference_variance = float(np.var(np.asarray(reference, dtype=np.float64)))
    if reference_variance <= np.finfo(np.float64).tiny:
        raise ValueError("reference variance is zero")
    return candidate_variance / reference_variance


_FORBIDDEN_SHARPENING_SUBSTRINGS = {
    "unsharp",
    "unsharpmask",
    "clahe",
    "equalizeadapthist",
    "localtonemap",
    "localtonemapping",
    "bloom",
    "glow",
    "edgeenhance",
    "superresolution",
    "deconvolution",
}
_FORBIDDEN_SHARPENING_EXACT_IDENTIFIERS = {
    "sharpen",
    "detailenhance",
}


def _normalized_identifier(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


@dataclass(frozen=True, slots=True)
class ForbiddenOperatorFinding:
    line: int
    identifier: str


def forbidden_sharpening_findings(source: str) -> tuple[ForbiddenOperatorFinding, ...]:
    """Return source locations for forbidden enhancement identifiers.

    ``sharpen`` is an exact-name rule so the legacy ``self_sharpen`` ratio
    stage remains measurable until M4.  The other identifiers are matched as
    normalized substrings so spelling variants such as
    ``local_tone_mapping`` cannot evade V18.
    """

    tree = ast.parse(source)
    findings: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        identifier: str | None = None
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.Attribute):
            identifier = node.attr
        if identifier is None:
            continue
        normalized = _normalized_identifier(identifier)
        if normalized in _FORBIDDEN_SHARPENING_EXACT_IDENTIFIERS or any(
            forbidden in normalized
            for forbidden in _FORBIDDEN_SHARPENING_SUBSTRINGS
        ):
            findings.add((node.lineno, identifier))
    return tuple(
        ForbiddenOperatorFinding(line=line, identifier=identifier)
        for line, identifier in sorted(findings, key=lambda item: (item[0], item[1].casefold()))
    )


def forbidden_sharpening_operators(source: str) -> tuple[str, ...]:
    """Return forbidden callable/name identifiers found in Python source."""

    return tuple(
        sorted(
            {finding.identifier for finding in forbidden_sharpening_findings(source)},
            key=str.casefold,
        )
    )


@dataclass(frozen=True, slots=True)
class ObservationCandidate:
    required_bands: frozenset[str]
    present_bands: frozenset[str]
    fill_fraction: float
    fill_floor: float
    change_token: str | None
    previous_change_token: str | None
    http_status: int = 200
    all_black: bool = False


@dataclass(frozen=True, slots=True)
class LivenessDecision:
    qualifies: bool
    reasons: tuple[str, ...]


def evaluate_changed_observation(candidate: ObservationCandidate) -> LivenessDecision:
    reasons: list[str] = []
    if not 200 <= candidate.http_status < 300:
        reasons.append("http_not_successful")
    if not candidate.required_bands.issubset(candidate.present_bands):
        reasons.append("required_band_missing")
    if candidate.all_black or candidate.fill_fraction <= candidate.fill_floor:
        reasons.append("insufficient_usable_fill")
    if not candidate.change_token or candidate.change_token == candidate.previous_change_token:
        reasons.append("observation_unchanged")
    return LivenessDecision(qualifies=not reasons, reasons=tuple(reasons))


REQUIRED_CACHE_IDENTITY_FIELDS = frozenset(
    {"ALGO_VERSION", "recipe_sha256", "delivered_px"}
)


def missing_cache_identity_fields(identity: Mapping[str, Any]) -> frozenset[str]:
    missing = {
        field
        for field in REQUIRED_CACHE_IDENTITY_FIELDS
        if field not in identity or identity[field] in {None, ""}
    }
    return frozenset(missing)


def cache_identity_digest(identity: Mapping[str, Any]) -> str:
    missing = missing_cache_identity_fields(identity)
    if missing:
        raise ValueError(f"cache identity missing required fields: {sorted(missing)}")
    if not isinstance(identity["delivered_px"], int) or identity["delivered_px"] <= 0:
        raise ValueError("delivered_px must be a positive integer")
    recipe_hash = str(identity["recipe_sha256"])
    if len(recipe_hash) != 64 or any(character not in "0123456789abcdef" for character in recipe_hash):
        raise ValueError("recipe_sha256 must be a lowercase SHA-256 hex digest")
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


class RayleighContractError(RuntimeError):
    pass


SensorKey = tuple[str, str]


def assert_rayleigh_startup_ready(
    configured_sensors: Iterable[SensorKey],
    make_corrector: Callable[[str, str], object | None],
    *,
    data_directory_writable: bool,
) -> Mapping[SensorKey, object]:
    """Call the real factory for every configured sensor and aggregate failures."""

    sensors = tuple(configured_sensors)
    if not sensors:
        raise RayleighContractError("no configured sensors were supplied")
    if len(sensors) != len(set(sensors)):
        raise RayleighContractError("configured sensor list contains duplicates")
    if not data_directory_writable:
        raise RayleighContractError("pyspectral data directory is not writable")

    correctors: dict[SensorKey, object] = {}
    failures: list[str] = []
    for platform, sensor in sensors:
        try:
            corrector = make_corrector(platform, sensor)
        except Exception as exc:  # the contract reports every sensor in one startup failure
            failures.append(f"{platform}/{sensor}: {type(exc).__name__}: {exc}")
            continue
        if corrector is None:
            failures.append(f"{platform}/{sensor}: returned None")
        else:
            correctors[(platform, sensor)] = corrector
    if failures:
        raise RayleighContractError("Rayleigh startup check failed: " + "; ".join(failures))
    return correctors


def rayleigh_frame_status_passes(*, required: bool, status: str) -> bool:
    if status not in C18_RAYLEIGH_STATUSES:
        return False
    if required:
        return status == "applied"
    return status in {"applied", "not_required"}
