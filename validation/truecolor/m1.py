"""M1 explicit-Hanning validation against TAT's real production path.

M1 is a deliberately narrow stopgap: it changes only the two regular-grid
Matplotlib draws from nearest-neighbour sampling to an explicit Hanning
kernel.  The canonical spatial gates in this module consume the pinned
NA_MELISSA_FLOATER ABI granules, build the 1527 x 1600 display array through
TAT's production band assembly and colour recipe, and compare delivered
pixels with an exact fractional-area reference derived from that *same*
display array.  Consequently these measurements score S13 and downstream
only.  They cannot score the upstream crop, registration, cap, or display-
domain defects reserved for M2 and later.

Synthetic measurements remain useful mutation discriminators, but are kept
in a separate evidence object and can never satisfy canonical V01--V03.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import hashlib
import inspect
import io
import os
import tempfile
import textwrap
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence
from unittest import mock

import numpy as np

from . import production_adapter
from .contracts import GATES
from .corpus import load_sample_manifest, sha256_file, verify_object
from .gates import RayleighStartupEvidence, ResearchInputs
from .metrics import high_frequency_power_ratio, rms_error
from .report import collect_dependency_metadata, collect_source_metadata, stable_json_bytes
from .synthetics import alias_stress_columns, deterministic_flat_noise


M1_REPORT_SCHEMA: Final = "tat_truecolor_m1_validation/v2"
M1_SCENE_ID: Final = "NA_MELISSA_FLOATER_20251027_1500"
M1_SAMPLE_MANIFEST_SHA256: Final = (
    "c8083dbef91a5cfa47dfcaa3c410e66b100ad2f357c4d9e6ae3305a9cb3acf76"
)
M1_BBOX: Final = (-84.1, 10.3, -72.1, 22.3)
M1_REQUIRED_ROLES: Final = ("red", "blue", "veggie", "ir")
M1_SOURCE_LONG_AXIS_PX: Final = 1600
M1_DELIVERED_PX: Final = 1188
M1_REQUIRED_GATE_IDS: Final[frozenset[str]] = frozenset(
    {"V01", "V02", "V03", "V04", "V15", "V16", "V18", "V25"}
)
M1_HARNESS_SOURCE_FILES: Final[tuple[str, ...]] = (
    "scripts/tat_truecolor_m1_validate.py",
    "validation/truecolor/__init__.py",
    "validation/truecolor/m1.py",
    "validation/truecolor/m1_gallery.py",
    "validation/truecolor/report.py",
    "validation/truecolor/production_adapter.py",
    "validation/truecolor/gates.py",
    "validation/truecolor/contracts.py",
    "validation/truecolor/metrics.py",
    "validation/truecolor/synthetics.py",
    "validation/truecolor/corpus.py",
    "validation/truecolor/reference_pixels.py",
)
M1_REFERENCE_CONVENTION: Final = (
    "Canonical edge-aligned exact fractional-area average of the same "
    "production 1527x1600 display-encoded array onto the programmatically "
    "captured Agg internal output grid. The array is quantized once with the "
    "declared production-floor quantizer, then sent through M0 nearest solely "
    "to apply the same canvas clip. A unique >=99.8%-identity contiguous "
    "internal-to-delivered slice is required and recorded; no internal size or "
    "offset is hard-coded. This handles Matplotlib 3.9's legacy 1189-row "
    "internal raster and Matplotlib 3.11's 1188-row raster without changing "
    "the scientific reference. This reference "
    "scores S13 and downstream only; S02 striding, S03 co-registration, and "
    "S04 capping are upstream and cannot score against it."
)


@dataclass(frozen=True, slots=True)
class LiteralKernelSite:
    branch_kernel: str | None
    interpolation_literal: str | None
    line: int


@dataclass(frozen=True, slots=True)
class StaticPolicyAudit:
    dispatcher_sites: tuple[LiteralKernelSite, ...]
    render_call_count: int
    render_calls_forward_policy_parameter: bool
    default_policy_is_m1: bool
    m0_rollback_registered: bool
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


@dataclass(frozen=True, slots=True)
class RuntimeKernelTrace:
    policy_version: str
    policy_stage: str
    product_path: str
    interpolations: tuple[str | None, ...]

    @property
    def passed(self) -> bool:
        expected = "nearest" if self.policy_stage == "M0" else "hanning"
        return self.interpolations == (expected,)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


@dataclass(frozen=True, slots=True)
class KernelCoverage:
    kernel: str
    source_pixels: int
    destination_pixels: int
    contributing_source_pixels: int
    discarded_source_pixels: int
    discarded_fraction: float
    minimum_weight: float
    maximum_weight: float
    maximum_weight_sum_error: float
    internal_destination_pixels: int | None = None
    delivered_internal_start: int = 0
    delivered_internal_stop: int | None = None

    @property
    def is_convex_full_coverage(self) -> bool:
        return (
            self.discarded_source_pixels == 0
            and self.minimum_weight >= -1e-7
        )


@dataclass(frozen=True, slots=True)
class SyntheticM1Evidence:
    evidence_kind: str
    source_width: int
    destination_width: int
    nearest_rms_dn: float
    hanning_rms_dn: float
    nearest_high_frequency_ratio: float
    hanning_high_frequency_ratio: float
    hanning_out_of_range_fraction: float
    edge_fixture_nominal_step_dn: float
    edge_fixture_noise_seed: int
    edge_fixture_left_plateau_sigma_dn: float
    edge_fixture_right_plateau_sigma_dn: float
    edge_fixture_edge_index: int
    nearest_edge_step_dn: float
    nearest_edge_overshoot_dn: float
    nearest_edge_undershoot_dn: float
    nearest_edge_ringing_dn: float
    hanning_edge_step_dn: float
    hanning_edge_overshoot_dn: float
    hanning_edge_undershoot_dn: float
    hanning_edge_ringing_dn: float
    flat_noise_variance_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CanonicalObjectIdentity:
    role: str
    file_identifier: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalM1Evidence:
    evidence_kind: str
    scene_id: str
    bbox: tuple[float, float, float, float]
    objects: tuple[CanonicalObjectIdentity, ...]
    source_shape: tuple[int, int, int]
    source_display_sha256: str
    source_lats_sha256: str
    source_lons_sha256: str
    source_geom_valid_sha256: str
    figure_shape: tuple[int, int, int]
    delivered_shape: tuple[int, int, int]
    map_pixel_bounds: tuple[int, int, int, int]
    agg_internal_shape: tuple[int, int, int]
    agg_transform_matrix: tuple[tuple[float, float, float], ...]
    agg_make_image_geometry: Mapping[str, Any]
    delivered_internal_pixel_bounds: tuple[int, int, int, int]
    internal_mapping_candidate_scores: tuple[
        tuple[tuple[int, int, int, int], float], ...
    ]
    sampling_model: str
    sampling_phase: tuple[float, float]
    reference_convention: str
    reference_compositor_mode: str
    reference_quantizer: str
    reference_compositor_caveat: str
    reference_identity_boundary_px: int
    reference_identity_full: float
    reference_identity_scored: float
    production_nearest_canvas_identity_scored: float
    explicit_hanning_canvas_identity_scored: float
    reference_expected_quantized_sha256: str
    production_nearest_rms_dn: float
    explicit_hanning_rms_dn: float
    error_removed_same_array_pct: float
    production_nearest_high_frequency_ratio: float
    explicit_hanning_high_frequency_ratio: float
    spectral_definition: Mapping[str, Any]
    hanning_global_out_of_range_fraction: float
    hanning_local_neighbourhood_out_of_range_fraction: float
    hanning_local_neighbourhood_comparisons: int
    hanning_local_neighbourhood_tolerance: float
    hanning_actual_phase_minimum_weight: float
    hanning_actual_phase_maximum_weight_sum_error: float
    hanning_actual_transform_maximum_dc_error: float
    hanning_float_to_rendered_identity_boundary_px: int
    hanning_float_to_rendered_identity_full: float
    hanning_float_to_rendered_identity_scored: float
    production_nearest_png_sha256: str
    explicit_hanning_png_sha256: str
    bandlimited_reference_png_sha256: str
    production_nearest_raster_sha256: str
    explicit_hanning_raster_sha256: str
    bandlimited_reference_raster_sha256: str
    repeatability_sha256: tuple[str, ...]
    repeatability_process_ids: tuple[int, ...]
    repeatability_start_method: str
    runtime_interpolations: Mapping[str, tuple[str | None, ...]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M1GateEvidence:
    gate_id: str
    name: str
    threshold: str
    outcome: str
    blocking: bool
    evidence_kind: str
    detail: str
    measurements: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RenderedRaster:
    png: bytes
    raster: np.ndarray
    interpolations: tuple[str | None, ...]
    agg_traces: tuple["AggResampleTrace", ...]
    make_image_traces: tuple["MakeImageTrace", ...]
    internal_resampled_rgb: np.ndarray | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    internal_make_image_rgb: np.ndarray | None = dataclasses.field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class AggResampleTrace:
    interpolation: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, int]
    transform_matrix: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class MakeImageTrace:
    interpolation: str
    input_shape: tuple[int, ...]
    in_bbox_bounds: tuple[float, float, float, float]
    out_bbox_bounds: tuple[float, float, float, float]
    clip_bbox_bounds: tuple[float, float, float, float]
    magnification: float
    unsampled: bool
    round_to_pixel_border: bool
    returned_rgba_shape: tuple[int, int, int]
    returned_origin: tuple[float, float]
    returned_transform_matrix: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class LocalNeighbourhoodRange:
    out_of_range_fraction: float
    comparisons: int
    tolerance: float
    minimum_weight: float
    maximum_weight_sum_error: float
    maximum_dc_error: float
    float_to_rendered_identity_boundary_px: int
    float_to_rendered_identity_full: float
    float_to_rendered_identity_scored: float


@dataclass(frozen=True, slots=True)
class InternalCanvasMapping:
    pixel_bounds: tuple[int, int, int, int]
    boundary_exclusion_px: int
    full_identity: float
    scored_identity: float
    candidate_scores: tuple[tuple[tuple[int, int, int, int], float], ...]


@dataclass(frozen=True, slots=True)
class M1RenderedBundle:
    production_nearest: np.ndarray
    explicit_hanning: np.ndarray
    bandlimited_reference: np.ndarray


@dataclass(frozen=True, slots=True)
class M1Run:
    research: ResearchInputs
    static_policy: StaticPolicyAudit
    runtime_traces: tuple[RuntimeKernelTrace, ...]
    horizontal_coverage: KernelCoverage
    vertical_coverage: KernelCoverage
    synthetic: SyntheticM1Evidence
    canonical: CanonicalM1Evidence | None
    rayleigh_startup: RayleighStartupEvidence
    gates: tuple[M1GateEvidence, ...]
    rendered: M1RenderedBundle | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    @property
    def gate_id_contract_valid(self) -> bool:
        gate_ids = tuple(gate.gate_id for gate in self.gates)
        return (
            len(gate_ids) == len(M1_REQUIRED_GATE_IDS)
            and frozenset(gate_ids) == M1_REQUIRED_GATE_IDS
        )

    @property
    def stage_advancement_passed(self) -> bool:
        """Whether the research-defined M1 stage may advance to M2.

        M1 records V01's hard threshold honestly, but the canonical stage
        table permits a measured V01 improvement to be non-blocking until M2.
        Every other assigned gate must pass, and the exact gate-ID contract is
        fail-closed so a missing or duplicate gate can never advance a stage.
        """

        return self.gate_id_contract_valid and all(
            not gate.blocking for gate in self.gates
        )

    @property
    def all_thresholds_passed(self) -> bool:
        return self.gate_id_contract_valid and all(
            gate.outcome == "PASS" for gate in self.gates
        )

    @property
    def threshold_fail_gate_ids(self) -> tuple[str, ...]:
        return tuple(gate.gate_id for gate in self.gates if gate.outcome == "FAIL")

    @property
    def unmeasured_gate_ids(self) -> tuple[str, ...]:
        return tuple(
            gate.gate_id for gate in self.gates if gate.outcome == "UNMEASURED"
        )

    @property
    def enforcement_passed(self) -> bool:
        """Backward-compatible name for M1 stage advancement, not all thresholds."""

        return self.stage_advancement_passed


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def m1_harness_source_hashes() -> dict[str, str]:
    """Hash every source file that generates or interprets M1 evidence."""

    root = production_adapter.REPO_ROOT.resolve()
    hashes: dict[str, str] = {}
    for relative_name in M1_HARNESS_SOURCE_FILES:
        path = root / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"required M1 harness source is absent: {path}")
        hashes[relative_name] = _sha256_bytes(path.read_bytes())
    return hashes


def find_internal_canvas_mapping(
    internal_raster: np.ndarray,
    delivered_raster: np.ndarray,
    *,
    boundary_exclusion_px: int = 1,
) -> InternalCanvasMapping:
    """Find the unique contiguous internal Agg slice delivered by the canvas."""

    internal = np.asarray(internal_raster, dtype=np.uint8)
    delivered = np.asarray(delivered_raster, dtype=np.uint8)
    if internal.ndim != 3 or delivered.ndim != 3:
        raise ValueError("canvas mapping expects RGB rasters")
    if internal.shape[-1] != 3 or delivered.shape[-1] != 3:
        raise ValueError("canvas mapping expects exactly three channels")
    excess_y = internal.shape[0] - delivered.shape[0]
    excess_x = internal.shape[1] - delivered.shape[1]
    if excess_y < 0 or excess_x < 0:
        raise RuntimeError("internal Agg raster is smaller than the delivered map")
    if excess_y > 8 or excess_x > 8:
        raise RuntimeError(
            "unexpectedly large internal/delivered geometry difference: "
            f"internal={internal.shape}, delivered={delivered.shape}"
        )
    candidates: list[tuple[tuple[int, int, int, int], float, float]] = []
    boundary = boundary_exclusion_px
    for row_start in range(excess_y + 1):
        for column_start in range(excess_x + 1):
            bounds = (
                row_start,
                row_start + delivered.shape[0],
                column_start,
                column_start + delivered.shape[1],
            )
            candidate = internal[
                bounds[0] : bounds[1],
                bounds[2] : bounds[3],
            ]
            equal = np.all(candidate == delivered, axis=-1)
            if boundary:
                scored = equal[boundary:-boundary, boundary:-boundary]
            else:
                scored = equal
            candidates.append((bounds, float(equal.mean()), float(scored.mean())))
    passing = [candidate for candidate in candidates if candidate[2] >= 0.998]
    if len(passing) != 1:
        raise RuntimeError(
            "internal-to-canvas mapping is not a unique >=99.8% identity: "
            + repr([(bounds, scored) for bounds, _, scored in candidates])
        )
    bounds, full_identity, scored_identity = passing[0]
    return InternalCanvasMapping(
        pixel_bounds=bounds,
        boundary_exclusion_px=boundary,
        full_identity=full_identity,
        scored_identity=scored_identity,
        candidate_scores=tuple(
            (candidate_bounds, candidate_scored)
            for candidate_bounds, _, candidate_scored in candidates
        ),
    )


def internal_to_canvas_orientation(
    array: np.ndarray,
    trace: AggResampleTrace,
) -> np.ndarray:
    """Orient an Agg-internal raster like the top-down encoded canvas."""

    result = np.asarray(array)
    matrix = np.asarray(trace.transform_matrix, dtype=np.float64)
    if matrix[1, 1] < 0.0:
        result = np.flip(result, axis=0)
    if matrix[0, 0] < 0.0:
        result = np.flip(result, axis=1)
    return result


def _literal_interpolation(call: ast.Call) -> str | None:
    keyword = next(
        (item.value for item in call.keywords if item.arg == "interpolation"),
        None,
    )
    if isinstance(keyword, ast.Constant) and isinstance(keyword.value, str):
        return keyword.value
    return None


def _kernel_compare_value(test: ast.expr) -> str | None:
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and isinstance(test.comparators[0].value, str)
        and isinstance(test.left, ast.Attribute)
        and test.left.attr == "resample_kernel"
    ):
        return None
    return test.comparators[0].value


def dispatcher_literal_sites(source: str) -> tuple[LiteralKernelSite, ...]:
    """Enumerate literal imshow calls with their guarding policy branch."""

    tree = ast.parse(textwrap.dedent(source))
    sites: list[LiteralKernelSite] = []

    def visit(nodes: Sequence[ast.stmt], branch: str | None = None) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node.body, branch)
                continue
            if isinstance(node, ast.If):
                declared = _kernel_compare_value(node.test)
                visit(node.body, declared if declared is not None else branch)
                visit(node.orelse, branch)
                continue
            for child in ast.walk(node):
                if not (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "imshow"
                ):
                    continue
                sites.append(
                    LiteralKernelSite(
                        branch_kernel=branch,
                        interpolation_literal=_literal_interpolation(child),
                        line=child.lineno,
                    )
                )

    visit(tree.body)
    return tuple(sorted(sites, key=lambda item: item.line))


def render_policy_forwarding_issues(source: str) -> tuple[int, tuple[str, ...]]:
    """Reject a production site that omits policy or hardcodes M0/M1."""

    tree = ast.parse(textwrap.dedent(source))
    calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_imshow_with_render_policy"
    )
    issues: list[str] = []
    if len(calls) != 2:
        issues.append(f"render_png has {len(calls)} dispatcher calls, expected 2")
    for call in calls:
        value = next((item.value for item in call.keywords if item.arg == "policy"), None)
        if not isinstance(value, ast.Name) or value.id != "policy":
            issues.append(
                f"render dispatcher call at line {call.lineno} does not forward policy=policy"
            )
    return len(calls), tuple(issues)


def audit_static_policy_sources(
    dispatcher_source: str,
    render_png_source: str,
    *,
    default_policy_is_m1: bool,
    m0_rollback_registered: bool,
) -> StaticPolicyAudit:
    sites = dispatcher_literal_sites(dispatcher_source)
    call_count, forwarding_issues = render_policy_forwarding_issues(render_png_source)
    issues = list(forwarding_issues)
    expected = {"nearest": "nearest", "hanning": "hanning"}
    observed: dict[str | None, str | None] = {}
    for site in sites:
        if site.branch_kernel in observed:
            issues.append(f"duplicate dispatcher arm for {site.branch_kernel!r}")
        observed[site.branch_kernel] = site.interpolation_literal
        if site.interpolation_literal in {None, "auto", "antialiased"}:
            issues.append(
                f"dispatcher line {site.line} has non-literal/automatic interpolation"
            )
    if observed != expected:
        issues.append(f"dispatcher literal map is {observed!r}, expected {expected!r}")
    if not default_policy_is_m1:
        issues.append("default render policy is not the registered M1 policy")
    if not m0_rollback_registered:
        issues.append("M0 nearest rollback policy is not registered")
    return StaticPolicyAudit(
        dispatcher_sites=sites,
        render_call_count=call_count,
        render_calls_forward_policy_parameter=not forwarding_issues,
        default_policy_is_m1=default_policy_is_m1,
        m0_rollback_registered=m0_rollback_registered,
        issues=tuple(issues),
    )


def audit_production_policy() -> StaticPolicyAudit:
    modules = production_adapter.production_modules()
    render = modules["render"]
    import render_policy

    return audit_static_policy_sources(
        inspect.getsource(render._imshow_with_render_policy),
        inspect.getsource(render.render_png),
        default_policy_is_m1=(
            render_policy.resolve_render_policy() is render_policy.M1_POLICY
            and render_policy.ALGO_VERSION == render_policy.M1_POLICY.algo_version
        ),
        m0_rollback_registered=(
            render_policy.POLICIES.get(render_policy.M0_POLICY.algo_version)
            is render_policy.M0_POLICY
            and render_policy.M0_POLICY.resample_kernel == "nearest"
        ),
    )


def fractional_area_resample_axis(
    values: np.ndarray,
    destination_size: int,
    *,
    axis: int,
    sampling_phase: float = 0.5,
) -> np.ndarray:
    """Exact overlap average along one axis, including device-phase offset.

    Output pixel centres are ``(i + sampling_phase) * input/output``.  A
    footprint extending beyond the image is normalized by its valid overlap,
    which is equivalent to edge support for a constant field and preserves
    DC.  The canonical research reference always uses phase 0.5; the phase
    argument exists for focused reference-operator tests, not to move the
    canonical grid to Matplotlib's figure-anchor phase.
    """

    source = np.asarray(values, dtype=np.float64)
    if source.ndim == 0:
        raise ValueError("fractional-area input must have at least one axis")
    if destination_size <= 0:
        raise ValueError("destination size must be positive")
    if not 0.0 <= sampling_phase <= 1.0:
        raise ValueError("sampling phase must be in [0, 1]")
    moved = np.moveaxis(source, axis, 0)
    source_size = moved.shape[0]
    if source_size <= 0:
        raise ValueError("source size must be positive")
    scale = source_size / destination_size
    result = np.empty((destination_size, *moved.shape[1:]), dtype=np.float64)
    for output_index in range(destination_size):
        start = (output_index + sampling_phase - 0.5) * scale
        stop = (output_index + sampling_phase + 0.5) * scale
        first = max(0, int(np.floor(start)))
        last = min(source_size, int(np.ceil(stop)))
        accumulator = np.zeros(moved.shape[1:], dtype=np.float64)
        valid_overlap = 0.0
        for source_index in range(first, last):
            overlap = max(
                0.0,
                min(stop, source_index + 1.0) - max(start, source_index),
            )
            accumulator += moved[source_index] * overlap
            valid_overlap += overlap
        if valid_overlap <= 0.0:
            raise RuntimeError("fractional-area footprint has no valid support")
        result[output_index] = accumulator / valid_overlap
    return np.moveaxis(result, 0, axis)


def fractional_area_resample(
    values: np.ndarray,
    output_shape: tuple[int, int],
    *,
    sampling_phase: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Separable exact fractional-area average of an HxW[xC] raster."""

    output_height, output_width = output_shape
    horizontal = fractional_area_resample_axis(
        values,
        output_width,
        axis=1,
        sampling_phase=sampling_phase[1],
    )
    return fractional_area_resample_axis(
        horizontal,
        output_height,
        axis=0,
        sampling_phase=sampling_phase[0],
    )


def matplotlib_kernel_coverage(
    source_pixels: int,
    destination_pixels: int,
    kernel: str,
) -> KernelCoverage:
    """Synthetic/test-only legacy geometry probe for an isolated Agg axis.

    Canonical M1 evidence uses ``kernel_coverage_from_trace`` with the captured
    production transform, internal output shape, and delivered canvas slice.
    """

    if source_pixels <= 0 or destination_pixels <= 0:
        raise ValueError("source and destination sizes must be positive")
    if kernel not in {"nearest", "hanning", "auto", "antialiased"}:
        raise ValueError(f"unsupported measured kernel: {kernel!r}")
    import matplotlib.image as mpl_image
    from matplotlib.figure import Figure
    from matplotlib.transforms import Affine2D

    impulses = np.eye(source_pixels, dtype=np.float32)
    figure = Figure(figsize=(1.0, 1.0))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    image = axes.imshow(impulses, interpolation=kernel)
    responses = np.asarray(
        mpl_image._resample(  # noqa: SLF001 - this is the measured production engine
            image,
            impulses,
            (source_pixels, destination_pixels),
            Affine2D().scale(destination_pixels / source_pixels, 1.0),
        ),
        dtype=np.float64,
    )
    per_source = np.max(np.abs(responses), axis=1)
    contributing = int(np.count_nonzero(per_source > np.finfo(np.float32).eps))
    discarded = source_pixels - contributing
    column_sums = responses.sum(axis=0)
    return KernelCoverage(
        kernel=kernel,
        source_pixels=source_pixels,
        destination_pixels=destination_pixels,
        contributing_source_pixels=contributing,
        discarded_source_pixels=discarded,
        discarded_fraction=discarded / source_pixels,
        minimum_weight=float(responses.min()),
        maximum_weight=float(responses.max()),
        maximum_weight_sum_error=float(np.max(np.abs(column_sums - 1.0))),
        internal_destination_pixels=destination_pixels,
        delivered_internal_start=0,
        delivered_internal_stop=destination_pixels,
    )


def matplotlib_axis_response(
    source_pixels: int,
    destination_pixels: int,
    kernel: str,
    *,
    sampling_phase: float,
) -> np.ndarray:
    """Return destination-by-source Agg weights at an explicit device phase."""

    if source_pixels <= 0 or destination_pixels <= 0:
        raise ValueError("response dimensions must be positive")
    if not 0.0 <= sampling_phase <= 1.0:
        raise ValueError("sampling phase must be in [0, 1]")
    import matplotlib.image as mpl_image
    from matplotlib.figure import Figure
    from matplotlib.transforms import Affine2D

    impulses = np.eye(source_pixels, dtype=np.float32)
    figure = Figure(figsize=(1.0, 1.0))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    image = axes.imshow(impulses, interpolation=kernel)
    transform = (
        Affine2D()
        .scale(destination_pixels / source_pixels, 1.0)
        .translate(0.5 - sampling_phase, 0.0)
    )
    response = mpl_image._resample(  # noqa: SLF001 - production Agg engine
        image,
        impulses,
        (source_pixels, destination_pixels),
        transform,
    )
    return np.asarray(response, dtype=np.float64).T


def matplotlib_axis_response_from_transform(
    source_pixels: int,
    destination_pixels: int,
    kernel: str,
    *,
    scale: float,
    translation: float,
    axis: int,
) -> np.ndarray:
    """Measure one separable Agg axis with the captured production affine."""

    if axis not in {0, 1}:
        raise ValueError("response axis must be 0 or 1")
    import matplotlib.image as mpl_image
    from matplotlib.figure import Figure
    from matplotlib.transforms import Affine2D

    impulses = np.eye(source_pixels, dtype=np.float32)
    figure = Figure(figsize=(1.0, 1.0))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    image = axes.imshow(impulses, interpolation=kernel, origin="upper")
    if axis == 1:
        transform = Affine2D.from_values(scale, 0.0, 0.0, 1.0, translation, 0.0)
        result = mpl_image._resample(  # noqa: SLF001
            image,
            impulses,
            (source_pixels, destination_pixels),
            transform,
        )
        response = np.asarray(result, dtype=np.float64).T
    else:
        transform = Affine2D.from_values(1.0, 0.0, 0.0, scale, 0.0, translation)
        result = mpl_image._resample(  # noqa: SLF001
            image,
            impulses,
            (destination_pixels, source_pixels),
            transform,
        )
        response = np.asarray(result, dtype=np.float64)
    if response.shape != (destination_pixels, source_pixels):
        raise RuntimeError(f"unexpected axis response shape: {response.shape}")
    return response


def kernel_coverage_from_trace(
    trace: AggResampleTrace,
    source_shape: tuple[int, int],
    delivered_internal_bounds: tuple[int, int, int, int],
) -> tuple[KernelCoverage, KernelCoverage]:
    """V02/V15 coverage for the exact production internal Agg operator."""

    source_height, source_width = source_shape
    output_height, output_width = trace.output_shape
    matrix = np.asarray(trace.transform_matrix, dtype=np.float64)
    response_y = matplotlib_axis_response_from_transform(
        source_height,
        output_height,
        trace.interpolation,
        scale=float(matrix[1, 1]),
        translation=float(matrix[1, 2]),
        axis=0,
    )
    response_x = matplotlib_axis_response_from_transform(
        source_width,
        output_width,
        trace.interpolation,
        scale=float(matrix[0, 0]),
        translation=float(matrix[0, 2]),
        axis=1,
    )
    if matrix[1, 1] < 0.0:
        response_y = response_y[::-1]
    if matrix[0, 0] < 0.0:
        response_x = response_x[::-1]
    row_start, row_stop, column_start, column_stop = delivered_internal_bounds
    response_y = response_y[row_start:row_stop]
    response_x = response_x[column_start:column_stop]

    def summarize(
        response: np.ndarray,
        source: int,
        destination: int,
        internal_destination: int,
        delivered_start: int,
        delivered_stop: int,
    ) -> KernelCoverage:
        contributing = int(
            np.count_nonzero(
                np.max(np.abs(response), axis=0) > np.finfo(np.float32).eps
            )
        )
        return KernelCoverage(
            kernel=trace.interpolation,
            source_pixels=source,
            destination_pixels=destination,
            contributing_source_pixels=contributing,
            discarded_source_pixels=source - contributing,
            discarded_fraction=(source - contributing) / source,
            minimum_weight=float(response.min()),
            maximum_weight=float(response.max()),
            maximum_weight_sum_error=float(
                np.max(np.abs(response.sum(axis=1) - 1.0))
            ),
            internal_destination_pixels=internal_destination,
            delivered_internal_start=delivered_start,
            delivered_internal_stop=delivered_stop,
        )

    return (
        summarize(
            response_x,
            source_width,
            column_stop - column_start,
            output_width,
            column_start,
            column_stop,
        ),
        summarize(
            response_y,
            source_height,
            row_stop - row_start,
            output_height,
            row_start,
            row_stop,
        ),
    )


def _float_hanning_at_geometry(
    source_rgb: np.ndarray,
    output_shape: tuple[int, int],
    sampling_phase: tuple[float, float],
) -> np.ndarray:
    """Run the production Agg RGBA resampler without 8-bit quantization."""

    import matplotlib.image as mpl_image
    from matplotlib.figure import Figure
    from matplotlib.transforms import Affine2D

    source = np.asarray(source_rgb, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("float Hanning probe expects HxWx3 RGB")
    output_height, output_width = output_shape
    phase_y, phase_x = sampling_phase
    rgba = np.concatenate(
        [source, np.ones((*source.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    figure = Figure(figsize=(1.0, 1.0))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    image = axes.imshow(rgba, interpolation="hanning", origin="upper")
    transform = (
        Affine2D()
        .scale(output_width / source.shape[1], output_height / source.shape[0])
        .translate(0.5 - phase_x, 0.5 - phase_y)
    )
    result = mpl_image._resample(  # noqa: SLF001 - production Agg engine
        image,
        rgba,
        (output_height, output_width),
        transform,
    )
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (output_height, output_width, 4):
        raise RuntimeError(f"unexpected Agg Hanning result shape: {result.shape}")
    if not np.allclose(result[..., 3], 1.0, atol=2e-6):
        raise RuntimeError("Agg Hanning probe produced non-opaque alpha")
    return result[..., :3]


def _float_resample_from_trace(
    source_rgb: np.ndarray,
    trace: AggResampleTrace,
) -> np.ndarray:
    """Run Agg float RGBA with the exact affine captured in production."""

    import matplotlib.image as mpl_image
    from matplotlib.figure import Figure
    from matplotlib.transforms import Affine2D

    source = np.asarray(source_rgb, dtype=np.float32)
    if source.shape != trace.input_shape[:2] + (3,):
        raise ValueError(
            f"trace input is {trace.input_shape}, received RGB {source.shape}"
        )
    rgba = np.concatenate(
        [source, np.ones((*source.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    figure = Figure(figsize=(1.0, 1.0))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    image = axes.imshow(rgba, interpolation=trace.interpolation, origin="upper")
    transform = Affine2D(np.asarray(trace.transform_matrix, dtype=np.float64))
    result = mpl_image._resample(  # noqa: SLF001
        image,
        rgba,
        trace.output_shape,
        transform,
    )
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (*trace.output_shape, 4):
        raise RuntimeError(f"trace replay returned unexpected shape: {result.shape}")
    return result[..., :3]


def local_neighbourhood_out_of_range_fraction(
    source_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    support_y: Sequence[np.ndarray],
    support_x: Sequence[np.ndarray],
    *,
    tolerance: float,
    sampling_phase: tuple[float, float] = (0.5, 0.5),
    background_y: Sequence[bool] | None = None,
    background_x: Sequence[bool] | None = None,
) -> tuple[float, int]:
    """Count candidate channels outside their contributing local min/max."""

    source = np.asarray(source_rgb, dtype=np.float64)
    candidate = np.asarray(candidate_rgb, dtype=np.float64)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("local range source must be HxWx3 RGB")
    if candidate.shape != (len(support_y), len(support_x), 3):
        raise ValueError("local range candidate/support shape mismatch")
    if tolerance < 0.0:
        raise ValueError("local range tolerance must be non-negative")
    if background_y is not None and len(background_y) != len(support_y):
        raise ValueError("vertical background flags do not match support")
    if background_x is not None and len(background_x) != len(support_x):
        raise ValueError("horizontal background flags do not match support")
    violations = 0
    comparisons = candidate.size
    for output_y, source_y in enumerate(support_y):
        if source_y.size == 0:
            raise ValueError("empty vertical source support")
        row_source = source[source_y, :, :]
        for output_x, source_x in enumerate(support_x):
            if source_x.size == 0:
                raise ValueError("empty horizontal source support")
            neighbourhood = row_source[:, source_x, :]
            lower = np.min(neighbourhood, axis=(0, 1))
            upper = np.max(neighbourhood, axis=(0, 1))
            if (
                (background_y is not None and background_y[output_y])
                or (background_x is not None and background_x[output_x])
                or (
                (output_y == 0 and sampling_phase[0] < 0.5)
                or (output_y == len(support_y) - 1 and sampling_phase[0] > 0.5)
                or (output_x == 0 and sampling_phase[1] < 0.5)
                or (output_x == len(support_x) - 1 and sampling_phase[1] > 0.5)
                )
            ):
                lower = np.minimum(lower, 0.0)
                upper = np.maximum(upper, 0.0)
            value = candidate[output_y, output_x]
            violations += int(
                np.count_nonzero(
                    (value < lower - tolerance) | (value > upper + tolerance)
                )
            )
    return violations / comparisons, comparisons


def measure_local_neighbourhood_range(
    source_rgb: np.ndarray,
    rendered_raster: np.ndarray,
    *,
    trace: AggResampleTrace,
    internal_resampled_rgb: np.ndarray,
    delivered_internal_bounds: tuple[int, int, int, int],
    tolerance: float = 2e-6,
) -> LocalNeighbourhoodRange:
    """Count V15 violations against each output's actual source support.

    The support is measured from Agg impulse responses at the installed
    figure phase.  Every output channel is compared with the min/max of only
    its contributing source neighbourhood.  A clipped footprint includes the
    compositor's zero background in its local range.
    """

    source = np.asarray(source_rgb, dtype=np.float32)
    rendered = np.asarray(rendered_raster, dtype=np.uint8)
    output_height, output_width = trace.output_shape
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("local range source must be HxWx3 RGB")
    row_start, row_stop, column_start, column_stop = delivered_internal_bounds
    if rendered.shape != (row_stop - row_start, column_stop - column_start, 3):
        raise ValueError("local range delivered raster shape mismatch")
    matrix = np.asarray(trace.transform_matrix, dtype=np.float64)
    response_y = matplotlib_axis_response_from_transform(
        source.shape[0],
        output_height,
        trace.interpolation,
        scale=float(matrix[1, 1]),
        translation=float(matrix[1, 2]),
        axis=0,
    )
    response_x = matplotlib_axis_response_from_transform(
        source.shape[1],
        output_width,
        trace.interpolation,
        scale=float(matrix[0, 0]),
        translation=float(matrix[0, 2]),
        axis=1,
    )
    if matrix[1, 1] < 0.0:
        response_y = response_y[::-1]
    if matrix[0, 0] < 0.0:
        response_x = response_x[::-1]
    minimum_weight = float(min(response_y.min(), response_x.min()))
    maximum_weight_sum_error = float(
        max(
            np.max(np.abs(response_y.sum(axis=1) - 1.0)),
            np.max(np.abs(response_x.sum(axis=1) - 1.0)),
        )
    )
    support_epsilon = np.finfo(np.float32).eps
    support_y_all = [
        np.flatnonzero(np.abs(row) > support_epsilon) for row in response_y
    ]
    support_x_all = [
        np.flatnonzero(np.abs(row) > support_epsilon) for row in response_x
    ]
    support_y = support_y_all[row_start:row_stop]
    support_x = support_x_all[column_start:column_stop]
    if any(indices.size == 0 for indices in (*support_y, *support_x)):
        raise RuntimeError("Agg Hanning response contains an output with no support")

    internal_candidate = internal_to_canvas_orientation(
        np.asarray(internal_resampled_rgb, dtype=np.float64),
        trace,
    )
    if internal_candidate.shape != (*trace.output_shape, 3):
        raise ValueError("captured internal candidate shape does not match trace")
    candidate = internal_candidate[
        row_start:row_stop,
        column_start:column_stop,
    ]
    background_y = response_y.sum(axis=1)[row_start:row_stop] < 0.999
    background_x = response_x.sum(axis=1)[column_start:column_stop] < 0.999
    out_of_range, comparisons = local_neighbourhood_out_of_range_fraction(
        source,
        candidate,
        support_y,
        support_x,
        tolerance=tolerance,
        background_y=background_y,
        background_x=background_x,
    )

    maximum_dc_error = 0.0
    for level in (0.1, 0.5, 1.0):
        constant = np.full(source.shape, level, dtype=np.float32)
        constant_result = internal_to_canvas_orientation(
            _float_resample_from_trace(constant, trace),
            trace,
        )
        constant_delivered = constant_result[
            row_start:row_stop,
            column_start:column_stop,
        ]
        maximum_dc_error = max(
            maximum_dc_error,
            float(np.max(np.abs(constant_delivered - level))),
        )

    expected_u8 = np.floor(np.clip(candidate, 0.0, 1.0) * 255.0).astype(np.uint8)
    identity_pixels = np.all(expected_u8 == rendered, axis=-1)
    identity_boundary = 1
    identity_scored = identity_pixels[
        identity_boundary:-identity_boundary,
        identity_boundary:-identity_boundary,
    ]
    return LocalNeighbourhoodRange(
        out_of_range_fraction=out_of_range,
        comparisons=comparisons,
        tolerance=tolerance,
        minimum_weight=minimum_weight,
        maximum_weight_sum_error=maximum_weight_sum_error,
        maximum_dc_error=maximum_dc_error,
        float_to_rendered_identity_boundary_px=identity_boundary,
        float_to_rendered_identity_full=float(identity_pixels.mean()),
        float_to_rendered_identity_scored=float(identity_scored.mean()),
    )


def _agg_canvas_resample_columns(
    values: np.ndarray,
    destination_size: int,
    kernel: str,
    *,
    height: int = 16,
) -> np.ndarray:
    """Drive the same Agg RGB resampler at an exact width and return uint8."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    source = np.asarray(values, dtype=np.float32)
    rgb = np.broadcast_to(source[None, :, None], (height, source.size, 3)).copy()
    dpi = 100
    figure = Figure(
        figsize=(destination_size / dpi, height / dpi),
        dpi=dpi,
        frameon=False,
    )
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.imshow(rgb, origin="upper", aspect="auto", interpolation=kernel)
    axes.set_axis_off()
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[height // 2, :, 0].copy()


def _float_rgb_resample_columns(
    values: np.ndarray,
    destination_size: int,
    kernel: str,
    *,
    height: int = 32,
) -> np.ndarray:
    import matplotlib.image as mpl_image
    from matplotlib.figure import Figure
    from matplotlib.transforms import Affine2D

    source = np.asarray(values, dtype=np.float32)
    rgb = np.broadcast_to(source[None, :, None], (height, source.size, 3)).copy()
    rgba = np.concatenate(
        [rgb, np.ones((*rgb.shape[:2], 1), dtype=np.float32)], axis=-1
    )
    figure = Figure(figsize=(1.0, 1.0))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    image = axes.imshow(rgba, interpolation=kernel)
    result = mpl_image._resample(  # noqa: SLF001 - measured production engine
        image,
        rgba,
        (height, destination_size),
        Affine2D().scale(destination_size / source.size, 1.0),
    )
    return np.asarray(result[height // 2, :, 0], dtype=np.float64)


def canonical_radial_power_profile(
    values: np.ndarray,
    *,
    bins: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical V03 radial PSD from the audited research metric.

    This intentionally preserves every consequential convention in
    ``prototypes/metrics.py::radial_power`` without importing the research
    prototype: Rec.601 luma, top-left square crop, mean removal, a separable
    two-dimensional Hann spatial window, full shifted FFT, 64 equal radial
    bins from zero through axial Nyquist, and the mean PSD within each bin.
    Corner frequencies above 0.5 cycles/pixel are outside those bins.
    """

    if bins <= 0:
        raise ValueError("radial bin count must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[-1] != 3:
            raise ValueError("radial RGB input must have exactly three channels")
        luma = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )
    elif array.ndim == 2:
        luma = array
    else:
        raise ValueError("radial power expects HxW or HxWx3 values")
    if min(luma.shape) < 2:
        raise ValueError("radial power input is too small")
    luma = luma - np.mean(luma)
    size = min(luma.shape)
    luma = luma[:size, :size]
    window = np.hanning(size)
    luma = luma * window[:, None] * window[None, :]
    power = np.fft.fftshift(np.abs(np.fft.fft2(luma)) ** 2)
    frequency = np.fft.fftshift(np.fft.fftfreq(size))
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    edges = np.linspace(0.0, 0.5, bins + 1)
    indices = np.digitize(radius.ravel(), edges) - 1
    profile = np.empty(bins, dtype=np.float64)
    flattened = power.ravel()
    for index in range(bins):
        selected = indices == index
        profile[index] = (
            float(np.mean(flattened[selected])) if np.any(selected) else np.nan
        )
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, profile


def radial_high_frequency_power(
    values: np.ndarray,
    *,
    cutoff: float = 0.3,
    bins: int = 64,
) -> float:
    """Sum canonical radial mean-PSD bins strictly above ``cutoff``."""

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
        raise ValueError("reference has no measurable high-frequency power")
    return (
        radial_high_frequency_power(candidate, cutoff=cutoff, bins=bins)
        / denominator
    )


@dataclass(frozen=True, slots=True)
class EdgeOvershootEvidence:
    step_dn: float
    overshoot_dn: float
    undershoot_dn: float
    ringing_dn: float


def local_edge_overshoot(
    profile: np.ndarray,
    edge_index: int,
    *,
    window: int = 12,
) -> EdgeOvershootEvidence:
    """Canonical plateau-relative V16 overshoot/undershoot in DN.

    Plateaus exclude the three samples next to the transition.  Unlike a
    global min/max guard, this detects a ringing excursion that remains well
    inside the image's unrelated global range.
    """

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("edge profile must be one-dimensional")
    left_start = max(0, edge_index - window)
    left_stop = max(0, edge_index - 3)
    right_start = min(values.size, edge_index + 3)
    right_stop = min(values.size, edge_index + window)
    if left_stop <= left_start or right_stop <= right_start:
        raise ValueError("edge profile does not contain both plateau windows")
    left_values = values[left_start:left_stop]
    right_values = values[right_start:right_stop]
    left = float(np.median(left_values))
    right = float(np.median(right_values))
    signed_step = right - left
    if abs(signed_step) < 1e-9:
        raise ValueError("edge plateau step is zero")
    if signed_step > 0.0:
        high, low = right, left
        high_side, low_side = right_values, left_values
    else:
        high, low = left, right
        high_side, low_side = left_values, right_values
    overshoot = max(0.0, float(np.max(high_side)) - high)
    undershoot = max(0.0, low - float(np.min(low_side)))
    return EdgeOvershootEvidence(
        step_dn=abs(signed_step),
        overshoot_dn=overshoot,
        undershoot_dn=undershoot,
        ringing_dn=overshoot + undershoot,
    )


def isolated_edge_fixture() -> tuple[np.ndarray, int]:
    """Build a fresh deterministic 46.1-DN step with 1.76-DN plateaus.

    The original pixels behind the research package's published 3.66 -> 2.60
    DN result are not present in its fixtures.  This is therefore an openly
    declared, independently seeded discriminator, not an attempt to recreate
    those extrema.  Each source plateau's noise is zero-mean and normalized
    to exactly 1.76 DN standard deviation before the render path sees it.
    """

    midpoint = M1_SOURCE_LONG_AXIS_PX // 2
    generator = np.random.default_rng(20251027)
    noise = generator.standard_normal(M1_SOURCE_LONG_AXIS_PX)
    for section in (slice(None, midpoint), slice(midpoint, None)):
        values = noise[section]
        noise[section] = (values - np.mean(values)) * (1.76 / np.std(values))
    source = np.full(M1_SOURCE_LONG_AXIS_PX, 100.0, dtype=np.float64)
    source[midpoint:] = 146.1
    source += noise
    selected = np.floor(
        (np.arange(M1_DELIVERED_PX, dtype=np.float64) + 0.5)
        * M1_SOURCE_LONG_AXIS_PX
        / M1_DELIVERED_PX
    ).astype(np.int64)
    edge_index = int(np.searchsorted(selected, midpoint))
    return source, edge_index


def run_synthetic_m1_discriminator() -> SyntheticM1Evidence:
    source = alias_stress_columns(M1_SOURCE_LONG_AXIS_PX)
    reference = fractional_area_resample_axis(
        source,
        M1_DELIVERED_PX,
        axis=0,
    )
    nearest = _agg_canvas_resample_columns(source, M1_DELIVERED_PX, "nearest")
    hanning = _agg_canvas_resample_columns(source, M1_DELIVERED_PX, "hanning")

    edge_source_dn, edge_index = isolated_edge_fixture()
    nearest_step_dn = 255.0 * _float_rgb_resample_columns(
        edge_source_dn / 255.0,
        M1_DELIVERED_PX,
        "nearest",
    )
    hanning_step_dn = 255.0 * _float_rgb_resample_columns(
        edge_source_dn / 255.0,
        M1_DELIVERED_PX,
        "hanning",
    )
    nearest_ring = local_edge_overshoot(nearest_step_dn, edge_index)
    hanning_ring = local_edge_overshoot(hanning_step_dn, edge_index)
    edge_midpoint = M1_SOURCE_LONG_AXIS_PX // 2

    noise = deterministic_flat_noise(size=M1_SOURCE_LONG_AXIS_PX)
    hanning_noise = _float_rgb_resample_columns(noise, M1_DELIVERED_PX, "hanning")
    noise_ratio = float(np.var(hanning_noise) / np.var(noise))
    return SyntheticM1Evidence(
        evidence_kind="synthetic_discriminator_not_canonical_gate_evidence",
        source_width=M1_SOURCE_LONG_AXIS_PX,
        destination_width=M1_DELIVERED_PX,
        nearest_rms_dn=rms_error(nearest, reference * 255.0),
        hanning_rms_dn=rms_error(hanning, reference * 255.0),
        nearest_high_frequency_ratio=high_frequency_power_ratio(
            nearest, reference * 255.0, cutoff=0.3
        ),
        hanning_high_frequency_ratio=high_frequency_power_ratio(
            hanning, reference * 255.0, cutoff=0.3
        ),
        hanning_out_of_range_fraction=float(
            np.mean((hanning < source.min() * 255.0) | (hanning > source.max() * 255.0))
        ),
        edge_fixture_nominal_step_dn=46.1,
        edge_fixture_noise_seed=20251027,
        edge_fixture_left_plateau_sigma_dn=float(
            np.std(edge_source_dn[:edge_midpoint])
        ),
        edge_fixture_right_plateau_sigma_dn=float(
            np.std(edge_source_dn[edge_midpoint:])
        ),
        edge_fixture_edge_index=edge_index,
        nearest_edge_step_dn=nearest_ring.step_dn,
        nearest_edge_overshoot_dn=nearest_ring.overshoot_dn,
        nearest_edge_undershoot_dn=nearest_ring.undershoot_dn,
        nearest_edge_ringing_dn=nearest_ring.ringing_dn,
        hanning_edge_step_dn=hanning_ring.step_dn,
        hanning_edge_overshoot_dn=hanning_ring.overshoot_dn,
        hanning_edge_undershoot_dn=hanning_ring.undershoot_dn,
        hanning_edge_ringing_dn=hanning_ring.ringing_dn,
        flat_noise_variance_ratio=noise_ratio,
    )


def _runtime_fixture(product_path: str):
    import datetime as dt

    from satellites import FetchResult

    width = 48
    height = 40
    bbox = [-80.0, 14.0, -76.0, 18.0]
    lons, lats = np.meshgrid(
        np.linspace(bbox[0], bbox[2], width, dtype=np.float32),
        np.linspace(bbox[3], bbox[1], height, dtype=np.float32),
    )
    if product_path == "rgb":
        y, x = np.mgrid[0:height, 0:width]
        scalar = 0.25 + 0.45 * (x / max(width - 1, 1)) + 0.1 * np.sin(y / 4.0)
        cmi = np.repeat(scalar[..., None], 3, axis=-1).astype(np.float32)
        units = "rgb"
        channel = 2
        generic_channel = "true_color"
        enhancement = "true_color"
    elif product_path == "scalar_regular_grid":
        y, x = np.mgrid[0:height, 0:width]
        cmi = (245.0 + 22.0 * np.sin(x / 7.0) * np.cos(y / 5.0)).astype(
            np.float32
        )
        units = "K"
        channel = 13
        generic_channel = "clean_ir"
        enhancement = "rainbow_ir"
    else:
        raise ValueError(f"unknown runtime product path: {product_path!r}")
    return (
        FetchResult(
            cmi=cmi,
            lats=lats,
            lons=lons,
            channel=channel,
            generic_channel=generic_channel,
            scan_start=dt.datetime(2026, 8, 20, 12, 0, tzinfo=dt.timezone.utc),
            product="M1-RUNTIME-PROBE",
            bucket="noaa-goes19",
            sat_name="GOES-19",
            sub_sat_lon=-75.2,
            units=units,
        ),
        bbox,
        channel,
        enhancement,
    )


def trace_runtime_kernel(policy, product_path: str) -> RuntimeKernelTrace:
    """Exercise one real render_png branch and capture the literal sent to Agg."""

    modules = production_adapter.production_modules()
    render = modules["render"]
    from cartopy.mpl.geoaxes import GeoAxes

    data, bbox, channel, enhancement = _runtime_fixture(product_path)
    original_imshow = GeoAxes.imshow
    interpolations: list[str | None] = []

    def traced_imshow(self, *args, **kwargs):
        interpolations.append(kwargs.get("interpolation"))
        return original_imshow(self, *args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(GeoAxes, "imshow", traced_imshow))
        stack.enter_context(mock.patch.object(GeoAxes, "text", lambda *a, **k: None))
        render.render_png(
            data,
            bbox,
            channel,
            "2026-08-20 12:00",
            enhancement,
            1,
            coastlines=False,
            gridlines=False,
            dpi=30,
            policy=policy,
        )
    return RuntimeKernelTrace(
        policy_version=policy.algo_version,
        policy_stage=policy.stage,
        product_path=product_path,
        interpolations=tuple(interpolations),
    )


def trace_all_runtime_kernels() -> tuple[RuntimeKernelTrace, ...]:
    import render_policy

    return tuple(
        trace_runtime_kernel(policy, product_path)
        for policy in (render_policy.M0_POLICY, render_policy.M1_POLICY)
        for product_path in ("rgb", "scalar_regular_grid")
    )


def _canonical_entries(
    research: ResearchInputs,
    corpus_root: str | os.PathLike[str] | Path,
) -> tuple[dict[str, Path], tuple[CanonicalObjectIdentity, ...]]:
    manifest = research.root / "tables" / "sample_manifest.csv"
    manifest_hash = sha256_file(manifest)
    if manifest_hash != M1_SAMPLE_MANIFEST_SHA256:
        raise RuntimeError(
            "sample_manifest.csv SHA-256 mismatch: "
            f"expected {M1_SAMPLE_MANIFEST_SHA256}, received {manifest_hash}"
        )
    root = Path(corpus_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"M1 corpus root is not a directory: {root}")
    if any(os.path.lexists(candidate / ".git") for candidate in (root, *root.parents)):
        raise RuntimeError(f"M1 raw corpus must stay outside Git worktrees: {root}")

    selected = [
        entry
        for entry in load_sample_manifest(manifest)
        if entry.scene_id == M1_SCENE_ID
    ]
    by_role = {entry.metadata["band_role"]: entry for entry in selected}
    if tuple(sorted(by_role)) != tuple(sorted(M1_REQUIRED_ROLES)):
        raise RuntimeError(
            f"{M1_SCENE_ID} roles are {sorted(by_role)!r}, expected "
            f"{sorted(M1_REQUIRED_ROLES)!r}"
        )

    paths: dict[str, Path] = {}
    identities: list[CanonicalObjectIdentity] = []
    for role in M1_REQUIRED_ROLES:
        entry = by_role[role]
        path = root / entry.file_identifier
        if path.is_symlink():
            raise RuntimeError(f"canonical corpus object must not be a symlink: {path}")
        if path.parent != root:
            raise RuntimeError(f"unsafe canonical corpus path: {path}")
        verify_object(entry, path)
        assert entry.expected_sha256 is not None
        paths[role] = path
        identities.append(
            CanonicalObjectIdentity(
                role=role,
                file_identifier=entry.file_identifier,
                bytes=entry.expected_size,
                sha256=entry.expected_sha256,
            )
        )
    return paths, tuple(identities)


def load_canonical_melissa_display_array(
    research: ResearchInputs,
    corpus_root: str | os.PathLike[str] | Path,
):
    """Build the pinned display RGB through the actual production ABI recipe."""

    import satellites

    paths, identities = _canonical_entries(research, corpus_root)
    satellite = satellites.GOESEastSatellite()
    paths_by_name = {path.name: path for path in paths.values()}

    def open_local(self, resolved):
        try:
            path = paths_by_name[Path(resolved.s3_key).name]
        except KeyError as exc:
            raise RuntimeError(
                f"production requested an unpinned M1 object: {resolved.s3_key}"
            ) from exc
        dataset = satellites.xr.open_dataset(
            path,
            engine="h5netcdf",
            decode_times=False,
        )
        cleanup = tempfile.mkdtemp(prefix="tat-m1-readonly-netcdf-")
        return dataset, cleanup

    satellite.open = types.MethodType(open_local, satellite)
    scan_starts = {
        role: satellites._parse_scan_start(identity.file_identifier)  # noqa: SLF001
        for role, identity in zip(M1_REQUIRED_ROLES, identities, strict=True)
    }
    if len(set(scan_starts.values())) != 1:
        raise RuntimeError(f"canonical M1 bands are not same-scan: {scan_starts!r}")
    scan_start = scan_starts["red"]
    band_files = {
        role: satellites.ResolvedFile(
            bucket="noaa-goes19",
            s3_key=paths[role].name,
            product="CMIPF",
            scan_start=scan_start,
            sat_name="GOES-19",
            sub_sat_lon=-75.2,
        )
        for role in M1_REQUIRED_ROLES
    }
    result = satellite._compose_true_color_sync(  # noqa: SLF001 - real production path
        band_files,
        list(M1_BBOX),
        band_files["red"],
        1,
    )
    if tuple(result.cmi.shape) != (1527, 1600, 3):
        raise RuntimeError(
            f"canonical production display shape drifted: {result.cmi.shape}"
        )
    if result.cmi.dtype != np.float32 or not np.isfinite(result.cmi).all():
        raise RuntimeError("canonical production display is not finite float32")
    return result, identities


def _render_canonical_raster(data, policy) -> RenderedRaster:
    modules = production_adapter.production_modules()
    render = modules["render"]
    import matplotlib.image as mpl_image
    from PIL import Image
    from cartopy.mpl.geoaxes import GeoAxes

    geometry = production_adapter.analytic_figure_geometry(12.0, 12.0)
    original_imshow = GeoAxes.imshow
    original_resample = mpl_image._resample  # noqa: SLF001
    original_make_image = mpl_image._ImageBase._make_image  # noqa: SLF001
    interpolations: list[str | None] = []
    traces: list[AggResampleTrace] = []
    make_image_traces: list[MakeImageTrace] = []
    internal_results: list[np.ndarray] = []
    make_image_results: list[np.ndarray] = []

    def traced_imshow(self, *args, **kwargs):
        interpolations.append(kwargs.get("interpolation"))
        return original_imshow(self, *args, **kwargs)

    def traced_resample(image, data_array, output_shape, transform, *args, **kwargs):
        result = original_resample(
            image,
            data_array,
            output_shape,
            transform,
            *args,
            **kwargs,
        )
        interpolation = str(image.get_interpolation())
        if (
            interpolation in {"nearest", "hanning"}
            and np.ndim(data_array) == 3
            and np.shape(data_array)[-1] == 4
        ):
            matrix = np.asarray(transform.get_matrix(), dtype=np.float64)
            traces.append(
                AggResampleTrace(
                    interpolation=interpolation,
                    input_shape=tuple(np.shape(data_array)),
                    output_shape=(int(output_shape[0]), int(output_shape[1])),
                    transform_matrix=tuple(
                        tuple(float(value) for value in row) for row in matrix
                    ),
                )
            )
            internal = np.asarray(result)
            if internal.ndim != 3 or internal.shape[-1] != 4:
                raise RuntimeError(
                    f"unexpected production Agg RGBA result: {internal.shape}"
                )
            internal_results.append(np.asarray(internal[..., :3]).copy())
        return result

    def traced_make_image(
        image,
        data_array,
        in_bbox,
        out_bbox,
        clip_bbox,
        magnification=1.0,
        unsampled=False,
        round_to_pixel_border=True,
    ):
        result = original_make_image(
            image,
            data_array,
            in_bbox,
            out_bbox,
            clip_bbox,
            magnification=magnification,
            unsampled=unsampled,
            round_to_pixel_border=round_to_pixel_border,
        )
        interpolation = str(image.get_interpolation())
        if (
            interpolation in {"nearest", "hanning"}
            and np.ndim(data_array) == 3
            and np.shape(data_array)[-1] in {3, 4}
        ):
            returned_rgba, returned_x, returned_y, returned_transform = result
            if returned_rgba is None:
                raise RuntimeError("production _make_image returned no RGBA raster")
            matrix = np.asarray(returned_transform.get_matrix(), dtype=np.float64)
            make_image_traces.append(
                MakeImageTrace(
                    interpolation=interpolation,
                    input_shape=tuple(np.shape(data_array)),
                    in_bbox_bounds=tuple(float(value) for value in in_bbox.bounds),
                    out_bbox_bounds=tuple(float(value) for value in out_bbox.bounds),
                    clip_bbox_bounds=tuple(float(value) for value in clip_bbox.bounds),
                    magnification=float(magnification),
                    unsampled=bool(unsampled),
                    round_to_pixel_border=bool(round_to_pixel_border),
                    returned_rgba_shape=tuple(np.shape(returned_rgba)),
                    returned_origin=(float(returned_x), float(returned_y)),
                    returned_transform_matrix=tuple(
                        tuple(float(value) for value in row) for row in matrix
                    ),
                )
            )
            make_image_results.append(np.asarray(returned_rgba[..., :3]).copy())
        return result

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(GeoAxes, "imshow", traced_imshow))
        stack.enter_context(mock.patch.object(mpl_image, "_resample", traced_resample))
        stack.enter_context(
            mock.patch.object(mpl_image._ImageBase, "_make_image", traced_make_image)
        )
        stack.enter_context(mock.patch.object(GeoAxes, "text", lambda *a, **k: None))
        png = render.render_png(
            data,
            list(M1_BBOX),
            2,
            "2025-10-27 15:00 UTC",
            "true_color",
            1,
            coastlines=False,
            gridlines=False,
            dpi=110,
            policy=policy,
        )
    with Image.open(io.BytesIO(png)) as image:
        figure_rgb = np.asarray(image.convert("RGB"))
    expected_figure = (*geometry.figure_shape, 3)
    if tuple(figure_rgb.shape) != expected_figure:
        raise RuntimeError(
            f"canonical M1 figure shape drifted: {figure_rgb.shape}, "
            f"expected {expected_figure}"
        )
    first_row, first_column = geometry.device_origin
    delivered_height, delivered_width = geometry.delivered_shape
    raster = figure_rgb[
        first_row : first_row + delivered_height,
        first_column : first_column + delivered_width,
    ].copy()
    if tuple(raster.shape) != (1188, 1188, 3):
        raise RuntimeError(f"canonical delivered raster shape drifted: {raster.shape}")
    if (
        len(traces) != 1
        or len(internal_results) != 1
        or len(make_image_traces) != 1
        or len(make_image_results) != 1
    ):
        raise RuntimeError(
            "canonical render did not expose exactly one regular-grid Agg "
            "image operation: "
            f"resamples={len(traces)}, makes={len(make_image_traces)}, "
            f"resample_results={len(internal_results)}, "
            f"make_results={len(make_image_results)}"
        )
    if make_image_traces[0].returned_rgba_shape[:2] != traces[0].output_shape:
        raise RuntimeError("_make_image and _resample internal shapes disagree")
    return RenderedRaster(
        png=png,
        raster=raster,
        interpolations=tuple(interpolations),
        agg_traces=tuple(traces),
        make_image_traces=tuple(make_image_traces),
        internal_resampled_rgb=internal_results[0],
        internal_make_image_rgb=make_image_results[0],
    )


def _reference_fetch(source_fetch, reference: np.ndarray):
    height, width = reference.shape[:2]
    lons, lats = np.meshgrid(
        np.linspace(M1_BBOX[0], M1_BBOX[2], width, dtype=np.float32),
        np.linspace(M1_BBOX[3], M1_BBOX[1], height, dtype=np.float32),
    )
    return dataclasses.replace(
        source_fetch,
        cmi=reference.astype(np.float32),
        lats=lats,
        lons=lons,
        cos_sza=None,
        geom_valid=np.ones((height, width), dtype=bool),
        product="M1-EXACT-AREA-REFERENCE",
    )


def _repeatability_worker(source_fetch, connection) -> None:
    """One isolated fork worker for canonical V25 evidence."""

    try:
        import render_policy

        rendered = _render_canonical_raster(source_fetch, render_policy.M1_POLICY)
        connection.send(("ok", _sha256_bytes(rendered.png), os.getpid()))
    except BaseException as exc:  # child must surface every failure to parent
        connection.send(("error", f"{type(exc).__name__}: {exc}", os.getpid()))
    finally:
        connection.close()


def separate_process_repeatability_hashes(
    source_fetch,
) -> tuple[tuple[str, str, str], tuple[int, int, int], str]:
    """Render the pinned M1 request in three distinct forked processes."""

    import multiprocessing

    available = multiprocessing.get_all_start_methods()
    if "fork" not in available:
        raise RuntimeError(
            "canonical V25 requires separate fork workers on the Linux renderer"
        )
    context = multiprocessing.get_context("fork")
    parents = []
    processes = []
    for _ in range(3):
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_repeatability_worker,
            args=(source_fetch, child_connection),
        )
        process.start()
        child_connection.close()
        parents.append(parent_connection)
        processes.append(process)
    results: list[str] = []
    process_ids: list[int] = []
    try:
        for connection, process in zip(parents, processes, strict=True):
            if not connection.poll(60.0):
                raise RuntimeError(f"M1 repeatability worker {process.pid} timed out")
            status, detail, child_pid = connection.recv()
            if status != "ok":
                raise RuntimeError(
                    f"M1 repeatability worker {process.pid} failed: {detail}"
                )
            results.append(detail)
            process_ids.append(int(child_pid))
        for process in processes:
            process.join(timeout=10.0)
            if process.exitcode != 0:
                raise RuntimeError(
                    f"M1 repeatability worker {process.pid} exited {process.exitcode}"
                )
    finally:
        for connection in parents:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
    if len(results) != 3:
        raise RuntimeError(f"M1 repeatability produced {len(results)} hashes")
    if len(set(process_ids)) != 3:
        raise RuntimeError(f"M1 repeatability did not use three PIDs: {process_ids}")
    return (
        (results[0], results[1], results[2]),
        (process_ids[0], process_ids[1], process_ids[2]),
        "fork",
    )


def _global_out_of_range_fraction(candidate: np.ndarray, source: np.ndarray) -> float:
    lower = np.floor(float(np.min(source)) * 255.0)
    upper = np.ceil(float(np.max(source)) * 255.0)
    return float(np.mean((candidate < lower) | (candidate > upper)))


def measure_canonical_m1(
    research: ResearchInputs,
    corpus_root: str | os.PathLike[str] | Path,
) -> tuple[
    CanonicalM1Evidence,
    M1RenderedBundle,
    KernelCoverage,
    KernelCoverage,
]:
    import render_policy

    source_fetch, identities = load_canonical_melissa_display_array(
        research,
        corpus_root,
    )
    geometry = production_adapter.analytic_figure_geometry(12.0, 12.0)
    nearest = _render_canonical_raster(source_fetch, render_policy.M0_POLICY)
    hanning = _render_canonical_raster(source_fetch, render_policy.M1_POLICY)
    nearest_trace = nearest.agg_traces[0]
    hanning_trace = hanning.agg_traces[0]
    if (
        nearest_trace.input_shape != hanning_trace.input_shape
        or nearest_trace.output_shape != hanning_trace.output_shape
        or not np.allclose(
            nearest_trace.transform_matrix,
            hanning_trace.transform_matrix,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError("M0 and M1 production Agg geometry does not match")
    internal_shape = hanning_trace.output_shape
    reference_source = fractional_area_resample(
        source_fetch.cmi,
        internal_shape,
        # Canonical audited research convention: edge-aligned detector-cell
        # overlap. Matplotlib's device phase belongs to the candidate path and
        # is recorded independently; it must not move the ground-truth grid.
        sampling_phase=(0.5, 0.5),
    )
    reference_raster = np.floor(
        np.clip(reference_source, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    reference = _render_canonical_raster(
        _reference_fetch(source_fetch, reference_source),
        render_policy.M0_POLICY,
    )
    reference_mapping = find_internal_canvas_mapping(
        reference_raster,
        reference.raster,
    )
    if reference.internal_make_image_rgb is None:
        raise RuntimeError("reference render did not capture _make_image RGBA")
    reference_make_mapping = find_internal_canvas_mapping(
        internal_to_canvas_orientation(
            reference.internal_make_image_rgb,
            reference.agg_traces[0],
        ),
        reference.raster,
    )
    if reference_make_mapping.pixel_bounds != reference_mapping.pixel_bounds:
        raise RuntimeError("direct reference and _make_image canvas mappings disagree")
    if (
        nearest.internal_make_image_rgb is None
        or hanning.internal_make_image_rgb is None
        or hanning.internal_resampled_rgb is None
    ):
        raise RuntimeError("candidate render did not capture internal Agg rasters")
    nearest_mapping = find_internal_canvas_mapping(
        internal_to_canvas_orientation(
            nearest.internal_make_image_rgb,
            nearest_trace,
        ),
        nearest.raster,
    )
    hanning_mapping = find_internal_canvas_mapping(
        internal_to_canvas_orientation(
            hanning.internal_make_image_rgb,
            hanning_trace,
        ),
        hanning.raster,
    )
    if (
        nearest_mapping.pixel_bounds != reference_mapping.pixel_bounds
        or hanning_mapping.pixel_bounds != reference_mapping.pixel_bounds
    ):
        raise RuntimeError("candidate and reference canvas mappings disagree")

    nearest_error = nearest.raster.astype(np.float64) - reference.raster.astype(
        np.float64
    )
    hanning_error = hanning.raster.astype(np.float64) - reference.raster.astype(
        np.float64
    )
    nearest_rms = float(np.sqrt(np.mean(np.square(nearest_error))))
    hanning_rms = float(np.sqrt(np.mean(np.square(hanning_error))))
    reduction = 100.0 * (1.0 - hanning_rms / nearest_rms)
    local_range = measure_local_neighbourhood_range(
        source_fetch.cmi,
        hanning.raster,
        trace=hanning_trace,
        internal_resampled_rgb=hanning.internal_resampled_rgb,
        delivered_internal_bounds=reference_mapping.pixel_bounds,
    )
    horizontal_coverage, vertical_coverage = kernel_coverage_from_trace(
        hanning_trace,
        source_fetch.cmi.shape[:2],
        reference_mapping.pixel_bounds,
    )
    repeatability, repeatability_pids, repeatability_start_method = (
        separate_process_repeatability_hashes(source_fetch)
    )
    if any(value != _sha256_bytes(hanning.png) for value in repeatability):
        raise RuntimeError(
            "separate-process M1 output differs from the parent production render"
        )
    if source_fetch.geom_valid is None:
        raise RuntimeError("canonical full compose omitted geom_valid")
    expected_delivered_reference = reference_raster[
        reference_mapping.pixel_bounds[0] : reference_mapping.pixel_bounds[1],
        reference_mapping.pixel_bounds[2] : reference_mapping.pixel_bounds[3],
    ]
    make_geometry = asdict(hanning.make_image_traces[0])
    canonical = CanonicalM1Evidence(
        evidence_kind="canonical_real_pinned_scene",
        scene_id=M1_SCENE_ID,
        bbox=M1_BBOX,
        objects=identities,
        source_shape=tuple(source_fetch.cmi.shape),
        source_display_sha256=_sha256_bytes(source_fetch.cmi.tobytes()),
        source_lats_sha256=_sha256_bytes(
            np.ascontiguousarray(source_fetch.lats).tobytes()
        ),
        source_lons_sha256=_sha256_bytes(
            np.ascontiguousarray(source_fetch.lons).tobytes()
        ),
        source_geom_valid_sha256=_sha256_bytes(
            np.ascontiguousarray(source_fetch.geom_valid).tobytes()
        ),
        figure_shape=(*geometry.figure_shape, 3),
        delivered_shape=tuple(hanning.raster.shape),
        map_pixel_bounds=(
            geometry.device_origin[0],
            geometry.device_origin[0] + geometry.delivered_shape[0],
            geometry.device_origin[1],
            geometry.device_origin[1] + geometry.delivered_shape[1],
        ),
        agg_internal_shape=(*internal_shape, 4),
        agg_transform_matrix=hanning_trace.transform_matrix,
        agg_make_image_geometry=make_geometry,
        delivered_internal_pixel_bounds=reference_mapping.pixel_bounds,
        internal_mapping_candidate_scores=reference_mapping.candidate_scores,
        sampling_model=geometry.sampling_model,
        sampling_phase=geometry.sampling_phase,
        reference_convention=M1_REFERENCE_CONVENTION,
        reference_compositor_mode=(
            "captured_internal_grid_exact_area_then_M0_nearest_canvas_clip"
        ),
        reference_quantizer="production_floor_u8_no_dither",
        reference_compositor_caveat=(
            "The internal Agg raster can differ from the 1188x1188 delivered "
            "map under Matplotlib's legacy ceil geometry. Internal shape and "
            "the unique contiguous canvas slice are captured and identity-"
            "proved at runtime; neither is hard-coded."
        ),
        reference_identity_boundary_px=reference_mapping.boundary_exclusion_px,
        reference_identity_full=reference_mapping.full_identity,
        reference_identity_scored=reference_mapping.scored_identity,
        production_nearest_canvas_identity_scored=(
            nearest_mapping.scored_identity
        ),
        explicit_hanning_canvas_identity_scored=(
            hanning_mapping.scored_identity
        ),
        reference_expected_quantized_sha256=_sha256_bytes(
            expected_delivered_reference.tobytes()
        ),
        production_nearest_rms_dn=nearest_rms,
        explicit_hanning_rms_dn=hanning_rms,
        error_removed_same_array_pct=reduction,
        production_nearest_high_frequency_ratio=radial_high_frequency_power_ratio(
            nearest.raster,
            reference.raster,
            cutoff=0.3,
        ),
        explicit_hanning_high_frequency_ratio=radial_high_frequency_power_ratio(
            hanning.raster,
            reference.raster,
            cutoff=0.3,
        ),
        spectral_definition={
            "luma": "Rec.601: 0.299 R + 0.587 G + 0.114 B",
            "square_crop": "top-left min(height,width)",
            "spatial_window": "separable 2-D Hann",
            "radial_bins": 64,
            "radial_range_cycles_per_pixel": (0.0, 0.5),
            "bin_statistic": "mean PSD",
            "integration": "sum finite bin means with centre > 0.3 cycles/pixel",
        },
        hanning_global_out_of_range_fraction=_global_out_of_range_fraction(
            hanning.raster,
            source_fetch.cmi,
        ),
        hanning_local_neighbourhood_out_of_range_fraction=(
            local_range.out_of_range_fraction
        ),
        hanning_local_neighbourhood_comparisons=local_range.comparisons,
        hanning_local_neighbourhood_tolerance=local_range.tolerance,
        hanning_actual_phase_minimum_weight=local_range.minimum_weight,
        hanning_actual_phase_maximum_weight_sum_error=(
            local_range.maximum_weight_sum_error
        ),
        hanning_actual_transform_maximum_dc_error=local_range.maximum_dc_error,
        hanning_float_to_rendered_identity_boundary_px=(
            local_range.float_to_rendered_identity_boundary_px
        ),
        hanning_float_to_rendered_identity_full=(
            local_range.float_to_rendered_identity_full
        ),
        hanning_float_to_rendered_identity_scored=(
            local_range.float_to_rendered_identity_scored
        ),
        production_nearest_png_sha256=_sha256_bytes(nearest.png),
        explicit_hanning_png_sha256=_sha256_bytes(hanning.png),
        bandlimited_reference_png_sha256=_sha256_bytes(reference.png),
        production_nearest_raster_sha256=_sha256_bytes(nearest.raster.tobytes()),
        explicit_hanning_raster_sha256=_sha256_bytes(hanning.raster.tobytes()),
        bandlimited_reference_raster_sha256=_sha256_bytes(
            reference.raster.tobytes()
        ),
        repeatability_sha256=repeatability,
        repeatability_process_ids=repeatability_pids,
        repeatability_start_method=repeatability_start_method,
        runtime_interpolations={
            "production_nearest": nearest.interpolations,
            "explicit_hanning": hanning.interpolations,
            "bandlimited_reference_internal_grid_M0_canvas_clip": (
                reference.interpolations
            ),
        },
    )
    rendered = M1RenderedBundle(
        production_nearest=nearest.raster,
        explicit_hanning=hanning.raster,
        bandlimited_reference=reference.raster,
    )
    return canonical, rendered, horizontal_coverage, vertical_coverage


_M1_GATE_SPECS: Final = {gate.gate_id: gate for gate in GATES}


def _m1_gate(
    gate_id: str,
    passed: bool | None,
    *,
    evidence_kind: str,
    detail: str,
    measurements: Mapping[str, Any],
    blocking: bool | None = None,
) -> M1GateEvidence:
    spec = _M1_GATE_SPECS[gate_id]
    outcome = "UNMEASURED" if passed is None else ("PASS" if passed else "FAIL")
    return M1GateEvidence(
        gate_id=gate_id,
        name=spec.name,
        threshold=spec.threshold,
        outcome=outcome,
        blocking=(passed is not True) if blocking is None else blocking,
        evidence_kind=evidence_kind,
        detail=detail,
        measurements=measurements,
    )


def evaluate_m1_gates(
    *,
    static_policy: StaticPolicyAudit,
    runtime_traces: Sequence[RuntimeKernelTrace],
    horizontal_coverage: KernelCoverage,
    vertical_coverage: KernelCoverage,
    synthetic: SyntheticM1Evidence,
    canonical: CanonicalM1Evidence | None,
) -> tuple[M1GateEvidence, ...]:
    """Evaluate only M1's assigned gates, failing closed on absent real data."""

    runtime_passed = len(runtime_traces) == 4 and all(
        trace.passed for trace in runtime_traces
    )
    canonical_kind = (
        canonical is not None and canonical.evidence_kind == "canonical_real_pinned_scene"
    )
    if not canonical_kind:
        v01_pass: bool | None = None
        v02_pass: bool | None = None
        v03_pass: bool | None = None
        canonical_measurements: Mapping[str, Any] = {
            "reason": (
                "canonical pinned 12-degree production scene was not supplied; "
                "synthetic discriminator is ineligible for V01-V03"
            ),
            "synthetic_evidence_kind": synthetic.evidence_kind,
        }
        v01_m1_progress = False
    else:
        assert canonical is not None
        v01_pass = canonical.explicit_hanning_rms_dn <= 1.5
        v02_pass = (
            horizontal_coverage.discarded_source_pixels == 0
            and vertical_coverage.discarded_source_pixels == 0
        )
        v03_pass = canonical.explicit_hanning_high_frequency_ratio <= 1.10
        v01_m1_progress = (
            canonical.explicit_hanning_rms_dn
            < canonical.production_nearest_rms_dn
            and canonical.error_removed_same_array_pct > 0.0
        )
        canonical_measurements = {
            "scene_id": canonical.scene_id,
            "source_shape": canonical.source_shape,
            "source_display_sha256": canonical.source_display_sha256,
            "source_lats_sha256": canonical.source_lats_sha256,
            "source_lons_sha256": canonical.source_lons_sha256,
            "source_geom_valid_sha256": canonical.source_geom_valid_sha256,
            "figure_shape": canonical.figure_shape,
            "delivered_shape": canonical.delivered_shape,
            "map_pixel_bounds": canonical.map_pixel_bounds,
            "agg_internal_shape": canonical.agg_internal_shape,
            "agg_transform_matrix": canonical.agg_transform_matrix,
            "agg_make_image_geometry": canonical.agg_make_image_geometry,
            "delivered_internal_pixel_bounds": (
                canonical.delivered_internal_pixel_bounds
            ),
            "internal_mapping_candidate_scores": (
                canonical.internal_mapping_candidate_scores
            ),
            "sampling_model": canonical.sampling_model,
            "sampling_phase": canonical.sampling_phase,
            "reference_convention": canonical.reference_convention,
            "reference_compositor_mode": canonical.reference_compositor_mode,
            "reference_quantizer": canonical.reference_quantizer,
            "reference_compositor_caveat": canonical.reference_compositor_caveat,
            "reference_identity_boundary_px": (
                canonical.reference_identity_boundary_px
            ),
            "reference_identity_full": canonical.reference_identity_full,
            "reference_identity_scored": canonical.reference_identity_scored,
            "production_nearest_rms_dn": canonical.production_nearest_rms_dn,
            "explicit_hanning_rms_dn": canonical.explicit_hanning_rms_dn,
            "production_nearest_high_frequency_ratio": (
                canonical.production_nearest_high_frequency_ratio
            ),
            "explicit_hanning_high_frequency_ratio": (
                canonical.explicit_hanning_high_frequency_ratio
            ),
            "spectral_definition": canonical.spectral_definition,
            "error_removed_same_array_pct": canonical.error_removed_same_array_pct,
            "m1_partial_progress_requirement": (
                "explicit Hanning RMS must be lower than production nearest; "
                "the <=1.5-DN hard threshold remains assigned to M2"
            ),
            "m1_partial_progress_passed": v01_m1_progress,
        }

    v04_pass = static_policy.passed and runtime_passed
    v15_pass = (
        horizontal_coverage.is_convex_full_coverage
        and vertical_coverage.is_convex_full_coverage
        and synthetic.hanning_out_of_range_fraction == 0.0
        and canonical is not None
        and canonical.hanning_global_out_of_range_fraction == 0.0
        and canonical.hanning_local_neighbourhood_out_of_range_fraction == 0.0
        and canonical.hanning_actual_phase_minimum_weight >= -2e-6
        and canonical.hanning_actual_transform_maximum_dc_error <= 2e-6
        and canonical.hanning_float_to_rendered_identity_scored >= 0.998
    )
    v16_pass = (
        horizontal_coverage.is_convex_full_coverage
        and vertical_coverage.is_convex_full_coverage
        and synthetic.hanning_edge_ringing_dn
        <= synthetic.nearest_edge_ringing_dn + 1e-7
    )
    forbidden = production_adapter.forbidden_operator_findings()
    v18_pass = not forbidden and synthetic.flat_noise_variance_ratio <= 1.0
    v25_pass = (
        canonical is not None
        and len(canonical.repeatability_sha256) == 3
        and len(set(canonical.repeatability_sha256)) == 1
    )
    return (
        _m1_gate(
            "V01",
            v01_pass,
            evidence_kind=(
                "canonical_real_pinned_scene" if canonical_kind else "unmeasured"
            ),
            detail=(
                "all 1188x1188 extracted map pixels versus compositor-matched "
                "same-array exact-area reference; M1 records the hard threshold "
                "honestly but advances on measured improvement because the "
                "canonical stage table assigns full V01 closure to M2"
            ),
            measurements=canonical_measurements,
            blocking=not v01_m1_progress,
        ),
        _m1_gate(
            "V02",
            v02_pass,
            evidence_kind=(
                "canonical_scale_kernel_response" if canonical_kind else "unmeasured"
            ),
            detail="Agg impulse-response enumeration at both canonical axis ratios",
            measurements={
                "horizontal": asdict(horizontal_coverage),
                "vertical": asdict(vertical_coverage),
            },
        ),
        _m1_gate(
            "V03",
            v03_pass,
            evidence_kind=(
                "canonical_real_pinned_scene" if canonical_kind else "unmeasured"
            ),
            detail=(
                "canonical Rec.601 luma, top-left square crop, 2-D Hann window, "
                "64 radial mean-PSD bins over 0..0.5; sum above 0.3 cycles/pixel"
            ),
            measurements=canonical_measurements,
        ),
        _m1_gate(
            "V04",
            v04_pass,
            evidence_kind="production_static_and_runtime",
            detail=(
                "literal dispatcher arms plus RGB/scalar runtime traces for active M1 "
                "and registered M0 rollback"
            ),
            measurements={
                "static_policy": static_policy.as_dict(),
                "runtime_traces": [trace.as_dict() for trace in runtime_traces],
            },
        ),
        _m1_gate(
            "V15",
            v15_pass,
            evidence_kind="kernel_convexity_plus_real_and_synthetic_ranges",
            detail="non-negative unit-sum Hanning response cannot manufacture extrema",
            measurements={
                "horizontal": asdict(horizontal_coverage),
                "vertical": asdict(vertical_coverage),
                "synthetic_out_of_range_fraction": (
                    synthetic.hanning_out_of_range_fraction
                ),
                "canonical_global_out_of_range_fraction": (
                    None
                    if canonical is None
                    else canonical.hanning_global_out_of_range_fraction
                ),
                "canonical_local_neighbourhood_out_of_range_fraction": (
                    None
                    if canonical is None
                    else canonical.hanning_local_neighbourhood_out_of_range_fraction
                ),
                "canonical_local_neighbourhood_comparisons": (
                    None
                    if canonical is None
                    else canonical.hanning_local_neighbourhood_comparisons
                ),
                "canonical_local_neighbourhood_tolerance": (
                    None
                    if canonical is None
                    else canonical.hanning_local_neighbourhood_tolerance
                ),
                "canonical_actual_phase_minimum_weight": (
                    None
                    if canonical is None
                    else canonical.hanning_actual_phase_minimum_weight
                ),
                "canonical_actual_phase_maximum_weight_sum_error": (
                    None
                    if canonical is None
                    else canonical.hanning_actual_phase_maximum_weight_sum_error
                ),
                "canonical_actual_transform_maximum_dc_error": (
                    None
                    if canonical is None
                    else canonical.hanning_actual_transform_maximum_dc_error
                ),
                "canonical_float_to_rendered_identity_boundary_px": (
                    None
                    if canonical is None
                    else canonical.hanning_float_to_rendered_identity_boundary_px
                ),
                "canonical_float_to_rendered_identity_full": (
                    None
                    if canonical is None
                    else canonical.hanning_float_to_rendered_identity_full
                ),
                "canonical_float_to_rendered_identity_scored": (
                    None
                    if canonical is None
                    else canonical.hanning_float_to_rendered_identity_scored
                ),
            },
        ),
        _m1_gate(
            "V16",
            v16_pass,
            evidence_kind="fresh_deterministic_local_plateau_edge_plus_kernel_convexity",
            detail=(
                "fresh fixed-seed 46.1-DN isolated step with 1.76-DN source "
                "plateau sigma; medians exclude three transition samples and "
                "local extrema are measured in DN (not a reproduction of the "
                "unavailable research edge pixels)"
            ),
            measurements={
                "nominal_step_dn": synthetic.edge_fixture_nominal_step_dn,
                "noise_seed": synthetic.edge_fixture_noise_seed,
                "left_plateau_sigma_dn": (
                    synthetic.edge_fixture_left_plateau_sigma_dn
                ),
                "right_plateau_sigma_dn": (
                    synthetic.edge_fixture_right_plateau_sigma_dn
                ),
                "edge_index": synthetic.edge_fixture_edge_index,
                "nearest_step_dn": synthetic.nearest_edge_step_dn,
                "nearest_overshoot_dn": synthetic.nearest_edge_overshoot_dn,
                "nearest_undershoot_dn": synthetic.nearest_edge_undershoot_dn,
                "nearest_ringing_dn": synthetic.nearest_edge_ringing_dn,
                "hanning_step_dn": synthetic.hanning_edge_step_dn,
                "hanning_overshoot_dn": synthetic.hanning_edge_overshoot_dn,
                "hanning_undershoot_dn": synthetic.hanning_edge_undershoot_dn,
                "hanning_ringing_dn": synthetic.hanning_edge_ringing_dn,
            },
        ),
        _m1_gate(
            "V18",
            v18_pass,
            evidence_kind="production_ast_plus_flat_field",
            detail="forbidden-operator scan and Hanning flat-noise variance guard",
            measurements={
                "forbidden_operator_findings": forbidden,
                "flat_noise_variance_ratio": synthetic.flat_noise_variance_ratio,
            },
        ),
        _m1_gate(
            "V25",
            v25_pass,
            evidence_kind=(
                "canonical_real_pinned_scene" if canonical is not None else "unmeasured"
            ),
            detail=(
                "three identical canonical M1 production PNG SHA-256 values "
                "from three distinct Linux fork child PIDs (not fresh interpreters)"
            ),
            measurements={
                "repeatability_sha256": ()
                if canonical is None
                else canonical.repeatability_sha256,
                "process_ids": ()
                if canonical is None
                else canonical.repeatability_process_ids,
                "start_method": None
                if canonical is None
                else canonical.repeatability_start_method,
            },
        ),
    )


def run_m1(
    research: ResearchInputs,
    *,
    corpus_root: str | os.PathLike[str] | Path,
    rayleigh_startup: RayleighStartupEvidence,
) -> M1Run:
    if not rayleigh_startup.passed:
        raise RuntimeError("M1 refuses to run without the eight-sensor Rayleigh contract")
    static_policy = audit_production_policy()
    runtime_traces = trace_all_runtime_kernels()
    synthetic = run_synthetic_m1_discriminator()
    canonical, rendered, horizontal, vertical = measure_canonical_m1(
        research,
        corpus_root,
    )
    gates = evaluate_m1_gates(
        static_policy=static_policy,
        runtime_traces=runtime_traces,
        horizontal_coverage=horizontal,
        vertical_coverage=vertical,
        synthetic=synthetic,
        canonical=canonical,
    )
    return M1Run(
        research=research,
        static_policy=static_policy,
        runtime_traces=runtime_traces,
        horizontal_coverage=horizontal,
        vertical_coverage=vertical,
        synthetic=synthetic,
        canonical=canonical,
        rayleigh_startup=rayleigh_startup,
        gates=gates,
        rendered=rendered,
    )


def build_m1_report(
    run: M1Run,
    *,
    gallery_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    dependency_metadata: Mapping[str, Any] | None = None,
    harness_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    gates = [gate.as_dict() for gate in run.gates]
    outcomes = {
        outcome: sum(gate["outcome"] == outcome for gate in gates)
        for outcome in ("PASS", "FAIL", "UNMEASURED")
    }
    report: dict[str, Any] = {
        "schema": M1_REPORT_SCHEMA,
        "command": "m1",
        "enforce_through": "M1",
        "research_inputs": run.research.as_dict(),
        "source": dict(
            collect_source_metadata() if source_metadata is None else source_metadata
        ),
        "dependencies": dict(
            collect_dependency_metadata()
            if dependency_metadata is None
            else dependency_metadata
        ),
        "summary": {
            "gate_count": len(gates),
            "required_gate_ids": sorted(M1_REQUIRED_GATE_IDS),
            "actual_gate_ids": [gate["gate_id"] for gate in gates],
            "exact_gate_id_contract_valid": run.gate_id_contract_valid,
            "outcomes": outcomes,
            "threshold_fail_gate_ids": list(run.threshold_fail_gate_ids),
            "unmeasured_gate_ids": list(run.unmeasured_gate_ids),
            "blocking_gate_count": sum(gate["blocking"] for gate in gates),
            "stage_advancement_passed": run.stage_advancement_passed,
            "all_thresholds_passed": run.all_thresholds_passed,
            "m1_partial_progress_rule": (
                "V01 may remain a non-blocking FAIL only when canonical real "
                "evidence shows explicit-Hanning RMS improved over production "
                "nearest; M2 remains responsible for closing V01 <=1.5 DN"
            ),
        },
        "m1_contracts": {
            "static_policy": run.static_policy.as_dict(),
            "runtime_traces": [trace.as_dict() for trace in run.runtime_traces],
            "horizontal_coverage": asdict(run.horizontal_coverage),
            "vertical_coverage": asdict(run.vertical_coverage),
            "synthetic_discriminator": run.synthetic.as_dict(),
            "canonical": None if run.canonical is None else run.canonical.as_dict(),
            "rayleigh_startup": run.rayleigh_startup.as_dict(),
        },
        "gates": gates,
        "gallery_artifacts": dict(gallery_artifacts or {}),
        "artifact_hashes": {
            "validation_gates_csv": run.research.validation_gates_sha256,
            "reference_pixels_json": run.research.reference_pixels_sha256,
            "sample_manifest_csv": M1_SAMPLE_MANIFEST_SHA256,
            "validation_harness_source_sha256": dict(
                m1_harness_source_hashes()
                if harness_source_hashes is None
                else harness_source_hashes
            ),
        },
    }
    report["artifact_hashes"]["evidence_payload_sha256"] = _sha256_bytes(
        stable_json_bytes(report)
    )
    return report
