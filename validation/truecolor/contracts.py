"""Canonical contracts for the staged true-colour validation harness.

This module deliberately contains no renderer implementation.  It gives the
M0 harness a small, immutable description of the V01--V46 gates and of the
red-team corrections that are easy to lose when prose is copied between
documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Iterable, Mapping


class GateSeverity(str, Enum):
    """Whether a non-passing result blocks progression to the next stage."""

    HARD = "hard"
    SOFT = "soft"


class BaselineStatus(str, Enum):
    """Status measured by the research package before implementation."""

    PASS = "pass"
    FAIL = "fail"
    UNMEASURED = "unmeasured"


class GateOutcome(str, Enum):
    """Result vocabulary used by the harness.

    ``UNMEASURED`` and ``SKIPPED`` are intentionally distinct from ``PASS``.
    A hard gate in either state remains blocking.
    """

    PASS = "pass"
    FAIL = "fail"
    SOFT_WARNING = "soft_warning"
    UNMEASURED = "unmeasured"
    SKIPPED = "skipped"
    HARNESS_ERROR = "harness_error"


@dataclass(frozen=True, slots=True)
class GateSpec:
    gate_id: str
    name: str
    threshold: str
    baseline_status: BaselineStatus
    severity: GateSeverity = GateSeverity.HARD


@dataclass(frozen=True, slots=True)
class GateResult:
    spec: GateSpec
    outcome: GateOutcome
    detail: str = ""

    @property
    def blocking(self) -> bool:
        if self.spec.severity is GateSeverity.SOFT:
            return False
        return self.outcome is not GateOutcome.PASS


@dataclass(frozen=True, slots=True)
class GeometryConvention:
    """Corrected C1 drawn-raster geometry."""

    low_px: int = 756
    default_px: int = 1188
    high_px: int = 2160
    capped_source_columns: int = 1600
    default_discarded_columns_under_nearest: int = 412

    @property
    def default_discarded_fraction(self) -> float:
        return self.default_discarded_columns_under_nearest / self.capped_source_columns


C1_GEOMETRY: Final = GeometryConvention()
C1_RETIRED_VALUES: Final = frozenset({757, 1189, 1214.4, 2162})


REFERENCE_PIXEL_DN_FLOOR: Final[Mapping[str, str]] = MappingProxyType(
    {
        "P1": "#D0CFCE",
        "P2": "#000006",
        "P3": "#193F10",
        "P4": "#838689",
        "P5": "#195710",
    }
)
REFERENCE_PIXEL_DN_ROUND_HALF_UP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "P1": "#D0D0CF",
        "P2": "#000006",
        "P3": "#193F11",
        "P4": "#84868A",
        "P5": "#195711",
    }
)
C3_DECLARED_QUANTIZERS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "production_floor": REFERENCE_PIXEL_DN_FLOOR,
        "round_half_up": REFERENCE_PIXEL_DN_ROUND_HALF_UP,
    }
)
C3_FLOAT_TOLERANCE: Final = 1e-6
C3_EIGHT_BIT_TOLERANCE_DN: Final = 0
C3_RECOMMENDED_QUANTIZER: Final = "round_half_up"


@dataclass(frozen=True, slots=True)
class FlickerScope:
    """C13 scope for V30; moving storm structure is never a reference patch."""

    full_daylight_only: bool = True
    stationary_patch_only: bool = True
    minimum_patch_width_px: int = 96
    minimum_patch_height_px: int = 96


C13_FLICKER_SCOPE: Final = FlickerScope()
C13_TERMINATOR_DAY_DEAD_MAX: Final = 1.0 / 255.0
C13_TERMINATOR_DEFINITION: Final = "day_rgb_black_clipped_to_zero"
C13_LOOSER_DEFINITION_RESIDUAL_DN: Final = 12.81


C18_RAYLEIGH_STATUSES: Final = frozenset(
    {"applied", "not_required", "SKIPPED_LUT_UNAVAILABLE"}
)
C18_STARTUP_CALLS_EVERY_CONFIGURED_SENSOR: Final = True
C18_PER_FRAME_STATUS_LOG_REQUIRED: Final = True

C19_SELF_SHARPEN_DISABLE_PREREQUISITES: Final = frozenset({"V46"})
C19_REGISTRATION_LIMIT_PX: Final = 0.25


@dataclass(frozen=True, slots=True)
class ReferenceConvention:
    """C20 convention attached to every alias-reduction percentage."""

    name: str
    source_columns: int
    cannot_score_stages: tuple[str, ...]
    quoted_unweighted_reduction_pct: float
    native_crop_kernel_reduction_pct: tuple[float, float]


C20_REFERENCE_CONVENTION: Final = ReferenceConvention(
    name="same_1600_pixel_intermediate",
    source_columns=1600,
    cannot_score_stages=("S02", "S03", "S04"),
    quoted_unweighted_reduction_pct=81.9,
    native_crop_kernel_reduction_pct=(47.7, 52.0),
)


_GATE_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("V01", "Aliasing RMS vs band-limited reference", "<= 1.5 DN"),
    ("V02", "Source-column coverage", "0 % discarded"),
    ("V03", "Spectral power ratio above 0.3 cyc/px", "<= 1.10x"),
    ("V04", "Explicit-kernel assertion", "kernel == 'hanning' (or the resize happens upstream and imshow draws 1:1)"),
    ("V05", "Native-resolution honesty", "0 frames with delivered_px > native_px"),
    ("V06", "Native-resolution preservation", ">= 0.90"),
    ("V07", "Tone-curve constant assertion", "exact equality"),
    ("V08", "Reference-pixel end-to-end fixtures", "float display stages exact to 1e-6 (AUTHORITATIVE). DN under production floor(x*255): P1 #D0CFCE, P2 #000006, P3 #193F10, P4 #838689, P5 #195710. DN under round-half-up (the unified quantiser): P1 #D0D0CF, P2 #000006, P3 #193F11, P4 #84868A, P5 #195711."),
    ("V09", "Radiance/reflectance unit test", "match to 1e-6; range 0..1"),
    ("V10", "No double gamma / no double transfer function", "exactly 1 encoding; 0 ICC profiles"),
    ("V11", "Sensor-specific channel test", "exact band identity; wavelength to 1e-6"),
    ("V12", "Synthetic-channel reference test", "coefficients exact; fixture values to 1e-6"),
    ("V13", "Linear-light resampling (domain gate)", "0 display-domain ops (GIBS excepted); mean signed DN error within +/- 0.05 DN of zero"),
    ("V14", "Signed-bias closed form", "|relative error| <= 0.5 % at every C"),
    ("V15", "No manufactured structure (out-of-range gate)", "exactly 0.000000"),
    ("V16", "Edge overshoot / ringing bound", "<= production, i.e. no increase"),
    ("V17", "Neutral-edge chroma gate", "== 0 at EVERY edge column parity"),
    ("V18", "No hidden sharpening", "0 forbidden operators; noise variance ratio <= 1.0"),
    ("V19", "self_sharpen disabled", "false everywhere"),
    ("V20", "Archive blockiness / no smooth upscale", "delivered at native px (V05); if displayed upscaled, mean flat run of the DELIVERED raster <= 12 px"),
    ("V21", "One quantiser, round-half-up", "1 implementation; V08 fixtures exact"),
    ("V22", "Still / loop / download parity", "byte-identical pre-encode"),
    ("V23", "No double resize (resample-count gate)", "== 1"),
    ("V24", "Browser / server delivered-size agreement", "no CSS resample of a correctly sized image"),
    ("V25", "Deterministic output checksum", "3 of 3 identical"),
    ("V26", "Output dimension invariance", "exactly 1 distinct size; no tolerance"),
    ("V27", "Source-grid invariance", "exactly 1 distinct shape"),
    ("V28", "Fixed-geography registration", "<= 0.25 px"),
    ("V29", "Chrome invariance", "exactly 1 distinct hash"),
    ("V30", "Stationary-region radiometric flicker", "mean/median/p1 luminance <= 1.0 DN; p95 <= 2.5; p999 <= 5.0; Laplacian RMS <= 2.0; R/G ratio <= 0.005; B/G <= 0.015; mean saturation <= 0.010; black-clipped fraction <= 0.005; white-clipped <= 0.005"),
    ("V31", "No per-frame adaptive transform", "byte-identical LUTs"),
    ("V32", "Terminator blend fidelity", "within 2.0 DN of pure clean-IR"),
    ("V33", "No terminator luminance step", "<= 12 DN"),
    ("V34", "Illumination label correctness", "exact string match; 0 mislabelled frames"),
    ("V35", "Product-class labelling honesty", "0 violations"),
    ("V36", "Rayleigh actually ran", "0 frames with SKIPPED_LUT_UNAVAILABLE where required; startup assertion present"),
    ("V37", "Cache and version integrity", "key contains all three; 1 distinct ALGO_VERSION per sequence; all hashes reproduce"),
    ("V38", "Liveness on CHANGED observations", "0 slots counted fresh without all three terms"),
    ("V39", "Stale / missing source-data failure tests", "4 of 4 injected faults produce the declared behaviour"),
    ("V40", "Missing-band behaviour", "slot rejected; declared in the manifest"),
    ("V41", "No-data transparency", "0 interpolated no-data pixels; composite domain == linear_premultiplied"),
    ("V42", "Observation-selection honesty", "displacement disclosed on every frame; > 30 min carries a notice"),
    ("V43", "Sequence completeness disclosure (SOFT)", "warn, not block: disclosure present whenever a gap exists"),
    ("V44", "Antimeridian correctness", "0 discontinuous seam pixels; graticule continuous"),
    ("V45", "Full-basin coverage", "0 claimed basin-era pairs that fail to render or fail a hard gate"),
    ("V46", "Channel registration", "<= 0.25 px, or the era's published offset if larger and declared"),
)

_BASELINE_PASS_IDS: Final = frozenset(
    {
        "V07",
        "V08",
        "V09",
        "V10",
        "V11",
        "V12",
        "V15",
        "V16",
        "V18",
        "V25",
        "V30",
        "V31",
    }
)


def _baseline_status(gate_id: str) -> BaselineStatus:
    if gate_id == "V46":
        return BaselineStatus.UNMEASURED
    if gate_id in _BASELINE_PASS_IDS:
        return BaselineStatus.PASS
    return BaselineStatus.FAIL


GATES: Final[tuple[GateSpec, ...]] = tuple(
    GateSpec(
        gate_id=gate_id,
        name=name,
        threshold=threshold,
        baseline_status=_baseline_status(gate_id),
        severity=GateSeverity.SOFT if gate_id == "V43" else GateSeverity.HARD,
    )
    for gate_id, name, threshold in _GATE_ROWS
)
GATES_BY_ID: Final[Mapping[str, GateSpec]] = MappingProxyType(
    {spec.gate_id: spec for spec in GATES}
)


def validate_gate_catalog(specs: Iterable[GateSpec] = GATES) -> None:
    """Raise if a catalog is not exactly one definition for each V01--V46."""

    materialized = tuple(specs)
    ids = tuple(spec.gate_id for spec in materialized)
    expected = tuple(f"V{number:02d}" for number in range(1, 47))
    if len(ids) != len(set(ids)):
        raise ValueError("gate IDs must be unique")
    if tuple(sorted(ids)) != expected:
        raise ValueError("gate catalog must contain exactly V01 through V46")
    soft = tuple(spec.gate_id for spec in materialized if spec.severity is GateSeverity.SOFT)
    if soft != ("V43",):
        raise ValueError("V43 must be the sole soft gate")
    unmeasured = tuple(
        spec.gate_id
        for spec in materialized
        if spec.baseline_status is BaselineStatus.UNMEASURED
    )
    if unmeasured != ("V46",):
        raise ValueError("V46 must remain the sole unmeasured research-baseline gate")


def gate_spec(gate_id: str) -> GateSpec:
    try:
        return GATES_BY_ID[gate_id]
    except KeyError as exc:
        raise KeyError(f"unknown true-colour validation gate: {gate_id}") from exc


def result_for_measurement(gate_id: str, passed: bool | None, detail: str = "") -> GateResult:
    """Create a result without ever converting missing evidence into a pass."""

    spec = gate_spec(gate_id)
    if passed is True:
        outcome = GateOutcome.PASS
    elif passed is None:
        outcome = GateOutcome.UNMEASURED
    elif spec.severity is GateSeverity.SOFT:
        outcome = GateOutcome.SOFT_WARNING
    else:
        outcome = GateOutcome.FAIL
    return GateResult(spec=spec, outcome=outcome, detail=detail)


def research_baseline_result(gate_id: str) -> GateResult:
    spec = gate_spec(gate_id)
    measured = {
        BaselineStatus.PASS: True,
        BaselineStatus.FAIL: False,
        BaselineStatus.UNMEASURED: None,
    }[spec.baseline_status]
    return result_for_measurement(gate_id, measured, "research-package baseline")


validate_gate_catalog()
