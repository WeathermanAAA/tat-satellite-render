"""M0 baseline evaluator for the canonical V01--V46 contracts.

The baseline command is intentionally honest about evidence provenance.  A
current-HEAD check is used only where M0 can run one deterministically; every
other result remains either the canonical research baseline or explicitly
unmeasured.  A historical failure is never converted into a passing fixture.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import numpy as np

from . import production_adapter
from .contracts import (
    GATES,
    GateOutcome,
    GateResult,
    GateSeverity,
    research_baseline_result,
    result_for_measurement,
)
from .metrics import (
    log_domain_pair_relative_bias,
    source_column_coverage,
)
from .reference_pixels import (
    DERIVED_INVARIANTS,
    DISPLAY_ATOL,
    EIGHT_BIT_DN_ATOL,
    EIGHT_BIT_DN_SCOPE,
    PROTECTED_CONSTANTS,
    REFERENCE_PIXELS,
    REFLECTANCE_ATOL,
    expected_dn,
    quantize_display,
)


EXPECTED_VALIDATION_GATES_SHA256 = (
    "ce1786c7b79c89885fa4b530cc3280c73ad79ce452027e4c71869fc4857f3322"
)
EXPECTED_REFERENCE_PIXELS_SHA256 = (
    "9205f9373264f614f4011d18bdb9b8f33c2f5a7a346f9d37cf6b98b80b42d38a"
)
RAYLEIGH_STARTUP_SENSORS = (
    ("GOES-16", "abi"),
    ("GOES-17", "abi"),
    ("GOES-18", "abi"),
    ("GOES-19", "abi"),
    ("Himawari-8", "ahi"),
    ("Himawari-9", "ahi"),
    ("GEO-KOMPSAT-2A", "ami"),
    ("Meteosat-12", "fci"),
)


class ResearchIntegrityError(RuntimeError):
    pass


class RayleighEnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RayleighSensorEvidence:
    platform: str
    sensor: str
    status: str
    corrector_class: str | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RayleighStartupEvidence:
    data_directory: str
    config_file: str
    config_sha256: str
    configured_rsr_directory: str
    configured_rayleigh_directory: str
    writable_probe_passed: bool
    sensors: tuple[RayleighSensorEvidence, ...]

    @property
    def passed(self) -> bool:
        return (
            self.writable_probe_passed
            and len(self.sensors) == len(RAYLEIGH_STARTUP_SENSORS)
            and all(item.status == "valid_corrector" for item in self.sensors)
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


@dataclass(frozen=True, slots=True)
class CanonicalGateRow:
    gate_id: str
    name: str
    procedure: str
    threshold: str
    current_value: str
    fails_today: str
    source_document: str


@dataclass(frozen=True, slots=True)
class ResearchInputs:
    root: Path
    validation_gates_path: Path
    validation_gates_sha256: str
    reference_pixels_path: Path
    reference_pixels_sha256: str
    gates: Mapping[str, CanonicalGateRow]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "validation_gates_csv": {
                "path": str(self.validation_gates_path),
                "sha256": self.validation_gates_sha256,
            },
            "reference_pixels_json": {
                "path": str(self.reference_pixels_path),
                "sha256": self.reference_pixels_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class GateEvidence:
    gate_id: str
    name: str
    threshold: str
    outcome: GateOutcome
    severity: GateSeverity
    blocking: bool
    evidence_provenance: str
    detail: str
    measurements: Mapping[str, Any]
    canonical_current_value: str
    canonical_fails_today: str
    canonical_procedure: str
    canonical_source_document: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outcome"] = self.outcome.value
        data["severity"] = self.severity.value
        return data


@dataclass(frozen=True, slots=True)
class M0Enforcement:
    requested: bool
    passed: bool
    issues: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BaselineRun:
    research: ResearchInputs
    gates: tuple[GateEvidence, ...]
    port_fidelity: tuple[production_adapter.PortFidelityResult, ...]
    rayleigh_startup: RayleighStartupEvidence | None
    enforcement: M0Enforcement

    @property
    def counts(self) -> Mapping[str, int]:
        counts = {outcome.value: 0 for outcome in GateOutcome}
        for gate in self.gates:
            counts[gate.outcome.value] += 1
        return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_external_existing_path(
    value: str | os.PathLike[str] | Path,
    *,
    label: str,
    directory: bool,
) -> Path:
    try:
        resolved = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RayleighEnvironmentError(f"{label} does not resolve: {value}") from exc
    if directory and not resolved.is_dir():
        raise RayleighEnvironmentError(f"{label} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise RayleighEnvironmentError(f"{label} is not a file: {resolved}")
    for ancestor in (resolved, *resolved.parents):
        if os.path.lexists(ancestor / ".git"):
            raise RayleighEnvironmentError(
                f"{label} must be outside every Git worktree: {resolved}"
            )
    return resolved


def _prove_directory_writable(directory: Path) -> None:
    descriptor = -1
    probe_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".tat-rayleigh-write-probe-", dir=directory
        )
        probe_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(b"tat-rayleigh-write-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        probe_path.unlink()
        probe_path = None
        directory_descriptor = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise RayleighEnvironmentError(
            f"pyspectral data directory is not durably writable: {directory}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def run_rayleigh_startup_check(
    data_directory: str | os.PathLike[str] | Path,
    *,
    config_file: str | os.PathLike[str] | Path | None = None,
    make_corrector: Callable[[str, str], object | None] | None = None,
) -> RayleighStartupEvidence:
    """Call the production factory for all eight required sensor/platform pairs."""

    data_path = _require_external_existing_path(
        data_directory, label="pyspectral data directory", directory=True
    )
    configured_path = config_file or os.environ.get("PSP_CONFIG_FILE")
    if not configured_path:
        raise RayleighEnvironmentError(
            "PSP_CONFIG_FILE must name an explicit external pyspectral config"
        )
    config_path = _require_external_existing_path(
        configured_path, label="pyspectral config", directory=False
    )

    try:
        import yaml
    except ImportError as exc:
        raise RayleighEnvironmentError(
            "PyYAML is required to verify the pyspectral configuration"
        ) from exc
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, yaml.YAMLError) as exc:
        raise RayleighEnvironmentError(
            f"cannot parse pyspectral config {config_path}: {exc}"
        ) from exc
    if not isinstance(config, Mapping):
        raise RayleighEnvironmentError(
            f"pyspectral config is not a mapping: {config_path}"
        )
    configured_directories: dict[str, Path] = {}
    for key in ("rsr_dir", "rayleigh_dir"):
        raw_value = config.get(key)
        if not isinstance(raw_value, str) or not raw_value:
            raise RayleighEnvironmentError(
                f"pyspectral config lacks a non-empty {key}: {config_path}"
            )
        configured_directories[key] = Path(raw_value).expanduser().resolve(
            strict=False
        )
        if configured_directories[key] != data_path:
            raise RayleighEnvironmentError(
                f"pyspectral {key} resolves to {configured_directories[key]}, "
                f"not declared data directory {data_path}"
            )

    if make_corrector is None:
        environment_config = os.environ.get("PSP_CONFIG_FILE")
        if not environment_config:
            raise RayleighEnvironmentError(
                "PSP_CONFIG_FILE is unset for the production Rayleigh factory"
            )
        environment_path = Path(environment_config).expanduser().resolve(strict=True)
        if environment_path != config_path:
            raise RayleighEnvironmentError(
                f"production PSP_CONFIG_FILE resolves to {environment_path}, "
                f"not validated config {config_path}"
            )
        truecolor = production_adapter.production_modules()["truecolor"]
        make_corrector = truecolor._make_rayleigh

    _prove_directory_writable(data_path)
    sensor_evidence: list[RayleighSensorEvidence] = []
    for platform_name, sensor in RAYLEIGH_STARTUP_SENSORS:
        try:
            corrector = make_corrector(platform_name, sensor)
        except Exception as exc:  # collect every sensor failure before blocking
            sensor_evidence.append(
                RayleighSensorEvidence(
                    platform=platform_name,
                    sensor=sensor,
                    status="exception",
                    corrector_class=None,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if corrector is None:
            sensor_evidence.append(
                RayleighSensorEvidence(
                    platform=platform_name,
                    sensor=sensor,
                    status="returned_none",
                    corrector_class=None,
                    detail="production factory returned None",
                )
            )
            continue
        corrector_type = type(corrector)
        sensor_evidence.append(
            RayleighSensorEvidence(
                platform=platform_name,
                sensor=sensor,
                status="valid_corrector",
                corrector_class=(
                    f"{corrector_type.__module__}.{corrector_type.__qualname__}"
                ),
            )
        )

    return RayleighStartupEvidence(
        data_directory=str(data_path),
        config_file=str(config_path),
        config_sha256=_sha256(config_path),
        configured_rsr_directory=str(configured_directories["rsr_dir"]),
        configured_rayleigh_directory=str(configured_directories["rayleigh_dir"]),
        writable_probe_passed=True,
        sensors=tuple(sensor_evidence),
    )


def _fixture_projection_from_json(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    projected: list[dict[str, Any]] = []
    for fixture in payload["fixtures"]:
        stages = fixture["stages"]
        floor = stages["s7_8bit_DN_floor"]
        round_half_up = stages["s7_8bit_DN_round_half_up"]
        projected.append(
            {
                "id": fixture["id"],
                "sensor": fixture["sensor"],
                "cos_sza": stages["geometry"]["cos_sza"],
                "reflectance_factor": {
                    role: stages["s1_cmi_reflectance_factor"].get(role)
                    for role in ("red", "blue", "veggie", "green")
                },
                "sunz_corrected": {
                    role: stages["s2_sunz_corrected"].get(role)
                    for role in ("red", "blue", "veggie", "green")
                },
                "rayleigh_corrected": {
                    role: stages["s3_rayleigh_corrected"].get(role)
                    for role in ("red", "blue", "veggie", "green")
                },
                "synthesized_green": stages["s4_green"]["value"],
                "linear_rgb": tuple(stages["s4_rgb_linear_raw"]),
                "display_rgb": tuple(stages["s6_cira_display"]),
                "dn_floor_production": (floor["R"], floor["G"], floor["B"]),
                "dn_round_half_up": (
                    round_half_up["R"],
                    round_half_up["G"],
                    round_half_up["B"],
                ),
                "dn_floor_hex": floor["hex"],
                "dn_round_half_up_hex": round_half_up["hex"],
            }
        )
    return tuple(projected)


def _fixture_projection_from_local() -> tuple[dict[str, Any], ...]:
    projected: list[dict[str, Any]] = []
    for fixture in REFERENCE_PIXELS:
        floor_hex = "#" + "".join(
            f"{value:02X}" for value in fixture.dn_floor_production
        )
        round_half_up_hex = "#" + "".join(
            f"{value:02X}" for value in fixture.dn_round_half_up
        )
        projected.append(
            {
                "id": fixture.id,
                "sensor": fixture.sensor,
                "cos_sza": fixture.cos_sza,
                "reflectance_factor": {
                    role: getattr(fixture.reflectance_factor, role)
                    for role in ("red", "blue", "veggie", "green")
                },
                "sunz_corrected": {
                    role: getattr(fixture.sunz_corrected, role)
                    for role in ("red", "blue", "veggie", "green")
                },
                "rayleigh_corrected": {
                    role: getattr(fixture.rayleigh_corrected, role)
                    for role in ("red", "blue", "veggie", "green")
                },
                "synthesized_green": fixture.synthesized_green,
                "linear_rgb": fixture.linear_rgb,
                "display_rgb": fixture.display_rgb,
                "dn_floor_production": fixture.dn_floor_production,
                "dn_round_half_up": fixture.dn_round_half_up,
                "dn_floor_hex": floor_hex,
                "dn_round_half_up_hex": round_half_up_hex,
            }
        )
    return tuple(projected)


def verify_local_reference_projection(payload: Mapping[str, Any]) -> None:
    """Prove executable fixtures are an exact projection of the verified JSON."""

    canonical_constants = dict(payload["pipeline_constants"])
    canonical_constants["_CIRA_LOG_ROOT"] = canonical_constants.pop("CIRA_LOG_ROOT")
    canonical_constants["_CIRA_DENOM"] = canonical_constants.pop("CIRA_DENOM")
    for name in ("ABI_GREEN_FRACTIONS", "NDVI_GREEN_LIMITS"):
        canonical_constants[name] = tuple(canonical_constants[name])
    if dict(PROTECTED_CONSTANTS) != canonical_constants:
        raise ResearchIntegrityError(
            "local protected constants diverge from canonical reference_pixels.json"
        )

    canonical_derived = dict(payload["derived_invariants"])
    canonical_derived["cira_dn_per_octave"] = canonical_derived.pop(
        "cira_DN_per_octave_of_reflectance"
    )
    canonical_derived["cira_dn_at_unit_reflectance_floor_production"] = (
        canonical_derived.pop("cira_DN_at_unit_reflectance_floor_production")
    )
    canonical_derived["cira_dn_at_unit_reflectance_round_half_up"] = (
        canonical_derived.pop("cira_DN_at_unit_reflectance_round_half_up")
    )
    canonical_derived["cira_headroom_dn_above_unit_reflectance_floor_production"] = (
        canonical_derived.pop(
            "cira_headroom_DN_above_unit_reflectance_floor_production"
        )
    )
    if dict(DERIVED_INVARIANTS) != canonical_derived:
        raise ResearchIntegrityError(
            "local derived invariants diverge from canonical reference_pixels.json"
        )

    tolerances = payload["tolerances"]
    local_tolerances = {
        "reflectance_stages": REFLECTANCE_ATOL,
        "display_stages": DISPLAY_ATOL,
        "eight_bit_DN": EIGHT_BIT_DN_ATOL,
        "eight_bit_DN_scope": EIGHT_BIT_DN_SCOPE,
    }
    if local_tolerances != tolerances:
        raise ResearchIntegrityError(
            "local tolerances diverge from canonical reference_pixels.json"
        )

    if _fixture_projection_from_local() != _fixture_projection_from_json(payload):
        raise ResearchIntegrityError(
            "local executable fixtures diverge from canonical reference_pixels.json"
        )


def verify_research_root(root: str | os.PathLike[str] | Path) -> ResearchInputs:
    """Verify and load the two canonical files required by every M0 run."""

    research_root = Path(root).expanduser().resolve(strict=True)
    if not research_root.is_dir():
        raise ResearchIntegrityError(f"research root is not a directory: {research_root}")
    gates_path = research_root / "tables" / "validation_gates.csv"
    pixels_path = research_root / "sources" / "reference_pixels.json"
    for path in (gates_path, pixels_path):
        if not path.is_file():
            raise ResearchIntegrityError(f"canonical research input is missing: {path}")

    gates_hash = _sha256(gates_path)
    pixels_hash = _sha256(pixels_path)
    if gates_hash != EXPECTED_VALIDATION_GATES_SHA256:
        raise ResearchIntegrityError(
            "validation_gates.csv SHA-256 mismatch: "
            f"expected {EXPECTED_VALIDATION_GATES_SHA256}, received {gates_hash}"
        )
    if pixels_hash != EXPECTED_REFERENCE_PIXELS_SHA256:
        raise ResearchIntegrityError(
            "reference_pixels.json SHA-256 mismatch: "
            f"expected {EXPECTED_REFERENCE_PIXELS_SHA256}, received {pixels_hash}"
        )

    with gates_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = tuple(csv.DictReader(handle))
    rows: dict[str, CanonicalGateRow] = {}
    for raw in raw_rows:
        gate_id = raw.get("gate_id", "")
        if gate_id in rows:
            raise ResearchIntegrityError(f"duplicate canonical gate ID: {gate_id}")
        try:
            rows[gate_id] = CanonicalGateRow(
                gate_id=gate_id,
                name=raw["name"],
                procedure=raw["procedure"],
                threshold=raw["threshold"],
                current_value=raw["current_value"],
                fails_today=raw["fails_today"],
                source_document=raw["source_document"],
            )
        except KeyError as exc:
            raise ResearchIntegrityError(
                f"validation_gates.csv lacks canonical column {exc.args[0]!r}"
            ) from exc
    expected_ids = tuple(gate.gate_id for gate in GATES)
    if tuple(rows) != expected_ids:
        raise ResearchIntegrityError("validation_gates.csv must contain ordered V01--V46")

    try:
        pixels = json.loads(pixels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchIntegrityError(f"reference_pixels.json is unreadable: {exc}") from exc
    fixtures = pixels.get("fixtures")
    if pixels.get("schema") != "tat_truecolor_reference_pixels/v1" or not isinstance(
        fixtures, list
    ) or len(fixtures) != 5:
        raise ResearchIntegrityError("reference pixel schema/fixture count is not canonical")
    for fixture in fixtures:
        stages = fixture.get("stages", {})
        if "s7_8bit_DN_floor" not in stages or "s7_8bit_DN_round_half_up" not in stages:
            raise ResearchIntegrityError("reference pixel is missing corrected C3 DN columns")
    verify_local_reference_projection(pixels)

    return ResearchInputs(
        root=research_root,
        validation_gates_path=gates_path,
        validation_gates_sha256=gates_hash,
        reference_pixels_path=pixels_path,
        reference_pixels_sha256=pixels_hash,
        gates=rows,
    )


@dataclass(frozen=True, slots=True)
class _Check:
    passed: bool | None
    detail: str
    measurements: Mapping[str, Any]
    evidence_provenance: str = "current_head"


def _check_v02() -> _Check:
    coverage = source_column_coverage(1600, 1188)
    passed = coverage.discarded_columns == 0
    return _Check(
        passed,
        "current nearest point sampling evaluated at corrected C1 geometry",
        asdict(coverage),
    )


def _check_v04() -> _Check:
    calls = production_adapter.imshow_calls()
    call_values = [
        {"branch": call.branch, "line": call.line, "interpolation": call.interpolation}
        for call in calls
    ]
    passed = bool(calls) and all(call.interpolation == "hanning" for call in calls)
    return _Check(passed, "literal interpolation policy from current render.py AST", {"calls": call_values})


def _check_v07() -> _Check:
    truecolor = production_adapter.production_modules()["truecolor"]
    mismatches: dict[str, dict[str, Any]] = {}
    for attribute, expected in PROTECTED_CONSTANTS.items():
        actual = getattr(truecolor, attribute, "<missing>")
        if actual != expected:
            mismatches[attribute] = {"expected": expected, "actual": actual}
    return _Check(
        not mismatches,
        "protected constants read from imported production truecolor module",
        {
            "mismatches": mismatches,
            "cira_stretch_source_sha256": production_adapter.cira_source_sha256(),
        },
    )


def _evaluate_reference_pixels() -> tuple[list[str], dict[str, str]]:
    truecolor = production_adapter.production_modules()["truecolor"]
    failures: list[str] = []
    delivered: dict[str, str] = {}
    for pixel in REFERENCE_PIXELS:
        cos_sza = np.asarray([pixel.cos_sza], dtype=np.float64)
        for role in ("red", "blue", "green"):
            source = getattr(pixel.reflectance_factor, role)
            expected = getattr(pixel.sunz_corrected, role)
            if source is None:
                continue
            actual = truecolor.sunz_correct(np.asarray([source], dtype=np.float32), cos_sza)
            if not np.allclose(actual, [expected], rtol=0.0, atol=REFLECTANCE_ATOL):
                failures.append(f"{pixel.id}: sunz {role}")
        veggie = truecolor.sunz_correct_pathlength(
            np.asarray([pixel.reflectance_factor.veggie], dtype=np.float32), cos_sza
        )
        if not np.allclose(
            veggie, [pixel.sunz_corrected.veggie], rtol=0.0, atol=REFLECTANCE_ATOL
        ):
            failures.append(f"{pixel.id}: sunz veggie")

        rayleigh = pixel.rayleigh_corrected
        green = truecolor.make_green(
            pixel.sensor,
            np.asarray([rayleigh.red], dtype=np.float32),
            np.asarray([rayleigh.blue], dtype=np.float32),
            None
            if rayleigh.green is None
            else np.asarray([rayleigh.green], dtype=np.float32),
            np.asarray([rayleigh.veggie], dtype=np.float32),
        )
        if not np.allclose(
            green, [pixel.synthesized_green], rtol=0.0, atol=REFLECTANCE_ATOL
        ):
            failures.append(f"{pixel.id}: synthesized green")
        linear_rgb = np.asarray(
            [[rayleigh.red, float(green[0]), rayleigh.blue]], dtype=np.float32
        )
        display = truecolor.cira_stretch(linear_rgb)[0]
        if not np.allclose(display, pixel.display_rgb, rtol=0.0, atol=DISPLAY_ATOL):
            failures.append(f"{pixel.id}: cira display")
        quantized = tuple(int(value) for value in quantize_display(display, "floor_production"))
        if quantized != expected_dn(pixel, "floor_production"):
            failures.append(f"{pixel.id}: declared floor DN")
        delivered[pixel.id] = "#" + "".join(f"{value:02X}" for value in quantized)
    return failures, delivered


def _check_v08() -> _Check:
    failures, delivered = _evaluate_reference_pixels()
    return _Check(
        not failures,
        "five fixtures executed through imported production pointwise functions; M0 declares floor",
        {"declared_quantizer": "production_floor", "failures": failures, "delivered_hex": delivered},
    )


def _check_v10(
    port_results: Sequence[production_adapter.PortFidelityResult],
) -> _Check:
    order = production_adapter.assemble_call_order()
    source_findings: list[str] = []
    for name in ("app.py", "render.py", "satellites.py", "truecolor.py"):
        text = (production_adapter.REPO_ROOT / name).read_text(encoding="utf-8").casefold()
        if "icc_profile" in text or "icc profile" in text:
            source_findings.append(name)
    png_evidence = [
        {
            "case_id": result.case_id,
            "mode": result.png_mode,
            "info_keys": list(result.png_info_keys),
            "icc_profile_present": result.png_icc_profile_present,
            "gamma_present": result.png_gamma_present,
            "srgb_present": result.png_srgb_present,
            "quantization_model": result.quantization_model,
            "floor_quantization_identity": (
                result.floor_quantization_scored_identity
            ),
            "round_half_up_mutant_identity": (
                result.round_half_up_mutant_scored_identity
            ),
        }
        for result in port_results
    ]
    emitted_png_passes = all(
        not result.png_icc_profile_present
        and not result.png_gamma_present
        and not result.png_srgb_present
        and result.quantization_model == "float_nearest_then_floor_u8"
        and result.floor_quantization_scored_identity
        >= result.minimum_scored_identity
        and result.round_half_up_mutant_scored_identity < 0.001
        for result in port_results
    )
    passed = (
        order.count("cira_stretch") == 1
        and not source_findings
        and emitted_png_passes
    )
    return _Check(
        passed,
        "production call order plus emitted-PNG transfer metadata and quantization mutant",
        {
            "assemble_call_order": list(order),
            "icc_source_findings": source_findings,
            "emitted_png": png_evidence,
        },
    )


def _check_v12() -> _Check:
    truecolor = production_adapter.production_modules()["truecolor"]
    coefficient_match = tuple(truecolor.ABI_GREEN_FRACTIONS) == (0.45, 0.45, 0.10)
    failures, _ = _evaluate_reference_pixels()
    green_failures = [failure for failure in failures if "green" in failure]
    return _Check(
        coefficient_match and not green_failures,
        "production ABI coefficients plus P1/P3/P5 production make_green fixtures",
        {
            "abi_green_fractions": list(truecolor.ABI_GREEN_FRACTIONS),
            "green_fixture_failures": green_failures,
        },
    )


def _check_v14() -> _Check:
    biases = {
        str(contrast): 100.0 * float(log_domain_pair_relative_bias(contrast))
        for contrast in (2, 5, 20, 50)
    }
    order = production_adapter.assemble_call_order()
    passed = all(abs(value) <= 0.5 for value in biases.values())
    return _Check(
        passed,
        "closed-form display/log-domain mutant selected by current cira-before-imshow order",
        {"relative_error_pct": biases, "assemble_call_order": list(order)},
    )


def _check_v18(
    port_results: Sequence[production_adapter.PortFidelityResult],
) -> _Check:
    findings = production_adapter.forbidden_operator_findings()
    square = next(result for result in port_results if result.case_id == "bbox_1x1")
    variance_passes = square.noise_variance_ratio <= 1.0 + 1e-12
    return _Check(
        not findings and variance_passes,
        "AST operator scan plus deterministic production-render noise measurement",
        {
            "forbidden_operator_findings": list(findings),
            "noise_variance_ratio": square.noise_variance_ratio,
            "noise_variance_threshold": "<= 1.0",
            "noise_fixture": "bbox_1x1 stationary seeded noise",
            "numeric_core_exclusion_px": square.numeric_core_exclusion_px,
        },
    )


def _check_v19() -> _Check:
    truecolor = production_adapter.production_modules()["truecolor"]
    default = inspect.signature(truecolor.assemble_truecolor).parameters[
        "do_ratio_sharpen"
    ].default
    blocks = dict(truecolor.SHARPEN_BLOCK)
    passed = default is False and all(int(value) == 1 for value in blocks.values())
    return _Check(
        passed,
        "production function default and per-sensor SHARPEN_BLOCK values",
        {"do_ratio_sharpen_default": default, "sharpen_block": blocks},
    )


def separate_process_repeatability_hashes() -> tuple[str, str, str]:
    """Render the same defined request in three fresh Python processes."""

    worker = (
        "from validation.truecolor.production_adapter import "
        "port_probe_png_sha256; print(port_probe_png_sha256('bbox_1x1'))"
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    hashes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="tat-m0-mplconfig-") as mpl_config:
        environment["MPLCONFIGDIR"] = mpl_config
        for _ in range(3):
            completed = subprocess.run(
                [sys.executable, "-c", worker],
                cwd=production_adapter.REPO_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            digest = completed.stdout.strip().splitlines()[-1]
            if len(digest) != 64:
                raise RuntimeError(f"repeatability worker returned invalid digest: {digest!r}")
            hashes.append(digest)
    return tuple(hashes)  # type: ignore[return-value]


def _check_v25(hashes: Sequence[str]) -> _Check:
    passed = len(hashes) == 3 and len(set(hashes)) == 1 and all(len(value) == 64 for value in hashes)
    return _Check(
        passed,
        "three identical production requests rendered in separate processes",
        {"sha256_runs": list(hashes), "distinct_hashes": len(set(hashes))},
    )


def gridline_boundary_probe(step_function: Callable[[float], float]) -> Mapping[str, Any]:
    exact = 12.0
    above = float(np.nextafter(exact, np.inf))
    exact_step = float(step_function(exact))
    above_step = float(step_function(above))
    return {
        "span_exact": exact,
        "span_nextafter": above,
        "step_exact": exact_step,
        "step_nextafter": above_step,
        "stable": exact_step == above_step,
    }


def _check_v29() -> _Check:
    render = production_adapter.production_modules()["render"]
    probe = gridline_boundary_probe(render._gridline_step)
    return _Check(
        False,
        "diagnostic ULP proxy only; exact sequence chrome hashes were not rendered",
        {
            **probe,
            "proxy_only": True,
            "pass_eligible": False,
            "missing_gate_evidence": "exactly one distinct chrome hash across sequence",
        },
        evidence_provenance="current_head_proxy",
    )


def _call_names(function: Callable[..., Any]) -> set[str]:
    import ast

    tree = ast.parse(inspect.getsource(function))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id.casefold())
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr.casefold())
    return names


def _check_v31() -> _Check:
    truecolor = production_adapter.production_modules()["truecolor"]
    checked = (
        truecolor.assemble_truecolor,
        truecolor.cira_stretch,
        truecolor.sunz_correct,
        truecolor.night_blend,
    )
    adaptive_names = {
        "percentile",
        "nanpercentile",
        "histogram",
        "histogram_equalize",
        "equalize_hist",
        "equalize_adapthist",
    }
    found = sorted(set().union(*(_call_names(function) for function in checked)) & adaptive_names)
    return _Check(
        not found,
        "AST call scan of production true-colour transforms for per-frame adaptive operators",
        {"adaptive_call_findings": found},
    )


def _check_v36() -> _Check:
    truecolor = production_adapter.production_modules()["truecolor"]
    factory_source = inspect.getsource(truecolor._make_rayleigh)
    frame_source = inspect.getsource(truecolor.assemble_truecolor)
    repo_sources = "\n".join(
        (production_adapter.REPO_ROOT / name).read_text(encoding="utf-8")
        for name in ("app.py", "render.py", "satellites.py", "truecolor.py")
    )
    startup_assertion_present = "assert_rayleigh_startup" in repo_sources
    frame_status_present = "rayleigh_status" in repo_sources
    silent_none_path_present = "return None" in factory_source and "_make_rayleigh" in frame_source
    passed = startup_assertion_present and frame_status_present and not silent_none_path_present
    return _Check(
        passed,
        "static current-HEAD startup/status/silent-None contract inspection; no LUT fetch attempted",
        {
            "startup_assertion_present": startup_assertion_present,
            "per_frame_rayleigh_status_present": frame_status_present,
            "silent_none_path_present": silent_none_path_present,
        },
    )


_ACTUAL_CHECK_IDS = frozenset(
    {"V02", "V04", "V07", "V08", "V10", "V12", "V14", "V18", "V19", "V25", "V29", "V31", "V36"}
)
_M0_PROTECTIVE_PASS_IDS = frozenset({"V07", "V08", "V10", "V12", "V18", "V25", "V31"})
_M0_PORT_FIDELITY_REQUIREMENTS: Final = (
    ("bbox_1x1", 0.998, False),
    ("bbox_2x1", 1.0, True),
)


def _run_actual_checks(
    repeatability_hashes: Sequence[str],
    port_results: Sequence[production_adapter.PortFidelityResult],
) -> Mapping[str, _Check]:
    checks: dict[str, _Check] = {}
    evaluators: Mapping[str, Callable[[], _Check]] = {
        "V02": _check_v02,
        "V04": _check_v04,
        "V07": _check_v07,
        "V08": _check_v08,
        "V12": _check_v12,
        "V14": _check_v14,
        "V19": _check_v19,
        "V29": _check_v29,
        "V31": _check_v31,
        "V36": _check_v36,
    }
    for gate_id, evaluator in evaluators.items():
        checks[gate_id] = evaluator()
    checks["V10"] = _check_v10(port_results)
    checks["V18"] = _check_v18(port_results)
    checks["V25"] = _check_v25(repeatability_hashes)
    return checks


def _evidence_from_result(
    row: CanonicalGateRow,
    result: GateResult,
    provenance: str,
    measurements: Mapping[str, Any],
) -> GateEvidence:
    return GateEvidence(
        gate_id=row.gate_id,
        name=row.name,
        threshold=row.threshold,
        outcome=result.outcome,
        severity=result.spec.severity,
        blocking=result.blocking,
        evidence_provenance=provenance,
        detail=result.detail,
        measurements=measurements,
        canonical_current_value=row.current_value,
        canonical_fails_today=row.fails_today,
        canonical_procedure=row.procedure,
        canonical_source_document=row.source_document,
    )


def port_fidelity_contract_issues(
    port_results: Sequence[production_adapter.PortFidelityResult],
) -> tuple[str, ...]:
    """Validate port results against an enforcement-owned canonical contract.

    These requirements intentionally do not come from the adapter's case
    registry.  A mutation of the probe configuration therefore cannot weaken,
    remove, or rename a case and then re-bless itself during M0 enforcement.
    """

    issues: list[str] = []
    ports = {result.case_id: result for result in port_results}
    if len(ports) != len(port_results):
        issues.append("duplicate port-fidelity case IDs")
    expected_ids = tuple(case_id for case_id, _, _ in _M0_PORT_FIDELITY_REQUIREMENTS)
    unexpected = sorted(set(ports) - set(expected_ids))
    if unexpected:
        issues.append(f"unexpected port-fidelity cases: {', '.join(unexpected)}")
    for case_id, minimum_identity, exact_identity_required in (
        _M0_PORT_FIDELITY_REQUIREMENTS
    ):
        result = ports.get(case_id)
        if result is None:
            issues.append(f"missing canonical port-fidelity case {case_id}")
            continue
        if result.minimum_scored_identity != minimum_identity:
            issues.append(
                f"{case_id} declared minimum {result.minimum_scored_identity!r}; "
                f"canonical minimum is {minimum_identity!r}"
            )
        if result.exact_identity_required is not exact_identity_required:
            issues.append(
                f"{case_id} exact-identity declaration drifted from "
                f"{exact_identity_required}"
            )
        if not np.isfinite(result.scored_pixel_identity) or (
            result.scored_pixel_identity < minimum_identity
        ):
            issues.append(
                f"{case_id} port identity {result.scored_pixel_identity:.9f} is below "
                f"{minimum_identity:.9f}"
            )
        if exact_identity_required and result.scored_pixel_identity != 1.0:
            issues.append(f"{case_id} port identity is not exactly 1.0")
    return tuple(issues)


def _enforce_m0(
    evidence: Sequence[GateEvidence],
    port_results: Sequence[production_adapter.PortFidelityResult],
    rayleigh_startup: RayleighStartupEvidence | None,
    requested: bool,
) -> M0Enforcement:
    if not requested:
        return M0Enforcement(requested=False, passed=True, issues=())
    issues: list[str] = []
    by_id = {gate.gate_id: gate for gate in evidence}
    if tuple(by_id) != tuple(f"V{number:02d}" for number in range(1, 47)):
        issues.append("report does not contain ordered V01--V46")
    soft = [gate.gate_id for gate in evidence if gate.severity is GateSeverity.SOFT]
    if soft != ["V43"]:
        issues.append("V43 is not the sole soft gate")
    v46 = by_id.get("V46")
    if v46 is None or v46.outcome is not GateOutcome.UNMEASURED or not v46.blocking:
        issues.append("V46 must remain unmeasured and blocking at M0")
    for gate_id in sorted(_ACTUAL_CHECK_IDS):
        gate = by_id.get(gate_id)
        expected_provenance = (
            "current_head_proxy" if gate_id == "V29" else "current_head"
        )
        if gate is None or gate.evidence_provenance != expected_provenance:
            issues.append(f"{gate_id} lacks expected {expected_provenance} evidence")
    for gate_id in sorted(_M0_PROTECTIVE_PASS_IDS):
        gate = by_id.get(gate_id)
        if gate is None or gate.outcome is not GateOutcome.PASS:
            issues.append(f"M0 protective gate {gate_id} did not pass")
    issues.extend(port_fidelity_contract_issues(port_results))
    if rayleigh_startup is None:
        issues.append("M0 lacks the required eight-sensor Rayleigh startup check")
    elif not rayleigh_startup.passed:
        failed = [
            f"{item.platform}/{item.sensor}:{item.status}"
            for item in rayleigh_startup.sensors
            if item.status != "valid_corrector"
        ]
        issues.append("Rayleigh startup check failed: " + ", ".join(failed))
    return M0Enforcement(requested=True, passed=not issues, issues=tuple(issues))


def run_baseline(
    research: ResearchInputs,
    *,
    enforce_through: str | None = None,
    repeatability_hashes: Sequence[str] | None = None,
    rayleigh_data_directory: str | os.PathLike[str] | Path | None = None,
    rayleigh_config_file: str | os.PathLike[str] | Path | None = None,
    rayleigh_factory: Callable[[str, str], object | None] | None = None,
) -> BaselineRun:
    if enforce_through not in {None, "M0"}:
        raise ValueError("M0 is the only implemented enforcement stage")
    hashes = (
        tuple(repeatability_hashes)
        if repeatability_hashes is not None
        else separate_process_repeatability_hashes()
    )
    port_results = tuple(
        production_adapter.port_fidelity_probe(case_id)
        for case_id in production_adapter.PORT_FIDELITY_CASES
    )
    rayleigh_startup = (
        None
        if rayleigh_data_directory is None
        else run_rayleigh_startup_check(
            rayleigh_data_directory,
            config_file=rayleigh_config_file,
            make_corrector=rayleigh_factory,
        )
    )
    actual = _run_actual_checks(hashes, port_results)
    evidence: list[GateEvidence] = []
    for spec in GATES:
        row = research.gates[spec.gate_id]
        check = actual.get(spec.gate_id)
        if check is None:
            result = research_baseline_result(spec.gate_id)
            provenance = (
                "unmeasured"
                if result.outcome is GateOutcome.UNMEASURED
                else "canonical_research_baseline"
            )
            measurements: Mapping[str, Any] = {
                "canonical_current_value": row.current_value,
                "reason_not_rerun": "requires corpus, browser, live service, or a later-stage implementation",
            }
        else:
            result = result_for_measurement(spec.gate_id, check.passed, check.detail)
            provenance = check.evidence_provenance
            measurements = check.measurements
        evidence.append(_evidence_from_result(row, result, provenance, measurements))
    enforcement = _enforce_m0(
        evidence,
        port_results,
        rayleigh_startup,
        enforce_through == "M0",
    )
    return BaselineRun(
        research=research,
        gates=tuple(evidence),
        port_fidelity=port_results,
        rayleigh_startup=rayleigh_startup,
        enforcement=enforcement,
    )


class HttpFailureClass(str, Enum):
    NONE = "none"
    PERMANENT_AUTH = "permanent_auth"
    TRANSIENT = "transient"
    OTHER_PERMANENT = "other_permanent"


def classify_http_failure(status: int) -> HttpFailureClass:
    if 200 <= status < 300:
        return HttpFailureClass.NONE
    if status in {401, 403}:
        return HttpFailureClass.PERMANENT_AUTH
    if status in {408, 425, 429} or 500 <= status < 600:
        return HttpFailureClass.TRANSIENT
    return HttpFailureClass.OTHER_PERMANENT


def retry_permitted(status: int) -> bool:
    return classify_http_failure(status) is HttpFailureClass.TRANSIENT


def sequence_dimensions_invariant(dimensions: Iterable[tuple[int, int]]) -> bool:
    values = tuple(dimensions)
    return bool(values) and len(set(values)) == 1


def source_grid_invariant(shapes: Iterable[tuple[int, int]]) -> bool:
    values = tuple(shapes)
    return bool(values) and len(set(values)) == 1
