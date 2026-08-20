from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import tat_harness, tat_truecolor_validate
from validation.truecolor import gates
from validation.truecolor.contracts import GateOutcome
from validation.truecolor.report import (
    UnsafeArtifactPath,
    build_report,
    stable_json_bytes,
    validate_artifact_directory,
    write_report,
)


RESEARCH_ROOT_ENV = "TAT_TRUECOLOR_RESEARCH_ROOT"
THREE_IDENTICAL_HASHES = ("d" * 64, "d" * 64, "d" * 64)


@pytest.fixture(scope="module")
def research() -> gates.ResearchInputs:
    configured = os.environ.get(RESEARCH_ROOT_ENV)
    if not configured:
        pytest.skip(
            f"external contract tests require explicit {RESEARCH_ROOT_ENV}"
        )
    root = Path(configured).expanduser()
    if not root.is_dir():
        pytest.fail(f"configured {RESEARCH_ROOT_ENV} is not a directory: {root}")
    return gates.verify_research_root(root)


def _rayleigh_environment(tmp_path: Path) -> tuple[Path, Path]:
    data_directory = tmp_path / "pyspectral-data"
    data_directory.mkdir()
    config_file = tmp_path / "pyspectral.yaml"
    config_file.write_text(
        f"rsr_dir: {data_directory}\nrayleigh_dir: {data_directory}\n",
        encoding="utf-8",
    )
    return data_directory, config_file


def test_expected_research_hashes_are_full_canonical_sha256_values() -> None:
    assert gates.EXPECTED_VALIDATION_GATES_SHA256 == (
        "ce1786c7b79c89885fa4b530cc3280c73ad79ce452027e4c71869fc4857f3322"
    )
    assert gates.EXPECTED_REFERENCE_PIXELS_SHA256 == (
        "9205f9373264f614f4011d18bdb9b8f33c2f5a7a346f9d37cf6b98b80b42d38a"
    )


def test_research_root_verifies_both_files_and_all_canonical_rows(research) -> None:
    assert research.validation_gates_sha256 == gates.EXPECTED_VALIDATION_GATES_SHA256
    assert research.reference_pixels_sha256 == gates.EXPECTED_REFERENCE_PIXELS_SHA256
    assert tuple(research.gates) == tuple(f"V{number:02d}" for number in range(1, 47))
    for spec in gates.GATES:
        canonical = research.gates[spec.gate_id]
        assert spec.name == canonical.name
        assert spec.threshold == canonical.threshold


def test_missing_or_repackaged_research_input_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "tables").mkdir()
    (tmp_path / "sources").mkdir()
    (tmp_path / "tables" / "validation_gates.csv").write_text("not canonical\n")
    (tmp_path / "sources" / "reference_pixels.json").write_text("{}\n")
    with pytest.raises(gates.ResearchIntegrityError, match="SHA-256 mismatch"):
        gates.verify_research_root(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ("constant", "derived", "tolerance", "floor_dn", "round_half_up_dn"),
)
def test_local_reference_reblessing_cannot_diverge_from_verified_json(
    research, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    if mutation == "constant":
        changed = dict(gates.PROTECTED_CONSTANTS)
        changed["_CIRA_DENOM"] = 2.0
        monkeypatch.setattr(gates, "PROTECTED_CONSTANTS", changed)
    elif mutation == "derived":
        changed = dict(gates.DERIVED_INVARIANTS)
        changed["cira_dn_per_octave"] = 38.0
        monkeypatch.setattr(gates, "DERIVED_INVARIANTS", changed)
    elif mutation == "tolerance":
        monkeypatch.setattr(gates, "DISPLAY_ATOL", 2e-6)
    else:
        first = gates.REFERENCE_PIXELS[0]
        if mutation == "floor_dn":
            first = dataclasses.replace(
                first, dn_floor_production=(209, 207, 206)
            )
        else:
            first = dataclasses.replace(
                first, dn_round_half_up=(209, 208, 207)
            )
        monkeypatch.setattr(
            gates, "REFERENCE_PIXELS", (first, *gates.REFERENCE_PIXELS[1:])
        )

    with pytest.raises(gates.ResearchIntegrityError, match="diverge"):
        gates.verify_research_root(research.root)


def test_baseline_reports_all_canonical_thresholds_and_honest_provenance(
    research, tmp_path: Path
) -> None:
    data_directory, config_file = _rayleigh_environment(tmp_path)
    rayleigh_calls: list[tuple[str, str]] = []

    def factory(platform: str, sensor: str) -> object:
        rayleigh_calls.append((platform, sensor))
        return object()

    run = gates.run_baseline(
        research,
        enforce_through="M0",
        repeatability_hashes=THREE_IDENTICAL_HASHES,
        rayleigh_data_directory=data_directory,
        rayleigh_config_file=config_file,
        rayleigh_factory=factory,
    )
    assert len(run.gates) == 46
    assert [gate.gate_id for gate in run.gates] == [f"V{number:02d}" for number in range(1, 47)]
    assert all(gate.threshold == research.gates[gate.gate_id].threshold for gate in run.gates)

    by_id = {gate.gate_id: gate for gate in run.gates}
    for gate_id in ("V02", "V04", "V14", "V19", "V36"):
        assert by_id[gate_id].outcome is GateOutcome.FAIL
        assert by_id[gate_id].evidence_provenance == "current_head"
    assert by_id["V29"].outcome is GateOutcome.FAIL
    assert by_id["V29"].evidence_provenance == "current_head_proxy"
    assert by_id["V29"].measurements["pass_eligible"] is False
    for gate_id in ("V07", "V08", "V10", "V12", "V18", "V25", "V31"):
        assert by_id[gate_id].outcome is GateOutcome.PASS
        assert by_id[gate_id].evidence_provenance == "current_head"
    assert all(
        not item["icc_profile_present"]
        and not item["gamma_present"]
        and not item["srgb_present"]
        and item["floor_quantization_identity"] >= 0.998
        and item["round_half_up_mutant_identity"] < 0.001
        for item in by_id["V10"].measurements["emitted_png"]
    )

    assert by_id["V43"].outcome is GateOutcome.SOFT_WARNING
    assert not by_id["V43"].blocking
    assert by_id["V46"].outcome is GateOutcome.UNMEASURED
    assert by_id["V46"].blocking
    ports = {result.case_id: result for result in run.port_fidelity}
    assert ports["bbox_1x1"].scored_pixel_identity >= 0.998
    assert ports["bbox_2x1"].scored_pixel_identity == 1.0
    assert run.rayleigh_startup is not None
    assert run.rayleigh_startup.passed
    assert tuple(rayleigh_calls) == gates.RAYLEIGH_STARTUP_SENSORS
    assert run.enforcement.passed, run.enforcement.issues


def test_port_fidelity_enforcement_cannot_be_reblessed_by_probe_metadata() -> None:
    canonical = tuple(
        gates.production_adapter.port_fidelity_probe(case_id)
        for case_id in gates.production_adapter.PORT_FIDELITY_CASES
    )
    weakened = dataclasses.replace(
        canonical[0],
        scored_pixel_identity=0.997,
        minimum_scored_identity=0.0,
    )
    issues = gates.port_fidelity_contract_issues((weakened, canonical[1]))
    assert any("canonical minimum is 0.998" in issue for issue in issues)
    assert any("below 0.998000000" in issue for issue in issues)

    missing = gates.port_fidelity_contract_issues(canonical[:1])
    assert "missing canonical port-fidelity case bbox_2x1" in missing


def test_corpus_browser_and_live_results_are_not_presented_as_current_checks(research) -> None:
    run = gates.run_baseline(research, repeatability_hashes=THREE_IDENTICAL_HASHES)
    actual = {"V02", "V04", "V07", "V08", "V10", "V12", "V14", "V18", "V19", "V25", "V29", "V31", "V36"}
    for gate in run.gates:
        if gate.gate_id not in actual:
            assert gate.evidence_provenance in {"canonical_research_baseline", "unmeasured"}
    assert next(gate for gate in run.gates if gate.gate_id == "V29").evidence_provenance == (
        "current_head_proxy"
    )


def test_report_json_is_stable_and_carries_source_dependency_and_artifact_hashes(
    research,
) -> None:
    run = gates.run_baseline(research, repeatability_hashes=THREE_IDENTICAL_HASHES)
    first = build_report(run, enforce_through=None)
    second = build_report(run, enforce_through=None)
    assert stable_json_bytes(first) == stable_json_bytes(second)
    assert first["schema"] == "tat_truecolor_validation_report/v1"
    assert first["summary"]["gate_count"] == 46
    assert len(first["m0_contracts"]["port_fidelity"]) == 2
    assert first["source"]["git_head"]
    assert "matplotlib" in first["dependencies"]["distributions"]
    assert first["dependencies"]["distributions"]["tat-palettes"]
    assert first["dependencies"]["module_origins"]["tat_palettes"]
    assert first["artifact_hashes"]["validation_gates_csv"] == gates.EXPECTED_VALIDATION_GATES_SHA256
    assert len(first["artifact_hashes"]["evidence_payload_sha256"]) == 64


def test_report_write_is_atomic_checksummed_and_directory_synced(
    research, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = gates.run_baseline(research, repeatability_hashes=THREE_IDENTICAL_HASHES)
    report = build_report(run, enforce_through=None)
    original_fsync = os.fsync
    synced_modes: list[int] = []

    def tracked_fsync(descriptor: int) -> None:
        synced_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    written = write_report(report, tmp_path / "review")
    payload = written.report_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == written.report_sha256
    assert written.checksum_path.read_text().startswith(written.report_sha256)
    assert stat.S_IMODE(written.report_path.stat().st_mode) == 0o644
    assert stat.S_IMODE(written.checksum_path.stat().st_mode) == 0o644
    assert json.loads(payload)["summary"]["gate_count"] == 46
    assert sum(stat.S_ISREG(mode) for mode in synced_modes) == 2
    assert sum(stat.S_ISDIR(mode) for mode in synced_modes) == 2
    with pytest.raises(FileExistsError):
        write_report(report, tmp_path / "review")


def test_artifact_path_inside_any_git_worktree_is_refused() -> None:
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(UnsafeArtifactPath, match="Git worktree"):
        validate_artifact_directory(repo / "review-artifacts" / "m0")


@pytest.mark.parametrize(
    "filename",
    ("../escaped.json", "nested/report.json", "/tmp/absolute.json", "..\\escaped.json"),
)
def test_report_filename_cannot_escape_the_validated_artifact_root(
    tmp_path: Path, filename: str
) -> None:
    with pytest.raises(UnsafeArtifactPath, match="plain filename"):
        write_report({"safe": True}, tmp_path / "review", filename=filename)


def test_m0_enforcement_blocks_without_embedded_rayleigh_startup_evidence(
    research,
) -> None:
    run = gates.run_baseline(
        research,
        enforce_through="M0",
        repeatability_hashes=THREE_IDENTICAL_HASHES,
    )
    assert not run.enforcement.passed
    assert any("eight-sensor Rayleigh" in issue for issue in run.enforcement.issues)


def test_rayleigh_startup_calls_all_eight_and_aggregates_none_and_exception(
    tmp_path: Path,
) -> None:
    data_directory, config_file = _rayleigh_environment(tmp_path)
    calls: list[tuple[str, str]] = []

    def factory(platform: str, sensor: str) -> object | None:
        calls.append((platform, sensor))
        if platform == "GOES-16":
            return None
        if platform == "Himawari-8":
            raise RuntimeError("injected LUT failure")
        return object()

    evidence = gates.run_rayleigh_startup_check(
        data_directory,
        config_file=config_file,
        make_corrector=factory,
    )
    assert tuple(calls) == gates.RAYLEIGH_STARTUP_SENSORS
    assert not evidence.passed
    statuses = {(item.platform, item.sensor): item.status for item in evidence.sensors}
    assert statuses[("GOES-16", "abi")] == "returned_none"
    assert statuses[("Himawari-8", "ahi")] == "exception"


def test_rayleigh_startup_rejects_git_worktree_and_unwritable_data_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = Path(__file__).resolve().parents[1]
    _, external_config = _rayleigh_environment(tmp_path)
    with pytest.raises(gates.RayleighEnvironmentError, match="outside every Git"):
        gates.run_rayleigh_startup_check(
            repo,
            config_file=external_config,
            make_corrector=lambda _platform, _sensor: object(),
        )

    data_directory = tmp_path / "second-data"
    data_directory.mkdir()
    config_file = tmp_path / "second-pyspectral.yaml"
    config_file.write_text(
        f"rsr_dir: {data_directory}\nrayleigh_dir: {data_directory}\n",
        encoding="utf-8",
    )

    def refuse_probe(*_args, **_kwargs):
        raise PermissionError("injected read-only directory")

    monkeypatch.setattr(gates.tempfile, "mkstemp", refuse_probe)
    with pytest.raises(gates.RayleighEnvironmentError, match="not durably writable"):
        gates.run_rayleigh_startup_check(
            data_directory,
            config_file=config_file,
            make_corrector=lambda _platform, _sensor: object(),
        )


def test_cli_requires_explicit_research_root_and_artifact_directory() -> None:
    with pytest.raises(SystemExit) as exc:
        tat_truecolor_validate.main(["baseline"])
    assert exc.value.code == 2


def test_cli_requires_rayleigh_data_directory_for_m0_enforcement(
    research, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exc:
        tat_truecolor_validate.main(
            [
                "baseline",
                "--research-root",
                str(research.root),
                "--artifact-dir",
                str(tmp_path / "evidence"),
                "--enforce-through",
                "M0",
            ]
        )
    assert exc.value.code == 2


def test_cli_baseline_and_m0_enforcement_write_external_evidence(
    research, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gates, "separate_process_repeatability_hashes", lambda: THREE_IDENTICAL_HASHES)
    data_directory, config_file = _rayleigh_environment(tmp_path)
    monkeypatch.setenv("PSP_CONFIG_FILE", str(config_file))
    rayleigh_calls: list[tuple[str, str]] = []

    def factory(platform: str, sensor: str) -> object:
        rayleigh_calls.append((platform, sensor))
        return object()

    truecolor = gates.production_adapter.production_modules()["truecolor"]
    monkeypatch.setattr(truecolor, "_make_rayleigh", factory)
    output = tmp_path / "cli-evidence"
    exit_code = tat_truecolor_validate.main(
        [
            "baseline",
            "--research-root",
            str(research.root),
            "--artifact-dir",
            str(output),
            "--enforce-through",
            "M0",
            "--rayleigh-data-dir",
            str(data_directory),
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["m0_enforcement"]["passed"]
    assert tuple(rayleigh_calls) == gates.RAYLEIGH_STARTUP_SENSORS
    assert (output / "m0-baseline.json").is_file()
    assert (output / "m0-baseline.json.sha256").is_file()
    report = json.loads((output / "m0-baseline.json").read_text(encoding="utf-8"))
    assert report["m0_contracts"]["rayleigh_startup"]["passed"]


def test_compatibility_entrypoint_delegates_to_the_same_main() -> None:
    assert tat_harness.main is tat_truecolor_validate.main
    completed = subprocess.run(
        [sys.executable, "scripts/tat_harness.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "tat_truecolor_validate" in completed.stdout
