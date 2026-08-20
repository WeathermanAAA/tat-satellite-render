"""Atomic, external-only publication contracts for the M1 evidence CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tat_truecolor_m1_validate as cli
from validation.truecolor import production_adapter
from validation.truecolor.report import UnsafeArtifactPath


@pytest.fixture
def external_tree():
    base = Path(
        tempfile.mkdtemp(
            prefix=".m1-cli-contract-",
            dir=production_adapter.REPO_ROOT.resolve().parent,
        )
    )
    try:
        for name in ("research", "corpus", "rayleigh", "config", "artifacts"):
            (base / name).mkdir()
        config = base / "config" / "pyspectral.yaml"
        config.write_text("rsr_dir: fixture\n", encoding="utf-8")
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _argv(base: Path, target: Path) -> list[str]:
    return [
        "--enforce-through",
        "M1",
        "--research-root",
        str(base / "research"),
        "--corpus-root",
        str(base / "corpus"),
        "--artifact-dir",
        str(target),
        "--rayleigh-data-dir",
        str(base / "rayleigh"),
        "--pyspectral-config",
        str(base / "config" / "pyspectral.yaml"),
    ]


def _fake_run() -> SimpleNamespace:
    gates = tuple(
        SimpleNamespace(gate_id=gate_id, outcome="PASS", blocking=False)
        for gate_id in ("V01", "V02", "V03", "V04", "V15", "V16", "V18", "V25")
    )
    canonical = SimpleNamespace(
        production_nearest_rms_dn=6.314,
        explicit_hanning_rms_dn=0.395,
        production_nearest_high_frequency_ratio=2.258,
        explicit_hanning_high_frequency_ratio=1.006,
        reference_convention="captured internal grid fixture",
    )
    return SimpleNamespace(
        canonical=canonical,
        gates=gates,
        stage_advancement_passed=True,
        all_thresholds_passed=True,
        threshold_fail_gate_ids=(),
        unmeasured_gate_ids=(),
    )


def _install_fast_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gallery_failure: bool = False,
    source_drift_at_call: int | None = None,
) -> None:
    monkeypatch.setattr(cli, "verify_research_root", lambda path: {"root": str(path)})
    source_calls = 0

    def source_metadata():
        nonlocal source_calls
        source_calls += 1
        revision = (
            "changed"
            if source_drift_at_call is not None
            and source_calls >= source_drift_at_call
            else "stable"
        )
        return {"git_head": revision, "source_sha256": {"render.py": "1" * 64}}

    monkeypatch.setattr(cli, "collect_source_metadata", source_metadata)
    monkeypatch.setattr(cli, "collect_dependency_metadata", lambda: {"python": "fixture"})
    harness_hashes = {"validation/truecolor/m1.py": "2" * 64}
    monkeypatch.setattr(cli, "m1_harness_source_hashes", lambda: harness_hashes)
    monkeypatch.setattr(cli, "run_rayleigh_startup_check", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "run_m1", lambda *args, **kwargs: _fake_run())

    def gallery(run, directory):
        payload_path = Path(directory) / "01_fixture.bin"
        payload_path.write_bytes(b"immutable gallery pixels")
        if gallery_failure:
            raise RuntimeError("injected gallery failure")
        return {
            payload_path.name: {
                "filename": payload_path.name,
                "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            }
        }

    monkeypatch.setattr(cli, "write_m1_gallery_images", gallery)

    def report(run, **kwargs):
        assert kwargs["source_metadata"]["git_head"] == "stable"
        assert kwargs["dependency_metadata"] == {"python": "fixture"}
        assert kwargs["harness_source_hashes"] == harness_hashes
        return {
            "schema": "fixture/v2",
            "source": kwargs["source_metadata"],
            "dependencies": kwargs["dependency_metadata"],
            "artifact_hashes": {
                "validation_harness_source_sha256": kwargs[
                    "harness_source_hashes"
                ]
            },
        }

    monkeypatch.setattr(cli, "build_m1_report", report)

    def manifest(run, directory, artifacts, *, report_filename, report_sha256):
        payload = json.dumps(
            {"report": {"filename": report_filename, "sha256": report_sha256}},
            sort_keys=True,
        ).encode("utf-8")
        path = Path(directory) / "gallery_manifest.json"
        path.write_bytes(payload)
        return SimpleNamespace(
            filename=path.name,
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    monkeypatch.setattr(cli, "write_m1_gallery_manifest", manifest)


def test_atomic_success_binds_report_manifest_and_complete_checksum_index(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fast_fakes(monkeypatch)
    target = external_tree / "artifacts" / "successful-run"
    assert cli.main(_argv(external_tree, target)) == 0
    assert target.is_dir()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))

    summary = json.loads(capsys.readouterr().out)
    assert summary["stage_advancement_passed"] is True
    assert summary["all_thresholds_passed"] is True
    assert summary["threshold_fail_gate_ids"] == []
    assert summary["unmeasured_gate_ids"] == []
    assert set(summary["gate_outcomes"].values()) == {"PASS"}
    assert Path(summary["report"]).parent == target
    assert Path(summary["gallery_manifest"]).parent == target

    report = json.loads((target / "m1-validation.json").read_text())
    manifest = json.loads((target / "gallery_manifest.json").read_text())
    report_digest = hashlib.sha256((target / "m1-validation.json").read_bytes()).hexdigest()
    assert manifest["report"]["sha256"] == report_digest
    assert report["source"]["git_head"] == "stable"
    assert report["artifact_hashes"]["validation_harness_source_sha256"]

    payload_names = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    indexed_names = {
        line.split("  ", 1)[1]
        for line in (target / "SHA256SUMS").read_text().splitlines()
    }
    assert indexed_names == payload_names
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=target, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS.sha256"], cwd=target, check=True)


def test_injected_gallery_failure_leaves_no_target_or_private_staging(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fast_fakes(monkeypatch, gallery_failure=True)
    target = external_tree / "artifacts" / "failed-run"
    assert cli.main(_argv(external_tree, target)) == 2
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))


def test_midrun_source_drift_fails_before_staging_or_publication(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fast_fakes(monkeypatch, source_drift_at_call=2)
    target = external_tree / "artifacts" / "source-drift"
    assert cli.main(_argv(external_tree, target)) == 2
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))


def test_source_drift_immediately_before_rename_cleans_complete_staging(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fast_fakes(monkeypatch, source_drift_at_call=4)
    target = external_tree / "artifacts" / "late-source-drift"
    assert cli.main(_argv(external_tree, target)) == 2
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))


def test_existing_target_is_preserved_and_rejected_before_validation(
    external_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = external_tree / "artifacts" / "existing"
    target.mkdir()
    marker = target / "owner-data"
    marker.write_text("preserve", encoding="utf-8")
    called = False

    def should_not_validate(path):
        nonlocal called
        called = True
        raise AssertionError("validation should not begin")

    monkeypatch.setattr(cli, "verify_research_root", should_not_validate)
    assert cli.main(_argv(external_tree, target)) == 2
    assert not called
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_preflight_rejects_nested_and_symlink_resolved_input_collisions(
    external_tree: Path,
) -> None:
    nested = external_tree / "research" / "artifact-run"
    arguments = cli.build_parser().parse_args(_argv(external_tree, nested))
    with pytest.raises(UnsafeArtifactPath, match="disjoint in both directions"):
        cli.preflight_paths(arguments)
    assert not nested.exists()

    link = external_tree / "artifacts" / "research-link"
    link.symlink_to(external_tree / "research", target_is_directory=True)
    linked_target = link / "artifact-run"
    arguments = cli.build_parser().parse_args(_argv(external_tree, linked_target))
    with pytest.raises(UnsafeArtifactPath, match="disjoint in both directions"):
        cli.preflight_paths(arguments)
    assert not linked_target.exists()


def test_preflight_rejects_volatile_target_and_research_inside_git(
    external_tree: Path,
) -> None:
    volatile = Path("/tmp") / f"m1-artifact-{external_tree.name}"
    arguments = cli.build_parser().parse_args(_argv(external_tree, volatile))
    with pytest.raises(UnsafeArtifactPath, match="durable artifacts"):
        cli.preflight_paths(arguments)

    arguments = argparse.Namespace(
        artifact_dir=external_tree / "artifacts" / "git-research",
        research_root=production_adapter.REPO_ROOT,
        corpus_root=external_tree / "corpus",
        rayleigh_data_dir=external_tree / "rayleigh",
        pyspectral_config=external_tree / "config" / "pyspectral.yaml",
    )
    with pytest.raises(UnsafeArtifactPath, match="outside every Git worktree"):
        cli.preflight_paths(arguments)
