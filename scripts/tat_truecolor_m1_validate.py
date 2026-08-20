#!/usr/bin/env python3
"""Run canonical M1 gates and write the external original-scale gallery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.truecolor.gates import (  # noqa: E402
    ResearchIntegrityError,
    run_rayleigh_startup_check,
    verify_research_root,
)
from validation.truecolor.m1 import (  # noqa: E402
    build_m1_report,
    m1_harness_source_hashes,
    run_m1,
)
from validation.truecolor.m1_gallery import (  # noqa: E402
    write_m1_gallery_images,
    write_m1_gallery_manifest,
)


@dataclass(frozen=True, slots=True)
class PreflightPaths:
    artifact_directory: Path
    research_root: Path
    corpus_root: Path
    rayleigh_data_directory: Path
    pyspectral_config: Path


def _resolve_directory(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory: {resolved}")
    return resolved


def _resolve_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be an existing file: {resolved}")
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _git_worktree_root(path: Path) -> Path | None:
    probe = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return None
    top = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(top).resolve(strict=True)


def preflight_paths(arguments: argparse.Namespace) -> PreflightPaths:
    """Resolve every path and reject collisions before any filesystem write."""

    raw_artifact_directory = arguments.artifact_dir.expanduser()
    if os.path.lexists(raw_artifact_directory):
        raise FileExistsError(
            "refusing an existing artifact leaf (including a symlink): "
            f"{raw_artifact_directory}"
        )
    artifact_directory = validate_artifact_directory(arguments.artifact_dir)
    if os.path.lexists(artifact_directory):
        raise FileExistsError(
            f"refusing to replace existing M1 artifact target: {artifact_directory}"
        )
    if not artifact_directory.parent.is_dir():
        raise FileNotFoundError(
            "M1 artifact parent must already exist; refusing to create paths "
            f"during preflight: {artifact_directory.parent}"
        )
    for volatile_root in (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm")):
        if _paths_overlap(artifact_directory, volatile_root):
            raise UnsafeArtifactPath(
                f"M1 durable artifacts may not be published under {volatile_root}"
            )

    research_root = _resolve_directory(
        arguments.research_root,
        label="research root",
    )
    corpus_root = _resolve_directory(arguments.corpus_root, label="corpus root")
    rayleigh_data_directory = _resolve_directory(
        arguments.rayleigh_data_dir,
        label="Rayleigh data directory",
    )
    pyspectral_config = _resolve_file(
        arguments.pyspectral_config,
        label="pyspectral config",
    )
    research_worktree = _git_worktree_root(research_root)
    if research_worktree is not None:
        raise UnsafeArtifactPath(
            "research package must remain outside every Git worktree; "
            f"found worktree {research_worktree}"
        )

    protected_roots = {
        "research root": research_root,
        "corpus root": corpus_root,
        "Rayleigh data root": rayleigh_data_directory,
        "pyspectral config root": pyspectral_config.parent,
    }
    for label, protected_root in protected_roots.items():
        if _paths_overlap(artifact_directory, protected_root):
            raise UnsafeArtifactPath(
                "artifact target must be disjoint in both directions from the "
                f"{label}: target={artifact_directory}, protected={protected_root}"
            )
    if not os.access(artifact_directory.parent, os.W_OK | os.X_OK):
        raise PermissionError(
            f"M1 artifact parent is not writable: {artifact_directory.parent}"
        )
    return PreflightPaths(
        artifact_directory=artifact_directory,
        research_root=research_root,
        corpus_root=corpus_root,
        rayleigh_data_directory=rayleigh_data_directory,
        pyspectral_config=pyspectral_config,
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_new_file(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite M1 integrity file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_run_checksum_index(directory: Path) -> tuple[Path, str, Path]:
    """Write a non-cyclic checksum index for every immutable payload file."""

    index_path = directory / "SHA256SUMS"
    sidecar_path = directory / "SHA256SUMS.sha256"
    excluded = {index_path.name, sidecar_path.name}
    payload_files = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.name not in excluded
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not payload_files:
        raise RuntimeError("refusing to publish an empty M1 checksum index")
    lines = []
    for path in payload_files:
        relative_name = path.relative_to(directory).as_posix()
        if "\n" in relative_name or "\r" in relative_name or "\\" in relative_name:
            raise ValueError(f"unsafe checksum-index filename: {relative_name!r}")
        lines.append(f"{_sha256_file(path)}  {relative_name}\n")
    _atomic_new_file(index_path, "".join(lines).encode("utf-8"))
    index_sha256 = _sha256_file(index_path)
    _atomic_new_file(
        sidecar_path,
        f"{index_sha256}  {index_path.name}\n".encode("ascii"),
    )

    indexed_names = {
        line.split("  ", 1)[1].rstrip("\n") for line in lines
    }
    expected_names = {
        path.relative_to(directory).as_posix() for path in payload_files
    }
    if indexed_names != expected_names:
        raise RuntimeError("M1 root checksum index did not cover every payload file")
    for path in payload_files:
        relative_name = path.relative_to(directory).as_posix()
        expected = next(
            line.split("  ", 1)[0]
            for line in lines
            if line.endswith(f"  {relative_name}\n")
        )
        if _sha256_file(path) != expected:
            raise RuntimeError(f"M1 checksum verification failed for {relative_name}")
    if _sha256_file(index_path) != index_sha256:
        raise RuntimeError("M1 checksum index sidecar verification failed")
    return index_path, index_sha256, sidecar_path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
from validation.truecolor.report import (  # noqa: E402
    UnsafeArtifactPath,
    collect_dependency_metadata,
    collect_source_metadata,
    validate_artifact_directory,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tat_truecolor_m1_validate",
        description=(
            "Canonical real-scene M1 explicit-Hanning gates and review gallery"
        ),
    )
    parser.add_argument(
        "--enforce-through",
        choices=("M1",),
        required=True,
        help="explicit staged enforcement target; only M1 is implemented here",
    )
    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
        help="verified extracted TAT_TRUECOLOR_RESEARCH directory",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="external directory containing the four pinned Melissa ABI objects",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="new external directory for report and original-scale review files",
    )
    parser.add_argument(
        "--rayleigh-data-dir",
        type=Path,
        required=True,
        help="external writable pyspectral RSR/Rayleigh data directory",
    )
    parser.add_argument(
        "--pyspectral-config",
        type=Path,
        required=True,
        help="external pyspectral YAML binding rsr_dir/rayleigh_dir to the data directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    staging_directory: Path | None = None
    try:
        paths = preflight_paths(arguments)
        artifact_directory = paths.artifact_directory
        research = verify_research_root(paths.research_root)
        try:
            source_metadata = collect_source_metadata()
            dependency_metadata = collect_dependency_metadata()
            harness_source_hashes = m1_harness_source_hashes()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "source/dependency/harness metadata preflight failed before "
                "artifact publication; mount the linked worktree's Git metadata "
                "at its recorded path"
            ) from exc

        os.environ["PSP_CONFIG_FILE"] = str(paths.pyspectral_config)
        rayleigh = run_rayleigh_startup_check(
            paths.rayleigh_data_directory,
            config_file=paths.pyspectral_config,
        )
        run = run_m1(
            research,
            corpus_root=paths.corpus_root,
            rayleigh_startup=rayleigh,
        )
        post_run_source_metadata = collect_source_metadata()
        if post_run_source_metadata != source_metadata:
            raise RuntimeError(
                "production source metadata changed during the M1 measurement run"
            )
        post_run_harness_hashes = m1_harness_source_hashes()
        if post_run_harness_hashes != harness_source_hashes:
            raise RuntimeError(
                "M1 validation harness sources changed during the measurement run"
            )
        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{artifact_directory.name}.staging-",
                dir=artifact_directory.parent,
            )
        )
        os.chmod(staging_directory, 0o700)
        gallery = write_m1_gallery_images(run, staging_directory)
        report = build_m1_report(
            run,
            gallery_artifacts=gallery,
            source_metadata=source_metadata,
            dependency_metadata=dependency_metadata,
            harness_source_hashes=harness_source_hashes,
        )
        written = write_report(
            report,
            staging_directory,
            filename="m1-validation.json",
        )
        gallery_manifest = write_m1_gallery_manifest(
            run,
            staging_directory,
            gallery,
            report_filename=written.report_path.name,
            report_sha256=written.report_sha256,
        )
        if collect_source_metadata() != source_metadata:
            raise RuntimeError(
                "production source metadata changed during M1 artifact generation"
            )
        if m1_harness_source_hashes() != harness_source_hashes:
            raise RuntimeError(
                "M1 validation harness sources changed during artifact generation"
            )
        checksum_index, checksum_index_sha256, checksum_sidecar = (
            write_run_checksum_index(staging_directory)
        )
        if collect_source_metadata() != source_metadata:
            raise RuntimeError(
                "production source metadata changed before atomic M1 publication"
            )
        if m1_harness_source_hashes() != harness_source_hashes:
            raise RuntimeError(
                "M1 validation harness sources changed before atomic publication"
            )
        if os.path.lexists(artifact_directory):
            raise FileExistsError(
                f"M1 artifact target appeared during staging: {artifact_directory}"
            )
        os.chmod(staging_directory, 0o755)
        _fsync_directory(staging_directory)
        os.rename(staging_directory, artifact_directory)
        staging_directory = None
        _fsync_directory(artifact_directory.parent)
    except (
        FileExistsError,
        OSError,
        ResearchIntegrityError,
        RuntimeError,
        subprocess.CalledProcessError,
        UnsafeArtifactPath,
        ValueError,
    ) as exc:
        print(f"tat_truecolor_m1_validate: {exc}", file=sys.stderr)
        return 2
    finally:
        if staging_directory is not None:
            shutil.rmtree(staging_directory, ignore_errors=True)

    assert run.canonical is not None
    gate_outcomes = {gate.gate_id: gate.outcome for gate in run.gates}
    summary = {
        "report": str(artifact_directory / written.report_path.name),
        "report_sha256": written.report_sha256,
        "gallery_manifest": str(artifact_directory / gallery_manifest.filename),
        "gallery_manifest_sha256": gallery_manifest.sha256,
        "sha256sums": str(artifact_directory / checksum_index.name),
        "sha256sums_sha256": checksum_index_sha256,
        "sha256sums_sidecar": str(artifact_directory / checksum_sidecar.name),
        "stage_advancement_passed": run.stage_advancement_passed,
        "all_thresholds_passed": run.all_thresholds_passed,
        "threshold_fail_gate_ids": list(run.threshold_fail_gate_ids),
        "unmeasured_gate_ids": list(run.unmeasured_gate_ids),
        "gate_outcomes": gate_outcomes,
        "blocking_gate_count": sum(gate.blocking for gate in run.gates),
        "production_nearest_rms_dn": run.canonical.production_nearest_rms_dn,
        "explicit_hanning_rms_dn": run.canonical.explicit_hanning_rms_dn,
        "production_nearest_high_frequency_ratio": (
            run.canonical.production_nearest_high_frequency_ratio
        ),
        "explicit_hanning_high_frequency_ratio": (
            run.canonical.explicit_hanning_high_frequency_ratio
        ),
        "reference_convention": run.canonical.reference_convention,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if run.stage_advancement_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
