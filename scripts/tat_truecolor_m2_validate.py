#!/usr/bin/env python3
"""Run the M2 contract oracle or close M2 with bound production evidence.

The production path deliberately keeps every derived review payload in memory
until the complete report, identities, and checksums can be staged as one
no-replace transaction.  A contract-only run remains an explicitly non-closing
``scratch-*`` artifact.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.truecolor import m2 as m2_validation  # noqa: E402
from validation.truecolor import m2_gallery as m2_gallery_module  # noqa: E402
from validation.truecolor import m2_production as m2_production_module  # noqa: E402
from validation.truecolor import production_adapter  # noqa: E402
from validation.truecolor.gates import (  # noqa: E402
    GATES,
    RAYLEIGH_DATA_INVENTORY_SCHEMA,
    RAYLEIGH_STARTUP_SENSORS,
    ResearchIntegrityError,
    run_rayleigh_startup_check,
    verify_research_root,
)
from validation.truecolor.m2 import (  # noqa: E402
    FORBIDDEN_M2_CLOSURE_GATE_IDS,
    M2_ALL_GATE_IDS,
    M2_CANONICAL_MUTATION_IDS,
    M5_DEFERRED_GATE_IDS,
    M5_SURVIVING_ENCODED_RESIZES,
    M2_ORACLE_SCHEMA,
    M2_REPORT_SCHEMA,
    M2_REQUIRED_GATE_IDS,
    build_m2_report,
    m2_harness_source_hashes,
    run_m2_contract_oracle,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
)
from validation.truecolor.m2_gallery import (  # noqa: E402
    GALLERY_SCHEMA,
    M2GalleryBundle,
    build_m2_gallery_payloads,
)
from validation.truecolor.m2_production import (  # noqa: E402
    M1_TRUSTED_CHECKSUM_INDEX_SHA256,
    M1_TRUSTED_GALLERY_MANIFEST_SHA256,
    M1_TRUSTED_GIT_HEAD,
    M1_TRUSTED_REPORT_SHA256,
    M2_PRODUCTION_SCHEMA,
    M2_REQUIRED_POLICY_VERSION,
    CanonicalM2Evidence,
    M2ProductionEvaluation,
    RepeatabilityCapture,
    SequenceEvaluation,
    TimestampedRendererFrame,
    TimestampedSequenceBinding,
    capture_three_process_repeatability,
    evaluate_canonical_m2,
    evaluate_timestamped_sequence,
    load_canonical_melissa_physical,
    measure_canonical_m2,
    verify_m1_artifact_baseline,
)
from validation.truecolor.m2_sequence import (  # noqa: E402
    V30_BINDING_SCHEMA,
    V30_CANONICAL_ACQUISITION_INDEX_SHA256,
    V30_CANONICAL_MANIFEST_SHA256,
    V30_CAPTURE_SCHEMA,
    V30_NATURAL_EARTH_ZIP_SHA256,
    V30_PATCH_EDGE_BINDING_TOLERANCE_PX,
    V30_PATCH_WORLD_BOUNDS,
    V30_ROLE_ORDER,
    V30Corpus,
    V30Object,
    V30SequenceCapture,
    capture_v30_sequence,
    verify_v30_corpus,
)
from validation.truecolor.report import (  # noqa: E402
    UnsafeArtifactPath,
    collect_dependency_metadata,
    collect_source_metadata,
    validate_artifact_directory,
)


MANIFEST_SCHEMA = "tat_truecolor_m2_artifact_manifest/v2"
M2_PRODUCT_SOURCE_FILES = (
    "app.py",
    "cache.py",
    "colormaps.py",
    "render.py",
    "render_policy.py",
    "render_resample.py",
    "satellites.py",
    "truecolor.py",
    "gridsat.py",
    "gridsat_goes.py",
    "mergir.py",
    "s2_pyramid.py",
    "s2_webmerc.py",
    "vendor/ahi_hsd.py",
    "vendor/ahi_loader.py",
    "requirements.txt",
    "runtime.txt",
)
M2_LOCAL_RENDER_DATA_FILES = (
    "cyclolab_ne_10m_states.geojson",
)


@dataclass(frozen=True, slots=True)
class PreflightPaths:
    artifact_directory: Path
    research_root: Path
    corpus_root: Path | None = None
    m1_artifact_directory: Path | None = None
    rayleigh_data_directory: Path | None = None
    pyspectral_config: Path | None = None
    v30_acquisition_directory: Path | None = None
    v30_raw_directory: Path | None = None
    v30_land_zip: Path | None = None

    @property
    def production(self) -> bool:
        values = (
            self.corpus_root,
            self.m1_artifact_directory,
            self.rayleigh_data_directory,
            self.pyspectral_config,
            self.v30_acquisition_directory,
            self.v30_raw_directory,
            self.v30_land_zip,
        )
        if any(value is None for value in values):
            if any(value is not None for value in values):
                raise RuntimeError("partial M2 production path set escaped preflight")
            return False
        return True


@dataclass(frozen=True, slots=True)
class PublishedArtifacts:
    directory: Path
    report_sha256: str
    manifest_sha256: str
    checksum_index_sha256: str


M2_AUTHORITY_AND_FALLBACK_LIMITATIONS = (
    "M2 primary imagery-layer authority is limited to ABI/AHI noncrossing, bounded, normal-DPI truecolor requests.",
    "M2 scalar authority is limited to noncrossing regular GridSat-B1 and GridSat-GOES archive scalar requests.",
    "MergIR remains on registered M1 because its legacy fetch may select an alternate slot after resolution; M2 does not claim its cache/source provenance.",
    "High-DPI truecolor remains on the registered M1 fallback and is not M2 evidence.",
    "Antimeridian/crossing truecolor is rejected by the M2 envelope; any separately registered crossing behavior remains outside M2 authority.",
    "Backdrop rendering remains on the registered M1 fallback.",
    "Non-archive or native-grid scalar rendering remains on the registered M1 fallback.",
    "btpng rendering remains on the registered M1 fallback.",
    "Unsupported bounding boxes or requests outside the M2 preflight envelope remain rejected or on explicitly registered M1 behavior.",
)
M2_EVIDENCE_DEPTH_LIMITATIONS = (
    "Real canonical imagery, V41, and V30 evidence use ABI GOES-19; AHI authority is supported by typed product/mutation tests and Rayleigh preflight, not a bound real-AHI corpus render.",
    "GridSat-B1/GridSat-GOES scalar authority is supported by contract and integration fixtures, not a canonical external scalar object/render.",
)
M2_GALLERY_REQUIRED_LIMITATIONS = (
    *M2_AUTHORITY_AND_FALLBACK_LIMITATIONS[:2],
    *M2_EVIDENCE_DEPTH_LIMITATIONS,
    *M2_AUTHORITY_AND_FALLBACK_LIMITATIONS[2:],
    "The M2 physical >1600 source cap is a separate upstream Hanning operation before Rayleigh and remains an M3 ordering residual.",
    "Accepted ratio self-sharpen remains an unchanged neighborhood spatial operator before CIRA and is parked to M4.",
    "S03 bilinear inter-band co-registration remains until M4.",
    "Encoded loop, S2 pyramid, Web-Mercator, and browser resizing remain until M5.",
    "V44, V45, and V46 are unmeasured and unclosed.",
)
PRODUCTION_LIMITATIONS = (
    *M2_AUTHORITY_AND_FALLBACK_LIMITATIONS[:2],
    *M2_EVIDENCE_DEPTH_LIMITATIONS,
    *M2_AUTHORITY_AND_FALLBACK_LIMITATIONS[2:],
    "The M2 physical >1600 source cap is a separate upstream Hanning operation before Rayleigh and remains an M3 ordering residual.",
    "Accepted ratio self-sharpen remains an unchanged neighborhood spatial operator before CIRA and is parked to M4.",
    "S03 bilinear inter-band co-registration remains until M4.",
    "Encoded loop, S2 pyramid, Web-Mercator, and browser resizing remain until M5.",
    "runtime.txt declares Python 3.11.10 while the accepted immutable validation "
    "image runs CPython 3.11.16; M2 enforces the 3.11 series and records both "
    "exact versions without silently repinning deployment.",
    "V44, V45, and V46 are unmeasured and unclosed.",
)
PRODUCTION_ARGUMENTS = (
    "corpus_root",
    "m1_artifact_dir",
    "rayleigh_data_dir",
    "pyspectral_config",
    "v30_acquisition_dir",
    "v30_raw_dir",
    "v30_land_zip",
)
V30_PATCH_COUNT = 19
V30_PATCH_SHAPE = (96, 96, 3)
_EXACT_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==([^\s;#]+)$"
)
TAT_PALETTES_VERSION = "0.2.2"
TAT_PALETTES_DIRECT_URL = {
    "url": "https://github.com/WeathermanAAA/Triple-A-Tropics.git",
    "subdirectory": "palette",
    "vcs_info": {
        "vcs": "git",
        "requested_revision": "tat-palettes-v0.2.2",
        "commit_id": "569450d82ee2b7b8a54c11df9976b32f7a8f10af",
    },
}
M2_RUNNER_IMAGE_ID = (
    "sha256:0abedbfafe94be25cca79ade77f06fea31e555ba176c5ab097c6cbb1b95e470d"
)
M2_RUNNER_IMAGE_PROVENANCE = {
    "image_id": M2_RUNNER_IMAGE_ID,
    "declaration_source": "external_operator_declared",
    "container_self_attested": False,
    "limitation": (
        "The validation container cannot self-attest its Docker image ID; this "
        "immutable ID is an external operator declaration bound into the report."
    ),
}


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _resolved_git_pointer() -> Mapping[str, Any]:
    dot_git = REPO_ROOT / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        prefix = "gitdir:"
        if not text.lower().startswith(prefix):
            raise RuntimeError("linked-worktree .git file has an invalid format")
        raw = text[len(prefix) :].strip()
        target = Path(raw)
        if not target.is_absolute():
            target = dot_git.parent / target
        resolved = target.resolve(strict=True)
        if not resolved.is_dir():
            raise RuntimeError("linked-worktree gitdir is not a directory")
        return {
            "dot_git_kind": "linked_worktree_pointer",
            "dot_git_path": str(dot_git),
            "pointer_text": text,
            "resolved_git_directory": str(resolved),
        }
    if dot_git.is_dir():
        return {
            "dot_git_kind": "directory",
            "dot_git_path": str(dot_git.resolve(strict=True)),
            "pointer_text": None,
            "resolved_git_directory": str(dot_git.resolve(strict=True)),
        }
    raise RuntimeError(f"repository .git metadata is absent: {dot_git}")


def collect_repository_identity() -> dict[str, Any]:
    """Resolve linked-worktree metadata and fail rather than invent identity."""

    pointer = dict(_resolved_git_pointer())
    top = Path(_git("rev-parse", "--show-toplevel").stdout.strip()).resolve(
        strict=True
    )
    if top != REPO_ROOT.resolve(strict=True):
        raise RuntimeError(f"Git top-level mismatch: expected {REPO_ROOT}, got {top}")
    absolute_git_dir = Path(
        _git("rev-parse", "--absolute-git-dir").stdout.strip()
    ).resolve(strict=True)
    common_raw = _git("rev-parse", "--git-common-dir").stdout.strip()
    common = Path(common_raw)
    if not common.is_absolute():
        common = REPO_ROOT / common
    common = common.resolve(strict=True)
    if absolute_git_dir != Path(pointer["resolved_git_directory"]):
        raise RuntimeError(
            "Git's resolved directory differs from the linked-worktree pointer"
        )

    upstream_probe = _git(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", check=False
    )
    upstream = (
        upstream_probe.stdout.strip() if upstream_probe.returncode == 0 else None
    )
    divergence: Mapping[str, int] | None = None
    upstream_head: str | None = None
    if upstream:
        upstream_head = _git("rev-parse", upstream).stdout.strip()
        counts = _git(
            "rev-list", "--left-right", "--count", f"HEAD...{upstream}"
        ).stdout.split()
        if len(counts) != 2:
            raise RuntimeError("Git returned an invalid upstream divergence")
        divergence = {"ahead": int(counts[0]), "behind": int(counts[1])}

    return {
        "repository_root": str(top),
        "branch": _git("branch", "--show-current").stdout.strip(),
        "head": _git("rev-parse", "HEAD").stdout.strip(),
        "head_tree": _git("rev-parse", "HEAD^{tree}").stdout.strip(),
        "upstream": upstream,
        "upstream_head": upstream_head,
        "divergence": divergence,
        "status_porcelain_v1": _git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout.splitlines(),
        "git_directory": str(absolute_git_dir),
        "git_common_directory": str(common),
        "git_pointer": pointer,
    }


def collect_m2_source_metadata() -> dict[str, Any]:
    """Hash every M2 authority, publisher dependency, and named residual."""

    metadata = dict(collect_source_metadata(REPO_ROOT))
    # The shared M1 metadata helper strips the whole Git stdout string, which
    # removes the leading porcelain X-column space from the first unstaged-only
    # row.  M2 binds the complete dirty source snapshot, so preserve Git's two
    # status columns byte-for-byte before comparing it with the repository
    # identity collected by this harness.
    metadata["git_status_porcelain"] = _git(
        "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout.splitlines()
    hashes = dict(metadata.get("source_sha256", {}))
    missing: list[str] = []
    for name in M2_PRODUCT_SOURCE_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            hashes[name] = sha256_file(path)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"M2 product source files are absent: {missing}")
    metadata["source_sha256"] = dict(sorted(hashes.items()))
    metadata["m2_bound_product_and_residual_source_files"] = list(
        M2_PRODUCT_SOURCE_FILES
    )
    local_data_hashes: dict[str, str] = {}
    for name in M2_LOCAL_RENDER_DATA_FILES:
        path = REPO_ROOT / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"M2 local render data file is absent: {name}")
        local_data_hashes[name] = sha256_file(path)
    metadata["local_render_data_sha256"] = dict(sorted(local_data_hashes.items()))
    metadata["m2_bound_local_render_data_files"] = list(
        M2_LOCAL_RENDER_DATA_FILES
    )
    return metadata


def _tat_palettes_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the exact installed tat-palettes supply-chain identity."""

    distributions = metadata.get("distributions")
    direct_urls = metadata.get("direct_urls")
    module_origins = metadata.get("module_origins")
    observed_version = (
        distributions.get("tat-palettes")
        if type(distributions) is dict
        else None
    )
    observed_direct_url = (
        direct_urls.get("tat-palettes") if type(direct_urls) is dict else None
    )
    observed_origin = (
        module_origins.get("tat_palettes")
        if type(module_origins) is dict
        else None
    )
    origin_issue: str | None = None
    resolved_origin: str | None = None
    if type(observed_origin) is not str or not observed_origin:
        origin_issue = "tat_palettes module origin is absent"
    else:
        raw_origin = Path(observed_origin)
        try:
            resolved = raw_origin.resolve(strict=True)
        except (OSError, RuntimeError):
            origin_issue = "tat_palettes module origin does not resolve to a file"
        else:
            resolved_origin = str(resolved)
            if (
                not raw_origin.is_absolute()
                or not resolved.is_file()
                or resolved.name != "__init__.py"
                or resolved.parent.name != "tat_palettes"
                or not any(
                    part in {"site-packages", "dist-packages"}
                    for part in resolved.parts
                )
            ):
                origin_issue = (
                    "tat_palettes module origin is not a plausible installed "
                    "site-packages package"
                )

    issues: list[str] = []
    if observed_version != TAT_PALETTES_VERSION:
        issues.append(
            "tat-palettes version mismatch: "
            f"installed={observed_version!r}, required={TAT_PALETTES_VERSION!r}"
        )
    if observed_direct_url != TAT_PALETTES_DIRECT_URL:
        issues.append(
            "tat-palettes PEP-610 VCS identity differs from the exact sanitized "
            "tag, commit, repository URL, and palette subdirectory binding"
        )
    if origin_issue is not None:
        issues.append(origin_issue)
    return {
        "expected_version": TAT_PALETTES_VERSION,
        "expected_direct_url": _json_value(TAT_PALETTES_DIRECT_URL),
        "observed_version": observed_version,
        "observed_direct_url": _json_value(observed_direct_url),
        "module_origin": observed_origin,
        "resolved_module_origin": resolved_origin,
        "issues": issues,
        "passed": not issues,
    }


def _declared_exact_requirement_pins() -> dict[str, str]:
    exact_pins: dict[str, str] = {}
    for raw_line in (REPO_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_REQUIREMENT.fullmatch(line)
        if match is not None:
            exact_pins[match.group(1)] = match.group(2)
    if not exact_pins:
        raise RuntimeError("requirements.txt exposes no exact production pins")
    return dict(sorted(exact_pins.items()))


def collect_m2_dependency_metadata(
    base_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind exact package pins and the production CPython runtime series."""

    metadata = _json_value(
        collect_dependency_metadata() if base_metadata is None else base_metadata
    )
    requirements_path = REPO_ROOT / "requirements.txt"
    runtime_path = REPO_ROOT / "runtime.txt"
    exact_pins = _declared_exact_requirement_pins()
    installed: dict[str, str | None] = {}
    for distribution in sorted(exact_pins, key=str.lower):
        try:
            installed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed[distribution] = None
    runtime_declaration = runtime_path.read_text(encoding="ascii").strip()
    runtime_prefix = "python-"
    if not runtime_declaration.startswith(runtime_prefix):
        raise RuntimeError("runtime.txt is not an exact python-* declaration")
    expected_python = runtime_declaration[len(runtime_prefix) :]
    version_pattern = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)$")
    expected_match = version_pattern.fullmatch(expected_python)
    actual_python = platform.python_version()
    actual_match = version_pattern.fullmatch(actual_python)
    if expected_match is None or actual_match is None:
        raise RuntimeError("Python runtime identity is not an exact X.Y.Z version")
    expected_series = ".".join(expected_match.groups()[:2])
    actual_series = ".".join(actual_match.groups()[:2])
    contract = {
        "requirements_sha256": sha256_file(requirements_path),
        "runtime_sha256": sha256_file(runtime_path),
        "runtime_declaration": runtime_declaration,
        "expected_python": expected_python,
        "actual_python": actual_python,
        "expected_python_series": expected_series,
        "actual_python_series": actual_series,
        "implementation": platform.python_implementation(),
        "runtime_declaration_matches_actual": (
            actual_python == expected_python
        ),
        "runtime_series_matches_actual": actual_series == expected_series,
        "exact_requirement_pins": dict(sorted(exact_pins.items())),
        "installed_exact_pin_versions": dict(sorted(installed.items())),
    }
    tat_palettes_identity = _tat_palettes_identity(metadata)
    contract["tat_palettes_identity"] = tat_palettes_identity
    contract["issues"] = [
        *(
            []
            if platform.python_implementation() == "CPython"
            else [
                "production interpreter is not CPython: "
                f"{platform.python_implementation()} {platform.python_version()}"
            ]
        ),
        *(
            []
            if actual_series == expected_series
            else [
                "production Python runtime series mismatch: "
                f"actual={actual_python!r}, declared={expected_python!r}"
            ]
        ),
        *(
            f"{name} installed={installed[name]!r}, required={version!r}"
            for name, version in sorted(exact_pins.items())
            if installed[name] != version
        ),
        *tat_palettes_identity["issues"],
    ]
    contract["passed"] = not contract["issues"]
    metadata["production_dependency_contract"] = contract
    return metadata


def require_production_dependency_identity(metadata: Mapping[str, Any]) -> None:
    contract = metadata.get("production_dependency_contract")
    recomputed_tat_palettes = _tat_palettes_identity(metadata)
    recorded_tat_palettes = (
        contract.get("tat_palettes_identity") if type(contract) is dict else None
    )
    requirements_path = REPO_ROOT / "requirements.txt"
    runtime_path = REPO_ROOT / "runtime.txt"
    runtime_declaration = runtime_path.read_text(encoding="ascii").strip()
    expected_python = runtime_declaration.removeprefix("python-")
    expected_series = ".".join(expected_python.split(".")[:2])
    exact_pins = _declared_exact_requirement_pins()
    distributions = metadata.get("distributions")
    internal_issues: list[str] = []
    if type(contract) is dict:
        if contract.get("issues") != []:
            internal_issues.append("dependency contract issues are not empty")
        if contract.get("requirements_sha256") != sha256_file(requirements_path):
            internal_issues.append("requirements.txt identity differs")
        if contract.get("runtime_sha256") != sha256_file(runtime_path):
            internal_issues.append("runtime.txt identity differs")
        if (
            contract.get("runtime_declaration") != runtime_declaration
            or contract.get("expected_python") != expected_python
            or contract.get("expected_python_series") != expected_series
        ):
            internal_issues.append("declared Python runtime identity differs")
        if (
            contract.get("implementation") != "CPython"
            or metadata.get("implementation") != "CPython"
            or contract.get("runtime_series_matches_actual") is not True
            or contract.get("actual_python_series") != expected_series
            or contract.get("actual_python") != metadata.get("python")
        ):
            internal_issues.append("actual CPython runtime-series binding differs")
        if contract.get("exact_requirement_pins") != exact_pins:
            internal_issues.append("exact requirements pin map differs")
        if contract.get("installed_exact_pin_versions") != exact_pins:
            internal_issues.append("installed exact-pin versions differ")
        if type(distributions) is not dict or any(
            distributions.get(name) != version
            for name, version in exact_pins.items()
        ):
            internal_issues.append("distribution versions differ from exact pins")
    else:
        internal_issues.append("production dependency contract is absent")
    if (
        type(contract) is not dict
        or contract.get("passed") is not True
        or recomputed_tat_palettes["passed"] is not True
        or recorded_tat_palettes != recomputed_tat_palettes
        or internal_issues
    ):
        issues = None if type(contract) is not dict else contract.get("issues")
        if recomputed_tat_palettes["issues"]:
            issues = list(issues or ()) + list(recomputed_tat_palettes["issues"])
        if (
            type(contract) is dict
            and recorded_tat_palettes != recomputed_tat_palettes
        ):
            issues = list(issues or ()) + [
                "recorded tat-palettes identity differs from recomputed metadata"
            ]
        issues = list(issues or ()) + internal_issues
        raise RuntimeError(
            "canonical M2 publication requires exact requirements pins, the "
            "declared CPython runtime series, and the pinned tat-palettes VCS "
            f"identity; issues={issues}"
        )


def _require_publication_dependency_identity(metadata: Mapping[str, Any]) -> None:
    require_production_dependency_identity(metadata)
    current = collect_m2_dependency_metadata()
    require_production_dependency_identity(current)
    if _json_value(metadata) != current:
        raise RuntimeError(
            "canonical M2 dependency identity differs from the live environment"
        )


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


def _resolve_real_path(path: Path, *, label: str, directory: bool) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise UnsafeArtifactPath(f"{label} may not be a symlink: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved.is_symlink():
        raise UnsafeArtifactPath(f"{label} may not resolve to a symlink: {resolved}")
    expected = resolved.is_dir() if directory else resolved.is_file()
    if not expected:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{label} must be an existing {kind}: {resolved}")
    return resolved


def _json_value(value: Any) -> Any:
    """Return an ordinary, strict-JSON copy (and reject NaN/proxies)."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _require_production_paths(paths: PreflightPaths) -> tuple[Path, ...]:
    if not paths.production:
        raise RuntimeError("M2 canonical production paths are absent")
    values = (
        paths.corpus_root,
        paths.m1_artifact_directory,
        paths.rayleigh_data_directory,
        paths.pyspectral_config,
        paths.v30_acquisition_directory,
        paths.v30_raw_directory,
        paths.v30_land_zip,
    )
    if any(value is None for value in values):  # narrowed by ``production``
        raise RuntimeError("M2 production path narrowing failed")
    return tuple(value for value in values if value is not None)


def preflight_paths(arguments: argparse.Namespace) -> PreflightPaths:
    """Resolve every input and reject every collision before measurement."""

    raw_target = arguments.artifact_dir.expanduser()
    if os.path.lexists(raw_target):
        raise FileExistsError(
            f"refusing an existing artifact leaf (including symlink): {raw_target}"
        )
    artifact = validate_artifact_directory(raw_target)
    if os.path.lexists(artifact):
        raise FileExistsError(f"refusing to replace M2 artifact target: {artifact}")
    if not artifact.parent.is_dir():
        raise FileNotFoundError(
            f"artifact parent must already exist during preflight: {artifact.parent}"
        )
    if not os.access(artifact.parent, os.W_OK | os.X_OK):
        raise PermissionError(f"artifact parent is not writable: {artifact.parent}")
    for volatile in (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm")):
        if _paths_overlap(artifact, volatile):
            raise UnsafeArtifactPath(
                f"durable M2 artifacts may not be published under {volatile}"
            )

    research = _resolve_real_path(
        arguments.research_root, label="research root", directory=True
    )
    research_worktree = _git_worktree_root(research)
    if research_worktree is not None:
        raise UnsafeArtifactPath(
            "canonical research input must remain outside every Git worktree; "
            f"found {research_worktree}"
        )
    contract_only = bool(getattr(arguments, "contract_only", False))
    runner_image_id = getattr(arguments, "runner_image_id", None)
    supplied = {
        name: getattr(arguments, name, None) for name in PRODUCTION_ARGUMENTS
    }
    supplied_names = tuple(name for name, value in supplied.items() if value is not None)
    if contract_only:
        if runner_image_id is not None:
            raise ValueError(
                "--contract-only cannot claim a canonical runner image identity"
            )
        if supplied_names:
            raise ValueError(
                "--contract-only cannot accept canonical-production inputs: "
                + ", ".join(supplied_names)
            )
        if not artifact.name.startswith("scratch-"):
            raise UnsafeArtifactPath(
                "contract-only evidence must use a scratch-* artifact leaf"
            )
        if _paths_overlap(artifact, research):
            raise UnsafeArtifactPath(
                f"artifact target and research root overlap: {artifact}, {research}"
            )
        return PreflightPaths(artifact_directory=artifact, research_root=research)

    if runner_image_id != M2_RUNNER_IMAGE_ID:
        raise ValueError(
            "canonical M2 production requires --runner-image-id equal to the "
            f"immutable accepted image {M2_RUNNER_IMAGE_ID}"
        )

    missing = tuple(name for name, value in supplied.items() if value is None)
    if missing:
        raise ValueError(
            "canonical M2 production requires every production input; missing: "
            + ", ".join(missing)
        )

    corpus = _resolve_real_path(
        supplied["corpus_root"], label="canonical corpus", directory=True
    )
    m1_artifact = _resolve_real_path(
        supplied["m1_artifact_dir"],
        label="verified M1 artifact",
        directory=True,
    )
    rayleigh = _resolve_real_path(
        supplied["rayleigh_data_dir"],
        label="Rayleigh data directory",
        directory=True,
    )
    config = _resolve_real_path(
        supplied["pyspectral_config"],
        label="pyspectral config",
        directory=False,
    )
    acquisition = _resolve_real_path(
        supplied["v30_acquisition_dir"],
        label="V30 acquisition directory",
        directory=True,
    )
    raw = _resolve_real_path(
        supplied["v30_raw_dir"], label="V30 raw directory", directory=True
    )
    land = _resolve_real_path(
        supplied["v30_land_zip"], label="V30 land ZIP", directory=False
    )

    inputs = {
        "research root": research,
        "canonical corpus": corpus,
        "verified M1 artifact": m1_artifact,
        "Rayleigh data directory": rayleigh,
        "pyspectral config": config,
        "V30 acquisition directory": acquisition,
        "V30 raw directory": raw,
        "V30 land ZIP": land,
    }
    for label, path in inputs.items():
        probe = path if path.is_dir() else path.parent
        worktree = _git_worktree_root(probe)
        if worktree is not None:
            raise UnsafeArtifactPath(
                f"{label} must remain outside every Git worktree; found {worktree}"
            )
        protected = path if path.is_dir() else path.parent
        if _paths_overlap(artifact, protected):
            raise UnsafeArtifactPath(
                "artifact target must be disjoint in both directions from "
                f"{label}: target={artifact}, protected={protected}"
            )

    rows = tuple(inputs.items())
    for index, (left_label, left) in enumerate(rows):
        for right_label, right in rows[index + 1 :]:
            if _paths_overlap(left, right):
                raise UnsafeArtifactPath(
                    "canonical M2 input roles may not alias or nest: "
                    f"{left_label}={left}, {right_label}={right}"
                )

    return PreflightPaths(
        artifact_directory=artifact,
        research_root=research,
        corpus_root=corpus,
        m1_artifact_directory=m1_artifact,
        rayleigh_data_directory=rayleigh,
        pyspectral_config=config,
        v30_acquisition_directory=acquisition,
        v30_raw_directory=raw,
        v30_land_zip=land,
    )


def _canonical_object_rows(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) is not tuple or len(value) != 4:
        raise RuntimeError("canonical corpus must bind exactly four immutable objects")
    rows: list[dict[str, Any]] = []
    expected_roles = ("red", "blue", "veggie", "ir")
    for expected_role, item in zip(expected_roles, value, strict=True):
        if hasattr(item, "role"):
            row = asdict(item)
        elif type(item) is dict:
            row = dict(item)
        else:
            raise TypeError("canonical corpus identity row is not registered")
        if set(row) != {"role", "file_identifier", "bytes", "sha256"}:
            raise RuntimeError("canonical corpus identity fields changed")
        filename = row["file_identifier"]
        digest = row["sha256"]
        if (
            row["role"] != expected_role
            or type(filename) is not str
            or not filename
            or Path(filename).name != filename
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("canonical corpus identity row is invalid")
        rows.append(
            {
                "role": expected_role,
                "file_identifier": filename,
                "bytes": row["bytes"],
                "sha256": digest,
            }
        )
    return tuple(rows)


def canonical_corpus_binding_from_evidence(
    evidence: CanonicalM2Evidence,
    corpus_root: Path,
) -> dict[str, Any]:
    if type(evidence) is not CanonicalM2Evidence:
        raise TypeError("canonical corpus binding requires exact M2 evidence")
    root = _resolve_real_path(corpus_root, label="canonical corpus", directory=True)
    rows = _canonical_object_rows(evidence.object_identities)
    record: dict[str, Any] = {
        "schema": "tat_truecolor_m2_canonical_corpus_binding/v1",
        "root": str(root),
        "objects": list(rows),
    }
    record["inventory_sha256"] = sha256_bytes(_strict_json_bytes(record))
    return record


def collect_canonical_corpus_binding(
    corpus_root: Path,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash only the four identity-bound canonical objects, without rendering."""

    root = _resolve_real_path(corpus_root, label="canonical corpus", directory=True)
    expected = _json_value(expected_binding)
    if expected.get("root") != str(root):
        raise RuntimeError("canonical corpus root identity changed")
    raw_rows = expected.get("objects")
    if type(raw_rows) is not list:
        raise RuntimeError("canonical corpus binding omitted object rows")
    rows = _canonical_object_rows(tuple(raw_rows))
    for row in rows:
        path = root / row["file_identifier"]
        if path.parent != root or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"canonical corpus object is absent or unsafe: {path}")
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"canonical corpus object identity changed: {path.name}")
    record: dict[str, Any] = {
        "schema": "tat_truecolor_m2_canonical_corpus_binding/v1",
        "root": str(root),
        "objects": list(rows),
    }
    record["inventory_sha256"] = sha256_bytes(_strict_json_bytes(record))
    if record != expected:
        raise RuntimeError("canonical corpus inventory binding changed")
    return record


def _reloaded_identity_rows(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) is not tuple:
        raise TypeError("independent canonical reload identities are not a tuple")
    return _canonical_object_rows(value)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = _strict_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(item) for item in array.shape]}
    )
    return sha256_bytes(header + array.tobytes(order="C"))


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_utc(value: Any) -> dt.datetime:
    if type(value) is not str:
        raise ValueError("timestamp is not a string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("timestamp is not explicit UTC")
    return parsed.astimezone(dt.timezone.utc)


def _array_record(name: str, values: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(item) for item in values.shape],
        "dtype": values.dtype.str,
        "sha256": _array_sha256(values),
    }


def _png_bytes(values: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(values, mode="RGB").save(
        buffer, format="PNG", optimize=False, compress_level=9
    )
    return buffer.getvalue()


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def _patch_metrics(patch: np.ndarray) -> dict[str, float]:
    rgb = patch.astype(np.float64)
    # Keep the independent publication replay bit-exact with the registered
    # sequence evaluator.  A BLAS-backed dot product is mathematically
    # equivalent, but it changes a few final float64 bits and makes the exact
    # aggregate binding reject the producer's own measured V30 record.
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    laplacian = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    means = np.mean(rgb, axis=(0, 1))
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    saturation = np.zeros(maximum.shape, dtype=np.float64)
    np.divide(maximum - minimum, maximum, out=saturation, where=maximum > 0.0)
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


def _artifact_row(
    filename: str,
    payload: bytes,
    *,
    media_type: str,
    description: str,
    source_binding_sha256: Mapping[str, str],
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "media_type": media_type,
        "width": width,
        "height": height,
        "description": description,
        "source_bindings": list(sorted(source_binding_sha256)),
        "source_binding_sha256": dict(sorted(source_binding_sha256.items())),
    }


def build_v30_patch_payloads(
    capture: V30SequenceCapture,
    sequence: SequenceEvaluation,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    """Serialize all 19 bound 96px V30 patches and their exact identities."""

    if type(capture) is not V30SequenceCapture:
        raise TypeError("V30 artifacts require an exact production capture")
    if (
        capture.schema != V30_CAPTURE_SCHEMA
        or type(capture.corpus) is not V30Corpus
        or capture.corpus.acquisition_index_sha256
        != V30_CANONICAL_ACQUISITION_INDEX_SHA256
        or capture.corpus.manifest_sha256 != V30_CANONICAL_MANIFEST_SHA256
        or capture.corpus.land_zip_sha256 != V30_NATURAL_EARTH_ZIP_SHA256
        or len(capture.corpus.objects) != 76
        or len(capture.corpus.groups) != V30_PATCH_COUNT
        or type(capture.frame_provenance) is not tuple
        or len(capture.frame_provenance) != V30_PATCH_COUNT
        or type(capture.screening) is not tuple
        or len(capture.screening) != V30_PATCH_COUNT
        or capture.frozen_output_mutant_killed is not True
        or any(type(item) is not V30Object for item in capture.corpus.objects)
    ):
        raise RuntimeError("V30 capture/corpus provenance is incomplete or noncanonical")
    land = capture.corpus.land_binding
    if (
        type(land) is not dict
        or land.get("dataset") != "Natural Earth 1:10m land"
        or land.get("zip_sha256") != V30_NATURAL_EARTH_ZIP_SHA256
        or land.get("patch_fully_covered") is not True
        or type(land.get("distance_to_land_boundary_deg")) is not float
        or float(land["distance_to_land_boundary_deg"]) < 0.5
        or land.get("minimum_required_distance_deg") != 0.5
    ):
        raise RuntimeError("V30 Natural Earth clear-land binding is invalid")
    if type(sequence) is not SequenceEvaluation or not sequence.passed:
        raise RuntimeError("V30 artifacts require a passing measured evaluation")
    recomputed = evaluate_timestamped_sequence(capture.binding)
    if _json_value(recomputed.as_dict()) != _json_value(sequence.as_dict()):
        raise RuntimeError("V30 evaluation differs from the retained renderer arrays")
    binding = capture.binding
    if type(binding) is not TimestampedSequenceBinding:
        raise TypeError("V30 capture binding is not the registered sequence type")
    if binding.schema != V30_BINDING_SCHEMA:
        raise RuntimeError("V30 capture uses an unregistered binding schema")
    if binding.stationary_world_bounds != V30_PATCH_WORLD_BOUNDS:
        raise RuntimeError("V30 capture does not use the predeclared world patch")
    if type(binding.frames) is not tuple or len(binding.frames) != V30_PATCH_COUNT:
        raise RuntimeError("V30 artifacts require exactly 19 retained renderer frames")
    if type(binding.patch_bounds) is not tuple or len(binding.patch_bounds) != 4:
        raise RuntimeError("V30 patch bounds are not an immutable four-tuple")
    top, bottom, left, right = binding.patch_bounds
    if (bottom - top, right - left, 3) != V30_PATCH_SHAPE:
        raise RuntimeError("predeclared V30 delivered patch is not exactly 96x96 RGB")

    patches: list[np.ndarray] = []
    patch_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    seen_timestamps: set[str] = set()
    for index, frame in enumerate(binding.frames, start=1):
        if type(frame) is not TimestampedRendererFrame:
            raise TypeError(f"V30 frame {index} is not the registered renderer type")
        raster = frame.raster_rgb_u8
        daylight = frame.full_daylight_mask
        if (
            type(raster) is not np.ndarray
            or raster.dtype != np.uint8
            or raster.ndim != 3
            or raster.shape[-1] != 3
            or type(daylight) is not np.ndarray
            or daylight.dtype != np.bool_
            or daylight.shape != raster.shape[:2]
            or not (0 <= top < bottom <= raster.shape[0])
            or not (0 <= left < right <= raster.shape[1])
        ):
            raise RuntimeError(f"V30 frame {index} raster/mask binding is invalid")
        if _array_sha256(raster) != frame.renderer_raster_sha256:
            raise RuntimeError(f"V30 frame {index} raster bytes changed")
        provenance = capture.frame_provenance[index - 1]
        screen = capture.screening[index - 1]
        corpus_group = capture.corpus.groups[index - 1]
        corpus_group_hashes = tuple(item.sha256 for item in corpus_group)
        exact_starts = {item.exact_scan_start_utc for item in corpus_group}
        nominal_slots = {item.nominal_slot_utc for item in corpus_group}
        source_rows = (
            provenance.get("source_objects", ())
            if type(provenance) is dict
            else ()
        )
        if (
            type(provenance) is not dict
            or set(provenance)
            != {
                "nominal_slot_utc",
                "exact_noaa_scan_start_utc",
                "production_scan_start_utc",
                "source_objects",
                "request_identity_sha256",
                "grid_identity_sha256",
                "renderer_png_sha256",
                "renderer_raster_sha256",
                "trace_identity_sha256",
                "delivered_shape",
                "patch_bounds",
                "world_patch_max_pixel_edge_residual",
                "world_patch_pixel_edge_tolerance",
            }
            or len(exact_starts) != 1
            or len(nominal_slots) != 1
            or frame.timestamp_utc != next(iter(exact_starts), None)
            or provenance.get("exact_noaa_scan_start_utc") != frame.timestamp_utc
            or provenance.get("nominal_slot_utc")
            != next(iter(nominal_slots), None)
            or provenance.get("production_scan_start_utc")
            != _parse_utc(frame.timestamp_utc).replace(microsecond=0).isoformat()
            or provenance.get("request_identity_sha256")
            != frame.request_identity_sha256
            or provenance.get("grid_identity_sha256") != frame.grid_identity_sha256
            or provenance.get("renderer_png_sha256") != frame.renderer_png_sha256
            or provenance.get("renderer_raster_sha256")
            != frame.renderer_raster_sha256
            or not _valid_sha256(provenance.get("trace_identity_sha256"))
            or provenance.get("delivered_shape") != list(raster.shape[:2])
            or provenance.get("patch_bounds") != list(binding.patch_bounds)
            or type(provenance.get("world_patch_max_pixel_edge_residual"))
            is not float
            or not math.isfinite(provenance["world_patch_max_pixel_edge_residual"])
            or provenance["world_patch_max_pixel_edge_residual"] < 0.0
            or provenance["world_patch_max_pixel_edge_residual"]
            > V30_PATCH_EDGE_BINDING_TOLERANCE_PX
            or provenance.get("world_patch_pixel_edge_tolerance")
            != V30_PATCH_EDGE_BINDING_TOLERANCE_PX
            or corpus_group_hashes != frame.source_object_sha256
            or type(source_rows) is not list
            or len(source_rows) != 4
            or any(type(row) is not dict for row in source_rows)
            or source_rows != [item.as_dict() for item in corpus_group]
        ):
            raise RuntimeError(f"V30 frame {index} provenance differs from arrays")
        if (
            type(screen) is not dict
            or set(screen)
            != {
                "nominal_slot_utc",
                "dqf_zero_fraction_by_role",
                "cloud_proxy_fraction",
                "c13_above_280k_fraction",
                "cloud_proxy_definition",
                "selection_policy",
            }
            or screen["nominal_slot_utc"] != next(iter(nominal_slots))
            or type(screen["dqf_zero_fraction_by_role"]) is not dict
            or tuple(screen["dqf_zero_fraction_by_role"]) != V30_ROLE_ORDER
            or any(
                type(value) is not float or value != 1.0
                for value in screen["dqf_zero_fraction_by_role"].values()
            )
            or type(screen["cloud_proxy_fraction"]) is not float
            or screen["cloud_proxy_fraction"] != 0.0
            or type(screen["c13_above_280k_fraction"]) is not float
            or screen["c13_above_280k_fraction"] != 1.0
            or screen["cloud_proxy_definition"]
            != "C02>0.30 or C13<270K; acquisition screen only"
            or screen["selection_policy"]
            != "predeclared_before_renderer_metric_evaluation"
        ):
            raise RuntimeError(f"V30 frame {index} raw clear-land screen is invalid")
        patch = raster[top:bottom, left:right]
        if patch.shape != V30_PATCH_SHAPE or not np.all(daylight[top:bottom, left:right]):
            raise RuntimeError(f"V30 frame {index} patch is not exact full-daylight 96px")
        if frame.timestamp_utc in seen_timestamps:
            raise RuntimeError("V30 patch timestamps are not unique")
        seen_timestamps.add(frame.timestamp_utc)
        immutable_patch = np.ascontiguousarray(patch).copy()
        patches.append(immutable_patch)
        filename = f"v30_patch_{index:02d}_96px.png"
        png = _png_bytes(immutable_patch)
        payloads[filename] = png
        record = _array_record(f"v30_patch_{index:02d}", immutable_patch)
        patch_rows.append(
            {
                "index": index - 1,
                "timestamp_utc": frame.timestamp_utc,
                "filename": filename,
                "png_sha256": sha256_bytes(png),
                "png_bytes": len(png),
                "array": record,
                "metrics": _patch_metrics(immutable_patch),
                "parent_renderer_raster_sha256": frame.renderer_raster_sha256,
                "parent_renderer_png_sha256": frame.renderer_png_sha256,
                "request_identity_sha256": frame.request_identity_sha256,
                "grid_identity_sha256": frame.grid_identity_sha256,
                "source_object_sha256": list(frame.source_object_sha256),
            }
        )
        metadata[filename] = _artifact_row(
            filename,
            png,
            media_type="image/png",
            width=96,
            height=96,
            description=(
                f"V30 frame {index:02d} predeclared clear-land full-daylight "
                "patch at exact delivered 100% scale."
            ),
            source_binding_sha256={
                "patch_array": record["sha256"],
                "renderer_raster": frame.renderer_raster_sha256,
            },
        )

    stack = np.stack(patches, axis=0)
    if stack.dtype != np.uint8 or stack.shape != (19, 96, 96, 3):
        raise RuntimeError("V30 retained patch stack identity changed")
    if (
        tuple(capture.raster_sha256)
        != tuple(frame.renderer_raster_sha256 for frame in binding.frames)
        or capture.request_identity_sha256
        != binding.frames[0].request_identity_sha256
        or capture.grid_identity_sha256 != binding.frames[0].grid_identity_sha256
    ):
        raise RuntimeError("V30 capture-level identities differ from retained frames")
    stack_name = "v30_patches_19x96x96_rgb_uint8.npy"
    stack_payload = _npy_bytes(stack)
    payloads[stack_name] = stack_payload
    stack_record = _array_record("v30_patch_stack", stack)
    metadata[stack_name] = _artifact_row(
        stack_name,
        stack_payload,
        media_type="application/x-npy",
        width=96,
        height=96,
        description="Raw uint8 stack for all 19 bound V30 patches in timestamp order.",
        source_binding_sha256={
            "v30_patch_stack": stack_record["sha256"],
        },
    )

    capture_summary = _json_value(capture.summary_dict())
    evaluation_summary = _json_value(sequence.as_dict())
    evidence = {
        "schema": "tat_truecolor_m2_v30_patch_evidence/v1",
        "patch_count": V30_PATCH_COUNT,
        "patch_shape": list(V30_PATCH_SHAPE),
        "patch_bounds": list(binding.patch_bounds),
        "stationary_world_bounds": list(binding.stationary_world_bounds),
        "stack": {
            "filename": stack_name,
            "payload_sha256": sha256_bytes(stack_payload),
            "payload_bytes": len(stack_payload),
            **stack_record,
        },
        "capture_summary_sha256": sha256_bytes(_strict_json_bytes(capture_summary)),
        "capture_summary": capture_summary,
        "evaluation_summary_sha256": sha256_bytes(
            _strict_json_bytes(evaluation_summary)
        ),
        "evaluation": evaluation_summary,
        "patches": patch_rows,
        "limitations": list(PRODUCTION_LIMITATIONS),
    }
    evidence_name = "v30_patch_evidence.json"
    evidence_payload = _strict_json_bytes(evidence)
    payloads[evidence_name] = evidence_payload
    metadata[evidence_name] = _artifact_row(
        evidence_name,
        evidence_payload,
        media_type="application/json",
        description="Machine-readable V30 patch identities and independently recomputed metrics.",
        source_binding_sha256={
            "v30_capture_summary": evidence["capture_summary_sha256"],
            "v30_evaluation_summary": evidence["evaluation_summary_sha256"],
            "v30_patch_stack": stack_record["sha256"],
        },
    )
    if set(payloads) != set(metadata):
        raise RuntimeError("V30 payload declarations are incomplete")
    return dict(sorted(payloads.items())), dict(sorted(metadata.items()))


def assemble_production_payloads(
    gallery: M2GalleryBundle,
    v30_payloads: Mapping[str, bytes],
    v30_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if type(gallery) is not M2GalleryBundle:
        raise TypeError("M2 gallery builder returned an unregistered bundle")
    gallery_manifest = gallery.manifest_dict()
    gallery_payloads = dict(gallery.payloads)
    gallery_metadata = _json_value(gallery_manifest.get("artifacts"))
    if set(gallery_payloads) != set(gallery_metadata):
        raise RuntimeError("M2 gallery payload/manifest coverage differs")
    for name, payload in gallery_payloads.items():
        row = gallery_metadata[name]
        if (
            type(payload) is not bytes
            or row.get("filename") != name
            or row.get("bytes") != len(payload)
            or row.get("sha256") != sha256_bytes(payload)
        ):
            raise RuntimeError(f"M2 gallery artifact identity differs: {name}")

    manifest_name = "m2_gallery_manifest.json"
    manifest_payload = _strict_json_bytes(gallery_manifest)
    gallery_payloads[manifest_name] = manifest_payload
    gallery_metadata[manifest_name] = _artifact_row(
        manifest_name,
        manifest_payload,
        media_type="application/json",
        description="Immutable manifest for all primary-layer M2 review payloads.",
        source_binding_sha256={
            "canonical_evidence_summary": gallery_manifest["source_evidence"][
                "summary_sha256"
            ]
        },
    )

    normalized_v30_payloads = dict(v30_payloads)
    normalized_v30_metadata = _json_value(v30_metadata)
    if set(normalized_v30_payloads) != set(normalized_v30_metadata):
        raise RuntimeError("V30 payload/metadata coverage differs")
    collision = set(gallery_payloads) & set(normalized_v30_payloads)
    if collision:
        raise RuntimeError(f"M2 gallery/V30 payload filename collision: {sorted(collision)}")
    for name, payload in normalized_v30_payloads.items():
        row = normalized_v30_metadata[name]
        if (
            type(payload) is not bytes
            or row.get("filename") != name
            or row.get("bytes") != len(payload)
            or row.get("sha256") != sha256_bytes(payload)
        ):
            raise RuntimeError(f"V30 artifact identity differs: {name}")
    return (
        dict(sorted({**gallery_payloads, **normalized_v30_payloads}.items())),
        dict(sorted({**gallery_metadata, **normalized_v30_metadata}.items())),
    )


_REQUIRED_M2_GALLERY_PAYLOADS = frozenset(
    {
        "01_m2_candidate_native.png",
        "02_exact_area_reference_native.png",
        "03_old_nearest_mutant_native.png",
        "04_display_first_mutant_native.png",
        "05_candidate_signed_rec601_error_native.png",
        "06_old_nearest_mutant_signed_rec601_error_native.png",
        "07_display_first_mutant_signed_rec601_error_native.png",
        "08_native_comparison_sheet_100pct.png",
        "09_signed_error_sheet_shared_scale_100pct.png",
        "10_signed_rec601_error_values_float32.npy",
        "11_signed_rec601_error_summary.csv",
        "12_canonical_spectral_comparison.csv",
        "13_canonical_spectral_comparison.png",
        "14_production_renderer.png",
        "README.md",
        "REVIEW_INSTRUCTIONS.md",
    }
)
_REQUIRED_V30_PAYLOADS = frozenset(
    {
        *(f"v30_patch_{index:02d}_96px.png" for index in range(1, 20)),
        "v30_patches_19x96x96_rgb_uint8.npy",
        "v30_patch_evidence.json",
    }
)


def _publication_mapping(
    value: Any,
    *,
    label: str,
    nonempty: bool = True,
) -> dict[str, Any]:
    if type(value) is not dict or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise RuntimeError(f"M2 publication {label} must be a {qualifier}JSON object")
    return value


def _exact_publication_mapping(
    value: Any,
    *,
    label: str,
    keys: set[str] | frozenset[str],
    nonempty: bool = True,
) -> dict[str, Any]:
    result = _publication_mapping(value, label=label, nonempty=nonempty)
    if set(result) != set(keys):
        raise RuntimeError(f"canonical M2 {label} JSON-object schema is not exact")
    return result


def _publication_list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeError(f"M2 publication {label} must be a JSON array")
    return value


def _publication_json_payload(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"M2 publication {label} is not strict JSON") from exc
    result = _publication_mapping(parsed, label=label)
    if payload != _strict_json_bytes(result):
        raise RuntimeError(f"M2 publication {label} is not canonical strict JSON")
    return result


def _valid_publication_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_UNKNOWN_STAGE_CLAIM_SCOPE = re.compile(
    r"(?:^|_)(?:m[3-8]|v(?:21|22|23|24|44|45|46))(?:_|$)"
)
_UNKNOWN_STAGE_CLAIM_TERM = re.compile(
    r"(?:approv|claim|clos|complet|pass|stage|advance|authorit|migrat)"
)


def _reject_unknown_stage_claim_keys(value: Any, *, path: str = "report") -> None:
    """Reject closure-shaped keys hiding inside otherwise variable measurements."""

    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise RuntimeError(f"canonical M2 JSON key is not a string at {path}")
            lowered = key.lower()
            if (
                _UNKNOWN_STAGE_CLAIM_SCOPE.search(lowered)
                and _UNKNOWN_STAGE_CLAIM_TERM.search(lowered)
            ):
                raise RuntimeError(
                    f"canonical M2 unknown nested stage/closure claim is forbidden: "
                    f"{path}.{key}"
                )
            if (
                "historical_true_color" in lowered
                or "historical_product_true_color" in lowered
                or "true_color_authority" in lowered
            ):
                raise RuntimeError(
                    "canonical M2 unsupported historical true-color claim is "
                    f"forbidden: {path}.{key}"
                )
            _reject_unknown_stage_claim_keys(child, path=f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_unknown_stage_claim_keys(child, path=f"{path}[{index}]")


def _validate_bound_production_inputs(
    input_bindings: Mapping[str, Any],
    rayleigh: Mapping[str, Any],
) -> None:
    corpus = _publication_mapping(
        input_bindings.get("canonical_corpus"), label="canonical corpus binding"
    )
    objects = _publication_list(
        corpus.get("objects"), label="canonical corpus objects"
    )
    roles = ("red", "blue", "veggie", "ir")
    if (
        set(corpus) != {"schema", "root", "objects", "inventory_sha256"}
        or corpus.get("schema")
        != "tat_truecolor_m2_canonical_corpus_binding/v1"
        or type(corpus.get("root")) is not str
        or not Path(corpus["root"]).is_absolute()
        or len(objects) != 4
    ):
        raise RuntimeError("canonical corpus input binding is incomplete")
    object_hashes: list[str] = []
    object_names: list[str] = []
    for role, raw_row in zip(roles, objects, strict=True):
        row = _publication_mapping(raw_row, label="canonical corpus object")
        name = row.get("file_identifier")
        digest = row.get("sha256")
        if (
            set(row) != {"role", "file_identifier", "bytes", "sha256"}
            or row.get("role") != role
            or type(name) is not str
            or Path(name).name != name
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not _valid_publication_sha256(digest)
        ):
            raise RuntimeError("canonical corpus object identity is invalid")
        object_names.append(name)
        object_hashes.append(digest)
    corpus_unsigned = {
        "schema": corpus["schema"],
        "root": corpus["root"],
        "objects": objects,
    }
    if (
        len(set(object_names)) != 4
        or len(set(object_hashes)) != 4
        or corpus.get("inventory_sha256")
        != sha256_bytes(_strict_json_bytes(corpus_unsigned))
    ):
        raise RuntimeError("canonical corpus inventory identity is inconsistent")

    m1_baseline = _exact_publication_mapping(
        input_bindings.get("verified_m1_artifact"),
        label="verified M1 binding",
        keys={
            "directory",
            "report_sha256",
            "checksum_index_sha256",
            "gallery_manifest_sha256",
            "git_head",
            "scene_id",
            "source_shape",
            "agg_internal_shape",
            "agg_affine",
            "delivered_internal_bounds",
            "v15_out_of_range_fraction",
            "v16_nearest_ringing_dn",
            "v16_hanning_ringing_dn",
            "v16_measurements",
            "object_identities",
            "source_display_sha256",
            "delivered_shape",
            "m0_nearest_raster_sha256",
            "m0_nearest_png_sha256",
            "m1_hanning_raster_sha256",
            "m1_hanning_png_sha256",
            "m0_runtime_interpolations",
            "m1_runtime_interpolations",
        },
    )
    trusted_m1 = {
        "report_sha256": M1_TRUSTED_REPORT_SHA256,
        "checksum_index_sha256": M1_TRUSTED_CHECKSUM_INDEX_SHA256,
        "gallery_manifest_sha256": M1_TRUSTED_GALLERY_MANIFEST_SHA256,
        "git_head": M1_TRUSTED_GIT_HEAD,
    }
    if any(m1_baseline.get(key) != value for key, value in trusted_m1.items()):
        raise RuntimeError("verified M1 input binding differs from trusted identities")

    v30_corpus = _exact_publication_mapping(
        input_bindings.get("v30_corpus"),
        label="V30 corpus binding",
        keys={
            "acquisition_directory",
            "raw_directory",
            "manifest",
            "acquisition_index_sha256",
            "land_zip",
            "land_binding",
            "object_count",
            "group_count",
            "total_bytes",
            "inventory_sha256",
        },
    )
    manifest = _exact_publication_mapping(
        v30_corpus.get("manifest"),
        label="V30 manifest binding",
        keys={"path", "sha256"},
    )
    land_zip = _exact_publication_mapping(
        v30_corpus.get("land_zip"),
        label="V30 land binding",
        keys={"path", "sha256"},
    )
    land_binding = _exact_publication_mapping(
        v30_corpus.get("land_binding"),
        label="V30 land-coverage binding",
        keys={
            "dataset",
            "zip_sha256",
            "patch_fully_covered",
            "distance_to_land_boundary_deg",
            "minimum_required_distance_deg",
        },
    )
    if (
        v30_corpus.get("acquisition_index_sha256")
        != V30_CANONICAL_ACQUISITION_INDEX_SHA256
        or manifest.get("sha256") != V30_CANONICAL_MANIFEST_SHA256
        or land_zip.get("sha256") != V30_NATURAL_EARTH_ZIP_SHA256
        or land_binding.get("zip_sha256") != V30_NATURAL_EARTH_ZIP_SHA256
        or land_binding.get("patch_fully_covered") is not True
        or v30_corpus.get("object_count") != 76
        or v30_corpus.get("group_count") != V30_PATCH_COUNT
        or not _valid_publication_sha256(v30_corpus.get("inventory_sha256"))
    ):
        raise RuntimeError("V30 corpus input binding is incomplete or noncanonical")

    _exact_publication_mapping(
        rayleigh,
        label="Rayleigh startup binding",
        keys={
            "data_directory",
            "config_file",
            "config_sha256",
            "configured_rsr_directory",
            "configured_rayleigh_directory",
            "writable_probe_passed",
            "sensors",
            "data_inventory_schema",
            "data_inventory_file_count",
            "data_inventory_total_bytes",
            "data_inventory_file_set_sha256",
            "data_inventory_sha256",
            "data_inventory_files",
            "factory_binding",
            "production_cache_session_started_empty",
            "production_cache_exact_expected_keys",
            "production_cache_returned_identities_match",
            "passed",
        },
    )
    files = _publication_list(
        rayleigh.get("data_inventory_files"), label="Rayleigh inventory files"
    )
    sensors = _publication_list(rayleigh.get("sensors"), label="Rayleigh correctors")
    expected_sensors = list(RAYLEIGH_STARTUP_SENSORS)
    observed_sensors: list[tuple[str, str]] = []
    for raw_sensor in sensors:
        sensor = _exact_publication_mapping(
            raw_sensor,
            label="Rayleigh corrector",
            keys={
                "platform",
                "sensor",
                "status",
                "corrector_class",
                "detail",
            },
        )
        if (
            sensor.get("status") != "valid_corrector"
            or type(sensor.get("corrector_class")) is not str
            or not sensor["corrector_class"]
            or sensor.get("detail") is not None
        ):
            raise RuntimeError("Rayleigh corrector binding is invalid")
        observed_sensors.append((sensor.get("platform"), sensor.get("sensor")))
    inventory_rows: list[dict[str, Any]] = []
    prior_path: str | None = None
    for raw_file in files:
        row = _publication_mapping(raw_file, label="Rayleigh inventory file")
        relative_path = row.get("relative_path")
        if (
            set(row) != {"relative_path", "bytes", "sha256"}
            or type(relative_path) is not str
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or (prior_path is not None and relative_path <= prior_path)
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not _valid_publication_sha256(row.get("sha256"))
        ):
            raise RuntimeError("Rayleigh inventory file binding is invalid")
        prior_path = relative_path
        inventory_rows.append(row)
    file_set_payload = {
        "schema": RAYLEIGH_DATA_INVENTORY_SCHEMA,
        "relative_paths": [row["relative_path"] for row in inventory_rows],
    }
    file_set_hash = sha256_bytes(
        (
            json.dumps(
                file_set_payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    total_bytes = sum(row["bytes"] for row in inventory_rows)
    inventory_payload = {
        "schema": RAYLEIGH_DATA_INVENTORY_SCHEMA,
        "file_count": len(inventory_rows),
        "total_bytes": total_bytes,
        "file_set_sha256": file_set_hash,
        "files": inventory_rows,
    }
    inventory_hash = sha256_bytes(
        (
            json.dumps(
                inventory_payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    if (
        rayleigh.get("passed") is not True
        or rayleigh.get("factory_binding")
        != "production_fresh_cache_session"
        or rayleigh.get("production_cache_session_started_empty") is not True
        or rayleigh.get("production_cache_exact_expected_keys") is not True
        or rayleigh.get("production_cache_returned_identities_match") is not True
        or rayleigh.get("writable_probe_passed") is not True
        or rayleigh.get("data_inventory_schema")
        != RAYLEIGH_DATA_INVENTORY_SCHEMA
        or not inventory_rows
        or rayleigh.get("data_inventory_file_count") != len(inventory_rows)
        or rayleigh.get("data_inventory_total_bytes") != total_bytes
        or rayleigh.get("data_inventory_file_set_sha256") != file_set_hash
        or rayleigh.get("data_inventory_sha256") != inventory_hash
        or observed_sensors != expected_sensors
        or not _valid_publication_sha256(rayleigh.get("config_sha256"))
        or rayleigh.get("configured_rsr_directory")
        != rayleigh.get("data_directory")
        or rayleigh.get("configured_rayleigh_directory")
        != rayleigh.get("data_directory")
    ):
        raise RuntimeError("Rayleigh startup/inventory input binding is inconsistent")


def _decode_publication_png(
    payload: bytes,
    *,
    label: str,
    opaque_rgba: bool = False,
) -> np.ndarray:
    from PIL import Image

    required_mode = "RGBA" if opaque_rgba else "RGB"
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if (
                image.format != "PNG"
                or image.mode != required_mode
                or getattr(image, "n_frames", 1) != 1
            ):
                required_description = (
                    "one opaque RGBA PNG" if opaque_rgba else "one RGB PNG"
                )
                raise RuntimeError(
                    f"canonical M2 {label} is not {required_description}"
                )
            array = np.asarray(image).copy()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"canonical M2 {label} is not a decodable PNG") from exc
    if opaque_rgba:
        if (
            type(array) is not np.ndarray
            or array.dtype != np.uint8
            or array.ndim != 3
            or array.shape[-1] != 4
            or min(array.shape[:2]) <= 0
            or not np.all(array[..., 3] == np.uint8(255))
        ):
            raise RuntimeError(
                f"canonical M2 {label} does not have exact opaque RGBA pixels"
            )
        array = array[..., :3]
    if type(array) is not np.ndarray or array.dtype != np.uint8 or array.ndim != 3:
        raise RuntimeError(f"canonical M2 {label} PNG array is invalid")
    if array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
        raise RuntimeError(f"canonical M2 {label} PNG shape is invalid")
    return np.ascontiguousarray(array)


def _validate_gallery_payload_semantics(
    *,
    canonical: Mapping[str, Any],
    repeatability: Mapping[str, Any],
    sequence: Mapping[str, Any],
    gallery_manifest: Mapping[str, Any],
    gallery_rows: Mapping[str, Mapping[str, Any]],
    extras: Mapping[str, bytes],
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    """Decode and independently recompute every non-V30 gallery payload."""

    arrays = _publication_mapping(
        canonical.get("arrays"), label="canonical gallery array registry"
    )
    native_specs = (
        (
            "candidate",
            "01_m2_candidate_native.png",
            "candidate_floor",
            "M2 candidate",
        ),
        (
            "exact_area_reference",
            "02_exact_area_reference_native.png",
            "exact_area_reference_floor",
            "Exact-area reference",
        ),
        (
            "old_nearest_mutant",
            "03_old_nearest_mutant_native.png",
            "old_nearest_mutant_floor",
            "Old nearest mutant",
        ),
        (
            "display_first_mutant",
            "04_display_first_mutant_native.png",
            "display_first_mutant_floor",
            "Display-first mutant",
        ),
    )
    decoded: dict[str, np.ndarray] = {}
    array_bindings: dict[str, str] = {}
    for logical_name, filename, array_name, _ in native_specs:
        raster = _decode_publication_png(extras[filename], label=filename)
        if extras[filename] != _png_bytes(raster):
            raise RuntimeError(
                f"canonical M2 native gallery PNG encoding differs: {filename}"
            )
        if arrays.get(array_name) != _array_record(array_name, raster):
            raise RuntimeError(
                f"canonical M2 native gallery pixels differ from report: {filename}"
            )
        decoded[logical_name] = raster
        array_bindings[array_name] = arrays[array_name]["sha256"]
    native_shape = decoded["candidate"].shape
    if (
        any(raster.shape != native_shape for raster in decoded.values())
        or gallery_manifest.get("native_shape") != list(native_shape)
    ):
        raise RuntimeError("canonical M2 native gallery raster shapes differ")
    height, width = native_shape[:2]

    candidate = decoded["candidate"]
    reference = decoded["exact_area_reference"]
    nearest = decoded["old_nearest_mutant"]
    display_first = decoded["display_first_mutant"]
    errors = {
        "candidate": m2_gallery_module._signed_rec601_error(candidate, reference),
        "old_nearest_mutant": m2_gallery_module._signed_rec601_error(
            nearest, reference
        ),
        "display_first_mutant": m2_gallery_module._signed_rec601_error(
            display_first, reference
        ),
    }
    expected_error_stack = np.ascontiguousarray(
        np.stack([errors[name] for name in m2_gallery_module.ERROR_ORDER]).astype(
            np.float32
        )
    )
    error_npy_name = "10_signed_rec601_error_values_float32.npy"
    try:
        error_stack = np.load(io.BytesIO(extras[error_npy_name]), allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise RuntimeError("canonical M2 signed-error NPY is invalid") from exc
    if (
        type(error_stack) is not np.ndarray
        or error_stack.dtype != np.float32
        or error_stack.shape != (len(m2_gallery_module.ERROR_ORDER), height, width)
        or extras[error_npy_name] != _npy_bytes(error_stack)
        or not np.array_equal(error_stack, expected_error_stack)
    ):
        raise RuntimeError(
            "canonical M2 signed-error NPY differs from decoded native rasters"
        )
    shared_scale = m2_gallery_module._canonical_error_scale(
        tuple(errors[name] for name in m2_gallery_module.ERROR_ORDER)
    )
    error_maps = {
        name: m2_gallery_module._signed_error_map(errors[name], shared_scale)
        for name in m2_gallery_module.ERROR_ORDER
    }
    for index, name in enumerate(m2_gallery_module.ERROR_ORDER, start=5):
        filename = f"{index:02d}_{name}_signed_rec601_error_native.png"
        raster = _decode_publication_png(extras[filename], label=filename)
        if (
            extras[filename] != _png_bytes(raster)
            or not np.array_equal(raster, error_maps[name])
        ):
            raise RuntimeError(
                f"canonical M2 signed-error PNG differs from NPY values: {filename}"
            )

    main_sheet, main_sheet_size, main_bounds = m2_gallery_module._comparison_sheet(
        tuple((label, decoded[name]) for name, _, _, label in native_specs)
    )
    if extras["08_native_comparison_sheet_100pct.png"] != main_sheet:
        raise RuntimeError("canonical M2 native comparison sheet is not reproducible")
    error_sheet, error_sheet_size, error_bounds = m2_gallery_module._comparison_sheet(
        tuple(
            (name.replace("_", " "), error_maps[name])
            for name in m2_gallery_module.ERROR_ORDER
        )
    )
    if extras["09_signed_error_sheet_shared_scale_100pct.png"] != error_sheet:
        raise RuntimeError("canonical M2 signed-error comparison sheet is not reproducible")
    if extras["11_signed_rec601_error_summary.csv"] != (
        m2_gallery_module._error_summary_csv(errors, shared_scale)
    ):
        raise RuntimeError("canonical M2 signed-error CSV is not reproducible")

    frequency, profiles = m2_gallery_module._spectral_profiles(decoded)
    spectral_csv = m2_gallery_module._spectral_csv(frequency, profiles)
    if extras["12_canonical_spectral_comparison.csv"] != spectral_csv:
        raise RuntimeError("canonical M2 spectral CSV differs from native rasters")
    ratios = {
        name: m2_validation.radial_high_frequency_power_ratio(
            raster,
            reference,
            cutoff=m2_gallery_module.SPECTRAL_CUTOFF,
            bins=m2_gallery_module.SPECTRAL_BINS,
        )
        for name, raster in decoded.items()
        if name != "exact_area_reference"
    }
    metrics = _exact_publication_mapping(
        canonical.get("metrics"),
        label="canonical gallery metric registry",
        keys={
            "V01", "V02", "V03", "V07", "V08", "V13", "V14", "V15",
            "V16", "V41", "M0_M1_ROLLBACK", "mutation_discriminators",
            "limitations",
        },
    )
    v01 = _exact_publication_mapping(
        metrics.get("V01"),
        label="canonical V01 metrics",
        keys={
            "candidate_rms_dn", "old_nearest_rms_dn", "reference_quantizer",
            "reference",
        },
    )
    v03 = _exact_publication_mapping(
        metrics.get("V03"),
        label="canonical V03 metrics",
        keys={
            "candidate_ratio", "old_nearest_ratio", "luma", "spatial_window",
            "radial_bins", "cutoff_cycles_per_pixel",
        },
    )
    if (
        v01
        != {
            "candidate_rms_dn": m2_validation.rms_dn(candidate, reference),
            "old_nearest_rms_dn": m2_validation.rms_dn(nearest, reference),
            "reference_quantizer": "production_floor_u8_no_dither",
            "reference": (
                "exact detector-cell area response derived from captured affine; "
                "physical reflectance/K/day-weight and coverage first, one CIRA last"
            ),
        }
        or v01["candidate_rms_dn"] > 1.5
        or v01["old_nearest_rms_dn"] <= 1.5
        or v03
        != {
            "candidate_ratio": ratios["candidate"],
            "old_nearest_ratio": ratios["old_nearest_mutant"],
            "luma": "Rec.601",
            "spatial_window": "separable_2D_Hann",
            "radial_bins": m2_gallery_module.SPECTRAL_BINS,
            "cutoff_cycles_per_pixel": m2_gallery_module.SPECTRAL_CUTOFF,
        }
        or v03["candidate_ratio"] > 1.10
        or v03["old_nearest_ratio"] <= 1.10
    ):
        raise RuntimeError("canonical M2 V01/V03 metrics differ from gallery rasters")

    v02 = _exact_publication_mapping(
        metrics.get("V02"),
        label="canonical V02 metrics",
        keys={"candidate", "old_nearest_mutant"},
    )
    coverage_keys = {
        "source_rows", "source_columns", "contributing_rows",
        "contributing_columns", "discarded_rows", "discarded_columns",
        "discarded_row_fraction", "discarded_column_fraction", "minimum_weight",
        "maximum_row_sum_error", "procedure",
    }
    coverage_rows = {
        name: _exact_publication_mapping(
            v02.get(name), label=f"canonical V02 {name}", keys=coverage_keys
        )
        for name in ("candidate", "old_nearest_mutant")
    }
    candidate_coverage = coverage_rows["candidate"]
    if (
        candidate_coverage.get("discarded_rows") != 0
        or candidate_coverage.get("discarded_columns") != 0
        or candidate_coverage.get("discarded_row_fraction") != 0.0
        or candidate_coverage.get("discarded_column_fraction") != 0.0
        or any(
            row.get("procedure") != "measured_Agg_impulse_response_source_support"
            or any(
                type(row.get(key)) is not int or row[key] <= 0
                for key in (
                    "source_rows", "source_columns", "contributing_rows",
                    "contributing_columns",
                )
            )
            or any(
                type(row.get(key)) not in {int, float}
                or not math.isfinite(float(row[key]))
                for key in (
                    "discarded_row_fraction", "discarded_column_fraction",
                    "minimum_weight", "maximum_row_sum_error",
                )
            )
            for row in coverage_rows.values()
        )
    ):
        raise RuntimeError("canonical M2 V02 support metrics are not closing")

    baseline = _publication_mapping(
        canonical.get("m1_baseline"), label="canonical M1 metric baseline"
    )
    v07 = _exact_publication_mapping(
        metrics.get("V07"),
        label="canonical V07 metrics",
        keys={"m1_report_sha256", "m1_git_head", "current_cira_source_sha256"},
    )
    if (
        v07.get("m1_report_sha256") != baseline.get("report_sha256")
        or v07.get("m1_git_head") != baseline.get("git_head")
        or v07.get("current_cira_source_sha256")
        != production_adapter.cira_source_sha256()
        or not _valid_publication_sha256(v07.get("current_cira_source_sha256"))
    ):
        raise RuntimeError("canonical M2 V07 baseline/source metrics differ")

    v08 = _exact_publication_mapping(
        metrics.get("V08"),
        label="canonical V08 metrics",
        keys={"passed", "failures", "pixels", "binding", "finalizer_source_sha256"},
    )
    if v08 != _json_value(
        m2_production_module._reference_pixel_finalizer_measurements()  # noqa: SLF001
    ):
        raise RuntimeError("canonical M2 V08 metrics differ from live reference pixels")

    v13 = _exact_publication_mapping(
        metrics.get("V13"),
        label="canonical V13 metrics",
        keys={
            "mean_signed_all_channel_dn", "mean_signed_rec601_luma_dn",
            "display_first_mutant_mean_signed_dn", "operation_trace", "trace_issues",
            "trace_identity_sha256", "geometry_identity_sha256",
            "terminator_order_mutant", "captured_endpoint_binding",
            "endpoint_identity_fixture",
        },
    )
    terminator = _exact_publication_mapping(
        v13.get("terminator_order_mutant"),
        label="canonical V13 terminator mutant",
        keys={
            "fixture", "approved_source_weight_then_resample_sha256",
            "mutant_resample_cos_then_weight_sha256", "maximum_abs_weight_difference",
            "mean_abs_weight_difference", "killed",
        },
    )
    primary_capture = _publication_mapping(
        canonical.get("primary_capture"), label="canonical V13 primary capture"
    )
    if (
        v13.get("trace_issues") != []
        or v13.get("operation_trace") != primary_capture.get("operations")
        or v13.get("trace_identity_sha256")
        != primary_capture.get("trace_identity_sha256")
        or v13.get("captured_endpoint_binding")
        != primary_capture.get("endpoint_binding")
        or v13.get("endpoint_identity_fixture")
        != _json_value(
            m2_production_module._endpoint_restoration_measurements()  # noqa: SLF001
        )
        or any(
            type(v13.get(key)) not in {int, float}
            or not math.isfinite(float(v13[key]))
            for key in (
                "mean_signed_all_channel_dn", "mean_signed_rec601_luma_dn",
                "display_first_mutant_mean_signed_dn",
            )
        )
        or abs(float(v13["mean_signed_all_channel_dn"])) > 0.05
        or abs(float(v13["mean_signed_rec601_luma_dn"])) > 0.05
        or abs(float(v13["display_first_mutant_mean_signed_dn"])) <= 0.05
        or terminator.get("killed") is not True
        or type(terminator.get("maximum_abs_weight_difference")) not in {int, float}
        or float(terminator["maximum_abs_weight_difference"]) <= 1.0e-3
    ):
        raise RuntimeError("canonical M2 V13 trace/domain metrics are not closing")

    v14 = _exact_publication_mapping(
        metrics.get("V14"),
        label="canonical V14 metrics",
        keys={
            "contrast_ratios", "approved_relative_error",
            "display_first_mutant_relative_error",
            "maximum_abs_approved_relative_error", "trace_binding",
        },
    )
    if v14 != _json_value(m2_production_module._contrast_measurements()):  # noqa: SLF001
        raise RuntimeError("canonical M2 V14 metrics differ from closed form")

    v15 = _exact_publication_mapping(
        metrics.get("V15"),
        label="canonical V15 metrics",
        keys={
            "maximum_out_of_range_fraction", "tolerance", "fields", "mapping",
            "m1_report_sha256", "m1_baseline_out_of_range_fraction",
        },
    )
    v15_fields = _exact_publication_mapping(
        v15.get("fields"),
        label="canonical V15 fields",
        keys={"shared_day_fields", "ir_bt_kelvin", "solar_geometry_and_day_weight"},
    )
    for name, raw_row in v15_fields.items():
        row = _exact_publication_mapping(
            raw_row,
            label=f"canonical V15 field {name}",
            keys={"out_of_range_fraction", "comparisons"},
        )
        if row.get("out_of_range_fraction") != 0.0 or (
            type(row.get("comparisons")) is not int or row["comparisons"] <= 0
        ):
            raise RuntimeError("canonical M2 V15 field range proof is not closing")
    if (
        v15.get("maximum_out_of_range_fraction") != 0.0
        or v15.get("tolerance") != 2.0e-6
        or v15.get("mapping")
        != "captured_Agg_impulse_support_and_positive_source_coverage"
        or v15.get("m1_report_sha256") != baseline.get("report_sha256")
        or v15.get("m1_baseline_out_of_range_fraction")
        != baseline.get("v15_out_of_range_fraction")
        or v15.get("m1_baseline_out_of_range_fraction") != 0.0
    ):
        raise RuntimeError("canonical M2 V15 local-support proof is not closing")

    v16 = _exact_publication_mapping(
        metrics.get("V16"),
        label="canonical V16 metrics",
        keys={
            "m1_report_sha256", "m1_git_head", "m1_nearest_ringing_dn",
            "m1_hanning_ringing_dn", "candidate", "candidate_edge_index",
            "candidate_geometry_identity_sha256", "candidate_source_pixels",
            "candidate_delivered_pixels", "candidate_x_scale",
            "candidate_x_translation", "procedure", "passed",
        },
    )
    candidate_ringing = _exact_publication_mapping(
        v16.get("candidate"),
        label="canonical V16 candidate ringing",
        keys={"step_dn", "overshoot_dn", "undershoot_dn", "ringing_dn"},
    )
    if (
        v16.get("passed") is not True
        or v16.get("m1_report_sha256") != baseline.get("report_sha256")
        or v16.get("m1_git_head") != baseline.get("git_head")
        or v16.get("m1_nearest_ringing_dn")
        != baseline.get("v16_nearest_ringing_dn")
        or v16.get("m1_hanning_ringing_dn")
        != baseline.get("v16_hanning_ringing_dn")
        or type(candidate_ringing.get("ringing_dn")) not in {int, float}
        or float(candidate_ringing["ringing_dn"])
        > float(v16["m1_nearest_ringing_dn"])
        or any(
            type(v16.get(key)) is not int or v16[key] <= 0
            for key in (
                "candidate_edge_index", "candidate_source_pixels",
                "candidate_delivered_pixels",
            )
        )
        or v16.get("procedure")
        != (
            "fixed_seed_46.1_DN_plateau_fixture_inverse_CIRA_to_physical_"
            "then_measured_Hanning_response_then_CIRA_and_local_plateaus"
        )
    ):
        raise RuntimeError("canonical M2 V16 ringing proof is not closing")
    spectral_plot = m2_gallery_module._spectral_plot(frequency, profiles, ratios)
    if extras["13_canonical_spectral_comparison.png"] != spectral_plot:
        raise RuntimeError("canonical M2 spectral plot differs from its CSV/native rasters")

    native_comparison = _exact_publication_mapping(
        gallery_manifest.get("native_comparison_sheet"),
        label="gallery native comparison sheet",
        keys={"artifact", "scale_percent", "panel_bounds", "panels_are_pixel_identical"},
    )
    signed_error = _exact_publication_mapping(
        gallery_manifest.get("signed_error"),
        label="gallery signed-error declaration",
        keys={
            "quantity",
            "rec601_coefficients_rgb",
            "raw_values_artifact",
            "raw_values_dtype",
            "raw_values_shape",
            "raw_values_order",
            "shared_scale_percentile",
            "shared_symmetric_limit_dn",
            "shared_scale_definition",
            "comparison_sheet",
        },
    )
    error_comparison = _exact_publication_mapping(
        signed_error.get("comparison_sheet"),
        label="gallery signed-error comparison sheet",
        keys={"artifact", "scale_percent", "panel_bounds", "panels_are_pixel_identical"},
    )
    spectral = _exact_publication_mapping(
        gallery_manifest.get("spectral"),
        label="gallery spectral declaration",
        keys={
            "plot",
            "csv",
            "procedure",
            "bins",
            "cutoff_cycles_per_delivered_pixel",
            "ratios_to_exact_area_reference",
        },
    )
    if (
        native_comparison
        != {
            "artifact": "08_native_comparison_sheet_100pct.png",
            "scale_percent": 100,
            "panel_bounds": main_bounds,
            "panels_are_pixel_identical": True,
        }
        or signed_error.get("quantity")
        != "candidate minus exact-area reference, signed Rec.601 luma DN"
        or signed_error.get("rec601_coefficients_rgb") != [0.299, 0.587, 0.114]
        or signed_error.get("raw_values_artifact") != error_npy_name
        or signed_error.get("raw_values_dtype") != "float32"
        or signed_error.get("raw_values_shape") != list(error_stack.shape)
        or signed_error.get("raw_values_order")
        != list(m2_gallery_module.ERROR_ORDER)
        or signed_error.get("shared_scale_percentile")
        != m2_gallery_module.ERROR_SCALE_PERCENTILE
        or signed_error.get("shared_symmetric_limit_dn") != shared_scale
        or signed_error.get("shared_scale_definition")
        != (
            "max(1 DN, p99.9 absolute candidate error, p99.9 absolute old-nearest "
            "error, p99.9 absolute display-first error)"
        )
        or error_comparison
        != {
            "artifact": "09_signed_error_sheet_shared_scale_100pct.png",
            "scale_percent": 100,
            "panel_bounds": error_bounds,
            "panels_are_pixel_identical": True,
        }
        or spectral
        != {
            "plot": "13_canonical_spectral_comparison.png",
            "csv": "12_canonical_spectral_comparison.csv",
            "procedure": "Rec.601_square_crop_2D_Hann_64_radial_bins_v1",
            "bins": m2_gallery_module.SPECTRAL_BINS,
            "cutoff_cycles_per_delivered_pixel": m2_gallery_module.SPECTRAL_CUTOFF,
            "ratios_to_exact_area_reference": ratios,
        }
    ):
        raise RuntimeError("canonical M2 gallery scientific declarations differ")

    primary_png = _publication_mapping(
        canonical.get("primary_png"), label="canonical production PNG identity"
    )
    production_png = extras["14_production_renderer.png"]
    production_array = _decode_publication_png(
        production_png,
        label="production renderer payload",
        opaque_rgba=True,
    )
    if (
        primary_png.get("bytes") != len(production_png)
        or primary_png.get("sha256") != sha256_bytes(production_png)
        or canonical.get("primary_capture", {}).get("png_bytes") != len(production_png)
        or canonical.get("primary_capture", {}).get("png_sha256")
        != sha256_bytes(production_png)
    ):
        raise RuntimeError("canonical M2 production renderer PNG identity differs")

    canonical_summary_sha256 = sha256_bytes(_strict_json_bytes(canonical))
    repeatability_sha256 = sha256_bytes(_strict_json_bytes(repeatability))
    sequence_sha256 = sha256_bytes(_strict_json_bytes(sequence))
    limitation_text = "\n".join(
        f"- {item}" for item in M2_GALLERY_REQUIRED_LIMITATIONS
    )
    expected_review = (
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
    expected_readme = (
        "# M2 primary-imagery-layer review payloads\n\n"
        "Andrew verdict: PENDING. This gallery is not itself a stage-advancement "
        "decision and cannot substitute for the canonical gate report.\n\n"
        f"Scene: `{canonical['scene_id']}`\n\n"
        f"Native delivered raster: `{width} x {height}` pixels\n\n"
        f"Canonical evidence summary SHA-256: `{canonical_summary_sha256}`\n\n"
        "Repeatability summary: supplied; "
        f"passed={repeatability['passed']}; summary SHA-256={repeatability_sha256}.\n\n"
        "Timestamped sequence summary: supplied; "
        f"outcome={sequence['outcome']}; summary SHA-256={sequence_sha256}.\n\n"
        "Every standalone scientific raster is lossless and native scale. The two "
        "comparison sheets add labels only outside the embedded raster bounds. The "
        "production PNG is copied byte-for-byte rather than decoded and re-encoded.\n\n"
        "Exact limitations:\n\n"
        f"{limitation_text}\n"
    ).encode("utf-8")
    if extras["REVIEW_INSTRUCTIONS.md"] != expected_review:
        raise RuntimeError("canonical M2 review instructions are not reproducible")
    if extras["README.md"] != expected_readme:
        raise RuntimeError("canonical M2 gallery README is not reproducible")

    expected_artifact_semantics: dict[
        str, tuple[str, int | None, int | None, tuple[str, ...]]
    ] = {
        "01_m2_candidate_native.png": (
            "image/png", width, height, ("candidate_floor",)
        ),
        "02_exact_area_reference_native.png": (
            "image/png", width, height, ("exact_area_reference_floor",)
        ),
        "03_old_nearest_mutant_native.png": (
            "image/png", width, height, ("old_nearest_mutant_floor",)
        ),
        "04_display_first_mutant_native.png": (
            "image/png", width, height, ("display_first_mutant_floor",)
        ),
        "05_candidate_signed_rec601_error_native.png": (
            "image/png", width, height,
            ("candidate_floor", "exact_area_reference_floor"),
        ),
        "06_old_nearest_mutant_signed_rec601_error_native.png": (
            "image/png", width, height,
            ("old_nearest_mutant_floor", "exact_area_reference_floor"),
        ),
        "07_display_first_mutant_signed_rec601_error_native.png": (
            "image/png", width, height,
            ("display_first_mutant_floor", "exact_area_reference_floor"),
        ),
        "08_native_comparison_sheet_100pct.png": (
            "image/png", main_sheet_size[0], main_sheet_size[1],
            (
                "candidate_floor", "exact_area_reference_floor",
                "old_nearest_mutant_floor", "display_first_mutant_floor",
            ),
        ),
        "09_signed_error_sheet_shared_scale_100pct.png": (
            "image/png", error_sheet_size[0], error_sheet_size[1],
            (
                "candidate_floor", "old_nearest_mutant_floor",
                "display_first_mutant_floor", "exact_area_reference_floor",
            ),
        ),
        error_npy_name: (
            "application/x-npy", width, height,
            (
                "candidate_floor", "old_nearest_mutant_floor",
                "display_first_mutant_floor", "exact_area_reference_floor",
            ),
        ),
        "11_signed_rec601_error_summary.csv": (
            "text/csv", None, None,
            (
                "candidate_floor", "old_nearest_mutant_floor",
                "display_first_mutant_floor", "exact_area_reference_floor",
            ),
        ),
        "12_canonical_spectral_comparison.csv": (
            "text/csv", None, None,
            (
                "candidate_floor", "exact_area_reference_floor",
                "old_nearest_mutant_floor", "display_first_mutant_floor",
            ),
        ),
        "13_canonical_spectral_comparison.png": (
            "image/png", 1200, 720,
            (
                "candidate_floor", "exact_area_reference_floor",
                "old_nearest_mutant_floor", "display_first_mutant_floor",
            ),
        ),
        "14_production_renderer.png": (
            "image/png", production_array.shape[1], production_array.shape[0],
            ("primary_png", "candidate_floor"),
        ),
        "README.md": ("text/markdown", None, None, ("evidence_summary",)),
        "REVIEW_INSTRUCTIONS.md": (
            "text/markdown", None, None, ("evidence_summary",)
        ),
    }
    expected_descriptions = {
        "01_m2_candidate_native.png": (
            "Registered M2 delivered map raster at exact native inspection scale."
        ),
        "02_exact_area_reference_native.png": (
            "Captured-affine exact-area physical reference at exact native "
            "inspection scale."
        ),
        "03_old_nearest_mutant_native.png": (
            "Rejected old nearest-neighbour behavior at exact native inspection scale."
        ),
        "04_display_first_mutant_native.png": (
            "Rejected display-domain-first behavior at exact native inspection scale."
        ),
        "05_candidate_signed_rec601_error_native.png": (
            "Signed Rec.601 luma error (candidate minus exact-area reference); "
            f"red positive, blue negative, shared symmetric scale {shared_scale:.6g} DN."
        ),
        "06_old_nearest_mutant_signed_rec601_error_native.png": (
            "Signed Rec.601 luma error (old_nearest_mutant minus exact-area reference); "
            f"red positive, blue negative, shared symmetric scale {shared_scale:.6g} DN."
        ),
        "07_display_first_mutant_signed_rec601_error_native.png": (
            "Signed Rec.601 luma error (display_first_mutant minus exact-area reference); "
            f"red positive, blue negative, shared symmetric scale {shared_scale:.6g} DN."
        ),
        "08_native_comparison_sheet_100pct.png": (
            "Four labeled native rasters embedded pixel-for-pixel at 100% scale."
        ),
        "09_signed_error_sheet_shared_scale_100pct.png": (
            "Three signed-error maps embedded pixel-for-pixel with one shared scale."
        ),
        error_npy_name: (
            "Raw float32 signed Rec.601 DN errors with order candidate, old-nearest, "
            "display-first and shape [3,H,W]."
        ),
        "11_signed_rec601_error_summary.csv": (
            "Machine-readable signed Rec.601 error statistics and canonical shared scale."
        ),
        "12_canonical_spectral_comparison.csv": (
            "Canonical Rec.601, square-crop, 2-D-Hann, 64-bin radial spectra."
        ),
        "13_canonical_spectral_comparison.png": (
            "Plot of the exact canonical spectral profiles stored in the companion CSV."
        ),
        "14_production_renderer.png": (
            "Exact PNG bytes returned by the bound primary M2 production renderer."
        ),
        "README.md": (
            "Gallery index, source identity, optional evidence status, and exact limits."
        ),
        "REVIEW_INSTRUCTIONS.md": (
            "Immutable append-only review procedure; Andrew remains PENDING."
        ),
    }
    binding_hashes = {
        **array_bindings,
        "primary_png": primary_png["sha256"],
        "evidence_summary": canonical_summary_sha256,
    }
    for name, (media_type, expected_width, expected_height, bindings) in (
        expected_artifact_semantics.items()
    ):
        row = _publication_mapping(gallery_rows.get(name), label=f"gallery row {name}")
        if (
            metadata.get(name) != row
            or row.get("media_type") != media_type
            or row.get("width") != expected_width
            or row.get("height") != expected_height
            or row.get("description") != expected_descriptions[name]
            or row.get("source_bindings") != list(bindings)
            or row.get("source_binding_sha256")
            != {binding: binding_hashes[binding] for binding in bindings}
        ):
            raise RuntimeError(
                f"canonical M2 gallery artifact semantics differ: {name}"
            )


def _validate_canonical_publication_semantics(
    report: Mapping[str, Any],
    extras: Mapping[str, bytes],
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove a canonical classification from mutually bound report structures."""

    summary = _publication_mapping(report.get("summary"), label="summary")
    expected_report_keys = {
        "schema",
        "command",
        "summary",
        "claim_boundary",
        "limitations",
        "research_inputs",
        "repository",
        "source",
        "dependencies",
        "harness_source_sha256",
        "oracle",
        "production_evaluation",
        "v30_capture",
        "gallery_artifacts",
        "runner_image",
        "invocation",
        "evidence_payload_sha256",
    }
    if set(report) != expected_report_keys:
        raise RuntimeError("canonical M2 report top-level schema is not exact")
    _reject_unknown_stage_claim_keys(report)
    if set(summary) != {
        "contract_passed",
        "stage_advancement_passed",
        "canonical_blockers",
        "mutation_count",
        "mutation_kill_count",
    }:
        raise RuntimeError("canonical M2 report summary schema is not exact")
    if (
        report.get("command") != "canonical-production"
        or summary.get("contract_passed") is not True
        or summary.get("stage_advancement_passed") is not True
        or summary.get("canonical_blockers") != []
    ):
        raise RuntimeError(
            "canonical M2 publication requires a passing canonical-production summary"
        )
    if report.get("limitations") != list(PRODUCTION_LIMITATIONS):
        raise RuntimeError(
            "canonical M2 report does not declare the exact bounded authority and M1 fallbacks"
        )
    research = _exact_publication_mapping(
        report.get("research_inputs"),
        label="research-input binding",
        keys={"root", "validation_gates_csv", "reference_pixels_json"},
    )
    if type(research.get("root")) is not str or not Path(research["root"]).is_absolute():
        raise RuntimeError("canonical M2 research root is not absolute")
    for name in ("validation_gates_csv", "reference_pixels_json"):
        row = _exact_publication_mapping(
            research.get(name),
            label=f"research input {name}",
            keys={"path", "sha256"},
        )
        if (
            type(row.get("path")) is not str
            or not Path(row["path"]).is_absolute()
            or not _valid_publication_sha256(row.get("sha256"))
        ):
            raise RuntimeError(f"canonical M2 research input is invalid: {name}")
    try:
        verified_research = verify_research_root(research["root"])
        live_research = _json_value(verified_research.as_dict())
    except (OSError, ResearchIntegrityError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "canonical M2 research inputs cannot be freshly reverified"
        ) from exc
    if live_research != research:
        raise RuntimeError("canonical M2 research input identity differs from live files")
    invocation = _publication_mapping(report.get("invocation"), label="invocation")
    if (
        report.get("runner_image") != M2_RUNNER_IMAGE_PROVENANCE
        or invocation
        != {
            "command": "canonical-production",
            "runner_image_id": M2_RUNNER_IMAGE_ID,
            "runner_image_identity_source": "external_operator_declared",
            "container_self_attested": False,
        }
    ):
        raise RuntimeError(
            "canonical M2 runner image must be the exact external/operator-declared immutable ID"
        )

    oracle = _exact_publication_mapping(
        report.get("oracle"),
        label="oracle",
        keys={
            "schema",
            "contract_passed",
            "stage_advancement_passed",
            "canonical_blockers",
            "gate_registry_issues",
            "mutation_registry_issues",
            "gates",
            "mutations",
            "array_identities",
            "product_color_protection",
            "claim_boundary",
        },
    )
    if (
        oracle.get("schema") != M2_ORACLE_SCHEMA
        or oracle.get("contract_passed") is not True
        or oracle.get("stage_advancement_passed") is not True
        or oracle.get("canonical_blockers") != []
        or oracle.get("gate_registry_issues") != []
        or oracle.get("mutation_registry_issues") != []
    ):
        raise RuntimeError("canonical M2 oracle semantics are not closing")
    array_identities = _publication_list(
        oracle.get("array_identities"), label="oracle array identities"
    )
    observed_array_names: list[str] = []
    for raw_identity in array_identities:
        identity = _exact_publication_mapping(
            raw_identity,
            label="oracle array identity",
            keys={"name", "shape", "dtype", "sha256"},
        )
        observed_array_names.append(identity.get("name"))
        shape = _publication_list(
            identity.get("shape"), label="oracle array-identity shape"
        )
        if (
            type(identity.get("name")) is not str
            or not identity["name"]
            or not shape
            or any(type(value) is not int or value <= 0 for value in shape)
            or type(identity.get("dtype")) is not str
            or not identity["dtype"]
            or not _valid_publication_sha256(identity.get("sha256"))
        ):
            raise RuntimeError("canonical M2 oracle array identity is invalid")
    if not array_identities or len(observed_array_names) != len(set(observed_array_names)):
        raise RuntimeError("canonical M2 oracle array identities are absent or duplicated")
    product_protection = _exact_publication_mapping(
        oracle.get("product_color_protection"),
        label="oracle product-color protection",
        keys={
            "v07_passed",
            "v07_mismatches",
            "v08_passed",
            "v08_failures",
            "v08_declared_quantizer",
            "v08_delivered_hex",
            "cira_stretch_source_sha256",
            "policy_rollback_binding",
        },
    )
    policy_rollback_binding = _exact_publication_mapping(
        product_protection.get("policy_rollback_binding"),
        label="oracle policy rollback binding",
        keys={
            "passed",
            "issues",
            "live_dispatcher_call_count",
            "live_dispatchers_forward_request_policy",
            "m0_rollback_registered",
            "m1_rollback_registered",
            "m1_literal_hanning_sites",
            "m2_registered_default",
        },
    )
    if (
        product_protection.get("v07_passed") is not True
        or product_protection.get("v07_mismatches") != {}
        or product_protection.get("v08_passed") is not True
        or product_protection.get("v08_failures") != []
        or product_protection.get("v08_declared_quantizer") != "production_floor"
        or not _valid_publication_sha256(
            product_protection.get("cira_stretch_source_sha256")
        )
        or policy_rollback_binding != _json_value(
            m2_validation.m2_policy_rollback_binding()
        )
        or policy_rollback_binding.get("passed") is not True
    ):
        raise RuntimeError("canonical M2 product-color protection is not closing")
    if report.get("claim_boundary") != oracle.get("claim_boundary"):
        raise RuntimeError("top-level and oracle claim boundaries differ")
    expected_claim_boundary = {
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
            "black_endpoint": 0.0223,
            "white_endpoint": 2.1730825067739694,
        },
        "day_weight_policy": "legacy_88_to_95_no_survival_gate",
        "glint": "post_CIRA_pointwise_M8_compatibility_exception",
        "forbidden_pass_gate_ids": sorted(FORBIDDEN_M2_CLOSURE_GATE_IDS),
    }
    if report.get("claim_boundary") != expected_claim_boundary:
        raise RuntimeError("canonical M2 claim boundary is not the registered bounded claim")

    gate_rows = _publication_list(oracle.get("gates"), label="oracle gates")
    gates: dict[str, dict[str, Any]] = {}
    for raw_row in gate_rows:
        row = _exact_publication_mapping(
            raw_row,
            label="oracle gate",
            keys={
                "gate_id",
                "outcome",
                "canonical_closure",
                "evidence_scope",
                "threshold",
                "detail",
                "measurements",
                "canonical_procedure",
                "canonical_source_document",
            },
        )
        _publication_mapping(
            row.get("measurements"), label="oracle gate measurements", nonempty=False
        )
        gate_id = row.get("gate_id")
        if type(gate_id) is not str or gate_id in gates:
            raise RuntimeError("canonical M2 oracle gate identities are invalid")
        gates[gate_id] = row
    if set(gates) != M2_ALL_GATE_IDS or len(gate_rows) != len(M2_ALL_GATE_IDS):
        raise RuntimeError("canonical M2 oracle gate registry is not exact")
    if any(
        gates[gate_id].get("outcome") != "PASS"
        or gates[gate_id].get("canonical_closure") is not True
        for gate_id in M2_REQUIRED_GATE_IDS
    ):
        raise RuntimeError("canonical M2 required gates are not all closing passes")
    if not FORBIDDEN_M2_CLOSURE_GATE_IDS <= gates.keys() or any(
        gates[gate_id].get("outcome") != "UNMEASURED"
        or gates[gate_id].get("canonical_closure") is not False
        for gate_id in FORBIDDEN_M2_CLOSURE_GATE_IDS
    ):
        raise RuntimeError("M2 publication must leave V44, V45, and V46 unclosed")
    if not M5_DEFERRED_GATE_IDS <= gates.keys() or any(
        gates[gate_id].get("outcome") != "DEFERRED"
        or gates[gate_id].get("canonical_closure") is not False
        for gate_id in M5_DEFERRED_GATE_IDS
    ):
        raise RuntimeError("M2 publication must leave V21 through V24 deferred")
    research_gates = getattr(verified_research, "gates", None)
    expected_research_gate_ids = {gate.gate_id for gate in GATES}
    if (
        not isinstance(research_gates, Mapping)
        or set(research_gates) != expected_research_gate_ids
        or not M2_ALL_GATE_IDS <= expected_research_gate_ids
    ):
        raise RuntimeError("canonical M2 live research gate registry is not exact")
    for gate_id, gate in gates.items():
        research_gate = research_gates[gate_id]
        if (
            gate.get("threshold") != getattr(research_gate, "threshold", None)
            or gate.get("canonical_procedure")
            != getattr(research_gate, "procedure", None)
            or gate.get("canonical_source_document")
            != getattr(research_gate, "source_document", None)
        ):
            raise RuntimeError(
                f"canonical M2 {gate_id} text differs from bound research inputs"
            )
    for gate_id in M2_REQUIRED_GATE_IDS:
        if gates[gate_id].get("evidence_scope") != "canonical_bound_production_renderer":
            raise RuntimeError(
                f"canonical M2 {gate_id} evidence scope is not production-bound"
            )
    for gate_id in M5_DEFERRED_GATE_IDS:
        if (
            gates[gate_id].get("evidence_scope") != "explicit_M5_residual"
            or gates[gate_id].get("detail")
            != "Surviving encoded-pixel resizes prevent M2 closure."
            or gates[gate_id].get("measurements")
            != {"surviving_resizes": list(M5_SURVIVING_ENCODED_RESIZES)}
        ):
            raise RuntimeError(f"canonical M2 {gate_id} deferred declaration differs")
    for gate_id in FORBIDDEN_M2_CLOSURE_GATE_IDS:
        if (
            gates[gate_id].get("evidence_scope") != "explicitly_unclosed"
            or gates[gate_id].get("detail")
            != "M2 has no evidence that can close this coverage gate."
            or gates[gate_id].get("measurements") != {}
        ):
            raise RuntimeError(f"canonical M2 {gate_id} unmeasured declaration differs")

    mutations = _publication_list(
        oracle.get("mutations"), label="oracle mutations"
    )
    mutation_rows = [
        _exact_publication_mapping(
            row,
            label="oracle mutation",
            keys={"mutation_id", "killed", "discriminator", "measurements"},
        )
        for row in mutations
    ]
    for row in mutation_rows:
        _publication_mapping(
            row.get("measurements"),
            label="oracle mutation measurements",
            nonempty=False,
        )
    mutation_ids = [row.get("mutation_id") for row in mutation_rows]
    if (
        len(mutation_ids) != len(set(mutation_ids))
        or set(mutation_ids) != M2_CANONICAL_MUTATION_IDS
        or any(row.get("killed") is not True for row in mutation_rows)
    ):
        raise RuntimeError("canonical M2 oracle has an absent or surviving mutation")
    if (
        summary.get("mutation_count") != len(mutations)
        or summary.get("mutation_kill_count") != len(mutations)
    ):
        raise RuntimeError("canonical M2 summary mutation counts differ from oracle")

    production = _exact_publication_mapping(
        report.get("production_evaluation"),
        label="production evaluation",
        keys={
            "schema",
            "stage_advancement_passed",
            "binding_issues",
            "canonical",
            "repeatability",
            "sequence",
            "oracle",
            "input_bindings",
            "rayleigh_startup",
        },
    )
    if (
        production.get("schema") != M2_PRODUCTION_SCHEMA
        or production.get("stage_advancement_passed") is not True
        or production.get("binding_issues") != []
        or production.get("oracle") != oracle
    ):
        raise RuntimeError("canonical M2 production evaluation is not bound and closing")
    canonical = _exact_publication_mapping(
        production.get("canonical"),
        label="canonical production capture",
        keys={
            "schema",
            "scene_id",
            "bbox",
            "objects",
            "primary_capture",
            "limb_capture",
            "primary_png",
            "arrays",
            "metrics",
            "m1_baseline",
        },
    )
    if canonical.get("schema") != M2_PRODUCTION_SCHEMA:
        raise RuntimeError("canonical M2 production capture schema is unregistered")
    canonical_arrays = _exact_publication_mapping(
        canonical.get("arrays"),
        label="canonical production array registry",
        keys={
            "candidate_float",
            "candidate_floor",
            "exact_area_reference_float",
            "exact_area_reference_floor",
            "old_nearest_mutant_floor",
            "display_first_mutant_float",
            "display_first_mutant_floor",
        },
    )
    for name, raw_record in canonical_arrays.items():
        record = _exact_publication_mapping(
            raw_record,
            label=f"canonical production array {name}",
            keys={"name", "shape", "dtype", "sha256"},
        )
        shape = _publication_list(
            record.get("shape"), label=f"canonical production array shape {name}"
        )
        expected_dtype = "<f4" if name.endswith("_float") else "|u1"
        if (
            record.get("name") != name
            or len(shape) != 3
            or shape[-1] != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or record.get("dtype") != expected_dtype
            or not _valid_publication_sha256(record.get("sha256"))
        ):
            raise RuntimeError(f"canonical M2 production array identity is invalid: {name}")
    if len({tuple(row["shape"]) for row in canonical_arrays.values()}) != 1:
        raise RuntimeError("canonical M2 production array shapes differ")
    _exact_publication_mapping(
        canonical.get("primary_png"),
        label="canonical production PNG identity",
        keys={"bytes", "sha256"},
    )
    expected_capture_keys = {
        "schema",
        "bbox",
        "dpi",
        "title_text",
        "policy_version",
        "png_bytes",
        "png_sha256",
        "geometry",
        "delivered_float_rgb",
        "delivered_floor_rgb",
        "figure_image_rgb",
        "identity_draw",
        "resample_calls",
        "pointwise_calls",
        "operations",
        "endpoint_binding",
        "trace_issues",
        "trace_identity_sha256",
    }
    primary_capture = _exact_publication_mapping(
        canonical.get("primary_capture"),
        label="canonical primary renderer capture",
        keys=expected_capture_keys,
    )
    limb_capture = canonical.get("limb_capture")
    if limb_capture is not None:
        _exact_publication_mapping(
            limb_capture,
            label="canonical limb renderer capture",
            keys=expected_capture_keys,
        )
    if primary_capture.get("trace_issues") != []:
        raise RuntimeError("canonical primary renderer capture has trace issues")
    for capture_array, canonical_name in (
        ("delivered_float_rgb", "candidate_float"),
        ("delivered_floor_rgb", "candidate_floor"),
        ("figure_image_rgb", "candidate_floor"),
    ):
        record = _exact_publication_mapping(
            primary_capture.get(capture_array),
            label=f"canonical primary capture array {capture_array}",
            keys={"name", "shape", "dtype", "sha256"},
        )
        if (
            record.get("shape") != canonical_arrays[canonical_name].get("shape")
            or record.get("dtype") != canonical_arrays[canonical_name].get("dtype")
            or record.get("sha256") != canonical_arrays[canonical_name].get("sha256")
        ):
            raise RuntimeError(
                f"canonical primary capture array differs: {capture_array}"
            )
    canonical_metrics = _exact_publication_mapping(
        canonical.get("metrics"),
        label="canonical production metrics",
        keys={
            "V01",
            "V02",
            "V03",
            "V07",
            "V08",
            "V13",
            "V14",
            "V15",
            "V16",
            "V41",
            "M0_M1_ROLLBACK",
            "mutation_discriminators",
            "limitations",
        },
    )
    for name in canonical_metrics:
        if name != "limitations":
            _publication_mapping(
                canonical_metrics[name],
                label=f"canonical production metric {name}",
                nonempty=False,
            )
    v41 = _exact_publication_mapping(
        canonical_metrics.get("V41"),
        label="canonical V41 measurement",
        keys={
            "scene_bbox",
            "source_geometric_boundary_edges",
            "source_has_valid_invalid_boundary",
            "captured_modality_union_unsupported_pixels",
            "delivery_support_escape_pixels",
            "exact_formula_max_abs_residual",
            "exact_formula_array_identity",
            "registered_coverage_tau",
            "registered_tau_matches_oracle",
            "support_proof",
            "numeric_crosscheck",
            "composite_domain",
            "background_contract",
            "background_contract_passed",
            "mutation_discriminators",
            "all_mutations_killed",
            "passed",
        },
    )
    numeric_crosscheck = _exact_publication_mapping(
        v41.get("numeric_crosscheck"),
        label="canonical V41 numeric crosscheck",
        keys={
            "procedure",
            "day_coverage_max_abs_error",
            "ir_coverage_max_abs_error",
            "absolute_tolerance",
            "tolerance_justification",
            "passed",
        },
    )
    v41_mutations = _exact_publication_mapping(
        v41.get("mutation_discriminators"),
        label="canonical V41 mutation discriminators",
        keys={
            "output_rgb_finite_proxy",
            "nearest_source_mask",
            "missing_geom_visibility_boundary",
            "delivery_outside_captured_modality_union",
            "over_tight_float64_proxy_tolerance",
        },
    )
    simple_v41_mutations = {
        "output_rgb_finite_proxy": {"difference_pixels", "killed"},
        "nearest_source_mask": {"difference_pixels", "killed"},
        "missing_geom_visibility_boundary": {"source_boundary_edges", "killed"},
        "delivery_outside_captured_modality_union": {
            "injectable_unsupported_pixels",
            "observed_escape_pixels",
            "killed",
        },
        "over_tight_float64_proxy_tolerance": {
            "rejected_tolerance",
            "observed_max_abs_residual",
            "registered_tolerance",
            "killed",
        },
    }
    normalized_v41_mutations = {
        name: _exact_publication_mapping(
            v41_mutations.get(name),
            label=f"canonical V41 mutation {name}",
            keys=keys,
        )
        for name, keys in simple_v41_mutations.items()
    }
    over_tight = normalized_v41_mutations["over_tight_float64_proxy_tolerance"]
    if (
        v41 != gates["V41"].get("measurements")
        or v41.get("passed") is not True
        or v41.get("source_has_valid_invalid_boundary") is not True
        or type(v41.get("source_geometric_boundary_edges")) is not int
        or v41["source_geometric_boundary_edges"] <= 0
        or type(v41.get("captured_modality_union_unsupported_pixels")) is not int
        or v41["captured_modality_union_unsupported_pixels"] <= 0
        or v41.get("delivery_support_escape_pixels") != 0
        or v41.get("exact_formula_max_abs_residual") != 0.0
        or v41.get("exact_formula_array_identity") is not True
        # The product threshold is stored and applied as float32.  Bind the
        # report to that exact delivered representation, rather than to the
        # unequal Python-float spelling of the same registered oracle value.
        or v41.get("registered_coverage_tau")
        != float(np.float32(m2_validation.COVERAGE_TAU))
        or v41.get("registered_tau_matches_oracle") is not True
        or v41.get("support_proof")
        != (
            "captured_delivered_day_ir_coverage_union_plus_exact_"
            "day_weight_finalizer_formula"
        )
        or numeric_crosscheck.get("procedure")
        != (
            "float64_sparse_measured_Agg_response_replay_only;_"
            "never_used_to_invent_support"
        )
        or numeric_crosscheck.get("absolute_tolerance") != 1.0e-4
        or numeric_crosscheck.get("passed") is not True
        or any(
            type(numeric_crosscheck.get(name)) not in {int, float}
            or not math.isfinite(float(numeric_crosscheck[name]))
            or float(numeric_crosscheck[name]) < 0.0
            or float(numeric_crosscheck[name]) > 1.0e-4
            for name in ("day_coverage_max_abs_error", "ir_coverage_max_abs_error")
        )
        or v41.get("composite_domain") != "linear_premultiplied"
        or v41.get("background_contract") != "opaque_pre_cira"
        or v41.get("background_contract_passed") is not True
        or v41.get("all_mutations_killed") is not True
        or any(row.get("killed") is not True for row in normalized_v41_mutations.values())
        or not (
            type(
                normalized_v41_mutations["output_rgb_finite_proxy"].get(
                    "difference_pixels"
                )
            )
            is int
            and normalized_v41_mutations["output_rgb_finite_proxy"][
                "difference_pixels"
            ]
            > 0
        )
        or not (
            type(
                normalized_v41_mutations["nearest_source_mask"].get(
                    "difference_pixels"
                )
            )
            is int
            and normalized_v41_mutations["nearest_source_mask"][
                "difference_pixels"
            ]
            > 0
        )
        or not (
            type(
                normalized_v41_mutations[
                    "missing_geom_visibility_boundary"
                ].get("source_boundary_edges")
            )
            is int
            and normalized_v41_mutations["missing_geom_visibility_boundary"][
                "source_boundary_edges"
            ]
            > 0
        )
        or not (
            type(
                normalized_v41_mutations[
                    "delivery_outside_captured_modality_union"
                ].get("injectable_unsupported_pixels")
            )
            is int
            and normalized_v41_mutations[
                "delivery_outside_captured_modality_union"
            ]["injectable_unsupported_pixels"]
            > 0
        )
        or normalized_v41_mutations[
            "delivery_outside_captured_modality_union"
        ].get("observed_escape_pixels")
        != 0
        or over_tight.get("rejected_tolerance") != 2.0e-6
        or over_tight.get("registered_tolerance") != 1.0e-4
        or type(over_tight.get("observed_max_abs_residual")) not in {int, float}
        or not (
            2.0e-6
            < float(over_tight["observed_max_abs_residual"])
            <= 1.0e-4
        )
    ):
        raise RuntimeError("canonical M2 V41 support proof is invalid or unbound")

    repeatability = _exact_publication_mapping(
        production.get("repeatability"),
        label="V25 repeatability",
        keys={"passed", "hashes", "output_bytes", "manifests", "hash_provenance"},
    )
    repeatability_hashes = _publication_list(
        repeatability.get("hashes"), label="V25 hashes"
    )
    repeatability_bytes = _publication_list(
        repeatability.get("output_bytes"), label="V25 output-byte counts"
    )
    repeatability_manifests = _publication_list(
        repeatability.get("manifests"), label="V25 process manifests"
    )
    normalized_manifests = []
    for raw_manifest in repeatability_manifests:
        normalized_manifests.append(_exact_publication_mapping(
            raw_manifest,
            label="V25 process manifest",
            keys={
                "pid",
                "parent_pid",
                "process_start_ticks",
                "start_method",
                "python",
                "platform",
                "numpy",
                "matplotlib",
                "pillow",
                "policy_version",
                "request_identity_sha256",
                "output_bytes",
                "child_output_sha256",
                "parent_recomputed_output_sha256",
            },
        ))
    dependency_versions = _publication_mapping(
        report.get("dependencies", {}).get("distributions"),
        label="dependency distribution versions",
    )
    request_identities = {
        row.get("request_identity_sha256") for row in normalized_manifests
    }
    canonical_primary_png = _publication_mapping(
        canonical.get("primary_png"), label="canonical V25 primary PNG"
    )
    if (
        repeatability.get("passed") is not True
        or repeatability.get("hash_provenance")
        != "parent_recomputed_from_child_renderer_bytes"
        or len(repeatability_hashes) != 3
        or len(repeatability_bytes) != 3
        or len(repeatability_manifests) != 3
        or len(set(repeatability_hashes)) != 1
        or any(
            value != canonical_primary_png.get("sha256")
            for value in repeatability_hashes
        )
        or any(
            value != canonical_primary_png.get("bytes")
            for value in repeatability_bytes
        )
        or any(not _valid_publication_sha256(value) for value in repeatability_hashes)
        or any(type(value) is not int or value <= 0 for value in repeatability_bytes)
        or len({row.get("pid") for row in normalized_manifests}) != 3
        or len(request_identities) != 1
        or any(not _valid_publication_sha256(value) for value in request_identities)
        or any(
            type(row.get("pid")) is not int
            or row["pid"] <= 0
            or row.get("parent_pid") != os.getpid()
            or type(row.get("process_start_ticks")) is not int
            or row["process_start_ticks"] <= 0
            or row.get("start_method") != "fork"
            or row.get("python") != report.get("dependencies", {}).get("python")
            or row.get("platform") != report.get("dependencies", {}).get("platform")
            or row.get("numpy") != dependency_versions.get("numpy")
            or row.get("matplotlib") != dependency_versions.get("matplotlib")
            or row.get("pillow") != dependency_versions.get("Pillow")
            or row.get("policy_version") != M2_REQUIRED_POLICY_VERSION
            or row.get("output_bytes") != repeatability_bytes[index]
            or row.get("child_output_sha256") != repeatability_hashes[index]
            or row.get("parent_recomputed_output_sha256")
            != repeatability_hashes[index]
            for index, row in enumerate(normalized_manifests)
        )
    ):
        raise RuntimeError("canonical M2 V25 repeatability evidence is incomplete")

    sequence = _exact_publication_mapping(
        production.get("sequence"),
        label="V30 production evaluation",
        keys={"outcome", "metrics", "issues", "frames", "patch_bounds", "binding", "procedure"},
    )
    sequence_metrics = _exact_publication_mapping(
        sequence.get("metrics"),
        label="V30 metrics",
        keys={
            "full_daylight",
            "stationary_patch",
            "patch_height",
            "patch_width",
            "luminance_mean",
            "luminance_median",
            "luminance_p1",
            "luminance_p95",
            "luminance_p999",
            "laplacian_rms",
            "red_green_ratio",
            "blue_green_ratio",
            "mean_saturation",
            "black_clipped_fraction",
            "white_clipped_fraction",
        },
    )
    if (
        sequence.get("outcome") != "PASS"
        or sequence.get("issues") != []
        or sequence_metrics.get("full_daylight") is not True
        or sequence_metrics.get("stationary_patch") is not True
        or sequence.get("procedure")
        != (
            "1.4826*MAD(per-frame metric second difference)/sqrt(6), "
            "recomputed from bound uint8 renderer arrays"
        )
    ):
        raise RuntimeError("canonical M2 V30 evaluation is not a measured pass")
    expected_gate_measurements = {
        gate_id: canonical_metrics[gate_id]
        for gate_id in (
            "V01",
            "V02",
            "V03",
            "V07",
            "V08",
            "V13",
            "V14",
            "V15",
            "V16",
            "V41",
        )
    }
    expected_gate_measurements.update(
        {
            "V25": {
                **repeatability,
                "canonical_primary_png": canonical_primary_png,
                "matches_canonical_primary_png": True,
            },
            "V30": sequence,
        }
    )
    expected_gate_details = {
        "V01": (
            "Real pinned ABI scene versus captured-affine exact-area physical reference."
        ),
        "V02": "Source support enumerated from measured Agg impulse responses.",
        "V03": (
            "Canonical Rec.601, 2-D Hann, 64-bin radial spectrum on delivered bytes."
        ),
        "V07": "Protected constants plus immutable M1 artifact/repository binding.",
        "V08": (
            "Five canonical pixels executed through the registered M2 finalizer and "
            "floor quantizer."
        ),
        "V13": (
            "Runtime operation trace binds reflectance/K/coverage/source day-weight "
            "delivery before the sole CIRA call."
        ),
        "V14": (
            "Closed-form contrast fixtures are bound to the captured physical-first trace."
        ),
        "V15": (
            "All delivered physical fields use actual local Agg support, not global extrema."
        ),
        "V16": (
            "Physical-first edge fixture compared with the verified M1 nearest baseline."
        ),
        "V25": (
            "Parent recomputed SHA-256 from bytes returned by three distinct renderer "
            "processes."
        ),
        "V30": (
            "No metric summary is accepted; flicker is recomputed from bound "
            "timestamped arrays."
        ),
        "V41": (
            "Real pinned full-disk objects rendered across the GOES limb with "
            "source-mask replay."
        ),
    }
    if (
        set(expected_gate_measurements) != M2_REQUIRED_GATE_IDS
        or set(expected_gate_details) != M2_REQUIRED_GATE_IDS
        or any(
            gates[gate_id].get("measurements") != measurements
            or gates[gate_id].get("detail") != expected_gate_details[gate_id]
            for gate_id, measurements in expected_gate_measurements.items()
        )
    ):
        raise RuntimeError(
            "canonical M2 required gate evidence differs from production bindings"
        )
    observed_mutations = {row["mutation_id"]: row for row in mutation_rows}
    try:
        live_synthetic_oracle = m2_validation.run_m2_contract_oracle(
            verified_research
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("canonical M2 synthetic mutation oracle cannot be rerun") from exc
    expected_mutations = {
        row.mutation_id: _json_value(row.as_dict())
        for row in live_synthetic_oracle.mutations
    }
    production_discriminators = _exact_publication_mapping(
        canonical_metrics.get("mutation_discriminators"),
        label="canonical production mutation discriminators",
        keys={
            "old_nearest",
            "display_first",
            "resample_cos_then_nonlinear_day_weight",
            "output_proxy_canonical_coverage",
            "current_m0_m1_rollback_bytes",
        },
    )
    repeatability_mutation = {
        "baseline_passed": True,
        "changed_byte_index": repeatability_bytes[0] // 2,
        "changed_byte_mutant_passed": False,
        "caller_hash_parameter_present": False,
        "killed": True,
    }
    expected_mutations.update(
        {
            "canonical_old_nearest_behavior": {
                "mutation_id": "canonical_old_nearest_behavior",
                "killed": True,
                "discriminator": "real-scene exact-area V01/V03",
                "measurements": production_discriminators["old_nearest"],
            },
            "canonical_display_first_behavior": {
                "mutation_id": "canonical_display_first_behavior",
                "killed": True,
                "discriminator": "real-scene signed physical-domain V13",
                "measurements": production_discriminators["display_first"],
            },
            "resample_cos_then_nonlinear_day_weight": {
                "mutation_id": "resample_cos_then_nonlinear_day_weight",
                "killed": True,
                "discriminator": (
                    "source-evaluated terminator taper through captured response"
                ),
                "measurements": production_discriminators[
                    "resample_cos_then_nonlinear_day_weight"
                ],
            },
            "output_proxy_canonical_coverage": {
                "mutation_id": "output_proxy_canonical_coverage",
                "killed": True,
                "discriminator": (
                    "V02/V41 require measured input support and source coverage"
                ),
                "measurements": production_discriminators[
                    "output_proxy_canonical_coverage"
                ],
            },
            "caller_supplied_repeatability_hashes": {
                "mutation_id": "caller_supplied_repeatability_hashes",
                "killed": True,
                "discriminator": "V25 hashes are recomputed from child renderer bytes",
                "measurements": repeatability_mutation,
            },
            "current_m0_m1_rollback_bytes": {
                "mutation_id": "current_m0_m1_rollback_bytes",
                "killed": True,
                "discriminator": (
                    "current real-corpus legacy compositor and registered M0/M1 "
                    "renderer bytes versus immutable M1 hashes"
                ),
                "measurements": production_discriminators[
                    "current_m0_m1_rollback_bytes"
                ],
            },
        }
    )
    rollback = _publication_mapping(
        canonical_metrics.get("M0_M1_ROLLBACK"),
        label="canonical M0/M1 rollback replay",
    )
    if (
        set(expected_mutations) != M2_CANONICAL_MUTATION_IDS
        or observed_mutations != expected_mutations
        or rollback.get("passed") is not True
        or production_discriminators["current_m0_m1_rollback_bytes"].get(
            "killed"
        )
        is not True
    ):
        raise RuntimeError(
            "canonical M2 mutation evidence differs from recomputed production bindings"
        )
    sequence_frames = _publication_list(
        sequence.get("frames"), label="V30 evaluation frames"
    )
    sequence_binding = _exact_publication_mapping(
        sequence.get("binding"),
        label="V30 evaluation binding",
        keys={
            "schema",
            "frame_count",
            "stationary_world_bounds",
            "stationary_class",
            "stationarity_method",
            "daylight_mask_producer_binding",
        },
    )
    if (
        len(sequence_frames) != V30_PATCH_COUNT
        or sequence_binding.get("schema") != V30_BINDING_SCHEMA
        or sequence_binding.get("frame_count") != V30_PATCH_COUNT
    ):
        raise RuntimeError("canonical M2 V30 evaluation frame binding is incomplete")
    expected_sequence_frame_keys = {
        "timestamp_utc",
        "raster",
        "daylight_mask",
        "request_identity_sha256",
        "grid_identity_sha256",
        "source_object_sha256",
        "renderer_png_sha256",
        "renderer_raster_sha256",
        "daylight_mask_producer_binding",
    }
    for raw_frame in sequence_frames:
        frame = _exact_publication_mapping(
            raw_frame,
            label="V30 evaluation frame",
            keys=expected_sequence_frame_keys,
        )
        raster_record = _exact_publication_mapping(
            frame.get("raster"),
            label="V30 evaluation-frame raster",
            keys={"name", "shape", "dtype", "sha256"},
        )
        daylight_record = _exact_publication_mapping(
            frame.get("daylight_mask"),
            label="V30 evaluation-frame daylight mask",
            keys={"name", "shape", "dtype", "sha256"},
        )
        source_hashes = _publication_list(
            frame.get("source_object_sha256"),
            label="V30 evaluation-frame source identities",
        )
        try:
            _parse_utc(frame.get("timestamp_utc"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("canonical M2 V30 frame timestamp is invalid") from exc
        if (
            raster_record.get("dtype") != "|u1"
            or daylight_record.get("dtype") != "|b1"
            or raster_record.get("shape")[:-1] != daylight_record.get("shape")
            or raster_record.get("shape")[-1:] != [3]
            or frame.get("renderer_raster_sha256") != raster_record.get("sha256")
            or len(source_hashes) != 4
            or any(not _valid_publication_sha256(value) for value in source_hashes)
            or any(
                not _valid_publication_sha256(frame.get(name))
                for name in (
                    "request_identity_sha256",
                    "grid_identity_sha256",
                    "renderer_png_sha256",
                    "renderer_raster_sha256",
                )
            )
            or frame.get("daylight_mask_producer_binding")
            != sequence_binding.get("daylight_mask_producer_binding")
        ):
            raise RuntimeError("canonical M2 V30 evaluation-frame schema is invalid")

    input_bindings = _exact_publication_mapping(
        production.get("input_bindings"),
        label="production input bindings",
        keys={
            "canonical_corpus",
            "verified_m1_artifact",
            "rayleigh_startup",
            "runner_image",
            "v30_corpus",
        },
    )
    required_inputs = {
        "canonical_corpus",
        "verified_m1_artifact",
        "rayleigh_startup",
        "runner_image",
        "v30_corpus",
    }
    for name in required_inputs:
        _publication_mapping(input_bindings[name], label=f"{name} input binding")
    rayleigh = _publication_mapping(
        production.get("rayleigh_startup"), label="Rayleigh startup binding"
    )
    if rayleigh.get("passed") is not True or rayleigh != input_bindings["rayleigh_startup"]:
        raise RuntimeError("canonical M2 Rayleigh startup binding is inconsistent")
    if input_bindings["runner_image"] != M2_RUNNER_IMAGE_PROVENANCE:
        raise RuntimeError("canonical M2 runner-image input binding is inconsistent")
    _validate_bound_production_inputs(input_bindings, rayleigh)
    if canonical.get("objects") != input_bindings["canonical_corpus"].get(
        "objects"
    ):
        raise RuntimeError(
            "canonical production objects differ from the canonical-corpus input binding"
        )
    if canonical.get("m1_baseline") != input_bindings["verified_m1_artifact"]:
        raise RuntimeError(
            "canonical production M1 baseline differs from the verified input binding"
        )

    v30_capture = _exact_publication_mapping(
        report.get("v30_capture"),
        label="V30 capture",
        keys={
            "schema",
            "corpus",
            "request_identity_sha256",
            "grid_identity_sha256",
            "raster_sha256",
            "distinct_raster_count",
            "frozen_output_mutant_killed",
            "patch_bounds",
            "stationary_world_bounds",
            "stationary_class",
            "stationarity_method",
            "frames",
            "screening",
        },
    )
    v30_provenance = _publication_list(
        v30_capture.get("frames"), label="V30 provenance"
    )
    v30_screening = _publication_list(
        v30_capture.get("screening"), label="V30 screening"
    )
    for raw_frame in v30_provenance:
        frame = _exact_publication_mapping(
            raw_frame,
            label="V30 capture provenance row",
            keys={
                "nominal_slot_utc",
                "exact_noaa_scan_start_utc",
                "production_scan_start_utc",
                "source_objects",
                "request_identity_sha256",
                "grid_identity_sha256",
                "renderer_png_sha256",
                "renderer_raster_sha256",
                "trace_identity_sha256",
                "delivered_shape",
                "patch_bounds",
                "world_patch_max_pixel_edge_residual",
                "world_patch_pixel_edge_tolerance",
            },
        )
        source_objects = _publication_list(
            frame.get("source_objects"), label="V30 provenance source objects"
        )
        if len(source_objects) != 4:
            raise RuntimeError("canonical M2 V30 provenance source set is incomplete")
        for raw_object in source_objects:
            _exact_publication_mapping(
                raw_object,
                label="V30 provenance source object",
                keys={
                    "nominal_slot_utc",
                    "exact_scan_start_utc",
                    "exact_scan_end_utc",
                    "date_created_utc",
                    "channel",
                    "role",
                    "key",
                    "filename",
                    "bytes",
                    "sha256",
                    "band_id",
                },
            )
    for raw_screen in v30_screening:
        _exact_publication_mapping(
            raw_screen,
            label="V30 capture screening row",
            keys={
                "nominal_slot_utc",
                "dqf_zero_fraction_by_role",
                "cloud_proxy_fraction",
                "c13_above_280k_fraction",
                "cloud_proxy_definition",
                "selection_policy",
            },
        )
    if (
        v30_capture.get("schema") != V30_CAPTURE_SCHEMA
        or v30_capture.get("corpus") != input_bindings["v30_corpus"]
        or v30_capture.get("frozen_output_mutant_killed") is not True
        or v30_capture.get("distinct_raster_count") != V30_PATCH_COUNT
        or len(
            _publication_list(
                v30_capture.get("raster_sha256"), label="V30 raster identities"
            )
        )
        != V30_PATCH_COUNT
        or len(v30_provenance) != V30_PATCH_COUNT
        or len(v30_screening) != V30_PATCH_COUNT
        or v30_capture.get("patch_bounds") != sequence.get("patch_bounds")
        or v30_capture.get("stationary_world_bounds")
        != sequence_binding.get("stationary_world_bounds")
    ):
        raise RuntimeError("canonical M2 V30 capture is incomplete or inconsistent")
    capture_raster_identities = _publication_list(
        v30_capture.get("raster_sha256"), label="V30 capture raster identities"
    )
    all_v30_source_rows: list[dict[str, Any]] = []
    role_channel_band = {
        "red": ("C02", 2),
        "blue": ("C01", 1),
        "veggie": ("C03", 3),
        "ir": ("C13", 13),
    }
    for index, (provenance, screen, sequence_frame) in enumerate(
        zip(v30_provenance, v30_screening, sequence_frames, strict=True)
    ):
        provenance = _publication_mapping(
            provenance, label=f"V30 capture provenance {index}"
        )
        screen = _publication_mapping(screen, label=f"V30 screening {index}")
        sequence_frame = _publication_mapping(
            sequence_frame, label=f"V30 sequence frame {index}"
        )
        source_rows = _publication_list(
            provenance.get("source_objects"),
            label=f"V30 provenance source objects {index}",
        )
        source_hashes = [row.get("sha256") for row in source_rows]
        source_roles = [row.get("role") for row in source_rows]
        try:
            provenance_nominal_time = _parse_utc(
                provenance.get("nominal_slot_utc")
            )
            exact_time = _parse_utc(provenance.get("exact_noaa_scan_start_utc"))
            sequence_time = _parse_utc(sequence_frame.get("timestamp_utc"))
            production_time = _parse_utc(provenance.get("production_scan_start_utc"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"canonical M2 V30 provenance timestamp differs at frame {index}"
            ) from exc
        for raw_source in source_rows:
            source = _publication_mapping(
                raw_source, label=f"V30 provenance source object {index}"
            )
            role = source.get("role")
            expected_channel_band = role_channel_band.get(role)
            try:
                nominal_time = _parse_utc(source.get("nominal_slot_utc"))
                source_start = _parse_utc(source.get("exact_scan_start_utc"))
                source_end = _parse_utc(source.get("exact_scan_end_utc"))
                source_created = _parse_utc(source.get("date_created_utc"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"canonical M2 V30 source-object time differs at frame {index}"
                ) from exc
            if (
                expected_channel_band is None
                or (source.get("channel"), source.get("band_id"))
                != expected_channel_band
                or nominal_time != provenance_nominal_time
                or source_start != exact_time
                or source_end < source_start
                or source_created < source_end
                or type(source.get("key")) is not str
                or not source["key"]
                or type(source.get("filename")) is not str
                or Path(source["filename"]).name != source["filename"]
                or Path(source["key"]).name != source["filename"]
                or type(source.get("bytes")) is not int
                or source["bytes"] <= 0
                or not _valid_publication_sha256(source.get("sha256"))
            ):
                raise RuntimeError(
                    f"canonical M2 V30 source-object identity differs at frame {index}"
                )
            all_v30_source_rows.append(source)
        raster_shape = sequence_frame.get("raster", {}).get("shape", [])
        dqf = _exact_publication_mapping(
            screen.get("dqf_zero_fraction_by_role"),
            label=f"V30 screening DQF fractions {index}",
            keys=set(V30_ROLE_ORDER),
        )
        if (
            exact_time != sequence_time
            or production_time != exact_time.replace(microsecond=0)
            or provenance.get("request_identity_sha256")
            != sequence_frame.get("request_identity_sha256")
            or provenance.get("request_identity_sha256")
            != v30_capture.get("request_identity_sha256")
            or provenance.get("grid_identity_sha256")
            != sequence_frame.get("grid_identity_sha256")
            or provenance.get("grid_identity_sha256")
            != v30_capture.get("grid_identity_sha256")
            or provenance.get("renderer_png_sha256")
            != sequence_frame.get("renderer_png_sha256")
            or provenance.get("renderer_raster_sha256")
            != sequence_frame.get("renderer_raster_sha256")
            or provenance.get("renderer_raster_sha256")
            != capture_raster_identities[index]
            or source_hashes != sequence_frame.get("source_object_sha256")
            or source_roles != list(V30_ROLE_ORDER)
            or not _valid_publication_sha256(
                provenance.get("trace_identity_sha256")
            )
            or provenance.get("delivered_shape") != raster_shape[:2]
            or provenance.get("patch_bounds") != v30_capture.get("patch_bounds")
            or type(provenance.get("world_patch_max_pixel_edge_residual"))
            is not float
            or provenance["world_patch_max_pixel_edge_residual"] < 0.0
            or provenance["world_patch_max_pixel_edge_residual"]
            > V30_PATCH_EDGE_BINDING_TOLERANCE_PX
            or provenance.get("world_patch_pixel_edge_tolerance")
            != V30_PATCH_EDGE_BINDING_TOLERANCE_PX
            or screen.get("nominal_slot_utc")
            != provenance.get("nominal_slot_utc")
            or any(type(value) is not float or value != 1.0 for value in dqf.values())
            or screen.get("cloud_proxy_fraction") != 0.0
            or screen.get("c13_above_280k_fraction") != 1.0
            or screen.get("cloud_proxy_definition")
            != "C02>0.30 or C13<270K; acquisition screen only"
            or screen.get("selection_policy")
            != "predeclared_before_renderer_metric_evaluation"
        ):
            raise RuntimeError(
                f"canonical M2 V30 provenance/screening differs at frame {index}"
            )
    v30_corpus_binding = _publication_mapping(
        v30_capture.get("corpus"), label="V30 capture corpus aggregate"
    )
    observed_filenames = [row["filename"] for row in all_v30_source_rows]
    observed_keys = [row["key"] for row in all_v30_source_rows]
    inventory_payload = "".join(
        f"{row['sha256']}  {row['filename']}\n"
        for row in sorted(all_v30_source_rows, key=lambda value: value["filename"])
    ).encode("ascii")
    if (
        len(all_v30_source_rows) != V30_PATCH_COUNT * len(V30_ROLE_ORDER)
        or len(set(observed_filenames)) != len(observed_filenames)
        or len(set(observed_keys)) != len(observed_keys)
        or v30_corpus_binding.get("object_count") != len(all_v30_source_rows)
        or v30_corpus_binding.get("group_count") != V30_PATCH_COUNT
        or v30_corpus_binding.get("total_bytes")
        != sum(row["bytes"] for row in all_v30_source_rows)
        or v30_corpus_binding.get("inventory_sha256")
        != sha256_bytes(inventory_payload)
    ):
        raise RuntimeError("canonical M2 V30 corpus aggregate differs from provenance")

    report_artifacts = _publication_mapping(
        report.get("gallery_artifacts"), label="gallery artifact map"
    )
    if report_artifacts != dict(metadata) or set(extras) != set(metadata):
        raise RuntimeError("canonical M2 report/gallery/payload coverage differs")
    expected_artifact_row_keys = {
        "filename",
        "sha256",
        "bytes",
        "media_type",
        "width",
        "height",
        "description",
        "source_bindings",
        "source_binding_sha256",
    }
    for name, raw_row in metadata.items():
        row = _publication_mapping(raw_row, label=f"artifact metadata {name}")
        bindings = _publication_list(
            row.get("source_bindings"), label=f"artifact source bindings {name}"
        )
        binding_hashes = _publication_mapping(
            row.get("source_binding_sha256"),
            label=f"artifact source-binding hashes {name}",
            nonempty=False,
        )
        if (
            set(row) != expected_artifact_row_keys
            or type(row.get("media_type")) is not str
            or not row["media_type"]
            or type(row.get("description")) is not str
            or not row["description"]
            or any(
                value is not None and (type(value) is not int or value <= 0)
                for value in (row.get("width"), row.get("height"))
            )
            or any(type(binding) is not str or not binding for binding in bindings)
            or len(bindings) != len(set(bindings))
            or set(bindings) != set(binding_hashes)
            or any(
                not _valid_publication_sha256(digest)
                for digest in binding_hashes.values()
            )
        ):
            raise RuntimeError(
                f"canonical M2 artifact metadata schema is invalid: {name}"
            )
    required_artifacts = {
        *_REQUIRED_M2_GALLERY_PAYLOADS,
        *_REQUIRED_V30_PAYLOADS,
        "m2_gallery_manifest.json",
    }
    if not required_artifacts <= extras.keys():
        raise RuntimeError("canonical M2 gallery or V30 payload set is incomplete")

    gallery_manifest = _publication_json_payload(
        extras["m2_gallery_manifest.json"], label="gallery manifest"
    )
    gallery_rows = _publication_mapping(
        gallery_manifest.get("artifacts"), label="gallery manifest artifacts"
    )
    if (
        set(gallery_manifest)
        != {
            "schema",
            "scene_id",
            "bbox",
            "native_shape",
            "source_evidence",
            "artifacts",
            "payload_index_sha256",
            "native_comparison_sheet",
            "signed_error",
            "spectral",
            "repeatability",
            "sequence",
            "andrew_review",
            "claim_boundary",
        }
        or
        gallery_manifest.get("schema") != GALLERY_SCHEMA
        or set(gallery_rows) != _REQUIRED_M2_GALLERY_PAYLOADS
        or any(metadata.get(name) != row for name, row in gallery_rows.items())
    ):
        raise RuntimeError("canonical M2 gallery manifest coverage is inconsistent")
    source_evidence = _exact_publication_mapping(
        gallery_manifest.get("source_evidence"),
        label="gallery source evidence",
        keys={
            "schema",
            "summary_sha256",
            "array_records",
            "primary_png",
            "production_policy",
            "source_objects",
        },
    )
    canonical_summary_sha256 = sha256_bytes(_strict_json_bytes(canonical))
    primary_capture = _publication_mapping(
        canonical.get("primary_capture"), label="canonical primary capture"
    )
    if (
        set(source_evidence)
        != {
            "schema",
            "summary_sha256",
            "array_records",
            "primary_png",
            "production_policy",
            "source_objects",
        }
        or source_evidence.get("schema") != M2_PRODUCTION_SCHEMA
        or source_evidence.get("summary_sha256") != canonical_summary_sha256
        or source_evidence.get("array_records") != canonical.get("arrays")
        or source_evidence.get("primary_png") != canonical.get("primary_png")
        or source_evidence.get("source_objects") != canonical.get("objects")
        or source_evidence.get("production_policy")
        != primary_capture.get("policy_version")
        or gallery_manifest.get("scene_id") != canonical.get("scene_id")
        or gallery_manifest.get("bbox") != canonical.get("bbox")
    ):
        raise RuntimeError("canonical M2 gallery source binding differs from production")
    payload_index = {
        name: {"sha256": row["sha256"], "bytes": row["bytes"]}
        for name, row in sorted(gallery_rows.items())
    }
    if gallery_manifest.get("payload_index_sha256") != sha256_bytes(
        _strict_json_bytes(payload_index)
    ):
        raise RuntimeError("canonical M2 gallery payload-index identity differs")
    manifest_name = "m2_gallery_manifest.json"
    expected_manifest_metadata = _artifact_row(
        manifest_name,
        extras[manifest_name],
        media_type="application/json",
        description="Immutable manifest for all primary-layer M2 review payloads.",
        source_binding_sha256={
            "canonical_evidence_summary": canonical_summary_sha256
        },
    )
    if metadata.get(manifest_name) != expected_manifest_metadata:
        raise RuntimeError("canonical M2 gallery-manifest metadata differs")
    native_shape = _publication_list(
        gallery_manifest.get("native_shape"), label="gallery native shape"
    )
    native_comparison = _exact_publication_mapping(
        gallery_manifest.get("native_comparison_sheet"),
        label="native comparison sheet",
        keys={"artifact", "scale_percent", "panel_bounds", "panels_are_pixel_identical"},
    )
    signed_error = _exact_publication_mapping(
        gallery_manifest.get("signed_error"),
        label="signed-error evidence",
        keys={
            "quantity",
            "rec601_coefficients_rgb",
            "raw_values_artifact",
            "raw_values_dtype",
            "raw_values_shape",
            "raw_values_order",
            "shared_scale_percentile",
            "shared_symmetric_limit_dn",
            "shared_scale_definition",
            "comparison_sheet",
        },
    )
    error_comparison = _exact_publication_mapping(
        signed_error.get("comparison_sheet"),
        label="signed-error comparison sheet",
        keys={"artifact", "scale_percent", "panel_bounds", "panels_are_pixel_identical"},
    )
    spectral = _exact_publication_mapping(
        gallery_manifest.get("spectral"),
        label="spectral evidence",
        keys={
            "plot",
            "csv",
            "procedure",
            "bins",
            "cutoff_cycles_per_delivered_pixel",
            "ratios_to_exact_area_reference",
        },
    )
    if (
        len(native_shape) != 3
        or native_shape[-1] != 3
        or any(type(value) is not int or value <= 0 for value in native_shape)
        or native_comparison.get("scale_percent") != 100
        or native_comparison.get("panels_are_pixel_identical") is not True
        or native_comparison.get("artifact") not in gallery_rows
        or signed_error.get("raw_values_artifact") not in gallery_rows
        or error_comparison.get("scale_percent") != 100
        or error_comparison.get("panels_are_pixel_identical") is not True
        or error_comparison.get("artifact") not in gallery_rows
        or spectral.get("csv") not in gallery_rows
        or spectral.get("plot") not in gallery_rows
    ):
        raise RuntimeError("canonical M2 native-scale gallery semantics are incomplete")
    gallery_repeatability = _exact_publication_mapping(
        gallery_manifest.get("repeatability"),
        label="gallery repeatability binding",
        keys={"present", "summary", "summary_sha256"},
    )
    gallery_sequence = _exact_publication_mapping(
        gallery_manifest.get("sequence"),
        label="gallery V30 binding",
        keys={"present", "summary", "summary_sha256"},
    )
    gallery_claim = _exact_publication_mapping(
        gallery_manifest.get("claim_boundary"),
        label="gallery claim boundary",
        keys={"stage_advancement_claimed_by_gallery", "scope", "limitations"},
    )
    andrew_review = _exact_publication_mapping(
        gallery_manifest.get("andrew_review"),
        label="Andrew review state",
        keys={
            "verdict",
            "instructions",
            "supplement_pattern",
            "supplement_must_bind_final_gallery_manifest_sha256",
            "published_payloads_are_immutable",
        },
    )
    if (
        gallery_repeatability.get("present") is not True
        or gallery_repeatability.get("summary") != repeatability
        or gallery_repeatability.get("summary_sha256")
        != sha256_bytes(_strict_json_bytes(repeatability))
        or gallery_sequence.get("present") is not True
        or gallery_sequence.get("summary") != sequence
        or gallery_sequence.get("summary_sha256")
        != sha256_bytes(_strict_json_bytes(sequence))
        or set(gallery_claim)
        != {"stage_advancement_claimed_by_gallery", "scope", "limitations"}
        or gallery_claim.get("stage_advancement_claimed_by_gallery") is not False
        or gallery_claim.get("scope")
        != "M2 primary imagery-layer finalizer only"
        or gallery_claim.get("limitations")
        != list(M2_GALLERY_REQUIRED_LIMITATIONS)
        or andrew_review
        != {
            "verdict": "PENDING",
            "instructions": "REVIEW_INSTRUCTIONS.md",
            "supplement_pattern": "ANDREW_VERDICT_<UTC_TIMESTAMP>.md",
            "supplement_must_bind_final_gallery_manifest_sha256": True,
            "published_payloads_are_immutable": True,
        }
    ):
        raise RuntimeError("canonical M2 gallery evaluation bindings differ")

    _validate_gallery_payload_semantics(
        canonical=canonical,
        repeatability=repeatability,
        sequence=sequence,
        gallery_manifest=gallery_manifest,
        gallery_rows=gallery_rows,
        extras=extras,
        metadata=metadata,
    )

    v30_evidence = _publication_json_payload(
        extras["v30_patch_evidence.json"], label="V30 patch evidence"
    )
    patches = [
        _publication_mapping(row, label="V30 patch row")
        for row in _publication_list(v30_evidence.get("patches"), label="V30 patches")
    ]
    expected_patch_names = {
        f"v30_patch_{index:02d}_96px.png" for index in range(1, 20)
    }
    observed_patch_names = {row.get("filename") for row in patches}
    stack = _publication_mapping(v30_evidence.get("stack"), label="V30 stack")
    stack_name = "v30_patches_19x96x96_rgb_uint8.npy"
    try:
        stack_array = np.load(io.BytesIO(extras[stack_name]), allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise RuntimeError("canonical M2 V30 stack is not a safe NPY array") from exc
    if (
        type(stack_array) is not np.ndarray
        or stack_array.dtype != np.uint8
        or stack_array.shape != (V30_PATCH_COUNT, *V30_PATCH_SHAPE)
        or extras[stack_name] != _npy_bytes(stack_array)
    ):
        raise RuntimeError("canonical M2 V30 stack dtype, shape, or NPY bytes differ")
    expected_stack = {
        "filename": stack_name,
        "payload_sha256": sha256_bytes(extras[stack_name]),
        "payload_bytes": len(extras[stack_name]),
        **_array_record("v30_patch_stack", stack_array),
    }
    if (
        set(v30_evidence)
        != {
            "schema",
            "patch_count",
            "patch_shape",
            "patch_bounds",
            "stationary_world_bounds",
            "stack",
            "capture_summary_sha256",
            "capture_summary",
            "evaluation_summary_sha256",
            "evaluation",
            "patches",
            "limitations",
        }
        or v30_evidence.get("schema")
        != "tat_truecolor_m2_v30_patch_evidence/v1"
        or v30_evidence.get("patch_count") != V30_PATCH_COUNT
        or v30_evidence.get("patch_shape") != list(V30_PATCH_SHAPE)
        or v30_evidence.get("patch_bounds") != v30_capture.get("patch_bounds")
        or v30_evidence.get("stationary_world_bounds")
        != v30_capture.get("stationary_world_bounds")
        or len(patches) != V30_PATCH_COUNT
        or observed_patch_names != expected_patch_names
        or v30_evidence.get("capture_summary") != v30_capture
        or v30_evidence.get("capture_summary_sha256")
        != sha256_bytes(_strict_json_bytes(v30_capture))
        or v30_evidence.get("evaluation") != sequence
        or v30_evidence.get("evaluation_summary_sha256")
        != sha256_bytes(_strict_json_bytes(sequence))
        or stack != expected_stack
        or v30_evidence.get("limitations") != list(PRODUCTION_LIMITATIONS)
    ):
        raise RuntimeError("canonical M2 V30 patch evidence bindings differ")
    expected_stack_metadata = _artifact_row(
        stack_name,
        extras[stack_name],
        media_type="application/x-npy",
        width=V30_PATCH_SHAPE[1],
        height=V30_PATCH_SHAPE[0],
        description="Raw uint8 stack for all 19 bound V30 patches in timestamp order.",
        source_binding_sha256={
            "v30_patch_stack": expected_stack["sha256"],
        },
    )
    evidence_name = "v30_patch_evidence.json"
    expected_evidence_metadata = _artifact_row(
        evidence_name,
        extras[evidence_name],
        media_type="application/json",
        description=(
            "Machine-readable V30 patch identities and independently recomputed metrics."
        ),
        source_binding_sha256={
            "v30_capture_summary": v30_evidence["capture_summary_sha256"],
            "v30_evaluation_summary": v30_evidence["evaluation_summary_sha256"],
            "v30_patch_stack": expected_stack["sha256"],
        },
    )
    if (
        metadata.get(stack_name) != expected_stack_metadata
        or metadata.get(evidence_name) != expected_evidence_metadata
    ):
        raise RuntimeError("canonical M2 V30 stack/evidence metadata differs")

    patch_metric_rows = [
        _patch_metrics(np.ascontiguousarray(stack_array[index]))
        for index in range(V30_PATCH_COUNT)
    ]
    flicker_metrics: dict[str, float] = {}
    for metric_name in patch_metric_rows[0]:
        values = np.asarray(
            [row[metric_name] for row in patch_metric_rows], dtype=np.float64
        )
        second_difference = values[2:] - 2.0 * values[1:-1] + values[:-2]
        median = float(np.median(second_difference))
        mad = float(np.median(np.abs(second_difference - median)))
        flicker_metrics[metric_name] = 1.4826 * mad / math.sqrt(6.0)
    recomputed_sequence_metrics: dict[str, Any] = {
        "full_daylight": True,
        "stationary_patch": True,
        "patch_height": V30_PATCH_SHAPE[0],
        "patch_width": V30_PATCH_SHAPE[1],
        **flicker_metrics,
    }
    if sequence_metrics != recomputed_sequence_metrics:
        raise RuntimeError(
            "canonical M2 V30 aggregate flicker metrics differ from patch stack"
        )
    registered_flicker = m2_validation.StationaryFlickerMetrics(
        **recomputed_sequence_metrics
    )
    if m2_validation.stationary_flicker_issues(registered_flicker):
        raise RuntimeError("canonical M2 V30 patch stack fails registered thresholds")

    from PIL import Image

    expected_patch_keys = {
        "index",
        "timestamp_utc",
        "filename",
        "png_sha256",
        "png_bytes",
        "array",
        "metrics",
        "parent_renderer_raster_sha256",
        "parent_renderer_png_sha256",
        "request_identity_sha256",
        "grid_identity_sha256",
        "source_object_sha256",
    }
    capture_rasters = _publication_list(
        v30_capture.get("raster_sha256"), label="V30 capture raster identities"
    )
    for index, (row, raw_frame) in enumerate(
        zip(patches, sequence_frames, strict=True)
    ):
        frame = _publication_mapping(raw_frame, label="V30 evaluation frame")
        name = f"v30_patch_{index + 1:02d}_96px.png"
        patch_array = np.ascontiguousarray(stack_array[index])
        try:
            row_time = _parse_utc(row.get("timestamp_utc"))
            frame_time = _parse_utc(frame.get("timestamp_utc"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"canonical M2 V30 patch timestamp differs: {name}"
            ) from exc
        if (
            set(row) != expected_patch_keys
            or row.get("index") != index
            or row.get("filename") != name
            or row_time != frame_time
            or row.get("png_sha256") != sha256_bytes(extras[name])
            or row.get("png_bytes") != len(extras[name])
            or row.get("array")
            != _array_record(f"v30_patch_{index + 1:02d}", patch_array)
            or row.get("metrics") != _patch_metrics(patch_array)
            or row.get("parent_renderer_raster_sha256")
            != frame.get("renderer_raster_sha256")
            or row.get("parent_renderer_raster_sha256") != capture_rasters[index]
            or row.get("parent_renderer_png_sha256")
            != frame.get("renderer_png_sha256")
            or row.get("request_identity_sha256")
            != frame.get("request_identity_sha256")
            or row.get("request_identity_sha256")
            != v30_capture.get("request_identity_sha256")
            or row.get("grid_identity_sha256")
            != frame.get("grid_identity_sha256")
            or row.get("grid_identity_sha256")
            != v30_capture.get("grid_identity_sha256")
            or row.get("source_object_sha256")
            != frame.get("source_object_sha256")
        ):
            raise RuntimeError(f"canonical M2 V30 patch identity differs: {name}")
        expected_patch_metadata = _artifact_row(
            name,
            extras[name],
            media_type="image/png",
            width=V30_PATCH_SHAPE[1],
            height=V30_PATCH_SHAPE[0],
            description=(
                f"V30 frame {index + 1:02d} predeclared clear-land full-daylight "
                "patch at exact delivered 100% scale."
            ),
            source_binding_sha256={
                "patch_array": row["array"]["sha256"],
                "renderer_raster": row["parent_renderer_raster_sha256"],
            },
        )
        if metadata.get(name) != expected_patch_metadata:
            raise RuntimeError(f"canonical M2 V30 patch metadata differs: {name}")
        try:
            with Image.open(io.BytesIO(extras[name])) as image:
                image.load()
                if (
                    image.format != "PNG"
                    or image.mode != "RGB"
                    or image.size != (V30_PATCH_SHAPE[1], V30_PATCH_SHAPE[0])
                    or getattr(image, "n_frames", 1) != 1
                ):
                    raise RuntimeError(
                        f"canonical M2 V30 patch PNG format differs: {name}"
                    )
                png_array = np.asarray(image).copy()
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"canonical M2 V30 patch is not a lossless PNG: {name}"
            ) from exc
        if (
            png_array.dtype != np.uint8
            or png_array.shape != V30_PATCH_SHAPE
            or not np.array_equal(png_array, patch_array)
        ):
            raise RuntimeError(
                f"canonical M2 V30 patch PNG pixels differ from stack: {name}"
            )

    dependencies = _publication_mapping(
        report.get("dependencies"), label="production dependency identity"
    )
    dependency_contract = _publication_mapping(
        dependencies.get("production_dependency_contract"),
        label="production dependency contract",
    )
    if dependency_contract.get("passed") is not True:
        raise RuntimeError("canonical M2 production dependency contract did not pass")
    _require_publication_dependency_identity(dependencies)

    evidence_hash = report.get("evidence_payload_sha256")
    unsigned_report = dict(report)
    unsigned_report.pop("evidence_payload_sha256", None)
    if evidence_hash != sha256_bytes(stable_json_bytes(unsigned_report)):
        raise RuntimeError("canonical M2 report evidence-payload identity differs")


def _head_blob_sha256(head: str, name: str) -> str:
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != name
    ):
        raise RuntimeError(f"unsafe repository-bound source path: {name!r}")
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{head}:{name}"],
        check=True,
        capture_output=True,
    )
    return sha256_bytes(completed.stdout)


def _repository_publication_issues(
    report: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split fatal repository faults from valid dirty/pre-commit state."""

    fatal: list[str] = []
    precommit: list[str] = []
    repository = _publication_mapping(
        report.get("repository"), label="repository identity"
    )
    try:
        fresh_repository = collect_repository_identity()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        fatal.append(f"fresh repository identity cannot be collected: {exc}")
    else:
        if set(repository) != set(fresh_repository):
            fatal.append(
                "reported repository identity object keys differ from fresh identity"
            )
        if repository != fresh_repository:
            fatal.append(
                "reported repository identity, Git directories, or linked-worktree "
                "pointer differs from fresh identity"
            )
    branch = repository.get("branch")
    upstream = repository.get("upstream")
    head = repository.get("head")
    head_tree = repository.get("head_tree")
    upstream_head = repository.get("upstream_head")
    divergence = repository.get("divergence")
    status = repository.get("status_porcelain_v1")
    valid_sha1 = re.compile(r"^[0-9a-f]{40}$")
    if type(branch) is not str or not branch.strip():
        fatal.append("repository branch is absent")
    if type(upstream) is not str or not upstream.strip():
        fatal.append("repository upstream is absent")
    if type(head) is not str or valid_sha1.fullmatch(head) is None:
        fatal.append("repository HEAD is invalid")
    if type(head_tree) is not str or valid_sha1.fullmatch(head_tree) is None:
        fatal.append("repository HEAD tree is invalid")
    if type(upstream_head) is not str or valid_sha1.fullmatch(upstream_head) is None:
        fatal.append("repository upstream HEAD is invalid")
    if (
        type(divergence) is not dict
        or type(divergence.get("ahead")) is not int
        or divergence.get("ahead", -1) < 0
        or divergence.get("behind") != 0
    ):
        fatal.append("repository is behind or has invalid upstream divergence")
    if type(status) is not list or any(type(row) is not str for row in status):
        fatal.append("repository status_porcelain_v1 is invalid")
        dirty = False
    else:
        dirty = bool(status)
        if dirty:
            precommit.append("repository status_porcelain_v1 is not empty")

    source_metadata = _publication_mapping(
        report.get("source"), label="source metadata"
    )
    if (
        source_metadata.get("repository_root") != repository.get("repository_root")
        or source_metadata.get("git_head") != head
        or source_metadata.get("git_branch") != branch
        or source_metadata.get("git_status_porcelain") != status
    ):
        fatal.append("source Git identity fields differ from repository identity")
    if source_metadata.get("m2_bound_product_and_residual_source_files") != list(
        M2_PRODUCT_SOURCE_FILES
    ):
        fatal.append("M2 bound product-source registry differs")
    if source_metadata.get("m2_bound_local_render_data_files") != list(
        M2_LOCAL_RENDER_DATA_FILES
    ):
        fatal.append("M2 bound local render-data registry differs")
    if (
        source_metadata.get("cira_stretch_source_sha256")
        != production_adapter.cira_source_sha256()
    ):
        fatal.append("CIRA source identity differs from live protected source")

    try:
        live_root = Path(_git("rev-parse", "--show-toplevel").stdout.strip()).resolve(
            strict=True
        )
        reported_root = Path(str(repository.get("repository_root", ""))).resolve(
            strict=True
        )
        live_head = _git("rev-parse", "HEAD").stdout.strip()
        live_tree = _git("rev-parse", "HEAD^{tree}").stdout.strip()
        live_branch = _git("branch", "--show-current").stdout.strip()
        live_upstream = _git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        ).stdout.strip()
        live_upstream_head = _git("rev-parse", "@{upstream}").stdout.strip()
        live_counts = _git(
            "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
        ).stdout.split()
        live_status = _git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout.splitlines()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        fatal.append(f"live repository identity cannot be reverified: {exc}")
    else:
        live_divergence = (
            {
                "ahead": int(live_counts[0]),
                "behind": int(live_counts[1]),
            }
            if len(live_counts) == 2
            else None
        )
        if live_root != REPO_ROOT.resolve(strict=True):
            fatal.append("live repository root differs from validator root")
        if reported_root != live_root:
            fatal.append("reported repository root differs from live root")
        if head != live_head or head_tree != live_tree:
            fatal.append("reported HEAD/tree differs from live repository")
        if branch != live_branch or upstream != live_upstream:
            fatal.append("reported branch/upstream differs from live repository")
        if upstream_head != live_upstream_head or divergence != live_divergence:
            fatal.append("reported upstream identity/divergence differs from live repository")
        if status != live_status:
            fatal.append("reported repository status differs from live repository")

    hash_maps = (
        (
            "source",
            source_metadata.get("source_sha256"),
        ),
        (
            "local render data",
            source_metadata.get("local_render_data_sha256"),
        ),
        ("harness", report.get("harness_source_sha256")),
    )
    for label, raw_hashes in hash_maps:
        try:
            hashes = _publication_mapping(
                raw_hashes,
                label=f"{label} source SHA-256 map",
                nonempty=False,
            )
        except RuntimeError as exc:
            fatal.append(str(exc))
            continue
        if label == "source" and set(hashes) != set(M2_PRODUCT_SOURCE_FILES):
            fatal.append("M2 source SHA-256 coverage differs from product registry")
        if label == "local render data" and set(hashes) != set(
            M2_LOCAL_RENDER_DATA_FILES
        ):
            fatal.append("M2 local render-data SHA-256 coverage differs from registry")
        if label == "harness":
            try:
                current_harness = m2_harness_source_hashes(REPO_ROOT)
            except (OSError, RuntimeError) as exc:
                fatal.append(f"M2 harness identity cannot be recomputed: {exc}")
            else:
                if hashes != current_harness:
                    fatal.append("M2 harness SHA-256 coverage or bytes differ")
        for name, expected in hashes.items():
            if (
                type(name) is not str
                or type(expected) is not str
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                fatal.append(f"{label} source identity is invalid: {name!r}")
                continue
            path = REPO_ROOT / name
            if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
                fatal.append(f"{label} source bytes differ from report: {name}")
                continue
            if type(head) is not str or valid_sha1.fullmatch(head) is None:
                continue
            try:
                head_digest = _head_blob_sha256(head, name)
            except (RuntimeError, subprocess.CalledProcessError):
                head_digest = None
            if head_digest != expected:
                issue = f"{label} source does not match a HEAD-tree blob: {name}"
                (precommit if dirty else fatal).append(issue)
    return tuple(fatal), tuple(precommit)


def _classify_report_for_publication(
    report: Mapping[str, Any],
    extras: Mapping[str, bytes],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    artifact_leaf: str,
) -> str:
    """Return a manifest evidence class only after fail-closed semantics."""

    if not isinstance(report, Mapping) or report.get("schema") != M2_REPORT_SCHEMA:
        raise RuntimeError("M2 publication report schema is unregistered")
    summary = _publication_mapping(report.get("summary"), label="summary")
    stage_passed = summary.get("stage_advancement_passed")
    blockers = summary.get("canonical_blockers")
    if type(stage_passed) is not bool or type(blockers) is not list:
        raise RuntimeError("M2 publication summary has invalid stage semantics")
    if stage_passed:
        _validate_canonical_publication_semantics(report, extras, metadata)
        fatal_repository_issues, precommit_issues = _repository_publication_issues(
            report
        )
        if fatal_repository_issues:
            raise RuntimeError(
                "canonical M2 repository identity is invalid: "
                + "; ".join(fatal_repository_issues)
            )
        if precommit_issues:
            if not artifact_leaf.startswith("scratch-"):
                raise RuntimeError(
                    "dirty/pre-commit M2 evidence requires a scratch-* artifact leaf: "
                    + "; ".join(precommit_issues)
                )
            return "noncanonical_precommit_candidate"
        if artifact_leaf.startswith("scratch-"):
            raise RuntimeError(
                "a clean canonical M2 report must use a final non-scratch artifact leaf"
            )
        return "canonical_stage_evidence"

    command = report.get("command")
    if command not in {None, "contract-oracle"}:
        raise RuntimeError("nonclosing M2 evidence is not a contract oracle")
    if not blockers:
        raise RuntimeError("nonclosing M2 contract evidence must name blockers")
    if "production_evaluation" in report or "v30_capture" in report:
        raise RuntimeError("nonclosing M2 contract evidence carries production claims")
    oracle = report.get("oracle")
    if oracle is not None:
        oracle_row = _publication_mapping(oracle, label="contract oracle")
        if (
            oracle_row.get("schema") != M2_ORACLE_SCHEMA
            or oracle_row.get("stage_advancement_passed") is not False
            or oracle_row.get("canonical_blockers") != blockers
        ):
            raise RuntimeError("nonclosing M2 oracle semantics differ from summary")
    return "non_closing_contract_oracle"


def _write_new_file(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite staged evidence: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish a sibling directory without replacing any leaf.

    Evidence publication is supported only where libc exposes Linux
    ``renameat2(RENAME_NOREPLACE)``.  Failing closed on an older platform is
    safer than a check-then-rename fallback that can replace a concurrently
    created empty directory or symlink.
    """

    if any(
        not name or name in {".", ".."} or Path(name).name != name
        for name in (source_name, destination_name)
    ):
        raise ValueError("atomic publication names must be plain sibling leaves")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace directory publication is unavailable",
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "artifact target appeared before atomic no-replace publication",
            destination_name,
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _checksum_index(directory: Path) -> tuple[str, str]:
    index_path = directory / "SHA256SUMS"
    sidecar_path = directory / "SHA256SUMS.sha256"
    payloads = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.name not in {index_path.name, sidecar_path.name}
        ),
        key=lambda path: path.name,
    )
    if not payloads:
        raise RuntimeError("refusing an empty M2 artifact transaction")
    lines: list[str] = []
    for path in payloads:
        if any(character in path.name for character in ("\n", "\r", "\\")):
            raise ValueError(f"unsafe checksum filename: {path.name!r}")
        lines.append(f"{sha256_file(path)}  {path.name}\n")
    _write_new_file(index_path, "".join(lines).encode("ascii"))
    index_hash = sha256_file(index_path)
    _write_new_file(
        sidecar_path, f"{index_hash}  {index_path.name}\n".encode("ascii")
    )
    for line in lines:
        expected, name = line.rstrip("\n").split("  ", 1)
        if sha256_file(directory / name) != expected:
            raise RuntimeError(f"checksum verification failed for {name}")
    if sha256_file(index_path) != index_hash:
        raise RuntimeError("checksum-index verification failed")
    return index_hash, sha256_file(sidecar_path)


def _verify_staged_transaction_unchanged(
    staging: Path,
    *,
    staging_identity: tuple[int, int, int],
    expected_payloads: Mapping[str, bytes],
) -> None:
    """Re-prove the exact staged inode and bytes after the caller's last check."""

    observed_stat = os.stat(staging, follow_symlinks=False)
    observed_identity = (
        observed_stat.st_dev,
        observed_stat.st_ino,
        observed_stat.st_mode,
    )
    if observed_identity != staging_identity or (observed_stat.st_mode & 0o170000) != 0o040000:
        raise RuntimeError("M2 staging directory identity changed after pre-publish check")
    observed_names = {entry.name for entry in staging.iterdir()}
    if observed_names != set(expected_payloads):
        raise RuntimeError("M2 staged file set changed after pre-publish check")
    for name, expected in expected_payloads.items():
        path = staging / name
        path_stat = os.stat(path, follow_symlinks=False)
        if (path_stat.st_mode & 0o170000) != 0o100000 or path.is_symlink():
            raise RuntimeError(
                f"M2 staged artifact type changed after pre-publish check: {name}"
            )
        if path_stat.st_size != len(expected) or path.read_bytes() != expected:
            raise RuntimeError(
                f"M2 staged artifact bytes changed after pre-publish check: {name}"
            )


def publish_report_transactionally(
    report: Mapping[str, Any],
    artifact_directory: Path,
    *,
    additional_payloads: Mapping[str, bytes] | None = None,
    artifact_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    pre_publish_check: Callable[[], None] | None = None,
) -> PublishedArtifacts:
    """Publish one immutable sibling-staged transaction or nothing."""

    requested_artifact = Path(artifact_directory).expanduser()
    if os.path.lexists(requested_artifact):
        raise FileExistsError(
            f"refusing existing requested artifact leaf: {requested_artifact}"
        )
    try:
        report_snapshot = _json_value(report)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("M2 publication report is not strict JSON") from exc
    if type(report_snapshot) is not dict:
        raise RuntimeError("M2 publication report must be a JSON object")
    report = report_snapshot
    requested_absolute = requested_artifact.absolute()
    artifact = validate_artifact_directory(requested_artifact)
    if artifact != requested_absolute or os.path.lexists(requested_artifact):
        raise UnsafeArtifactPath(
            "artifact path changed through a symlink or noncanonical component"
        )
    extras = dict(additional_payloads or {})
    try:
        metadata_snapshot = _json_value(artifact_metadata or {})
    except (TypeError, ValueError) as exc:
        raise RuntimeError("M2 artifact metadata is not strict JSON") from exc
    if type(metadata_snapshot) is not dict:
        raise RuntimeError("M2 artifact metadata must be a JSON object")
    metadata = metadata_snapshot
    reserved = {
        "m2-validation.json",
        "m2-validation.json.sha256",
        "manifest.json",
        "manifest.json.sha256",
        "SHA256SUMS",
        "SHA256SUMS.sha256",
    }
    for name, payload in extras.items():
        if (
            not name
            or Path(name).name != name
            or name in reserved
            or any(character in name for character in ("\n", "\r", "\\"))
            or type(payload) is not bytes
        ):
            raise ValueError(f"unsafe M2 additional payload: {name!r}")
    if set(metadata) != set(extras):
        raise RuntimeError("M2 artifact metadata must cover exactly every extra payload")
    normalized_metadata: dict[str, Mapping[str, Any]] = {}
    for name, payload in extras.items():
        row = _publication_mapping(metadata[name], label=f"artifact metadata {name}")
        expected = {
            "filename": name,
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"M2 artifact metadata differs from payload bytes: {name}")
        normalized_metadata[name] = row
    evidence_class = _classify_report_for_publication(
        report,
        extras,
        normalized_metadata,
        artifact_leaf=artifact.name,
    )
    if (
        evidence_class
        in {"canonical_stage_evidence", "noncanonical_precommit_candidate"}
        and not callable(pre_publish_check)
    ):
        raise RuntimeError(
            "canonical or precommit M2 production evidence requires a final "
            "pre_publish_check callback"
        )
    if os.path.lexists(artifact):
        raise FileExistsError(f"refusing existing artifact target: {artifact}")
    if not artifact.parent.is_dir():
        raise FileNotFoundError(f"artifact parent is absent: {artifact.parent}")
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor: int | None = os.open(artifact.parent, parent_flags)
    descriptor_stat = os.fstat(parent_descriptor)
    path_stat = os.stat(artifact.parent, follow_symlinks=False)
    if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
        path_stat.st_dev,
        path_stat.st_ino,
    ):
        os.close(parent_descriptor)
        raise RuntimeError("artifact parent identity changed during preflight")
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{artifact.name}.staging-", dir=artifact.parent)
        )
        os.chmod(staging, 0o700)
        initial_staging_stat = os.stat(staging, follow_symlinks=False)
        staging_identity = (
            initial_staging_stat.st_dev,
            initial_staging_stat.st_ino,
            initial_staging_stat.st_mode,
        )
        for name in sorted(extras):
            _write_new_file(staging / name, extras[name])
        report_payload = stable_json_bytes(report)
        report_hash = sha256_bytes(report_payload)
        _write_new_file(staging / "m2-validation.json", report_payload)
        _write_new_file(
            staging / "m2-validation.json.sha256",
            f"{report_hash}  m2-validation.json\n".encode("ascii"),
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "evidence_class": evidence_class,
            "evaluation_stage_advancement_passed": report["summary"][
                "stage_advancement_passed"
            ],
            "report": {"filename": "m2-validation.json", "sha256": report_hash},
            "repository": {
                "root": report["repository"]["repository_root"],
                "branch": report["repository"]["branch"],
                "head": report["repository"]["head"],
                "git_directory": report["repository"]["git_directory"],
                "git_common_directory": report["repository"][
                    "git_common_directory"
                ],
            },
            "research_inputs": report["research_inputs"],
            "harness_source_sha256": report["harness_source_sha256"],
            "bound_source_sha256": report["source"]["source_sha256"],
            "bound_local_render_data_sha256": report["source"].get(
                "local_render_data_sha256", {}
            ),
            "stage_advancement_passed": (
                evidence_class == "canonical_stage_evidence"
            ),
            "canonical_blockers": (
                report["summary"]["canonical_blockers"]
                if evidence_class != "noncanonical_precommit_candidate"
                else ["REPOSITORY_PRECOMMIT_STATE"]
            ),
            "claim_boundary": report["claim_boundary"],
            "artifacts": dict(sorted(normalized_metadata.items())),
        }
        manifest_payload = stable_json_bytes(manifest)
        manifest_hash = sha256_bytes(manifest_payload)
        _write_new_file(staging / "manifest.json", manifest_payload)
        _write_new_file(
            staging / "manifest.json.sha256",
            f"{manifest_hash}  manifest.json\n".encode("ascii"),
        )

        parsed_report = json.loads((staging / "m2-validation.json").read_text())
        parsed_manifest = json.loads((staging / "manifest.json").read_text())
        if parsed_manifest["report"]["sha256"] != sha256_file(
            staging / "m2-validation.json"
        ):
            raise RuntimeError("manifest/report identity verification failed")
        if parsed_manifest["repository"]["head"] != parsed_report["repository"]["head"]:
            raise RuntimeError("manifest repository identity differs from report")
        if parsed_manifest["claim_boundary"] != parsed_report["claim_boundary"]:
            raise RuntimeError("manifest claim boundary differs from report")
        if parsed_manifest["artifacts"] != dict(sorted(normalized_metadata.items())):
            raise RuntimeError("manifest artifact metadata changed during staging")
        if parsed_report.get("gallery_artifacts", {}) != parsed_manifest["artifacts"]:
            if normalized_metadata:
                raise RuntimeError("report/manifest gallery artifact maps differ")
        if (
            parsed_manifest["bound_source_sha256"]
            != parsed_report["source"]["source_sha256"]
        ):
            raise RuntimeError("manifest source identities differ from report")
        if parsed_manifest["bound_local_render_data_sha256"] != parsed_report[
            "source"
        ].get("local_render_data_sha256", {}):
            raise RuntimeError("manifest local render-data identities differ from report")
        for name, row in normalized_metadata.items():
            path = staging / name
            if path.is_symlink() or path.stat().st_size != row["bytes"]:
                raise RuntimeError(f"staged M2 artifact identity failed: {name}")
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"staged M2 artifact SHA-256 failed: {name}")
        checksum_hash, checksum_sidecar_hash = _checksum_index(staging)
        expected_staged_payloads = {
            **extras,
            "m2-validation.json": report_payload,
            "m2-validation.json.sha256": (
                f"{report_hash}  m2-validation.json\n".encode("ascii")
            ),
            "manifest.json": manifest_payload,
            "manifest.json.sha256": (
                f"{manifest_hash}  manifest.json\n".encode("ascii")
            ),
        }
        expected_checksum_payload = "".join(
            f"{sha256_bytes(payload)}  {name}\n"
            for name, payload in sorted(expected_staged_payloads.items())
        ).encode("ascii")
        expected_staged_payloads["SHA256SUMS"] = expected_checksum_payload
        expected_staged_payloads["SHA256SUMS.sha256"] = (
            f"{sha256_bytes(expected_checksum_payload)}  SHA256SUMS\n".encode(
                "ascii"
            )
        )
        if (
            checksum_hash != sha256_bytes(expected_checksum_payload)
            or checksum_sidecar_hash
            != sha256_bytes(expected_staged_payloads["SHA256SUMS.sha256"])
        ):
            raise RuntimeError("M2 checksum transaction differs from staged payloads")

        if pre_publish_check is not None:
            pre_publish_check()

        _verify_staged_transaction_unchanged(
            staging,
            staging_identity=staging_identity,
            expected_payloads=expected_staged_payloads,
        )

        current_parent_stat = os.stat(artifact.parent, follow_symlinks=False)
        if (current_parent_stat.st_dev, current_parent_stat.st_ino) != (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ):
            raise RuntimeError("artifact parent identity changed before publication")

        os.chmod(staging, 0o755)
        _fsync_directory(staging)
        _rename_directory_noreplace(
            parent_descriptor,
            staging.name,
            artifact.name,
        )
        staging = None
        _fsync_directory(artifact.parent)
        return PublishedArtifacts(
            directory=artifact,
            report_sha256=report_hash,
            manifest_sha256=manifest_hash,
            checksum_index_sha256=checksum_hash,
        )
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tat_truecolor_m2_validate",
        description="Independent M2 domain/coverage mutation oracle",
    )
    parser.add_argument("--enforce-through", choices=("M2",), required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--canonical-corpus",
        "--corpus-root",
        dest="corpus_root",
        type=Path,
        help="external directory containing the four pinned Melissa ABI objects",
    )
    parser.add_argument(
        "--verified-m1-artifact",
        "--m1-artifact-dir",
        dest="m1_artifact_dir",
        type=Path,
        help="immutable, checksum-verified canonical M1 artifact directory",
    )
    parser.add_argument(
        "--rayleigh-dir",
        "--rayleigh-data-dir",
        dest="rayleigh_data_dir",
        type=Path,
        help="external writable pyspectral RSR/Rayleigh data directory",
    )
    parser.add_argument(
        "--rayleigh-config",
        "--pyspectral-config",
        dest="pyspectral_config",
        type=Path,
        help="pyspectral YAML that binds rsr_dir/rayleigh_dir",
    )
    parser.add_argument(
        "--v30-acquisition",
        "--v30-acquisition-dir",
        dest="v30_acquisition_dir",
        type=Path,
        help="verified offline V30 acquisition package directory",
    )
    parser.add_argument(
        "--v30-raw",
        "--v30-raw-dir",
        dest="v30_raw_dir",
        type=Path,
        help="directory containing exactly the 76 pinned V30 NetCDF objects",
    )
    parser.add_argument(
        "--v30-land-zip",
        type=Path,
        help="pinned Natural Earth 1:10m land ZIP used for V30 stationarity",
    )
    parser.add_argument(
        "--runner-image-id",
        help=(
            "external operator declaration of the immutable Docker image ID; "
            "the container cannot self-attest this value"
        ),
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help=(
            "explicitly permit a non-closing scratch artifact; the leaf name "
            "must start with scratch-"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    prior_pyspectral_config = os.environ.get("PSP_CONFIG_FILE")
    run = None
    try:
        paths = preflight_paths(arguments)
        research = verify_research_root(paths.research_root)
        repository_identity = collect_repository_identity()
        source_metadata = collect_m2_source_metadata()
        dependency_metadata = collect_dependency_metadata()
        harness_hashes = m2_harness_source_hashes(REPO_ROOT)

        if arguments.contract_only:
            run = run_m2_contract_oracle(research)
            if run.stage_advancement_passed:
                raise RuntimeError(
                    "contract-only oracle unexpectedly claimed canonical closure"
                )
            report = build_m2_report(
                run,
                research=research,
                repository_identity=repository_identity,
                source_metadata=source_metadata,
                dependency_metadata=dependency_metadata,
                harness_source_hashes=harness_hashes,
                command="contract-oracle",
            )

            def verify_inputs_unchanged() -> None:
                if collect_repository_identity() != repository_identity:
                    raise RuntimeError("repository identity changed during M2 oracle run")
                if collect_m2_source_metadata() != source_metadata:
                    raise RuntimeError(
                        "product source metadata changed during M2 oracle run"
                    )
                if collect_dependency_metadata() != dependency_metadata:
                    raise RuntimeError(
                        "dependency metadata changed during M2 oracle run"
                    )
                if m2_harness_source_hashes(REPO_ROOT) != harness_hashes:
                    raise RuntimeError("M2 harness sources changed during M2 oracle run")
                if (
                    verify_research_root(paths.research_root).as_dict()
                    != research.as_dict()
                ):
                    raise RuntimeError(
                        "canonical research bindings changed during M2 oracle run"
                    )

            published = publish_report_transactionally(
                report,
                paths.artifact_directory,
                pre_publish_check=verify_inputs_unchanged,
            )
        else:
            (
                corpus_root,
                m1_artifact_directory,
                rayleigh_data_directory,
                pyspectral_config,
                v30_acquisition_directory,
                v30_raw_directory,
                v30_land_zip,
            ) = _require_production_paths(paths)
            dependency_metadata = collect_m2_dependency_metadata(dependency_metadata)
            require_production_dependency_identity(dependency_metadata)
            os.environ["PSP_CONFIG_FILE"] = str(pyspectral_config)
            rayleigh = run_rayleigh_startup_check(
                rayleigh_data_directory,
                config_file=pyspectral_config,
            )
            if rayleigh.passed is not True:
                raise RuntimeError(
                    f"M2 Rayleigh startup did not pass: {rayleigh.as_dict()}"
                )

            synthetic = run_m2_contract_oracle(research)
            m1_baseline = verify_m1_artifact_baseline(m1_artifact_directory)
            evidence = measure_canonical_m2(
                research,
                corpus_root=corpus_root,
                m1_artifact_directory=m1_artifact_directory,
                include_limb_v41=True,
            )
            if type(evidence) is not CanonicalM2Evidence:
                raise TypeError("canonical measurement returned an unregistered object")
            if m1_baseline.as_dict() != evidence.m1_baseline.as_dict():
                raise RuntimeError("canonical measurement used a different M1 baseline")
            evidence_summary = _json_value(evidence.summary_dict())
            evidence_summary_sha256 = sha256_bytes(
                _strict_json_bytes(evidence_summary)
            )
            corpus_binding = canonical_corpus_binding_from_evidence(
                evidence, corpus_root
            )

            repeatability_data, repeatability_identities = (
                load_canonical_melissa_physical(research, corpus_root)
            )
            if _reloaded_identity_rows(repeatability_identities) != tuple(
                corpus_binding["objects"]
            ):
                raise RuntimeError(
                    "independent V25 reload used different canonical objects"
                )
            repeatability = capture_three_process_repeatability(
                repeatability_data,
                bbox=evidence.bbox,
            )
            del repeatability_data
            if type(repeatability) is not RepeatabilityCapture or not repeatability.passed:
                raise RuntimeError("canonical V25 repeatability did not pass")
            repeatability_summary = _json_value(repeatability.summary_dict())

            v30_corpus = verify_v30_corpus(
                v30_acquisition_directory,
                v30_raw_directory,
                v30_land_zip,
            )
            if (
                v30_corpus.acquisition_directory != v30_acquisition_directory
                or v30_corpus.raw_directory != v30_raw_directory
                or v30_corpus.land_zip_path != v30_land_zip
            ):
                raise RuntimeError("verified V30 corpus paths differ from preflight")
            v30_capture = capture_v30_sequence(v30_corpus)
            if type(v30_capture) is not V30SequenceCapture:
                raise TypeError("V30 producer returned an unregistered capture")
            v30_capture_summary = _json_value(v30_capture.summary_dict())

            evaluation = evaluate_canonical_m2(
                synthetic,
                evidence,
                repeatability=repeatability,
                sequence_binding=v30_capture.binding,
            )
            if type(evaluation) is not M2ProductionEvaluation:
                raise TypeError("M2 evaluator returned an unregistered evaluation")
            if evaluation.binding_issues:
                raise RuntimeError(
                    f"canonical M2 evidence has binding issues: {evaluation.binding_issues}"
                )
            if evaluation.stage_advancement_passed is not True:
                raise RuntimeError(
                    "canonical M2 evaluation did not pass stage advancement: "
                    f"{evaluation.oracle.canonical_blockers}"
                )
            run = evaluation.oracle
            evaluation_summary = _json_value(evaluation.as_dict())

            gallery = build_m2_gallery_payloads(
                evidence,
                repeatability_summary=repeatability_summary,
                sequence_summary=evaluation.sequence.as_dict(),
                expected_evidence_summary_sha256=evidence_summary_sha256,
            )
            v30_payloads, v30_metadata = build_v30_patch_payloads(
                v30_capture, evaluation.sequence
            )
            additional_payloads, artifact_metadata = assemble_production_payloads(
                gallery, v30_payloads, v30_metadata
            )
            input_bindings = {
                "canonical_corpus": corpus_binding,
                "verified_m1_artifact": _json_value(m1_baseline.as_dict()),
                "rayleigh_startup": _json_value(rayleigh.as_dict()),
                "v30_corpus": _json_value(v30_corpus.summary_dict()),
                "runner_image": _json_value(M2_RUNNER_IMAGE_PROVENANCE),
            }
            production_report_evaluation = dict(evaluation_summary)
            production_report_evaluation["input_bindings"] = input_bindings
            production_report_evaluation["rayleigh_startup"] = _json_value(
                rayleigh.as_dict()
            )
            report = build_m2_report(
                run,
                research=research,
                repository_identity=repository_identity,
                source_metadata=source_metadata,
                dependency_metadata=dependency_metadata,
                harness_source_hashes=harness_hashes,
                command="canonical-production",
                production_evaluation=production_report_evaluation,
                v30_capture=v30_capture_summary,
                gallery_artifacts=artifact_metadata,
                limitations=PRODUCTION_LIMITATIONS,
            )
            report["runner_image"] = _json_value(M2_RUNNER_IMAGE_PROVENANCE)
            report["invocation"] = {
                "command": "canonical-production",
                "runner_image_id": arguments.runner_image_id,
                "runner_image_identity_source": "external_operator_declared",
                "container_self_attested": False,
            }
            report.pop("evidence_payload_sha256", None)
            report["evidence_payload_sha256"] = sha256_bytes(
                stable_json_bytes(report)
            )

            def verify_inputs_unchanged() -> None:
                # This callback is invoked after every payload/checksum has been
                # verified, immediately before the no-replace directory rename.
                if collect_repository_identity() != repository_identity:
                    raise RuntimeError(
                        "repository identity changed during canonical M2 run"
                    )
                if collect_m2_source_metadata() != source_metadata:
                    raise RuntimeError(
                        "product source metadata changed during canonical M2 run"
                    )
                current_dependencies = collect_m2_dependency_metadata()
                require_production_dependency_identity(current_dependencies)
                if current_dependencies != dependency_metadata:
                    raise RuntimeError(
                        "dependency metadata changed during canonical M2 run"
                    )
                if m2_harness_source_hashes(REPO_ROOT) != harness_hashes:
                    raise RuntimeError(
                        "M2 harness sources changed during canonical M2 run"
                    )
                if (
                    verify_research_root(paths.research_root).as_dict()
                    != research.as_dict()
                ):
                    raise RuntimeError(
                        "canonical research bindings changed during M2 run"
                    )
                if (
                    collect_canonical_corpus_binding(corpus_root, corpus_binding)
                    != corpus_binding
                ):
                    raise RuntimeError("canonical corpus binding changed during M2 run")
                current_v30 = verify_v30_corpus(
                    v30_acquisition_directory,
                    v30_raw_directory,
                    v30_land_zip,
                )
                if _json_value(current_v30.summary_dict()) != input_bindings["v30_corpus"]:
                    raise RuntimeError("V30 acquisition/raw/land binding changed")
                configured = os.environ.get("PSP_CONFIG_FILE")
                if configured is None or Path(configured).resolve(strict=True) != pyspectral_config:
                    raise RuntimeError("PSP_CONFIG_FILE changed during canonical M2 run")
                current_rayleigh = run_rayleigh_startup_check(
                    rayleigh_data_directory,
                    config_file=pyspectral_config,
                    expected_production_evidence=rayleigh,
                )
                if _json_value(current_rayleigh.as_dict()) != input_bindings[
                    "rayleigh_startup"
                ]:
                    raise RuntimeError("Rayleigh startup binding changed during M2 run")
                if _json_value(evidence.summary_dict()) != evidence_summary:
                    raise RuntimeError("canonical evidence arrays changed after measurement")
                if _json_value(repeatability.summary_dict()) != repeatability_summary:
                    raise RuntimeError("V25 retained renderer bytes changed")
                if _json_value(v30_capture.summary_dict()) != v30_capture_summary:
                    raise RuntimeError("V30 capture metadata changed")
                rebound = evaluate_canonical_m2(
                    synthetic,
                    evidence,
                    repeatability=repeatability,
                    sequence_binding=v30_capture.binding,
                )
                # canonical_evidence_binding_issues(), reached by the evaluator,
                # re-verifies every indexed M1 artifact byte here.  Avoid a
                # second consecutive full M1 checksum traversal.
                if (
                    rebound.binding_issues
                    or rebound.stage_advancement_passed is not True
                    or _json_value(rebound.as_dict()) != evaluation_summary
                ):
                    raise RuntimeError(
                        "canonical evaluation changed before M2 publication"
                    )

            published = publish_report_transactionally(
                report,
                paths.artifact_directory,
                additional_payloads=additional_payloads,
                artifact_metadata=artifact_metadata,
                pre_publish_check=verify_inputs_unchanged,
            )
    except (
        FileExistsError,
        OSError,
        PermissionError,
        ResearchIntegrityError,
        RuntimeError,
        subprocess.CalledProcessError,
        UnsafeArtifactPath,
        TypeError,
        ValueError,
    ) as exc:
        print(f"tat_truecolor_m2_validate: {exc}", file=sys.stderr)
        return 2

    finally:
        if prior_pyspectral_config is None:
            os.environ.pop("PSP_CONFIG_FILE", None)
        else:
            os.environ["PSP_CONFIG_FILE"] = prior_pyspectral_config

    if run is None:
        raise RuntimeError("M2 validator completed without a run")
    summary = {
        "artifact_directory": str(published.directory),
        "report_sha256": published.report_sha256,
        "manifest_sha256": published.manifest_sha256,
        "sha256sums_sha256": published.checksum_index_sha256,
        "contract_passed": run.contract_passed,
        "stage_advancement_passed": run.stage_advancement_passed,
        "canonical_blockers": list(run.canonical_blockers),
        "command": "contract-oracle" if arguments.contract_only else "canonical-production",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if run.stage_advancement_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
