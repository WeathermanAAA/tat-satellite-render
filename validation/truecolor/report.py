"""Stable JSON reporting and external-artifact safety for validation stages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit

from . import production_adapter
from .gates import BaselineRun


REPORT_SCHEMA = "tat_truecolor_validation_report/v1"
DEPENDENCIES = (
    "numpy",
    "scipy",
    "matplotlib",
    "cartopy",
    "Pillow",
    "satpy",
    "pyspectral",
    "scikit-image",
    "tat-palettes",
)

_SECRET_METADATA_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
_TOKEN_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+|"
    r"sk-[A-Za-z0-9_-]{8,})"
)


class UnsafeArtifactPath(ValueError):
    pass


def stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sanitize_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme:
        return value
    hostname = parsed.hostname
    if hostname is None:
        sanitized_netloc = parsed.netloc
    else:
        sanitized_host = f"[{hostname}]" if ":" in hostname else hostname
        sanitized_netloc = sanitized_host
        try:
            if parsed.port is not None:
                sanitized_netloc += f":{parsed.port}"
        except ValueError:
            return "<redacted-invalid-url>"
    return urlunsplit(
        SplitResult(
            parsed.scheme,
            sanitized_netloc,
            parsed.path,
            "",
            "",
        )
    )


def sanitize_direct_url_metadata(value: Any, *, key: str | None = None) -> Any:
    """Remove credentials and token-bearing fields from PEP-610 metadata."""

    if key is not None:
        normalized_key = key.strip().lower().replace("-", "_")
        if (
            normalized_key in _SECRET_METADATA_KEYS
            or normalized_key.endswith("_token")
            or normalized_key.endswith("_secret")
            or normalized_key.endswith("_password")
            or normalized_key.endswith("_credential")
        ):
            return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): sanitize_direct_url_metadata(
                child_value,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [sanitize_direct_url_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_direct_url_metadata(item) for item in value]
    if isinstance(value, str):
        if urlsplit(value).scheme:
            return _sanitize_url(value)
        if _TOKEN_VALUE_PATTERN.search(value):
            return "<redacted>"
        return value
    return value


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def collect_source_metadata(repo_root: Path = production_adapter.REPO_ROOT) -> dict[str, Any]:
    root = repo_root.resolve()
    source_names = (
        "app.py",
        "render.py",
        "render_policy.py",
        "satellites.py",
        "truecolor.py",
        "requirements.txt",
        "runtime.txt",
    )
    hashes = {
        name: _sha256_file(root / name)
        for name in source_names
        if (root / name).is_file()
    }
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return {
        "repository_root": str(root),
        "git_head": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_status_porcelain": status,
        "source_sha256": hashes,
        "cira_stretch_source_sha256": production_adapter.cira_source_sha256(),
    }


def collect_dependency_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    direct_urls: dict[str, Mapping[str, Any] | None] = {}
    for distribution in DEPENDENCIES:
        try:
            metadata = importlib.metadata.distribution(distribution)
            versions[distribution] = metadata.version
            raw_direct_url = metadata.read_text("direct_url.json")
            direct_urls[distribution] = (
                sanitize_direct_url_metadata(json.loads(raw_direct_url))
                if raw_direct_url
                else None
            )
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
            direct_urls[distribution] = None
        except json.JSONDecodeError:
            direct_urls[distribution] = {"error": "invalid direct_url.json"}
    palette_spec = importlib.util.find_spec("tat_palettes")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "distributions": versions,
        "direct_urls": direct_urls,
        "module_origins": {
            "tat_palettes": None
            if palette_spec is None or palette_spec.origin is None
            else str(Path(palette_spec.origin).resolve()),
        },
    }


def build_report(run: BaselineRun, *, enforce_through: str | None) -> dict[str, Any]:
    gates = [gate.as_dict() for gate in run.gates]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "command": "baseline",
        "enforce_through": enforce_through,
        "research_inputs": run.research.as_dict(),
        "source": collect_source_metadata(),
        "dependencies": collect_dependency_metadata(),
        "summary": {
            "gate_count": len(gates),
            "outcomes": dict(run.counts),
            "blocking_gate_count": sum(gate["blocking"] for gate in gates),
            "m0_enforcement": run.enforcement.as_dict(),
        },
        "m0_contracts": {
            "port_fidelity": [result.as_dict() for result in run.port_fidelity],
            "rayleigh_startup": None
            if run.rayleigh_startup is None
            else run.rayleigh_startup.as_dict(),
        },
        "gates": gates,
        "artifact_hashes": {
            "validation_gates_csv": run.research.validation_gates_sha256,
            "reference_pixels_json": run.research.reference_pixels_sha256,
        },
    }
    evidence_payload = stable_json_bytes(report)
    report["artifact_hashes"]["evidence_payload_sha256"] = _sha256_bytes(evidence_payload)
    return report


def validate_artifact_directory(path: str | os.PathLike[str] | Path) -> Path:
    """Reject a prospective artifact directory inside any Git worktree."""

    resolved = Path(path).expanduser().resolve(strict=False)
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / ".git").exists():
            raise UnsafeArtifactPath(
                f"review artifacts may not be written inside Git worktree {ancestor}"
            )

    existing = resolved
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if existing.exists():
        probe = subprocess.run(
            ["git", "-C", str(existing), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            top = subprocess.run(
                ["git", "-C", str(existing), "rev-parse", "--show-toplevel"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            raise UnsafeArtifactPath(
                f"review artifacts may not be written inside Git worktree {top or existing}"
            )
    return resolved


@dataclass(frozen=True, slots=True)
class WrittenReport:
    report_path: Path
    report_sha256: str
    checksum_path: Path


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_report(
    report: Mapping[str, Any],
    artifact_directory: str | os.PathLike[str] | Path,
    *,
    filename: str = "m0-baseline.json",
) -> WrittenReport:
    directory = validate_artifact_directory(artifact_directory)
    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or any(ord(character) < 32 for character in filename)
    ):
        raise UnsafeArtifactPath(
            f"report filename must be one plain filename: {filename!r}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / filename
    checksum_path = directory / f"{filename}.sha256"
    if report_path.exists() or checksum_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {report_path}")
    payload = stable_json_bytes(report)
    digest = _sha256_bytes(payload)
    _atomic_write(report_path, payload)
    _atomic_write(checksum_path, f"{digest}  {filename}\n".encode("ascii"))
    return WrittenReport(report_path, digest, checksum_path)
