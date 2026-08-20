"""External-only, resumable resolver for the pinned true-color corpus.

The resolver has no default manifest location and performs no network access
unless the caller explicitly opts in.  Raw objects and the progress ledger are
both required to live outside any Git worktree.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import urlparse


SAMPLE_MANIFEST_COLUMNS = (
    "scene_id",
    "storm",
    "basin",
    "datetime_utc",
    "sensor",
    "platform",
    "product",
    "band_role",
    "processing_level",
    "source_url",
    "file_identifier",
    "bytes",
    "sha256",
    "bbox",
    "native_red_px",
    "target_grid_px",
    "uncapped_grid_px",
    "sza_min_deg",
    "sza_max_deg",
    "vza_min_deg",
    "vza_max_deg",
    "scene_properties",
    "caveats",
)
PROGRESS_SCHEMA = "tat_truecolor_corpus_progress/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class CorpusError(RuntimeError):
    """Base class for corpus safety and integrity failures."""


class ManifestError(CorpusError):
    """The supplied CSV does not satisfy the pinned manifest contract."""


class UnsafeRootError(CorpusError):
    """A writable corpus path lies in a Git worktree or is otherwise unsafe."""


class NetworkDisabledError(CorpusError):
    """A missing object would require network access without explicit opt-in."""


class DownloadProtocolError(CorpusError):
    """The server response cannot be safely appended or verified."""


class SizeMismatchError(CorpusError):
    """An object does not have its manifest-pinned size."""


class ChecksumMismatchError(CorpusError):
    """An object does not have its manifest-pinned SHA-256."""


class EntryKind(str, Enum):
    CHECKSUMMED_OBJECT = "checksummed_object"
    EXTERNAL_REFERENCE = "external_reference"


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    row_number: int
    scene_id: str
    sensor: str
    product: str
    source_url: str
    file_identifier: str
    expected_size: int
    expected_sha256: str | None
    digest_note: str | None
    kind: EntryKind
    metadata: Mapping[str, str]

    @property
    def is_checksummed_object(self) -> bool:
        return self.kind is EntryKind.CHECKSUMMED_OBJECT


class ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Iterable[bytes]: ...

    def close(self) -> None: ...


class SessionLike(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        stream: bool,
        timeout: tuple[float, float],
    ) -> ResponseLike: ...


def _is_git_worktree_path(path: Path) -> bool:
    """Return true when ``path`` is at or below a Git worktree marker."""

    resolved = path.expanduser().resolve(strict=False)
    return any(
        os.path.lexists(candidate / ".git")
        for candidate in (resolved, *resolved.parents)
    )


def require_external_root(path: str | os.PathLike[str], *, label: str) -> Path:
    """Resolve and create a writable root only after proving it is external."""

    raw = os.fspath(path)
    if not raw:
        raise UnsafeRootError(f"{label} root must be explicit and non-empty")
    resolved = Path(raw).expanduser().resolve(strict=False)
    if _is_git_worktree_path(resolved):
        raise UnsafeRootError(f"{label} root is inside a Git worktree: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise UnsafeRootError(f"{label} root is not a directory: {resolved}")
    if _is_git_worktree_path(resolved):
        raise UnsafeRootError(f"{label} root became a Git worktree: {resolved}")
    return resolved


def _validate_identifier(identifier: str, row_number: int) -> None:
    if not identifier or identifier in {".", ".."}:
        raise ManifestError(f"row {row_number}: empty or unsafe file_identifier")
    if Path(identifier).name != identifier or "/" in identifier or "\\" in identifier:
        raise ManifestError(
            f"row {row_number}: file_identifier must be a plain filename: {identifier!r}"
        )


def _parse_entry(row: dict[str, str], row_number: int) -> CorpusEntry:
    if None in row or any(value is None for value in row.values()):
        raise ManifestError(f"row {row_number}: field count differs from pinned schema")
    try:
        expected_size = int(row["bytes"])
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"row {row_number}: bytes must be a non-negative integer") from exc
    if expected_size < 0:
        raise ManifestError(f"row {row_number}: bytes must be a non-negative integer")

    source_url = row["source_url"].strip()
    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ManifestError(f"row {row_number}: unsupported source_url {source_url!r}")

    identifier = row["file_identifier"].strip()
    _validate_identifier(identifier, row_number)
    digest_field = row["sha256"].strip().lower()
    digest_is_sha256 = bool(_SHA256_RE.fullmatch(digest_field))

    if expected_size > 0:
        if not digest_is_sha256:
            raise ManifestError(
                f"row {row_number}: positive-size object requires a 64-hex SHA-256"
            )
        expected_sha256 = digest_field
        digest_note = None
        kind = EntryKind.CHECKSUMMED_OBJECT
    else:
        if digest_is_sha256:
            raise ManifestError(
                f"row {row_number}: zero-byte reference must not masquerade as an object"
            )
        expected_sha256 = None
        digest_note = row["sha256"].strip() or None
        kind = EntryKind.EXTERNAL_REFERENCE

    return CorpusEntry(
        row_number=row_number,
        scene_id=row["scene_id"].strip(),
        sensor=row["sensor"].strip(),
        product=row["product"].strip(),
        source_url=source_url,
        file_identifier=identifier,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        digest_note=digest_note,
        kind=kind,
        metadata=MappingProxyType(dict(row)),
    )


def load_sample_manifest(manifest_path: str | os.PathLike[str]) -> tuple[CorpusEntry, ...]:
    """Parse the exact supplied sample-manifest schema from an explicit path."""

    raw = os.fspath(manifest_path)
    if not raw:
        raise ManifestError("sample-manifest path must be explicit and non-empty")
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ManifestError(f"sample-manifest is not a file: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLE_MANIFEST_COLUMNS:
            raise ManifestError(
                "sample-manifest columns differ from the pinned schema: "
                f"{tuple(reader.fieldnames or ())!r}"
            )
        entries = tuple(_parse_entry(row, row_number) for row_number, row in enumerate(reader, 2))

    if not entries:
        raise ManifestError("sample-manifest contains no rows")
    identifiers: set[str] = set()
    for entry in entries:
        if entry.file_identifier in identifiers:
            raise ManifestError(
                f"row {entry.row_number}: duplicate file_identifier {entry.file_identifier!r}"
            )
        identifiers.add(entry.file_identifier)
    return entries


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Make a completed atomic publication durable across interruption."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_object(entry: CorpusEntry, path: Path) -> None:
    """Verify one pinned object without modifying it."""

    if not entry.is_checksummed_object or entry.expected_sha256 is None:
        raise ManifestError(f"{entry.file_identifier} is an external reference, not an object")
    actual_size = path.stat().st_size
    if actual_size != entry.expected_size:
        raise SizeMismatchError(
            f"{entry.file_identifier}: expected {entry.expected_size} bytes, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != entry.expected_sha256:
        raise ChecksumMismatchError(
            f"{entry.file_identifier}: expected SHA-256 {entry.expected_sha256}, "
            f"got {actual_sha256}"
        )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


class CorpusResolver:
    """Resolve pinned objects into an external corpus with durable progress."""

    def __init__(
        self,
        *,
        manifest_path: str | os.PathLike[str],
        corpus_root: str | os.PathLike[str],
        artifact_root: str | os.PathLike[str],
        session: SessionLike | None = None,
        chunk_size: int = 1024 * 1024,
        timeout: tuple[float, float] = (10.0, 120.0),
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        self.entries = load_sample_manifest(self.manifest_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.corpus_root = require_external_root(corpus_root, label="corpus")
        self.artifact_root = require_external_root(artifact_root, label="artifact")
        self.progress_path = self.artifact_root / "corpus_progress.json"
        self._progress_tmp_path = self.artifact_root / "corpus_progress.json.tmp"
        self._session = session
        self.chunk_size = chunk_size
        self.timeout = timeout
        self._progress = self._load_progress()

    def _load_progress(self) -> dict[str, Any]:
        if not self.progress_path.exists():
            return {
                "schema": PROGRESS_SCHEMA,
                "manifest_path": str(self.manifest_path),
                "manifest_sha256": self.manifest_sha256,
                "objects": {},
                "external_references": {},
            }
        if self.progress_path.is_symlink():
            raise UnsafeRootError(f"progress ledger must not be a symlink: {self.progress_path}")
        try:
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusError(f"cannot read progress ledger: {self.progress_path}") from exc
        if progress.get("schema") != PROGRESS_SCHEMA:
            raise CorpusError(f"unexpected progress schema in {self.progress_path}")
        if progress.get("manifest_path") != str(self.manifest_path):
            raise CorpusError("progress ledger belongs to a different sample manifest")
        if progress.get("manifest_sha256") != self.manifest_sha256:
            raise CorpusError(
                "progress ledger belongs to different sample-manifest bytes"
            )
        if not isinstance(progress.get("objects"), dict) or not isinstance(
            progress.get("external_references"), dict
        ):
            raise CorpusError(f"malformed progress ledger: {self.progress_path}")
        return progress

    def _persist_progress(self) -> None:
        if self._progress_tmp_path.is_symlink():
            raise UnsafeRootError(
                f"progress temporary path must not be a symlink: {self._progress_tmp_path}"
            )
        payload = json.dumps(self._progress, indent=2, sort_keys=True) + "\n"
        with self._progress_tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self._progress_tmp_path, self.progress_path)
        _fsync_directory(self.artifact_root)

    def _record_object(
        self,
        entry: CorpusEntry,
        *,
        status: str,
        downloaded_bytes: int,
        detail: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "scene_id": entry.scene_id,
            "source_url": entry.source_url,
            "expected_size": entry.expected_size,
            "expected_sha256": entry.expected_sha256,
            "downloaded_bytes": downloaded_bytes,
            "status": status,
            "path": str(self._destination(entry)),
        }
        if detail:
            record["detail"] = detail
        self._progress["objects"][entry.file_identifier] = record
        self._persist_progress()

    def _record_external_reference(self, entry: CorpusEntry) -> None:
        self._progress["external_references"][entry.file_identifier] = {
            "scene_id": entry.scene_id,
            "source_url": entry.source_url,
            "status": "reference_only",
            "digest_note": entry.digest_note,
        }
        self._persist_progress()

    def _destination(self, entry: CorpusEntry) -> Path:
        _validate_identifier(entry.file_identifier, entry.row_number)
        destination = self.corpus_root / entry.file_identifier
        if destination.parent != self.corpus_root:
            raise UnsafeRootError(f"unsafe corpus destination: {destination}")
        return destination

    def _paths(self, entry: CorpusEntry) -> tuple[Path, Path]:
        destination = self._destination(entry)
        partial = destination.with_name(destination.name + ".part")
        if destination.is_symlink() or partial.is_symlink():
            raise UnsafeRootError(f"corpus object paths must not be symlinks: {destination}")
        return destination, partial

    def _network_session(self) -> SessionLike:
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def resolve(
        self, entry: CorpusEntry, *, allow_network: bool = False
    ) -> Path | None:
        """Return a verified local object, downloading only with explicit opt-in."""

        if entry.kind is EntryKind.EXTERNAL_REFERENCE:
            self._record_external_reference(entry)
            return None

        destination, partial = self._paths(entry)
        if destination.exists():
            try:
                verify_object(entry, destination)
            except CorpusError as exc:
                self._record_object(
                    entry,
                    status="failed",
                    downloaded_bytes=destination.stat().st_size,
                    detail=str(exc),
                )
                raise
            self._record_object(
                entry, status="verified", downloaded_bytes=entry.expected_size
            )
            return destination

        if partial.exists() and partial.stat().st_size == entry.expected_size:
            try:
                verify_object(entry, partial)
            except CorpusError as exc:
                self._record_object(
                    entry,
                    status="failed",
                    downloaded_bytes=partial.stat().st_size,
                    detail=str(exc),
                )
                raise
            os.replace(partial, destination)
            _fsync_directory(self.corpus_root)
            self._record_object(
                entry, status="verified", downloaded_bytes=entry.expected_size
            )
            return destination

        if not allow_network:
            downloaded = partial.stat().st_size if partial.exists() else 0
            self._record_object(
                entry,
                status="network_disabled",
                downloaded_bytes=downloaded,
                detail="network access requires allow_network=True",
            )
            raise NetworkDisabledError(
                f"{entry.file_identifier} is not verified locally; network is disabled"
            )

        self._download_to_partial(entry, partial)
        try:
            verify_object(entry, partial)
        except CorpusError as exc:
            self._record_object(
                entry,
                status="failed",
                downloaded_bytes=partial.stat().st_size,
                detail=str(exc),
            )
            raise
        os.replace(partial, destination)
        _fsync_directory(self.corpus_root)
        self._record_object(entry, status="verified", downloaded_bytes=entry.expected_size)
        return destination

    def _download_to_partial(self, entry: CorpusEntry, partial: Path) -> None:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry.expected_size:
            error = SizeMismatchError(
                f"{entry.file_identifier}.part exceeds expected size: "
                f"{offset} > {entry.expected_size}"
            )
            self._record_object(
                entry, status="failed", downloaded_bytes=offset, detail=str(error)
            )
            raise error

        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = self._network_session().get(
            entry.source_url,
            headers=headers,
            stream=True,
            timeout=self.timeout,
        )
        try:
            append = offset > 0 and response.status_code == 206
            if offset > 0 and response.status_code == 200:
                # The server ignored Range. Restart safely rather than append a
                # complete response to an existing prefix.
                offset = 0
                append = False
            elif response.status_code == 206:
                content_range = _header(response.headers, "Content-Range")
                match = _CONTENT_RANGE_RE.fullmatch(content_range or "")
                if match is None:
                    raise DownloadProtocolError(
                        f"{entry.file_identifier}: missing/invalid Content-Range"
                    )
                start, end, total = (int(value) for value in match.groups())
                if start != offset or end < start or total != entry.expected_size:
                    raise DownloadProtocolError(
                        f"{entry.file_identifier}: unsafe Content-Range {content_range!r}"
                    )
            elif response.status_code != 200:
                raise DownloadProtocolError(
                    f"{entry.file_identifier}: HTTP {response.status_code}"
                )

            content_length = _header(response.headers, "Content-Length")
            if content_length is not None:
                try:
                    response_bytes = int(content_length)
                except ValueError as exc:
                    raise DownloadProtocolError(
                        f"{entry.file_identifier}: invalid Content-Length"
                    ) from exc
                expected_response_bytes = entry.expected_size - offset
                if response_bytes != expected_response_bytes:
                    raise SizeMismatchError(
                        f"{entry.file_identifier}: response declares {response_bytes} bytes; "
                        f"expected {expected_response_bytes}"
                    )

            mode = "ab" if append else "wb"
            self._record_object(
                entry, status="downloading", downloaded_bytes=offset
            )
            try:
                with partial.open(mode) as handle:
                    downloaded = offset
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > entry.expected_size:
                            raise SizeMismatchError(
                                f"{entry.file_identifier}: received more than "
                                f"{entry.expected_size} bytes"
                            )
                        handle.flush()
                        self._record_object(
                            entry,
                            status="downloading",
                            downloaded_bytes=downloaded,
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception as exc:
                downloaded = partial.stat().st_size if partial.exists() else 0
                self._record_object(
                    entry,
                    status="partial",
                    downloaded_bytes=downloaded,
                    detail=str(exc),
                )
                raise

            actual_size = partial.stat().st_size
            if actual_size != entry.expected_size:
                error = SizeMismatchError(
                    f"{entry.file_identifier}: expected {entry.expected_size} bytes, "
                    f"downloaded {actual_size}"
                )
                self._record_object(
                    entry,
                    status="partial",
                    downloaded_bytes=actual_size,
                    detail=str(error),
                )
                raise error
        finally:
            response.close()
