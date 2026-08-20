"""Safety and mutation tests for the external-only M0 corpus resolver."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from validation.truecolor.corpus import (
    SAMPLE_MANIFEST_COLUMNS,
    ChecksumMismatchError,
    CorpusError,
    CorpusResolver,
    DownloadProtocolError,
    EntryKind,
    ManifestError,
    NetworkDisabledError,
    SizeMismatchError,
    UnsafeRootError,
    load_sample_manifest,
)


def _manifest_row(
    payload: bytes,
    *,
    identifier: str = "object.bin",
    sha256: str | None = None,
    size: int | None = None,
    source_url: str = "https://example.invalid/object.bin",
) -> dict[str, str]:
    row = {column: "" for column in SAMPLE_MANIFEST_COLUMNS}
    row.update(
        {
            "scene_id": "TEST_SCENE",
            "storm": "TEST",
            "basin": "North Atlantic",
            "datetime_utc": "2026-08-20T00:00:00Z",
            "sensor": "ABI",
            "platform": "GOES-19",
            "product": "ABI-L2-CMIP",
            "band_role": "red",
            "processing_level": "L2 CMI",
            "source_url": source_url,
            "file_identifier": identifier,
            "bytes": str(len(payload) if size is None else size),
            "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
        }
    )
    return row


def _gibs_row() -> dict[str, str]:
    row = _manifest_row(
        b"",
        identifier="VIIRS_NOAA20_2025-10-27",
        size=0,
        sha256="(WMS response, not archived)",
        source_url="https://gibs.earthdata.nasa.gov/wms/example",
    )
    row.update(
        {
            "scene_id": "GIBS_TEST",
            "sensor": "VIIRS",
            "platform": "NOAA20",
            "product": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
        }
    )
    return row


def _write_manifest(path: Path, *rows: dict[str, str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        status_code: int,
        headers: dict[str, str] | None = None,
        fail_after_first_chunk: bool = False,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.fail_after_first_chunk = fail_after_first_chunk
        self.closed = False

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self.payload), chunk_size):
            yield self.payload[index : index + chunk_size]
            if self.fail_after_first_chunk:
                raise ConnectionError("injected interruption")

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, *responses: _Response) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url, *, headers, stream, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "stream": stream,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected network call")
        return self.responses.pop(0)


def _resolver(tmp_path: Path, manifest: Path, session=None, *, chunk_size=4):
    return CorpusResolver(
        manifest_path=manifest,
        corpus_root=tmp_path / "external-corpus",
        artifact_root=tmp_path / "external-artifacts",
        session=session,
        chunk_size=chunk_size,
    )


def test_manifest_parser_distinguishes_objects_from_gibs_references(tmp_path):
    payload = b"pinned-object"
    manifest = _write_manifest(
        tmp_path / "sample_manifest.csv", _manifest_row(payload), _gibs_row()
    )

    object_entry, gibs_entry = load_sample_manifest(manifest)
    assert object_entry.kind is EntryKind.CHECKSUMMED_OBJECT
    assert object_entry.expected_size == len(payload)
    assert object_entry.expected_sha256 == hashlib.sha256(payload).hexdigest()
    assert gibs_entry.kind is EntryKind.EXTERNAL_REFERENCE
    assert gibs_entry.expected_size == 0
    assert gibs_entry.expected_sha256 is None
    assert gibs_entry.digest_note == "(WMS response, not archived)"


def test_manifest_requires_exact_schema_and_valid_size_digest_pair(tmp_path):
    bad_schema = tmp_path / "bad-schema.csv"
    bad_schema.write_text("scene_id,bytes,sha256\nS,1,00\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="pinned schema"):
        load_sample_manifest(bad_schema)

    bad_digest = _write_manifest(
        tmp_path / "bad-digest.csv",
        _manifest_row(b"payload", sha256="not-a-digest"),
    )
    with pytest.raises(ManifestError, match="requires a 64-hex"):
        load_sample_manifest(bad_digest)


@pytest.mark.parametrize("unsafe_root", ("corpus", "artifact"))
def test_unsafe_git_worktree_roots_are_rejected_before_creation(
    tmp_path, unsafe_root
):
    payload = b"payload"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(payload))
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    corpus_root = tmp_path / "external-corpus"
    artifact_root = tmp_path / "external-artifacts"
    if unsafe_root == "corpus":
        unsafe_path = corpus_root = worktree / "corpus"
    else:
        unsafe_path = artifact_root = worktree / "artifacts"

    with pytest.raises(UnsafeRootError, match="inside a Git worktree"):
        CorpusResolver(
            manifest_path=manifest,
            corpus_root=corpus_root,
            artifact_root=artifact_root,
        )
    assert not unsafe_path.exists()


def test_network_is_disabled_by_default_and_progress_is_persisted(tmp_path):
    payload = b"network-required"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(payload))
    session = _Session()
    resolver = _resolver(tmp_path, manifest, session)

    with pytest.raises(NetworkDisabledError):
        resolver.resolve(resolver.entries[0])
    assert session.calls == []
    progress = json.loads(resolver.progress_path.read_text(encoding="utf-8"))
    assert progress["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    record = progress["objects"]["object.bin"]
    assert record["status"] == "network_disabled"
    assert record["downloaded_bytes"] == 0


def test_progress_cannot_resume_after_same_path_manifest_mutation(tmp_path):
    payload = b"manifest-bound-object"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(payload))
    resolver = _resolver(tmp_path, manifest)
    with pytest.raises(NetworkDisabledError):
        resolver.resolve(resolver.entries[0])

    changed = _manifest_row(payload)
    changed["caveats"] = "mutated after progress was recorded"
    _write_manifest(manifest, changed)

    with pytest.raises(CorpusError, match="different sample-manifest bytes"):
        _resolver(tmp_path, manifest)


def test_download_verifies_then_atomically_publishes(tmp_path):
    payload = b"correct-pinned-object"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(payload))
    response = _Response(
        payload,
        status_code=200,
        headers={"Content-Length": str(len(payload))},
    )
    resolver = _resolver(tmp_path, manifest, _Session(response))

    destination = resolver.resolve(resolver.entries[0], allow_network=True)

    assert destination is not None
    assert destination.read_bytes() == payload
    assert not destination.with_name(destination.name + ".part").exists()
    assert response.closed
    progress = json.loads(resolver.progress_path.read_text(encoding="utf-8"))
    assert progress["objects"][destination.name]["status"] == "verified"


def test_interrupted_download_resumes_from_part_with_http_range(tmp_path):
    payload = b"0123456789abcdef"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(payload))
    interrupted = _Response(
        payload,
        status_code=200,
        headers={"Content-Length": str(len(payload))},
        fail_after_first_chunk=True,
    )
    session = _Session(interrupted)
    resolver = _resolver(tmp_path, manifest, session, chunk_size=4)
    entry = resolver.entries[0]

    with pytest.raises(ConnectionError, match="injected interruption"):
        resolver.resolve(entry, allow_network=True)
    partial = resolver.corpus_root / "object.bin.part"
    assert partial.read_bytes() == payload[:4]
    progress = json.loads(resolver.progress_path.read_text(encoding="utf-8"))
    assert progress["objects"]["object.bin"]["status"] == "partial"
    assert progress["objects"]["object.bin"]["downloaded_bytes"] == 4

    remainder = payload[4:]
    resumed_response = _Response(
        remainder,
        status_code=206,
        headers={
            "Content-Range": f"bytes 4-{len(payload) - 1}/{len(payload)}",
            "Content-Length": str(len(remainder)),
        },
    )
    resumed_session = _Session(resumed_response)
    resumed_resolver = _resolver(
        tmp_path, manifest, resumed_session, chunk_size=4
    )
    destination = resumed_resolver.resolve(
        resumed_resolver.entries[0], allow_network=True
    )

    assert destination.read_bytes() == payload
    assert resumed_session.calls[0]["headers"] == {"Range": "bytes=4-"}
    assert not partial.exists()


def test_partial_resume_rejects_mutated_content_range(tmp_path):
    payload = b"0123456789"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(payload))
    response = _Response(
        payload[4:],
        status_code=206,
        headers={
            "Content-Range": f"bytes 3-{len(payload) - 1}/{len(payload)}",
            "Content-Length": str(len(payload) - 4),
        },
    )
    resolver = _resolver(tmp_path, manifest, _Session(response))
    partial = resolver.corpus_root / "object.bin.part"
    partial.write_bytes(payload[:4])

    with pytest.raises(DownloadProtocolError, match="unsafe Content-Range"):
        resolver.resolve(resolver.entries[0], allow_network=True)
    assert partial.read_bytes() == payload[:4]


def test_wrong_checksum_mutation_is_not_published(tmp_path):
    expected = b"expected"
    mutation = b"mutated!"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(expected))
    response = _Response(
        mutation,
        status_code=200,
        headers={"Content-Length": str(len(mutation))},
    )
    resolver = _resolver(tmp_path, manifest, _Session(response))

    with pytest.raises(ChecksumMismatchError):
        resolver.resolve(resolver.entries[0], allow_network=True)
    assert not (resolver.corpus_root / "object.bin").exists()
    assert (resolver.corpus_root / "object.bin.part").read_bytes() == mutation


def test_wrong_size_mutation_is_not_published(tmp_path):
    expected = b"expected-size"
    truncated = expected[:-1]
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(expected))
    response = _Response(truncated, status_code=200)
    resolver = _resolver(tmp_path, manifest, _Session(response))

    with pytest.raises(SizeMismatchError, match="downloaded"):
        resolver.resolve(resolver.entries[0], allow_network=True)
    assert not (resolver.corpus_root / "object.bin").exists()
    assert (resolver.corpus_root / "object.bin.part").read_bytes() == truncated


def test_oversize_partial_mutation_is_rejected_without_network(tmp_path):
    payload = b"small"
    manifest = _write_manifest(tmp_path / "sample.csv", _manifest_row(payload))
    session = _Session()
    resolver = _resolver(tmp_path, manifest, session)
    partial = resolver.corpus_root / "object.bin.part"
    partial.write_bytes(payload + b"x")

    with pytest.raises(SizeMismatchError, match="exceeds expected size"):
        resolver.resolve(resolver.entries[0], allow_network=True)
    assert session.calls == []


def test_gibs_reference_never_creates_raw_bytes_or_calls_network(tmp_path):
    manifest = _write_manifest(tmp_path / "sample.csv", _gibs_row())
    session = _Session()
    resolver = _resolver(tmp_path, manifest, session)

    assert resolver.resolve(resolver.entries[0], allow_network=True) is None
    assert session.calls == []
    assert list(resolver.corpus_root.iterdir()) == []
    progress = json.loads(resolver.progress_path.read_text(encoding="utf-8"))
    assert progress["objects"] == {}
    reference = progress["external_references"]["VIIRS_NOAA20_2025-10-27"]
    assert reference["status"] == "reference_only"
    assert reference["digest_note"] == "(WMS response, not archived)"
