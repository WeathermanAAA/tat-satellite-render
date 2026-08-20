"""Secret-redaction contracts for external M1 dependency evidence."""

from __future__ import annotations

from validation.truecolor.report import sanitize_direct_url_metadata


def test_pep610_metadata_strips_url_credentials_query_fragment_and_nested_tokens() -> None:
    raw = {
        "url": (
            "git+https://build-user:placeholder-credential@example.invalid/"
            "org/repo.git?access_token=placeholder-query#subdirectory=package"
        ),
        "vcs_info": {
            "vcs": "git",
            "commit_id": "a" * 40,
            "requested_revision": "main",
        },
        "installer": {
            "access_token": "placeholder-token-value",
            "client-secret": "placeholder-client-secret",
            "headers": {"Authorization": "placeholder-authorization"},
        },
        "mirrors": [
            "https://name:placeholder@example.invalid/archive.whl?sig=value#fragment",
            {"api_key": "placeholder-api-key"},
        ],
        # Assemble a token-shaped value at runtime so the repository itself
        # never contains a credential-like literal that secret scanners could
        # mistake for a real key.
        "free_text": "g" + "hp_" + "A" * 36,
    }
    sanitized = sanitize_direct_url_metadata(raw)
    assert sanitized["url"] == "git+https://example.invalid/org/repo.git"
    assert sanitized["vcs_info"] == raw["vcs_info"]
    assert sanitized["installer"]["access_token"] == "<redacted>"
    assert sanitized["installer"]["client-secret"] == "<redacted>"
    assert sanitized["installer"]["headers"]["Authorization"] == "<redacted>"
    assert sanitized["mirrors"][0] == "https://example.invalid/archive.whl"
    assert sanitized["mirrors"][1]["api_key"] == "<redacted>"
    assert sanitized["free_text"] == "<redacted>"
    serialized = repr(sanitized)
    for secret in (
        "placeholder-credential",
        "placeholder-query",
        "placeholder-token-value",
        "placeholder-client-secret",
        "placeholder-authorization",
        "placeholder-api-key",
    ):
        assert secret not in serialized
