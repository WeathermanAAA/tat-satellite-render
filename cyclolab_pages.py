"""CycloLab per-storm page writer - key contract (Stage 0/2).

THE EDGE CONTRACT (locked against the deployed cyclolab-router Worker,
version 8b77a818, Stage-0 gate-verified 2026-06-06): the Worker serves
``triple-a-tropics.com/cyclolab/{sid}/`` from bucket
``triple-a-tropics-media`` at key ``cyclolab/{sid}/index.html`` with the
stored Content-Type. The writer side therefore MUST produce exactly:

    key          {live_prefix}/{sid}/index.html
    content-type text/html; charset=utf-8   (R2Sink.write_html)
    bucket       R2_BUCKET env, default triple-a-tropics-media

Shadow-first discipline composes cleanly: shadow pages live under
``shadow/cyclolab/...`` which the Worker CANNOT serve (it only reads
``cyclolab/...``) - shadow content is unreachable from the edge by
construction, and promotion is just writing under the live prefix.

tests/test_cyclolab_pages.py pins the key shapes; the main repo's
tests/test_cyclolab_router.py pins the Worker's resolve() to the SAME
strings - the two suites together are the cross-repo contract.
"""
from __future__ import annotations

from storm_ids import StormIds, parse_sid

LIVE_PREFIX = "cyclolab"          # what the deployed Worker reads


def page_key(sid: str, prefix: str = LIVE_PREFIX) -> str:
    """R2 key for a storm's shell page. Validates the sid (designated
    storms only - parse_sid raises on invests/malformed)."""
    ids: StormIds = parse_sid(sid)
    return f"{prefix}/{ids.sid}/index.html"


def adv_key(sid: str, prefix: str = LIVE_PREFIX) -> str:
    """R2 key for a storm's cached advisory JSON (the §8.3 contract the
    shell hydrates from; same shape the cyclolab-adv Source writes)."""
    ids: StormIds = parse_sid(sid)
    return f"{prefix}/adv/{ids.sid}.json"


def page_url_path(sid: str) -> str:
    """The public path the Worker serves for a storm (deep-link target +
    og:url value)."""
    ids: StormIds = parse_sid(sid)
    return f"/cyclolab/{ids.sid}/"
