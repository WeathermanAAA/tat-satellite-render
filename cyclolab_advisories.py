"""CycloLab advisories+cone Source (CYCLOLAB_DESIGN.md §9) - Stage 1.

A poller_framework Source that rides the intensity-poller engine (same
process; per-source isolation means an NHC outage can never stale the
ACE/tracks feeds). Each poll:

1. fetches CurrentStorms.json (cheap index; the SAME authoritative list
   the live-names path reads),
2. for every DESIGNATED AL/EP/CP storm (storm_ids.parse_sid - invests
   rejected, the V1 scope guard), reads the advisory number from the
   GIS product entries,
3. change-gates on the (storm, advNum) set - the heavy KMZ work fires
   only when an advisory is NEW,
4. fetches the CONE + TRACK KMZs, parses them via kml_advisories into
   the §8.3 contract, enforces the ISSUANCE-REGRESSION guard (a parsed
   issuance older than the last cached one is rejected - the
   91W/recycled-deck lesson: never let a stale mirror replace fresh
   state), and writes ``{prefix}/adv/{sid}.json`` to R2.

JTWC (WP) derived cones: the geometry ships in derived_cone.py
(tested), but the UPSTREAM forecast-track source for WP storms still
needs live verification (design rule: source-freshness verified before
wiring) and there is no live WP designated storm to verify against -
the JTWC sub-path is therefore NOT wired in Stage 1.

Kill-switch: CYCLOLAB_ADVISORIES (house idiom; default on). Prefix
defaults to the SHADOW path - promoting to live is a config change.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
from typing import Callable, Optional

import requests

import poller_framework as pf
from kml_advisories import AdvisoryParseError, build_advisory_json
from storm_ids import InvestSidError, parse_sid

log = logging.getLogger("cyclolab-advisories")


def _env(n, d=None):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


CYCLOLAB_ENABLED = (_env("CYCLOLAB_ADVISORIES", "1") or "1").lower() \
    not in ("0", "false", "no")
# SHADOW-FIRST (the HAFS_R2_PREFIX pattern): the default prefix changes
# nothing user-facing. Promote = set CYCLOLAB_PREFIX=cyclolab.
CYCLOLAB_PREFIX = (_env("CYCLOLAB_PREFIX", "shadow/cyclolab") or "").rstrip("/")

_NHC_BASINS = {"AL", "EP", "CP"}


def _iso_z(t: dt.datetime) -> str:
    return t.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_fetch_bytes(session: requests.Session, url: str,
                         policy: pf.FetchPolicy) -> Optional[bytes]:
    """GET binary via resilient_fetch - same 404/transient semantics as
    intensity_poller._get_text, but returning raw bytes (KMZ zips)."""
    def _do():
        r = session.get(url, timeout=policy.timeout,
                        headers={"User-Agent": "tat-cyclolab/1.0"})
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            raise pf.TransientFetchError(f"{r.status_code} {url}")
        r.raise_for_status()
        return r.content
    return pf.resilient_fetch(_do, policy)


def _storm_entries(current_storms: dict) -> list[dict]:
    """The designated NHC storms this source owns, with everything the
    adv-gate + fetch need. Invests and non-NHC basins are skipped (the
    V1 scope); a storm missing its GIS products is skipped this poll
    (pre-first-advisory window)."""
    out = []
    for s in (current_storms or {}).get("activeStorms", []) or []:
        nhc_id = str(s.get("id") or "").upper()
        sid = f"NHC_{nhc_id}"
        try:
            ids = parse_sid(sid)
        except (InvestSidError, ValueError, KeyError):
            continue
        if ids.basin not in _NHC_BASINS:
            continue
        cone = s.get("trackCone") or {}
        track = s.get("forecastTrack") or {}
        if not (cone.get("kmzFile") and track.get("kmzFile")):
            continue
        try:
            adv = int(str(cone.get("advNum")).lstrip("0") or "0")
        except ValueError:
            continue
        out.append({
            "sid": sid, "ids": ids, "adv": adv,
            "cone_url": cone["kmzFile"], "track_url": track["kmzFile"],
            "text": {
                "tcp_url": (s.get("publicAdvisory") or {}).get("url"),
                "tcd_url": (s.get("forecastDiscussion") or {}).get("url"),
            },
        })
    return out


def make_advisories_source(session: requests.Session, sink: pf.Sink, *,
                           prefix: str = CYCLOLAB_PREFIX,
                           current_storms_url: str = None,
                           policy: pf.FetchPolicy = None,
                           fetch_text: Optional[Callable] = None,
                           fetch_bytes: Optional[Callable] = None,
                           clock: Callable[[], dt.datetime] = pf.utcnow,
                           ) -> pf.Source:
    """Build the Source. fetch_text/fetch_bytes are injectable so the
    offline tests drive the full poll cycle with fixture bytes."""
    import intensity_poller as ip   # late import: shared helpers/config
    current_storms_url = current_storms_url or ip.CURRENT_STORMS_URL
    policy = policy or ip.FETCH_POLICY
    get_text = fetch_text or (lambda url: ip._get_text(session, url, policy))
    get_bytes = fetch_bytes or (
        lambda url: _default_fetch_bytes(session, url, policy))

    # Per-storm ledger: {nhc_sid: {"adv": int, "issued": iso}}. In-memory;
    # a restart simply re-fetches the current advisory and overwrites the
    # SAME content at the SAME key - idempotent by construction.
    ledger: dict[str, dict] = {}

    def fetch():
        import json as _json
        text = get_text(current_storms_url)
        if not text:
            raise pf.TransientFetchError("CurrentStorms.json unavailable")
        return {"storms": _storm_entries(_json.loads(text))}

    def change_key(data):
        return tuple(sorted((e["sid"], e["adv"]) for e in data["storms"]))

    def valid_time(data):
        return None   # freshness is the advisory issuance, stamped per write

    def process(ctx: pf.ProcessContext):
        errors = []
        for e in ctx.data["storms"]:
            sid, adv = e["sid"], e["adv"]
            known = ledger.get(sid)
            if known and known["adv"] >= adv:
                continue   # adv-gate: nothing new for this storm
            try:
                cone_b = get_bytes(e["cone_url"])
                track_b = get_bytes(e["track_url"])
                if not cone_b or not track_b:
                    raise AdvisoryParseError(
                        f"{sid} adv {adv}: KMZ missing "
                        f"(cone={bool(cone_b)}, track={bool(track_b)})")
                payload = build_advisory_json(sid, cone_b, track_b,
                                              text_urls=e["text"])
                # ISSUANCE-REGRESSION GUARD: never replace cached state
                # with an OLDER advisory (stale mirror / CDN echo).
                if known and payload["issued_utc"] < known["issued"]:
                    raise AdvisoryParseError(
                        f"{sid}: parsed issuance {payload['issued_utc']} "
                        f"regresses cached {known['issued']} - rejected")
                payload["provenance"]["parsed_utc"] = _iso_z(clock())
                payload["provenance"]["source_index"] = current_storms_url
                sink.write(f"{prefix}/adv/{sid}.json", payload)
                ledger[sid] = {"adv": adv, "issued": payload["issued_utc"]}
                log.info("cyclolab adv cached: %s adv %d (%d cone vtx, "
                         "%d points)", sid, adv, len(payload["cone"]),
                         len(payload["points"]))
            except Exception as ex:  # noqa: BLE001 - isolate per storm
                errors.append(f"{sid} adv {adv}: {type(ex).__name__}: {ex}")
        if errors:
            # Raise AFTER the successful storms persisted - the engine
            # holds the change signature and retries only the failures
            # next poll (their adv numbers are still un-ledgered).
            raise RuntimeError("; ".join(errors)[:1000])

    return pf.Source(name="cyclolab-adv", fetch=fetch,
                     change_key=change_key, process=process,
                     valid_time=valid_time)
