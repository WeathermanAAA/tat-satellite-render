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
import re
from typing import Callable, Optional

import requests

import poller_framework as pf
from cyclolab_intensity import basin_entry
from cyclolab_og import render_intensity_card
from kml_advisories import (AdvisoryParseError, build_advisory_json,
                            product_text,
                            parse_next_advisory)
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

# §7.4 TEXT VERIFICATION (final-gate-3 #4, the third-strike fix): the
# TCP/TCD URLs are ROLLING pages (MIATCPEP1.shtml serves whatever
# advisory is current AT FETCH TIME). At advisory-issuance the index
# and the text pages roll at slightly different instants, so a fetch
# can return the PREVIOUS advisory's product - the repo's own captured
# fixtures show a real adv-14 TCP under an adv-13 CurrentStorms. Every
# product is therefore verified against the advisory being cached via
# its own "Advisory/Discussion Number" line before it is attached;
# intermediate letters (4A) belong to the same advisory family and
# pass. Unverifiable text (no Number line: error pages, outage
# interstitials) is NEVER attached - a placeholder panel beats wrong
# text. Mismatch/missing -> the product stays PENDING and the
# text-heal path retries it every poll (see process()).
_PRODUCT_NUM_RE = {
    "tcp": re.compile(r"Advisory\s+Number\s+(\d+)\s*[A-Za-z]?", re.I),
    "tcd": re.compile(r"Discussion\s+Number\s+(\d+)", re.I),
}


def _verified_product(kind: str, raw: str, adv: int) -> str:
    """Strip the .shtml wrapper and return the product body iff its own
    advisory number matches ``adv``; raise AdvisoryParseError otherwise."""
    body = product_text(raw)
    m = _PRODUCT_NUM_RE[kind].search(body or "")
    if not m:
        raise AdvisoryParseError(
            f"{kind.upper()} has no Advisory/Discussion Number line - "
            "unverifiable, not attached")
    got = int(m.group(1))
    if got != adv:
        raise AdvisoryParseError(
            f"{kind.upper()} page is at advisory {got}, caching advisory "
            f"{adv} - rolling URL out of sync, not attached")
    return body


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
        if ids.is_invest:          # invests have no NHC advisory product (Stage C)
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
            "name": str(s.get("name") or nhc_id),
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

    # Per-storm ledger: {nhc_sid: {"adv": int, "issued": iso,
    # "text_done": bool, "payload": dict}}. In-memory; a restart simply
    # re-fetches the current advisory and overwrites the SAME content at
    # the SAME key - idempotent by construction. ``payload`` is retained
    # (~25 KB/storm) so the TEXT-HEAL path can merge late products and
    # rewrite WITHOUT refetching the KMZs; ``text_done`` records whether
    # every text product with a URL was attached VERIFIED.
    ledger: dict[str, dict] = {}
    poll_seq = {"n": 0}

    def fetch():
        import json as _json
        text = get_text(current_storms_url)
        if not text:
            raise pf.TransientFetchError("CurrentStorms.json unavailable")
        poll_seq["n"] += 1
        return {"storms": _storm_entries(_json.loads(text))}

    def change_key(data):
        base = tuple(sorted((e["sid"], e["adv"]) for e in data["storms"]))
        # TEXT-HEAL PULSE (final-gate-3 #4): the engine only runs
        # process() on a change_key change, so a storm whose advisory
        # shipped text-less (mid-roll page / transient failure / outage
        # restart) would otherwise stay blank for the WHOLE 3-6 h
        # advisory cycle - the user's third-strike blank. While any
        # ledgered storm still owes verified text, every poll gets a
        # distinct key so process() runs and retries JUST the text
        # (two cheap .shtml GETs - no KMZ work). Settles back to the
        # plain advisory-set key once everything is attached.
        live = {e["sid"] for e in data["storms"]}
        if any(not k.get("text_done") for s, k in ledger.items()
               if s in live):
            return (base, poll_seq["n"])
        return (base,)

    def valid_time(data):
        return None   # freshness is the advisory issuance, stamped per write

    def _attach_text(entry: dict, adv: int, payload: dict) -> bool:
        """§7.4 ADVISORY TEXT PANELS: fetch + VERIFY + attach the TCP/TCD
        product text alongside the URLs - the browser cannot fetch
        nhc.gov cross-origin. Best-effort PER PRODUCT: missing text never
        blocks the cone, the countdown, or the other product. Idempotent:
        already-attached products are never refetched, so the heal path
        retries ONLY what is still owed. Returns True iff every product
        that has a URL is attached (vacuously True when no URLs - e.g. a
        basin with no NHC text products)."""
        sid = entry["sid"]
        urls = entry.get("text") or {}
        # Defensive: build_advisory_json always ships the URL dict, but a
        # malformed payload must degrade to "pending", never crash the
        # storm out of its cone.
        if not isinstance(payload.get("text"), dict):
            payload["text"] = {"tcp_url": urls.get("tcp_url"),
                               "tcd_url": urls.get("tcd_url")}
        complete = True
        tcp_raw = None
        for kind in ("tcp", "tcd"):
            url = urls.get(kind + "_url")
            if not url or payload["text"].get(kind):
                continue
            try:
                raw = get_text(url)
                if not raw:
                    raise AdvisoryParseError("product page missing/empty")
                payload["text"][kind] = _verified_product(kind, raw, adv)
                if kind == "tcp":
                    tcp_raw = raw
            except Exception as tx:  # noqa: BLE001
                complete = False
                log.warning("%s adv %d: %s text pending: %s",
                            sid, adv, kind.upper(), tx)
        # NEXT-ADVISORY countdown source (the advisory's OWN stated
        # time - never wall-clock cadence). Rides a successful TCP
        # attach; best-effort: a parse failure leaves the fields absent
        # and the shell simply shows no countdown this cycle.
        try:
            if tcp_raw:
                nxt = parse_next_advisory(tcp_raw, payload["issued_utc"])
                payload["next_advisory_utc"] = nxt["next_advisory_utc"]
                payload["next_advisory_kind"] = nxt["kind"]
                payload["next_complete_utc"] = nxt["next_complete_utc"]
                payload["next_advisory_stated"] = nxt["stated"]
        except Exception as nx:  # noqa: BLE001
            log.warning("%s adv %d: next-advisory parse skipped: %s",
                        sid, adv, nx)
        return complete

    def _heal_text(entry: dict, known: dict) -> None:
        """TEXT-HEAL (final-gate-3 #4): same advisory, text still owed -
        retry the missing products and rewrite the JSON if anything
        landed. Cheap (text GETs only; the retained payload spares the
        KMZ refetch) and best-effort: a failed heal never flags the
        source, the next poll simply tries again."""
        sid, adv = entry["sid"], known["adv"]
        payload = known["payload"]
        before = (payload["text"].get("tcp"), payload["text"].get("tcd"))
        known["text_done"] = _attach_text(entry, adv, payload)
        after = (payload["text"].get("tcp"), payload["text"].get("tcd"))
        if after != before:
            payload["provenance"]["text_healed_utc"] = _iso_z(clock())
            sink.write(f"{prefix}/adv/{sid}.json", payload)
            log.info("cyclolab adv text healed: %s adv %d (tcp=%d ch, "
                     "tcd=%d ch)", sid, adv,
                     len(payload["text"].get("tcp") or ""),
                     len(payload["text"].get("tcd") or ""))

    def process(ctx: pf.ProcessContext):
        errors = []
        live = set()
        for e in ctx.data["storms"]:
            sid, adv = e["sid"], e["adv"]
            live.add(sid)
            known = ledger.get(sid)
            if known and known["adv"] >= adv:
                # adv-gate: nothing NEW for this storm - but late text
                # products still heal in place (the third-strike fix).
                if known["adv"] == adv and not known.get("text_done"):
                    try:
                        _heal_text(e, known)
                    except Exception as hx:  # noqa: BLE001
                        log.warning("%s adv %d: text heal failed: %s",
                                    sid, adv, hx)
                continue
            try:
                cone_b = get_bytes(e["cone_url"])
                track_b = get_bytes(e["track_url"])
                if not cone_b or not track_b:
                    raise AdvisoryParseError(
                        f"{sid} adv {adv}: KMZ missing "
                        f"(cone={bool(cone_b)}, track={bool(track_b)})")
                payload = build_advisory_json(sid, cone_b, track_b,
                                              text_urls=e["text"])
                text_done = _attach_text(e, adv, payload)
                # ISSUANCE-REGRESSION GUARD: never replace cached state
                # with an OLDER advisory (stale mirror / CDN echo).
                if known and payload["issued_utc"] < known["issued"]:
                    raise AdvisoryParseError(
                        f"{sid}: parsed issuance {payload['issued_utc']} "
                        f"regresses cached {known['issued']} - rejected")
                payload["provenance"]["parsed_utc"] = _iso_z(clock())
                payload["provenance"]["source_index"] = current_storms_url
                sink.write(f"{prefix}/adv/{sid}.json", payload)
                # §8.6 second placement: the server-rendered intensity
                # OG/share card, refreshed per advisory. Best-effort and
                # honesty-guarded: no registry entry (or a sink without
                # binary writes, e.g. tests) -> no card, never a
                # borrowed envelope; failure never blocks the advisory.
                try:
                    entry = basin_entry(e["ids"].basin)
                    if entry and hasattr(sink, "write_png"):
                        png = render_intensity_card(
                            payload, entry, storm_name=e["name"])
                        sink.write_png(f"{prefix}/og/{sid}.png", png)
                except Exception as og:  # noqa: BLE001
                    log.warning("%s adv %d: OG card skipped: %s",
                                sid, adv, og)
                ledger[sid] = {"adv": adv, "issued": payload["issued_utc"],
                               "text_done": text_done, "payload": payload}
                log.info("cyclolab adv cached: %s adv %d (%d cone vtx, "
                         "%d points, text %s)", sid, adv,
                         len(payload["cone"]), len(payload["points"]),
                         "complete" if text_done else "PENDING-HEAL")
            except Exception as ex:  # noqa: BLE001 - isolate per storm
                errors.append(f"{sid} adv {adv}: {type(ex).__name__}: {ex}")
        # A storm that left CurrentStorms (dissipated/post-storm) can
        # never heal - close its text debt so the heal pulse stops.
        for sid, k in ledger.items():
            if sid not in live and not k.get("text_done"):
                k["text_done"] = True
                log.info("%s: storm left the index with text pending - "
                         "heal closed", sid)
        if errors:
            # Raise AFTER the successful storms persisted - the engine
            # holds the change signature and retries only the failures
            # next poll (their adv numbers are still un-ledgered).
            raise RuntimeError("; ".join(errors)[:1000])

    return pf.Source(name="cyclolab-adv", fetch=fetch,
                     change_key=change_key, process=process,
                     valid_time=valid_time)
