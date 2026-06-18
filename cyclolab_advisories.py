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
import json
import logging
import os
import re
from typing import Callable, Optional

import requests

import poller_framework as pf
from cyclolab_intensity import basin_entry
from cyclolab_og import render_intensity_card
from derived_cone import build_derived_advisory_json
from jtwc_warning import JtwcParseError, parse_jtwc_warning, parse_warning_number
from kml_advisories import (AdvisoryParseError, build_advisory_json,
                            parse_nws_alert_zones,
                            product_text, NWS_TC_EVENTS,
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


# Inland county/zone WW FILLS (Phase 4 follow-up): the NWS public alerts API.
# National (not per-storm), so fetched ONCE per poll and attributed to each
# storm by its cone bbox. Kill-switch CYCLOLAB_WW_ZONES (house idiom, default on).
NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"
WW_ZONES_ENABLED = (_env("CYCLOLAB_WW_ZONES", "1") or "1").lower() \
    not in ("0", "false", "no")
# How far (deg) beyond the storm's cone+track bbox a watched/warned zone is
# still "this storm's" (v2 #5: widened 3.0 -> 4.0 + the box now spans the FULL
# forecast track, not just the cone, so ALL warned counties along the track -
# not just the coastal row - are attributed).
_WW_ZONES_MARGIN_DEG = 4.0


def _storm_extent_box(payload: dict) -> tuple:
    """The lon/lat bbox spanning the storm's cone AND its full forecast track -
    the attribution extent for the inland W/W fills (v2 #5)."""
    xs = [c[0] for c in payload.get("cone") or []]
    ys = [c[1] for c in payload.get("cone") or []]
    for p in payload.get("points") or []:
        if p.get("lon") is not None and p.get("lat") is not None:
            xs.append(p["lon"])
            ys.append(p["lat"])
    return (min(xs), min(ys), max(xs), max(ys))


def _default_fetch_alerts(session: requests.Session,
                          policy: pf.FetchPolicy) -> Optional[dict]:
    """GET the active TC watches/warnings (the six event types) as a GeoJSON
    FeatureCollection. NWS requires a descriptive User-Agent. Best-effort:
    404 -> None, transient -> retried, anything else -> None to the caller (the
    fills are ADDITIVE - a down alerts API must never block the cone)."""
    def _do():
        r = session.get(
            NWS_ALERTS_URL,
            params={"status": "actual", "message_type": "alert",
                    "event": ",".join(NWS_TC_EVENTS)},
            headers={"User-Agent": "tat-cyclolab/1.0 (triple-a-tropics.com)",
                     "Accept": "application/geo+json"},
            timeout=policy.timeout)
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            raise pf.TransientFetchError(f"{r.status_code} alerts")
        r.raise_for_status()
        return r.json()
    return pf.resilient_fetch(_do, policy)


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
        # Coastal watch/warning layer (Phase 4): the WW KMZ from the SAME GIS
        # package. Optional - absent before the first watch/warning is issued,
        # and the renderer/poller degrade gracefully when it is.
        ww = s.get("windWatchesWarnings") or {}
        out.append({
            "sid": sid, "ids": ids, "adv": adv,
            "name": str(s.get("name") or nhc_id),
            "cone_url": cone["kmzFile"], "track_url": track["kmzFile"],
            "ww_url": ww.get("kmzFile"),
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
                           fetch_alerts: Optional[Callable] = None,
                           clock: Callable[[], dt.datetime] = pf.utcnow,
                           ) -> pf.Source:
    """Build the Source. fetch_text/fetch_bytes/fetch_alerts are injectable so
    the offline tests drive the full poll cycle with fixture bytes."""
    import intensity_poller as ip   # late import: shared helpers/config
    current_storms_url = current_storms_url or ip.CURRENT_STORMS_URL
    policy = policy or ip.FETCH_POLICY
    get_text = fetch_text or (lambda url: ip._get_text(session, url, policy))
    get_bytes = fetch_bytes or (
        lambda url: _default_fetch_bytes(session, url, policy))
    get_alerts = fetch_alerts or (
        lambda: _default_fetch_alerts(session, policy))
    # National TC alerts fetched ONCE per poll, cached by poll sequence. A
    # fetch/parse failure caches None -> ww_zones stays [] (never blocks a cone).
    alerts_cache = {"poll": -1, "raw": None}

    # Per-storm ledger: {nhc_sid: {"adv": int, "issued": iso,
    # "text_done": bool, "payload": dict}}. In-memory; a restart simply
    # re-fetches the current advisory and overwrites the SAME content at
    # the SAME key - idempotent by construction. ``payload`` is retained
    # (~25 KB/storm) so the TEXT-HEAL path can merge late products and
    # rewrite WITHOUT refetching the KMZs; ``text_done`` records whether
    # every text product with a URL was attached VERIFIED.
    ledger: dict[str, dict] = {}
    poll_seq = {"n": 0}

    def _alerts_for_poll() -> Optional[dict]:
        """The national TC-alert FeatureCollection for THIS poll (fetched once,
        cached by poll_seq). None when disabled or the fetch failed."""
        if not WW_ZONES_ENABLED:
            return None
        if alerts_cache["poll"] != poll_seq["n"]:
            alerts_cache["poll"] = poll_seq["n"]
            try:
                alerts_cache["raw"] = get_alerts()
            except Exception as ax:  # noqa: BLE001 - additive, never blocks cone
                log.warning("cyclolab ww_zones: alerts fetch failed: %s", ax)
                alerts_cache["raw"] = None
        return alerts_cache["raw"]

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
        retries ONLY what is still owed.

        TCP + TCD are ALWAYS-EXPECTED for an NHC designated storm - the only
        kind this source handles (``_storm_entries`` restricts to AL/EP/CP
        DESIGNATED, non-invest). So the debt stays OPEN (returns False) until
        BOTH land, EVEN when a URL has not been populated yet. The bug this
        closes: at a storm's FIRST advisory CurrentStorms can carry the cone/
        track KMZ a poll or two BEFORE it populates the publicAdvisory /
        forecastDiscussion URLs (observed at a PTC's first advisory). The old
        code skipped a URL-less product and left ``complete`` True, so
        ``text_done`` went VACUOUSLY true, the heal pulse stopped, and the panel
        stayed blank for the WHOLE advisory cycle even after the URLs appeared.
        Returns True iff BOTH TCP and TCD are attached VERIFIED."""
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
            if payload["text"].get(kind):
                continue                      # already attached VERIFIED
            url = urls.get(kind + "_url")
            if not url:
                # ALWAYS-EXPECTED: a not-yet-populated URL is owed text, not
                # "no product" - keep the debt open so the heal pulse retries.
                complete = False
                continue
            try:
                raw = get_text(url)
                if not raw:
                    raise AdvisoryParseError("product page missing/empty")
                payload["text"][kind] = _verified_product(kind, raw, adv)
                # A late-populated URL must also land in the payload so the
                # panel can link it (the prior poll may have written None).
                payload["text"][kind + "_url"] = url
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
                # Watches/warnings (Phase 4): best-effort fetch of the WW KMZ.
                # A missing URL or a fetch hiccup -> None -> build_advisory_json
                # writes ww=[]; the cone is NEVER blocked by the WW layer.
                ww_b = None
                if e.get("ww_url"):
                    try:
                        ww_b = get_bytes(e["ww_url"])
                    except Exception as wx:  # noqa: BLE001
                        log.warning("%s adv %d: WW fetch failed: %s",
                                    sid, adv, wx)
                        ww_b = None
                payload = build_advisory_json(sid, cone_b, track_b,
                                              text_urls=e["text"],
                                              ww_kmz_bytes=ww_b)
                # Inland county/zone FILLS (Phase 4 follow-up): attribute the
                # national NWS TC alerts to THIS storm by its cone bbox. ADDITIVE
                # + GRACEFUL - any failure leaves ww_zones=[]; the cone is never
                # blocked. NHC AL/EP/CP US only (this source's scope already).
                try:
                    raw_alerts = _alerts_for_poll()
                    if raw_alerts:
                        zones = parse_nws_alert_zones(
                            raw_alerts,
                            cone_box=_storm_extent_box(payload),
                            margin_deg=_WW_ZONES_MARGIN_DEG)
                        if zones:
                            payload["ww_zones"] = zones
                            payload["provenance"]["ww_zones_count"] = len(zones)
                except Exception as zx:  # noqa: BLE001
                    log.warning("%s adv %d: ww_zones skipped: %s",
                                sid, adv, zx)
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


# ===========================================================================
# JTWC / WP derived-cone sub-path (CYCLOLAB_DESIGN.md §8.4)
# ---------------------------------------------------------------------------
# A SEPARATE, fully-isolated pf.Source (name="cyclolab-adv-jtwc"). JTWC has no
# CurrentStorms and no official cone: knackwx is the live DESIGNATION feed (same
# fix the tracks map uses), and the per-storm JTWC TC Warning text IS the
# product. We parse the warning's forecast track, buffer it by JTWC's published
# mean track-forecast error (derived_cone + the method-versioned radii blob) into
# the SAME {prefix}/adv/{sid}.json contract the NHC path writes, so the existing
# shell renders a derived cone + two text panels with no rearchitecture.
#
# Isolation: a JTWC fetch/parse failure can only flag THIS source (per-source
# isolation) -- it can never stale ACE/tracks nor break the NHC cones. metoc is
# primary; a JTWC outage = no cone this poll, never an exception that escapes.
# Kill-switch CYCLOLAB_ADVISORIES_JTWC (house idiom, default on), INDEPENDENT of
# the NHC CYCLOLAB_ADVISORIES.
# ===========================================================================
CYCLOLAB_JTWC_ENABLED = (_env("CYCLOLAB_ADVISORIES_JTWC", "1") or "1").lower() \
    not in ("0", "false", "no")

# metoc per-storm products: wp{NN}{YY}web.txt (warning) / ...prog.txt (reasoning).
JTWC_PRODUCT_BASE = (_env("JTWC_PRODUCT_BASE",
                          "https://www.metoc.navy.mil/jtwc/products") or "").rstrip("/")
_LONG_ATCF_RE = re.compile(r"^wp(\d{2})(\d{4})$", re.I)

# The WP mean-error radii blob (method-versioned). Loaded once; a re-pin to a
# newer verification year is a data edit (the file), not a code edit.
_JTWC_RADII = json.load(open(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cyclolab_radii_jtwc_wpac_mean_2015.json"), encoding="utf-8"))


def _jtwc_url(nn: int, yy: int, kind: str) -> str:
    """kind in {'web','prog'} -> the metoc product URL (wp0726web.txt etc.)."""
    return f"{JTWC_PRODUCT_BASE}/wp{nn:02d}{yy:02d}{kind}.txt"


def _jtwc_designated_storms(knack_data, default_year: int) -> list[dict]:
    """The DESIGNATED WP storms (basin letter 'W', number 1..49 - invests 90-99
    excluded) from a knackwx ATCF payload. NN + year come from long_atcf_id
    ('wp072026'); sid = JTWC_WP{NN}{YYYY} (the tracks-feed SID). De-duped."""
    out: dict[str, dict] = {}
    for it in knack_data if isinstance(knack_data, list) else []:
        aid = str(it.get("atcf_id") or "").strip().upper()
        if not aid.endswith("W"):
            continue
        try:
            num = int(aid[:-1])
        except (ValueError, IndexError):
            continue
        if not (1 <= num <= 49):     # designated only (invests are not our product)
            continue
        m = _LONG_ATCF_RE.match(str(it.get("long_atcf_id") or "").strip())
        if m:
            nn, year = int(m.group(1)), int(m.group(2))
        else:
            nn, year = num, default_year   # rare: long_atcf_id absent
        yy = year % 100
        raw = str(it.get("storm_name") or "").strip().upper()
        name = raw if raw and raw not in {"INVEST", "NAMELESS", "UNNAMED"} \
            else f"{num:02d}W"
        sid = f"JTWC_WP{nn:02d}{year}"
        out[sid] = {"sid": sid, "nn": nn, "yy": yy, "year": year, "name": name,
                    "web_url": _jtwc_url(nn, yy, "web"),
                    "prog_url": _jtwc_url(nn, yy, "prog")}
    return list(out.values())


def make_jtwc_advisories_source(session: requests.Session, sink: pf.Sink, *,
                                prefix: str = CYCLOLAB_PREFIX,
                                knackwx_url: str = None,
                                policy: pf.FetchPolicy = None,
                                fetch_text: Optional[Callable] = None,
                                clock: Callable[[], dt.datetime] = pf.utcnow,
                                ) -> pf.Source:
    """Build the JTWC/WP derived-cone Source. ``fetch_text`` is injectable so the
    offline tests drive the full poll cycle with fixture text."""
    import intensity_poller as ip   # late import: shared helpers/config
    knackwx_url = knackwx_url or ip.KNACKWX_ATCF_URL
    policy = policy or ip.FETCH_POLICY
    get_text = fetch_text or (lambda url: ip._get_text(session, url, policy))

    # Per-storm ledger: {sid: {"warning_number": int, "issued": iso,
    # "text_done": bool, "payload": dict}}. In-memory; a restart re-fetches the
    # current warning and overwrites the SAME key - idempotent by construction.
    ledger: dict[str, dict] = {}
    poll_seq = {"n": 0}

    def fetch():
        text = get_text(knackwx_url)
        if not text:
            raise pf.TransientFetchError("knackwx ATCF unavailable")
        poll_seq["n"] += 1
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as e:
            raise pf.TransientFetchError(f"knackwx JSON parse: {e}")
        storms = []
        for s in _jtwc_designated_storms(data, clock().year):
            # The warning text is the JTWC 'cheap index': fetch+parse it so the
            # change-gate keys on (sid, warning_number). PER-STORM ISOLATION - a
            # single warning fetch/parse failure (metoc 403/down, mid-issue) just
            # SKIPS that storm this poll; it never fails the whole fetch.
            try:
                web = get_text(s["web_url"])
                if not web:
                    continue
                parsed = parse_jtwc_warning(web)
            except (JtwcParseError, pf.TransientFetchError, pf.PermanentFetchError,
                    Exception) as ex:  # noqa: BLE001 - graceful per-storm degrade
                log.warning("JTWC %s: warning fetch/parse skipped this poll: %s",
                            s["sid"], ex)
                continue
            s["parsed"] = parsed
            s["warning_number"] = parsed["warning_number"]
            s["web_text"] = web
            storms.append(s)
        return {"storms": storms}

    def change_key(data):
        base = tuple(sorted((s["sid"], s["warning_number"])
                            for s in data["storms"]))
        live = {s["sid"] for s in data["storms"]}
        # Text-heal pulse (mirror NHC): while any live storm still owes its
        # VERIFIED prog text, every poll gets a distinct key so process() runs
        # and retries JUST the prog fetch (cheap). Settles once all text lands.
        if any(not k.get("text_done") for s, k in ledger.items() if s in live):
            return (base, poll_seq["n"])
        return (base,)

    def valid_time(data):
        return None

    def _attach_text(payload: dict, wn: int, web_text: str, web_url: str,
                     prog_url: str) -> bool:
        """tcp = the warning body (verified by construction: its own WARNING NR
        IS ``wn``, parsed in fetch). tcd = the Prognostic Reasoning, attached
        ONLY when its own WARNING NR matches ``wn`` (cross-product verification,
        mirror of NHC _verified_product). Missing/mismatched prog -> tcd PENDING,
        healed next poll; it NEVER blocks the cone. Returns True iff tcd verified."""
        text = payload.get("text")
        if not isinstance(text, dict):
            text = {}
            payload["text"] = text
        text["tcp"] = web_text
        text["tcp_url"] = web_url
        text["tcd_url"] = prog_url
        if text.get("tcd"):
            return True                       # already attached VERIFIED
        try:
            prog = get_text(prog_url)
            if not prog:
                log.info("JTWC %s wn %d: prog.txt missing - tcd pending",
                         payload.get("sid"), wn)
                return False
            got = parse_warning_number(prog)
            if got != wn:
                log.info("JTWC %s: prog warning nr %s != %d - tcd pending",
                         payload.get("sid"), got, wn)
                return False
            text["tcd"] = prog
            return True
        except Exception as tx:  # noqa: BLE001 - additive; never blocks the cone
            log.warning("JTWC %s wn %d: prog fetch pending: %s",
                        payload.get("sid"), wn, tx)
            return False

    def _build_payload(s: dict) -> dict:
        parsed = s["parsed"]
        wn = s["warning_number"]
        points = [dict(p) for p in parsed["points"]]
        for p in points:                      # so build_* reports the advisory
            p["advisory"] = wn
        payload = build_derived_advisory_json(s["sid"], points, _JTWC_RADII)
        # issued_utc := the WARNING issuance (header DTG), not the tau0 valid
        # time build_* defaulted to - the true monotonic issuance the
        # regression guard compares.
        payload["issued_utc"] = parsed["issued_utc"]
        payload["advisory"] = wn
        payload["name"] = s["name"]
        payload["provenance"]["discovery_source"] = knackwx_url
        payload["provenance"]["warning_text_url"] = s["web_url"]
        payload["provenance"]["reasoning_text_url"] = s["prog_url"]
        return payload

    def _heal_text(s: dict, known: dict) -> None:
        """Same warning, prog text still owed - retry + rewrite if it landed.
        Cheap (one text GET; the retained payload spares the cone rebuild)."""
        payload = known["payload"]
        before = payload.get("text", {}).get("tcd")
        known["text_done"] = _attach_text(payload, known["warning_number"],
                                          s["web_text"], s["web_url"], s["prog_url"])
        if payload.get("text", {}).get("tcd") != before:
            payload["provenance"]["text_healed_utc"] = _iso_z(clock())
            sink.write(f"{prefix}/adv/{s['sid']}.json", payload)
            log.info("JTWC adv text healed: %s wn %d (tcd=%d ch)", s["sid"],
                     known["warning_number"],
                     len(payload.get("text", {}).get("tcd") or ""))

    def process(ctx: pf.ProcessContext):
        errors = []
        live = set()
        for s in ctx.data["storms"]:
            sid, wn = s["sid"], s["warning_number"]
            live.add(sid)
            known = ledger.get(sid)
            if known and known["warning_number"] >= wn:
                # change-gate: nothing NEW (or a stale-mirror OLDER warning,
                # rejected by construction). Late prog text still heals in place.
                if known["warning_number"] == wn and not known.get("text_done"):
                    try:
                        _heal_text(s, known)
                    except Exception as hx:  # noqa: BLE001
                        log.warning("JTWC %s wn %d: text heal failed: %s",
                                    sid, wn, hx)
                continue
            try:
                payload = _build_payload(s)
                text_done = _attach_text(payload, wn, s["web_text"],
                                         s["web_url"], s["prog_url"])
                # ISSUANCE-REGRESSION GUARD: never replace cached state with an
                # OLDER warning (stale mirror / CDN echo) - the 91W lesson.
                if known and payload["issued_utc"] < known["issued"]:
                    raise JtwcParseError(
                        f"{sid}: parsed issuance {payload['issued_utc']} "
                        f"regresses cached {known['issued']} - rejected")
                payload["provenance"]["parsed_utc"] = _iso_z(clock())
                sink.write(f"{prefix}/adv/{sid}.json", payload)
                # Best-effort OG card (same honesty guard as NHC): no basin
                # registry entry (WP may have none) or a sink without binary
                # writes (tests) -> skipped, never a borrowed envelope.
                try:
                    ids = parse_sid(sid)
                    entry = basin_entry(ids.basin)
                    if entry and hasattr(sink, "write_png"):
                        png = render_intensity_card(payload, entry,
                                                    storm_name=s["name"])
                        sink.write_png(f"{prefix}/og/{sid}.png", png)
                except Exception as og:  # noqa: BLE001
                    log.warning("JTWC %s wn %d: OG card skipped: %s", sid, wn, og)
                ledger[sid] = {"warning_number": wn, "issued": payload["issued_utc"],
                               "text_done": text_done, "payload": payload}
                log.info("cyclolab JTWC adv cached: %s wn %d (%d cone vtx, "
                         "%d points, text %s)", sid, wn, len(payload["cone"]),
                         len(payload["points"]),
                         "complete" if text_done else "PENDING-HEAL")
            except Exception as ex:  # noqa: BLE001 - isolate per storm
                errors.append(f"{sid} wn {wn}: {type(ex).__name__}: {ex}")
        # A storm that left knackwx (dissipated) can never heal - close its debt.
        for sid, k in ledger.items():
            if sid not in live and not k.get("text_done"):
                k["text_done"] = True
        if errors:
            raise RuntimeError("; ".join(errors)[:1000])

    return pf.Source(name="cyclolab-adv-jtwc", fetch=fetch,
                     change_key=change_key, process=process,
                     valid_time=valid_time)
