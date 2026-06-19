#!/usr/bin/env python3
"""
cyclolab_guidance_poller.py
---------------------------
ALWAYS-ON poller (Railway worker) for the CycloLab guidance DATA layer. For every
ACTIVE NHC storm AND invest (AL/EP/CP) it fetches + parses the public a-deck
(model tracks + intensity) and the SHIPS text, and writes clean per-entity JSON to
R2 for the CycloLab pages to hydrate later. DATA LAYER ONLY - no renderers, no UI.

Always-on (not a GitHub cron) so it is immune to the scheduled-event throttling that
stalls the enscenters ingest. Per-entity isolation (one storm's failure never affects
another), an always-emitted heartbeat, idempotent never-miss re-writes (a-decks update
each 00/06/12/18Z synoptic; early aids land first, full guidance trickles over hours -
re-poll + re-write), and graceful degrade (fresh invest = statistical-only; missing
SHIPS = "unavailable"). Does NOT touch ACE/track/climo or floater code - the active-
entity list is READ from the same public global feed the home map already publishes.

Parsers live in cyclolab_guidance.py (pure, unit-tested). This module is the I/O.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

import cyclolab_guidance as cg

UTC = dt.timezone.utc
log = logging.getLogger("cyclolab_guidance")


def _env(n: str, d: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(n)
    return v if v not in (None, "") else d


POLL_INTERVAL_S = float(_env("GUIDANCE_POLL_INTERVAL_S", "900"))   # 15 min
R2_PREFIX = (_env("GUIDANCE_R2_PREFIX", "cyclolab") or "cyclolab").strip("/")
GLOBAL_FEED_URL = _env("GLOBAL_GEOJSON_URL", "https://cdn.triple-a-tropics.com/global_storms.geojson")
NHC_BASINS = {"AL", "EP", "CP"}
# JTWC basins we publish guidance for. WPAC (07W et al.) — the a-deck comes from
# the DTC open mirror, NOT NHC aid_public (which is AL/EP/CP only), and there is
# no public WPAC SHIPS text. A WP failure is per-entity isolated (run_once), so it
# can never stale/break the NHC guidance.
JTWC_BASINS = {"WP"}
GUIDANCE_BASINS = NHC_BASINS | JTWC_BASINS
ADECK_BASE = "https://ftp.nhc.noaa.gov/atcf/aid_public"
# DTC open a-deck mirror — the only clean PUBLIC source carrying WPAC model aids
# (the JTWC operational a-deck is restricted/403s from cloud; the b-deck proxy
# chain serves btk only). Plain .dat (not gzipped); fetch_adeck handles both.
ADECK_WP_BASE = "https://hurricanes.ral.ucar.edu/repository/data/adecks_open"
SHIPS_BASE = "https://ftp.nhc.noaa.gov/atcf/stext"
CACHE_CONTROL = "public, max-age=120"
UA = {"User-Agent": "triple-a-tropics-cyclolab-guidance/1.0 (+triple-a-tropics.com)"}
_TIMEOUT = 25.0


def _iso_now() -> str:
    return dt.datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Discovery: the active NHC entities (named storms + invests) from the public
# global feed's active markers - the authoritative "active now" set (a-deck files
# linger after an invest dies, so file-existence is NOT a liveness signal).
# --------------------------------------------------------------------------
def sid_parts(sid: str) -> Optional[Tuple[str, str, str]]:
    """'NHC_EP932026' -> ('EP','93','2026'), 'JTWC_WP072026' -> ('WP','07','2026');
    None if not a guidance basin (NHC AL/EP/CP or JTWC WP)."""
    if not sid or "_" not in sid:
        return None
    rest = sid.split("_", 1)[1]
    if len(rest) != 8 or not rest[2:].isdigit():
        return None
    basin, nn, year = rest[:2], rest[2:4], rest[4:]
    return (basin, nn, year) if basin in GUIDANCE_BASINS else None


def discover_entities(feed: dict) -> List[str]:
    """Active guidance sids (NHC AL/EP/CP + JTWC WP) from the global feed
    (kind == active_marker)."""
    out: List[str] = []
    for f in (feed or {}).get("features", []):
        p = f.get("properties", {})
        if p.get("kind") != "active_marker":
            continue
        sid = p.get("storm_id") or p.get("sid")
        if sid and sid_parts(sid) and sid not in out:
            out.append(sid)
    return out


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def adeck_url(basin: str, nn: str, year: str) -> str:
    # WPAC: DTC open mirror, plain .dat (awp{nn}{year}.dat). NHC: aid_public, gzipped.
    if basin.upper() in JTWC_BASINS:
        return f"{ADECK_WP_BASE}/a{basin.lower()}{nn}{year}.dat"
    return f"{ADECK_BASE}/a{basin.lower()}{nn}{year}.dat.gz"


def ships_url(cycle: str, basin: str, nn: str, year: str) -> str:
    # stext/{YYMMDDCC}{BASIN}{NN}{YY}_ships.txt  (uppercase basin, 2-digit years f+b)
    return f"{SHIPS_BASE}/{cycle[2:]}{basin}{nn}{year[2:]}_ships.txt"


def fetch_adeck(session: requests.Session, url: str) -> Optional[str]:
    try:
        r = session.get(url, headers=UA, timeout=_TIMEOUT)
        if r.status_code != 200 or not r.content:
            return None
        data = r.content
        if data[:2] == b"\x1f\x8b":      # gzip magic (NHC .dat.gz); WPAC DTC is plain .dat
            data = gzip.decompress(data)
        return data.decode("latin-1", "ignore")
    except Exception as e:  # noqa: BLE001
        log.warning("a-deck fetch %s failed: %s", url, e)
        return None


def _prior_cycle(cycle: str) -> str:
    t = dt.datetime.strptime(cycle, "%Y%m%d%H").replace(tzinfo=UTC) - dt.timedelta(hours=6)
    return t.strftime("%Y%m%d%H")


def fetch_ships(session: requests.Session, cycle: str, basin: str, nn: str, year: str
                ) -> Optional[str]:
    """Try the a-deck's latest-init SHIPS, then one synoptic prior (SHIPS can lag the
    a-deck by a cycle). None if neither is posted (-> 'unavailable')."""
    for cyc in (cycle, _prior_cycle(cycle)):
        if not (cyc and len(cyc) == 10):
            continue
        url = ships_url(cyc, basin, nn, year[:4] if len(year) == 4 else year)
        try:
            r = session.get(url, headers=UA, timeout=_TIMEOUT)
            txt = r.text if r.status_code == 200 else ""
        except Exception as e:  # noqa: BLE001
            log.warning("ships fetch %s failed: %s", url, e)
            txt = ""
        if txt and "TIME (HR)" in txt:    # a real SHIPS body (not a 404 HTML page)
            return txt
    return None


# --- NHC Tropical Weather Outlook (TWO) -> invest formation chances ----------
TWO_BASE = "https://www.nhc.noaa.gov/xml"
# tracks-feed basin -> TWO product code (Atlantic's product is "AT", not "AL").
_TWO_PRODUCT = {"AL": "AT", "EP": "EP", "CP": "CP"}


def two_url(basin: str) -> str:
    return f"{TWO_BASE}/TWO{_TWO_PRODUCT.get(basin.upper(), basin.upper())}.xml"


def fetch_two(session: requests.Session, basin: str, year: int) -> Dict[str, dict]:
    """Fetch + parse the NHC TWO for a basin -> {sid: formation}. {} on any
    failure (best-effort - a missing TWO just hides the banner pill)."""
    try:
        r = session.get(two_url(basin), headers=UA, timeout=_TIMEOUT)
        if r.status_code != 200 or not r.text:
            return {}
        return cg.parse_two(r.text, year)
    except Exception as e:  # noqa: BLE001
        log.warning("TWO fetch %s failed: %s", basin, e)
        return {}


# --------------------------------------------------------------------------
# QC + per-entity processing
# --------------------------------------------------------------------------
def _qc_guidance(g: dict) -> List[str]:
    """Light QC (the NHC README mandates it). Returns a list of warnings."""
    warns = []
    if not g.get("init_time"):
        warns.append("no init_time")
    if not g.get("present_aids"):
        warns.append("no curated aids present (fresh-invest/statistical-only or pre-guidance)")
    for tech, pts in g.get("aids", {}).items():
        for p in pts:
            if p["lat"] is not None and not (-90 <= p["lat"] <= 90):
                warns.append(f"{tech} tau{p['tau']} lat out of range")
            if p["lon"] is not None and not (-180 <= p["lon"] <= 180):
                warns.append(f"{tech} tau{p['tau']} lon out of range")
    return warns


def process_entity(sid: str, session: requests.Session,
                   put_json: Callable[[str, dict], bool],
                   formation_map: Optional[Dict[str, dict]] = None) -> dict:
    """Fetch + parse + write one entity's guidance.json + ships.json (+ for an
    invest, formation.json from the pre-parsed TWO map). Returns a per-entity
    status dict (for the heartbeat). Raises nothing the caller must catch beyond
    its own isolation guard."""
    parts = sid_parts(sid)
    if not parts:
        return {"sid": sid, "ok": False, "reason": "bad sid"}
    basin, nn, year = parts
    is_wp = basin.upper() in JTWC_BASINS
    status: dict = {"sid": sid, "ok": False, "guidance": False, "ships": False, "warns": []}

    # NHC formation chance (INVESTS only) - the genesis odds for the banner pill.
    # Written even when the TWO has no area for this invest (nulls -> pill hides),
    # so the eager fetch is a clean 200 rather than a 404. JTWC/WP has no NHC TWO,
    # so WP invests get no formation.json (the pill simply never shows).
    if int(nn) >= 90 and not is_wp:
        fc = (formation_map or {}).get(sid)
        formation = dict(fc) if fc else {"sid": sid, "p48": None, "p7": None,
                                         "level": None, "area": None}
        formation.update({"generated_at": _iso_now(), "source": "nhc-two"})
        status["formation"] = put_json(f"{R2_PREFIX}/{sid}/formation.json", formation)
        status["formation_p7"] = formation.get("p7")

    raw = fetch_adeck(session, adeck_url(basin, nn, year))
    if raw is None:
        status["reason"] = "a-deck unavailable"
        return status
    guidance = cg.parse_adeck(raw, basin=basin)
    guidance.update({"sid": sid, "basin": basin, "generated_at": _iso_now(),
                     "source": "dtc-atcf-adecks_open" if is_wp else "nhc-atcf-aid_public"})
    status["warns"] = _qc_guidance(guidance)
    status["guidance"] = put_json(f"{R2_PREFIX}/{sid}/guidance.json", guidance)
    status["init"] = guidance.get("init_cycle")
    status["present_aids"] = len(guidance.get("present_aids", []))

    cycle = guidance.get("init_cycle")
    ships_obj: dict
    if is_wp:
        # No public WPAC SHIPS text is published (NHC stext is AL/EP/CP only;
        # JTWC runs STIPS, not a redistributable SHIPS product). Honest stub -
        # never fabricate, and SHIPS-absent never blocks the track/intensity plots.
        ships_obj = {"available": False, "reason": "SHIPS not published for WPAC",
                     "sid": sid, "generated_at": _iso_now()}
    else:
        txt = fetch_ships(session, cycle, basin, nn, year) if cycle else None
        if txt is None:
            ships_obj = {"available": False, "reason": "unavailable", "sid": sid,
                         "generated_at": _iso_now()}
        else:
            ships_obj = cg.parse_ships(txt)
            ships_obj.update({"sid": sid, "basin": basin, "init_cycle": cycle,
                              "init_time": cg.init_to_iso(cycle), "generated_at": _iso_now(),
                              "source": "nhc-atcf-stext"})
    status["ships"] = put_json(f"{R2_PREFIX}/{sid}/ships.json", ships_obj)
    status["ships_available"] = ships_obj.get("available", False)
    status["ok"] = bool(status["guidance"] and status["ships"])
    return status


def run_once(session: requests.Session, put_json: Callable[[str, dict], bool],
             feed_url: str = GLOBAL_FEED_URL) -> List[dict]:
    """One poll: discover active entities, process each in isolation, ALWAYS emit a
    heartbeat. Returns the per-entity statuses."""
    try:
        feed = session.get(feed_url, headers=UA, timeout=_TIMEOUT).json()
    except Exception as e:  # noqa: BLE001
        log.warning("guidance heartbeat: active-feed fetch failed (%s); skipping tick", e)
        return []
    sids = discover_entities(feed)
    # NHC formation chances (TWO) for the invest basins present, fetched ONCE per
    # basin per poll then shared across that basin's invests.
    year = dt.datetime.now(UTC).year
    formation_map: Dict[str, dict] = {}
    for b in {sid_parts(s)[0] for s in sids
              if sid_parts(s) and int(sid_parts(s)[1]) >= 90
              and sid_parts(s)[0] in NHC_BASINS}:   # WP has no NHC TWO
        formation_map.update(fetch_two(session, b, year))
    statuses: List[dict] = []
    for sid in sids:
        try:
            statuses.append(process_entity(sid, session, put_json, formation_map))
        except Exception as e:  # noqa: BLE001 - PER-ENTITY isolation
            log.warning("guidance: entity %s FAILED (isolated): %s", sid, e)
            statuses.append({"sid": sid, "ok": False, "reason": f"exception: {e}"})
    ok = sum(1 for s in statuses if s.get("ok"))
    log.info("guidance heartbeat: %d active NHC entit%s, %d ok | %s",
             len(sids), "y" if len(sids) == 1 else "ies", ok,
             "; ".join(f"{s['sid']}[init={s.get('init','?')} aids={s.get('present_aids','?')} "
                       f"ships={'Y' if s.get('ships_available') else 'n'}]" for s in statuses) or "none active")
    return statuses


# --------------------------------------------------------------------------
# R2 sink + main loop
# --------------------------------------------------------------------------
class _R2:
    def __init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig
        self.bucket = _env("R2_BUCKET", "triple-a-tropics-media")
        self.s3 = boto3.client(
            "s3", endpoint_url=_env("R2_ENDPOINT"),
            aws_access_key_id=_env("R2_ACCESS_KEY_ID") or _env("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY") or _env("AWS_SECRET_ACCESS_KEY"),
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}))

    def put_json(self, key: str, obj: dict) -> bool:
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key,
                               Body=json.dumps(obj, separators=(",", ":")).encode(),
                               ContentType="application/json", CacheControl=CACHE_CONTROL)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("R2 put %s failed: %s", key, e)
            return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    r2 = _R2()
    session = requests.Session()
    log.info("cyclolab_guidance_poller armed (prefix=%s interval=%ss bucket=%s)",
             R2_PREFIX, POLL_INTERVAL_S, r2.bucket)
    while True:
        try:
            run_once(session, r2.put_json)
        except Exception as e:  # noqa: BLE001 - the loop must never die
            log.warning("guidance tick crashed (continuing): %s", e)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
