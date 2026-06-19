#!/usr/bin/env python3
"""
intensity_poller.py
-------------------
Streaming ACE/tracks intensity poller (Piece 3). One PollerEngine with a Source
per NHC/JTWC basin (wp/al/ep). Each cycle a Source:
  1. reads its slow-moving ARCHIVE BASE from R2 (feeds/base/{basin}_*_base.json),
  2. fetches the fresh live data: NAMED b-decks 01-40 via the SAME proxy chain
     the generators use (Cloudflare worker -> ftp.nhc / natyphoon mirror -> JTWC)
     parsed with the FROZEN ace_core.parse_bdeck, PLUS active invests 90-99 from
     the SAME knackwx source the cron uses. Named drives both feeds; invests go to
     the TRACKS feed ONLY (they never enter ACE / season counts). Each named
     storm's NAME/designation is re-derived per poll from NHC CurrentStorms.json
     (the authoritative live list, which leads the b-deck name column by up to a
     few advisory cycles) so ONE -> AMANDA reaches the banner at poll speed,
  3. recomputes the live feed (current curve + ytd@doy + rank + storms[] + header
     + freshness) via ace_core's shared assembly (feed_recompute), preserving the
     EXACT current feed shape, ONLY when a new fix actually lands (change-gated),
  4. writes the feeds to the injected Sink (R2Sink in prod; Dict/FileSink offline).

Run as a Railway worker (`python intensity_poller.py`) with the R2 sink; the
deliberate cron->poller cutover (cron stops writing the live feeds) is gated on
the main-repo WRITE_LIVE_FEEDS flag and stays reversible.

Anti-freeze (inherited from poller_framework): per-source isolation (one basin's
b-deck fetch failing never freezes or stales the others; each keeps its own
last-known-good), resilient_fetch (timeout + backoff retries), always-on health
heartbeat. ace_core is pinned at ace-core-v0.5.1 (invest guard by
construction: ATCF 90-99 can never accrue ACE or count as named; unified
invest marker: every invest is invest_x regardless of active state; stage
rule: every active designated storm is marker_type "hurricane" - the
peak-keyed td_circle ring is retired, current_category picks the glyph
letter; NaN-safe NATURE so a blank b-deck NATURE column can never crash
the feed build) - this
never alters ACE methodology, never rebuilds the archive, never touches
climo or /historical.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
from typing import Callable, Optional

import pandas as pd
import requests

import ace_core as ac
import poller_framework as pf
import feed_recompute as fr

# ---------------------------------------------------------------------------
# Config (env-driven, safe defaults)
# ---------------------------------------------------------------------------
def _env(n, d=None):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


BASE_URL = _env("FEED_BASE_URL", "https://cdn.triple-a-tropics.com/feeds/base").rstrip("/")
BASINS = tuple((_env("POLLER_BASINS", "wp,al,ep")).split(","))
MAX_STORM_NUM = int(_env("MAX_STORM_NUM", "40"))   # named b-decks 01..40
# Invests (90-99) for the TRACKS feed ONLY - the same knackwx source the cron
# uses, so cutover keeps the invest cards (90E etc.). Invests NEVER enter ACE:
# the ACE recompute is given the named frame only.
KNACKWX_ATCF_URL = _env("KNACKWX_ATCF_URL", "https://api.knackwx.com/atcf/v2")
INVESTS_ENABLED = (_env("POLLER_INVESTS", "1") or "1").lower() not in ("0", "false", "no")
# NHC CurrentStorms.json - the AUTHORITATIVE live name/designation list (AL/EP/
# CP; JTWC basins have no entry). The b-deck name column (col 27, what
# parse_bdeck reads) lags a rename by up to a few advisory cycles - observed
# 2026-06-03: CurrentStorms said AMANDA at 15:00Z while bep012026.dat col 27
# still said ONE - so each poll re-derives the display NAME from here, the same
# authoritative source floater_poller.fetch_current_named reads. (Failure
# semantics deliberately differ from the floater: we return {} and let the
# b-deck name stand for the poll - stateless, worst case a brief name
# flicker during an NHC outage - where the floater holds last-known-good.)
# Display-only by construction: ACE math reads wind/time/nature, never NAME
# (see apply_live_names for the one NAME-adjacent hazard and how it is closed).
CURRENT_STORMS_URL = _env("CURRENT_STORMS_URL",
                          "https://www.nhc.noaa.gov/CurrentStorms.json")
LIVE_NAMES_ENABLED = (_env("POLLER_LIVE_NAMES", "1") or "1").lower() not in ("0", "false", "no")
# Phase 3 (poller-primary storm-display): the poller assembles the SAME global
# FeatureCollection the cron's --basin global mode builds (the shared
# ace_core.build_global_geojson, pinned >= ace-core-v0.2.0) and writes it to
# GLOBAL_GEOJSON_KEY after each basin's recompute - once EVERY configured basin
# has reported since startup (cold-start guard: never publish a partial map).
# SHADOW-FIRST cutover, mirroring the HAFS_R2_PREFIX pattern: the DEFAULT key
# is the shadow one, so deploying this changes NOTHING live - the cron still
# owns global_storms.geojson. Promote = set GLOBAL_GEOJSON_KEY=
# global_storms.geojson (reversible; rollback = set back / unset). "off"
# disables the writer entirely.
GLOBAL_GEOJSON_KEY = (_env("GLOBAL_GEOJSON_KEY",
                           "shadow/global_storms.geojson") or "").strip()
# Same placeholder set the invest path guards with (and ace_core's
# _PLACEHOLDER_NAMES, which is private to the pinned package).
_NAME_PLACEHOLDERS = {"", "INVEST", "NAMELESS", "UNNAMED"}
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")

# Resilient-fetch policy (tuple timeout + exponential backoff). Mirrors the
# generators' tolerance for a sluggish proxy/mirror.
FETCH_POLICY = pf.FetchPolicy(connect_timeout_s=10.0, read_timeout_s=30.0,
                              max_retries=3, backoff_base_s=2.0, backoff_max_s=8.0)


# ---------------------------------------------------------------------------
# Resilient HTTP (genuine 404 = absence -> no retry; 5xx/403/429 = transient)
# ---------------------------------------------------------------------------
def _get_text(session: requests.Session, url: str,
              policy: pf.FetchPolicy) -> Optional[str]:
    """GET text via resilient_fetch. Returns None on a genuine 404 (absence, not
    retried); raises TransientFetchError on 5xx/403/429 (retried); returns the
    body on 200."""
    def _do():
        r = session.get(url, timeout=policy.timeout,
                        headers={"User-Agent": UA})
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            raise pf.TransientFetchError(f"{r.status_code} {url}")
        r.raise_for_status()
        return r.text
    return pf.resilient_fetch(_do, policy)


# ---------------------------------------------------------------------------
# Base (R2) + live b-decks (proxy chain) fetch
# ---------------------------------------------------------------------------
def read_base(session: requests.Session, basin: str, kind: str,
              policy: pf.FetchPolicy = FETCH_POLICY) -> dict:
    """Read one base file (kind in {'ace','tracks'}) from R2. Raises on failure
    (a Source whose base is unreadable must NOT publish a feed - cold-start guard:
    keep last-known-good, never publish a feed with an empty archive)."""
    url = f"{BASE_URL}/{basin}_{kind}_base.json"
    text = _get_text(session, url, policy)
    if text is None:
        raise pf.TransientFetchError(f"base 404: {url}")
    return json.loads(text)


def fetch_live_bdecks(session: requests.Session, basin_cfg: dict, year: int,
                      policy: pf.FetchPolicy = FETCH_POLICY,
                      max_storm_num: int = MAX_STORM_NUM) -> pd.DataFrame:
    """Fetch + parse the current-season named b-decks via the proxy chain. Same
    chain + same parser (ace_core.parse_bdeck) as the main-repo generators, so a
    named storm yields the IDENTICAL canonical track. Returns a (possibly empty)
    live frame; stops after 3 consecutive missing storm numbers."""
    patterns = basin_cfg["atcf_patterns"]
    yy = year % 100
    frames = []
    misses = 0
    for nn in range(1, max_storm_num + 1):
        text = None
        for pat in patterns:
            url = pat.format(nn=f"{nn:02d}", yy=f"{yy:02d}", year=year)
            try:
                t = _get_text(session, url, policy)
            except (pf.TransientFetchError, pf.PermanentFetchError,
                    requests.exceptions.RequestException) as e:
                # A single bad mirror (SSL/connection/timeout/HTTP error) must
                # NEVER crash the whole basin fetch and discard storms already
                # collected this pass - that is what lost EP01 tonight when the
                # WP-only natyphoon mirror SSL-failed (TLSV1_UNRECOGNIZED_NAME)
                # in the AL/EP chain. resilient_fetch already retried this mirror
                # per policy; fall through to the NEXT mirror in the chain.
                log.debug("b-deck mirror failed (%s): %s -- trying next mirror",
                          url, type(e).__name__)
                continue            # try the next mirror in the chain
            if t and "BEST" in t:
                text = t
                break
        if text is not None:
            frames.append(ac.parse_bdeck(text, year, basin_cfg))
            misses = 0
        else:
            misses += 1
            if misses >= 3:
                break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_live_invests(session: requests.Session, basin_cfg: dict, year: int,
                       policy: pf.FetchPolicy = FETCH_POLICY) -> pd.DataFrame:
    """Active invests (90-99) for THIS basin from the knackwx API, in
    parse_bdeck schema - byte-for-byte the same rows the cron's fetch_live_invests
    builds, so the tracks feed keeps its invest cards after cutover. Invests carry
    storm_num 90-99 (merge_and_extract_storms marks them is_invest) and never
    enter ACE. Empty on any failure or when disabled: a flaky knackwx must NEVER
    drop the named-storm cards (per-source-guarded)."""
    if not INVESTS_ENABLED:
        return pd.DataFrame()
    letter = (basin_cfg.get("invest_letter") or "").upper()
    if not letter:
        return pd.DataFrame()
    try:
        text = _get_text(session, KNACKWX_ATCF_URL, policy)
    except pf.TransientFetchError:
        return pd.DataFrame()
    if not text:
        return pd.DataFrame()
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if not isinstance(data, list):
        return pd.DataFrame()
    rows = []
    for it in data:
        atcf_id = (it.get("atcf_id") or "").strip().upper()
        # Basin match off the ATCF id's trailing letter (the authoritative ATCF
        # basin designator: "93E" -> E.Pac, "92W" -> W.Pac, "96P" -> S.Pac).
        # knackwx's separate origin_basin field is sometimes null (observed on
        # EP invests like 93E, which dropped them off the live tracks map), so
        # deriving the basin from the id itself keeps EP/AL invests from being
        # silently dropped; origin_basin is only a fallback when the id has no
        # usable trailing letter. Mirrors generate_tracks_plot.fetch_live_invests.
        id_letter = atcf_id[-1] if atcf_id[-1:].isalpha() else ""
        basin_letter = id_letter or (it.get("origin_basin") or "").strip().upper()
        if basin_letter != letter:
            continue
        try:
            storm_num = int(atcf_id[:-1])
        except (ValueError, IndexError):
            continue
        # JTWC has no CurrentStorms equivalent, so a JUST-designated JTWC TD
        # (the former invest, before its b-deck BEST file is written) falls
        # through both the b-deck sweep (file absent) and the 90-99 invest
        # filter. Treat knackwx as the JTWC live-DESIGNATION source: accept
        # designated numbers 1..49 too. NHC keeps CurrentStorms authoritative
        # (no-op here when agency_name != JTWC). Mirror of generate_tracks_plot.
        is_jtwc = (basin_cfg.get("agency_name") or "").strip().upper() == "JTWC"
        is_invest_num = 90 <= storm_num <= 99
        is_designated = is_jtwc and (1 <= storm_num <= 49)
        if not (is_invest_num or is_designated):
            continue
        ts = it.get("analysis_time")
        if not ts:
            continue
        try:
            t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, AttributeError):
            continue
        try:
            lat = float(it.get("latitude"))
            lon = float(it.get("longitude"))
        except (TypeError, ValueError):
            continue
        try:
            vmax = float(it["winds"]) if it.get("winds") is not None else float("nan")
        except (TypeError, ValueError):
            vmax = float("nan")
        pres_raw = it.get("pressure")
        try:
            pres = float(pres_raw) if pres_raw not in (None, 0) else float("nan")
        except (TypeError, ValueError):
            pres = float("nan")
        devlvl = (it.get("cyclone_nature") or "").strip().upper()
        nature = ac.STATUS_TO_NATURE.get(devlvl, "")
        if not nature:
            nature = "TS" if (pd.notna(vmax) and vmax > 0) else "DS"
        name_raw = (it.get("storm_name") or "").strip()
        name = (name_raw if name_raw and name_raw not in {"INVEST", "NAMELESS", "UNNAMED"}
                else f"{storm_num}{letter}")
        # 92W->07W carry (mirror of the cron). knackwx gives the prior invest as
        # transitioned_from ("92W"); feed its NUMBER as spawn_invest so ace_core's
        # number-keyed superseding-invest dedup retires it the cycle the
        # designation appears. FRAME-COINCIDENT (recycle-safe, stateless): carry
        # only while the SAME payload still lists that 9x invest.
        spawn_invest = None
        if is_designated:
            tf = (it.get("transitioned_from") or "").strip().upper()
            mtf = re.fullmatch(r"(\d{1,2})[A-Z]", tf)
            if mtf:
                tf_num = int(mtf.group(1))
                tf_letter = tf[-1] if tf[-1:].isalpha() else ""
                if 90 <= tf_num <= 99 and tf_letter == letter and any(
                    (str((d.get("atcf_id") or "")).strip().upper())
                        == f"{tf_num:02d}{letter}"
                    for d in data):
                    spawn_invest = tf_num
        rows.append({
            "SID": f"{basin_cfg['agency_name']}_{basin_cfg['short'].upper()}"
                   f"{storm_num:02d}{year}",
            "NAME": name, "season": year, "time": t, "lat": lat, "lon": lon,
            "wind_kt": vmax, "pressure_mb": pres, "nature": nature,
            "source": "live-knackwx-designated" if is_designated else "live-knackwx",
            "storm_num": storm_num, "spawn_invest": spawn_invest,
        })
    return pd.DataFrame(rows)


def parse_current_storm_names(data, basin_cfg: dict, year: int,
                              max_storm_num: int = MAX_STORM_NUM) -> dict[int, str]:
    """CurrentStorms.json payload -> ``{storm_num: NAME}`` for THIS basin/year.
    Pure (offline-testable). Only real names pass: placeholders, malformed ids,
    other basins, other years, and invest-range numbers are skipped. Names are
    upper-cased to the NAME-column convention ("Amanda" -> "AMANDA")."""
    short = (basin_cfg.get("short") or "").strip().lower()
    out: dict[int, str] = {}
    if not isinstance(data, dict) or not short:
        return out
    storms = data.get("activeStorms")
    if not isinstance(storms, list):       # truthy non-list (true/1/{}) included
        return out
    for s in storms:
        if not isinstance(s, dict):
            continue
        sid = s.get("id")                                  # "ep012026"
        if not isinstance(sid, str):       # numeric/None id: skip, never .strip()
            continue
        sid = sid.strip().lower()
        if len(sid) != 8 or sid[:2] != short:
            continue
        try:
            num, yr = int(sid[2:4]), int(sid[4:8])
        except ValueError:
            continue
        if yr != year or not (1 <= num <= max_storm_num):
            continue
        name = s.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip().upper()
        if name in _NAME_PLACEHOLDERS:
            continue
        out[num] = name
    return out


def fetch_current_storm_names(session: requests.Session, basin_cfg: dict,
                              year: int,
                              policy: pf.FetchPolicy = FETCH_POLICY) -> dict[int, str]:
    """``{storm_num: authoritative live NAME}`` from NHC CurrentStorms.json.
    Empty for non-NHC basins (JTWC/WP has no CurrentStorms entry - their b-deck
    name is already the live designation) and on ANY failure: a flaky NHC
    endpoint must NEVER fail the basin fetch. This is a display-name nicety,
    per-source-guarded exactly like invests."""
    if not LIVE_NAMES_ENABLED:
        return {}
    if (basin_cfg.get("agency_name") or "").strip().upper() != "NHC":
        return {}
    try:
        text = _get_text(session, CURRENT_STORMS_URL, policy)
        data = json.loads(text) if text else None
        # Parse INSIDE the guard: well-formed JSON of an unexpected shape must
        # also degrade to {} - a CurrentStorms surprise can never fail the
        # basin fetch (the b-deck name simply stands this poll).
        return parse_current_storm_names(data, basin_cfg, year)
    except Exception:  # noqa: BLE001 - any failure -> no override this poll
        return {}


def apply_live_names(named: pd.DataFrame, live_names: dict[int, str],
                     ace_base: dict, tracks_base: dict):
    """Override the b-deck NAME with NHC's authoritative current name, renaming
    the bases' current-year records IN STEP (copy-on-write).

    WHY both sides: ace_core dedups ib-vs-live per NAME (merge_named_sources)
    but groups storms per SID, and IBTrACS SIDs never equal ATCF SIDs - so
    renaming only the live frame while the base still carries the storm under
    the OLD name would un-contest the pair and surface the same storm twice
    (duplicate card + double-counted ACE). Applying the same old->new rename to
    the base records keeps the contest intact in every window; with the bases
    not yet carrying the storm (the common just-named case) this degenerates to
    a pure rename of the live rows. Beyond the contest, NAME is display-only:
    ACE math reads wind/time/nature and groups by SID.

    Returns ``(named, ace_base, tracks_base)`` - new objects where changed, the
    inputs never mutated (an injected base_reader may hand the same dict to
    every poll)."""
    if (not live_names or named is None or named.empty
            or "storm_num" not in named.columns):
        return named, ace_base, tracks_base
    renames: dict[str, str] = {}        # normalized old NAME -> new NAME
    out = named.copy()
    for num, new in live_names.items():
        mask = out["storm_num"] == num
        if not mask.any():
            continue
        old = str(out.loc[mask, "NAME"].iloc[0] or "").strip().upper()
        if old == new:
            continue
        out.loc[mask, "NAME"] = new
        if old not in _NAME_PLACEHOLDERS:
            renames[old] = new
    if not renames:
        return out, ace_base, tracks_base

    def _patch(base: dict, key: str) -> dict:
        recs, hit = [], False
        for r in (base.get(key) or []):
            old = str(r.get("NAME") or "").strip().upper()
            if old in renames:
                r = {**r, "NAME": renames[old]}
                hit = True
            recs.append(r)
        return {**base, key: recs} if hit else base

    return (out, _patch(ace_base, "current_year_canon"),
            _patch(tracks_base, "current_year_ibtracs"))


def _combine(named: pd.DataFrame, invests: pd.DataFrame) -> pd.DataFrame:
    """named + invests for the TRACKS frame (ace gets named only).

    b-deck PRIMARY: a knackwx-DESIGNATED row (source 'live-knackwx-designated',
    the JTWC live-designation fill for a fresh TD the b-deck has not yet
    written) is dropped once the b-deck named frame carries the SAME SID -- the
    b-deck is the authoritative 6-hourly track when present. 90-99 invests are
    never JTWC-b-decked here, so they pass through untouched. Mirror of the
    cron's fetch_live_season b-deck-primary reconciliation."""
    frames = [f for f in (named, invests) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if named is not None and not named.empty and "source" in out.columns \
            and "SID" in out.columns:
        bdeck_sids = set(named["SID"])
        drop = ((out["source"] == "live-knackwx-designated")
                & out["SID"].isin(bdeck_sids))
        if drop.any():
            out = out[~drop].reset_index(drop=True)
    return out


# NHC CurrentStorms.json — the authoritative final-advisory signal feeding
# ace_core 0.7.0's prompt is_active retirement (status only: a dissipated
# NHC storm drops from active counts / live markers while its track and
# season ACE stay byte-identical). One fetch per cycle shared across the
# three basin sources via a short TTL cache; None (failed fetch) makes the
# retirement a no-op for that build — ace_core's contract, never
# mass-retire on missing information.
_NHC_ACTIVE_TTL_S = 120.0
_nhc_active_cache: dict = {"t": 0.0, "sids": None}
_nhc_active_lock = threading.Lock()


def _nhc_active_sids() -> "set[str] | None":
    with _nhc_active_lock:
        now = time.monotonic()
        if now - _nhc_active_cache["t"] > _NHC_ACTIVE_TTL_S:
            _nhc_active_cache["sids"] = ac.fetch_nhc_active_sids()
            _nhc_active_cache["t"] = now
        return _nhc_active_cache["sids"]


# --- NHC formation chances (Tropical Weather Outlook) -----------------------
# The genesis-odds pill on the invest/PTC CycloLab page reads
# cyclolab/{sid}/formation.json. The cyclolab GUIDANCE poller writes it, but
# ONLY for invests still in the ACTIVE feed — so once a system's invest is
# dropped from the feed (the 90L->01L handoff hands a PTC its REAL id and drops
# the invest), its formation.json FREEZES at its last value while the TWO keeps
# updating (observed: 60% frozen while the live TWO read 70%). The always-on
# poller refreshes formation.json for EVERY invest the TWO references, every
# poll, independent of active-feed membership — so the PTC's pill (which reads
# its spawning invest's formation.json) stays current. Same content + same key
# as the guidance poller, so the two are idempotent. TTL-cached (the TWO moves
# slowly; the poll cadence is ~60s). Kill-switch CYCLOLAB_FORMATION (default on).
_NHC_TWO_BASINS = {"al", "ep", "cp"}
_FORMATION_PREFIX = (_env("GUIDANCE_R2_PREFIX", "cyclolab") or "cyclolab").strip("/")
_FORMATION_ON = (_env("CYCLOLAB_FORMATION", "1") or "1").lower() \
    not in ("0", "false", "no")
_TWO_TTL_S = float(_env("TWO_TTL_S", "300"))
_two_cache: dict = {}            # basin -> (monotonic_t, {sid: formation})
_two_lock = threading.Lock()


def _formation_map(session, basin: str, year: int) -> dict:
    """TTL-cached NHC TWO formation chances for a basin -> {sid: formation}.
    Empty {} on a quiet outlook OR a transient fetch failure (best-effort)."""
    key = basin.lower()
    with _two_lock:
        cached = _two_cache.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < _TWO_TTL_S:
            return cached[1]
    from cyclolab_guidance_poller import fetch_two   # light; late import
    fmap = fetch_two(session, basin, year)
    with _two_lock:
        _two_cache[key] = (time.monotonic(), fmap)
    return fmap


def _refresh_formation(sink, session, basin: str, now_naive) -> None:
    """Write cyclolab/{sid}/formation.json for every TWO-referenced invest in
    ``basin`` (NHC only). Best-effort; an empty map writes nothing (so a
    transient TWO outage never blanks an existing pill). Idempotent with the
    guidance poller (identical key + content)."""
    if not _FORMATION_ON or basin.lower() not in _NHC_TWO_BASINS:
        return
    fmap = _formation_map(session, basin, now_naive.year)
    for sid, fc in fmap.items():
        formation = {**fc, "generated_at": ac.iso_z(now_naive),
                     "source": "nhc-two"}
        sink.write(f"{_FORMATION_PREFIX}/{sid}/formation.json", formation)


class GlobalGeojsonComposer:
    """Holds each basin's latest tracks-feed storms and re-emits the composed
    global geojson (feed_recompute.build_global_geojson_feed -> the shared
    ace_core assembly) after every basin update - but only once ALL configured
    basins have reported at least once since startup, so a cold start never
    publishes a partial map. Per-source isolation: a basin that later fails
    keeps its last-known-good storms on the map (matching the engine's
    last-known-good feed semantics). Best-effort by construction: a geojson
    compose/write failure is logged and NEVER fails the basin's feed publish -
    the display layer must not flag a healthy data source unhealthy."""

    def __init__(self, sink: pf.Sink, key: str, basins,
                 clock: Callable[[], dt.datetime] = pf.utcnow):
        self.sink = sink
        self.key = (key or "").strip()
        self.basins = tuple(basins)
        self.clock = clock
        self._storms: dict[str, list] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.key) and self.key.lower() != "off"

    def update(self, basin: str, tracks_feed: dict,
               now: Optional[dt.datetime] = None) -> bool:
        """Record ``basin``'s latest storms; compose + write when every
        configured basin has reported. Returns True iff written."""
        # EVERYTHING inside the guard: no code path out of update() may raise,
        # or the engine's process-level catch would flip a healthy basin's
        # HEALTH over a display-layer blip. (The geojson is re-composed and
        # re-written on each basin's update - up to 3 small PUTs per cycle.
        # Each write is internally consistent (last-known-good per basin);
        # writing only on the final basin would instead couple the map's
        # freshness to that one basin's health, so the redundancy is the
        # deliberate trade.)
        try:
            if not self.enabled:
                return False
            self._storms[basin] = list(tracks_feed.get("storms") or [])
            if not all(b in self._storms for b in self.basins):
                return False                   # cold start: partial coverage
            now = now or self.clock()
            payload = fr.build_global_geojson_feed(
                self._storms, build_now=now.replace(tzinfo=None))
            self.sink.write(self.key, payload)
            return True
        except Exception as e:  # noqa: BLE001 - display-layer best effort
            log.warning("global geojson write failed (%s): %s", self.key, e)
            return False


# ---------------------------------------------------------------------------
# Per-basin Source (fetch base+live -> change-gate on newest fix -> recompute)
# ---------------------------------------------------------------------------
def _latest_fix(live: pd.DataFrame) -> Optional[dt.datetime]:
    if live is None or live.empty:
        return None
    times = [t for t in live["time"] if t is not None]
    return max(times) if times else None


def make_basin_source(basin: str, session: requests.Session,
                      sink: pf.Sink,
                      live_fetcher: Optional[Callable[[dict, int], pd.DataFrame]] = None,
                      invest_fetcher: Optional[Callable[[dict, int], pd.DataFrame]] = None,
                      base_reader: Optional[Callable[[str, str], dict]] = None,
                      names_fetcher: Optional[Callable[[dict, int], dict]] = None,
                      geojson: Optional[GlobalGeojsonComposer] = None,
                      pages=None,
                      clock: Callable[[], dt.datetime] = pf.utcnow) -> pf.Source:
    """Build the Source for one basin. ``live_fetcher`` / ``invest_fetcher`` /
    ``base_reader`` / ``names_fetcher`` are injectable for offline tests; the
    defaults hit the proxy chain + knackwx + CurrentStorms.json + R2.

    ``named`` = b-decks 01-40 -> drives BOTH feeds, with each storm's NAME
    re-derived per poll from NHC CurrentStorms.json (apply_live_names) so a
    designation/rename (ONE -> AMANDA) reaches the banner at poll speed instead
    of waiting for NHC to backfill the b-deck name column. ``invests`` = knackwx
    90-99 -> the TRACKS feed only (invests never enter ACE)."""
    year = clock().year
    # NEVER-REGRESS high-water-mark (CycloLab cluster): per-basin, per-storm
    # last-known-good full track, so a transient JTWC b-deck/mirror failure can
    # never clobber a designated storm's track down to the lone live fix. In the
    # Source closure (one per basin) -> no cross-basin leakage. ACE is untouched
    # (the ACE feed is a separate recompute_ace_feed call).
    _track_hwm: dict = {}

    def _is_designated(storm: dict) -> bool:
        """Designated (non-invest) storms get the never-regress guard; invests
        are left as the live knackwx snapshot (Part E owns recycle-safe invest
        accumulation). Uses ace_core's own is_invest flag, with a storm-number
        backstop (90-99 = invest)."""
        if storm.get("is_invest"):
            return False
        try:
            return not (90 <= int(str(storm.get("sid") or "")[-6:-4]) <= 99)
        except (ValueError, IndexError):
            return not storm.get("is_invest", False)

    def _read_base(kind: str) -> dict:
        if base_reader is not None:
            return base_reader(basin, kind)
        return read_base(session, basin, kind)

    def fetch():
        # Base first: a Source whose base is unreadable raises (handled as a
        # fetch failure -> last-known-good preserved, no half-written feed).
        ace_base = _read_base("ace")
        tracks_base = _read_base("tracks")
        cfg = ace_base["basin_cfg"]
        named = (live_fetcher(cfg, year) if live_fetcher is not None
                 else fetch_live_bdecks(session, cfg, year))
        # Invests are tracks-only and per-source-guarded: a flaky knackwx returns
        # empty and never drops the named cards.
        invests = (invest_fetcher(cfg, year) if invest_fetcher is not None
                   else fetch_live_invests(session, cfg, year))
        # Authoritative live names (per-source-guarded: {} on any failure ->
        # the b-deck name stands). Applied to named + bases IN STEP; with
        # restamp=True the rename reaches the feeds next cycle even when no new
        # fix landed (process re-runs every cycle on the fresh fetch).
        live_names = (names_fetcher(cfg, year) if names_fetcher is not None
                      else fetch_current_storm_names(session, cfg, year))
        named, ace_base, tracks_base = apply_live_names(
            named, live_names, ace_base, tracks_base)
        return {"ace_base": ace_base, "tracks_base": tracks_base,
                "named": named, "invests": invests}

    def change_key(data):
        # New data iff a newer fix landed anywhere (named OR invest) or the base
        # regenerated - so a fresh invest advisory also triggers a tracks rebuild.
        lt = _latest_fix(_combine(data["named"], data["invests"]))
        return (pf.iso_z(lt) if lt else None,
                data["ace_base"].get("generated_utc"))

    def valid_time(data):
        return _latest_fix(_combine(data["named"], data["invests"]))

    def process(ctx: pf.ProcessContext):
        data = ctx.data
        now_naive = ctx.now.replace(tzinfo=None)
        # ACE: named frame ONLY (invests never enter ACE / season counts).
        ace_feed = fr.recompute_ace_feed(data["ace_base"], data["named"],
                                         build_now=now_naive)
        # Tracks: named + invests (preserves the cron's invest cards).
        tracks_live = _combine(data["named"], data["invests"])
        tracks_feed = fr.recompute_tracks_feed(data["tracks_base"], tracks_live,
                                               build_now=now_naive,
                                               nhc_active_sids=_nhc_active_sids())
        # NEVER-REGRESS (the cure for the clobbered JTWC Track History/Wind&Press
        # + the sparse per-basin map): a transient b-deck/mirror miss must not
        # collapse a designated storm's track to the lone live fix. ACE-safe by
        # construction -- ace_feed above is already computed off the canon frame
        # and is NOT touched here. Best-effort: a guard hiccup never blocks the
        # feed write.
        try:
            n_rg = fr.apply_never_regress(tracks_feed, _track_hwm, _is_designated)
            if n_rg:
                log.info("%s: never-regress republished %d designated storm(s) "
                         "(transient sparse track preserved)", basin, n_rg)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: never-regress skipped (%s)", basin, e)
        ctx.sink.write(f"feeds/{basin}_ace_data.json", ace_feed)
        ctx.sink.write(f"feeds/{basin}_tracks_data.json", tracks_feed)
        # Hardening: a system present in a LIVE source (knackwx / b-deck) but
        # absent from the published feed = the next discovery crack -> WARN
        # loudly. A handoff legitimately retires the prior 9x invest once its
        # designation appears (spawn_invest), so suppress those. Best-effort:
        # a logging bug must NEVER stale the just-written live feed.
        try:
            if not tracks_live.empty and "SID" in tracks_live.columns:
                feed_sids = {s.get("sid") for s in (tracks_feed.get("storms") or [])}
                cfg = data["tracks_base"]["basin_cfg"]
                superseded = set()
                if "spawn_invest" in tracks_live.columns:
                    for n in tracks_live["spawn_invest"].dropna().unique():
                        superseded.add(f"{cfg['agency_name']}_"
                                       f"{cfg['short'].upper()}{int(n):02d}{year}")
                missing = (set(tracks_live["SID"]) - feed_sids) - superseded
                if missing:
                    log.warning("%s: %d system(s) present in a live source but "
                                "absent from the published feed: %s",
                                basin, len(missing), sorted(missing))
        except Exception:  # noqa: BLE001
            pass
        # Phase 3: feed the global-map composer AFTER the basin feeds are
        # safely written. Best-effort inside (a geojson blip never fails or
        # stales this basin's source); gated by GLOBAL_GEOJSON_KEY (default
        # shadow - the cron still owns the live geojson until promotion).
        if geojson is not None:
            geojson.update(basin, tracks_feed, now=ctx.now)
        # CycloLab per-storm page lifecycle (CYCLOLAB_DESIGN.md §3.4):
        # birth/refresh/ended pages from the freshly-written feed. Same
        # best-effort contract as the geojson composer (update() never
        # raises); gated by CYCLOLAB_PAGES at engine assembly.
        if pages is not None:
            pages.update(basin, tracks_feed, now=ctx.now)
        # Keep the formation-chance pill FRESH on the always-on cadence: refresh
        # formation.json for every invest the NHC TWO references in this basin,
        # regardless of active-feed membership (the guidance poller only covers
        # active-feed invests, so a PTC's spawning invest — dropped by the
        # handoff — would otherwise freeze). Best-effort; never fails the basin.
        try:
            _refresh_formation(ctx.sink, session, basin, now_naive)
        except Exception as e:  # noqa: BLE001 - best-effort, never fail the feed
            log.warning("formation refresh failed (%s): %s", basin, e)

    # restamp=True: re-emit the feeds EVERY cycle so generated_utc ticks on the
    # poll cadence (the "poller alive / last checked" stamp) and staleness_minutes
    # stays continuously accurate. staleness is computed FROM latest_fix_valid_utc
    # (the data anchor, which only moves on a new fix), so it grows honestly
    # between advisories - re-stamping never resets it to ~0 / masks true data age.
    return pf.Source(name=basin, fetch=fetch, change_key=change_key,
                     process=process, valid_time=valid_time, restamp=True)


def build_engine(sink: pf.Sink, *, basins=BASINS,
                 session: Optional[requests.Session] = None,
                 interval_s: float = 60.0,
                 clock: Callable[[], dt.datetime] = pf.utcnow,
                 sleep: Callable[[float], None] = time.sleep,
                 geojson_key: Optional[str] = None,
                 cyclolab: Optional[bool] = None,
                 cyclolab_pages_on: Optional[bool] = None,
                 **source_kwargs) -> pf.PollerEngine:
    session = session or requests.Session()
    composer = GlobalGeojsonComposer(
        sink, GLOBAL_GEOJSON_KEY if geojson_key is None else geojson_key,
        basins, clock=clock)
    pages = None
    if cyclolab_pages_on is None:
        from cyclolab_pages import PAGES_ENABLED as cyclolab_pages_on
    if cyclolab_pages_on:
        from cyclolab_pages import CycloLabPageWriter
        pages = CycloLabPageWriter(sink)
    sources = [make_basin_source(b, session, sink, clock=clock,
                                 geojson=composer, pages=pages,
                                 **source_kwargs)
               for b in basins]
    # CycloLab advisories+cone Source (CYCLOLAB_DESIGN.md §9) rides the
    # same engine - per-source isolation means an NHC outage can never
    # stale the feeds, and vice versa. Kill-switch: CYCLOLAB_ADVISORIES.
    if cyclolab is None:
        from cyclolab_advisories import CYCLOLAB_ENABLED as cyclolab
    if cyclolab:
        from cyclolab_advisories import make_advisories_source
        sources.append(make_advisories_source(session, sink, clock=clock))
    # JTWC/WP derived-cone Source (CYCLOLAB_DESIGN.md §8.4) - a SEPARATE,
    # fully-isolated Source so a JTWC parse/fetch failure can never stale ACE/
    # tracks NOR break the NHC cones. INDEPENDENT kill-switch
    # (CYCLOLAB_ADVISORIES_JTWC), not gated by the NHC CYCLOLAB_ADVISORIES flag.
    from cyclolab_advisories import CYCLOLAB_JTWC_ENABLED
    if CYCLOLAB_JTWC_ENABLED:
        from cyclolab_advisories import make_jtwc_advisories_source
        sources.append(make_jtwc_advisories_source(session, sink, clock=clock))
    return pf.PollerEngine(
        sources, name="intensity-poller", interval_s=interval_s,
        stale_after_s=float(_env("STALE_AFTER_S", "1800")),
        sink=sink, heartbeat=pf.sink_heartbeat(sink, "feeds/poller_health.json"),
        clock=clock, sleep=sleep, policy=FETCH_POLICY)


log = logging.getLogger("intensity-poller")


class R2Sink(pf.Sink):
    """Writes each feed JSON to the R2 bucket with the cron's headers
    (application/json, max-age=30) so the frontend's cache behavior is unchanged.
    Raises on failure -> the engine records a process failure, holds the change
    signature, and retries the feed next cycle (a heartbeat-write failure is
    swallowed by emit_health, so a transient R2 blip never crashes the loop)."""

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig
        self.bucket = _env("R2_BUCKET", "triple-a-tropics-media")
        self.s3 = boto3.client(
            "s3", endpoint_url=_env("R2_ENDPOINT"),
            aws_access_key_id=_env("R2_ACCESS_KEY_ID") or _env("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY") or _env("AWS_SECRET_ACCESS_KEY"),
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}))

    def write(self, key: str, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                           ContentType="application/json",
                           CacheControl="public, max-age=30")

    def write_html(self, key: str, html: str,
                   cache: str = "public, max-age=30") -> None:
        """CycloLab page writer (CYCLOLAB_DESIGN.md §3.1): per-storm HTML
        PUT with text/html so the cyclolab-router Worker serves it as a
        real document. Same raise-on-failure semantics as write()."""
        self.s3.put_object(Bucket=self.bucket, Key=key,
                           Body=html.encode("utf-8"),
                           ContentType="text/html; charset=utf-8",
                           CacheControl=cache)

    def write_png(self, key: str, data: bytes,
                  cache: str = "public, max-age=300") -> None:
        """Binary PNG PUT (the CycloLab intensity OG card). Same
        raise-on-failure semantics as write()."""
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data,
                           ContentType="image/png", CacheControl=cache)


def main() -> None:   # pragma: no cover - Railway worker entrypoint
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    missing = [n for n in ("R2_ENDPOINT",) if not _env(n)]
    if not (_env("R2_ACCESS_KEY_ID") or _env("AWS_ACCESS_KEY_ID")):
        missing.append("R2_ACCESS_KEY_ID/AWS_ACCESS_KEY_ID")
    if missing:
        raise SystemExit("intensity_poller: missing required env: " + ", ".join(missing))
    sink = R2Sink()
    interval = float(_env("POLL_INTERVAL_S", "120"))
    eng = build_engine(sink, interval_s=interval)
    log.info("intensity poller starting | base=%s | basins=%s | interval=%gs | "
             "invests=%s | live_names=%s | geojson_key=%s",
             BASE_URL, ",".join(BASINS), interval, INVESTS_ENABLED,
             LIVE_NAMES_ENABLED, GLOBAL_GEOJSON_KEY or "off")
    eng.run_forever()


if __name__ == "__main__":
    main()
