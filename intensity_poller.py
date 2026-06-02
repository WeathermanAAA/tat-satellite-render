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
     the TRACKS feed ONLY (they never enter ACE / season counts),
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
heartbeat. ace_core is pinned at ace-core-v0.1.0 - this never alters ACE
methodology, never rebuilds the archive, never touches climo or /historical.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
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
            except pf.TransientFetchError:
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
        if (it.get("origin_basin") or "").upper() != letter:
            continue
        atcf_id = (it.get("atcf_id") or "").strip()
        try:
            storm_num = int(atcf_id[:-1])
        except (ValueError, IndexError):
            continue
        if not (90 <= storm_num <= 99):
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
        rows.append({
            "SID": f"{basin_cfg['agency_name']}_{basin_cfg['short'].upper()}"
                   f"{storm_num:02d}{year}",
            "NAME": name, "season": year, "time": t, "lat": lat, "lon": lon,
            "wind_kt": vmax, "pressure_mb": pres, "nature": nature,
            "source": "live-knackwx", "storm_num": storm_num,
        })
    return pd.DataFrame(rows)


def _combine(named: pd.DataFrame, invests: pd.DataFrame) -> pd.DataFrame:
    """named + invests for the TRACKS frame (ace gets named only)."""
    frames = [f for f in (named, invests) if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
                      clock: Callable[[], dt.datetime] = pf.utcnow) -> pf.Source:
    """Build the Source for one basin. ``live_fetcher`` / ``invest_fetcher`` /
    ``base_reader`` are injectable for offline tests; the defaults hit the proxy
    chain + knackwx + R2.

    ``named`` = b-decks 01-40 -> drives BOTH feeds. ``invests`` = knackwx 90-99
    -> the TRACKS feed only (invests never enter ACE)."""
    year = clock().year

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
                                               build_now=now_naive)
        ctx.sink.write(f"feeds/{basin}_ace_data.json", ace_feed)
        ctx.sink.write(f"feeds/{basin}_tracks_data.json", tracks_feed)

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
                 **source_kwargs) -> pf.PollerEngine:
    session = session or requests.Session()
    sources = [make_basin_source(b, session, sink, clock=clock, **source_kwargs)
               for b in basins]
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
    log.info("intensity poller starting | base=%s | basins=%s | interval=%gs | invests=%s",
             BASE_URL, ",".join(BASINS), interval, INVESTS_ENABLED)
    eng.run_forever()


if __name__ == "__main__":
    main()
