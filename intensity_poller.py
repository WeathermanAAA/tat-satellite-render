#!/usr/bin/env python3
"""
intensity_poller.py
-------------------
Streaming ACE/tracks intensity poller (Piece 3). One PollerEngine with a Source
per NHC/JTWC basin (wp/al/ep). Each cycle a Source:
  1. reads its slow-moving ARCHIVE BASE from R2 (feeds/base/{basin}_*_base.json),
  2. fetches ONLY the fresh live b-decks for the current season via the SAME
     proxy chain the main-repo generators use (Cloudflare worker -> ftp.nhc /
     natyphoon mirror -> JTWC), parsing them with the FROZEN ace_core.parse_bdeck,
  3. recomputes the live feed (current curve + ytd@doy + rank + storms[] + header
     + freshness) via ace_core's shared assembly (feed_recompute), preserving the
     EXACT current feed shape, ONLY when a new fix actually lands (change-gated),
  4. writes the feeds to the injected Sink.

STAGE A is OFFLINE-ONLY: the proofs run this against a local/R2 base and write to
a DictSink/FileSink. NO Railway deploy, NO prod-R2 write, NO cutover - the cron
still owns the live feeds. The deliberate cron->poller cutover is Stage B.

Anti-freeze (inherited from poller_framework): per-source isolation (one basin's
b-deck fetch failing never freezes or stales the others; each keeps its own
last-known-good), resilient_fetch (timeout + backoff retries), always-on health
heartbeat. ace_core is pinned at ace-core-v0.1.0 - this never alters ACE
methodology, never rebuilds the archive, never touches climo or /historical.
"""
from __future__ import annotations

import datetime as dt
import json
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
                      base_reader: Optional[Callable[[str, str], dict]] = None,
                      clock: Callable[[], dt.datetime] = pf.utcnow) -> pf.Source:
    """Build the Source for one basin. ``live_fetcher`` / ``base_reader`` are
    injectable for offline tests; the defaults hit the proxy chain + R2."""
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
        if live_fetcher is not None:
            live = live_fetcher(cfg, year)
        else:
            live = fetch_live_bdecks(session, cfg, year)
        return {"ace_base": ace_base, "tracks_base": tracks_base, "live": live}

    def change_key(data):
        # New data iff a newer fix landed (or the base regenerated).
        lt = _latest_fix(data["live"])
        return (pf.iso_z(lt) if lt else None,
                data["ace_base"].get("generated_utc"))

    def valid_time(data):
        return _latest_fix(data["live"])

    def process(ctx: pf.ProcessContext):
        data = ctx.data
        now_naive = ctx.now.replace(tzinfo=None)
        ace_feed = fr.recompute_ace_feed(data["ace_base"], data["live"], build_now=now_naive)
        tracks_feed = fr.recompute_tracks_feed(data["tracks_base"], data["live"], build_now=now_naive)
        ctx.sink.write(f"feeds/{basin}_ace_data.json", ace_feed)
        ctx.sink.write(f"feeds/{basin}_tracks_data.json", tracks_feed)

    return pf.Source(name=basin, fetch=fetch, change_key=change_key,
                     process=process, valid_time=valid_time)


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


if __name__ == "__main__":   # pragma: no cover - Stage B deploy entrypoint
    # NOT wired in Stage A. Stage B sets the R2 sink + run_forever.
    raise SystemExit("intensity_poller: Stage A is offline-only (run the proofs); "
                     "the deploy entrypoint + R2 sink land in Stage B.")
