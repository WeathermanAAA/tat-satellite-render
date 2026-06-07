#!/usr/bin/env python3
"""
feed_recompute.py
-----------------
Reassemble the LIVE ACE + tracks feeds from (archive base + fresh live b-decks),
using ace_core's SHARED assembly verbatim. This is the poller's brain; it is
PURE (no network, no I/O) so it is trivially testable offline and produces output
byte-shape-identical to the cron's feeds.

Contract: given the same fixes (the base's current-year IBTrACS canon + the live
b-deck frame) the cron would see, this reproduces the cron's feed EXACTLY,
because every number flows through the same frozen ace_core functions:
parse_bdeck / merge_named_sources / current_year_storms / eligible_points_from_canon
/ cumulative_by_doy / build_payload (ACE) and merge_and_extract_storms /
compute_header_stats (tracks). Nothing is reimplemented here.

The poller owns ONLY the fresh slice. The historical archive (curves, climo,
past storms, ranking backbone) comes straight from the base; this never rebuilds
it, never touches climo or /historical, and never alters ACE methodology
(ace_core is pinned at ace-core-v0.5.1: invest guard by construction +
unified invest_x marker + stage-rule hurricane marker for every active
designated storm + NaN-safe NATURE).
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

import ace_core as ac
import poller_framework as pf

# The current-year ACE canon schema (ace_core.parse_bdeck / the cron's
# current_year_ibtracs_fixes share this) and the tracks IBTrACS frame schema.
_CANON_COLS = ["SID", "NAME", "season", "time", "wind_kt",
               "nature", "ace_nature", "source", "storm_num"]
_TRACKS_COLS = ["SID", "NAME", "season", "time", "lat", "lon", "wind_kt",
                "pressure_mb", "nature", "ace_nature", "source"]


def _parse_naive(s: Optional[str]) -> Optional[dt.datetime]:
    """ISO-Z string -> tz-NAIVE UTC datetime, matching ace_core.parse_bdeck /
    the IBTrACS loaders (which use naive UTC). Keeping naive throughout avoids
    naive/aware mixing and reproduces the cron's arithmetic exactly."""
    if not s:
        return None
    d = pf.parse_iso(s)
    return d.replace(tzinfo=None) if d is not None else None


def _df_from_records(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """Rebuild a frame from JSON records, parsing ISO-Z 'time' back to a NAIVE
    datetime (matching parse_bdeck / the cron)."""
    rows = []
    for r in records or []:
        rr = dict(r)
        if rr.get("time") is not None:
            rr["time"] = _parse_naive(rr["time"])
        rows.append(rr)
    return pd.DataFrame(rows, columns=columns)


def _cum_from_base(ace_base: dict) -> pd.DataFrame:
    cols = {int(s): [float(v) for v in vals]
            for s, vals in ace_base["cum_hist"].items()}
    cum = pd.DataFrame(cols, index=range(1, 367))
    cum.index.name = "doy"
    return cum


def _climo_from_base(ace_base: dict) -> pd.DataFrame:
    climo = pd.DataFrame(
        {k: [float(v) for v in vals] for k, vals in ace_base["climo"].items()},
        index=range(1, 367))
    climo.index.name = "doy"
    return climo


def merge_current_canon(ib_cur: pd.DataFrame,
                        live: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge the live b-deck frame onto the base's current-year IBTrACS canon,
    EXACTLY as the cron does (ace_core.merge_named_sources -> concat -> dedup).
    Live empty -> the IBTrACS canon stands alone (graceful, never raises)."""
    if live is not None and not live.empty and not ib_cur.empty:
        ib_keep, live_keep = ac.merge_named_sources(ib_cur, live, name_col="NAME")
    else:
        ib_keep = ib_cur
        live_keep = live if live is not None else pd.DataFrame(columns=_CANON_COLS)
    frames = [f for f in (ib_keep, live_keep) if f is not None and not f.empty]
    canon = (pd.concat(frames, ignore_index=True)
             if frames else pd.DataFrame(columns=_CANON_COLS))
    if not canon.empty:
        canon = canon.drop_duplicates(subset=["SID", "time"])
    return canon


def recompute_ace_feed(ace_base: dict, live: Optional[pd.DataFrame],
                       build_now: Optional[dt.datetime] = None) -> dict:
    """Base + live -> the live ACE feed (the exact current shape). Reuses
    ace_core.build_payload, so every key the frontend reads is produced
    identically to the cron."""
    build_now = build_now or pf.utcnow().replace(tzinfo=None)
    cfg = ace_base["basin_cfg"]
    year = int(ace_base["base_year"])
    prior_year = year - 1

    cum = _cum_from_base(ace_base)
    climo = _climo_from_base(ace_base)
    storms_by_year = {int(y): v for y, v in ace_base["storms_by_year"].items()}
    last_obs_doy = {int(s): int(v) for s, v in ace_base["last_obs_doy"].items()}
    ib_cur = _df_from_records(ace_base["current_year_canon"], _CANON_COLS)

    canon_cur = merge_current_canon(ib_cur, live)

    latest_fix_dt = None
    if not canon_cur.empty:
        fix_times = [t for t in canon_cur["time"] if t is not None]
        latest_fix_dt = max(fix_times) if fix_times else None

    cur_storms = ac.current_year_storms(canon_cur, cfg, year)
    season_ace_current = ac.season_ace([s["ace_total"] for s in cur_storms])
    cur_points = ac.eligible_points_from_canon(canon_cur, cfg, year)

    # Add the current-year curve column (cumulative_by_doy is separable by
    # season, so cum_hist + this == the cron's full cum).
    if not cur_points.empty:
        cur_cum = ac.cumulative_by_doy(cur_points)
        cum[year] = cur_cum[year].reindex(range(1, 367)).fillna(0.0).values
        last_obs_doy[year] = int(cur_points.groupby("season")["doy"].max().get(year))
    else:
        cum[year] = 0.0
    cum = cum.reindex(columns=sorted(cum.columns))

    if cur_storms:
        storms_by_year[year] = cur_storms
    else:
        storms_by_year.pop(year, None)

    return ac.build_payload(
        cum, climo, year, prior_year, last_obs_doy,
        storms_by_year=storms_by_year,
        season_ace_current=season_ace_current,
        latest_fix_dt=latest_fix_dt,
        build_now=build_now)


# The cron's global composition order (generate_tracks_plot.py BASINS["global"]
# compose_from_basins). Feature order in the geojson follows storm order, which
# follows this - keep aligned so poller output is byte-comparable to the cron's.
GLOBAL_COMPOSE_ORDER = ("al", "ep", "wp")


def build_global_geojson_feed(storms_by_basin: dict[str, list],
                              build_now: Optional[dt.datetime] = None,
                              compose_order=GLOBAL_COMPOSE_ORDER) -> dict:
    """Per-basin storms (each list = a tracks feed's ``storms``) -> the global
    FeatureCollection for /global_tracks.html, via the SHARED
    ace_core.build_global_geojson - the identical assembly the cron's
    ``--basin global`` mode runs, composed in the identical basin order with
    ``basin`` stamped per storm (generate_tracks_plot.py:3022-3026). Pure: no
    network, no I/O, inputs never mutated (storms are shallow-copied for the
    basin stamp).

    On top of the cron's bare FeatureCollection this adds the poller's
    freshness stamps (``generated_utc`` / ``updated`` / ``latest_fix_valid_utc``
    / ``staleness_minutes`` - the same fields the feeds carry) as top-level
    foreign members, so the live "As of" can read the map's true freshness at
    runtime. RFC 7946 permits foreign members; MapLibre only reads
    ``features``. Invests ride along exactly as in the per-basin feeds:
    DISPLAYED on the map (invest markers), never in ACE/counts (they were
    excluded upstream by the ACE recompute)."""
    build_now = build_now or pf.utcnow().replace(tzinfo=None)
    storms: list[dict] = []
    for sub in compose_order:
        for s in storms_by_basin.get(sub) or []:
            storms.append({**s, "basin": sub})   # copy-on-write basin stamp
    fc = ac.build_global_geojson(storms)
    # Freshest fix across all composed basins (per-storm field carries through
    # from each sub-feed; ISO-Z strings compare lexicographically) - the same
    # reduction the cron's global mode applies to its payload.
    fix_times = [s.get("latest_fix_valid_utc") for s in storms
                 if s.get("latest_fix_valid_utc")]
    latest_fix_z = max(fix_times) if fix_times else None
    return {
        "type": fc["type"],
        "generated_utc": ac.now_iso_z(build_now),
        "updated": build_now.strftime("%Y-%m-%d %H:%M UTC"),
        "latest_fix_valid_utc": latest_fix_z,
        "staleness_minutes": ac.staleness_minutes(_parse_naive(latest_fix_z),
                                                  build_now)
        if latest_fix_z else None,
        "features": fc["features"],
    }


def recompute_tracks_feed(tracks_base: dict, live: Optional[pd.DataFrame],
                          build_now: Optional[dt.datetime] = None) -> dict:
    """Base + live -> the live tracks feed (exact current shape). Reuses
    ace_core.merge_and_extract_storms + compute_header_stats, so the tracks
    storm set + header match the cron and ace == tracks holds by construction."""
    build_now = build_now or pf.utcnow().replace(tzinfo=None)
    cfg = tracks_base["basin_cfg"]
    ibtracs_frame = _df_from_records(tracks_base["current_year_ibtracs"], _TRACKS_COLS)
    live_frame = live if live is not None else pd.DataFrame()

    storms = ac.merge_and_extract_storms(ibtracs_frame, live_frame, cfg)
    header = ac.compute_header_stats(storms)

    fix_times = [s["latest_fix_valid_utc"] for s in storms
                 if s.get("latest_fix_valid_utc")]
    latest_fix_z = max(fix_times) if fix_times else None
    return {
        "basin": tracks_base["basin"],
        "basin_name": tracks_base["basin_name"],
        "year": int(tracks_base["year"]),
        "updated": build_now.strftime("%Y-%m-%d %H:%M UTC"),
        "generated_utc": ac.now_iso_z(build_now),
        "latest_fix_valid_utc": latest_fix_z,
        "staleness_minutes": ac.staleness_minutes(_parse_naive(latest_fix_z), build_now)
        if latest_fix_z else None,
        "header": header,
        "vocab": tracks_base["vocab"],
        "storms": storms,
    }
