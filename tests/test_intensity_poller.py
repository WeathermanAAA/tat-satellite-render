#!/usr/bin/env python3
"""Offline proofs for the streaming intensity poller (Stage A).

No network, no prod-R2, no cron: a small synthetic archive base + live frame
exercise the recompute + engine, asserting the four Stage-A guarantees:
  SHAPE       - the recomputed feeds carry exactly the live feed's key set.
  CORRECTNESS - the cross-feed invariant holds (ace.current == sum(storms) ==
                tracks.total_ace), and the recompute routes every number through
                the frozen ace_core assembly (no reimplementation).
  FRESHNESS   - latest_fix_valid_utc tracks the newest fix; staleness is the
                genuine data lag, dropping to minutes the moment a fresh fix
                lands.
  ISOLATION   - one basin's fetch failing never freezes the others; last-known
                -good + signature are preserved; the health heartbeat still fires.

The full byte-for-byte parity-with-the-cron proof (poller output == cron output
for the same fixes) is run against the real base + a captured live frame; see
PROOFS.md (it needs the main-repo cron, so it is not a self-contained unit test).

Run:  python tests/test_intensity_poller.py -v
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feed_recompute as fr          # noqa: E402
import poller_framework as pf        # noqa: E402
import intensity_poller as ip        # noqa: E402

YEAR = 2026


def _cfg(basin="wp"):
    return {"short": basin, "agency_name": "JTWC", "invest_letter": "W",
            "atcf_patterns": []}


def _fix(name, day, hour, wind, lat=15.0, lon=135.0, pres=985.0, sid="T01"):
    return {"SID": sid, "NAME": name, "season": YEAR,
            "time": dt.datetime(YEAR, 1, 1) + dt.timedelta(days=day - 1, hours=hour),
            "lat": lat, "lon": lon, "wind_kt": wind, "pressure_mb": pres,
            "nature": "TS", "ace_nature": "TS", "source": "IBTrACS",
            "storm_num": None}


def _synthetic_ace_base(basin="wp"):
    # two past seasons + climo (366-length bands) + one current-year canon storm
    cum_hist = {"2024": [round(min(i * 0.5, 60.0), 3) for i in range(366)],
                "2025": [round(min(i * 0.4, 48.0), 3) for i in range(366)]}
    climo = {k: [0.0] * 366 for k in ("min", "p10", "p25", "mean", "p75", "p90", "max")}
    canon = [{**_fix("TESTER", 100, h, 55.0), "time": fr._parse_naive(
        (dt.datetime(YEAR, 4, 9) + dt.timedelta(hours=h)).replace(microsecond=0).isoformat() + "Z")}
        for h in (0, 6, 12, 18, 24, 30)]
    return {
        "schema_version": 1, "kind": "ace_base", "basin": basin, "base_year": YEAR,
        "doy": list(range(1, 367)), "cum_hist": cum_hist, "climo": climo,
        "storms_by_year": {"2025": [{"name": "OLDIE", "formation": "2025-08-01T00:00:00",
                                     "dissipation": "2025-08-05T00:00:00",
                                     "peak_wind_kt": 80.0, "peak_wind_time": "2025-08-03T00:00:00",
                                     "ace_total": 12.5}]},
        "last_obs_doy": {"2024": 366, "2025": 366},
        "current_year_canon": [{**c, "time": ip.ac.iso_z(c["time"])} for c in canon],
        "basin_cfg": _cfg(basin), "generated_utc": "2026-06-02T00:00:00Z",
    }


def _synthetic_tracks_base(basin="wp"):
    ib = [_fix("TESTER", 100, h, 55.0) for h in (0, 6, 12, 18, 24, 30)]
    return {
        "schema_version": 1, "kind": "tracks_base", "basin": basin,
        "basin_name": "West Pacific", "year": YEAR,
        "vocab": {"named": "named storms", "cat1plus": "typhoons",
                  "cat3plus": "major", "cat5": "super typhoons"},
        "current_year_ibtracs": [{**c, "time": ip.ac.iso_z(c["time"])} for c in ib],
        "basin_cfg": _cfg(basin), "generated_utc": "2026-06-02T00:00:00Z",
    }


# Live feed key sets (the exact shape the frontend reads).
ACE_KEYS = {"doy", "climo", "current", "prior_year", "today_doy", "rankings",
            "current_rank", "total_seasons", "all_years", "storms_by_year",
            "generated_utc", "latest_fix_valid_utc", "staleness_minutes"}
TRK_KEYS = {"basin", "basin_name", "year", "updated", "generated_utc",
            "latest_fix_valid_utc", "staleness_minutes", "header", "vocab", "storms"}


class TestRecompute(unittest.TestCase):
    def setUp(self):
        self.ab = _synthetic_ace_base()
        self.tb = _synthetic_tracks_base()
        self.bn = dt.datetime(2026, 4, 10, 12, 0, 0)

    def test_shape_ace(self):
        feed = fr.recompute_ace_feed(self.ab, pd.DataFrame(), build_now=self.bn)
        self.assertEqual(set(feed), ACE_KEYS)

    def test_shape_tracks(self):
        feed = fr.recompute_tracks_feed(self.tb, pd.DataFrame(), build_now=self.bn)
        self.assertEqual(set(feed), TRK_KEYS)

    def test_cross_feed_invariant(self):
        af = fr.recompute_ace_feed(self.ab, pd.DataFrame(), build_now=self.bn)
        tf = fr.recompute_tracks_feed(self.tb, pd.DataFrame(), build_now=self.bn)
        cur = af["current"]["latest_value"]
        ssum = round(sum(s["ace_total"] for s in af["storms_by_year"].get(str(YEAR), [])), 3)
        self.assertEqual(cur, ssum)               # ace headline == sum of per-storm
        self.assertEqual(cur, tf["header"]["total_ace"])  # ace == tracks
        self.assertGreater(cur, 0.0)              # the canon storm produced ACE

    def test_freshness_tracks_newest_and_minutes(self):
        # newest canon fix is day 100 (2026-04-09 30h => 2026-04-10 06:00Z)
        af = fr.recompute_ace_feed(self.ab, pd.DataFrame(),
                                   build_now=dt.datetime(2026, 4, 10, 6, 5, 0))
        self.assertEqual(af["latest_fix_valid_utc"], "2026-04-10T06:00:00Z")
        self.assertEqual(af["staleness_minutes"], 5)   # MINUTES the moment it is fresh

    def test_live_merges_and_advances_freshness(self):
        # a NEW live storm (additive; no merge contest) advances freshness
        live = pd.DataFrame([_fix("FRESHIE", 102, 0, 45.0, sid="T02")])  # 2026-04-12 00:00Z
        af = fr.recompute_ace_feed(self.ab, live, build_now=dt.datetime(2026, 4, 12, 0, 5, 0))
        self.assertEqual(af["latest_fix_valid_utc"], "2026-04-12T00:00:00Z")
        self.assertEqual(af["staleness_minutes"], 5)

    def test_invests_never_enter_ace(self):
        # an invest in an ACE frame contributes 0 ACE and never lands in storms_by_year
        named = pd.DataFrame([_fix("TESTER", 100, h, 55.0) for h in (0, 6, 12, 18, 24, 30)])
        inv = _fix("90W", 100, 0, 20.0, sid="JTWC_WP902026")
        inv["storm_num"], inv["nature"] = 90, "DS"
        with_inv = pd.concat([named, pd.DataFrame([inv])], ignore_index=True)
        a0 = fr.recompute_ace_feed(self.ab, named, build_now=self.bn)
        a1 = fr.recompute_ace_feed(self.ab, with_inv, build_now=self.bn)
        self.assertEqual(a0["current"]["latest_value"], a1["current"]["latest_value"])  # invest adds 0
        self.assertNotIn("90W", {s["name"] for s in a1["storms_by_year"].get(str(YEAR), [])})

    def test_invest_appears_on_tracks(self):
        # a RECENT invest (within the active window) shows on the tracks feed
        now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        syn = now.replace(hour=(now.hour // 6) * 6)
        inv = {"SID": "JTWC_WP902026", "NAME": "90W", "season": YEAR, "time": syn,
               "lat": 12.0, "lon": 130.0, "wind_kt": 25.0, "pressure_mb": 1004.0,
               "nature": "DS", "source": "live-knackwx", "storm_num": 90}
        tf = fr.recompute_tracks_feed(self.tb, pd.DataFrame([inv]), build_now=now)
        invs = [s["name"] for s in tf["storms"] if s.get("is_invest")]
        self.assertEqual(invs, ["90W"])


class TestEngineIsolation(unittest.TestCase):
    def _engine(self, sink, live_fetcher, clock):
        base_reader = lambda b, k: (_synthetic_ace_base(b) if k == "ace"
                                    else _synthetic_tracks_base(b))
        return ip.build_engine(sink, basins=("wp", "al", "ep"), session=None,
                               base_reader=base_reader, live_fetcher=live_fetcher,
                               clock=clock, sleep=lambda _: None)

    def test_one_basin_failure_isolated(self):
        sink = pf.DictSink()
        T = dt.datetime(2026, 4, 10, 6, 0, 0, tzinfo=dt.timezone.utc)

        def lf(cfg, year):
            if cfg["short"] == "wp":
                raise pf.TransientFetchError("simulated wp outage")
            return pd.DataFrame()
        eng = self._engine(sink, lf, lambda: T)
        res = eng.poll_once()
        self.assertEqual(res["wp"].status, pf.FETCH_FAILED)
        self.assertTrue(res["al"].ok and res["ep"].ok)        # others refreshed
        self.assertNotIn("feeds/wp_ace_data.json", sink.store)  # no half-write
        self.assertIn("feeds/al_ace_data.json", sink.store)
        self.assertIn("feeds/poller_health.json", sink.store)   # heartbeat fired
        self.assertIsInstance(res, dict)                       # no exception escaped

    def test_last_known_good_preserved_then_failure(self):
        sink = pf.DictSink()
        state = {"c": 1}
        T = {1: dt.datetime(2026, 4, 10, 6, 0, tzinfo=dt.timezone.utc),
             2: dt.datetime(2026, 4, 10, 6, 1, tzinfo=dt.timezone.utc)}

        def lf(cfg, year):
            if cfg["short"] == "wp" and state["c"] >= 2:
                raise pf.TransientFetchError("outage")
            return pd.DataFrame([_fix("TESTER", 100, 0, 55.0)]) if cfg["short"] == "wp" else pd.DataFrame()
        eng = self._engine(sink, lf, lambda: T[state["c"]])
        eng.poll_once()
        wp1 = json.dumps(sink.store["feeds/wp_ace_data.json"], sort_keys=True)
        sig1 = eng.health("wp").last_signature
        state["c"] = 2
        r2 = eng.poll_once()
        wp2 = json.dumps(sink.store["feeds/wp_ace_data.json"], sort_keys=True)
        self.assertEqual(r2["wp"].status, pf.FETCH_FAILED)
        self.assertEqual(wp1, wp2)                              # last-known-good intact
        self.assertEqual(sig1, eng.health("wp").last_signature)  # signature preserved
        self.assertTrue(r2["al"].ok and r2["ep"].ok)

    def test_change_gated_no_reprocess_when_unchanged(self):
        sink = pf.DictSink()
        T = dt.datetime(2026, 4, 10, 6, 0, 0, tzinfo=dt.timezone.utc)
        live = pd.DataFrame([_fix("TESTER", 100, 0, 55.0)])
        eng = self._engine(sink, lambda cfg, y: live if cfg["short"] == "wp" else pd.DataFrame(),
                           lambda: T)
        r1 = eng.poll_once()
        r2 = eng.poll_once()
        self.assertEqual(r1["wp"].status, pf.CHANGED)
        self.assertEqual(r2["wp"].status, pf.UNCHANGED)        # same fix -> skipped


if __name__ == "__main__":
    unittest.main(verbosity=2)
