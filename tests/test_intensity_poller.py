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

    def test_restamp_staleness_is_true_data_age_not_reset(self):
        # Re-stamping advances generated_utc, but staleness is (now - latest_fix),
        # so it GROWS between fixes and is never reset to ~0 (the Fix 3 guardrail:
        # never mask genuine data age). Base canon's newest fix is 2026-04-10T06:00Z.
        a1 = fr.recompute_ace_feed(self.ab, pd.DataFrame(), build_now=dt.datetime(2026, 4, 10, 7, 0, 0))
        a2 = fr.recompute_ace_feed(self.ab, pd.DataFrame(), build_now=dt.datetime(2026, 4, 10, 8, 0, 0))
        self.assertEqual(a1["latest_fix_valid_utc"], "2026-04-10T06:00:00Z")     # data anchor
        self.assertEqual(a1["latest_fix_valid_utc"], a2["latest_fix_valid_utc"])  # anchor unchanged
        self.assertNotEqual(a1["generated_utc"], a2["generated_utc"])            # generated advances
        self.assertEqual(a1["staleness_minutes"], 60)    # now - fix
        self.assertEqual(a2["staleness_minutes"], 120)   # GROWS (not reset to ~0)


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

    def test_restamp_rewrites_every_cycle_but_status_unchanged(self):
        # restamp=True: unchanged data across cycles -> status UNCHANGED, but the
        # feed is re-emitted each cycle with a fresh generated_utc (advancing
        # clock), while last_change_utc stays put (honest data-change signal).
        sink = pf.DictSink()
        clk = {"t": dt.datetime(2026, 4, 10, 6, 0, 0, tzinfo=dt.timezone.utc)}
        live = pd.DataFrame([_fix("TESTER", 100, 0, 55.0)])
        eng = self._engine(sink, lambda cfg, y: live if cfg["short"] == "wp" else pd.DataFrame(),
                           lambda: clk["t"])
        r1 = eng.poll_once()
        gen1 = sink.store["feeds/wp_ace_data.json"]["generated_utc"]
        lc1 = eng.health("wp").last_change_utc
        clk["t"] = dt.datetime(2026, 4, 10, 6, 2, 0, tzinfo=dt.timezone.utc)  # +2 min
        r2 = eng.poll_once()
        gen2 = sink.store["feeds/wp_ace_data.json"]["generated_utc"]
        lc2 = eng.health("wp").last_change_utc
        self.assertEqual(r1["wp"].status, pf.CHANGED)
        self.assertEqual(r2["wp"].status, pf.UNCHANGED)   # data unchanged
        self.assertNotEqual(gen1, gen2)                   # but re-stamped: generated_utc advanced
        self.assertEqual(lc1, lc2)                        # last_change_utc honest (data did not change)


class TestLiveNames(unittest.TestCase):
    """Phase 2 (storm-display): per-poll authoritative-name override from NHC
    CurrentStorms.json. Display-only: ACE totals are byte-identical with and
    without the override, and the ib-vs-live merge contest survives a rename."""

    @staticmethod
    def _named_one(n_fixes=6, wind=45.0):
        """A live b-deck frame for storm 01 named 'ONE' (parse_bdeck schema)."""
        rows = []
        for i in range(n_fixes):
            r = _fix("ONE", 100, 6 * i, wind, sid="NHC_EP012026")
            r["storm_num"], r["source"] = 1, "live-atcf"
            rows.append(r)
        return pd.DataFrame(rows)

    @staticmethod
    def _current_storms(name="Amanda", sid="ep012026"):
        return {"activeStorms": [{"id": sid, "binNumber": "EP1", "name": name,
                                  "classification": "TS", "intensity": "35"}]}

    def _ep_cfg(self):
        return {"short": "ep", "agency_name": "NHC", "invest_letter": "E",
                "atcf_patterns": []}

    # --- parser -----------------------------------------------------------
    def test_parser_filters_basin_year_range_and_placeholders(self):
        cfg = self._ep_cfg()
        data = {"activeStorms": [
            {"id": "ep012026", "name": "Amanda"},        # match -> {1: AMANDA}
            {"id": "al022026", "name": "Barry"},         # other basin
            {"id": "ep902026", "name": "Ninety"},        # invest-range num
            {"id": "ep012025", "name": "Stale"},         # wrong year
            {"id": "ep032026", "name": "  "},            # placeholder/blank
            {"id": "epxx2026", "name": "Bad"},           # malformed id
            "not-a-dict",
        ]}
        self.assertEqual(ip.parse_current_storm_names(data, cfg, YEAR),
                         {1: "AMANDA"})
        self.assertEqual(ip.parse_current_storm_names(None, cfg, YEAR), {})
        self.assertEqual(ip.parse_current_storm_names({}, {"short": ""}, YEAR), {})

    # --- apply: rename is display-only ------------------------------------
    def test_rename_reaches_tracks_feed_and_ace_total_unchanged(self):
        ab, tb = _synthetic_ace_base("ep"), _synthetic_tracks_base("ep")
        named = self._named_one()
        bn = dt.datetime(2026, 4, 11, 0, 0, 0)

        a0 = fr.recompute_ace_feed(ab, named, build_now=bn)
        t0 = fr.recompute_tracks_feed(tb, named, build_now=bn)

        named1, ab1, tb1 = ip.apply_live_names(named, {1: "AMANDA"}, ab, tb)
        a1 = fr.recompute_ace_feed(ab1, named1, build_now=bn)
        t1 = fr.recompute_tracks_feed(tb1, named1, build_now=bn)

        # ACE math identical (headline + per-storm values).
        self.assertEqual(a0["current"]["latest_value"], a1["current"]["latest_value"])
        self.assertEqual(
            sorted(round(s["ace_total"], 3) for s in a0["storms_by_year"][str(YEAR)]),
            sorted(round(s["ace_total"], 3) for s in a1["storms_by_year"][str(YEAR)]))
        # The display name flipped ONE -> AMANDA in BOTH feeds.
        self.assertIn("AMANDA", {s["name"] for s in a1["storms_by_year"][str(YEAR)]})
        self.assertNotIn("ONE", {s["name"] for s in a1["storms_by_year"][str(YEAR)]})
        names1 = {s["name"] for s in t1["storms"]}
        self.assertIn("AMANDA", names1)
        self.assertNotIn("ONE", names1)
        self.assertEqual(len(t0["storms"]), len(t1["storms"]))  # no card gained/lost
        # Inputs were not mutated (copy-on-write).
        self.assertEqual(named.iloc[0]["NAME"], "ONE")
        self.assertEqual(t0["storms"] and tb["current_year_ibtracs"][0]["NAME"], "TESTER")

    def test_rename_keeps_ib_vs_live_contest_no_double_count(self):
        # The ONE hazard: the base already carries the storm under the OLD name
        # (different SID). Renaming only the live side would un-contest the pair
        # -> duplicate card + double-counted ACE. apply_live_names renames the
        # base records in step, so the contest (and ACE) is identical.
        ab, tb = _synthetic_ace_base("ep"), _synthetic_tracks_base("ep")
        ib_rows = [{**_fix("ONE", 100, 6 * i, 40.0, sid="2026152N10250")}
                   for i in range(4)]                       # 4 obs: live (6) wins
        ab = {**ab, "current_year_canon":
              ab["current_year_canon"] + [{**r, "time": ip.ac.iso_z(r["time"])}
                                          for r in ib_rows]}
        tb = {**tb, "current_year_ibtracs":
              tb["current_year_ibtracs"] + [{**r, "time": ip.ac.iso_z(r["time"])}
                                            for r in ib_rows]}
        named = self._named_one()
        bn = dt.datetime(2026, 4, 11, 0, 0, 0)

        a0 = fr.recompute_ace_feed(ab, named, build_now=bn)        # baseline: ONE everywhere
        t0 = fr.recompute_tracks_feed(tb, named, build_now=bn)

        named1, ab1, tb1 = ip.apply_live_names(named, {1: "AMANDA"}, ab, tb)
        a1 = fr.recompute_ace_feed(ab1, named1, build_now=bn)
        t1 = fr.recompute_tracks_feed(tb1, named1, build_now=bn)

        self.assertEqual(a0["current"]["latest_value"], a1["current"]["latest_value"])
        self.assertEqual(len(a0["storms_by_year"][str(YEAR)]),
                         len(a1["storms_by_year"][str(YEAR)]))     # no extra ACE storm
        self.assertEqual(len(t0["storms"]), len(t1["storms"]))     # no duplicate card
        names1 = {s["name"] for s in t1["storms"]}
        self.assertIn("AMANDA", names1)
        self.assertNotIn("ONE", names1)                            # base rows renamed too

    def test_noop_cases(self):
        ab, tb = _synthetic_ace_base("ep"), _synthetic_tracks_base("ep")
        named = self._named_one()
        # Same name -> no-op; unknown storm_num -> no-op; empty map -> no-op.
        for live_names in ({1: "ONE"}, {7: "GHOST"}, {}):
            n1, ab1, tb1 = ip.apply_live_names(named, live_names, ab, tb)
            self.assertEqual(list(n1["NAME"].unique()), ["ONE"])
            self.assertIs(ab1, ab)
            self.assertIs(tb1, tb)
        empty = pd.DataFrame()
        n1, ab1, tb1 = ip.apply_live_names(empty, {1: "AMANDA"}, ab, tb)
        self.assertTrue(n1.empty)

    def test_parser_malformed_but_valid_json_shapes(self):
        # Well-formed JSON of the WRONG shape must degrade to {} (review
        # finding: these used to raise through the fetch and fail the basin).
        cfg = self._ep_cfg()
        for data in ({"activeStorms": True},            # truthy non-list
                     {"activeStorms": 1},
                     {"activeStorms": [{"id": 12345678, "name": "X"}]},   # numeric id
                     {"activeStorms": [{"id": "ep012026", "name": 123}]},  # numeric name
                     ["not", "a", "dict"], "html error page", 42):
            self.assertEqual(ip.parse_current_storm_names(data, cfg, YEAR), {},
                             f"shape {data!r} must yield no overrides")

    def test_fetch_survives_malformed_payload_end_to_end(self):
        # The exact prod trace: malformed CurrentStorms must NOT fail the basin
        # fetch - the feed still publishes with the b-deck name standing.
        orig = ip._get_text

        def bad_payload(session, url, policy):
            return '{"activeStorms": true}'
        ip._get_text = bad_payload
        try:
            self.assertEqual(
                ip.fetch_current_storm_names(None, self._ep_cfg(), YEAR), {})
            sink = pf.DictSink()
            named = self._named_one()
            base_reader = lambda b, k: (_synthetic_ace_base(b) if k == "ace"
                                        else _synthetic_tracks_base(b))
            eng = ip.build_engine(
                sink, basins=("ep",), session=None, base_reader=base_reader,
                live_fetcher=lambda cfg, y: named,
                invest_fetcher=lambda cfg, y: pd.DataFrame(),
                clock=lambda: dt.datetime(2026, 4, 11, 0, 0, 0,
                                          tzinfo=dt.timezone.utc),
                sleep=lambda _: None)   # names_fetcher NOT injected: real path
            res = eng.poll_once()
            self.assertTrue(res["ep"].ok)                       # fetch survived
            t = sink.store["feeds/ep_tracks_data.json"]
            self.assertIn("ONE", {s["name"] for s in t["storms"]})  # b-deck stands
        finally:
            ip._get_text = orig

    # --- fetch guards ------------------------------------------------------
    def test_fetch_failure_returns_empty(self):
        def boom(session, url, policy):
            raise pf.TransientFetchError("nhc down")
        orig = ip._get_text
        ip._get_text = boom
        try:
            self.assertEqual(
                ip.fetch_current_storm_names(None, self._ep_cfg(), YEAR), {})
        finally:
            ip._get_text = orig

    def test_non_nhc_basin_never_fetches(self):
        def fail_if_called(session, url, policy):
            raise AssertionError("HTTP fetch attempted for a JTWC basin")
        orig = ip._get_text
        ip._get_text = fail_if_called
        try:
            self.assertEqual(
                ip.fetch_current_storm_names(None, _cfg("wp"), YEAR), {})
        finally:
            ip._get_text = orig

    # --- end-to-end: rename propagates on restamp, no new fix needed -------
    def test_rename_propagates_on_restamp_without_new_fix(self):
        sink = pf.DictSink()
        clk = {"t": dt.datetime(2026, 4, 11, 0, 0, 0, tzinfo=dt.timezone.utc)}
        names = {"m": {}}
        named = self._named_one()
        base_reader = lambda b, k: (_synthetic_ace_base(b) if k == "ace"
                                    else _synthetic_tracks_base(b))
        eng = ip.build_engine(
            sink, basins=("ep",), session=None, base_reader=base_reader,
            live_fetcher=lambda cfg, y: named,
            invest_fetcher=lambda cfg, y: pd.DataFrame(),
            names_fetcher=lambda cfg, y: names["m"],
            clock=lambda: clk["t"], sleep=lambda _: None)

        r1 = eng.poll_once()
        t1 = sink.store["feeds/ep_tracks_data.json"]
        self.assertEqual(r1["ep"].status, pf.CHANGED)
        self.assertIn("ONE", {s["name"] for s in t1["storms"]})
        ace1 = sink.store["feeds/ep_ace_data.json"]["current"]["latest_value"]
        lc1 = eng.health("ep").last_change_utc

        # NHC names the storm between polls; NO new fix lands.
        names["m"] = {1: "AMANDA"}
        clk["t"] = dt.datetime(2026, 4, 11, 0, 2, 0, tzinfo=dt.timezone.utc)
        r2 = eng.poll_once()
        t2 = sink.store["feeds/ep_tracks_data.json"]
        names2 = {s["name"] for s in t2["storms"]}
        self.assertEqual(r2["ep"].status, pf.UNCHANGED)   # no new fix: restamp path
        self.assertIn("AMANDA", names2)                   # ...but the rename shipped
        self.assertNotIn("ONE", names2)
        ace2 = sink.store["feeds/ep_ace_data.json"]["current"]["latest_value"]
        self.assertEqual(ace1, ace2)                      # ACE untouched
        self.assertEqual(lc1, eng.health("ep").last_change_utc)  # honest signal


class TestMirrorFallthrough(unittest.TestCase):
    """HARDENING: a single bad mirror raising a RAW fetch error (SSLError /
    connection / timeout - not a TransientFetchError) must fall through to the
    NEXT mirror in the chain, never crash the basin fetch and discard storms
    already collected this pass. This is the bug that lost EP01 tonight (the
    WP-only natyphoon mirror SSL-failed inside the AL/EP chain)."""

    def test_raw_mirror_error_falls_through_and_keeps_storm(self):
        import intensity_poller as ip
        import requests

        cfg = {"atcf_patterns": ["https://bad.example/b{nn}{year}.dat",
                                 "https://good.example/b{nn}{year}.dat"]}
        calls = []

        def fake_get_text(session, url, policy):
            calls.append(url)
            if "bad.example" in url:               # raw SSL error, NOT Transient
                raise requests.exceptions.SSLError("TLSV1_UNRECOGNIZED_NAME")
            tail = url.rsplit("/", 1)[-1]           # "b012026.dat"
            return "BEST track storm 01" if tail.startswith("b01") else None

        sentinel = pd.DataFrame([{"storm": "01"}])
        orig_get_text, orig_parse = ip._get_text, ip.ac.parse_bdeck
        ip._get_text = fake_get_text
        ip.ac.parse_bdeck = lambda text, year, basin_cfg: sentinel
        try:
            out = ip.fetch_live_bdecks(session=None, basin_cfg=cfg, year=2026,
                                       max_storm_num=4)
        finally:
            ip._get_text, ip.ac.parse_bdeck = orig_get_text, orig_parse

        # Did NOT crash on the bad mirror; the good mirror's storm survived.
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["storm"], "01")
        # The bad mirror was tried, then fell through to the good one.
        self.assertTrue(any("bad.example" in u for u in calls))
        self.assertTrue(any("good.example" in u for u in calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
