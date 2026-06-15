#!/usr/bin/env python3
"""
Unit tests for ens_watchdog (the Ensemble Cyclone Centers freshness watchdog).
No network: the manifest fetch + GitHub dispatch are injected. Deterministic
(injected clock). Run: python -m unittest discover -s tests -v

Contract:
  (a) expected_latest_cycle floors (now - lag) to the 6-hourly boundary
  (b) a model BEHIND its expected cycle is dispatched; current/ahead is not
  (c) a model absent from the manifest is dispatched (bootstrap)
  (d) cooldown: a just-poked model is not re-poked while its run is in flight
  (e) run_once fires the right workflow, records the dispatch, is idempotent, and
      swallows a manifest-fetch error (never crashes the host poller)
  (f) a non-2xx dispatch is NOT recorded (so it retries next tick)
  (g) dispatch_workflow builds the correct GitHub API request (blank inputs)
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ens_watchdog as w  # noqa: E402

UTC = dt.timezone.utc


def _manifest(**latests):
    return {"models": [{"slug": s, "latest": c} for s, c in latests.items()]}


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._p


class _Session:
    """Minimal fake: .get returns the manifest; .post records dispatch calls."""
    def __init__(self, manifest, get_status=200, post_status=204):
        self._m, self._gs, self._ps = manifest, get_status, post_status
        self.posts = []

    def get(self, url, params=None, timeout=None):
        return _Resp(self._m, self._gs)

    def post(self, url, json=None, timeout=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return _Resp({}, self._ps)


NOW = dt.datetime(2026, 6, 15, 19, 30, tzinfo=UTC)


class TestExpectedCycle(unittest.TestCase):
    def test_floors_to_6h_boundary(self):
        # 19:30Z minus 6h lag = 13:30 -> floor to 12Z today
        self.assertEqual(w.expected_latest_cycle(NOW, 6), "2026061512")
        # 9h lag = 10:30 -> 06Z
        self.assertEqual(w.expected_latest_cycle(NOW, 9), "2026061506")
        # large lag rolls to the previous day
        self.assertEqual(w.expected_latest_cycle(dt.datetime(2026, 6, 15, 1, 0, tzinfo=UTC), 8),
                         "2026061412")


class TestDecide(unittest.TestCase):
    def test_behind_is_dispatched_current_is_not(self):
        # at 19:30, ecaie(lag6)->expect 12z, ecens(lag9)->expect 06z
        man = _manifest(ecaie="2026061506", ecens="2026061506", gefs="2026061512",
                        fnv3="2026061512", genc="2026061512")
        out = dict(w.decide_dispatches(man, NOW, {}))
        self.assertIn("ecaie", out)            # 06z < expected 12z -> behind
        self.assertNotIn("ecens", out)         # 06z == expected 06z -> current
        self.assertNotIn("gefs", out)          # ahead/current
        self.assertNotIn("fnv3", out)
        self.assertNotIn("genc", out)

    def test_missing_model_is_dispatched(self):
        out = dict(w.decide_dispatches(_manifest(ecens="2026061506"), NOW, {}))
        self.assertIn("ecaie", out)            # absent from manifest -> bootstrap poke

    def test_ahead_not_dispatched(self):
        man = _manifest(ecaie="2026061518", ecens="2026061512", gefs="2026061512",
                        fnv3="2026061512", genc="2026061512")
        self.assertEqual(w.decide_dispatches(man, NOW, {}), [])

    def test_cooldown_blocks_repoke(self):
        man = _manifest(ecaie="2026061506", ecens="2026061506", gefs="2026061512",
                        fnv3="2026061512", genc="2026061512")
        last = {"ecaie": NOW - dt.timedelta(minutes=10)}   # poked 10 min ago (< 40)
        self.assertNotIn("ecaie", dict(w.decide_dispatches(man, NOW, last)))
        last = {"ecaie": NOW - dt.timedelta(minutes=50)}   # 50 min ago (> 40) -> poke again
        self.assertIn("ecaie", dict(w.decide_dispatches(man, NOW, last)))


class TestRunOnce(unittest.TestCase):
    def _current_manifest(self):
        # everything current at NOW so only an explicitly-stale model fires
        return _manifest(ecens="2026061506", ecaie="2026061512", gefs="2026061512",
                         fnv3="2026061512", genc="2026061512")

    def test_fires_and_records_then_idempotent(self):
        man = self._current_manifest()
        man["models"][1]["latest"] = "2026061506"   # ecaie behind
        sess = _Session(man)
        fired_wfs = []
        last = {}
        fired = w.run_once(sess, "TOK", "o/r", last, now=NOW,
                           dispatch=lambda wf: (fired_wfs.append(wf) or 204))
        self.assertEqual([s for s, _ in fired], ["ecaie"])
        self.assertEqual(fired_wfs, ["update-aifs-ens.yml"])
        self.assertIn("ecaie", last)                 # recorded -> cooldown armed
        # immediate second tick: cooldown blocks re-poke
        fired2 = w.run_once(sess, "TOK", "o/r", last, now=NOW + dt.timedelta(minutes=5),
                            dispatch=lambda wf: (fired_wfs.append(wf) or 204))
        self.assertEqual(fired2, [])
        self.assertEqual(fired_wfs, ["update-aifs-ens.yml"])   # not fired again

    def test_no_token_does_not_dispatch(self):
        man = self._current_manifest(); man["models"][1]["latest"] = "2026061506"
        sess = _Session(man); calls = []
        fired = w.run_once(sess, None, "o/r", {}, now=NOW,
                           dispatch=lambda wf: (calls.append(wf) or 204))
        self.assertEqual(fired, [])                  # no token -> log only, no dispatch
        self.assertEqual(calls, [])

    def test_non_2xx_not_recorded(self):
        man = self._current_manifest(); man["models"][1]["latest"] = "2026061506"
        last = {}
        w.run_once(_Session(man), "TOK", "o/r", last, now=NOW, dispatch=lambda wf: 403)
        self.assertNotIn("ecaie", last)              # failed dispatch -> retry next tick

    def test_manifest_error_swallowed(self):
        sess = _Session({}, get_status=500)
        self.assertEqual(w.run_once(sess, "TOK", "o/r", {}, now=NOW), [])  # no crash


class TestDispatchRequest(unittest.TestCase):
    def test_builds_blank_inputs_request(self):
        sess = _Session({})
        code = w.dispatch_workflow(sess, "TOK", "WeathermanAAA/Triple-A-Tropics",
                                   "update-gefs.yml")
        self.assertEqual(code, 204)
        p = sess.posts[0]
        self.assertEqual(p["url"],
            "https://api.github.com/repos/WeathermanAAA/Triple-A-Tropics/"
            "actions/workflows/update-gefs.yml/dispatches")
        self.assertEqual(p["json"], {"ref": "main", "inputs": {}})   # blank cycle = never-miss
        self.assertEqual(p["headers"]["Authorization"], "Bearer TOK")


if __name__ == "__main__":
    unittest.main()
