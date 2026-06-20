#!/usr/bin/env python3
"""Self-heal watchdog tests for the meso poller (never-stale).

Proves the exit-on-stale wedge recovery WITHOUT a box: the decision fires ONLY
on a true full stall, never on normal 60 s / 2.5-min cadence or a cold start,
and the watchdog step force-exits (os._exit) when wedged so docker
`restart: always` recovers it clean.

Run: python -m pytest tests/test_meso_selfheal.py
"""
import datetime as dt
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import meso_poller as mp                       # noqa: E402


def _frames(ages: dict):
    """{slug: last_frame_utc} where ages maps slug -> seconds-since-last-upload
    (or None = never produced since (re)start). Returns (last_frame_utc, now)."""
    now = mp.utcnow()
    lf = {}
    for s in mp.MESO_SECTORS:
        a = ages.get(s.slug, None)
        if a is not None:
            lf[s.slug] = now - dt.timedelta(seconds=a)
    return lf, now


def _families():
    goes = [s for s in mp.MESO_SECTORS if s.family != "himawari"]
    hima = [s for s in mp.MESO_SECTORS if s.family == "himawari"]
    return goes, hima


class SelfHealDecide(unittest.TestCase):
    def test_all_stale_fires(self):
        h, now = _frames({s.slug: 5000 for s in mp.MESO_SECTORS})  # past any threshold
        wedged, why = mp.selfheal_decide(h, now)
        self.assertTrue(wedged)
        self.assertIn(">", why)

    def test_one_fresh_blocks_fire(self):
        ages = {s.slug: 5000 for s in mp.MESO_SECTORS}
        ages[mp.MESO_SECTORS[0].slug] = 30        # one sector still producing
        h, now = _frames(ages)
        self.assertFalse(mp.selfheal_decide(h, now)[0])

    def test_normal_cadence_no_false_trip(self):
        # worst HEALTHY upload gap (a couple of missed scans): GOES ~180 s,
        # Himawari ~270 s -- both must read fresh.
        ages = {s.slug: (270 if s.family == "himawari" else 180)
                for s in mp.MESO_SECTORS}
        h, now = _frames(ages)
        self.assertFalse(mp.selfheal_decide(h, now)[0])

    def test_single_late_scan_no_false_trip(self):
        # a one-off doubled gap (GOES ~120 s, Himawari ~300 s) is well under
        # the thresholds -> never trips on a single late scan.
        ages = {s.slug: (300 if s.family == "himawari" else 120)
                for s in mp.MESO_SECTORS}
        h, now = _frames(ages)
        self.assertFalse(mp.selfheal_decide(h, now)[0])

    def test_cold_start_all_none_no_fire(self):
        h, now = _frames({s.slug: None for s in mp.MESO_SECTORS})
        self.assertFalse(mp.selfheal_decide(h, now)[0])

    def test_per_source_thresholds(self):
        goes, hima = _families()
        self.assertTrue(goes and hima, "need both families in MESO_SECTORS")
        # 700 s: past GOES 600 but under Himawari 900 -> Himawari fresh -> no fire
        ages = {s.slug: 700 for s in mp.MESO_SECTORS}
        h, now = _frames(ages)
        self.assertFalse(mp.selfheal_decide(h, now)[0])
        # push Himawari past 900 too -> ALL stale -> fire
        for s in hima:
            ages[s.slug] = 1000
        h, now = _frames(ages)
        self.assertTrue(mp.selfheal_decide(h, now)[0])


class SelfHealAct(unittest.TestCase):
    def _poller(self, ages):
        p = mp.MesoPoller.__new__(mp.MesoPoller)   # bypass __init__ (no R2 / net)
        p._last_frame_utc, _ = _frames(ages)
        return p

    def test_step_force_exits_on_wedge(self):
        p = self._poller({s.slug: 5000 for s in mp.MESO_SECTORS})
        with mock.patch.object(mp.os, "_exit") as ex:
            p._selfheal_step()
            ex.assert_called_once_with(1)

    def test_step_noop_at_normal_cadence(self):
        p = self._poller({s.slug: (270 if s.family == "himawari" else 180)
                          for s in mp.MESO_SECTORS})
        with mock.patch.object(mp.os, "_exit") as ex:
            self.assertFalse(p._selfheal_step())
            ex.assert_not_called()

    def test_step_noop_at_cold_start(self):
        p = self._poller({s.slug: None for s in mp.MESO_SECTORS})
        with mock.patch.object(mp.os, "_exit") as ex:
            p._selfheal_step()
            ex.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
