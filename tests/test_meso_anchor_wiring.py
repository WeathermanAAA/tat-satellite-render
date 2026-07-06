"""Anchor wiring inside MesoPoller: _update_anchor holds/snaps + persists, and
_load_anchors restores across a restart. Proves the discovery thread keeps the
loop locked and that a restart doesn't snap a locked loop.

Run from the repo root:  python -m unittest tests.test_meso_anchor_wiring
"""

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meso_anchor as ma  # noqa: E402
import meso_poller as mp  # noqa: E402
from meso_sectors import MESO_SECTORS_BY_SLUG  # noqa: E402

NOW = dt.datetime(2026, 6, 20, 22, 40, tzinfo=dt.timezone.utc)
HIMA = [128.016, 10.133, 137.944, 19.768]
GOES = [-111.911, 32.246, -90.641, 46.912]


class StubR2:
    def __init__(self):
        self.store = {}

    def put_json(self, k, o, c):
        self.store[k] = o
        return True

    def get_json(self, k):
        return self.store.get(k)


def _poller():
    p = mp.MesoPoller.__new__(mp.MesoPoller)
    p.anchors = {}
    p.extents = {}
    p.r2 = StubR2()
    return p


def _ext(bbox, sat="Himawari-9"):
    return mp.SectorExtent(bbox=list(bbox), scan_start=NOW, sat_name=sat)


def _shift(bbox, dlon, dlat):
    return [round(bbox[0] + dlon, 3), round(bbox[1] + dlat, 3),
            round(bbox[2] + dlon, 3), round(bbox[3] + dlat, 3)]


class UpdateAnchorTests(unittest.TestCase):
    def test_himawari_inits_holds_drifts(self):
        p = _poller()
        sec = MESO_SECTORS_BY_SLUG["himawari9-meso"]
        p._update_anchor(sec, _ext(HIMA), NOW)
        a0 = p.anchors[sec.slug].bbox
        self.assertTrue(ma.bbox_contains(HIMA, a0))         # inset, covered
        self.assertLess(ma.lon_span(a0), ma.lon_span(HIMA))  # himawari margin > 0
        # persisted to R2
        self.assertIsNotNone(p.r2.get_json(mp.anchor_key(sec.slug)))
        # identical + small-drift scans HOLD -> the loop stays pixel-locked
        p._update_anchor(sec, _ext(HIMA), NOW)
        self.assertEqual(p.anchors[sec.slug].bbox, a0)
        p._update_anchor(sec, _ext(_shift(HIMA, -0.1, 0.05)), NOW)
        self.assertEqual(p.anchors[sec.slug].bbox, a0)
        # a drift past the margin re-anchors, KEEPS span, stays covered
        p._update_anchor(sec, _ext(_shift(HIMA, -0.8, 0.0)), NOW)
        a1 = p.anchors[sec.slug].bbox
        self.assertNotEqual(a1, a0)
        self.assertAlmostEqual(ma.lon_span(a1), ma.lon_span(a0), places=2)
        self.assertTrue(ma.bbox_contains(_shift(HIMA, -0.8, 0.0), a1))

    def test_goes_anchor_equals_live_box(self):
        p = _poller()
        sec = MESO_SECTORS_BY_SLUG["goes19-m1"]
        p._update_anchor(sec, _ext(GOES, sat="GOES-19"), NOW)
        # margin 0 -> anchor is the operator box exactly (old behaviour)
        self.assertEqual(p.anchors[sec.slug].bbox, GOES)
        # genuine reposition snaps to the new box exactly
        moved = [-90.0, 30.0, -68.73, 44.67]
        p._update_anchor(sec, _ext(moved, sat="GOES-19"), NOW)
        self.assertEqual(p.anchors[sec.slug].bbox, moved)

    def test_load_anchors_restores(self):
        p = _poller()
        sec = MESO_SECTORS_BY_SLUG["himawari9-meso"]
        p._update_anchor(sec, _ext(HIMA), NOW)
        saved = p.anchors[sec.slug].bbox
        # a fresh poller sharing the same R2 store reloads the anchor (no snap)
        p2 = mp.MesoPoller.__new__(mp.MesoPoller)
        p2.anchors = {}
        p2.r2 = p.r2
        p2._load_anchors()
        self.assertIn(sec.slug, p2.anchors)
        self.assertEqual(p2.anchors[sec.slug].bbox, saved)
        # and the reloaded anchor HOLDS against the same live box (no restart snap)
        p2.extents = {}
        p2._update_anchor(sec, _ext(HIMA), NOW)
        self.assertEqual(p2.anchors[sec.slug].bbox, saved)


if __name__ == "__main__":
    unittest.main()
