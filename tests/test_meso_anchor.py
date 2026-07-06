"""Stable common-extent ("anchor") logic -- the meso loop-jitter cure.

The meso poller used to render every band frame to the live per-scan box. GOES
M1/M2 is byte-stable within an operator position (so that was fine between the
1-2 genuine repositions/day), but the Himawari Target is re-steered continuously
to keep a storm centred, so it crept every slot -> the loop wandered. meso_anchor
pins each sector's loop to ONE reference box, HELD pixel-locked until the live box
no longer covers it (drift/reposition) or its span changes (zoom), then SNAPS.

These tests prove the invariants the loop's smoothness AND correctness rest on:
locked-while-covered, no-blank-edges (the anchor is always backed by data),
constant dimensions across a drift snap, GOES reduces to the old behaviour, and
antimeridian safety.

Run from the repo root:  python -m unittest tests.test_meso_anchor
"""

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meso_anchor as ma  # noqa: E402
from meso_anchor import Anchor, plan_anchor  # noqa: E402


# Representative live boxes (rounded to the poller's 3-decimal discovery).
HIMA = [128.016, 10.133, 137.944, 19.768]   # ~9.93 x 9.64 deg
GOES = [-111.911, 32.246, -90.641, 46.912]  # ~21.27 x 14.67 deg
HIMA_MARGIN = 0.6
GOES_MARGIN = 0.0
KW = dict(cover_buffer=0.02, span_tol=0.12)


def _shift(bbox, dlon, dlat):
    return [round(bbox[0] + dlon, 3), round(bbox[1] + dlat, 3),
            round(bbox[2] + dlon, 3), round(bbox[3] + dlat, 3)]


def _plan(prev, disc, margin, drift_mult=1.5, max_span=80.0):
    return plan_anchor(prev, disc, margin_lon=margin, margin_lat=margin,
                       drift_limit=drift_mult * margin, max_span=max_span, **KW)


class GeometryTests(unittest.TestCase):
    def test_cwh_roundtrip(self):
        for bb in (HIMA, GOES):
            clon, clat, w, h = ma.bbox_to_cwh(bb)
            back = ma.cwh_to_bbox(clon, clat, w, h)
            for a, b in zip(bb, back):
                self.assertAlmostEqual(a, b, places=2)

    def test_lon_span_and_center_antimeridian(self):
        cross = [170.0, -30.0, -170.0, -20.0]  # 20 deg wide across +/-180
        self.assertAlmostEqual(ma.lon_span(cross), 20.0, places=6)
        self.assertAlmostEqual(ma.center_lon(cross), 180.0, places=6)
        # non-crossing
        self.assertAlmostEqual(ma.lon_span(HIMA), 9.928, places=3)
        self.assertAlmostEqual(ma.center_lon(HIMA), 132.98, places=2)

    def test_inset_shrinks_and_never_inverts(self):
        ins = ma.inset_bbox(HIMA, 0.6, 0.6)
        self.assertGreater(ma.lon_span(ins), 0)
        self.assertLess(ma.lon_span(ins), ma.lon_span(HIMA))
        self.assertTrue(ma.bbox_contains(HIMA, ins))
        # margin larger than the box: clamped, stays positive + inside.
        tiny = [0.0, 0.0, 1.0, 1.0]
        big = ma.inset_bbox(tiny, 5.0, 5.0)
        self.assertGreater(ma.lon_span(big), 0)
        self.assertGreater(big[3] - big[1], 0)
        self.assertTrue(ma.bbox_contains(tiny, big))

    def test_contains_buffer(self):
        inner = ma.inset_bbox(HIMA, 0.6, 0.6)
        self.assertTrue(ma.bbox_contains(HIMA, inner))
        # a box poking 0.1 deg west of HIMA is NOT contained
        out = _shift(inner, -1.0, 0)
        self.assertFalse(ma.bbox_contains(HIMA, out))

    def test_contains_antimeridian(self):
        outer = [170.0, -30.0, -170.0, -20.0]   # crosses, 20 deg
        inner = [175.0, -28.0, -175.0, -22.0]   # inside, 10 deg
        self.assertTrue(ma.bbox_contains(outer, inner))
        self.assertFalse(ma.bbox_contains(outer, [160.0, -28.0, -175.0, -22.0]))


class HoldAndDriftTests(unittest.TestCase):
    def test_init_then_hold_byte_identical(self):
        a0, r0 = _plan(None, HIMA, HIMA_MARGIN)
        self.assertEqual(r0, "init")
        self.assertTrue(ma.bbox_contains(HIMA, a0))
        # next scan identical live box -> HOLD (loop pixel-locked).
        a1, r1 = _plan(a0, HIMA, HIMA_MARGIN)
        self.assertEqual(r1, "hold")
        self.assertEqual(a1, a0)

    def test_small_drift_within_margin_holds(self):
        a0, _ = _plan(None, HIMA, HIMA_MARGIN)
        # creep the live box 0.1 deg (well within the 0.6 margin) for many slots
        live = HIMA
        a = a0
        for _ in range(4):
            live = _shift(live, -0.1, 0.05)
            a, r = _plan(a, live, HIMA_MARGIN)
            self.assertEqual(r, "hold")     # stays locked
            self.assertEqual(a, a0)         # identical frame extent

    def test_drift_beyond_margin_recenters_constant_span(self):
        a0, _ = _plan(None, HIMA, HIMA_MARGIN)
        live = _shift(HIMA, -0.8, 0.0)      # drift past the margin
        a1, r1 = _plan(a0, live, HIMA_MARGIN)
        self.assertEqual(r1, "recenter")
        # span (=> pixel dimensions) preserved across the drift snap
        self.assertAlmostEqual(ma.lon_span(a1), ma.lon_span(a0), places=2)
        self.assertAlmostEqual(a1[3] - a1[1], a0[3] - a0[1], places=2)
        # still fully backed by data (no blank edge)
        self.assertTrue(ma.bbox_contains(live, a1))

    def test_zoom_reinsets_new_dims(self):
        a0, _ = _plan(None, HIMA, HIMA_MARGIN)
        # operator widens the Target by ~40% -> a genuine zoom
        wide = [126.0, 8.0, 140.0, 22.0]
        a1, r1 = _plan(a0, wide, HIMA_MARGIN)
        self.assertEqual(r1, "zoom")
        self.assertGreater(ma.lon_span(a1), ma.lon_span(a0))
        self.assertTrue(ma.bbox_contains(wide, a1))

    def test_glitch_near_global_discovery_holds_prev(self):
        # A glitchy discovery whose bounding box ballooned (bad nav / wrong
        # product) must NOT balloon the anchor -- the last-known-good anchor is
        # held until a plausible scan returns.
        a0, _ = _plan(None, HIMA, HIMA_MARGIN)
        glitch = [-178.0, -80.0, 178.0, 80.0]   # span 356 x 160 deg
        a1, r1 = _plan(a0, glitch, HIMA_MARGIN)
        self.assertEqual(r1, "hold")
        self.assertEqual(a1, a0)                 # anchor unchanged
        self.assertLess(ma.lon_span(a1), 40.0)   # never near-global
        # and the next good scan re-anchors normally
        a2, r2 = _plan(a1, _shift(HIMA, -0.05, 0.0), HIMA_MARGIN)
        self.assertEqual(r2, "hold")

    def test_reposition_backstop_when_recenter_uncovered(self):
        # A small-margin case where the live box shrinks within span_tol at the
        # SAME centre: a recenter keeping the old (larger) span would poke
        # outside the smaller live box -> the backstop falls back to an inset of
        # the live box, which IS covered. (reason 'reposition'.)
        prev = [125.0, 10.0, 135.0, 20.0]      # span 10 x 10
        live = [125.5, 10.5, 134.5, 19.5]      # span 9 x 9, same centre (within tol)
        a1, r1 = plan_anchor(prev, live, margin_lon=0.1, margin_lat=0.1,
                             drift_limit=0.15, **KW)
        self.assertEqual(r1, "reposition")
        self.assertTrue(ma.bbox_contains(live, a1))  # ALWAYS covered, no blank edge
        self.assertLess(ma.lon_span(a1), ma.lon_span(live))


class GoesReducesToOldBehaviourTests(unittest.TestCase):
    def test_margin_zero_anchor_equals_live(self):
        a0, r0 = _plan(None, GOES, GOES_MARGIN)
        self.assertEqual(r0, "init")
        self.assertEqual(a0, GOES)                     # full operator-box coverage
        # byte-stable within a position -> hold
        a1, r1 = _plan(a0, GOES, GOES_MARGIN)
        self.assertEqual(r1, "hold")
        self.assertEqual(a1, GOES)

    def test_genuine_reposition_snaps_to_new_live(self):
        a0, _ = _plan(None, GOES, GOES_MARGIN)
        moved = _shift([-90.0, 30.0, -68.7, 44.7], 0, 0)  # ~26 deg away, same span
        a1, r1 = _plan(a0, moved, GOES_MARGIN)
        self.assertNotEqual(r1, "hold")
        self.assertEqual(a1, moved)                    # exactly the new operator box
        self.assertTrue(ma.bbox_contains(moved, a1))

    def test_goes_zoom_snaps_to_new_box(self):
        a0, _ = _plan(None, [-100.0, 30.0, -87.0, 42.0], GOES_MARGIN)  # 13 deg
        wide = [-112.0, 32.0, -90.0, 47.0]   # 22 deg, a real M1 widen
        a1, r1 = _plan(a0, wide, GOES_MARGIN)
        self.assertEqual(r1, "zoom")
        self.assertEqual(a1, wide)


class CoverageInvariantTests(unittest.TestCase):
    def test_every_anchor_is_covered_over_a_long_drift(self):
        """The render box must stay backed by data. Walk a long, jittering
        Himawari drift (the real failure mode) and assert the chosen anchor pokes
        outside the live box by at most cover_buffer (the guaranteed worst-case
        blank strip) on every single step -- and that GOES (margin 0) never pokes."""
        cover_buffer = KW["cover_buffer"]
        a = None
        live = HIMA
        steps = 0
        snaps = 0
        worst_poke = 0.0
        # deterministic pseudo-walk (no Math.random): a westward creep with a
        # small lat sawtooth, like the measured 07W Target track.
        for i in range(200):
            dlon = -0.12 if i % 3 else -0.18
            dlat = 0.07 if i % 2 else -0.05
            live = _shift(live, dlon, dlat)
            new, r = _plan(a, live, HIMA_MARGIN)
            poke = ma.coverage_poke(live, new)
            worst_poke = max(worst_poke, poke)
            self.assertLessEqual(poke, cover_buffer + 1e-9,
                                 f"step {i}: anchor {new} pokes {poke} deg outside "
                                 f"live {live} (> buffer {cover_buffer})")
            if a is not None and r != "hold":
                snaps += 1
            a = new
            steps += 1
        # GOES (margin 0): the anchor equals the live data box -> poke <= the
        # sub-pixel cover_buffer (and 0 on any real reposition step).
        ga = None
        gl = GOES
        for i in range(50):
            gl = _shift(gl, -0.15, 0.05)
            gnew, _ = _plan(ga, gl, GOES_MARGIN)
            self.assertLessEqual(ma.coverage_poke(gl, gnew), cover_buffer + 1e-9)
            ga = gnew
        holds = steps - snaps - 1  # -1 for the init step
        # Over this (deliberately fast) drift the anchor must snap sometimes
        # (locked != frozen) but MOST steps must be holds -- a locked loop, not
        # the old per-step wander. (At the real ~0.008 deg/frame Target creep the
        # snap rate is far lower still -- see the real-data simulation.)
        self.assertGreater(snaps, 0)
        self.assertGreater(holds, steps // 2)
        self.assertLess(snaps, steps // 3)


class AnchorPersistenceTests(unittest.TestCase):
    def test_json_roundtrip(self):
        a = Anchor(bbox=HIMA, source_scan=dt.datetime(2026, 6, 20, 22, 30,
                   tzinfo=dt.timezone.utc), set_utc=dt.datetime(2026, 6, 20, 22,
                   31, tzinfo=dt.timezone.utc), reason="recenter")
        b = Anchor.from_json(a.to_json())
        self.assertIsNotNone(b)
        self.assertEqual(b.bbox, [round(v, 3) for v in HIMA])
        self.assertEqual(b.reason, "recenter")
        self.assertEqual(b.source_scan, a.source_scan)

    def test_corrupt_returns_none(self):
        ok_ts = {"source_scan": "20260620T223000Z", "set_utc": "20260620T223100Z"}
        bad_boxes = (
            {},                                            # no bbox
            {"bbox": [1, 2, 3], **ok_ts},                  # wrong length
            {"bbox": "x", "source_scan": "y", "set_utc": "z"},  # unparseable
            {"bbox": [0, 5, 1, 4], **ok_ts},               # lat inverted
            {"bbox": [float("nan"), 0, 1, 1], **ok_ts},    # NaN lon edge
            {"bbox": [float("inf"), 0, 1, 1], **ok_ts},    # Inf lon edge
            {"bbox": [10, 0, 10, 1], **ok_ts},             # zero-width (w==e)
        )
        for bad in bad_boxes:
            self.assertIsNone(Anchor.from_json(bad), bad)
        # a valid antimeridian-crossing box still round-trips
        good = {"bbox": [170.0, -30.0, -170.0, -20.0], **ok_ts}
        self.assertIsNotNone(Anchor.from_json(good))


if __name__ == "__main__":
    unittest.main()
