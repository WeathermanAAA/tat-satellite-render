"""Per-family cold-lane cadence in the hot/cold LANE model.

The meso poller runs two isolated lanes (hot ir/irbd vs cold wv/true-color/swir),
each on its own thread + render container. lane_cadence() floors the COLD lane
per family: Himawari at COLD_CADENCE_TARGET_HIMA_S (150 s) so all six Himawari
bands track the ~2.5-min Target sub-scan cadence; GOES at COLD_CADENCE_TARGET_S
(300 s). The hot lane is a flat CADENCE_TARGET_S (60 s) regardless. Because the
cold lane services only the most-overdue unit per pass, a Himawari unit due every
150 s is simply picked more often than a GOES unit due every 300 s -- the hot
lane is never touched.

Run from the repo root:  python -m unittest tests.test_meso_cadence
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meso_poller as mp  # noqa: E402


def _poller_and_lanes():
    """A MesoPoller (bypass __init__, no R2/net) plus hot/cold lanes built the
    same way __init__ partitions them."""
    p = mp.MesoPoller.__new__(mp.MesoPoller)
    cold = [mp.Unit(sector=s, band=b)
            for s in mp.MESO_SECTORS for b in mp.BANDS if not b.hot]
    hot = [mp.Unit(sector=s, band=b)
           for s in mp.MESO_SECTORS for b in mp.BANDS if b.hot]
    cold_lane = mp.Lane("cold", cold, "http://x/render",
                        mp.RATE_MIN_SPACING_S, drain_all=False)
    hot_lane = mp.Lane("hot", hot, "http://x/render",
                       mp.RATE_MIN_SPACING_S, drain_all=True)
    return p, hot_lane, cold_lane


def _unit(lane, family):
    return next(u for u in lane.units if u.sector.family == family)


class LaneCadenceFamily(unittest.TestCase):
    def setUp(self):
        self.p, self.hot, self.cold = _poller_and_lanes()

    def test_defaults(self):
        self.assertEqual(mp.COLD_CADENCE_TARGET_S, 300.0)
        self.assertEqual(mp.COLD_CADENCE_TARGET_HIMA_S, 150.0)
        self.assertEqual(mp.CADENCE_TARGET_S, 60.0)

    def test_himawari_cold_is_150(self):
        self.assertEqual(
            self.p.lane_cadence(self.cold, _unit(self.cold, "himawari")),
            mp.COLD_CADENCE_TARGET_HIMA_S)

    def test_goes_cold_is_300(self):
        self.assertEqual(
            self.p.lane_cadence(self.cold, _unit(self.cold, "goes")),
            mp.COLD_CADENCE_TARGET_S)

    def test_himawari_cold_strictly_faster_than_goes(self):
        self.assertLess(
            self.p.lane_cadence(self.cold, _unit(self.cold, "himawari")),
            self.p.lane_cadence(self.cold, _unit(self.cold, "goes")))

    def test_hot_lane_is_flat_60_regardless_of_family(self):
        for fam in ("himawari", "goes"):
            self.assertEqual(
                self.p.lane_cadence(self.hot, _unit(self.hot, fam)),
                mp.CADENCE_TARGET_S)

    def test_no_unit_defaults_to_goes_floor(self):
        # A stray callsite without a unit must NOT accidentally speed a lane up.
        self.assertEqual(self.p.lane_cadence(self.cold), mp.COLD_CADENCE_TARGET_S)

    def test_family_floor_dominates_internal_rate_budget(self):
        # On the internal render URL (min_spacing ~1 s) the rate-budget term
        # (len(units) x spacing) stays under either floor, so the family floor is
        # what sets the cadence -- not the rate term.
        self.assertLess(len(self.cold.units) * self.cold.limiter.min_spacing,
                        mp.COLD_CADENCE_TARGET_HIMA_S)

    def test_both_families_present_in_cold_lane(self):
        fams = {u.sector.family for u in self.cold.units}
        self.assertIn("himawari", fams)
        self.assertIn("goes", fams)


if __name__ == "__main__":
    unittest.main(verbosity=2)
