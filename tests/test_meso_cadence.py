"""Per-family cold-band cadence.

The meso poller stretches cold bands (WV / true-color / SWIR) past the hot
target so they never starve the hot IR/IRBD lane. The floor is per-family:
GOES at COLD_CADENCE_TARGET_S (300 s), Himawari at COLD_CADENCE_TARGET_HIMA_S
(150 s) so all six Himawari bands track the ~2.5-min Target sub-scan cadence its
hot bands already ride. These tests pin that split and prove the scheduler uses
the unit's own family floor when it reschedules a cold unit.

Run from the repo root:  python -m unittest tests.test_meso_cadence
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meso_poller as mp  # noqa: E402


def _poller():
    """A MesoPoller with the real sector×band unit set but no R2/network (bypass
    __init__)."""
    p = mp.MesoPoller.__new__(mp.MesoPoller)
    p.limiter = mp.RateLimiter(mp.RATE_MIN_SPACING_S)
    p.units = {}
    for sector in mp.MESO_SECTORS:
        for band in mp.BANDS:
            p.units[(sector.slug, band.key)] = mp.Unit(sector=sector, band=band)
    p._circuit_open_until = 0.0
    p._consec_render_fail = 0
    return p


class ColdCadenceFamily(unittest.TestCase):
    def setUp(self):
        self.p = _poller()

    def test_defaults(self):
        # The shipped defaults: GOES 300 s, Himawari 150 s.
        self.assertEqual(mp.COLD_CADENCE_TARGET_S, 300.0)
        self.assertEqual(mp.COLD_CADENCE_TARGET_HIMA_S, 150.0)

    def test_himawari_cold_is_150(self):
        self.assertEqual(self.p.cold_cadence("himawari"),
                         mp.COLD_CADENCE_TARGET_HIMA_S)

    def test_goes_cold_is_300(self):
        self.assertEqual(self.p.cold_cadence("goes"), mp.COLD_CADENCE_TARGET_S)

    def test_default_family_is_goes(self):
        # No argument == GOES (the conservative floor), so a stray callsite can
        # never accidentally speed a sector up.
        self.assertEqual(self.p.cold_cadence(), self.p.cold_cadence("goes"))

    def test_himawari_strictly_faster_than_goes(self):
        self.assertLess(self.p.cold_cadence("himawari"),
                        self.p.cold_cadence("goes"))

    def test_floor_dominates_rate_budget(self):
        # On the internal render URL min_spacing is ~1 s, so n_cold*min_spacing
        # (a couple dozen seconds) stays well under either floor -> the family
        # floor is what actually sets the cadence, not the rate-budget term.
        n_cold = sum(1 for u in self.p.units.values() if not u.band.hot)
        self.assertLess(n_cold * self.p.limiter.min_spacing,
                        mp.COLD_CADENCE_TARGET_HIMA_S)

    def test_himawari_sector_exists(self):
        fams = {s.family for s in mp.MESO_SECTORS}
        self.assertIn("himawari", fams)
        self.assertIn("goes", fams)


class TickRescheduleUsesFamilyFloor(unittest.TestCase):
    """The cold lane reschedules a processed unit at that unit's OWN family
    cadence -- a Himawari cold unit ~150 s out, a GOES cold unit ~300 s out."""

    def setUp(self):
        self.p = _poller()

    def _only_due(self, target):
        # Park everything far in the future, then make exactly `target` due now.
        future = time.monotonic() + 10_000
        for u in self.p.units.values():
            u.next_due = future
        target.next_due = time.monotonic() - 5

    def _cold_unit(self, family):
        return next(u for u in self.p.units.values()
                    if u.sector.family == family and not u.band.hot)

    def test_himawari_cold_reschedules_near_150(self):
        u = self._cold_unit("himawari")
        self._only_due(u)
        with mock.patch.object(self.p, "process_unit", lambda unit: None):
            t0 = time.monotonic()
            self.p.tick()
        self.assertAlmostEqual(u.next_due - t0,
                               mp.COLD_CADENCE_TARGET_HIMA_S, delta=2.0)

    def test_goes_cold_reschedules_near_300(self):
        u = self._cold_unit("goes")
        self._only_due(u)
        with mock.patch.object(self.p, "process_unit", lambda unit: None):
            t0 = time.monotonic()
            self.p.tick()
        self.assertAlmostEqual(u.next_due - t0,
                               mp.COLD_CADENCE_TARGET_S, delta=2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
