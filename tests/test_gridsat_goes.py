"""GridSat-GOES per-satellite deep tier: pure-logic locks (no network)."""
import datetime as dt
import unittest

import gridsat_goes as gg

UTC = dt.timezone.utc


class TestSlots(unittest.TestCase):
    def test_slot_rounding_hourly(self):
        self.assertEqual(gg.slot_for(dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC)),
                         dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC))
        self.assertEqual(gg.slot_for(dt.datetime(2005, 8, 28, 18, 29, tzinfo=UTC)),
                         dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC))
        self.assertEqual(gg.slot_for(dt.datetime(2005, 8, 28, 18, 31, tzinfo=UTC)),
                         dt.datetime(2005, 8, 28, 19, 0, tzinfo=UTC))

    def test_url_layout_katrina(self):
        u = gg.url_for(12, dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC))
        self.assertEqual(
            u, "https://www.ncei.noaa.gov/data/gridsat-goes/access/goes/2005/08/"
               "GridSat-GOES.goes12.2005.08.28.1800.v01.nc")

    def test_candidates_ordered_by_distance(self):
        t = dt.datetime(2005, 8, 28, 18, 20, tzinfo=UTC)
        c = gg.candidate_slots(t)
        self.assertEqual(c[0], dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC))
        self.assertEqual(set(c[1:3]),
                         {dt.datetime(2005, 8, 28, 17, 0, tzinfo=UTC),
                          dt.datetime(2005, 8, 28, 19, 0, tzinfo=UTC)})


class TestPositions(unittest.TestCase):
    def test_katrina_era_pair(self):
        pos = gg.positions_at(dt.datetime(2005, 8, 28, tzinfo=UTC))
        self.assertEqual(pos.get(12), -75.0)     # East
        self.assertEqual(pos.get(10), -135.0)    # West
        self.assertEqual(pos.get(9), 155.0)      # WPac backup era

    def test_modern_era_pair(self):
        pos = gg.positions_at(dt.datetime(2016, 10, 6, tzinfo=UTC))   # Matthew
        self.assertEqual(pos.get(13), -75.0)
        self.assertEqual(pos.get(15), -135.0)
        self.assertNotIn(12, pos)                # S.America stint ended 2013

    def test_record_bounds(self):
        with self.assertRaises(Exception):
            gg.GRIDSAT_GOES.resolve(dt.datetime(1993, 1, 1, tzinfo=UTC))
        with self.assertRaises(Exception):
            gg.GRIDSAT_GOES.resolve(dt.datetime(2018, 6, 1, tzinfo=UTC))
        gg.GRIDSAT_GOES.resolve(dt.datetime(2005, 8, 28, tzinfo=UTC))  # ok


class TestRanking(unittest.TestCase):
    def setUp(self):
        # seed the month-listing cache: Katrina hour has East+West files
        slot = dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC)
        self.slot = slot
        stamp = f"{slot:%Y.%m.%d.%H%M}"
        with gg._listing_lock:
            gg._listing_cache[(2005, 8)] = {(12, stamp), (10, stamp), (9, stamp)}

    def tearDown(self):
        with gg._listing_lock:
            gg._listing_cache.clear()

    def test_gulf_prefers_east(self):
        r = gg.rank_candidates([-95, 20, -80, 32], self.slot)
        self.assertEqual(r[0][0], 12)            # GOES-12 East first
        self.assertIn(10, [n for n, _ in r])     # West still a fallback

    def test_epac_prefers_west(self):
        r = gg.rank_candidates([-135, 5, -115, 25], self.slot)
        self.assertEqual(r[0][0], 10)

    def test_wpac_uses_goes9_backup(self):
        r = gg.rank_candidates([135, 10, 155, 30], self.slot)
        self.assertEqual(r[0][0], 9)             # 155°E WPac backup
        self.assertEqual(r[0][1], 155.0)

    def test_out_of_disk_is_empty(self):
        # central Indian Ocean: none of the 2005 GOES fleet sees it
        r = gg.rank_candidates([60, -20, 80, 0], self.slot)
        self.assertEqual(r, [])


class TestChannels(unittest.TestCase):
    def test_channel_map(self):
        self.assertEqual(gg._CHANNEL_VARS["clean_ir"], "ch4")
        self.assertEqual(gg._CHANNEL_VARS["wv_upper"], "ch3")
        self.assertEqual(gg._CHANNEL_VARS["visible_red"], "ch1")
        self.assertEqual(
            set(gg.GridSatGoesSatellite.generic_to_band) - set(gg._CHANNEL_VARS),
            set())


if __name__ == "__main__":
    unittest.main()
