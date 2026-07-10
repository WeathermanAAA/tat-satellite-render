"""GridSat-B1 deep-archive tier: pure-logic locks (no network)."""
import datetime as dt
import io
import unittest

import numpy as np

import gridsat

UTC = dt.timezone.utc


class TestSlots(unittest.TestCase):
    def test_slot_rounding(self):
        self.assertEqual(gridsat.slot_for(dt.datetime(1988, 9, 13, 18, 0, tzinfo=UTC)),
                         dt.datetime(1988, 9, 13, 18, 0, tzinfo=UTC))
        self.assertEqual(gridsat.slot_for(dt.datetime(1988, 9, 13, 19, 29, tzinfo=UTC)),
                         dt.datetime(1988, 9, 13, 18, 0, tzinfo=UTC))
        self.assertEqual(gridsat.slot_for(dt.datetime(1988, 9, 13, 19, 31, tzinfo=UTC)),
                         dt.datetime(1988, 9, 13, 21, 0, tzinfo=UTC))
        self.assertEqual(gridsat.slot_for(dt.datetime(1992, 8, 24, 23, 0, tzinfo=UTC)),
                         dt.datetime(1992, 8, 25, 0, 0, tzinfo=UTC))

    def test_key_layout(self):
        k = gridsat.key_for(dt.datetime(1988, 9, 13, 18, 0, tzinfo=UTC))
        self.assertEqual(
            k, "noaa-cdr-gridsat-b1-pds/data/1988/GRIDSAT-B1.1988.09.13.18.v02r01.nc")

    def test_candidates_ordered_by_distance(self):
        t = dt.datetime(2005, 8, 28, 12, 0, tzinfo=UTC)
        c = gridsat.candidate_slots(t, max_steps=2)
        self.assertEqual(c[0], t)
        deltas = [abs((s - t).total_seconds()) for s in c]
        self.assertEqual(deltas, sorted(deltas))

    def test_cutover_is_pre_abi(self):
        self.assertEqual(gridsat.ABI_CUTOVER,
                         dt.datetime(2017, 3, 1, tzinfo=UTC))

    def test_resolve_bounds(self):
        with self.assertRaises(gridsat.UnsupportedTimeError):
            gridsat.GRIDSAT.resolve(dt.datetime(1979, 12, 31, tzinfo=UTC))
        r = gridsat.GRIDSAT.resolve(dt.datetime(1980, 1, 1, tzinfo=UTC))
        self.assertEqual(r.bucket, gridsat.GRIDSAT_BUCKET)


class TestChannels(unittest.TestCase):
    def test_only_ir_and_wv(self):
        self.assertEqual(set(gridsat.GridSatB1Satellite.generic_to_band),
                         {"clean_ir", "wv_upper"})

    def test_can_see_is_lat_gated(self):
        sat = gridsat.GRIDSAT
        self.assertTrue(sat.can_see([-95, 12, -75, 28], None))
        self.assertFalse(sat.can_see([-95, 75, -75, 85], None))


class TestBtPng(unittest.TestCase):
    def test_lossless_roundtrip_with_nodata(self):
        from PIL import Image
        bt = np.array([[-90.0, 0.0], [40.0, np.nan]])
        png = gridsat.encode_bt_png(bt)
        im = np.array(Image.open(io.BytesIO(png)))
        dec = (im[..., 0].astype(float) * 256 + im[..., 1]) * gridsat.BT_SCALE + gridsat.BT_OFFSET
        self.assertEqual(im[1, 1, 3], 0)                    # NaN -> alpha 0
        self.assertTrue((im[..., 3][np.isfinite(bt)] == 255).all())
        for (i, j), v in np.ndenumerate(bt):
            if np.isfinite(v):
                self.assertAlmostEqual(dec[i, j], v, places=6)

    def test_from_fetch_flips_ascending_lat_to_row0_north(self):
        lats1 = np.array([10.0, 11.0, 12.0])               # ascending (GridSat)
        lons1 = np.array([100.0, 101.0])
        LON, LAT = np.meshgrid(lons1, lats1)
        cmi = np.array([[273.15, 273.15], [283.15, 283.15], [293.15, 293.15]])
        data = gridsat.FetchResult(
            cmi=cmi, lats=LAT.astype(np.float32), lons=LON.astype(np.float32),
            channel=1, generic_channel="clean_ir",
            scan_start=dt.datetime(2000, 1, 1, tzinfo=UTC),
            product="GRIDSAT-B1", bucket=gridsat.GRIDSAT_BUCKET,
            sat_name="GridSat-B1", sub_sat_lon=0.0, units="K")
        from PIL import Image
        im = np.array(Image.open(io.BytesIO(gridsat.bt_png_from_fetch(data))))
        dec = (im[..., 0].astype(float) * 256 + im[..., 1]) * 0.01 - 120.0
        self.assertAlmostEqual(dec[0, 0], 20.0, places=5)   # row 0 = NORTH = 20 C
        self.assertAlmostEqual(dec[2, 0], 0.0, places=5)


class TestCropIndices(unittest.TestCase):
    def test_basic_and_empty(self):
        lon = np.arange(-180.0, 180.0, 0.07)
        sl = gridsat._crop_indices(lon, -95.0, -75.0)
        self.assertTrue(lon[sl.start] >= -95.0)
        self.assertTrue(lon[sl.stop - 1] <= -75.0)
        empty = gridsat._crop_indices(lon, 200.0, 210.0)
        self.assertEqual(empty, slice(0, 0))


if __name__ == "__main__":
    unittest.main()
