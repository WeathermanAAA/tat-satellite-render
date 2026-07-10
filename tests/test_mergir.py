"""MergIR archive-tier locks: slot/url math, the best-available selector
gates, and the decode/crop seam against a SYNTHETIC granule (the live GES
DISC fetch needs Earthdata credentials -- no anonymous path exists, verified
2026-07-10 -- so the pure seams carry the test burden; a creds-gated live
smoke runs when the env has them)."""
import datetime as dt
import os
import unittest

import numpy as np

import gridsat
import mergir

UTC = dt.timezone.utc


class TestSlots(unittest.TestCase):
    def test_slot_rounding_30min(self):
        self.assertEqual(mergir.slot_for(dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC)),
                         dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC))
        self.assertEqual(mergir.slot_for(dt.datetime(2005, 8, 28, 18, 14, tzinfo=UTC)),
                         dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC))
        self.assertEqual(mergir.slot_for(dt.datetime(2005, 8, 28, 18, 16, tzinfo=UTC)),
                         dt.datetime(2005, 8, 28, 18, 30, tzinfo=UTC))

    def test_url_layout_and_half_index(self):
        url, half = mergir.url_for(dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC))
        self.assertEqual(url, "https://data.gesdisc.earthdata.nasa.gov/data/"
                              "MERGED_IR/GPM_MERGIR.1/2005/240/merg_2005082818_4km-pixel.nc4")
        self.assertEqual(half, 0)
        url2, half2 = mergir.url_for(dt.datetime(2005, 8, 28, 18, 30, tzinfo=UTC))
        self.assertEqual(url2, url)          # same hourly granule
        self.assertEqual(half2, 1)

    def test_record_start(self):
        self.assertEqual(mergir.MERGIR_START, dt.datetime(2000, 2, 7, tzinfo=UTC))
        with self.assertRaises(mergir.UnsupportedTimeError):
            mergir.MERGIR.resolve(dt.datetime(1999, 12, 31, tzinfo=UTC))

    def test_candidates_ordered(self):
        t = dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC)
        c = mergir.candidate_slots(t, max_steps=2)
        deltas = [abs((s - t).total_seconds()) for s in c]
        self.assertEqual(deltas, sorted(deltas))


class TestSelectorGates(unittest.TestCase):
    """The best-available-per-date rules the /render dispatch encodes."""

    def test_channel_set(self):
        # MergIR is 11 um ONLY -- WV must stay on GridSat for 2000-2017
        self.assertEqual(set(mergir.MergIRSatellite.generic_to_band), {"clean_ir"})
        self.assertIn("wv_upper", gridsat.GridSatB1Satellite.generic_to_band)

    def test_creds_gate(self):
        old = {k: os.environ.pop(k, None) for k in
               ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")}
        try:
            self.assertFalse(mergir.have_credentials())
            os.environ["EARTHDATA_TOKEN"] = "x"
            self.assertTrue(mergir.have_credentials())
            del os.environ["EARTHDATA_TOKEN"]
            os.environ["EARTHDATA_USERNAME"] = "u"
            self.assertFalse(mergir.have_credentials())   # password missing
            os.environ["EARTHDATA_PASSWORD"] = "p"
            self.assertTrue(mergir.have_credentials())
        finally:
            for k in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
                os.environ.pop(k, None)
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v

    def test_download_refuses_without_creds(self):
        old = {k: os.environ.pop(k, None) for k in
               ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")}
        try:
            with self.assertRaises(RuntimeError):
                mergir.download_granule("https://example.invalid/x.nc4")
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v


def _synthetic_granule(path):
    """A tiny merg-format netCDF4: Tb(time=2, lat, lon) with a known field."""
    import xarray as xr
    lat = np.arange(-59.982, 60.0, 0.5)          # coarse stand-in grid
    lon = np.arange(-179.982, 180.0, 0.5)
    LON, LAT = np.meshgrid(lon, lat)
    tb0 = (220.0 + LAT * 0.1 + LON * 0.01).astype(np.float32)
    tb1 = tb0 + 5.0
    ds = xr.Dataset(
        {"Tb": (("time", "lat", "lon"),
                np.stack([tb0, tb1]).astype(np.float32))},
        coords={"time": [0, 1], "lat": lat, "lon": lon})
    ds.to_netcdf(path, engine="h5netcdf")


class TestCropSeam(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix=".nc4", delete=False)
        _synthetic_granule(self.tmp.name)
        self.raw = open(self.tmp.name, "rb").read()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_crop_values_and_orientation(self):
        tb, lats, lons = mergir.crop_from_bytes(self.raw, 0, [-95.0, 12.0, -75.0, 28.0])
        self.assertTrue((lats >= 12.0).all() and (lats <= 28.0).all())
        self.assertTrue((lons >= -95.0).all() and (lons <= -75.0).all())
        # value check at a known cell (formula field)
        expect = 220.0 + lats[0] * 0.1 + lons[0] * 0.01
        self.assertAlmostEqual(float(tb[0, 0]), expect, places=3)

    def test_half_hour_selection(self):
        a, _, _ = mergir.crop_from_bytes(self.raw, 0, [-95, 12, -75, 28])
        b, _, _ = mergir.crop_from_bytes(self.raw, 1, [-95, 12, -75, 28])
        self.assertAlmostEqual(float(b[0, 0] - a[0, 0]), 5.0, places=4)

    def test_antimeridian_wrap(self):
        tb, lats, lons = mergir.crop_from_bytes(self.raw, 0, [170.0, 0.0, -170.0, 20.0])
        self.assertTrue((np.diff(lons) > 0).all())    # unwrapped monotonic
        self.assertGreater(lons.max(), 180.0)

    def test_lat_limit_honest(self):
        with self.assertRaises(mergir.CoverageError):
            mergir.crop_from_bytes(self.raw, 0, [-95.0, 65.0, -75.0, 80.0])


@unittest.skipUnless(mergir.have_credentials(), "needs Earthdata credentials")
class TestLiveKatrina(unittest.TestCase):
    def test_katrina_fetch(self):
        import asyncio
        t = dt.datetime(2005, 8, 28, 18, 0, tzinfo=UTC)
        bbox = [-95.0, 18.0, -80.0, 33.0]
        r = asyncio.run(mergir.MERGIR.find_file(t, "clean_ir", bbox, True))
        data = asyncio.run(mergir.MERGIR.fetch(r, bbox, "clean_ir"))
        self.assertEqual(data.units, "K")
        self.assertLess(float(np.nanmin(data.cmi)), 220.0)   # Katrina's cold tops


if __name__ == "__main__":
    unittest.main()
