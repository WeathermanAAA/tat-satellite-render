"""Pure-logic unit tests for the wide-area day/night mosaic (mosaic.py) — the
solar-zenith terminator, the view-quality compositing weight, and the scatter
rasterization. No network / no satellite fetch (that path is exercised live)."""
import datetime as dt
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mosaic  # noqa: E402


class TestSmoothstep(unittest.TestCase):
    def test_monotonic_clamped(self):
        x = np.linspace(-5, 15, 50)
        s = mosaic._smoothstep(0.0, 10.0, x)
        self.assertAlmostEqual(float(s[0]), 0.0)
        self.assertAlmostEqual(float(s[-1]), 1.0)
        self.assertTrue(np.all(np.diff(s) >= -1e-9))   # non-decreasing


class TestSolarZenith(unittest.TestCase):
    def test_subsolar_vs_antipode(self):
        # 2026-06-27 12:00Z: sun ~over 0degE near the Tropic of Cancer (solstice-ish).
        when = dt.datetime(2026, 6, 27, 12, 0, tzinfo=dt.timezone.utc)
        lat = np.array([[23.0], [23.0]])
        lon = np.array([[0.0], [180.0]])
        z = mosaic.solar_zenith_grid(lat, lon, when)
        self.assertLess(float(z[0, 0]), 25.0)      # near-overhead at 0E noon
        self.assertGreater(float(z[1, 0]), 120.0)  # deep night on the far side

    def test_shape_and_range(self):
        when = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
        lat, lon = np.meshgrid(np.linspace(-60, 60, 7), np.linspace(0, 359, 9), indexing="ij")
        z = mosaic.solar_zenith_grid(lat, lon, when)
        self.assertEqual(z.shape, lat.shape)
        self.assertTrue(np.all((z >= 0) & (z <= 180)))


class TestViewQuality(unittest.TestCase):
    def test_peaks_at_subsat_fades_at_limb(self):
        lat = np.array([[0.0, 0.0, 0.0]])
        lon = np.array([[140.0, 140.0 + 70.0, 140.0 + 90.0]])  # sub-sat, 70deg, 90deg away
        vq = mosaic._view_quality(lat, lon, 140.0)
        self.assertAlmostEqual(float(vq[0, 0]), 1.0, places=5)   # head-on
        self.assertGreater(float(vq[0, 0]), float(vq[0, 1]))     # fades outward
        self.assertAlmostEqual(float(vq[0, 2]), 0.0, places=5)   # 90deg -> 0
        self.assertTrue(np.all(vq >= 0.0))


class TestScatter(unittest.TestCase):
    def test_bins_by_lon_mod_360(self):
        H, W = mosaic.TARGET_H, mosaic.TARGET_W
        acc = np.zeros((H, W)); cnt = np.zeros((H, W))
        # a point at lon -100 (==260E) and lat 0 -> col ~ 260/360*W, row ~ middle
        gray = np.array([0.5]); lat = np.array([0.0]); lon = np.array([-100.0])
        mosaic._scatter(gray, lat, lon, acc, cnt)
        self.assertEqual(int(cnt.sum()), 1)
        r, c = np.unravel_index(int(np.argmax(cnt)), cnt.shape)
        self.assertAlmostEqual(c / W * 360.0, 260.0, delta=1.0)
        self.assertAlmostEqual(mosaic.LAT_LIM - r / H * 2 * mosaic.LAT_LIM, 0.0, delta=1.0)
        self.assertAlmostEqual(float(acc[r, c]), 0.5)

    def test_drops_nonfinite(self):
        H, W = mosaic.TARGET_H, mosaic.TARGET_W
        acc = np.zeros((H, W)); cnt = np.zeros((H, W))
        gray = np.array([0.5, np.nan, 0.7])
        lat = np.array([0.0, 10.0, np.nan])
        lon = np.array([10.0, 20.0, 30.0])
        mosaic._scatter(gray, lat, lon, acc, cnt)
        self.assertEqual(int(cnt.sum()), 1)   # only the first point is fully finite


class TestGrayField(unittest.TestCase):
    def _meta(self, cmi, units):
        class D:  # minimal FetchResult stand-in
            pass
        d = D(); d.cmi = cmi; d.units = units
        return d

    def test_visible_0_1(self):
        cmi = np.array([[0.0, 0.5, 1.0]])
        g = mosaic._gray_field(self._meta(cmi, "1"), is_visible=True)
        self.assertEqual(g.shape, cmi.shape)
        fin = g[np.isfinite(g)]
        self.assertTrue(np.all((fin >= 0.0) & (fin <= 1.0)))

    def test_swir_grayscale_nan_masked(self):
        cmi = np.array([[273.15 - 60.0, 273.15 + 20.0, np.nan]])  # K
        g = mosaic._gray_field(self._meta(cmi, "K"), is_visible=False)
        self.assertTrue(np.isnan(g[0, 2]))          # masked stays NaN
        fin = g[np.isfinite(g)]
        self.assertTrue(np.all((fin >= 0.0) & (fin <= 1.0)))


class TestConfig(unittest.TestCase):
    def test_three_disks_himawari_swir_only(self):
        self.assertEqual(len(mosaic.MOSAIC_SATS), 3)
        # GOES get vis; Himawari (0.5km AHI vis too big to fetch wide) is SWIR-only.
        flags = [fv for (_s, _b, fv) in mosaic.MOSAIC_SATS]
        self.assertEqual(flags, [True, True, False])

    def test_bounds_pacific_centered(self):
        self.assertEqual(mosaic.LAT_LIM, 65.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
