"""Masked / non-finite geolocation guard at BOTH pcolormesh sites.

Regression for the Time-Machine archive 500 ("x and y arguments to pcolormesh
cannot have non-finite values or be of type numpy.ma.MaskedArray"): early-era
ABI sectors (GOES-16 CONUS from the 89.5W checkout slot, 2017) hand back
MASKED lat/lon inside the sector, and the MAIN scalar render path passed them
to pcolormesh unguarded (the backdrop path already guarded). One shared guard
(_guard_mesh_coords), both call sites.
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import render  # noqa: E402
from satellites import FetchResult  # noqa: E402

BBOX = [-80.0, 20.0, -60.0, 40.0]


def _grid(n=60, masked_corner=False, nan_corner=False):
    """Synthetic BT grid over BBOX; optionally poison a corner's geolocation."""
    lons1 = np.linspace(BBOX[0], BBOX[2], n)
    lats1 = np.linspace(BBOX[3], BBOX[1], n)
    lons, lats = np.meshgrid(lons1, lats1)
    cmi = np.full((n, n), 260.0)
    cmi[20:30, 20:30] = 200.0            # a cold blob so the field has range
    if masked_corner:
        m = np.zeros((n, n), dtype=bool)
        m[:8, -8:] = True                # off-earth block, NE corner
        lons = np.ma.masked_array(lons, mask=m)
        lats = np.ma.masked_array(lats, mask=m)
    if nan_corner:
        lons = lons.copy(); lats = lats.copy()
        lons[-6:, :6] = np.nan
        lats[-6:, :6] = np.nan
    return cmi, lats, lons


def _fetch_result(cmi, lats, lons):
    return FetchResult(
        cmi=cmi, lats=lats, lons=lons, channel=13,
        generic_channel="clean_ir",
        scan_start=dt.datetime(2017, 9, 5, 10, 15, tzinfo=dt.timezone.utc),
        product="ABI-L2-CMIPC", bucket="noaa-goes16", sat_name="GOES-16",
        sub_sat_lon=-89.5, units="K",
    )


class GuardUnitTests(unittest.TestCase):
    def test_passthrough_when_clean(self):
        cmi, lats, lons = _grid()
        f = np.ma.masked_invalid(cmi)
        glons, glats, gfield = render._guard_mesh_coords(lons, lats, f)
        self.assertTrue(np.isfinite(glons).all())
        self.assertTrue(np.isfinite(glats).all())
        np.testing.assert_array_equal(glons, lons)
        # no extra masking introduced
        self.assertEqual(np.ma.getmaskarray(gfield).sum(),
                         np.ma.getmaskarray(f).sum())

    def test_masked_coords_become_finite_and_field_masked(self):
        cmi, lats, lons = _grid(masked_corner=True)
        f = np.ma.masked_invalid(cmi)
        glons, glats, gfield = render._guard_mesh_coords(lons, lats, f)
        # coords: plain float ndarrays, fully finite (pcolormesh-acceptable)
        self.assertNotIsInstance(glons, np.ma.MaskedArray)
        self.assertNotIsInstance(glats, np.ma.MaskedArray)
        self.assertTrue(np.isfinite(glons).all())
        self.assertTrue(np.isfinite(glats).all())
        # the poisoned block is masked in the FIELD (plus the 1-cell erosion)
        gm = np.ma.getmaskarray(gfield)
        self.assertTrue(gm[:8, -8:].all())
        self.assertGreater(gm.sum(), 64)          # erosion widened it
        # good interior cells survive
        self.assertFalse(gm[25, 25])
        # fill coords are in-extent (the mean of the finite coords)
        self.assertTrue((glons.min() >= BBOX[0]) and (glons.max() <= BBOX[2]))

    def test_nan_coords_equivalent_to_masked(self):
        cmi, lats, lons = _grid(nan_corner=True)
        f = np.ma.masked_invalid(cmi)
        glons, glats, gfield = render._guard_mesh_coords(lons, lats, f)
        self.assertTrue(np.isfinite(glons).all())
        self.assertTrue(np.ma.getmaskarray(gfield)[-6:, :6].all())


class RenderEndToEndTests(unittest.TestCase):
    def _render(self, cmi, lats, lons):
        return render.render_png(
            _fetch_result(cmi, lats, lons), BBOX, 13,
            "2017-09-05T10:15Z", "rainbow_ir", dpi=60,
        )

    def test_scalar_render_survives_masked_geolocation(self):
        # THE regression: 2017 GOES-16 CONUS-class masked coords on the MAIN
        # scalar path must render, not raise inside pcolormesh.
        cmi, lats, lons = _grid(masked_corner=True)
        png = self._render(cmi, lats, lons)
        self.assertGreater(len(png), 5000)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_scalar_render_survives_nan_geolocation(self):
        cmi, lats, lons = _grid(nan_corner=True)
        png = self._render(cmi, lats, lons)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_clean_geolocation_still_renders(self):
        cmi, lats, lons = _grid()
        png = self._render(cmi, lats, lons)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
