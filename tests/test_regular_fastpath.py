"""Regular-grid imshow fast path (Time Machine archive speed).

The archive tiers (GridSat-B1 / GridSat-GOES / MergIR) deliver uniform
lat/lon grids that the fetchers meshgrid into 2-D coords; render_png's
pcolormesh then transforms millions of quad vertices per frame. The fast
path detects the separable uniform shape and renders via imshow with a
cell-EDGE extent instead. These tests prove:

  1. the detector accepts exactly the meshgrid-of-uniform-axes shape and
     rejects everything else (curvilinear geos grids, non-uniform axes,
     masked/NaN geolocation) -- the frozen floater/meso path can never
     take the branch;
  2. an end-to-end render of the same field through BOTH paths is visually
     identical (small mean pixel delta; identical output geometry), with
     the colorbar wired (mesh handle) on the fast path.

Run: python -m unittest tests.test_regular_fastpath
"""
from __future__ import annotations

import datetime as dt
import io
import unittest

import numpy as np

import render
from satellites import FetchResult


def _regular_coords(nx=90, ny=60, dlon=0.07, dlat=0.07, lon0=-100.0, lat0=10.0,
                    ascending_lat=True):
    lon1 = lon0 + dlon * np.arange(nx)
    lat1 = lat0 + dlat * np.arange(ny)
    if not ascending_lat:
        lat1 = lat1[::-1]
    lons, lats = np.meshgrid(lon1, lat1)
    return lons.astype(np.float32), lats.astype(np.float32), lon1, lat1


class TestRegularGridDetector(unittest.TestCase):
    def test_accepts_uniform_meshgrid(self):
        lons, lats, lon1, lat1 = _regular_coords()
        out = render._regular_grid_axes(lons, lats)
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out[0], lon1, atol=1e-4)
        np.testing.assert_allclose(out[1], lat1, atol=1e-4)

    def test_accepts_descending_lat(self):
        lons, lats, _, lat1 = _regular_coords(ascending_lat=False)
        out = render._regular_grid_axes(lons, lats)
        self.assertIsNotNone(out)
        self.assertGreater(out[1][0], out[1][-1])

    def test_rejects_curvilinear_geos_like(self):
        # geos grids vary lon BY ROW (and lat by column) by construction
        lons, lats, _, _ = _regular_coords()
        lons = lons + np.linspace(0, 0.5, lons.shape[0])[:, None]
        self.assertIsNone(render._regular_grid_axes(lons, lats))

    def test_rejects_nonuniform_axis(self):
        lons, lats, _, _ = _regular_coords()
        lon1 = np.concatenate([np.arange(0, 3, 0.07), np.arange(3, 6, 0.14)])
        lons2, lats2 = np.meshgrid(lon1, lats[:, 0])
        self.assertIsNone(render._regular_grid_axes(lons2, lats2))

    def test_rejects_masked_or_nan_coords(self):
        lons, lats, _, _ = _regular_coords()
        self.assertIsNone(render._regular_grid_axes(np.ma.masked_array(lons), lats))
        lons_nan = lons.copy()
        lons_nan[0, 0] = np.nan
        self.assertIsNone(render._regular_grid_axes(lons_nan, lats))

    def test_rejects_non_monotonic_lat(self):
        lons, lats, _, _ = _regular_coords()
        lats2 = lats.copy()
        lats2[3, :] = lats2[2, :]   # a repeated row breaks strict monotonicity
        self.assertIsNone(render._regular_grid_axes(lons, lats2))


class TestFastPathRenderParity(unittest.TestCase):
    """Same field through imshow (fast) and pcolormesh (forced): near-identical."""

    def _fetch(self, ascending_lat=True):
        lons, lats, _, _ = _regular_coords(nx=160, ny=110, dlon=0.25, dlat=0.25,
                                           lon0=-100.0, lat0=5.0,
                                           ascending_lat=ascending_lat)
        # a smooth plausible BT field (K) with an off-grid NaN notch to prove
        # masked cells stay transparent-to-background on both paths
        yy, xx = np.mgrid[0:lats.shape[0], 0:lats.shape[1]]
        cmi = (230.0 + 40.0 * np.cos(xx / 17.0) * np.sin(yy / 11.0)).astype(np.float32)
        cmi[:8, :8] = np.nan
        return FetchResult(
            cmi=cmi, lats=lats, lons=lons, channel=13,
            generic_channel="clean_ir",
            scan_start=dt.datetime(2005, 7, 1, 18, tzinfo=dt.timezone.utc),
            product="GRIDSAT-TEST", bucket="test", sat_name="TEST-SAT",
            sub_sat_lon=-75.0, units="K",
        )

    def _render_pair(self, ascending_lat=True):
        fetch = self._fetch(ascending_lat=ascending_lat)
        bbox = [-99.0, 6.0, -62.0, 31.0]
        kw = dict(bbox=bbox, channel=13, time_str="2005-07-01 18:00 UTC",
                  enhancement="rainbow_ir", dpi=70)
        fast = render.render_png(fetch, **kw)
        orig = render._regular_grid_axes
        render._regular_grid_axes = lambda *a: None   # force pcolormesh
        try:
            slow = render.render_png(fetch, **kw)
        finally:
            render._regular_grid_axes = orig
        return fast, slow

    def _decode(self, png):
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)

    def test_paths_visually_identical(self):
        for ascending in (True, False):
            fast, slow = self._render_pair(ascending_lat=ascending)
            a, b = self._decode(fast), self._decode(slow)
            self.assertEqual(a.shape, b.shape)
            mean_delta = float(np.abs(a - b).mean())
            # sub-1/255 mean delta = the two rasterizations differ only in
            # cell-edge antialiasing, not placement or color mapping
            self.assertLess(
                mean_delta, 1.0,
                f"fast/slow render diverged (mean|Δ|={mean_delta:.3f}, "
                f"ascending_lat={ascending})")

    def test_fast_path_taken_and_colorbar_wired(self):
        fetch = self._fetch()
        calls = {"imshow": 0}
        orig = render._regular_grid_axes

        def spy(lons, lats):
            out = orig(lons, lats)
            if out is not None:
                calls["imshow"] += 1
            return out

        render._regular_grid_axes = spy
        try:
            png = render.render_png(
                fetch, bbox=[-99.0, 6.0, -62.0, 31.0], channel=13,
                time_str="2005-07-01 18:00 UTC", enhancement="rainbow_ir",
                dpi=70)
        finally:
            render._regular_grid_axes = orig
        self.assertEqual(calls["imshow"], 1, "fast path not taken for a regular grid")
        img = self._decode(png)
        # colorbar strip on the right must be populated (mesh handle wired):
        # a missing colorbar leaves the strip at the dark background
        strip = img[img.shape[0] // 4: -img.shape[0] // 4,
                    int(img.shape[1] * 0.90): int(img.shape[1] * 0.93)]
        self.assertGreater(float(strip.std()), 10.0, "colorbar missing on fast path")


if __name__ == "__main__":
    unittest.main()
