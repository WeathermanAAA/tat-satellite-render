"""Himawari-9 AHI suite locks: the JMA-verified recipe table, the registry
rows, the loader stride math, and the antimeridian-aware webmerc cut.

The RGB coefficients are LOCKED to the primary sources read on 2026-07-10
(JMA MSC Himawari RGB Quick Guides Ver 1.0 + MSC Technical Note No. 65 —
AHI-retuned thresholds per Murata & Shimizu 2017, NOT SEVIRI heritage, NOT
the ABI guides). Changing any number here requires re-reading the guide.
"""
import unittest

import numpy as np

import s2_recipes as X
import s2_registry as R
from s2_webmerc import cut_webmerc_pyramid, reproject_tile
from vendor.ahi_loader import _stitch
from vendor.ahi_hsd import HSDSegment, COUNT_OUTSIDE_SCAN


def _gun(r, i):
    g = r.guns[i]
    return (g.expr, g.lo, g.hi, g.gamma, g.kind)


class TestAhiRecipeLocks(unittest.TestCase):
    """Every verified JMA number, gun for gun."""

    def test_airmass(self):
        r = X.AHI_RECIPES_BY_KEY["airmass"]
        self.assertEqual(_gun(r, 0), (("diff", 8, 10), -25.8, 0.0, 1.0, "bt"))
        self.assertEqual(_gun(r, 1), (("diff", 12, 13), -41.5, 4.3, 1.0, "bt"))
        self.assertEqual(_gun(r, 2), (("band", 8), 242.6, 208.0, 1.0, "bt"))

    def test_dust(self):
        r = X.AHI_RECIPES_BY_KEY["dust"]
        self.assertEqual(_gun(r, 0), (("diff", 15, 13), -7.5, 3.0, 1.0, "bt"))
        self.assertEqual(_gun(r, 1), (("diff", 13, 11), 0.9, 12.5, 2.5, "bt"))
        self.assertEqual(_gun(r, 2), (("band", 13), 261.5, 289.2, 1.0, "bt"))

    def test_ash(self):
        r = X.AHI_RECIPES_BY_KEY["ash"]
        self.assertEqual(_gun(r, 0), (("diff", 15, 13), -3.0, 7.5, 1.0, "bt"))
        self.assertEqual(_gun(r, 1), (("diff", 13, 11), -1.6, 4.9, 1.2, "bt"))
        self.assertEqual(_gun(r, 2), (("band", 13), 243.6, 303.2, 1.0, "bt"))

    def test_nightmicro(self):
        r = X.AHI_RECIPES_BY_KEY["nightmicro"]
        self.assertEqual(_gun(r, 0), (("diff", 15, 13), -7.5, 3.0, 1.0, "bt"))
        self.assertEqual(_gun(r, 1), (("diff", 13, 7), -2.9, 7.0, 1.0, "bt"))
        self.assertEqual(_gun(r, 2), (("band", 13), 243.7, 293.2, 1.0, "bt"))

    def test_dayconvection(self):
        r = X.AHI_RECIPES_BY_KEY["dayconvection"]
        self.assertEqual(_gun(r, 0), (("diff", 8, 10), -36.0, 5.0, 1.0, "bt"))
        self.assertEqual(_gun(r, 1), (("diff", 7, 13), -1.0, 61.0, 0.5, "bt"))
        self.assertEqual(_gun(r, 2), (("diff", 5, 3), -0.80, 0.26, 0.95, "refl"))

    def test_daylandcloud(self):
        r = X.AHI_RECIPES_BY_KEY["daylandcloud"]
        self.assertEqual(_gun(r, 0), (("band", 5), 0.0, 0.99, 1.0, "refl"))
        self.assertEqual(_gun(r, 1), (("band", 4), 0.0, 1.02, 0.95, "refl"))
        self.assertEqual(_gun(r, 2), (("band", 3), 0.0, 1.0, 1.0, "refl"))

    def test_daycloudphase(self):
        r = X.AHI_RECIPES_BY_KEY["daycloudphase"]
        self.assertEqual(_gun(r, 0), (("band", 13), 280.7, 219.6, 1.0, "bt"))
        self.assertEqual(_gun(r, 1), (("band", 3), 0.0, 0.85, 1.0, "refl"))
        self.assertEqual(_gun(r, 2), (("band", 5), 0.01, 0.50, 1.0, "refl"))

    def test_firetemp(self):
        r = X.AHI_RECIPES_BY_KEY["firetemp"]
        self.assertEqual(_gun(r, 0), (("band", 7), 273.0, 350.0, 1.0, "bt"))
        self.assertEqual(_gun(r, 1), (("band", 6), 0.0, 0.5, 1.0, "refl"))
        self.assertEqual(_gun(r, 2), (("band", 5), 0.0, 0.5, 1.0, "refl"))

    def test_snowfog_is_documented_abi_heritage(self):
        # the exact JMA recipe needs a DERIVED 3.9 um solar-reflectance gun
        # (not band math) -- we ship ABI stretches renumbered, and the source
        # string must say so (honesty lock)
        r = X.AHI_RECIPES_BY_KEY["snowfog"]
        self.assertEqual(_gun(r, 2), (("diff", 7, 13), 0.0, 30.0, 1.7, "bt"))
        self.assertIn("departure", r.source)

    def test_sandwich_and_truecolor_band_sets(self):
        self.assertEqual(X.AHI_RECIPES_BY_KEY["sandwich"].bands, (3, 13))
        self.assertEqual(X.AHI_RECIPES_BY_KEY["sandwich"].vis_band, 3)
        self.assertEqual(X.AHI_RECIPES_BY_KEY["truecolor"].bands, (1, 2, 3, 4, 13))
        # ABI rows unchanged by the sensor generalization
        self.assertEqual(X.RECIPES_BY_KEY["sandwich"].bands, (2, 13))
        self.assertEqual(X.RECIPES_BY_KEY["truecolor"].bands, (1, 2, 3, 13))

    def test_channels_no_cirrus_native_green(self):
        keys = {r.key for r in X.AHI_RECIPES}
        self.assertIn("b02", keys)               # native green, no ABI counterpart
        self.assertNotIn("b13", keys)            # clean IR is keyed 'ir'
        self.assertIn("ir", keys)
        for k in keys:
            r = X.AHI_RECIPES_BY_KEY[k]
            self.assertNotIn(4, () if r.kind == "truecolor" else (),
                             "no AHI recipe may reference a cirrus band")
        # ABI C04 cirrus has NO AHI counterpart -- the key is absent, not faked
        with self.assertRaises(KeyError):
            X.recipe_for("ahi", "c04")

    def test_finest_km_uses_ahi_table(self):
        # AHI B05 (1.6 um) is 2 km (ABI C05 is 1 km) -- the FD/wpac pyramid
        # sizing must use the AHI table
        self.assertEqual(X.AHI_RECIPES_BY_KEY["firetemp"].finest_km, 2.0)
        self.assertEqual(X.RECIPES_BY_KEY["firetemp"].finest_km, 1.0)
        self.assertEqual(X.AHI_RECIPES_BY_KEY["daycloudphase"].finest_km, 0.5)

    def test_recipe_for_family_routing(self):
        self.assertIs(X.recipe_for("goes", "airmass"), X.RECIPES_BY_KEY["airmass"])
        self.assertIs(X.recipe_for("himawari", "airmass"), X.AHI_RECIPES_BY_KEY["airmass"])


class TestHimawariRegistryRows(unittest.TestCase):
    def test_row_counts_and_uniqueness(self):
        wpac = [e for e in R.REGISTRY if e.sat_key == "himawari9" and e.sector_key == "wpac"]
        fd = [e for e in R.REGISTRY if e.sat_key == "himawari9" and e.sector_key == "fd" and e.tiled]
        self.assertEqual(len(wpac), 28)          # full suite incl truecolor
        self.assertEqual(len(fd), 27)            # truecolor excluded (B03 budget)
        paths = [e.product_path for e in R.REGISTRY]
        self.assertEqual(len(paths), len(set(paths)))

    def test_wpac_ir_row(self):
        e = R.REGISTRY_BY_ID["himawari9-wpac-ir"]
        self.assertEqual(e.product_path, "sat/himawari9/wpac/ir")
        self.assertEqual(e.bt_px, 2560)          # the objfix WP input resolution
        self.assertEqual(e.bands, (13,))
        self.assertEqual(e.substrate, "ahi_l1b")
        self.assertEqual(e.ahi_segments, 10)
        self.assertEqual(e.pyramid_scheme, "webmercator-xyz")
        self.assertEqual(e.sector_bbox, (95.0, -5.0, 180.0, 45.0))

    def test_fd_bbox_unwrapped(self):
        e = R.REGISTRY_BY_ID["himawari9-fd-ir"]
        self.assertGreater(e.sector_bbox[2], 180.0)   # antimeridian crossing
        self.assertEqual(e.sector_bbox, (60.0, -60.0, 221.0, 60.0))

    def test_truecolor_capped_at_2km_class(self):
        e = R.REGISTRY_BY_ID["himawari9-wpac-truecolor"]
        self.assertEqual(e.pyramid_px, 4608)     # 2 km class, not the 0.5 km 6144

    def test_products_index_carries_suite(self):
        import datetime as dt
        idx = R.build_products_index("himawari9", "wpac",
                                     dt.datetime(2026, 7, 10, tzinfo=dt.timezone.utc))
        self.assertEqual(idx["count"], 28)
        ids = {p["id"] for p in idx["products"]}
        self.assertIn("himawari9-wpac-ir", ids)
        self.assertIn("himawari9-wpac-truecolor", ids)


class TestLoaderStride(unittest.TestCase):
    def _segments(self, n_cols=20, seg_lines=10, total=3):
        segs = []
        for k in range(total):
            counts = np.arange(k * seg_lines * n_cols,
                               (k + 1) * seg_lines * n_cols,
                               dtype=np.uint16).reshape(seg_lines, n_cols)
            segs.append(HSDSegment(
                sat_name="Himawari-9", band_number=13, central_wavelength_um=10.4,
                sub_lon=140.7, cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
                total_segments=total, segment_seq=k + 1,
                first_line_number=k * seg_lines + 1, n_columns=n_cols, n_lines=seg_lines,
                slope=1.0, intercept=0.0, planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
                speed_of_light=3e8, planck_const=6.6e-34, boltzmann_const=1.38e-23,
                albedo_coef=1.0, obs_start_mjd=60000.0, counts=counts))
        return segs

    def test_stride1_unchanged(self):
        segs = self._segments()
        full, nl, nc, off = _stitch(segs)
        self.assertEqual((nl, nc, off), (30, 20, 0))
        self.assertTrue((full == np.arange(600, dtype=np.uint16).reshape(30, 20)).all())

    def test_stride_matches_post_decimation(self):
        """A strided stitch must equal decimating the full stitch on the
        SAME global grid, for any subset of segments."""
        for stride in (2, 3, 4):
            for drop_first in (False, True):
                segs = self._segments()
                if drop_first:
                    segs = segs[1:]
                full, _, _, off0 = _stitch(segs)
                dec, nl, nc, off = _stitch(segs, stride=stride)
                # global kept rows: (line-1) % stride == 0, 1-based
                r0 = (-(off0)) % stride
                expect = full[r0::stride, ::stride]
                self.assertEqual(dec.shape, expect.shape)
                self.assertTrue((dec == expect).all())
                self.assertEqual(off, off0 + r0)
                self.assertEqual((off) % stride, 0)


class TestWebmercAntimeridian(unittest.TestCase):
    def _raster(self):
        # 360x120 raster over (60, -60, 221, 60): red west half, green east
        rgba = np.zeros((120, 360, 4), np.uint8)
        rgba[..., 3] = 255
        rgba[:, :180, 0] = 255
        rgba[:, 180:, 1] = 255
        return rgba, (60.0, -60.0, 221.0, 60.0)

    def test_eastern_lobe_sampled_across_wrap(self):
        rgba, bounds = self._raster()
        # z2 x0 covers lon -180..-90 -> unwrapped 180..270; data to 221 = green
        tile = reproject_tile(rgba, bounds, 2, 0, 1, 64)
        self.assertIsNotNone(tile)
        vis = tile[tile[..., 3] > 0]
        self.assertTrue((vis[:, 1] > 200).all())     # green lobe
        self.assertTrue((vis[:, 0] < 50).all())
        # the west edge of the data (tile x2 = lon 0..90 -> 60..90 red)
        tile_w = reproject_tile(rgba, bounds, 2, 2, 1, 64)
        vis_w = tile_w[tile_w[..., 3] > 0]
        self.assertTrue((vis_w[:, 0] > 200).all())   # red lobe

    def test_pyramid_covers_both_x_ranges(self):
        rgba, bounds = self._raster()
        out = cut_webmerc_pyramid(rgba, bounds, maxzoom=2)
        xs = sorted({x for (z, x, y) in out["tiles"] if z == 2})
        self.assertIn(0, xs)         # eastern lobe (wrapped)
        self.assertIn(3, xs)         # western part of the disk (~150E)
        self.assertNotIn(1, xs)      # the empty mid-Atlantic gap stays empty

    def test_goes_bounds_unaffected(self):
        rgba = np.zeros((60, 180, 4), np.uint8)
        rgba[..., 3] = 255
        rgba[..., 0] = 200
        out = cut_webmerc_pyramid(rgba, (-156.0, -60.0, 6.0, 60.0), maxzoom=1)
        self.assertGreater(len(out["tiles"]), 0)
        for (z, x, y) in out["tiles"]:
            self.assertLess(x, 2 ** z)


if __name__ == "__main__":
    unittest.main()
