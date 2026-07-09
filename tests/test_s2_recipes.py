#!/usr/bin/env python3
"""Tests for the Phase-3 recipe engine (s2_recipes) + the registry suite rows +
the emit dispatch. Synthetic arrays only -- no network, no matplotlib figures.
The RGB scalings themselves are the CIRA/RAMMB quick-guide values, verified
against the primary PDFs (see each Recipe.source)."""
import datetime as dt
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_recipes as X  # noqa: E402
import s2_registry as R  # noqa: E402

UTC = dt.timezone.utc


class TestGunMath(unittest.TestCase):
    def test_scale_linear(self):
        g = X.Gun(("band", 13), 0.0, 10.0)
        x = np.array([-5.0, 0.0, 5.0, 10.0, 20.0])
        np.testing.assert_allclose(X.scale_gun(x, g), [0, 0, 0.5, 1, 1])

    def test_scale_inverted(self):
        # lo > hi = inverted gun (Air Mass blue: 243.9 -> 208.5 K)
        g = X.Gun(("band", 8), 243.9, 208.5)
        x = np.array([243.9, 208.5, 226.2])
        v = X.scale_gun(x, g)
        self.assertAlmostEqual(v[0], 0.0)
        self.assertAlmostEqual(v[1], 1.0)
        self.assertAlmostEqual(v[2], 0.5, places=2)

    def test_scale_gamma(self):
        # Dust green gamma 2.5: mid-range rises (x^(1/2.5))
        g = X.Gun(("diff", 14, 11), 0.0, 1.0, 2.5)
        v = X.scale_gun(np.array([0.25]), g)
        self.assertAlmostEqual(float(v[0]), 0.25 ** (1 / 2.5), places=6)

    def test_nan_propagates(self):
        g = X.Gun(("band", 13), 0.0, 10.0)
        v = X.scale_gun(np.array([np.nan, 5.0]), g)
        self.assertTrue(np.isnan(v[0]))
        self.assertAlmostEqual(v[1], 0.5)

    def test_gun_input_diff(self):
        bands = {15: np.full((2, 2), 250.0), 13: np.full((2, 2), 245.0)}
        g = X.Gun(("diff", 15, 13), -6.7, 2.6)
        np.testing.assert_allclose(X.gun_input(g, bands), 5.0)


class TestRecipeTable(unittest.TestCase):
    def test_headline_products_present(self):
        for key in ("truecolor", "sandwich", "airmass", "dust", "firetemp",
                    "daycloudphase", "nightmicro", "c08"):
            self.assertIn(key, X.RECIPES_BY_KEY)

    def test_sixteen_channels_covered(self):
        # c01..c16: c13 is the existing goes19-conus-ir row; irbd adds Dvorak BD.
        keys = {r.key for r in X.RECIPES}
        for n in range(1, 17):
            if n == 13:
                self.assertIn("irbd", keys)
            else:
                self.assertIn(f"c{n:02d}", keys)

    def test_bands_and_kinds_consistent(self):
        for r in X.RECIPES:
            self.assertIn(r.group, ("composite", "rgb", "channel"))
            self.assertTrue(r.bands, f"{r.key} needs at least one band")
            for b in r.bands:
                self.assertIn(b, X.BAND_NATIVE_KM)
            if r.kind == "rgb_guns":
                self.assertEqual(len(r.guns), 3)
                for g in r.guns:
                    self.assertIn(g.expr[0], ("band", "diff"))
                    self.assertIn(g.kind, ("bt", "refl"))
                    # refl guns must reference reflective bands; bt guns emissive
                    for b in g.bands:
                        if g.kind == "refl":
                            self.assertNotIn(b, X.EMISSIVE_BANDS,
                                             f"{r.key}: refl gun on emissive C{b:02d}")
                        else:
                            self.assertIn(b, X.EMISSIVE_BANDS,
                                          f"{r.key}: bt gun on reflective C{b:02d}")
            if r.kind == "single_palette":
                self.assertTrue(r.band and r.enhancement)
            if r.bt_band:
                self.assertIn(r.bt_band, X.EMISSIVE_BANDS)

    def test_quickguide_airmass_numbers_locked(self):
        # Lock the verified CIRA quick-guide scalings against silent edits.
        r = X.RECIPES_BY_KEY["airmass"]
        self.assertEqual([g.expr for g in r.guns],
                         [("diff", 8, 10), ("diff", 12, 13), ("band", 8)])
        self.assertEqual([(g.lo, g.hi, g.gamma) for g in r.guns],
                         [(-26.2, 0.6, 1.0), (-43.2, 6.7, 1.0), (243.9, 208.5, 1.0)])

    def test_quickguide_nightmicro_numbers_locked(self):
        r = X.RECIPES_BY_KEY["nightmicro"]
        self.assertEqual([(g.lo, g.hi) for g in r.guns],
                         [(-6.7, 2.6), (-3.1, 5.2), (243.55, 292.65)])

    def test_quickguide_ash_and_dayconvection_locked(self):
        # ABI-tuned Ash values (NOT the SEVIRI heritage -4..2/-4..5/243..303)
        a = X.RECIPES_BY_KEY["ash"]
        self.assertEqual([(g.lo, g.hi, g.gamma) for g in a.guns],
                         [(-6.7, 2.6, 1.0), (-6.0, 6.3, 1.0), (243.6, 302.4, 1.0)])
        # Day Convection: green gamma is 1 per the ABI quick guide (EUMETSAT
        # heritage 0.5 deliberately NOT used); blue is reflectance FRACTION.
        d = X.RECIPES_BY_KEY["dayconvection"]
        self.assertEqual([g.expr for g in d.guns],
                         [("diff", 8, 10), ("diff", 7, 13), ("diff", 5, 2)])
        self.assertEqual([(g.lo, g.hi, g.gamma) for g in d.guns],
                         [(-35.0, 5.0, 1.0), (-5.0, 60.0, 1.0), (-0.75, 0.25, 1.0)])
        self.assertEqual(d.guns[2].kind, "refl")
        # Day Land Cloud = EUMETSAT Natural Color, ABI-stretched
        n = X.RECIPES_BY_KEY["daylandcloud"]
        self.assertEqual([(g.expr, g.lo, g.hi) for g in n.guns],
                         [(("band", 5), 0.0, 0.975), (("band", 3), 0.0, 1.086),
                          (("band", 2), 0.0, 1.0)])

    def test_quickguide_snowfog_locked(self):
        # Day Snow-Fog: R C03 0-100% g1.7, G C05 0-70% g1.7, B = C07-C13
        # BT difference 0-30 K g1.7 (all three guns gamma 1.7 per the guide).
        s = X.RECIPES_BY_KEY["snowfog"]
        self.assertEqual([g.expr for g in s.guns],
                         [("band", 3), ("band", 5), ("diff", 7, 13)])
        self.assertEqual([(g.lo, g.hi, g.gamma) for g in s.guns],
                         [(0.0, 1.0, 1.7), (0.0, 0.7, 1.7), (0.0, 30.0, 1.7)])
        self.assertEqual([g.kind for g in s.guns], ["refl", "refl", "bt"])
        self.assertTrue(s.day_only)
        self.assertEqual(s.bands, (3, 5, 7, 13))

    def test_ir_rgbs_carry_c13_bt(self):
        for key in ("airmass", "dust", "ash", "nightmicro", "daycloudphase"):
            self.assertEqual(X.RECIPES_BY_KEY[key].bt_band, 13, key)


class TestEngine(unittest.TestCase):
    def test_compute_rgb_shape_and_alpha(self):
        r = X.RECIPES_BY_KEY["nightmicro"]
        shape = (4, 5)
        bands = {7: np.full(shape, 260.0), 13: np.full(shape, 263.0),
                 15: np.full(shape, 261.0)}
        bands[15][0, 0] = np.nan     # one off-disk pixel
        rgb = X.compute_rgb(r, bands)
        self.assertEqual(rgb.shape, shape + (3,))
        rgba = X.rgba_from_rgb(rgb)
        self.assertEqual(rgba.dtype, np.uint8)
        self.assertEqual(rgba[0, 0, 3], 0)      # NaN input -> transparent
        self.assertEqual(rgba[1, 1, 3], 255)

    def test_dust_signature_direction(self):
        # A dusty pixel (positive 12.3-10.3 split, small 11.2-8.4, warm surface)
        # must come out red-heavy (magenta/pink family), per the quick guide.
        r = X.RECIPES_BY_KEY["dust"]
        dusty = {15: np.array([[302.0]]), 13: np.array([[300.0]]),
                 14: np.array([[301.0]]), 11: np.array([[299.0]])}
        rgb = X.compute_rgb(r, dusty)[0, 0]
        self.assertGreater(rgb[0], 0.85)         # R: +2 K diff near top of range
        self.assertGreater(rgb[2], 0.9)          # B: 300 K warm -> bright
        # thick ICE cloud: R near floor (negative split), B dark (cold)
        icy = {15: np.array([[215.0]]), 13: np.array([[220.0]]),
               14: np.array([[218.0]]), 11: np.array([[219.0]])}
        rgb2 = X.compute_rgb(r, icy)[0, 0]
        self.assertLess(rgb2[0], 0.2)
        self.assertLess(rgb2[2], 0.05)

    def test_reflective_channel_gray(self):
        r = X.RECIPES_BY_KEY["c02"]
        bands = {2: np.array([[0.25, 1.0]])}
        rgb = X.compute_rgb(r, bands)
        # 3 identical guns -> gray; gamma 2 -> sqrt stretch
        np.testing.assert_allclose(rgb[0, 0], [0.5] * 3, atol=1e-6)
        np.testing.assert_allclose(rgb[0, 1], [1.0] * 3, atol=1e-6)

    def test_sandwich_luma_modulation(self):
        ir_rgb = np.full((1, 2, 3), 0.8, np.float32)
        vis = np.array([[0.0, 1.0]], np.float32)
        out = X.sandwich_rgb(vis, ir_rgb)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.8 * X.SANDWICH_LUMA_FLOOR, places=5)
        self.assertAlmostEqual(float(out[0, 1, 0]), 0.8, places=5)


class TestRegistrySuite(unittest.TestCase):
    def test_rows_generated_for_every_recipe(self):
        for r in X.RECIPES:
            e = R.REGISTRY_BY_ID.get(f"goes19-conus-{r.key}")
            self.assertIsNotNone(e, f"no registry row for recipe {r.key}")
            self.assertTrue(e.tiled)
            self.assertEqual(e.recipe_id, r.key)
            self.assertEqual(e.bands, tuple(r.bands))
            self.assertEqual(e.pyramid_scheme, "webmercator-xyz")
            self.assertEqual(e.sector_bbox, R._CONUS_BBOX)

    def test_existing_ir_row_untouched(self):
        e = R.REGISTRY_BY_ID["goes19-conus-ir"]
        self.assertIsNone(e.recipe_id)
        self.assertEqual(e.pyramid_px, 6144)
        self.assertEqual(e.render_enhancement, "rainbow_ir")

    def test_claims_routes_suite_bands(self):
        # a C15 CONUS object is claimed by dust + nightmicro + c15 (at least)
        slot = R.parse_key("OR_ABI-L2-CMIPC-M6C15_G19_s20261891801171_e0_c0.nc")
        ids = {e.product_id for e in R.matching_entries(slot)}
        for want in ("goes19-conus-dust", "goes19-conus-nightmicro", "goes19-conus-c15"):
            self.assertIn(want, ids)

    def test_fd_rows_generated(self):
        # every recipe except truecolor gets an fd row (truecolor excluded: its
        # product_path would collide with the Phase-1 goes19-fd-mcmip row)
        for r in X.RECIPES:
            e = R.REGISTRY_BY_ID.get(f"goes19-fd-{r.key}")
            if r.key == "truecolor":
                self.assertIsNone(e)
                continue
            self.assertIsNotNone(e, f"no fd row for {r.key}")
            self.assertEqual(e.render_product_hint, "fd")
            self.assertEqual(e.s3_prefix, "ABI-L2-CMIPF/")
            self.assertEqual(e.sector_bbox, R._FD_BBOX)
        # no product_path collisions anywhere in the registry among TILED rows
        # (the fd-mcmip placeholder is tiled=False and may alias a future path)
        paths = [e.product_path for e in R.REGISTRY if e.tiled]
        self.assertEqual(len(paths), len(set(paths)))

    def test_products_index(self):
        idx = R.build_products_index("goes19", "conus", dt.datetime(2026, 7, 8, tzinfo=UTC))
        self.assertEqual(idx["count"], len(idx["products"]))
        ids = [p["id"] for p in idx["products"]]
        self.assertIn("goes19-conus-ir", ids)
        self.assertIn("goes19-conus-truecolor", ids)
        by_id = {p["id"]: p for p in idx["products"]}
        self.assertTrue(by_id["goes19-conus-ir"]["bt"])
        self.assertFalse(by_id["goes19-conus-truecolor"]["bt"])
        self.assertTrue(by_id["goes19-conus-firetemp"]["day_only"])
        self.assertEqual(R.products_index_key("shadow", "goes19", "conus"),
                         "shadow/sat/goes19/conus/products.json")


class TestImageryHelpers(unittest.TestCase):
    def test_decimate_bt_exact_values(self):
        import s2_imagery as I
        g = np.arange(100, dtype=np.float32).reshape(10, 10)
        bt, (w, h) = I._decimate_bt(g, (-100.0, 20.0, -90.0, 30.0), out_w=5)
        self.assertEqual((w, h), (5, 5))
        self.assertIn(float(bt[0, 0]), set(g.ravel().tolist()))  # exact, no interp

    def test_colorize_bt_nan_transparent(self):
        import s2_imagery as I
        bt_k = np.array([[220.0, np.nan]])
        rgb = I._colorize_bt(bt_k, "rainbow_ir")
        self.assertTrue(np.isfinite(rgb[0, 0]).all())
        self.assertTrue(np.isnan(rgb[0, 1]).all())
        rgba = X.rgba_from_rgb(rgb)
        self.assertEqual(rgba[0, 1, 3], 0)


if __name__ == "__main__":
    unittest.main()
