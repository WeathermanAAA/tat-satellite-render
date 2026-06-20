"""Custom-zoom resolution tiers + map-overlay toggles (the draw-a-box page).

Covers the app-layer plumbing: resolution tier -> (pixel budget, output format),
the budget-parameterised downsample, RenderRequest validation/defaults, and the
cache-key behaviour (tiers cache separately; the webp LOOP path is untouched;
overlay toggles key only when turned OFF, preserving default-on cache continuity).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app


BBOX = [-80.0, 20.0, -60.0, 40.0]   # 20° x 20°


class ResolveQualityTests(unittest.TestCase):
    def test_png_tiers_map_to_budget_and_format(self):
        self.assertEqual(app.resolve_quality("png", "high"),
                         (app.PIXEL_BUDGET, "png"))
        self.assertEqual(app.resolve_quality("png", "default"),
                         (app.QUALITY_BUDGETS["default"], "png"))
        # low re-encodes to lossy WebP for the smallest download
        self.assertEqual(app.resolve_quality("png", "low"),
                         (app.QUALITY_BUDGETS["low"], "webp"))

    def test_unknown_tier_falls_back_to_default(self):
        self.assertEqual(app.resolve_quality("png", "ultra"),
                         (app.QUALITY_BUDGETS["default"], "png"))

    def test_webp_loop_path_ignores_quality(self):
        # The poller loop path keeps the full budget + a webp frame regardless
        # of any quality value, so loop frames never change.
        for q in ("low", "default", "high", "junk"):
            self.assertEqual(app.resolve_quality("webp", q),
                             (app.PIXEL_BUDGET, "webp"))

    def test_budgets_are_ordered_low_lt_default_lt_high(self):
        self.assertLess(app.QUALITY_BUDGETS["low"], app.QUALITY_BUDGETS["default"])
        self.assertLess(app.QUALITY_BUDGETS["default"], app.QUALITY_BUDGETS["high"])


class DownsampleBudgetTests(unittest.TestCase):
    def test_smaller_budget_downsamples_more(self):
        n_high = app.compute_downsample_factor(BBOX, "clean_ir",
                                               app.QUALITY_BUDGETS["high"])
        n_def = app.compute_downsample_factor(BBOX, "clean_ir",
                                              app.QUALITY_BUDGETS["default"])
        n_low = app.compute_downsample_factor(BBOX, "clean_ir",
                                              app.QUALITY_BUDGETS["low"])
        self.assertLessEqual(n_high, n_def)
        self.assertLess(n_def, n_low)            # low always strides hardest here

    def test_default_budget_matches_pixel_budget(self):
        # back-compat: no budget arg == the full PIXEL_BUDGET path
        self.assertEqual(
            app.compute_downsample_factor(BBOX, "clean_ir"),
            app.compute_downsample_factor(BBOX, "clean_ir", app.PIXEL_BUDGET),
        )

    def test_low_tier_caps_output_area_near_500_squared(self):
        # The budget caps output AREA (~500x500 = 0.25M px); a non-square crop
        # keeps that area, so its geomean side ~= 500 (width/height vary by aspect).
        big = [-100.0, 0.0, -40.0, 40.0]   # 60x40 deg, 0.5 km visible
        n = app.compute_downsample_factor(big, "visible_red", app.QUALITY_BUDGETS["low"])
        km = app._native_km_per_pixel("visible_red")
        out_w = (big[2] - big[0]) * app.DEG_TO_KM / km / n
        out_h = (big[3] - big[1]) * app.DEG_TO_KM / km / n
        self.assertLessEqual(out_w * out_h, app.QUALITY_BUDGETS["low"])   # within budget
        self.assertLess(abs((out_w * out_h) ** 0.5 - 500), 120)          # geomean ~500


class RenderRequestTests(unittest.TestCase):
    def test_defaults_preserve_existing_look(self):
        r = app.RenderRequest(bbox=BBOX, channel="clean_ir")
        self.assertEqual(r.quality, "default")
        self.assertTrue(r.coastlines)
        self.assertTrue(r.gridlines)
        self.assertEqual(r.format, "png")

    def test_quality_is_coerced(self):
        self.assertEqual(app.RenderRequest(bbox=BBOX, channel="clean_ir",
                                           quality="HIGH").quality, "high")
        self.assertEqual(app.RenderRequest(bbox=BBOX, channel="clean_ir",
                                           quality="nonsense").quality, "default")

    def test_overlay_toggles_parse(self):
        r = app.RenderRequest(bbox=BBOX, channel="clean_ir",
                              coastlines=False, gridlines=False)
        self.assertFalse(r.coastlines)
        self.assertFalse(r.gridlines)


class CacheKeyTests(unittest.TestCase):
    def _key(self, **kw):
        body = app.RenderRequest(bbox=BBOX, channel="clean_ir", **kw)
        return app._request_key(body, "clean_ir", "2026-06-20T12:00:00", "noaa-goes19",
                                body.quality)

    def test_tiers_cache_separately(self):
        keys = {self._key(quality=q) for q in ("low", "default", "high")}
        self.assertEqual(len(keys), 3)

    def test_overlay_off_changes_key_but_default_on_does_not(self):
        base = self._key()                                   # coast+grid on (default)
        self.assertNotEqual(base, self._key(gridlines=False))
        self.assertNotEqual(base, self._key(coastlines=False))
        self.assertNotEqual(self._key(gridlines=False), self._key(coastlines=False))

    def test_webp_loop_key_unaffected_by_quality(self):
        # The poller loop path (format=webp) must key identically regardless of
        # quality, so the frame cache stays continuous across this deploy.
        def wkey(q):
            body = app.RenderRequest(bbox=BBOX, channel="clean_ir", format="webp", quality=q)
            return app._request_key(body, "clean_ir", "2026-06-20T12:00:00",
                                    "noaa-goes19", body.quality)
        self.assertEqual(wkey("low"), wkey("high"))
        self.assertEqual(wkey("default"), wkey("low"))


if __name__ == "__main__":
    unittest.main()
