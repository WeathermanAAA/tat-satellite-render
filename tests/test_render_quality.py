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

    def test_backdrop_field_default_false_and_parses(self):
        # PART 4: additive opt-in -> every existing caller omits it -> False.
        self.assertFalse(app.RenderRequest(bbox=BBOX, channel="clean_ir").backdrop)
        self.assertTrue(app.RenderRequest(bbox=BBOX, channel="clean_ir",
                                          backdrop=True).backdrop)


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

    def test_backdrop_keys_separately_from_chromed(self):
        # PART 4: the bare grayscale backdrop is a DISTINCT artifact at the same
        # bbox/channel -> its own cache slot, never clobbering the chromed render.
        self.assertNotEqual(self._key(), self._key(backdrop=True))


class TierDpiTests(unittest.TestCase):
    """The tier knob is the OUTPUT DPI (not a whole-figure bitmap resize)."""

    def test_tier_dpi_ordered_and_default_unchanged(self):
        self.assertLess(app.TIER_DPI["low"], app.TIER_DPI["default"])
        self.assertLess(app.TIER_DPI["default"], app.TIER_DPI["high"])
        # default stays 110 dpi == today (the byte-identical contract).
        self.assertEqual(app.TIER_DPI["default"], 110)
        self.assertEqual(app.DEFAULT_DPI, 110)

    def test_loop_path_renders_at_default_dpi(self):
        # The webp LOOP path (pollers) must render at 110 regardless of quality,
        # so loop frames are unaffected by the tier dpi. Exercise the REAL
        # selection expression the endpoint uses (pick_tier_dpi).
        for q in ("low", "default", "high", "junk"):
            self.assertEqual(app.pick_tier_dpi("webp", q), 110)
        # the custom-zoom png path DOES honor the tier dpi
        self.assertEqual(app.pick_tier_dpi("png", "low"), app.TIER_DPI["low"])
        self.assertEqual(app.pick_tier_dpi("png", "high"), app.TIER_DPI["high"])
        self.assertEqual(app.pick_tier_dpi("png", "junk"), app.DEFAULT_DPI)


class EncodeWebpNoResizeTests(unittest.TestCase):
    """The 'low' tier re-encodes its NATIVE render as WebP -- no whole-figure
    bitmap resize (that is what used to pixelate the chrome)."""

    def test_encode_webp_preserves_dimensions(self):
        import io
        from PIL import Image
        import render
        src = Image.new("RGB", (837, 611), (10, 20, 30))
        buf = io.BytesIO(); src.save(buf, "PNG")
        out = render.encode_webp(buf.getvalue(), app.LOWRES_WEBP_QUALITY)
        im = Image.open(io.BytesIO(out))
        self.assertEqual(im.format, "WEBP")
        self.assertEqual(im.size, (837, 611))   # NOT downscaled to 500


def _synthetic_ir() -> "tuple":
    """A small synthetic scalar-IR FetchResult + bbox for render_png (no network,
    no pyorbital/pyspectral -- those are only used by the true-color FETCH)."""
    import numpy as np
    from satellites import FetchResult
    import datetime as dt
    H = W = 40
    bbox = [-80.0, 20.0, -60.0, 40.0]
    lons, lats = np.meshgrid(np.linspace(bbox[0], bbox[2], W),
                             np.linspace(bbox[3], bbox[1], H))
    bt = np.full((H, W), 270.0, np.float32)
    bt[10:30, 10:30] = 220.0
    data = FetchResult(cmi=bt, lats=lats.astype(np.float32),
                       lons=lons.astype(np.float32), channel=13,
                       generic_channel="clean_ir",
                       scan_start=dt.datetime(2026, 6, 20, 12, 0,
                                              tzinfo=dt.timezone.utc),
                       product="CMIPF", bucket="noaa-goes19", sat_name="GOES-19",
                       sub_sat_lon=-75.2, units="K")
    return data, bbox


class RenderPngDpiTests(unittest.TestCase):
    """render_png's dpi knob scales the OUTPUT pixels uniformly; default==today."""

    def _dims(self, dpi):
        import io
        from PIL import Image
        import render
        data, bbox = _synthetic_ir()
        png = render.render_png(data, bbox, 13, "2026-06-20 12:00", "rainbow_ir",
                                1, dpi=dpi)
        return Image.open(io.BytesIO(png)).size

    def test_dpi_scales_output_resolution(self):
        w_low, h_low = self._dims(70)
        w_def, h_def = self._dims(110)
        w_high, h_high = self._dims(200)
        # low smaller, high bigger -- the tiers now differ in OUTPUT resolution.
        self.assertLess(w_low, w_def)
        self.assertLess(w_def, w_high)
        # 12in figure at 110 dpi -> ~1320 px wide (the unchanged default).
        self.assertAlmostEqual(w_def, 1320, delta=4)
        self.assertAlmostEqual(w_low, 840, delta=6)    # 12in x 70dpi
        self.assertAlmostEqual(w_high, 2400, delta=8)  # 12in x 200dpi

    def test_default_dpi_is_the_param_default(self):
        # render_png()'s dpi default IS 110, so the default tier == calling it
        # with no dpi arg -> byte-identical to the pre-change call site.
        import io
        from PIL import Image
        import render
        data, bbox = _synthetic_ir()
        a = render.render_png(data, bbox, 13, "t", "rainbow_ir", 1)
        b = render.render_png(data, bbox, 13, "t", "rainbow_ir", 1, dpi=110)
        self.assertEqual(a, b)


class RenderBackdropTests(unittest.TestCase):
    """PART 4: render_backdrop_webp = bare GRAYSCALE Clean-IR cutout, WebP,
    georeferenced to bbox, zero chrome -- the ASCAT viewer's clean backdrop."""

    def test_returns_grayscale_webp_tracking_bbox_aspect(self):
        import io
        from PIL import Image
        import numpy as np
        import render
        data, bbox = _synthetic_ir()                      # 20x20deg square bbox
        out = render.render_backdrop_webp(data, bbox)
        self.assertEqual(out[:4], b"RIFF")
        self.assertEqual(out[8:12], b"WEBP")
        im = Image.open(io.BytesIO(out)).convert("RGB")
        # square bbox -> ~square raster (no chrome margins skewing the aspect).
        self.assertAlmostEqual(im.width / im.height, 1.0, delta=0.06)
        # grayscale: R==G==B on every sampled pixel (bare gray palette).
        arr = np.asarray(im)
        s = arr[::40, ::40].reshape(-1, 3)
        self.assertTrue(np.all(s[:, 0] == s[:, 1]) and np.all(s[:, 1] == s[:, 2]))

    def test_degenerate_field_raises(self):
        import numpy as np
        import render
        data, bbox = _synthetic_ir()
        data.cmi[:] = np.nan                              # mostly-NaN -> bail
        with self.assertRaises(RuntimeError):
            render.render_backdrop_webp(data, bbox)

    def test_non_gray_enhancement_raises(self):
        import render
        data, bbox = _synthetic_ir()
        with self.assertRaises(ValueError):
            render.render_backdrop_webp(data, bbox, enhancement="rainbow_ir")


if __name__ == "__main__":
    unittest.main()
