"""Vis/SWIR satellite-backdrop revision: render_backdrop_webp's VISIBLE branch,
day/night band selection, per-storm bd_product stamping, and the basin-extent
backdrop emitter + floaters/backdrops.json manifest block. Network-free."""
import datetime as dt
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import render  # noqa: E402
import floater_poller as fp  # noqa: E402
from satellites import FetchResult  # noqa: E402


def _fetch(units):
    H = W = 32
    bbox = [-80.0, 20.0, -60.0, 40.0]
    lons, lats = np.meshgrid(np.linspace(bbox[0], bbox[2], W),
                             np.linspace(bbox[3], bbox[1], H))
    if units == "1":
        cmi = np.linspace(0.05, 0.9, H * W).reshape(H, W).astype(np.float32)
    else:
        cmi = np.full((H, W), 270.0, np.float32)
        cmi[8:24, 8:24] = 215.0
    return FetchResult(cmi=cmi, lats=lats.astype(np.float32),
                       lons=lons.astype(np.float32), channel=2,
                       generic_channel="visible_red",
                       scan_start=dt.datetime(2026, 6, 21, 16, 0, tzinfo=dt.timezone.utc),
                       product="CMIPF", bucket="noaa-goes19", sat_name="GOES-19",
                       sub_sat_lon=-75.2, units=units), bbox


class TestVisibleBackdrop(unittest.TestCase):
    def test_visible_branch_renders_grayscale_webp(self):
        data, bbox = _fetch("1")
        out = render.render_backdrop_webp(data, bbox)
        self.assertEqual(out[:4], b"RIFF")
        self.assertEqual(out[8:12], b"WEBP")
        from PIL import Image
        im = Image.open(io.BytesIO(out)).convert("RGB")
        px = list(im.getdata())
        for r, g, b in px[:: max(1, len(px) // 60)]:
            self.assertLessEqual(abs(r - g), 2)
            self.assertLessEqual(abs(g - b), 2)

    def test_visible_branch_ignores_nongray_enhancement(self):
        # the gray-enhancement guard is the THERMAL path only; the visible
        # (reflectance) branch renders regardless of the enhancement arg.
        data, bbox = _fetch("1")
        self.assertEqual(render.render_backdrop_webp(data, bbox, enhancement="rainbow_ir")[8:12], b"WEBP")

    def test_thermal_path_still_rejects_nongray(self):
        data, bbox = _fetch("K")
        with self.assertRaises(ValueError):
            render.render_backdrop_webp(data, bbox, enhancement="rainbow_ir")


class TestDayNightBand(unittest.TestCase):
    def test_day_returns_vis(self):
        ch, prod = fp.backdrop_band(
            0.0, 0.0, dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.timezone.utc))
        self.assertEqual((ch, prod), ("visible_red", "Vis"))

    def test_night_returns_swir(self):
        ch, prod = fp.backdrop_band(
            0.0, 0.0, dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.timezone.utc))
        self.assertEqual((ch, prod), ("shortwave_ir", "SWIR"))


class _RecR2:
    def __init__(self):
        self.json_puts = {}
        self.byte_puts = {}

    def get_json(self, key):
        return None

    def put_json(self, key, obj, cache):
        self.json_puts[key] = obj
        return True

    def put_bytes(self, key, data, content_type, cache):
        self.byte_puts[key] = content_type
        return True

    def delete(self, keys):
        pass


def _poller(rec_calls):
    with mock.patch.object(fp, "R2", _RecR2):
        p = fp.Poller()
    p.limiter = mock.Mock()

    def fake_render(session, bbox, channel, enhancement, storm=None, backdrop=False):
        rec_calls.append({"channel": channel, "enh": enhancement,
                          "backdrop": backdrop, "bbox": bbox})
        return (b"BDWEBP" if backdrop else b"CHROMEDWEBP"), {"Content-Type": "image/webp"}
    return p, fake_render


class TestPerStormProduct(unittest.TestCase):
    def test_backdrop_uses_daynight_band_and_stamps_bd_product(self):
        calls = []
        p, fake_render = _poller(calls)
        storm = fp.Storm(sid="JTWC_WP082026", slug="wp08", name="HIGOS", basin="WP",
                         lat=18.0, lon=235.0, category="TS", intensity_kt=45.0,
                         last_fix="2026-06-27T00:00:00Z", current_wind_kt=45.0,
                         current_pressure_mb=995.0, nature="TS")
        u = fp.Unit(storm=storm, band=fp.BANDS_BY_KEY["ir"])
        with mock.patch.object(fp, "call_render", fake_render), \
             mock.patch.object(fp, "FLOATER_BACKDROP_ENABLED", True), \
             mock.patch.object(fp, "FLOATER_BACKDROP_BAND", "ir"):
            p.process_unit(u)
        man = next(v for k, v in p.r2.json_puts.items() if k.endswith("/manifest.json"))
        fr = man["bands"]["ir"]["frames"][-1]
        self.assertIn(fr["bd_product"], ("Vis", "SWIR"))
        # the backdrop render used the matching day/night channel (not the IR band)
        bd_call = next(c for c in calls if c["backdrop"])
        expect_ch = "visible_red" if fr["bd_product"] == "Vis" else "shortwave_ir"
        self.assertEqual(bd_call["channel"], expect_ch)
        self.assertEqual(len(fr["bounds"]), 4)


class TestBasinBackdrops(unittest.TestCase):
    def test_basin_emitter_publishes_backdrops_index(self):
        calls = []
        p, fake_render = _poller(calls)
        with mock.patch.object(fp, "call_render", fake_render), \
             mock.patch.object(fp, "FLOATER_BACKDROP_ENABLED", True):
            p.refresh_basin_backdrops()
        idx = p.r2.json_puts.get(f"{fp.R2_PREFIX}/backdrops.json")
        self.assertIsNotNone(idx, "backdrops.json not written")
        bd = idx["backdrops"]
        self.assertEqual(set(bd), set(fp.BASIN_BACKDROP_REGIONS))
        for region, entry in bd.items():
            self.assertIn(entry["product"], ("Vis", "SWIR"))
            self.assertEqual(len(entry["bounds"]), 4)
            self.assertTrue(entry["key"].startswith(f"{fp.R2_PREFIX}/backdrops/{region}/"))
            self.assertIn(entry["key"], p.r2.byte_puts)
        # every basin render forced the bare-backdrop path
        self.assertTrue(all(c["backdrop"] for c in calls))

    def test_disabled_emits_nothing(self):
        calls = []
        p, fake_render = _poller(calls)
        with mock.patch.object(fp, "call_render", fake_render), \
             mock.patch.object(fp, "FLOATER_BACKDROP_ENABLED", False):
            p.refresh_basin_backdrops()
        self.assertNotIn(f"{fp.R2_PREFIX}/backdrops.json", p.r2.json_puts)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
