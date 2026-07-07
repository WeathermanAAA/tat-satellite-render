#!/usr/bin/env python3
"""Tests for the Web-Mercator (EPSG:3857) reprojection (s2_webmerc, §5.5) + the
scheme-aware emit/manifest path. Synthetic rasters, no network. Speed: method=0."""
import datetime as dt
import io
import json
import math
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_pyramid as P  # noqa: E402
import s2_registry as R  # noqa: E402
import s2_webmerc as WM  # noqa: E402

UTC = dt.timezone.utc
FAST = P.PyramidSpec(method=0)
STAMP = "20260707T013617Z"
BOUNDS = (-100.0, 20.0, -60.0, 50.0)   # 40deg lon x 30deg lat


def _opaque(h, w):
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    a[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    a[..., 2] = 120
    a[..., 3] = 255
    return a


class TestWebMercGeo(unittest.TestCase):
    def test_merc_y_to_lat(self):
        self.assertAlmostEqual(WM.merc_y_to_lat(0.5), 0.0, places=6)         # equator
        self.assertAlmostEqual(WM.merc_y_to_lat(0.0), WM.WEBMERC_LAT_LIMIT, places=4)
        self.assertAlmostEqual(WM.merc_y_to_lat(1.0), -WM.WEBMERC_LAT_LIMIT, places=4)

    def test_tile_geo_bounds_z0_is_world(self):
        w, s, e, n = WM.tile_geo_bounds(0, 0, 0)
        self.assertAlmostEqual(w, -180.0); self.assertAlmostEqual(e, 180.0)
        self.assertAlmostEqual(n, WM.WEBMERC_LAT_LIMIT, places=4)
        self.assertAlmostEqual(s, -WM.WEBMERC_LAT_LIMIT, places=4)

    def test_z1_nw_quadrant(self):
        w, s, e, n = WM.tile_geo_bounds(1, 0, 0)   # north-west quadrant
        self.assertAlmostEqual(w, -180.0); self.assertAlmostEqual(e, 0.0)
        self.assertAlmostEqual(s, 0.0, places=6)   # equator at the bottom
        self.assertAlmostEqual(n, WM.WEBMERC_LAT_LIMIT, places=4)


class TestReproject(unittest.TestCase):
    def test_native_max_zoom_scales_with_resolution(self):
        # 800px over 40deg lon = 20 px/deg -> ceil(log2(20*360/512)) = ceil(3.81) = 4
        self.assertEqual(WM.native_max_zoom(BOUNDS, (800, 600)), 4)
        self.assertGreater(WM.native_max_zoom(BOUNDS, (1600, 1200)), 4)  # finer -> deeper

    def test_reproject_tile_inside_vs_outside(self):
        src = _opaque(600, 800)
        # z0/0/0 covers the whole world -> the CONUS-ish source is a small opaque
        # patch inside a mostly-transparent tile: non-empty.
        t = WM.reproject_tile(src, BOUNDS, 0, 0, 0)
        self.assertEqual(t.shape, (512, 512, 4))
        self.assertGreater(int(t[..., 3].max()), 0)
        # a tile on the far side of the planet is entirely off-data -> None.
        self.assertIsNone(WM.reproject_tile(src, BOUNDS, 2, 3, 1))  # ~ +90..180 lon


class TestCutWebMerc(unittest.TestCase):
    def test_z0_global_counts_and_scheme(self):
        cut = WM.cut_webmerc_pyramid(_opaque(600, 800), BOUNDS, FAST, maxzoom=3)
        self.assertEqual(cut["scheme"], "webmercator-xyz")
        self.assertEqual(cut["maxzoom"], 3)
        self.assertIn((0, 0, 0), cut["tiles"])          # global z0 tile present
        for (z, x, y), b in cut["tiles"].items():
            self.assertEqual(Image.open(io.BytesIO(b)).size, (512, 512))

    def test_deterministic(self):
        a = _opaque(400, 600)
        c1 = WM.cut_webmerc_pyramid(a, BOUNDS, FAST, maxzoom=3)
        c2 = WM.cut_webmerc_pyramid(a, BOUNDS, FAST, maxzoom=3)
        self.assertEqual(c1["tiles"].keys(), c2["tiles"].keys())
        for k in c1["tiles"]:
            self.assertEqual(c1["tiles"][k], c2["tiles"][k])

    def test_lat_clamped_to_webmerc_limit(self):
        # a source spanning past the pole clamps to +/-85.05 without error.
        wide = (-160.0, -89.0, -20.0, 89.0)
        cut = WM.cut_webmerc_pyramid(_opaque(400, 600), wide, FAST, maxzoom=2)
        self.assertIn((0, 0, 0), cut["tiles"])


class TestEmitWebMerc(unittest.TestCase):
    def setUp(self):
        self.e = R.REGISTRY_BY_ID["goes19-conus-ir"]   # pyramid_scheme webmercator

    def test_emit_webmercator_writes_tiles_and_marker(self):
        r2 = _FakeR2()
        meta = P.emit_pyramid(self.e, r2, "shadow", STAMP, _opaque(600, 900),
                              BOUNDS, FAST, scheme="webmercator-xyz")
        self.assertEqual(meta["outcome"], "rendered")
        self.assertEqual(meta["scheme"], "webmercator-xyz")
        self.assertIn(self.e.ready_key("shadow", STAMP), r2.store)          # marker
        self.assertTrue(any(k.endswith(".webp") for k in r2.store))
        # complete_stamps recovers it with its webmerc maxzoom
        frames = dict(P.complete_stamps(self.e, r2, "shadow"))
        self.assertIn(STAMP, frames)
        self.assertEqual(frames[STAMP], meta["maxzoom"])

    def test_manifest_scheme_webmercator(self):
        lt = self.e.build_tiled_latest_times(
            [STAMP], bounds=BOUNDS, image_px=(900, 600), maxzoom=5,
            as_of=dt.datetime(2026, 7, 7, tzinfo=UTC), scheme="webmercator-xyz")
        self.assertEqual(lt["scheme"], "webmercator-xyz")
        self.assertEqual(lt["maxzoom"], 5)
        self.assertEqual(lt["tile"], "sat/goes19/conus/ir/{t}/{z}/{x}/{y}.webp")

    def test_registry_rows_declare_webmercator(self):
        for pid in ("goes19-conus-ir", "goes19-fd-ir"):
            self.assertEqual(R.REGISTRY_BY_ID[pid].pyramid_scheme, "webmercator-xyz")


class _FakeR2:
    def __init__(self): self.store = {}
    def put_bytes(self, k, d, c, ca): self.store[k] = d; return True
    def put_json(self, k, o, ca): self.store[k] = json.dumps(o).encode(); return True
    def head(self, k): return k in self.store
    def delete(self, ks):
        for k in ks: self.store.pop(k, None)
    def list_keys(self, p): return [k for k in self.store if k.startswith(p)]


if __name__ == "__main__":
    unittest.main()
