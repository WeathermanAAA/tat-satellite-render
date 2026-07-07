#!/usr/bin/env python3
"""Tests for the calibrated BT data raster (s2_bt) + its emit/manifest wiring."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_bt  # noqa: E402
import s2_pyramid as P  # noqa: E402
import s2_registry as R  # noqa: E402

STAMP = "20260707T013617Z"
BOUNDS = (-125.0, 15.0, -66.0, 49.0)


class _FakeR2:
    def __init__(self): self.store = {}
    def put_bytes(self, k, d, c, ca): self.store[k] = d; return True
    def put_json(self, k, o, ca): self.store[k] = json.dumps(o).encode(); return True
    def head(self, k): return k in self.store
    def delete(self, ks):
        for k in ks: self.store.pop(k, None)
    def list_keys(self, p): return [k for k in self.store if k.startswith(p)]


class TestBTCodec(unittest.TestCase):
    def test_roundtrip_lossless(self):
        a = np.array([[-95.0, -80.0, 0.0, 40.0],
                      [np.nan, -40.5, 12.34, -70.0]], np.float32)
        back = s2_bt.decode_bt_png(s2_bt.encode_bt_png(a))
        fin = np.isfinite(a)
        self.assertLess(float(np.abs(back[fin] - a[fin]).max()), 0.005)  # 0.01 scale
        self.assertTrue(np.isnan(back[1, 0]))            # nodata preserved

    def test_decode_value_matches_grid(self):
        a = np.array([[-73.21, 5.5]], np.float32)
        png = s2_bt.encode_bt_png(a)
        from PIL import Image
        import io
        px = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
        self.assertAlmostEqual(s2_bt.decode_value(*px[0, 0, [0, 1, 3]]), -73.21, places=2)

    def test_nodata_decode_none(self):
        self.assertIsNone(s2_bt.decode_value(0, 0, 0))   # alpha 0 -> None

    def test_descriptor_shape(self):
        d = s2_bt.bt_descriptor("sat/goes19/conus/ir", BOUNDS, (1280, 566))
        self.assertEqual(d["path"], "sat/goes19/conus/ir/{t}/bt.png")
        self.assertEqual((d["scale"], d["offset"], d["units"]), (0.01, -120.0, "degC"))
        self.assertEqual(d["dims"], [1280, 566])
        self.assertEqual(d["bounds"], [-125.0, 15.0, -66.0, 49.0])


class TestBTEmit(unittest.TestCase):
    def setUp(self):
        self.e = R.REGISTRY_BY_ID["goes19-conus-ir"]

    def _raster(self):
        a = np.zeros((256, 512, 4), np.uint8); a[..., 3] = 255
        a[..., 0] = np.linspace(0, 255, 512, dtype=np.uint8)[None, :]
        return a

    def test_emit_writes_bt_beside_tiles_and_manifest_block(self):
        r2 = _FakeR2()
        bt = s2_bt.encode_bt_png(np.full((100, 200), -50.0, np.float32))
        meta = P.emit_pyramid(self.e, r2, "shadow", STAMP, self._raster(), BOUNDS,
                              P.PyramidSpec(method=0), bt_png=bt)
        self.assertTrue(meta["has_bt"])
        self.assertIn(self.e.bt_key("shadow", STAMP), r2.store)          # bt.png beside tiles
        self.assertEqual(r2.store[self.e.bt_key("shadow", STAMP)], bt)   # exact bytes
        desc = s2_bt.bt_descriptor(self.e.product_path, BOUNDS, (200, 100))
        P.write_tiled_manifest(self.e, r2, "shadow", [STAMP], BOUNDS, (512, 256), 0,
                               __import__("datetime").datetime(2026, 7, 7,
                                   tzinfo=__import__("datetime").timezone.utc),
                               scheme="webmercator-xyz", bt=desc)
        man = json.loads(r2.store[self.e.latest_times_key("shadow")])
        self.assertEqual(man["bt"]["path"], "sat/goes19/conus/ir/{t}/bt.png")
        self.assertEqual(man["bt"]["scale"], 0.01)

    def test_no_bt_leaves_manifest_bt_null(self):
        r2 = _FakeR2()
        import datetime as dt
        man = self.e.build_tiled_latest_times(
            [STAMP], bounds=BOUNDS, image_px=(512, 256), maxzoom=0,
            as_of=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
        self.assertIsNone(man["bt"])                     # default: no inspector block


if __name__ == "__main__":
    unittest.main()
