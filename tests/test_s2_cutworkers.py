"""The threaded tile cut must be BYTE-IDENTICAL to the serial one.

S2_CUT_WORKERS (2026-07-25) parallelizes cut_webmerc_pyramid's per-tile unit so
one lane can use more than one core -- the cut is ~90% of a product-slot's CPU
and was strictly single-threaded, which capped the whole ring at one satellite
per core. Each tile is an independent, deterministic function of the source
raster, so parallelizing must not change a single byte. If it ever does, tiles
on R2 would silently depend on how many workers the lane happened to run with.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import s2_pyramid as P            # noqa: E402
import s2_webmerc as W            # noqa: E402


def _raster(h=512, w=1024, seed=7):
    rng = np.random.default_rng(seed)
    # structured, not pure noise: gradients + blobs compress like real imagery
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((np.sin(xx / 40.0) + np.cos(yy / 30.0) + 2) * 60).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = base
    rgba[..., 1] = np.roll(base, 13, axis=1)
    rgba[..., 2] = rng.integers(0, 255, (h, w), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


BOUNDS = (-156.0, -60.0, 6.0, 60.0)
AM_BOUNDS = (48.0, -60.0, 208.0, 60.0)     # antimeridian-crossing (GK-2A/AHI)


def cut_with(workers, rgba, bounds, maxzoom=3):
    prev = os.environ.get("S2_CUT_WORKERS")
    os.environ["S2_CUT_WORKERS"] = str(workers)
    try:
        return W.cut_webmerc_pyramid(rgba, bounds, P.PyramidSpec(),
                                     maxzoom=maxzoom)
    finally:
        if prev is None:
            os.environ.pop("S2_CUT_WORKERS", None)
        else:
            os.environ["S2_CUT_WORKERS"] = prev


class TestCutWorkerParity(unittest.TestCase):
    def _assert_identical(self, a, b):
        self.assertEqual(a["maxzoom"], b["maxzoom"])
        self.assertEqual(a["image_px"], b["image_px"])
        self.assertEqual(a["tile_counts"], b["tile_counts"])
        self.assertEqual(sorted(a["tiles"]), sorted(b["tiles"]))
        for k in a["tiles"]:
            self.assertEqual(a["tiles"][k], b["tiles"][k],
                             f"tile {k} differs between worker counts")

    def test_serial_vs_threaded_bytes(self):
        r = _raster()
        self._assert_identical(cut_with(1, r, BOUNDS), cut_with(4, r, BOUNDS))

    def test_many_workers_still_identical(self):
        r = _raster(seed=11)
        self._assert_identical(cut_with(1, r, BOUNDS), cut_with(16, r, BOUNDS))

    def test_antimeridian_two_lobe_range_parity(self):
        # the crossing case builds TWO x ranges and dedups between them --
        # exactly where a threaded rewrite could drop or double a tile
        r = _raster(seed=3)
        a, b = cut_with(1, r, AM_BOUNDS), cut_with(8, r, AM_BOUNDS)
        self._assert_identical(a, b)
        self.assertTrue(sum(a["tile_counts"].values()) > 0)

    def test_tile_count_matches_dedup_semantics(self):
        # a wrapped range must not produce duplicate (z,x,y) keys
        r = _raster(seed=5)
        out = cut_with(8, r, AM_BOUNDS)
        self.assertEqual(len(out["tiles"]), sum(out["tile_counts"].values()))


class TestEncodeEffort(unittest.TestCase):
    def test_method_is_a_spec_knob_not_a_constant(self):
        # the encoder effort must stay tunable per spec: the ring's CPU budget
        # depends on it, and a hardcoded value would strand that lever
        self.assertIn("method", {f.name for f in
                                 __import__("dataclasses").fields(P.PyramidSpec)})
        fast = P.PyramidSpec(method=4)
        slow = P.PyramidSpec(method=6)
        self.assertEqual(fast.quality, slow.quality)   # quality sets the ceiling


if __name__ == "__main__":
    unittest.main()
