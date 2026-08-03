#!/usr/bin/env python3
"""s2_container: zoom-banded USTAR blocks with verified byte offsets (the
hafs_render #27 pattern ported to tile pyramids; 2026-08-03 cost incident).

The safety argument mirrors hafs_container's: offsets are computed
arithmetically and VERIFIED by decode-back before publishing -- a wrong
offset serves a viewer garbage bytes -- and a container-published frame must
remain indistinguishable to every consumer of the emit contract (ready
marker last, complete_stamps geometry answer, byte-identical tile payloads
recoverable at the recorded ranges)."""
import io
import os
import shutil
import sys
import tarfile
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_container as C          # noqa: E402
import s2_pyramid as P            # noqa: E402
import s2_registry as R           # noqa: E402


def _tiles(spec_counts):
    """{z: n} -> synthetic tile dict {(z,x,y): distinct bytes}."""
    out = {}
    for z, n in spec_counts.items():
        for i in range(n):
            out[(z, i, 0)] = (f"tile-{z}-{i}-".encode() + os.urandom(11))
    return out


class TestPlanZoomBands(unittest.TestCase):
    def test_overview_band_plus_deep_levels(self):
        self.assertEqual(C.plan_zoom_bands([0, 1, 2, 3, 4, 5, 6, 7]),
                         [[0, 1, 2, 3, 4], [5], [6], [7]])

    def test_operates_on_present_zooms_only(self):
        self.assertEqual(C.plan_zoom_bands([0, 5]), [[0], [5]])
        self.assertEqual(C.plan_zoom_bands([0]), [[0]])


class TestBlockOffsets(unittest.TestCase):
    def test_every_member_recoverable_at_recorded_range(self):
        tiles = _tiles({0: 1, 1: 2, 5: 7})
        built = C.build_frame_containers(tiles, maxzoom=5)
        idx = built["index"]
        self.assertEqual(idx["count"], len(tiles))
        for name, (bkey, off, size) in idx["tiles"].items():
            data = built["blocks"][bkey]
            payload = data[off:off + size]
            z, x, y = name[:-len(".webp")].split("/")
            self.assertEqual(payload, tiles[(int(z), int(x), int(y))],
                             f"range mismatch for {name}")

    def test_verify_catches_corrupt_offsets(self):
        data, index = C.build_block([("0/0/0.webp", b"abc")])
        bad = dict(index)
        bad["0/0/0.webp"] = [index["0/0/0.webp"][0] + 1,
                             index["0/0/0.webp"][1]]
        with self.assertRaises(RuntimeError):
            C._verify_block(data, bad)

    def test_blocks_are_valid_tars_with_fixed_mtime(self):
        tiles = _tiles({0: 1, 6: 3})
        built = C.build_frame_containers(tiles, maxzoom=6)
        for data in built["blocks"].values():
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                for m in tar.getmembers():
                    self.assertEqual(m.mtime, 0)

    def test_index_key_roundtrip(self):
        self.assertEqual(C.index_key_name(7), "tiles.z7.json")
        self.assertEqual(C.maxzoom_from_index_key("a/b/20260803T000000Z/tiles.z7.json"), 7)
        self.assertIsNone(C.maxzoom_from_index_key("a/b/latest_times.json"))
        self.assertIsNone(C.maxzoom_from_index_key("a/b/tiles.zx.json"))


class TestEmitPyramidContainerMode(unittest.TestCase):
    """emit_pyramid with S2_CONTAINER_TILES=1: same contract, ~6 objects."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = P.FilesystemStore(self.tmp)
        self.entry = next(e for e in R.REGISTRY if e.tiled)
        os.environ["S2_CONTAINER_TILES"] = "1"
        self.addCleanup(os.environ.pop, "S2_CONTAINER_TILES", None)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _emit(self, stamp="20260803T120000Z", px=1200):
        raster = np.full((px, px, 4), 180, dtype=np.uint8)
        return P.emit_pyramid(self.entry, self.store, "shadow", stamp, raster,
                              (-125.0, 15.0, -66.0, 49.0),
                              P.PyramidSpec(quality=60),
                              scheme="flat-native-xyz")

    def test_container_frame_completes_and_probes_maxzoom(self):
        meta = self._emit()
        self.assertEqual(meta["outcome"], "rendered")
        # objects in the stamp dir: blocks + tiles index + ready marker; no
        # per-tile objects
        keys = self.store.list_keys(
            self.entry.tile_stamp_prefix("shadow", "20260803T120000Z"))
        exts = sorted({k.rsplit(".", 1)[-1] for k in keys})
        self.assertNotIn("webp", exts)
        self.assertTrue(any(k.endswith(".tar") for k in keys))
        self.assertTrue(any(k.endswith(f"tiles.z{meta['maxzoom']}.json")
                            for k in keys))
        # complete_stamps answers the same (stamp, maxzoom) from key layout
        frames = P.complete_stamps(self.entry, self.store, "shadow")
        self.assertEqual(frames, [("20260803T120000Z", meta["maxzoom"])])

    def test_dedup_and_flat_mode_unaffected(self):
        self._emit()
        again = self._emit()                    # ready marker: duplicate
        self.assertEqual(again["outcome"], "duplicate")
        os.environ["S2_CONTAINER_TILES"] = "0"
        meta = self._emit("20260803T121000Z")
        self.assertEqual(meta["outcome"], "rendered")
        keys = self.store.list_keys(
            self.entry.tile_stamp_prefix("shadow", "20260803T121000Z"))
        self.assertTrue(any(k.endswith(".webp") for k in keys))


if __name__ == "__main__":
    unittest.main()
