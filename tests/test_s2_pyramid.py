#!/usr/bin/env python3
"""Pure-logic tests for the Stage-2 shadow pyramid emitter (s2_pyramid) + the
tiled ProductEntry helpers (s2_registry). No network, no boto3, no matplotlib --
synthetic rasters + a hand FakeR2/FilesystemStore (the repo convention; moto is
reserved for SQS only). Covers: the pyramid cut (maxzoom/levels/512/skip-empty/
lossless-exactness/determinism), deterministic tile keys, the SLIDER manifest,
idempotency + PUT-failure, per-stamp prune, cold-start stamp recovery, and the
new tiled registry rows + routing (no cross-substrate leak).

Speed: encode with WebP method=0 (FAST) -- these assert geometry/keys/logic, not
codec weight, and method=0 stays deterministic + lossless-exact."""
import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_pyramid as P  # noqa: E402
import s2_registry as R  # noqa: E402

UTC = dt.timezone.utc
STAMP = "20260707T003117Z"
STAMP2 = "20260707T003617Z"
BOUNDS = (-125.0, 15.0, -66.0, 49.0)
FAST = P.PyramidSpec(method=0)                 # fast encode; same geometry/keys
FAST_LL = P.PyramidSpec(method=0, lossless=True)


class FakeR2:
    """In-memory S3/R2 stand-in (the test_s1_ingest convention)."""

    def __init__(self):
        self.store = {}

    def put_bytes(self, key, data, content_type, cache):
        self.store[key] = data
        return True

    def put_json(self, key, obj, cache):
        self.store[key] = json.dumps(obj, separators=(",", ":")).encode()
        return True

    def head(self, key):
        return key in self.store

    def delete(self, keys):
        for k in keys:
            self.store.pop(k, None)

    def list_keys(self, prefix):
        return [k for k in self.store if k.startswith(prefix)]


class FailingPutR2(FakeR2):
    def put_bytes(self, key, data, content_type, cache):
        if key.endswith(".webp"):
            return False
        return super().put_bytes(key, data, content_type, cache)


class FlakyR2(FakeR2):
    """Fails the Nth .webp PUT once (leaving a PARTIAL pyramid + no marker),
    then succeeds -- models a transient mid-emit R2 error / crash for the
    self-heal (completion-marker) regression test."""

    def __init__(self, fail_at=3):
        super().__init__(); self.n = 0; self.fail_at = fail_at

    def put_bytes(self, key, data, content_type, cache):
        if key.endswith(".webp"):
            self.n += 1
            if self.n == self.fail_at:
                return False
        return super().put_bytes(key, data, content_type, cache)


def _gradient(h, w, alpha=255):
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    a[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    a[..., 2] = 96
    a[..., 3] = alpha
    return a


def _reassemble_maxzoom(cut):
    T, (W, H), maxz = 512, cut["image_px"], cut["maxzoom"]
    canvas = np.zeros((((H + T - 1) // T) * T, ((W + T - 1) // T) * T, 4), np.uint8)
    for (z, x, y), b in cut["tiles"].items():
        if z != maxz:
            continue
        canvas[y * T:y * T + T, x * T:x * T + T] = np.asarray(
            Image.open(io.BytesIO(b)).convert("RGBA"))
    return canvas[:H, :W]


class TestCutMath(unittest.TestCase):
    def test_max_zoom_for(self):
        self.assertEqual(P.max_zoom_for(512, 512), 0)
        self.assertEqual(P.max_zoom_for(4096, 1812), 3)   # ceil(log2(8))
        self.assertEqual(P.max_zoom_for(8192, 3000), 4)
        self.assertEqual(P.max_zoom_for(300, 200), 0)     # smaller than one tile

    def test_level_dims_native_at_maxzoom(self):
        self.assertEqual(P.level_dims(4096, 1812, 3, 3), (4096, 1812))  # z=max is native
        self.assertEqual(P.level_dims(4096, 1812, 3, 0), (512, 227))    # z0 fits a tile

    def test_tile_counts_and_size(self):
        cut = P.cut_pyramid(_gradient(1812, 4096), FAST)
        self.assertEqual(cut["maxzoom"], 3)
        self.assertEqual(cut["image_px"], (4096, 1812))
        # full grid (opaque everywhere): z0=1, z1=2x1, z2=4x2, z3=8x4
        self.assertEqual(cut["tile_counts"], {0: 1, 1: 2, 2: 8, 3: 32})
        for (z, x, y), data in cut["tiles"].items():
            self.assertEqual(Image.open(io.BytesIO(data)).size, (512, 512))

    def test_rgb_input_coerced(self):
        cut = P.cut_pyramid(_gradient(600, 600)[..., :3], FAST)  # HxWx3
        self.assertIn((0, 0, 0), cut["tiles"])

    def test_skip_empty_drops_transparent_tiles(self):
        # left half opaque, right half fully transparent -> right tiles skipped.
        a = _gradient(1024, 2048)
        a[:, 1024:, 3] = 0
        cut = P.cut_pyramid(a, P.PyramidSpec(method=0, skip_empty=True))
        # z=maxzoom(2): grid 4x2=8; right half (x>=2) transparent -> 4 kept
        self.assertEqual(cut["tile_counts"][cut["maxzoom"]], 4)
        keep = P.cut_pyramid(a, P.PyramidSpec(method=0, skip_empty=False))
        self.assertEqual(keep["tile_counts"][keep["maxzoom"]], 8)

    def test_deterministic(self):
        a = _gradient(512, 900)
        c1, c2 = P.cut_pyramid(a, FAST), P.cut_pyramid(a, FAST)
        self.assertEqual(c1["tiles"].keys(), c2["tiles"].keys())
        for k in c1["tiles"]:
            self.assertEqual(c1["tiles"][k], c2["tiles"][k])

    def test_lossless_tiling_is_pixel_exact(self):
        a = _gradient(600, 900)
        reasm = _reassemble_maxzoom(P.cut_pyramid(a, FAST_LL))
        self.assertTrue(np.array_equal(reasm, a))  # byte-for-byte, no tiling drift


class TestStores(unittest.TestCase):
    def test_filesystem_store_parity_with_faker2(self):
        d = tempfile.mkdtemp()
        for store in (FakeR2(), P.FilesystemStore(d)):
            self.assertTrue(store.put_bytes("shadow/a/b/1.webp", b"xx", "image/webp", "c"))
            self.assertTrue(store.put_json("shadow/a/b/m.json", {"k": 1}, "c"))
            self.assertTrue(store.head("shadow/a/b/1.webp"))
            self.assertFalse(store.head("shadow/a/b/missing.webp"))
            self.assertEqual(sorted(store.list_keys("shadow/a/b/")),
                             ["shadow/a/b/1.webp", "shadow/a/b/m.json"])
            store.delete(["shadow/a/b/1.webp"])
            self.assertFalse(store.head("shadow/a/b/1.webp"))

    def test_filesystem_json_is_compact(self):
        d = tempfile.mkdtemp()
        P.FilesystemStore(d).put_json("shadow/m.json", {"a": 1, "b": [1, 2]}, "c")
        with open(os.path.join(d, "shadow/m.json"), "rb") as fh:
            self.assertEqual(fh.read(), b'{"a":1,"b":[1,2]}')  # separators (',',':')


class TestEmit(unittest.TestCase):
    def setUp(self):
        self.e = R.REGISTRY_BY_ID["goes19-conus-ir"]
        self.raster = _gradient(600, 1024)

    def test_emit_writes_deterministic_keys_under_prefix(self):
        r2 = FakeR2()
        meta = P.emit_pyramid(self.e, r2, "shadow", STAMP, self.raster, BOUNDS, FAST)
        self.assertEqual(meta["outcome"], "rendered")
        self.assertTrue(all(k.startswith("shadow/sat/goes19/conus/ir/" + STAMP)
                            for k in r2.store))
        self.assertIn(self.e.ready_key("shadow", STAMP), r2.store)      # completion marker
        self.assertEqual(len(r2.store), meta["n_tiles"] + 1)           # tiles + marker

    def test_idempotent_reemit_is_duplicate(self):
        r2 = FakeR2()
        P.emit_pyramid(self.e, r2, "shadow", STAMP, self.raster, BOUNDS, FAST)
        n = len(r2.store)
        meta2 = P.emit_pyramid(self.e, r2, "shadow", STAMP, self.raster, BOUNDS, FAST)
        self.assertEqual(meta2["outcome"], "duplicate")
        self.assertEqual(len(r2.store), n)   # no new PUTs

    def test_put_failure_raises(self):
        with self.assertRaises(IOError):
            P.emit_pyramid(self.e, FailingPutR2(), "shadow", STAMP, self.raster, BOUNDS, FAST)

    def test_partial_emit_leaves_no_marker_and_self_heals(self):
        # F1 regression: a transient failure mid-pyramid must NOT leave a frame
        # that later runs skip as 'duplicate' -- the marker (written last) is
        # absent, so the frame is not complete and the retry re-renders it.
        store = FlakyR2(fail_at=3)
        with self.assertRaises(IOError):
            P.emit_pyramid(self.e, store, "shadow", STAMP, self.raster, BOUNDS, FAST)
        self.assertFalse(store.head(self.e.ready_key("shadow", STAMP)))   # no marker
        self.assertEqual(P.complete_stamps(self.e, store, "shadow"), [])  # not complete
        meta = P.emit_pyramid(self.e, store, "shadow", STAMP, self.raster, BOUNDS, FAST)
        self.assertEqual(meta["outcome"], "rendered")                    # self-healed
        self.assertTrue(store.head(self.e.ready_key("shadow", STAMP)))
        self.assertEqual(P.stamps_from_store(self.e, store, "shadow"), [STAMP])
        # now idempotent
        self.assertEqual(P.emit_pyramid(self.e, store, "shadow", STAMP, self.raster,
                                        BOUNDS, FAST)["outcome"], "duplicate")

    def test_min_zoom_gt0_still_idempotent(self):
        # F4 regression: with min_zoom>0 there is NO z0 tile, but the marker
        # (not a hardcoded z0 tile) still makes a re-emit a no-op.
        r2 = FakeR2()
        spec = P.PyramidSpec(method=0, min_zoom=1)
        m1 = P.emit_pyramid(self.e, r2, "shadow", STAMP, self.raster, BOUNDS, spec)
        self.assertEqual(m1["outcome"], "rendered")
        self.assertNotIn(self.e.tile_key("shadow", STAMP, 0, 0, 0), r2.store)  # no z0
        m2 = P.emit_pyramid(self.e, r2, "shadow", STAMP, self.raster, BOUNDS, spec)
        self.assertEqual(m2["outcome"], "duplicate")


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.e = R.REGISTRY_BY_ID["goes19-conus-ir"]

    def test_tiled_manifest_shape(self):
        r2 = FakeR2()
        as_of = dt.datetime(2026, 7, 7, 0, 40, tzinfo=UTC)
        lt = P.write_tiled_manifest(self.e, r2, "shadow", [STAMP2, STAMP, STAMP2],
                                    BOUNDS, (4096, 1812), 3, as_of)
        self.assertEqual(lt["product"], "sat/goes19/conus/ir")
        self.assertIsNone(lt["path"])                       # tiled: no single-frame path
        self.assertEqual(lt["tile"], "sat/goes19/conus/ir/{t}/{z}/{x}/{y}.webp")
        self.assertEqual(lt["scheme"], "flat-native-xyz")
        self.assertEqual((lt["minzoom"], lt["maxzoom"], lt["tile_size"]), (0, 3, 512))
        self.assertEqual(lt["image_px"], [4096, 1812])
        self.assertEqual(lt["bounds"], [-125.0, 15.0, -66.0, 49.0])
        self.assertEqual(lt["times"], [STAMP, STAMP2])      # sorted+deduped
        self.assertEqual(lt["latest"], STAMP2)
        self.assertEqual(lt["count"], 2)
        self.assertEqual(lt["as_of"], "2026-07-07T00:40:00Z")
        self.assertIn(self.e.latest_times_key("shadow"), r2.store)

    def test_manifest_round_trips_from_r2(self):
        r2 = FakeR2()
        P.write_tiled_manifest(self.e, r2, "shadow", [STAMP], BOUNDS, (4096, 1812),
                               3, dt.datetime(2026, 7, 7, tzinfo=UTC))
        raw = r2.store[self.e.latest_times_key("shadow")]
        self.assertEqual(json.loads(raw)["latest"], STAMP)


class TestPruneAndColdStart(unittest.TestCase):
    def setUp(self):
        self.e = R.REGISTRY_BY_ID["goes19-conus-ir"]

    def test_prune_deletes_all_tiles_of_dead_stamp_only(self):
        r2 = FakeR2()
        raster = _gradient(768, 768)
        P.emit_pyramid(self.e, r2, "shadow", STAMP, raster, BOUNDS, FAST)
        P.emit_pyramid(self.e, r2, "shadow", STAMP2, raster, BOUNDS, FAST)
        n = P.prune_tiles(self.e, r2, "shadow", [STAMP])
        self.assertGreater(n, 1)                              # many keys per stamp
        self.assertNotIn(self.e.tile_key("shadow", STAMP, 0, 0, 0), r2.store)
        self.assertIn(self.e.tile_key("shadow", STAMP2, 0, 0, 0), r2.store)  # survives
        self.assertEqual(P.stamps_from_store(self.e, r2, "shadow"), [STAMP2])

    def test_stamps_from_store_ignores_manifest_and_marker(self):
        r2 = FakeR2()
        P.emit_pyramid(self.e, r2, "shadow", STAMP, _gradient(600, 600), BOUNDS, FAST)
        P.write_tiled_manifest(self.e, r2, "shadow", [STAMP], BOUNDS, (600, 600), 1,
                               dt.datetime(2026, 7, 7, tzinfo=UTC))
        # neither latest_times.json nor _ready.json is mistaken for a frame stamp
        self.assertEqual(P.stamps_from_store(self.e, r2, "shadow"), [STAMP])

    def test_complete_stamps_reports_per_frame_maxzoom(self):
        # F3 regression: frames cut at different pyramid_px have different maxzoom;
        # complete_stamps reports each, so the runner keeps only the current
        # geometry in the manifest (mixing would 404 old frames at deep zoom).
        r2 = FakeR2()
        P.emit_pyramid(self.e, r2, "shadow", STAMP, _gradient(600, 1024), BOUNDS, FAST)    # maxzoom 1
        P.emit_pyramid(self.e, r2, "shadow", STAMP2, _gradient(1200, 2048), BOUNDS, FAST)  # maxzoom 2
        self.assertEqual(dict(P.complete_stamps(self.e, r2, "shadow")), {STAMP: 1, STAMP2: 2})
        kept = [s for s, mz in P.complete_stamps(self.e, r2, "shadow") if mz == 2]
        self.assertEqual(kept, [STAMP2])   # only current-geometry frames advertised


class TestRegistryTiled(unittest.TestCase):
    def test_new_tiled_rows_present(self):
        for pid in ("goes19-conus-ir", "goes19-fd-ir"):
            e = R.REGISTRY_BY_ID[pid]
            self.assertTrue(e.tiled)
            self.assertEqual(e.tile_size, 512)
            self.assertIsNotNone(e.sector_bbox)
            self.assertIsNone(e.prod_meso_slug)   # new product, reference-render gated

    def test_tile_key_and_template(self):
        e = R.REGISTRY_BY_ID["goes19-conus-ir"]
        self.assertEqual(e.tile_key("shadow", STAMP, 3, 4, 2),
                         f"shadow/sat/goes19/conus/ir/{STAMP}/3/4/2.webp")
        self.assertEqual(e.tile_template(), "sat/goes19/conus/ir/{t}/{z}/{x}/{y}.webp")
        self.assertEqual(e.tile_stamp_prefix("shadow", STAMP),
                         f"shadow/sat/goes19/conus/ir/{STAMP}/")

    def test_stamp_from_tile_key_roundtrip_and_rejects(self):
        e = R.REGISTRY_BY_ID["goes19-conus-ir"]
        self.assertEqual(e.stamp_from_tile_key(e.tile_key("shadow", STAMP, 3, 4, 2)), STAMP)
        self.assertIsNone(e.stamp_from_tile_key(e.latest_times_key("shadow")))
        self.assertIsNone(e.stamp_from_tile_key(e.ready_key("shadow", STAMP)))  # marker != tile
        self.assertIsNone(e.stamp_from_tile_key("shadow/sat/goes19/conus/ir/health.json"))
        self.assertIsNone(e.stamp_from_tile_key(   # non-numeric z/x/y is not a tile
            f"shadow/sat/goes19/conus/ir/{STAMP}/a/b/c.webp"))

    def test_ready_key_roundtrip_and_rejects(self):
        e = R.REGISTRY_BY_ID["goes19-conus-ir"]
        rk = e.ready_key("shadow", STAMP)
        self.assertEqual(rk, f"shadow/sat/goes19/conus/ir/{STAMP}/_ready.json")
        self.assertEqual(e.stamp_from_ready_key(rk), STAMP)
        self.assertIsNone(e.stamp_from_ready_key(e.tile_key("shadow", STAMP, 0, 0, 0)))
        self.assertIsNone(e.stamp_from_ready_key(e.latest_times_key("shadow")))

    def test_scheme_is_single_source(self):
        # F5 regression: manifest scheme comes from R.TILE_SCHEME (no dup literal),
        # and the dead constant is gone from s2_pyramid.
        e = R.REGISTRY_BY_ID["goes19-conus-ir"]
        lt = e.build_tiled_latest_times([STAMP], bounds=BOUNDS, image_px=(4096, 1812),
                                        maxzoom=3, as_of=dt.datetime(2026, 7, 7, tzinfo=UTC))
        self.assertEqual(lt["scheme"], R.TILE_SCHEME)
        self.assertFalse(hasattr(P, "TILE_SCHEME"))

    def test_routing_no_cross_substrate_leak(self):
        CMIPF_C13 = ("ABI-L2-CMIPF/2026/001/00/"
                     "OR_ABI-L2-CMIPF-M6C13_G19_s20260010000226_e20260010009546_c20260010010005.nc")
        MCMIPF = ("ABI-L2-MCMIPF/2026/001/00/"
                  "OR_ABI-L2-MCMIPF-M6_G19_s20260010000226_e20260010009546_c20260010010005.nc")
        self.assertEqual({e.product_id for e in R.matching_entries(R.parse_key(CMIPF_C13))},
                         {"goes19-fd-ir"})            # C13 FD -> the new tiled IR product
        self.assertEqual({e.product_id for e in R.matching_entries(R.parse_key(MCMIPF))},
                         {"goes19-fd-mcmip"})         # MCMIP still only the mcmip product


if __name__ == "__main__":
    unittest.main()
