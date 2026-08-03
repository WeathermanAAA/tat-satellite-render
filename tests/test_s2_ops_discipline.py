#!/usr/bin/env python3
"""Ops-discipline tests (2026-08-03, the R2 Class A cost incident).

The emitter's listing volume -- NOT its tile PUTs -- was ~65% of a ~10.7M
ops/day bill: min(retention,300) stamp probes per product-pass, re-listed
per slot, plus a products.json rebuild probing every sibling product on
every pass. These tests pin the replacement behavior:

  * complete_stamps(after=...) tail-bounds the stamp listing (StartAfter);
  * the backfill loops re-check coverage IN MEMORY from each emit's returned
    stamp (exactly one _covered_times listing per pass);
  * the per-product manifest is maintained INCREMENTALLY (one GET + append),
    with the full rebuild-from-R2 reserved for cold/geometry/heal passes --
    and the geometry refuse-to-clobber guard preserved verbatim;
  * a duplicate frame already advertised writes NOTHING;
  * products.json rebuilds are gated on a frame actually landing and
    throttled per sector.

Every test is hermetic: FilesystemStore under a tempdir, synthetic imagery,
STATE_DIR patched into the sandbox.
"""
import dataclasses
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_pyramid as P            # noqa: E402
import s2_pyramid_emit as E       # noqa: E402
import s2_registry as R           # noqa: E402

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# _manifest_action: the pure decision
# ---------------------------------------------------------------------------
class TestManifestAction(unittest.TestCase):
    CUR = {"scheme": "flat-native-xyz", "maxzoom": 5,
           "times": ["20260803T100000Z"]}

    def test_no_manifest_rebuilds(self):
        self.assertEqual(E._manifest_action(None, "s", 5, "flat-native-xyz",
                                            heal_due=False), "rebuild")

    def test_scheme_mismatch_rebuilds(self):
        self.assertEqual(E._manifest_action(self.CUR, "s", 5,
                                            "webmercator-xyz",
                                            heal_due=False), "rebuild")

    def test_heal_due_rebuilds(self):
        self.assertEqual(E._manifest_action(self.CUR, "20260803T101000Z", 5,
                                            "flat-native-xyz",
                                            heal_due=True), "rebuild")

    def test_same_geometry_new_stamp_appends(self):
        self.assertEqual(E._manifest_action(self.CUR, "20260803T101000Z", 5,
                                            "flat-native-xyz",
                                            heal_due=False), "append")

    def test_rendered_stamp_already_advertised_skips(self):
        self.assertEqual(E._manifest_action(self.CUR, "20260803T100000Z", 5,
                                            "flat-native-xyz",
                                            heal_due=False), "skip")

    def test_geometry_change_rebuilds(self):
        # the z5-cron/z6-on-demand clobber trap routes through the guard path
        self.assertEqual(E._manifest_action(self.CUR, "20260803T101000Z", 6,
                                            "flat-native-xyz",
                                            heal_due=False), "rebuild")

    def test_duplicate_advertised_skips_but_unknown_rebuilds(self):
        # frame_mz None == duplicate (geometry not re-derived): appending a
        # frame of unknown geometry is forbidden
        self.assertEqual(E._manifest_action(self.CUR, "20260803T100000Z",
                                            None, "flat-native-xyz",
                                            heal_due=False), "skip")
        self.assertEqual(E._manifest_action(self.CUR, "20260803T101000Z",
                                            None, "flat-native-xyz",
                                            heal_due=False), "rebuild")


# ---------------------------------------------------------------------------
# complete_stamps(after=...): the StartAfter tail bound
# ---------------------------------------------------------------------------
class TestAfterBound(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = P.FilesystemStore(self.tmp)
        self.entry = next(e for e in R.REGISTRY if e.tiled)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _frame(self, stamp):
        self.store.put_bytes(self.entry.tile_key("shadow", stamp, 0, 0, 0),
                             b"x", "image/webp", "c")
        self.store.put_json(self.entry.ready_key("shadow", stamp),
                            {"maxzoom": 0}, "c")

    def test_after_excludes_older_stamps(self):
        self._frame("20260801T000000Z")
        self._frame("20260803T000000Z")
        self._frame("20260803T120000Z")
        allf = P.complete_stamps(self.entry, self.store, "shadow")
        self.assertEqual(len(allf), 3)
        tail = P.complete_stamps(self.entry, self.store, "shadow",
                                 after="20260802T000000Z")
        self.assertEqual([s for s, _ in tail],
                         ["20260803T000000Z", "20260803T120000Z"])

    def test_covered_times_passes_window(self):
        self._frame("20260801T000000Z")
        recent = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._frame(recent)
        out = E._covered_times([self.entry], self.store, "shadow",
                               window_min=60)
        self.assertEqual(len(out[0]), 1)     # only the recent frame probed


# ---------------------------------------------------------------------------
# emit_one end-to-end against FilesystemStore: incremental / skip / guard /
# heal. Imagery is synthetic; scheme forced flat-native so the cut is tiny.
# ---------------------------------------------------------------------------
class _Img:
    def __init__(self, stamp, px=256):
        self.rgba = np.full((px, px, 4), 200, dtype=np.uint8)
        self.bounds = (-125.0, 15.0, -66.0, 49.0)
        self.stamp = stamp
        self.product = "test"
        self.s3_key = "a/b"
        self.bt_grid = None


def _args(tmp, keep=5):
    return type("A", (), {
        "prefix": "shadow", "quality": 80, "scheme": "flat-native-xyz",
        "max_zoom": None, "keep": keep, "allow_geometry_change": False,
        "store": f"local:{tmp}", "step": None, "backfill": None,
    })()


class TestEmitOneManifestDiscipline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.state = tempfile.mkdtemp()
        self.store = P.FilesystemStore(self.tmp)
        self.entry = next(e for e in R.REGISTRY
                          if e.tiled and e.sector_key == "conus")
        self.args = _args(self.tmp)
        patches = [
            mock.patch.object(E, "STATE_DIR", self.state),
            mock.patch.object(E.I, "produce_imagery",
                              side_effect=self._imagery),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.next_img = None

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.state, ignore_errors=True)

    def _imagery(self, entry, time=None, nearest=True, band_cache=None):
        return self.next_img

    def _emit(self, stamp, px=256, counts=None):
        self.next_img = _Img(stamp, px=px)
        with mock.patch.object(E.P, "complete_stamps",
                               wraps=P.complete_stamps) as cs:
            res = E.emit_one(self.entry, None, self.store, self.args)
        if counts is not None:
            counts.append(cs.call_count)
        return res

    def _manifest(self):
        import json
        raw = self.store.get_bytes(self.entry.latest_times_key("shadow"))
        return json.loads(raw)

    def test_cold_then_incremental_then_dup_skip(self):
        calls = []
        self._emit("20260803T100000Z", counts=calls)
        self.assertEqual(calls[-1], 1)               # cold: one rebuild listing
        m = self._manifest()
        self.assertEqual(m["times"], ["20260803T100000Z"])

        self._emit("20260803T101000Z", counts=calls)
        self.assertEqual(calls[-1], 0)               # incremental: NO listing
        m = self._manifest()
        self.assertEqual(m["times"],
                         ["20260803T100000Z", "20260803T101000Z"])

        # duplicate (ready marker present) already advertised: no listing AND
        # no manifest rewrite
        before = os.stat(os.path.join(
            self.tmp, self.entry.latest_times_key("shadow"))).st_mtime_ns
        res = self._emit("20260803T101000Z", counts=calls)
        self.assertEqual(res["outcome"], "duplicate")
        self.assertEqual(calls[-1], 0)
        after = os.stat(os.path.join(
            self.tmp, self.entry.latest_times_key("shadow"))).st_mtime_ns
        self.assertEqual(before, after)

    def test_keep_trims_incremental_manifest(self):
        self.args = _args(self.tmp, keep=3)
        for i in range(5):
            self._emit(f"20260803T10{i}000Z")
        m = self._manifest()
        self.assertEqual(len(m["times"]), 3)
        self.assertEqual(m["times"][-1], "20260803T104000Z")

    def test_geometry_change_still_refuses(self):
        # the incremental path must NOT bypass the refuse-to-clobber guard: a
        # different pyramid geometry routes to the rebuild path and raises
        self._emit("20260803T100000Z", px=256)               # maxzoom 0
        with self.assertRaises(RuntimeError):
            self._emit("20260803T101000Z", px=2048)          # maxzoom 2
        # ...and --allow-geometry-change still permits the migration
        self.args.allow_geometry_change = True
        self._emit("20260803T102000Z", px=2048)
        m = self._manifest()
        self.assertNotIn("20260803T100000Z", m["times"])     # old geometry out

    def test_heal_tick_forces_rebuild(self):
        calls = []
        self._emit("20260803T100000Z", counts=calls)
        # a frame completed by an interrupted run: marker present, manifest
        # never told (simulates a kill between marker PUT and manifest PUT)
        orphan = "20260803T100500Z"
        self.store.put_bytes(self.entry.tile_key("shadow", orphan, 0, 0, 0),
                             b"x", "image/webp", "c")
        self.store.put_json(self.entry.ready_key("shadow", orphan),
                            {"maxzoom": 0}, "c")
        self._emit("20260803T101000Z", counts=calls)
        self.assertEqual(calls[-1], 0)                       # not heal time yet
        self.assertNotIn(orphan, self._manifest()["times"])
        # age the heal stamp past the window -> next emit reconciles
        heal = E._state_path("heal", "shadow", self.entry.product_id)
        old = dt.datetime.now(UTC).timestamp() - E.MANIFEST_HEAL_S - 60
        os.utime(heal, (old, old))
        self._emit("20260803T102000Z", counts=calls)
        self.assertEqual(calls[-1], 1)                       # heal: one listing
        self.assertIn(orphan, self._manifest()["times"])


# ---------------------------------------------------------------------------
# The backfill loop lists coverage exactly ONCE per pass
# ---------------------------------------------------------------------------
class TestSingleListingPerPass(unittest.TestCase):
    def test_product_pass_lists_once(self):
        entry = next(e for e in R.REGISTRY
                     if e.tiled and e.sector_key == "conus")
        stamps = []

        def fake_emit(entry, when, store, args, band_cache=None):
            s = when.strftime("%Y%m%dT%H%M%SZ")
            stamps.append(s)
            return {"stamp": s, "outcome": "rendered", "manifest": {}}

        with mock.patch.object(E, "emit_one", side_effect=fake_emit), \
             mock.patch.object(E.P, "complete_stamps",
                               return_value=[]) as cs, \
             mock.patch.object(E, "_write_products_index"):
            rc = E.main(["--product", entry.product_id,
                         "--store", "local:/tmp/nope",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        self.assertGreaterEqual(len(stamps), 2)   # several slots emitted...
        self.assertEqual(cs.call_count, 1)        # ...ONE coverage listing


# ---------------------------------------------------------------------------
# products.json gating: idle passes skip, rendered passes throttle per sector
# ---------------------------------------------------------------------------
class TestIndexGating(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp()
        p = mock.patch.object(E, "STATE_DIR", self.state)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.state, True)

    def test_idle_pass_writes_no_index(self):
        entry = next(e for e in R.REGISTRY
                     if e.tiled and e.sector_key == "conus")
        now = dt.datetime.now(UTC)
        covered = [[now - dt.timedelta(minutes=10 * i)] for i in range(1)]

        with mock.patch.object(E, "_covered_times",
                               return_value=[[now - dt.timedelta(minutes=m)
                                             for m in range(0, 45, 5)]]), \
             mock.patch.object(E, "emit_one") as eo, \
             mock.patch.object(E, "_write_products_index") as wi:
            rc = E.main(["--product", entry.product_id,
                         "--store", "local:/tmp/nope",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        eo.assert_not_called()
        wi.assert_not_called()

    def test_throttle_second_write_within_window(self):
        puts = {}

        class _Store:
            def put_json(self, key, obj, cache):
                puts[key] = obj
                return True

        with mock.patch.object(E.P, "has_complete_frame", return_value=True):
            k1 = E._write_products_index(_Store(), "shadow", "goes19", "conus")
            k2 = E._write_products_index(_Store(), "shadow", "goes19", "conus")
            k3 = E._write_products_index(_Store(), "shadow", "goes19", "conus",
                                         force=True)
        self.assertIsNotNone(k1)
        self.assertIsNone(k2)          # throttled
        self.assertIsNotNone(k3)       # force bypasses


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Review fixes (2026-08-03 adversarial panel)
# ---------------------------------------------------------------------------
class TestStalePinProbe(unittest.TestCase):
    """A pin OLDER than the coverage window (upstream stall: GK-2A ~2 h,
    FCI 6 h) must dedup against its ready marker BEFORE any fetch."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = P.FilesystemStore(self.tmp)
        self.entry = next(e for e in R.REGISTRY if e.tiled)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_marker_hit_appends_to_covered(self):
        pinned = dt.datetime.now(UTC) - dt.timedelta(hours=3)
        stamp = pinned.strftime("%Y%m%dT%H%M%SZ")
        self.store.put_json(self.entry.ready_key("shadow", stamp),
                            {"maxzoom": 5}, "c")
        covered = [[]]
        E._probe_stale_pin([self.entry], covered, pinned, self.store,
                           "shadow", window_min=110)
        self.assertEqual(covered[0], [pinned])
        self.assertFalse(E._slot_missing(pinned, covered,
                                         dt.timedelta(seconds=90)))

    def test_fresh_pin_costs_nothing(self):
        pinned = dt.datetime.now(UTC) - dt.timedelta(minutes=5)
        covered = [[]]
        heads = []
        self.store.head = lambda k: heads.append(k)      # would explode count
        E._probe_stale_pin([self.entry], covered, pinned, self.store,
                           "shadow", window_min=110)
        self.assertEqual(heads, [])                       # in-window: no probe

    def test_marker_miss_leaves_slot_missing(self):
        pinned = dt.datetime.now(UTC) - dt.timedelta(hours=3)
        covered = [[]]
        E._probe_stale_pin([self.entry], covered, pinned, self.store,
                           "shadow", window_min=110)
        self.assertTrue(E._slot_missing(pinned, covered,
                                        dt.timedelta(seconds=90)))


class TestReconcileManifest(unittest.TestCase):
    """Marker-complete frames missing from times[] (kill or failed PUT
    between marker and manifest) re-advertise within ONE pass."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = P.FilesystemStore(self.tmp)
        self.entry = next(e for e in R.REGISTRY if e.tiled)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _frame(self, stamp, maxzoom=5):
        self.store.put_bytes(
            self.entry.tile_key("shadow", stamp, maxzoom, 0, 0),
            b"x", "image/webp", "c")
        self.store.put_json(self.entry.ready_key("shadow", stamp),
                            {"maxzoom": maxzoom}, "c")

    def _manifest(self, times, maxzoom=5):
        lt = self.entry.build_tiled_latest_times(
            times, bounds=[-125.0, 15.0, -66.0, 49.0], image_px=[256, 256],
            maxzoom=maxzoom, as_of=dt.datetime.now(UTC))
        self.store.put_json(self.entry.latest_times_key("shadow"), lt, "c")

    def test_orphan_readvertised_in_one_pass(self):
        t1 = dt.datetime(2026, 8, 3, 10, 0, tzinfo=UTC)
        t2 = dt.datetime(2026, 8, 3, 10, 10, tzinfo=UTC)
        s1, s2 = (t.strftime("%Y%m%dT%H%M%SZ") for t in (t1, t2))
        self._frame(s1)
        self._frame(s2)
        self._manifest([s1])                     # s2 is the orphan
        E._reconcile_manifest(self.entry, self.store, "shadow", [t1, t2], 90)
        import json as _j
        cur = _j.loads(self.store.get_bytes(
            self.entry.latest_times_key("shadow")))
        self.assertEqual(cur["times"], [s1, s2])

    def test_foreign_geometry_orphan_left_for_heal(self):
        t1 = dt.datetime(2026, 8, 3, 10, 0, tzinfo=UTC)
        t2 = dt.datetime(2026, 8, 3, 10, 10, tzinfo=UTC)
        s1, s2 = (t.strftime("%Y%m%dT%H%M%SZ") for t in (t1, t2))
        self._frame(s1, maxzoom=5)
        self._frame(s2, maxzoom=7)               # z7 stray: NOT reconciled
        self._manifest([s1], maxzoom=5)
        E._reconcile_manifest(self.entry, self.store, "shadow", [t1, t2], 90)
        import json as _j
        cur = _j.loads(self.store.get_bytes(
            self.entry.latest_times_key("shadow")))
        self.assertEqual(cur["times"], [s1])

    def test_no_manifest_is_a_noop(self):
        t1 = dt.datetime(2026, 8, 3, 10, 0, tzinfo=UTC)
        self._frame(t1.strftime("%Y%m%dT%H%M%SZ"))
        E._reconcile_manifest(self.entry, self.store, "shadow", [t1], 90)
        self.assertIsNone(self.store.get_bytes(
            self.entry.latest_times_key("shadow")))


class TestCheckedManifestPut(unittest.TestCase):
    def test_failed_manifest_put_raises(self):
        entry = next(e for e in R.REGISTRY if e.tiled)

        class _BrokenStore:
            def put_json(self, key, obj, cache):
                return False

        with self.assertRaises(IOError):
            P.write_tiled_manifest(entry, _BrokenStore(), "shadow",
                                   ["20260803T100000Z"],
                                   [-125.0, 15.0, -66.0, 49.0], [256, 256],
                                   5, dt.datetime.now(UTC))


class TestHorizonFilter(unittest.TestCase):
    def test_append_drops_prune_horizon_stamps(self):
        # a pre-prune GET carrying ancient stamps must not resurrect them
        old = "20240101T000000Z"
        self.assertLess(old, E._horizon_stamp())
        fresh = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.assertGreater(fresh, E._horizon_stamp())
