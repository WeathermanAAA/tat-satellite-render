#!/usr/bin/env python3
"""Coverage-from-manifest tests (2026-08-25, change C of the R2 cost audit).

The --step loops used to learn "which slots already have a complete frame"
by LISTing every stamp in the window -- one Class A request per stamp,
every pass, idle or not (~378K of the fleet's ~385K ListObjects/day). They
now read it from the product's own writer-maintained latest_times.json with
ONE Class B GET (s2_pyramid_emit._covered_from_manifest via _pass_coverage),
falling back to the listing whenever the manifest cannot stand in, and
taking the listing once per S2_COVER_VERIFY_S per product as a VERIFY.
These tests pin that contract:

  (a) manifest-derived covered == listing-derived covered on a store holding
      complete frames + a matching manifest;
  (b) a missing / invalid / not-reaching-past-the-window manifest falls back
      to the listing, with the reason in the log line (both call sites);
  (c) an orphan (marker present, absent from times[]) is never rendered
      twice and is advertised after ONE pass -- through the REAL emit_one /
      emit_pyramid / _manifest_action path, not a mock -- for one GET + one
      PUT, no listing;
  (c') a STRAY (complete at a foreign geometry) never flips the manifest's
      geometry, is memoed, and costs one fetch per container lifetime;
  (d) an idle product pass on the manifest path costs 1 GET, 0 LIST pages;
  (e) S2_COVER_FROM_MANIFEST=0 takes the listing path unconditionally;
  (v) the periodic verify: a fresh container lists on its first pass, an
      expired gate lists again, and the verify pass is what heals an orphan
      whose slot fell off the grid and re-renders a frame deleted under a
      live manifest (the two things the manifest path cannot see).

Hermetic: a counting FilesystemStore under a tempdir (the s1_ingest.R2
`ops` convention, one page per list call), synthetic imagery, STATE_DIR
patched into the sandbox, the clock pinned the way tests/test_s2_backfill
pins it. Stamps are derived from the pinned clock so the tests do not rot
past the prune horizon the way fixed-date stamps do.
"""
import contextlib
import datetime as dt
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_pyramid as P            # noqa: E402
import s2_pyramid_emit as E       # noqa: E402
import s2_registry as R           # noqa: E402

UTC = dt.timezone.utc
FMT = "%Y%m%dT%H%M%SZ"
NOW = dt.datetime(2026, 8, 25, 10, 7, 42, tzinfo=UTC)
BOUNDS = [-125.0, 15.0, -66.0, 49.0]
PREFIX = "shadow"
SCHEME = "flat-native-xyz"
# the hwfd-leads lane's geometry: --step 10 --backfill 60 -> window 80 min
WINDOW = 80


def _stamp(t):
    return t.strftime(FMT)


def _ago(minutes, now=NOW):
    return now - dt.timedelta(minutes=minutes)


class CountingStore(P.FilesystemStore):
    """FilesystemStore + the s1_ingest.R2 request counters. One LIST page per
    list call (the fake never paginates, so the count is a LOWER bound of
    what R2 would bill -- the safe direction for a zero-assertion)."""

    def __init__(self, root):
        super().__init__(root)
        self.ops = {}
        self.put_log = []

    def _op(self, name):
        self.ops[name] = self.ops.get(name, 0) + 1

    def put_bytes(self, key, data, content_type, cache):
        self._op("put")
        self.put_log.append(key)
        return super().put_bytes(key, data, content_type, cache)

    def head(self, key):
        self._op("head")
        return super().head(key)

    def get_bytes(self, key):
        self._op("get")
        return super().get_bytes(key)

    def list_keys(self, prefix):
        self._op("list_pages")
        return super().list_keys(prefix)

    def list_prefixes(self, prefix, start_after=""):
        self._op("list_pages")
        return super().list_prefixes(prefix, start_after)

    def list_level(self, prefix):
        self._op("list_pages")
        return super().list_level(prefix)


def _frame(store, entry, stamp, maxzoom=0, marker=None):
    store.put_bytes(entry.tile_key(PREFIX, stamp, maxzoom, 0, 0), b"x",
                    "image/webp", "c")
    store.put_json(entry.ready_key(PREFIX, stamp),
                   {"maxzoom": maxzoom} if marker is None else marker, "c")


def _manifest(store, entry, stamps, maxzoom=0, scheme=SCHEME):
    lt = entry.build_tiled_latest_times(
        stamps, bounds=BOUNDS, image_px=[256, 256], maxzoom=maxzoom,
        as_of=NOW, scheme=scheme)
    store.put_json(entry.latest_times_key(PREFIX), lt, "c")
    return lt


@contextlib.contextmanager
def _clock(now=NOW):
    """Pin every dt.datetime.now() the emitter takes (the test_s2_backfill
    pattern); strptime/timedelta pass through untouched."""
    with mock.patch.object(E.dt, "datetime", wraps=dt.datetime) as md:
        md.now.return_value = now
        yield


def _conus_entry():
    return next(e for e in R.REGISTRY if e.tiled and e.sector_key == "conus")


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s2cover-")
        self.state = tempfile.mkdtemp(prefix="s2state-")
        self.store = CountingStore(self.tmp)
        self.entry = _conus_entry()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(shutil.rmtree, self.state, True)
        # E.main builds its own store from --store; hand it THIS counting
        # store so the pass's ops land in self.store.ops
        for p in (mock.patch.object(E, "_make_store",
                                    return_value=(self.store, False)),
                  mock.patch.object(E, "STATE_DIR", self.state)):
            p.start()
            self.addCleanup(p.stop)
        # the periodic verify gate is FRESH for the product under test: these
        # tests exercise the manifest path (TestPeriodicVerify owns the gate)
        self._verified(self.entry)

    def _verified(self, *entries):
        for e in entries:
            E._state_touch(E._state_path("cover", PREFIX, e.product_id))

    def _expire(self, *entries):
        for e in entries:
            g = E._state_path("cover", PREFIX, e.product_id)
            old = time.time() - E.COVER_VERIFY_S - 60
            os.utime(g, (old, old))

    def _times(self):
        return json.loads(self.store.get_bytes(
            self.entry.latest_times_key(PREFIX)))["times"]

    def _live(self):
        return json.loads(self.store.get_bytes(
            self.entry.latest_times_key(PREFIX)))


# ---------------------------------------------------------------------------
# (a) manifest-derived == listing-derived
# ---------------------------------------------------------------------------
class TestManifestEqualsListing(_Base):
    def test_same_covered_set_inside_the_window(self):
        inside = [_ago(5), _ago(25), _ago(55)]
        outside = [_ago(95), _ago(6 * 60)]
        for t in inside + outside:
            _frame(self.store, self.entry, _stamp(t))
        _manifest(self.store, self.entry,
                  [_stamp(t) for t in inside + outside])
        with _clock():
            man, reason = E._covered_from_manifest(
                [self.entry], self.store, PREFIX, WINDOW)
            lst = E._covered_times([self.entry], self.store, PREFIX,
                                   window_min=WINDOW)
        self.assertEqual(reason, "manifest")
        self.assertEqual(man, lst)
        self.assertEqual(man[0], sorted(inside))

    def test_multi_entry_shape_matches(self):
        stamps = [_stamp(_ago(m)) for m in (5, 15, 95)]
        for s in stamps:
            _frame(self.store, self.entry, s)
        _manifest(self.store, self.entry, stamps)
        with _clock():
            man, _ = E._covered_from_manifest(
                [self.entry, self.entry], self.store, PREFIX, WINDOW)
            lst = E._covered_times([self.entry, self.entry], self.store,
                                   PREFIX, window_min=WINDOW)
        self.assertEqual(len(man), 2)
        self.assertEqual(man, lst)

    def test_manifest_path_costs_one_get_and_no_listing(self):
        stamps = [_stamp(_ago(m)) for m in (5, 15, 25, 95)]
        for s in stamps:
            _frame(self.store, self.entry, s)
        _manifest(self.store, self.entry, stamps)
        self.store.ops = {}
        with _clock():
            E._covered_from_manifest([self.entry], self.store, PREFIX,
                                     WINDOW)
        self.assertEqual(self.store.ops, {"get": 1})
        self.store.ops = {}
        with _clock():
            lst = E._covered_times([self.entry], self.store, PREFIX,
                                   window_min=WINDOW)
        self.assertEqual(self.store.ops.get("get", 0), 0)
        # the listing path: one tail listing + one probe per stamp listed
        self.assertGreaterEqual(self.store.ops["list_pages"],
                                1 + len(lst[0]))


# ---------------------------------------------------------------------------
# (b) fallback reasons + the log line at both call sites
# ---------------------------------------------------------------------------
class TestFallbackReasons(_Base):
    def _cover(self):
        with _clock():
            return E._covered_from_manifest([self.entry], self.store, PREFIX,
                                            WINDOW)

    def test_missing(self):
        self.assertEqual(self._cover(),
                         (None, f"{self.entry.product_id} (missing)"))

    def test_unparseable_json(self):
        self.store.put_bytes(self.entry.latest_times_key(PREFIX),
                             b"{not json", "application/json", "c")
        cov, reason = self._cover()
        self.assertIsNone(cov)
        self.assertIn("unparseable JSON", reason)

    def test_no_scheme_or_maxzoom(self):
        self.store.put_json(self.entry.latest_times_key(PREFIX),
                            {"times": [_stamp(_ago(5))], "maxzoom": 0}, "c")
        cov, reason = self._cover()
        self.assertIsNone(cov)
        self.assertIn("no scheme/maxzoom", reason)

    def test_times_not_a_list(self):
        self.store.put_json(self.entry.latest_times_key(PREFIX),
                            {"scheme": SCHEME, "maxzoom": 0, "times": "x"},
                            "c")
        cov, reason = self._cover()
        self.assertIsNone(cov)
        self.assertIn("times[] is not a list", reason)

    def test_unparseable_stamp(self):
        self.store.put_json(self.entry.latest_times_key(PREFIX),
                            {"scheme": SCHEME, "maxzoom": 0,
                             "times": ["garbage", _stamp(_ago(5))]}, "c")
        cov, reason = self._cover()
        self.assertIsNone(cov)
        self.assertIn("unparseable stamp", reason)

    def test_not_reaching_past_the_window_is_refused(self):
        # every stamp INSIDE the window (a cold product, a keep-trimmed
        # manifest, or the one-frame manifest a geometry migration's rebuild
        # leaves): it cannot prove that an earlier slot is missing --
        # review finding 2026-08-25 (the migration flip-flop): the old
        # `len(times) < keep` acceptance trusted exactly this manifest
        stamps = [_stamp(_ago(m)) for m in (5, 15, 25)]
        _manifest(self.store, self.entry, stamps)
        cov, reason = self._cover()
        self.assertIsNone(cov)
        self.assertTrue(reason.startswith(
            f"{self.entry.product_id} (does not reach past the window"),
            reason)

    def test_reaching_past_the_window_is_valid(self):
        stamps = [_stamp(_ago(m)) for m in (5, 25, 95)]
        _manifest(self.store, self.entry, stamps)
        cov, reason = self._cover()
        self.assertEqual(reason, "manifest")
        self.assertEqual(cov, [[_ago(25), _ago(5)]])   # 95 min is outside

    def test_oldest_exactly_at_the_window_start_is_valid(self):
        stamps = [_stamp(_ago(5)), _stamp(_ago(WINDOW))]
        _manifest(self.store, self.entry, stamps)
        cov, reason = self._cover()
        self.assertEqual(reason, "manifest")
        self.assertEqual(cov, [[_ago(WINDOW), _ago(5)]])   # inclusive bound

    def test_empty_times_is_refused(self):
        _manifest(self.store, self.entry, [])
        cov, reason = self._cover()
        self.assertIsNone(cov)
        self.assertIn("does not reach past the window: 0 stamps", reason)

    def test_all_stamps_older_than_the_window_is_valid_and_empty(self):
        # an upstream outage: the manifest reaches past the window but
        # nothing in it is young enough to cover a slot
        _manifest(self.store, self.entry, [_stamp(_ago(m)) for m in (95, 200)])
        self.assertEqual(self._cover(), ([[]], "manifest"))

    def test_store_without_get_bytes_falls_back(self):
        class _Flat:                          # the FlatR2 fake shape
            pass
        cov, reason = E._covered_from_manifest([self.entry], _Flat(), PREFIX,
                                               WINDOW)
        self.assertIsNone(cov)
        self.assertIn("no get_bytes", reason)

    def test_no_window_falls_back(self):
        cov, reason = E._covered_from_manifest([self.entry], self.store,
                                               PREFIX, None)
        self.assertIsNone(cov)
        self.assertIn("no coverage window", reason)

    def test_product_pass_falls_back_and_logs(self):
        # no manifest in the store: the pass takes the listing path exactly
        # once, reconciles, and says why
        all_slots = [[_ago(m) for m in range(0, 45, 5)]]
        buf = io.StringIO()
        with mock.patch.object(E, "_covered_times",
                               return_value=all_slots) as ct, \
             mock.patch.object(E, "_reconcile_manifest") as rm, \
             mock.patch.object(E, "emit_one") as eo, \
             mock.patch.object(E, "_write_products_index"), \
             _clock(), contextlib.redirect_stdout(buf):
            rc = E.main(["--product", self.entry.product_id,
                         "--store", f"local:{self.tmp}",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        self.assertEqual(ct.call_count, 1)
        self.assertEqual(rm.call_count, 1)
        eo.assert_not_called()
        self.assertIn(f"[cover] manifest unusable for "
                      f"{self.entry.product_id} (missing) -- listing fallback",
                      buf.getvalue())

    def test_migration_leftover_lists_and_stays_idle(self):
        # review finding 2026-08-25 (R1/R7): B, C complete at maxzoom 0 (the
        # OLD geometry), A at maxzoom 1 (NEW); the migration's rebuild left
        # manifest {maxzoom 1, [A]}. The manifest cannot vouch for B and C,
        # so the pass lists (which counts a complete frame at ANY maxzoom),
        # renders NOTHING and leaves the manifest exactly as it was -- the
        # old acceptance rule fetched B and C and flipped the manifest to
        # maxzoom 0 (dropping A), then back, every pass for a window
        A, B, C = (_stamp(_ago(7)), _stamp(_ago(17)), _stamp(_ago(27)))
        _frame(self.store, self.entry, A, maxzoom=1)
        _frame(self.store, self.entry, B, maxzoom=0)
        _frame(self.store, self.entry, C, maxzoom=0)
        _manifest(self.store, self.entry, [A], maxzoom=1)
        before = self._live()
        fetched = []
        with mock.patch.object(E.I, "produce_imagery",
                               side_effect=lambda *a, **k: fetched.append(1)), \
             mock.patch.object(E, "_write_products_index"), \
             _clock(), contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = E.main(["--product", self.entry.product_id,
                         "--store", f"local:{self.tmp}",
                         "--allow-geometry-change",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        self.assertEqual(fetched, [])
        self.assertEqual(self._live(), before)
        self.assertIn("(does not reach past the window", buf.getvalue())

    def test_suite_pass_falls_back_when_one_member_is_unusable(self):
        # 3 mtgi1-fd products: two valid manifests, one missing -> the WHOLE
        # call falls back, the line names the culprit, reconcile runs per entry
        entries = [e for e in R.REGISTRY
                   if e.tiled and e.sat_key == "mtgi1" and e.sector_key == "fd"]
        self.assertEqual(len(entries), 3)
        self._verified(*entries)
        stamps = [_stamp(_ago(m)) for m in list(range(0, 45, 5)) + [95]]
        for e in entries[:2]:
            _manifest(self.store, e, stamps)
        all_slots = [[_ago(m) for m in range(0, 45, 5)] for _ in entries]
        buf = io.StringIO()
        with mock.patch.object(E, "_covered_times",
                               return_value=all_slots) as ct, \
             mock.patch.object(E, "_reconcile_manifest") as rm, \
             mock.patch.object(E, "_pin_suite_scan", return_value=_ago(3)), \
             mock.patch.object(E, "emit_one") as eo, \
             mock.patch.object(E, "_write_products_index"), \
             _clock(), contextlib.redirect_stdout(buf):
            rc = E.main(["--suite", "mtgi1-fd", "--store", f"local:{self.tmp}",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        self.assertEqual(ct.call_count, 1)
        self.assertEqual(rm.call_count, 3)
        eo.assert_not_called()
        self.assertIn(f"[cover] manifest unusable for "
                      f"{entries[2].product_id} (missing) -- listing fallback",
                      buf.getvalue())
        self.assertNotIn(": manifest (", buf.getvalue())   # no success lines

    def test_suite_pass_on_manifest_path_lists_nothing(self):
        entries = [e for e in R.REGISTRY
                   if e.tiled and e.sat_key == "mtgi1" and e.sector_key == "fd"]
        self._verified(*entries)
        stamps = [_stamp(_ago(m)) for m in list(range(0, 45, 5)) + [95]]
        for e in entries:
            _manifest(self.store, e, stamps)
        self.store.ops = {}
        buf = io.StringIO()
        with mock.patch.object(E, "_covered_times") as ct, \
             mock.patch.object(E, "_reconcile_manifest") as rm, \
             mock.patch.object(E, "_pin_suite_scan", return_value=_ago(3)), \
             mock.patch.object(E, "emit_one") as eo, \
             mock.patch.object(E, "_write_products_index"), \
             _clock(), contextlib.redirect_stdout(buf):
            rc = E.main(["--suite", "mtgi1-fd", "--store", f"local:{self.tmp}",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        ct.assert_not_called()
        rm.assert_not_called()
        eo.assert_not_called()
        self.assertEqual(self.store.ops, {"get": 3})
        for e in entries:
            self.assertIn(f"[cover] {e.product_id}: manifest (9 stamps in "
                          f"window) get=1", buf.getvalue())


# ---------------------------------------------------------------------------
# (c) the orphan: one source fetch, ZERO tile PUTs, advertised after one pass
# ---------------------------------------------------------------------------
class _Img:
    def __init__(self, stamp, px=256):
        self.rgba = np.full((px, px, 4), 200, dtype=np.uint8)
        self.bounds = tuple(BOUNDS)
        self.stamp = stamp
        self.product = "test"
        self.s3_key = "a/b"
        self.bt_grid = None


class _EmitBase(_Base):
    """slots (step 10, backfill 30 @ 10:07:42): 10:00, 09:50, 09:40.
    A=10:00 complete + advertised; B=09:50 the frame under test (complete,
    marker present, NOT in times[]); C=09:40 genuinely missing; D=08:00
    complete + advertised so the manifest reaches past the 50 min window."""

    A, B, C, D = (_stamp(dt.datetime(2026, 8, 25, 10, 0, tzinfo=UTC)),
                  _stamp(dt.datetime(2026, 8, 25, 9, 50, tzinfo=UTC)),
                  _stamp(dt.datetime(2026, 8, 25, 9, 40, tzinfo=UTC)),
                  _stamp(dt.datetime(2026, 8, 25, 8, 0, tzinfo=UTC)))
    B_MAXZOOM = 0
    B_MARKER = None

    def setUp(self):
        super().setUp()
        self.fetched = []

        def _imagery(entry, time=None, nearest=True, band_cache=None):
            # "the source scan nearest the slot": the slot's own stamp
            self.fetched.append(_stamp(time))
            return _Img(_stamp(time))
        p2 = mock.patch.object(E.I, "produce_imagery", side_effect=_imagery)
        p2.start()
        self.addCleanup(p2.stop)
        _frame(self.store, self.entry, self.A)
        _frame(self.store, self.entry, self.D)
        _frame(self.store, self.entry, self.B, maxzoom=self.B_MAXZOOM,
               marker=self.B_MARKER)
        _manifest(self.store, self.entry, [self.D, self.A])
        # heal stamp FRESH: the manifest decision below must come from the
        # 'duplicate not advertised' branch, not heal_due
        E._state_touch(E._state_path("heal", PREFIX, self.entry.product_id))
        self.store.ops = {}
        self.store.put_log = []

    def _pass(self, extra=()):
        buf = io.StringIO()
        with mock.patch.object(E, "_write_products_index"), \
             mock.patch.object(E.P, "complete_stamps",
                               wraps=P.complete_stamps) as cs, \
             _clock(), contextlib.redirect_stdout(buf):
            rc = E.main(["--product", self.entry.product_id,
                         "--store", f"local:{self.tmp}",
                         "--scheme", SCHEME,
                         "--step", "10", "--backfill", "30"] + list(extra))
        self.out = buf.getvalue()
        return rc, cs.call_count

    def _manifest_puts(self):
        return [k for k in self.store.put_log
                if k == self.entry.latest_times_key(PREFIX)]


class TestOrphanSelfHeals(_EmitBase):
    def test_orphan_costs_one_fetch_one_get_one_put_and_no_listing(self):
        rc, listings = self._pass()
        ops = dict(self.store.ops)            # before this test's own reads
        self.assertEqual(rc, 0)
        # newest-first: the orphan's slot, then the real hole; A never fetched
        self.assertEqual(self.fetched, [self.B, self.C])
        # NOT rendered twice: nothing was PUT under the orphan's stamp
        b_dir = self.entry.tile_stamp_prefix(PREFIX, self.B)
        self.assertEqual([k for k in self.store.put_log
                          if k.startswith(b_dir)], [])
        # ...while the real hole was cut and marked
        c_dir = self.entry.tile_stamp_prefix(PREFIX, self.C)
        self.assertTrue(any(k.startswith(c_dir) for k in self.store.put_log))
        self.assertTrue(self.store.head(self.entry.ready_key(PREFIX, self.C)))
        # advertised after ONE pass, in order
        self.assertEqual(self._times(), [self.D, self.C, self.B, self.A])
        # the orphan heals on the APPEND path (review 2026-08-25): its own
        # marker answers the geometry with one GET, so NO rebuild listing;
        # two incremental manifest PUTs (B, then C)
        self.assertEqual(listings, 0)
        self.assertEqual(len(self._manifest_puts()), 2)
        self.assertEqual(ops["get"], 4)       # cover + B manifest + B marker
        self.assertEqual(ops["head"], 2)      # + C manifest; dedup HEAD x2
        self.assertNotIn("list_pages", ops)
        self.assertIn(f"[done] {self.entry.product_id}: frames=3 "
                      f"latest={self.A}", self.out)     # B's (incremental)

    def test_next_pass_is_idle_on_the_manifest_path(self):
        self._pass()
        self.fetched.clear()
        self.store.ops = {}
        rc, listings = self._pass()
        self.assertEqual(rc, 0)
        self.assertEqual(self.fetched, [])
        self.assertEqual(listings, 0)
        self.assertEqual(self.store.ops, {"get": 1})

    def test_heal_due_orphan_still_rebuilds(self):
        # the heal tick keeps its full rebuild-from-listing (unchanged)
        os.remove(E._state_path("heal", PREFIX, self.entry.product_id))
        rc, listings = self._pass()
        self.assertEqual(rc, 0)
        self.assertEqual(listings, 1)         # B's heal rebuild; C appends
        self.assertEqual(self._times(), [self.D, self.C, self.B, self.A])


class TestOrphanLegacyMarker(_EmitBase):
    """A marker WITHOUT maxzoom (unknown geometry) keeps the pre-review
    rebuild: the listing derives the geometry from the frame's own tiles."""
    B_MARKER = {}

    def test_unknown_geometry_orphan_rebuilds_as_before(self):
        rc, listings = self._pass()
        self.assertEqual(rc, 0)
        self.assertEqual(self.fetched, [self.B, self.C])
        self.assertEqual(listings, 1)
        self.assertEqual(self._times(), [self.D, self.C, self.B, self.A])
        self.assertEqual(self._live()["maxzoom"], 0)


# ---------------------------------------------------------------------------
# (c') the STRAY: a complete frame at a FOREIGN geometry inside the window
# ---------------------------------------------------------------------------
class TestStrayGeometry(_EmitBase):
    """Review finding 2026-08-25 (R4/R7, reproduced on the builder's code):
    the unadvertised duplicate took the REBUILD path, which anchors on the
    emitted frame's own geometry -> the live manifest flipped to the stray's
    maxzoom (every real frame dropped), then back on the next rendered
    frame, every pass until the stray aged out of the window."""
    B_MAXZOOM = 1

    def test_stray_never_flips_the_manifest_and_is_memoed(self):
        rc, listings = self._pass()                  # NO --allow-geometry-change
        ops = dict(self.store.ops)
        self.assertEqual(rc, 0)
        self.assertEqual(self.fetched, [self.B, self.C])   # visited once...
        self.assertIn(f"[manifest] stray geometry {self.B} (maxzoom 1 != 0)",
                      self.out)
        live = self._live()
        self.assertEqual(live["maxzoom"], 0)               # ...geometry kept
        self.assertEqual(live["times"], [self.D, self.C, self.A])   # B never in
        self.assertNotIn("WARNING: dropped", self.out)
        self.assertNotIn("[FAIL]", self.out)               # no RuntimeError
        b_dir = self.entry.tile_stamp_prefix(PREFIX, self.B)
        self.assertEqual([k for k in self.store.put_log
                          if k.startswith(b_dir)], [])
        self.assertEqual(len(self._manifest_puts()), 1)    # C's append only
        self.assertEqual(listings, 0)
        self.assertEqual(ops["get"], 4)       # cover + B manifest + B marker
        self.assertNotIn("list_pages", ops)   # + C manifest
        self.assertEqual(E._stray_stamps(PREFIX, self.entry.product_id),
                         [self.B])

    def test_next_pass_covers_the_stray_from_the_memo(self):
        self._pass()
        self.fetched.clear()
        self.store.ops = {}
        rc, listings = self._pass()
        self.assertEqual(rc, 0)
        self.assertEqual(self.fetched, [])                 # not re-fetched
        self.assertEqual(listings, 0)
        self.assertEqual(self.store.ops, {"get": 1})
        self.assertIn(f"[cover] {self.entry.product_id}: manifest (2 stamps "
                      f"in window) get=1 stray=1", self.out)   # A, C; D outside
        self.assertEqual(self._live()["times"], [self.D, self.C, self.A])

    def test_heal_due_does_not_adopt_the_stray_geometry(self):
        os.remove(E._state_path("heal", PREFIX, self.entry.product_id))
        rc, listings = self._pass(["--allow-geometry-change"])
        self.assertEqual(rc, 0)
        live = self._live()
        self.assertEqual(live["maxzoom"], 0)
        self.assertNotIn(self.B, live["times"])
        self.assertIn(self.C, live["times"])
        self.assertIn(self.A, live["times"])
        self.assertEqual(listings, 1)         # C's heal rebuild, anchored on C

    def test_single_tick_time_emit_keeps_the_rebuild(self):
        # the guard is --step only: a --time emit of the stray WITH
        # --allow-geometry-change is the operator's explicit migration retry
        # (tests/test_s2_emit_suite pins the refuse-then-allow flow) and
        # still rebuilds the manifest around the emitted frame's geometry
        when = dt.datetime(2026, 8, 25, 9, 50, tzinfo=UTC)
        with mock.patch.object(E, "_write_products_index"), \
             _clock(), contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = E.main(["--product", self.entry.product_id,
                         "--store", f"local:{self.tmp}", "--scheme", SCHEME,
                         "--time", when.isoformat(), "--allow-geometry-change"])
        self.assertEqual(rc, 0)
        self.assertNotIn("stray geometry", buf.getvalue())
        live = self._live()
        self.assertEqual((live["maxzoom"], live["times"]), (1, [self.B]))
        self.assertEqual(E._stray_stamps(PREFIX, self.entry.product_id), [])

    def test_verify_listing_counts_the_stray_as_covered(self):
        # the listing path (any maxzoom covers its slot) is unchanged
        self._expire(self.entry)
        rc, listings = self._pass()
        self.assertEqual(rc, 0)
        self.assertEqual(self.fetched, [self.C])           # B covered by listing
        self.assertNotIn(self.B, self._live()["times"])    # foreign: not reconciled


# ---------------------------------------------------------------------------
# (d) idle product pass: 1 GET, 0 LIST pages, nothing else
# ---------------------------------------------------------------------------
class TestIdlePassOps(_Base):
    def test_idle_pass_is_one_get(self):
        stamps = [_stamp(_ago(m)) for m in (7, 17, 27, 37, 47, 100)]
        for s in stamps:
            _frame(self.store, self.entry, s)
        _manifest(self.store, self.entry, stamps)
        self.store.ops = {}
        buf = io.StringIO()
        with mock.patch.object(E, "emit_one") as eo, \
             mock.patch.object(E, "_write_products_index") as wi, \
             mock.patch.object(E, "_reconcile_manifest") as rm, \
             _clock(), contextlib.redirect_stdout(buf):
            rc = E.main(["--product", self.entry.product_id,
                         "--store", f"local:{self.tmp}",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        eo.assert_not_called()
        wi.assert_not_called()
        rm.assert_not_called()                # manifest path: no reconcile
        self.assertEqual(self.store.ops, {"get": 1})
        self.assertIn(f"[cover] {self.entry.product_id}: manifest (5 stamps "
                      f"in window) get=1", buf.getvalue())
        self.assertIn("[ops]", buf.getvalue())
        self.assertNotIn("list_pages", buf.getvalue())


# ---------------------------------------------------------------------------
# (e) the rollback lever
# ---------------------------------------------------------------------------
class TestKnob(_Base):
    def setUp(self):
        super().setUp()
        self.stamps = [_stamp(_ago(m)) for m in (7, 17, 27, 37, 47, 100)]
        for s in self.stamps:
            _frame(self.store, self.entry, s)
        _manifest(self.store, self.entry, self.stamps)
        self.store.ops = {}

    def test_zero_refuses_the_manifest(self):
        with mock.patch.dict(os.environ, {"S2_COVER_FROM_MANIFEST": "0"}), \
             _clock():
            cov, reason = E._covered_from_manifest(
                [self.entry], self.store, PREFIX, WINDOW)
        self.assertIsNone(cov)
        self.assertEqual(reason,
                         f"{self.entry.product_id} (S2_COVER_FROM_MANIFEST=0)")
        self.assertEqual(self.store.ops, {})      # not even the GET

    def test_zero_pass_lists_like_before(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"S2_COVER_FROM_MANIFEST": "0"}), \
             mock.patch.object(E, "_covered_times",
                               wraps=E._covered_times) as ct, \
             mock.patch.object(E, "emit_one") as eo, \
             mock.patch.object(E, "_write_products_index"), \
             _clock(), contextlib.redirect_stdout(buf):
            rc = E.main(["--product", self.entry.product_id,
                         "--store", f"local:{self.tmp}",
                         "--step", "10", "--backfill", "30"])
        self.assertEqual(rc, 0)
        self.assertEqual(ct.call_count, 1)
        eo.assert_not_called()
        self.assertIn("(S2_COVER_FROM_MANIFEST=0) -- listing fallback",
                      buf.getvalue())
        # the pre-change cost: tail listing + per-stamp probes (+ reconcile GET)
        self.assertGreaterEqual(self.store.ops["list_pages"], 1 + 5)
        self.assertEqual(self.store.ops["get"], 1)

    def test_default_and_one_use_the_manifest(self):
        for val in ({}, {"S2_COVER_FROM_MANIFEST": "1"}):
            env = dict(os.environ)
            env.pop("S2_COVER_FROM_MANIFEST", None)
            env.update(val)
            self.store.ops = {}
            with mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch.object(E, "_covered_times") as ct, \
                 mock.patch.object(E, "emit_one"), \
                 mock.patch.object(E, "_write_products_index"), \
                 _clock(), contextlib.redirect_stdout(io.StringIO()):
                rc = E.main(["--product", self.entry.product_id,
                             "--store", f"local:{self.tmp}",
                             "--step", "10", "--backfill", "30"])
            self.assertEqual(rc, 0)
            ct.assert_not_called()
            self.assertEqual(self.store.ops, {"get": 1})


# ---------------------------------------------------------------------------
# (v) the periodic listing VERIFY: what the manifest path cannot see
# ---------------------------------------------------------------------------
class TestPeriodicVerify(_Base):
    def setUp(self):
        super().setUp()
        # grid-aligned stamps (10:00, 09:50, 09:40, 09:30, 09:20, 08:27,
        # 07:27): the mocked fetcher returns the slot's own stamp, so a
        # re-rendered slot must land on the SAME stamp as the frame it heals
        base = NOW.replace(second=0)
        self.stamps = [_stamp(_ago(m, base))
                       for m in (7, 17, 27, 37, 47, 100, 160)]
        for s in self.stamps:
            _frame(self.store, self.entry, s)
        _manifest(self.store, self.entry, self.stamps)
        E._state_touch(E._state_path("heal", PREFIX, self.entry.product_id))
        self.store.ops = {}
        self.gate = E._state_path("cover", PREFIX, self.entry.product_id)

    def _pass(self, now=NOW, emit=None):
        buf = io.StringIO()
        ctx = (mock.patch.object(E, "emit_one") if emit is None
               else contextlib.nullcontext())
        with mock.patch.object(E, "_covered_times",
                               wraps=E._covered_times) as ct, \
             mock.patch.object(E, "_reconcile_manifest",
                               wraps=E._reconcile_manifest) as rm, \
             mock.patch.object(E, "_write_products_index"), \
             ctx as eo, _clock(now), contextlib.redirect_stdout(buf):
            rc = E.main(["--product", self.entry.product_id,
                         "--store", f"local:{self.tmp}", "--scheme", SCHEME,
                         "--step", "10", "--backfill", "30"])
        self.out = buf.getvalue()
        return rc, ct.call_count, rm.call_count, eo

    def test_fresh_container_lists_first_then_uses_the_manifest(self):
        os.remove(self.gate)                  # a fresh container: no gate
        rc, ct, rm, eo = self._pass()
        self.assertEqual((rc, ct, rm), (0, 1, 1))
        eo.assert_not_called()
        self.assertIn(f"[cover] listing verify for {self.entry.product_id}: "
                      f"1/1 gate(s) due (every {E.COVER_VERIFY_S}s per product)",
                      self.out)
        self.assertGreaterEqual(self.store.ops["list_pages"], 1 + 5)
        self.assertEqual(self.store.ops["get"], 1)         # the reconcile's
        self.assertTrue(os.path.exists(self.gate))          # re-armed
        self.store.ops = {}
        rc, ct, rm, eo = self._pass()
        self.assertEqual((rc, ct, rm), (0, 0, 0))
        self.assertEqual(self.store.ops, {"get": 1})

    def test_expired_gate_lists_again(self):
        self._expire(self.entry)
        rc, ct, rm, _ = self._pass()
        self.assertEqual((rc, ct, rm), (0, 1, 1))
        self.assertIn("[cover] listing verify", self.out)
        self.store.ops = {}
        rc, ct, rm, _ = self._pass()
        self.assertEqual((rc, ct, rm), (0, 0, 0))

    def test_zero_interval_lists_every_pass(self):
        with mock.patch.object(E, "COVER_VERIFY_S", 0):
            for _ in range(2):
                rc, ct, rm, _ = self._pass()
                self.assertEqual((rc, ct, rm), (0, 1, 1))

    def test_orphan_off_the_grid_heals_on_the_verify_pass(self):
        # review finding 2026-08-25 (R2): pass @10:17:42 -> grid 10:10, 10:00,
        # 09:50; an orphan at 09:40 (the oldest slot of the PREVIOUS pass,
        # killed between its marker PUT and its manifest PUT) is no longer a
        # slot, so the manifest path never visits it...
        now2 = dt.datetime(2026, 8, 25, 10, 17, 42, tzinfo=UTC)
        orphan = _stamp(dt.datetime(2026, 8, 25, 9, 40, tzinfo=UTC))
        grid = [_stamp(dt.datetime(2026, 8, 25, 10, m, tzinfo=UTC))
                for m in (10, 0)] + [_stamp(dt.datetime(2026, 8, 25, 9, 50,
                                                        tzinfo=UTC))]
        for s in grid + [orphan]:
            _frame(self.store, self.entry, s)
        _manifest(self.store, self.entry,                        # orphan absent
                  [s for s in self.stamps if s != orphan] + grid)
        rc, ct, rm, eo = self._pass(now=now2)
        self.assertEqual((rc, ct, rm), (0, 0, 0))
        eo.assert_not_called()
        self.assertNotIn(orphan, self._times())
        # ...the verify listing's reconcile re-advertises it within
        # COVER_VERIFY_S instead of the 6 h heal tick
        self._expire(self.entry)
        rc, ct, rm, eo = self._pass(now=now2)
        self.assertEqual((rc, ct, rm), (0, 1, 1))
        eo.assert_not_called()
        self.assertIn("[manifest] reconciled 1 orphan frame(s)", self.out)
        self.assertIn(orphan, self._times())

    def test_deleted_frame_re_renders_on_the_verify_pass(self):
        # review finding 2026-08-25 (R3/R6): every key of an advertised frame
        # is deleted under the live manifest (an operator purge, the canary
        # watchdog's rollback cleanup). The manifest path cannot see it...
        ghost = self.stamps[1]                                   # 17 min ago
        g_dir = self.entry.tile_stamp_prefix(PREFIX, ghost)
        self.store.delete(self.store.list_keys(g_dir))
        self.assertFalse(self.store.head(self.entry.ready_key(PREFIX, ghost)))
        fetched = []

        def _imagery(entry, time=None, nearest=True, band_cache=None):
            fetched.append(_stamp(time))
            return _Img(_stamp(time))
        self.store.ops = {}
        with mock.patch.object(E.I, "produce_imagery", side_effect=_imagery):
            rc, ct, rm, _ = self._pass(emit=True)
            self.assertEqual((rc, ct, rm), (0, 0, 0))
            self.assertEqual(fetched, [])                        # still a ghost
            self.assertIn(ghost, self._times())
            # ...the verify listing sees the slot open and re-renders it; the
            # manifest already advertises the stamp, so no manifest PUT
            self._expire(self.entry)
            self.store.put_log = []
            rc, ct, rm, _ = self._pass(emit=True)
        self.assertEqual((rc, ct, rm), (0, 1, 1))
        self.assertEqual(fetched, [ghost])
        self.assertTrue(self.store.head(self.entry.ready_key(PREFIX, ghost)))
        self.assertIn(f"[manifest] unchanged -- {ghost} already advertised",
                      self.out)
        self.assertEqual([k for k in self.store.put_log
                          if k == self.entry.latest_times_key(PREFIX)], [])
        self.assertEqual(self._times(), sorted(self.stamps))


if __name__ == "__main__":
    unittest.main()
