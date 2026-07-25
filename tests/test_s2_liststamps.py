"""complete_stamps: the delimiter (fast) path must answer EXACTLY what the
flat whole-subtree walk answers.

Why this test exists (2026-07-25): the manifest rebuild used to enumerate every
tile of every retained frame -- 460k keys / 239 s on geo-global, twice per
product-slot, growing with retention. That made native scan cadence
arithmetically impossible (one ABI FD product-slot needed 636 s of listing
against a 600 s budget). complete_stamps now reads the key LAYOUT with
delimiter listings at a cost independent of pyramid depth. The whole safety
argument is EQUIVALENCE: same frames, same maxzoom, same exclusion of partial
emits. A store that lacks the delimiter methods still gets the flat walk, so
both implementations must be kept honest against each other forever.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import s2_pyramid as P            # noqa: E402
import s2_registry as R           # noqa: E402


class FlatR2:
    """list_keys only -- drives complete_stamps' fallback path."""

    def __init__(self):
        self.store = {}

    def put_bytes(self, key, data, content_type, cache):
        self.store[key] = data
        return True

    def put_json(self, key, obj, cache):
        self.store[key] = b"{}"
        return True

    def head(self, key):
        return key in self.store

    def delete(self, keys):
        for k in keys:
            self.store.pop(k, None)

    def list_keys(self, prefix):
        return [k for k in self.store if k.startswith(prefix)]


class DelimR2(FlatR2):
    """Adds S3 delimiter semantics -- drives the fast path."""

    def list_prefixes(self, prefix, start_after=""):
        out = set()
        for k in self.store:
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            if "/" in rest:
                out.add(prefix + rest.split("/", 1)[0] + "/")
        return sorted(p for p in out if not start_after or p > start_after)

    def list_level(self, prefix):
        pres, keys = set(), []
        for k in self.store:
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            if "/" in rest:
                pres.add(prefix + rest.split("/", 1)[0] + "/")
            else:
                keys.append(k)
        return sorted(pres), sorted(keys)


PREFIX = "shadow"
ENTRY = R.REGISTRY_BY_ID["goes19-fd-ir"]


def seed(store, stamp, maxz, *, complete=True):
    """Write a frame's tiles z0..maxz, and the ready marker iff complete."""
    for z in range(maxz + 1):
        store.put_bytes(ENTRY.tile_key(PREFIX, stamp, z, 0, 0), b"x",
                        "image/webp", "c")
    if complete:
        store.put_json(ENTRY.ready_key(PREFIX, stamp), {"maxzoom": maxz}, "c")


class TestCompleteStampsEquivalence(unittest.TestCase):
    def _both(self, seeder):
        flat, delim = FlatR2(), DelimR2()
        for s in (flat, delim):
            seeder(s)
        # sanity: the fakes really do take different code paths
        self.assertFalse(hasattr(flat, "list_prefixes"))
        self.assertTrue(hasattr(delim, "list_prefixes"))
        return (P.complete_stamps(ENTRY, flat, PREFIX),
                P.complete_stamps(ENTRY, delim, PREFIX))

    def test_simple_series_agrees(self):
        def s(st):
            seed(st, "20260725T000000Z", 5)
            seed(st, "20260725T001000Z", 5)
            seed(st, "20260725T002000Z", 5)
        a, b = self._both(s)
        self.assertEqual(a, b)
        self.assertEqual([x[0] for x in b], ["20260725T000000Z",
                                             "20260725T001000Z",
                                             "20260725T002000Z"])
        self.assertEqual({x[1] for x in b}, {5})

    def test_partial_emit_excluded_by_both(self):
        # tiles present, marker absent -> never advertised (atomicity contract)
        def s(st):
            seed(st, "20260725T000000Z", 5)
            seed(st, "20260725T001000Z", 5, complete=False)
        a, b = self._both(s)
        self.assertEqual(a, b)
        self.assertEqual([x[0] for x in b], ["20260725T000000Z"])

    def test_mixed_geometry_maxzoom_agrees(self):
        # the geometry guard keys off per-frame maxzoom: both paths must report
        # the SAME depth per frame or a pyramid_px change would silently pass
        def s(st):
            seed(st, "20260725T000000Z", 5)
            seed(st, "20260725T001000Z", 7)
            seed(st, "20260725T002000Z", 3)
        a, b = self._both(s)
        self.assertEqual(a, b)
        self.assertEqual(dict(b), {"20260725T000000Z": 5,
                                   "20260725T001000Z": 7,
                                   "20260725T002000Z": 3})

    def test_empty_product_agrees(self):
        a, b = self._both(lambda st: None)
        self.assertEqual(a, b)
        self.assertEqual(b, [])

    def test_limit_keeps_the_newest(self):
        st = DelimR2()
        for i in range(12):
            seed(st, f"20260725T00{i:02d}00Z", 5)
        newest3 = P.complete_stamps(ENTRY, st, PREFIX, limit=3)
        self.assertEqual([x[0] for x in newest3],
                         ["20260725T000900Z", "20260725T001000Z",
                          "20260725T001100Z"])
        # unlimited still returns everything, in order
        self.assertEqual(len(P.complete_stamps(ENTRY, st, PREFIX)), 12)

    def test_limit_is_a_noop_below_the_window(self):
        st = DelimR2()
        seed(st, "20260725T000000Z", 5)
        self.assertEqual(P.complete_stamps(ENTRY, st, PREFIX, limit=90),
                         P.complete_stamps(ENTRY, st, PREFIX))

    def test_serial_path_matches_threaded(self):
        # >4 stamps takes the ThreadPoolExecutor branch; <=4 stays serial.
        st = DelimR2()
        for i in range(9):
            seed(st, f"20260725T00{i:02d}00Z", 5)
        threaded = P.complete_stamps(ENTRY, st, PREFIX)
        os.environ["S2_LIST_WORKERS"] = "1"
        try:
            serial = P.complete_stamps(ENTRY, st, PREFIX)
        finally:
            os.environ.pop("S2_LIST_WORKERS", None)
        self.assertEqual(threaded, serial)


class TestHasCompleteFrame(unittest.TestCase):
    def test_true_when_any_frame_complete(self):
        for cls in (FlatR2, DelimR2):
            st = cls()
            seed(st, "20260725T000000Z", 5)
            self.assertTrue(P.has_complete_frame(ENTRY, st, PREFIX), cls.__name__)

    def test_false_on_empty_and_on_partial_only(self):
        for cls in (FlatR2, DelimR2):
            self.assertFalse(P.has_complete_frame(ENTRY, cls(), PREFIX),
                             cls.__name__)
            st = cls()
            seed(st, "20260725T000000Z", 5, complete=False)
            self.assertFalse(P.has_complete_frame(ENTRY, st, PREFIX),
                             cls.__name__)

    def test_finds_a_complete_frame_behind_many_partials(self):
        # newest-first probing must not stop at the partial tail
        for cls in (FlatR2, DelimR2):
            st = cls()
            seed(st, "20260725T000000Z", 5)
            for i in range(1, 12):
                seed(st, f"20260725T00{i:02d}00Z", 5, complete=False)
            self.assertTrue(P.has_complete_frame(ENTRY, st, PREFIX),
                            cls.__name__)


if __name__ == "__main__":
    unittest.main()
