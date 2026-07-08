#!/usr/bin/env python3
"""Suite-mode orchestration tests for s2_pyramid_emit: one pinned scan, shared
band cache, per-product failure isolation, products.json refresh. The imagery
producers are mocked (no network); the tiler + store are real."""
import datetime as dt
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_imagery as I  # noqa: E402
import s2_pyramid_emit as E  # noqa: E402
import s2_registry as R  # noqa: E402

UTC = dt.timezone.utc
SCAN = dt.datetime(2026, 7, 8, 18, 6, 17, tzinfo=UTC)


def _fake_img(entry):
    rgba = np.zeros((64, 96, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    res = I.ImageryResult(
        rgba=rgba, bounds=entry.sector_bbox, scan_start=SCAN,
        product="CMIPC", bucket="noaa-goes19", s3_key="k.nc",
        generic_channel=entry.band_key, enhancement="x")
    return res


class TestSuiteEmit(unittest.TestCase):
    def _run(self, fail_ids=(), fail_plain=False, argv_extra=()):
        tmp = tempfile.mkdtemp(prefix="s2suite-")
        calls = {"recipe": [], "plain": [], "caches": []}

        def fake_recipe(entry, time=None, nearest=True, band_cache=None):
            if entry.product_id in fail_ids:
                raise RuntimeError("boom")
            calls["recipe"].append((entry.product_id, time))
            calls["caches"].append(band_cache)
            return _fake_img(entry)

        def fake_plain(entry, time=None, nearest=True):
            if fail_plain:
                raise RuntimeError("boom")
            calls["plain"].append((entry.product_id, time))
            return _fake_img(entry)

        with mock.patch.object(E.I, "produce_recipe_imagery", fake_recipe), \
             mock.patch.object(E.I, "produce_imagery", fake_plain), \
             mock.patch.object(E, "_pin_suite_scan", lambda entries, when: SCAN):
            rc = E.main(["--suite", "conus", "--store", f"local:{tmp}",
                         "--max-zoom", "2", *argv_extra])
        return rc, tmp, calls

    def test_suite_emits_every_conus_product_off_one_scan(self):
        rc, tmp, calls = self._run()
        self.assertEqual(rc, 0)
        n_suite = len([e for e in R.REGISTRY if e.tiled and e.sector_key == "conus"])
        self.assertEqual(len(calls["recipe"]) + len(calls["plain"]), n_suite)
        # every recipe product got the SAME pinned time + the SAME cache object
        self.assertTrue(all(t == SCAN for _, t in calls["recipe"] + calls["plain"]))
        self.assertEqual(len({id(c) for c in calls["caches"]}), 1)
        # the plain (non-recipe) clean-IR row went through the frozen path
        self.assertIn("goes19-conus-ir", [p for p, _ in calls["plain"]])
        # products index written
        idx_path = os.path.join(tmp, R.products_index_key("shadow", "goes19", "conus"))
        self.assertTrue(os.path.exists(idx_path))
        # a sample product's manifest + capped tiles exist
        man = os.path.join(tmp, "shadow/sat/goes19/conus/airmass/latest_times.json")
        self.assertTrue(os.path.exists(man))
        import json
        m = json.load(open(man))
        self.assertLessEqual(m["maxzoom"], 2)          # --max-zoom respected
        self.assertEqual(m["times"], [SCAN.strftime("%Y%m%dT%H%M%SZ")])

    def test_one_bad_product_never_kills_the_suite(self):
        rc, tmp, calls = self._run(fail_ids={"goes19-conus-dust"})
        self.assertEqual(rc, 0)                        # partial success = 0
        emitted = [p for p, _ in calls["recipe"]]
        self.assertNotIn("goes19-conus-dust", emitted)
        self.assertIn("goes19-conus-airmass", emitted)
        self.assertFalse(os.path.exists(
            os.path.join(tmp, "shadow/sat/goes19/conus/dust/latest_times.json")))

    def test_geometry_change_refused_then_allowed(self):
        # Q7 on-demand-z6 guard: an emit whose maxzoom differs from the
        # prefix's existing frames REFUSES (would clobber the loop) unless
        # --allow-geometry-change is passed.
        tmp = tempfile.mkdtemp(prefix="s2geom-")
        later = SCAN + dt.timedelta(minutes=5)

        def fake(entry, time=None, nearest=True, band_cache=None):
            img = _fake_img(entry)
            img.scan_start = time or SCAN
            return img

        with mock.patch.object(E.I, "produce_recipe_imagery", fake):
            args = ["--product", "goes19-conus-airmass", "--store", f"local:{tmp}"]
            rc = E.main(args + ["--max-zoom", "2", "--time", SCAN.isoformat()])
            self.assertEqual(rc, 0)
            with self.assertRaises(RuntimeError):
                E.main(args + ["--max-zoom", "1", "--time", later.isoformat()])
            rc = E.main(args + ["--max-zoom", "1", "--time", later.isoformat(),
                                "--allow-geometry-change"])
            self.assertEqual(rc, 0)
        import json
        man = json.load(open(os.path.join(
            tmp, "shadow/sat/goes19/conus/airmass/latest_times.json")))
        self.assertEqual(man["maxzoom"], 1)
        self.assertEqual(man["count"], 1)   # migration dropped the old-geometry frame

    def test_total_failure_exits_nonzero(self):
        all_ids = {e.product_id for e in R.REGISTRY
                   if e.tiled and e.sector_key == "conus" and e.recipe_id}
        rc, tmp, calls = self._run(fail_ids=all_ids, fail_plain=True)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
