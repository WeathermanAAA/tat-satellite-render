#!/usr/bin/env python3
"""Tests for the never-miss audit comparison + the pixel-diff core (pure)."""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime as dt  # noqa: E402
import s1_audit as A  # noqa: E402
import s1_slots as S  # noqa: E402

UTC = dt.timezone.utc


def _stamp(mins_ago, now):
    return (now - dt.timedelta(minutes=mins_ago)).strftime(S.STAMP_FMT)


class TestClassifyCoverage(unittest.TestCase):
    def setUp(self):
        self.now = dt.datetime(2026, 6, 18, 22, 0, 0, tzinfo=UTC)

    def test_all_covered(self):
        gt = [_stamp(m, self.now) for m in (10, 9, 8, 7)]
        c = S.classify_coverage(gt, gt, self.now, settle_s=180)
        self.assertEqual(c["missed"], [])
        self.assertEqual(c["pending"], [])
        self.assertEqual(len(c["covered"]), 4)

    def test_settled_missing_is_a_miss(self):
        # worker up (first_shadow=10min ago), slot at 8min ago settled, not shipped
        gt = [_stamp(10, self.now), _stamp(8, self.now), _stamp(6, self.now)]
        shadow = [_stamp(10, self.now), _stamp(6, self.now)]   # 8-min slot missing
        c = S.classify_coverage(gt, shadow, self.now, settle_s=180)
        self.assertEqual(c["missed"], [_stamp(8, self.now)])

    def test_recent_missing_is_pending(self):
        # slot 1 min ago (< settle_s=180) not shipped -> in-flight PENDING
        gt = [_stamp(10, self.now), _stamp(1, self.now)]
        shadow = [_stamp(10, self.now)]
        c = S.classify_coverage(gt, shadow, self.now, settle_s=180)
        self.assertEqual(c["missed"], [])
        self.assertIn(_stamp(1, self.now), c["pending"])

    def test_empty_shadow_all_pending(self):
        gt = [_stamp(m, self.now) for m in (10, 8, 6)]
        c = S.classify_coverage(gt, [], self.now, settle_s=180)
        self.assertEqual(c["missed"], [])
        self.assertEqual(len(c["pending"]), 3)

    def test_pre_worker_slot_is_pending(self):
        # slot older than the first shadow frame -> backfill not caught up -> PENDING
        gt = [_stamp(30, self.now), _stamp(10, self.now), _stamp(8, self.now)]
        shadow = [_stamp(10, self.now), _stamp(8, self.now)]   # first_shadow=10min
        c = S.classify_coverage(gt, shadow, self.now, settle_s=180)
        self.assertEqual(c["missed"], [])                       # 30-min predates worker
        self.assertIn(_stamp(30, self.now), c["pending"])


class TestAuditCompare(unittest.TestCase):
    def test_all_covered(self):
        gt = ["20260618T210057Z", "20260618T210157Z"]
        r = A.audit_compare(gt, gt)
        self.assertEqual(r["missing"], [])
        self.assertEqual(r["covered_in_window"], 2)

    def test_missing_slot_detected(self):
        gt = ["a", "b", "c"]
        r = A.audit_compare(gt, ["a", "c"])
        self.assertEqual(r["missing"], ["b"])

    def test_no_data_counts_as_covered(self):
        gt = ["a", "b"]
        r = A.audit_compare(gt, ["a"], no_data_stamps=["b"])
        self.assertEqual(r["missing"], [])          # b covered by logged no-data

    def test_extra_published(self):
        r = A.audit_compare(["a"], ["a", "b"])
        self.assertEqual(r["extra"], ["b"])
        self.assertEqual(r["missing"], [])


class TestDiffFrames(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            self.skipTest("numpy/PIL not available")

    def _webp(self, color, size=(32, 24)):
        from PIL import Image
        im = Image.new("RGB", size, color)
        b = io.BytesIO()
        im.save(b, "WEBP", quality=90, method=6)
        return b.getvalue()

    def test_byte_identical(self):
        w = self._webp((10, 20, 30))
        r = A and __import__("s1_pixeldiff").diff_frames(w, w)
        self.assertTrue(r["byte_equal"])
        self.assertTrue(r["pixel_equal"])
        self.assertEqual(r["max_abs_diff"], 0)

    def test_different_colors(self):
        import s1_pixeldiff as P
        a = self._webp((10, 20, 30))
        b = self._webp((200, 200, 200))
        r = P.diff_frames(a, b)
        self.assertFalse(r["pixel_equal"])
        self.assertGreater(r["max_abs_diff"], 0)
        self.assertGreater(r["diff_frac"], 0.5)

    def test_shape_mismatch(self):
        import s1_pixeldiff as P
        a = self._webp((10, 20, 30), size=(32, 24))
        b = self._webp((10, 20, 30), size=(16, 12))
        r = P.diff_frames(a, b)
        self.assertFalse(r["shape_match"])


if __name__ == "__main__":
    unittest.main()
