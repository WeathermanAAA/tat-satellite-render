#!/usr/bin/env python3
"""Tests for the never-miss audit comparison + the pixel-diff core (pure)."""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s1_audit as A  # noqa: E402


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
