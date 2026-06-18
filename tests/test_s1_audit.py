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


class TestShippedSetParsing(unittest.TestCase):
    """R2-list + CDN-manifest parsing + empty-/shadow/ tolerance (remote paths)."""

    def test_list_shadow_r2_moto(self):
        try:
            import boto3
            from moto import mock_aws
        except Exception:
            self.skipTest("moto not available")
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
        old_ep = os.environ.pop("R2_ENDPOINT", None)  # None -> moto intercepts
        try:
            with mock_aws():
                s3 = boto3.client("s3", region_name="us-east-1")
                bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")
                s3.create_bucket(Bucket=bucket)
                for st in ("20260618T210057Z", "20260618T210157Z"):
                    s3.put_object(Bucket=bucket,
                                  Key=S.shadow_frame_key("shadow", st), Body=b"x")
                s3.put_object(Bucket=bucket,
                              Key=S.latest_times_key("shadow"), Body=b"{}")  # ignored
                got = A.list_shadow_r2("shadow")
                self.assertEqual(set(got),
                                 {"20260618T210057Z", "20260618T210157Z"})
                for t in got.values():
                    self.assertIsNotNone(t)   # LastModified present
        finally:
            if old_ep is not None:
                os.environ["R2_ENDPOINT"] = old_ep

    def test_list_shadow_cdn_parse(self):
        import json
        orig = A._cdn_get
        A._cdn_get = lambda url: json.dumps(
            {"product": S.S1_PRODUCT_PATH,
             "times": ["20260618T210057Z", "20260618T210157Z"]}).encode()
        try:
            got = A.list_shadow_cdn("shadow")
            self.assertEqual(set(got),
                             {"20260618T210057Z", "20260618T210157Z"})
        finally:
            A._cdn_get = orig

    def test_list_shadow_cdn_404_empty(self):
        import urllib.error
        orig = A._cdn_get

        def boom(url):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        A._cdn_get = boom
        try:
            self.assertEqual(A.list_shadow_cdn("shadow"), {})   # worker not writing yet
        finally:
            A._cdn_get = orig


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


class TestDecomposeDiff(unittest.TestCase):
    """The cross-build-floor vs real-delta decomposition (§7.2/§9)."""
    def setUp(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            self.skipTest("numpy/PIL not available")

    def _noise_rgb(self, seed=0, size=(128, 96)):
        import numpy as np
        rng = np.random.default_rng(seed)
        # high-frequency content (like cloud tops) so the lossy floor is non-trivial
        return rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)

    def _webp(self, rgb):
        import io
        from PIL import Image
        b = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(b, "WEBP", quality=90, method=6)
        return b.getvalue()

    def test_real_zero_when_only_encode_floor(self):
        # prod = a re-encode of the SAME decoded source as shadow -> the only
        # difference is encode quantization -> REAL must be 0.
        import io
        import numpy as np
        from PIL import Image
        import s1_pixeldiff as P
        src = self._noise_rgb(seed=1)
        shadow = self._webp(src)
        decoded = np.asarray(Image.open(io.BytesIO(shadow)).convert("RGB"))
        prod = self._webp(decoded)            # 2nd-gen encode of the same content
        d = P.decompose_diff(shadow, prod)
        self.assertTrue(d["shape_match"])
        self.assertEqual(d["real_frac"], 0.0, f"expected real=0, got {d}")

    def test_real_positive_when_source_differs(self):
        # prod has a real content block changed far beyond any encode floor.
        import s1_pixeldiff as P
        src = self._noise_rgb(seed=2)
        changed = src.copy()
        changed[10:60, 10:60, :] = (changed[10:60, 10:60, :].astype(int) ^ 0xFF).astype("uint8")
        d = P.decompose_diff(self._webp(src), self._webp(changed))
        self.assertTrue(d["shape_match"])
        self.assertGreater(d["real_frac"], 0.0)
        self.assertGreater(d["real_max"], d["floor_ceiling"])

    def test_shape_mismatch_is_all_real(self):
        import s1_pixeldiff as P
        a = self._webp(self._noise_rgb(seed=3, size=(64, 48)))
        b = self._webp(self._noise_rgb(seed=3, size=(32, 24)))
        d = P.decompose_diff(a, b)
        self.assertFalse(d["shape_match"])
        self.assertEqual(d["real_frac"], 1.0)

    def test_identical_bytes(self):
        import s1_pixeldiff as P
        w = self._webp(self._noise_rgb(seed=4))
        d = P.decompose_diff(w, w)
        self.assertTrue(d["byte_equal"])
        self.assertEqual(d["real_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
