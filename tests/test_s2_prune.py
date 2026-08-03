"""s2_prune -- object-level shadow TTL (replaces the bucket-lifecycle rule the
box token can't set). Repo convention: hand FakeS3, no moto, no network."""
import datetime as dt
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import s2_prune as PR

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 7, 9, 0, 0, tzinfo=UTC)


def stamp(days_ago: float) -> str:
    return (NOW - dt.timedelta(days=days_ago)).strftime(PR.STAMP_FMT)


def tiled_keys(product: str, s: str):
    return [(f"{product}/{s}/_ready.json", 50),
            (f"{product}/{s}/bt.png", 4000),
            (f"{product}/{s}/0/0/0.webp", 1000),
            (f"{product}/{s}/5/8/11.webp", 1000)]


class FakeS3:
    """Just enough of the S3 client surface, with 1-key pages to prove the
    paginator, and optional AccessDenied on DeleteObjects."""

    def __init__(self, objects, deny_batch=False):
        self.objects = dict(objects)          # key -> bytes payload or size int
        self.deny_batch = deny_batch
        self.deleted_order = []
        self.puts = {}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000,
                        ContinuationToken=None, Delimiter=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if Delimiter:
            # grouped listing (the per-product walk uses this): keys with the
            # delimiter beyond Prefix collapse into CommonPrefixes
            pres, own = [], []
            for k in keys:
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    pre = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                    if pre not in pres:
                        pres.append(pre)
                else:
                    own.append(k)
            return {"CommonPrefixes": [{"Prefix": p} for p in pres],
                    "Contents": [{"Key": k, "Size": self._size(k)} for k in own],
                    "IsTruncated": False}
        start = int(ContinuationToken or 0)
        page = keys[start:start + 1]          # 1-key pages: exercise pagination
        out = {"Contents": [{"Key": k, "Size": self._size(k)} for k in page],
               "IsTruncated": start + 1 < len(keys)}
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + 1)
        return out

    def _size(self, k):
        v = self.objects[k]
        return len(v) if isinstance(v, (bytes, str)) else int(v)

    def delete_objects(self, Bucket, Delete):
        if self.deny_batch:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "DeleteObjects")
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)
            self.deleted_order.append(o["Key"])
        return {}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)
        self.deleted_order.append(Key)
        return {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects or not isinstance(self.objects[Key], bytes):
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body
        self.puts[Key] = Body
        return {}


def build_bucket():
    """Two tiled products + one single-frame (S1-shape) product + manifests."""
    P1, P2 = "shadow/sat/goes19/conus/ir", "shadow/sat/goes19/conus/airmass"
    S1 = "shadow/sat/goes19/meso2/ir"
    objs = []
    stamps1 = [stamp(20), stamp(16), stamp(10), stamp(0.5)]
    for s in stamps1:
        objs += tiled_keys(P1, s)
    stamps2 = [stamp(30), stamp(20), stamp(16)]          # ALL over-age
    for s in stamps2:
        objs += tiled_keys(P2, s)
    objs += [(f"{S1}/{stamp(41)}.webp", 70000), (f"{S1}/{stamp(40)}.webp", 70000),
             (f"{S1}/{stamp(1)}.webp", 70000)]
    # never-touch keys
    objs += [("shadow/sat/goes19/conus/products.json", 900),
             (f"{S1}/health.json", 300)]
    m1 = json.dumps({"times": stamps1, "latest": stamps1[-1], "count": 4,
                     "tile": "x/{t}/{z}/{x}/{y}.webp"}).encode()
    m2 = json.dumps({"times": stamps2, "latest": stamps2[-1], "count": 3}).encode()
    objs += [(f"{P1}/latest_times.json", m1), (f"{P2}/latest_times.json", m2)]
    return objs, (P1, stamps1), (P2, stamps2), S1


class TestClassify(unittest.TestCase):
    def test_tiled_single_and_nonframe(self):
        s = stamp(3)
        self.assertEqual(PR.classify_key(f"a/b/{s}/0/0/0.webp"), ("a/b", s, False))
        self.assertEqual(PR.classify_key(f"a/b/{s}/_ready.json"), ("a/b", s, True))
        self.assertEqual(PR.classify_key(f"a/b/{s}/bt.png"), ("a/b", s, False))
        self.assertEqual(PR.classify_key(f"a/b/{s}.webp"), ("a/b", s, False))
        for k in ("a/b/latest_times.json", "a/b/products.json", "a/b/health.json",
                  "a/b/20261340T000000Z/0/0/0.webp"):     # invalid date too
            self.assertIsNone(PR.classify_key(k), k)


class TestPlan(unittest.TestCase):
    def test_ttl_keepmin_and_marker_first(self):
        objs, (P1, s1), (P2, s2), S1 = build_bucket()
        plans = {p.product: p for p in
                 PR.plan_prune(objs, days=14, keep_min=2, now=NOW)}
        # P1: 20d + 16d over-age, both outside the newest-2 -> condemned
        self.assertEqual(plans[P1].condemned, [s1[0], s1[1]])
        # marker first within each stamp's key group
        self.assertTrue(plans[P1].keys[0].endswith("_ready.json"))
        self.assertEqual(len(plans[P1].keys), 8)
        # P2: all 3 over-age, newest 2 held by keep-min -> only the 30d goes
        self.assertEqual(plans[P2].condemned, [s2[0]])
        self.assertEqual(plans[P2].kept_old, [s2[1], s2[2]])
        # S1 single-frame: 41d condemned; 40d held by keep-min; fresh kept
        self.assertEqual(plans[S1].condemned, [stamp(41)])
        self.assertEqual(plans[S1].kept_old, [stamp(40)])
        # size accounting: one tiled stamp = 50+4000+1000+1000
        self.assertEqual(plans[P2].bytes, 6050)

    def test_keepmin_zero_allows_empty(self):
        objs, _, (P2, s2), _ = build_bucket()
        plans = {p.product: p for p in
                 PR.plan_prune(objs, days=14, keep_min=0, now=NOW)}
        self.assertEqual(plans[P2].condemned, s2)

    def test_manifest_rewrite(self):
        m = {"times": ["a", "b", "c"], "latest": "c", "count": 3, "bt": None}
        self.assertIsNone(PR.rewrite_manifest(m, {"z"}))
        out = PR.rewrite_manifest(m, {"a", "b"})
        self.assertEqual((out["times"], out["latest"], out["count"]), (["c"], "c", 1))
        out = PR.rewrite_manifest(m, {"a", "b", "c"})
        self.assertEqual((out["times"], out["latest"], out["count"]), ([], None, 0))


class TestMainApply(unittest.TestCase):
    def run_main(self, s3, *extra):
        return PR.main(["--prefix", "shadow/sat/", "--days", "14",
                        "--keep-min", "2", *extra], s3=s3)

    def test_dry_run_deletes_nothing(self):
        objs, *_ = build_bucket()
        s3 = FakeS3(objs)
        n = len(s3.objects)
        self.assertEqual(self.run_main(s3), 0)
        self.assertEqual(len(s3.objects), n)
        self.assertEqual(s3.puts, {})

    def test_apply_deletes_and_rewrites(self):
        objs, (P1, s1), (P2, s2), S1 = build_bucket()
        s3 = FakeS3(objs)
        self.assertEqual(self.run_main(s3, "--apply"), 0)
        # condemned stamps fully gone (marker + bt + tiles) -- scoped to their
        # OWN product: the same stamp string can legitimately survive in a
        # different product where keep-min protects it.
        for prod, s in ((P1, s1[0]), (P1, s1[1]), (P2, s2[0])):
            self.assertFalse(any(k.startswith(f"{prod}/{s}/") for k in s3.objects), s)
        # survivors + never-touch keys intact
        self.assertTrue(any(k.startswith(f"{P1}/{s1[2]}/") for k in s3.objects))
        self.assertIn("shadow/sat/goes19/conus/products.json", s3.objects)
        self.assertIn(f"{S1}/health.json", s3.objects)
        # manifests rewritten to surviving stamps only
        man1 = json.loads(s3.puts[f"{P1}/latest_times.json"])
        self.assertEqual(man1["times"], [s1[2], s1[3]])
        self.assertEqual(man1["count"], 2)
        self.assertEqual(man1["latest"], s1[3])
        man2 = json.loads(s3.puts[f"{P2}/latest_times.json"])
        self.assertEqual(man2["times"], [s2[1], s2[2]])
        # markers deleted before that stamp's tiles
        for s in (s1[0], s1[1], s2[0]):
            group = [k for k in s3.deleted_order if f"/{s}/" in k]
            self.assertTrue(group[0].endswith("_ready.json"), group)

    def test_batch_denied_falls_back_per_key(self):
        objs, (P1, s1), _, _ = build_bucket()
        s3 = FakeS3(objs, deny_batch=True)
        self.assertEqual(self.run_main(s3, "--apply"), 0)
        self.assertFalse(any(k.startswith(f"{P1}/{s1[0]}/") for k in s3.objects))

    def test_prefix_guard(self):
        s3 = FakeS3([])
        with self.assertRaises(SystemExit):
            PR.main(["--prefix", "models/hafs/", "--apply"], s3=s3)
        with self.assertRaises(SystemExit):
            PR.main(["--prefix", "shadow/sat/", "--days", "0"], s3=s3)


if __name__ == "__main__":
    unittest.main()
