"""meso manifest==storage invariant (reconcile_manifests + memory-authoritative
append_frame). The bug: a stale GET-modify-PUT writer (e.g. two poller
containers across a rebuild) let thinning delete R2 objects that another
manifest copy still listed -- the viewer 404'd on scattered phantom frames."""

import datetime as dt
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meso_poller as mp  # noqa: E402


def _stamp(t: dt.datetime) -> str:
    return t.strftime("%Y%m%dT%H%M%SZ")


def _key(slug: str, band_key: str, t: dt.datetime) -> str:
    return f"{mp.R2_PREFIX}/{slug}/{band_key}/{_stamp(t)}.png"


class FakeR2:
    def __init__(self, storage=None, remote=None, fail_prefixes=()):
        self.storage = set(storage or [])
        self.remote = dict(remote or {})
        self.fail_prefixes = tuple(fail_prefixes)
        self.deleted = []
        self.put = {}

    def list_keys(self, prefix):
        for fp in self.fail_prefixes:
            if prefix.startswith(fp):
                raise RuntimeError("list boom")
        return sorted(k for k in self.storage if k.startswith(prefix))

    def get_json(self, key):
        return self.remote.get(key)

    def put_json(self, key, obj, cache):
        self.put[key] = json.loads(json.dumps(obj))
        return True

    def delete(self, keys):
        keys = list(keys)
        self.deleted.extend(keys)
        self.storage -= set(keys)


def _bare_poller(fake):
    import threading
    p = object.__new__(mp.MesoPoller)   # skip __init__ (no boto3 client)
    p.r2 = fake
    p.manifests = {}
    p._manifest_lock = threading.Lock()  # lanes + reconcile serialize here
    return p


class FrameTsFromKeyTests(unittest.TestCase):
    def test_second_and_minute_precision_and_garbage(self):
        t = dt.datetime(2026, 6, 11, 4, 41, 26, tzinfo=dt.timezone.utc)
        self.assertEqual(mp.frame_ts_from_key("x/ir/20260611T044126Z.png"), t)
        self.assertEqual(
            mp.frame_ts_from_key("x/ir/20260611T0441Z.png"),
            t.replace(second=0))
        self.assertIsNone(mp.frame_ts_from_key("x/ir/manifest.json"))
        self.assertIsNone(mp.frame_ts_from_key("x/ir/notatime.png"))

    def test_webp_frames_reconcile_like_png(self):
        # Post-cutover keys are .webp; a reconcile listing holds BOTH codecs
        # during the retention overlap and every frame must keep its stamp.
        t = dt.datetime(2026, 6, 11, 4, 41, 26, tzinfo=dt.timezone.utc)
        self.assertEqual(mp.frame_ts_from_key("x/ir/20260611T044126Z.webp"), t)
        self.assertEqual(
            mp.frame_ts_from_key("x/ir/20260611T0441Z.webp"),
            t.replace(second=0))
        self.assertIsNone(mp.frame_ts_from_key("x/ir/notatime.webp"))

    def test_meso_frame_key_carries_extension(self):
        ts = dt.datetime(2026, 6, 12, 6, 0, 29, tzinfo=dt.timezone.utc)
        self.assertTrue(
            mp.frame_key("goes19-m1", "ir", ts, ".webp")
            .endswith("/20260612T060029Z.webp"))
        # Default stays .png -- pre-existing callers/fixtures unaffected.
        self.assertTrue(
            mp.frame_key("goes19-m1", "ir", ts).endswith("/20260612T060029Z.png"))
        # Round-trip: what frame_key writes, frame_ts_from_key recovers.
        self.assertEqual(
            mp.frame_ts_from_key(mp.frame_key("goes19-m1", "ir", ts, ".webp")), ts)


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.sector = mp.MESO_SECTORS[0]
        self.band = mp.BANDS[0]
        self.slug = self.sector.slug
        self.now = mp.utcnow()

    def _times(self, *mins_ago):
        return [self.now - dt.timedelta(minutes=m) for m in mins_ago]

    def test_phantoms_dropped_and_storage_wins(self):
        real = self._times(30, 20, 10)
        stored = [_key(self.slug, self.band.key, t) for t in real]
        phantom = [_key(self.slug, self.band.key, t)
                   for t in self._times(25, 15)]
        mkey = mp.sector_manifest_key(self.slug)
        remote = {mkey: {"slug": self.slug, "bands": {self.band.key: {
            "label": self.band.label,
            "frames": [{"t": mp.iso_z(mp.frame_ts_from_key(k)), "key": k}
                       for k in sorted(stored + phantom)],
            "last_hash": "h-keep",
        }}}}
        fake = FakeR2(storage=stored, remote=remote)
        p = _bare_poller(fake)
        p.reconcile_manifests()
        man = fake.put[mkey]
        keys = [f["key"] for f in man["bands"][self.band.key]["frames"]]
        self.assertEqual(keys, sorted(stored))          # phantoms gone
        self.assertEqual(man["bands"][self.band.key]["last_hash"], "h-keep")
        self.assertEqual(p.manifests[self.slug], man)   # memory seeded

    def test_out_of_retention_objects_pruned(self):
        old = self.now - dt.timedelta(hours=mp.HISTORY_WINDOW_H + 2)
        fresh = self.now - dt.timedelta(minutes=5)
        k_old = _key(self.slug, self.band.key, old)
        k_new = _key(self.slug, self.band.key, fresh)
        fake = FakeR2(storage=[k_old, k_new])
        p = _bare_poller(fake)
        p.reconcile_manifests()
        man = fake.put[mp.sector_manifest_key(self.slug)]
        keys = [f["key"] for f in man["bands"][self.band.key]["frames"]]
        self.assertEqual(keys, [k_new])
        self.assertIn(k_old, fake.deleted)

    def test_listing_failure_keeps_prior_band_body(self):
        prior_frames = [{"t": mp.iso_z(self.now), "key": "kept/by/failure.png"}]
        mkey = mp.sector_manifest_key(self.slug)
        remote = {mkey: {"bands": {self.band.key: {
            "label": self.band.label, "frames": prior_frames,
            "last_hash": "h1"}}}}
        fake = FakeR2(storage=[], remote=remote,
                      fail_prefixes=(f"{mp.R2_PREFIX}/{self.slug}/{self.band.key}/",))
        p = _bare_poller(fake)
        p.reconcile_manifests()
        man = fake.put[mkey]
        self.assertEqual(man["bands"][self.band.key]["frames"], prior_frames)


class AppendFrameMemoryAuthorityTests(unittest.TestCase):
    def test_append_never_rereads_remote(self):
        sector = mp.MESO_SECTORS[0]
        band = mp.BANDS[0]
        now = mp.utcnow()
        k0 = _key(sector.slug, band.key, now - dt.timedelta(minutes=5))
        fake = FakeR2()
        p = _bare_poller(fake)
        p.manifests[sector.slug] = {
            "slug": sector.slug, "satellite": sector.satellite,
            "sector": sector.sector, "label": sector.label,
            "bands": {band.key: {"label": band.label,
                                 "frames": [{"t": mp.iso_z(now - dt.timedelta(minutes=5)),
                                             "key": k0}]}}}
        # an alien writer corrupts the REMOTE copy; memory must win
        fake.remote[mp.sector_manifest_key(sector.slug)] = {
            "bands": {band.key: {"frames": [{"t": "1999-01-01T00:00:00Z",
                                             "key": "alien.png"}]}}}
        k1 = _key(sector.slug, band.key, now)
        p.append_frame(sector, band, k1, now, "h-new")
        man = fake.put[mp.sector_manifest_key(sector.slug)]
        keys = [f["key"] for f in man["bands"][band.key]["frames"]]
        self.assertEqual(keys, [k0, k1])
        self.assertNotIn("alien.png", keys)
        self.assertEqual(man["bands"][band.key]["last_hash"], "h-new")


if __name__ == "__main__":
    unittest.main()
