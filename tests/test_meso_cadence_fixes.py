"""Meso cadence fixes: internal rate-limit exemption, antimeridian bboxes,
satellite hint, Target sub-scan freshness, FLDK slot probe, hot/cold lanes.

The starved-poller post-mortem in one file: the render service 429'd its own
poller, the picker handed the Bering M2 box to the wrong satellite, the
validator rejected dateline crossings outright, Himawari quantized to 10-min
slots (and 500'd while the newest FLDK slot was still uploading), and one
slow cold render blocked every hot unit.
"""

import asyncio
import datetime as dt
import os
import sys
import time
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
import meso_poller as mp
import satellites
from render import _fill_coord_nan
from truecolor import _upsample_azimuth
from vendor import ahi_loader


def _req(host="172.18.0.5", xff=None):
    headers = {}
    if xff is not None:
        headers["x-forwarded-for"] = xff
    return types.SimpleNamespace(
        headers=headers,
        client=types.SimpleNamespace(host=host) if host else None,
    )


class InternalExemptionTests(unittest.TestCase):
    def test_private_peer_no_xff_is_internal(self):
        for host in ("172.18.0.5", "10.1.2.3", "192.168.1.9", "127.0.0.1",
                     "fd12:3456::1", "::1"):
            self.assertTrue(app.is_internal_request(_req(host)), host)

    def test_xff_always_public(self):
        # Railway's edge appends XFF to every proxied request; an attacker
        # can prepend entries but never remove the proxy's own.
        self.assertFalse(app.is_internal_request(_req("172.18.0.5", xff="1.2.3.4")))

    def test_public_peer_is_limited(self):
        self.assertFalse(app.is_internal_request(_req("8.8.8.8")))
        self.assertFalse(app.is_internal_request(_req("testclient")))  # non-IP
        self.assertFalse(app.is_internal_request(_req(None)))

    def test_kill_switch_restores_unconditional_limiting(self):
        old = app.RATE_LIMIT_EXEMPT_INTERNAL
        app.RATE_LIMIT_EXEMPT_INTERNAL = False
        try:
            self.assertFalse(app.is_internal_request(_req("127.0.0.1")))
        finally:
            app.RATE_LIMIT_EXEMPT_INTERNAL = old

    def test_sliding_window_blocks_then_frees(self):
        lim = app.SlidingWindowLimiter("3/second")
        for _ in range(3):
            self.assertIsNone(lim.check("k"))
        retry = lim.check("k")
        self.assertIsNotNone(retry)
        self.assertLessEqual(retry, 1.0)
        time.sleep(1.05)
        self.assertIsNone(lim.check("k"))

    def test_keys_are_independent(self):
        lim = app.SlidingWindowLimiter("1/minute")
        self.assertIsNone(lim.check("a"))
        self.assertIsNotNone(lim.check("a"))
        self.assertIsNone(lim.check("b"))


class AntimeridianBboxTests(unittest.TestCase):
    def test_validator_accepts_crossing(self):
        v = app.RenderRequest(
            bbox=[141.5, 46.318, -140.321, 71.477], channel="clean_ir")
        self.assertEqual(v.bbox[0], 141.5)

    def test_validator_still_rejects_garbage(self):
        for bad in ([200.0, 0, 10, 10], [0, 0, 200.0, 10], [5, 0, 5, 10],
                    [0, 50, 10, 40], [180.0, 0, -180.0, 10]):
            with self.assertRaises(Exception, msg=bad):
                app.RenderRequest(bbox=bad, channel="clean_ir")

    def test_full_width_box_still_legal(self):
        # [-180, 180] spans the whole world and predates the crossing
        # support; its span must read 360, not (360 % 360) == 0.
        v = app.RenderRequest(bbox=[-180.0, -30.0, 180.0, 30.0],
                              channel="clean_ir")
        self.assertEqual(v.bbox[2], 180.0)
        self.assertEqual(satellites.bbox_lon_span(v.bbox), 360.0)
        self.assertGreater(
            app.compute_downsample_factor(v.bbox, "clean_ir"), 1)

    def test_downsample_uses_wrapped_span(self):
        crossing = app.compute_downsample_factor(
            [170.0, 0.0, -170.0, 20.0], "clean_ir")   # 20 deg wrapped
        plain = app.compute_downsample_factor(
            [-10.0, 0.0, 10.0, 20.0], "clean_ir")     # 20 deg plain
        self.assertEqual(crossing, plain)

    def test_bbox_lon_span(self):
        self.assertAlmostEqual(satellites.bbox_lon_span([170, 0, -170, 1]), 20.0)
        self.assertAlmostEqual(satellites.bbox_lon_span([-10, 0, 10, 1]), 20.0)

    def test_bbox_inside_wrap_aware(self):
        bering = [141.5, 46.318, -140.321, 71.477]
        # A crossing bbox inside the identical crossing extent.
        self.assertTrue(satellites._bbox_inside(
            bering, (141.5, 46.318, -140.321, 71.477), buffer=0.5))
        # A crossing bbox is NOT inside the (non-crossing) PACUS footprint —
        # the old elementwise compare said it was, which picked a CONUS
        # product with a giant no-data wedge.
        self.assertFalse(satellites._bbox_inside(
            bering, satellites.PACUS_FOOTPRINT))
        # Plain containment still behaves, including the buffer slack.
        self.assertTrue(satellites._bbox_inside(
            [-100, 20, -90, 30], (-152, 14, -77, 51)))
        self.assertTrue(satellites._bbox_inside(
            [-152.3, 20, -90, 30], (-152, 14, -77, 51), buffer=0.5))
        self.assertFalse(satellites._bbox_inside(
            [-160, 20, -90, 30], (-152, 14, -77, 51)))

    def test_fill_coord_nan(self):
        a = np.array([[np.nan, 1.0, 2.0],
                      [np.nan, np.nan, 5.0],
                      [7.0, 8.0, np.nan]])
        f = _fill_coord_nan(a)
        self.assertTrue(np.isfinite(f).all())
        # Finite values are untouched.
        self.assertEqual(f[0, 1], 1.0)
        self.assertEqual(f[2, 0], 7.0)
        # All-NaN except one: everything collapses to it.
        b = np.full((4, 4), np.nan)
        b[2, 1] = 4.25
        self.assertTrue((_fill_coord_nan(b) == 4.25).all())


class SatelliteHintTests(unittest.TestCase):
    BERING = [141.5, 46.318, -140.321, 71.477]
    NOW = dt.datetime(2026, 6, 12, 20, 0, tzinfo=dt.timezone.utc)

    def test_default_pick_lands_on_himawari(self):
        # Documents WHY the hint exists: by sub-point distance Himawari
        # "wins" the Bering box (39.8 deg vs GOES-West's 42.3) yet barely
        # images it.
        self.assertIs(satellites.pick_satellite(self.BERING, self.NOW),
                      satellites.HIMAWARI_PACIFIC)

    def test_hint_selects_owning_satellite(self):
        self.assertIs(
            satellites.pick_satellite(self.BERING, self.NOW,
                                      family_hint="GOES-West"),
            satellites.GOES_WEST)
        self.assertIs(
            satellites.pick_satellite(self.BERING, self.NOW,
                                      family_hint="goes-west"),
            satellites.GOES_WEST)

    def test_hint_still_enforces_coverage(self):
        atlantic = [-60.0, 10.0, -40.0, 25.0]
        with self.assertRaises(satellites.CoverageError):
            satellites.pick_satellite(atlantic, self.NOW,
                                      family_hint="Himawari-Pacific")

    def test_unknown_hint_falls_back(self):
        self.assertIs(
            satellites.pick_satellite(self.BERING, self.NOW,
                                      family_hint="METEOSAT-11"),
            satellites.HIMAWARI_PACIFIC)

    def test_sector_family_hint_mapping(self):
        from meso_sectors import MESO_SECTORS_BY_SLUG as by_slug
        self.assertEqual(mp.sector_family_hint(by_slug["goes19-m1"]), "GOES-East")
        self.assertEqual(mp.sector_family_hint(by_slug["goes18-m2"]), "GOES-West")
        self.assertEqual(mp.sector_family_hint(by_slug["himawari9-meso"]),
                         "Himawari-Pacific")


class FindBandAtMesoTests(unittest.TestCase):
    def test_cmipm_sector_listing(self):
        """CMIPM1/CMIPM2 are filename tokens, not prefixes — _find_band_at
        must list CMIPM and filter (listing 'CMIPM1' returns [] and killed
        every GOES meso true-color render)."""
        calls = []
        scan = dt.datetime(2026, 6, 12, 19, 47, 26, tzinfo=dt.timezone.utc)
        fake_files = [
            "noaa-goes19/ABI-L2-CMIPM/2026/163/19/OR_ABI-L2-CMIPM1-M6C01_G19_s20261631947266_e1_c1.nc",
            "noaa-goes19/ABI-L2-CMIPM/2026/163/19/OR_ABI-L2-CMIPM2-M6C01_G19_s20261631947296_e1_c1.nc",
        ]
        orig = satellites._list_hour
        satellites._list_hour = lambda bucket, product, channel, t: (
            calls.append(product) or list(fake_files))
        # An installed, OPEN loop (not asyncio.run, which tears the policy
        # loop down): the legacy live-smoke tests (test_himawari /
        # test_goes_west) still use get_event_loop().run_until_complete and
        # RuntimeError on 3.12 if a prior test left no current loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            resolved = loop.run_until_complete(satellites.GOES_EAST._find_band_at(
                "noaa-goes19", "CMIPM1", 1, scan))
        finally:
            satellites._list_hour = orig
        self.assertEqual(calls, ["CMIPM"])
        self.assertIn("-CMIPM1-", resolved.s3_key)
        self.assertEqual(resolved.scan_start, scan)


class TargetFreshnessTests(unittest.TestCase):
    def test_current_slot_probed_first(self):
        """back=0: the current 10-min folder usually has the freshest R30x;
        starting at back=1 quantized Himawari to 10+ minutes stale."""
        probed = []

        def fake_latest(fs, bucket, slot, band):
            probed.append(slot)
            return 2 if len(probed) == 1 else None

        orig = ahi_loader.latest_target_subscan
        ahi_loader.latest_target_subscan = fake_latest
        try:
            now = dt.datetime(2026, 6, 12, 19, 48, 59, tzinfo=dt.timezone.utc)
            resolved = satellites.HIMAWARI_PACIFIC._resolve_target_sync(
                now, 13, False)
        finally:
            ahi_loader.latest_target_subscan = orig
        self.assertEqual(probed[0].minute, 40)            # current slot first
        self.assertEqual(resolved.product, "Target2")
        self.assertEqual(resolved.scan_start.minute, 42)  # slot + 2.5 min
        self.assertEqual(resolved.scan_start.second, 30)

    @staticmethod
    def _fldk_files(band: int, segs: range, total: int) -> list:
        return [f"HS_H09_20260612_2000_B{band:02d}_FLDK_R20_S{s:02d}{total:02d}.DAT.bz2"
                for s in segs]

    def test_fldk_probe_requires_complete_segment_sets(self):
        full = lambda b: self._fldk_files(b, range(1, 11), 10)
        listings = {
            "2010": [],                                          # nothing yet
            # B03 half-published: presence alone would pick this slot and
            # ship a part-black true-color frame past the degenerate guard.
            "2000": (full(1) + full(2) + full(4) + full(13)
                     + self._fldk_files(3, range(1, 6), 10)),
            "1950": full(1) + full(2) + full(3) + full(4) + full(13),
        }

        class FakeFS:
            def ls(self, prefix):
                key = prefix.rstrip("/").rsplit("/", 1)[-1]
                out = listings.get(key)
                if out is None:
                    raise FileNotFoundError(prefix)
                return out

        orig = satellites._get_fs
        satellites._get_fs = lambda: FakeFS()
        try:
            snapped = dt.datetime(2026, 6, 12, 20, 10, tzinfo=dt.timezone.utc)
            slot = satellites.HIMAWARI_PACIFIC._first_available_fldk_slot_sync(
                snapped, [1, 2, 3, 4, 13])
        finally:
            satellites._get_fs = orig
        self.assertEqual((slot.hour, slot.minute), (19, 50))

    def test_fldk_band_complete(self):
        comp = satellites.HimawariPacificSatellite._fldk_band_complete
        self.assertTrue(comp(self._fldk_files(3, range(1, 11), 10), 3))
        self.assertFalse(comp(self._fldk_files(3, range(1, 10), 10), 3))
        # H8-era single-file repack (S0101) is complete with one segment.
        self.assertTrue(comp(self._fldk_files(13, range(1, 2), 1), 13))
        self.assertFalse(comp([], 3))


class TargetSubScanEnumerationTests(unittest.TestCase):
    """True 2.5-min capture: address EVERY R30x Target sub-scan per 10-min slot, not
    just the freshest. Covers target_subscans, nearest-sub resolution for a specific
    requested time, and the poller's per-slot obs-time enumeration."""

    @staticmethod
    def _target_files(band, subs, res="R20"):
        return [f"HS_H09_20260612_1940_B{band:02d}_R30{s}_{res}_S0101.DAT.bz2"
                for s in subs]

    def _fakefs(self, files):
        class FakeFS:
            def ls(self, prefix):
                return files
        return FakeFS()

    def test_target_subscans_returns_full_set(self):
        slot = dt.datetime(2026, 6, 12, 19, 40, tzinfo=dt.timezone.utc)
        fs = self._fakefs(self._target_files(13, [1, 2, 3, 4]))
        self.assertEqual(ahi_loader.target_subscans(fs, "noaa-himawari9", slot, 13),
                         {1, 2, 3, 4})
        # latest_target_subscan stays the MAX of that set (the 'latest' resolve)
        self.assertEqual(ahi_loader.latest_target_subscan(fs, "x", slot, 13), 4)
        # a band missing two sub-scans reports only what it has
        fs2 = self._fakefs(self._target_files(13, [1, 3]))
        self.assertEqual(ahi_loader.target_subscans(fs2, "x", slot, 13), {1, 3})

    def test_resolve_nearest_picks_requested_subscan_not_latest(self):
        # only R301 and R303 present: a request NEAR R301 must resolve R301 (proving
        # 'nearest', not the old 'always max').
        orig = satellites._get_fs
        satellites._get_fs = lambda: self._fakefs(self._target_files(13, [1, 3]))
        try:
            slot = dt.datetime(2026, 6, 12, 19, 40, tzinfo=dt.timezone.utc)
            r1 = satellites.HIMAWARI_PACIFIC._resolve_target_sync(
                slot + dt.timedelta(seconds=10), 13, True)     # near R301
            r3 = satellites.HIMAWARI_PACIFIC._resolve_target_sync(
                slot + dt.timedelta(seconds=300), 13, True)    # near R303
            # nearest_to_target=False still returns the freshest (max)
            rlatest = satellites.HIMAWARI_PACIFIC._resolve_target_sync(
                slot + dt.timedelta(seconds=10), 13, False)
        finally:
            satellites._get_fs = orig
        self.assertEqual((r1.product, r1.scan_start.minute, r1.scan_start.second),
                         ("Target1", 40, 0))
        self.assertEqual((r3.product, r3.scan_start.minute, r3.scan_start.second),
                         ("Target3", 45, 0))
        self.assertEqual(rlatest.product, "Target3")           # max present

    def test_poller_enumerates_all_obs_times(self):
        orig = mp._get_fs
        mp._get_fs = lambda: self._fakefs(self._target_files(13, [1, 2, 3, 4]))
        try:
            sector = next(s for s in mp.MESO_SECTORS if s.family == "himawari")
            times = mp.himawari_target_subscan_times(sector, "clean_ir", back_slots=0)
        finally:
            mp._get_fs = orig
        self.assertEqual(len(times), 4)                        # all four R30x
        self.assertEqual({(b - a).total_seconds() for a, b in zip(times, times[1:])},
                         {150.0})                              # exactly 2.5 min apart
        # true-color resolves on visible_red (B03), so it enumerates too
        self.assertEqual(mp._himawari_ref_band("true_color"), 3)
        self.assertIsNone(mp._himawari_ref_band("nonsense"))


class LaneTests(unittest.TestCase):
    def test_units_split_hot_cold(self):
        p = object.__new__(mp.MesoPoller)
        units = {}
        for sector in mp.MESO_SECTORS:
            for band in mp.BANDS:
                units[(sector.slug, band.key)] = mp.Unit(sector=sector, band=band)
        hot = [u for u in units.values() if u.band.hot]
        cold = [u for u in units.values() if not u.band.hot]
        self.assertEqual(len(hot), 10)    # 5 sectors x (ir, irbd)
        self.assertEqual(len(cold), 20)   # 5 sectors x (wv_up, wv_low, tc, swir)

    def test_lane_cadence(self):
        p = object.__new__(mp.MesoPoller)
        hot = mp.Lane("hot", [None] * 10, "http://h", 1.0, drain_all=True)
        cold = mp.Lane("cold", [None] * 20, "http://c", 1.0, drain_all=False)
        self.assertEqual(p.lane_cadence(hot), mp.CADENCE_TARGET_S)
        self.assertEqual(p.lane_cadence(cold),
                         max(mp.COLD_CADENCE_TARGET_S, 20 * 1.0))

    def test_lanes_have_isolated_plumbing(self):
        a = mp.Lane("hot", [], "http://h", 1.0, drain_all=True)
        b = mp.Lane("cold", [], "http://c", 1.0, drain_all=False)
        self.assertIsNot(a.session, b.session)
        self.assertIsNot(a.limiter, b.limiter)
        self.assertNotEqual(a.render_url, b.render_url)
        # A tripped cold breaker must not read as a hot trip.
        b.circuit_open_until = time.monotonic() + 60
        self.assertLess(a.circuit_open_until, time.monotonic())


class HeaderCaseTests(unittest.TestCase):
    def test_scan_time_header_found_despite_lowercasing(self):
        """uvicorn lowercases response headers; the old literal
        .get("X-Scan-Time") missed, every frame fell back to the stale
        discovery slot time, and newer scans OVERWROTE the previous frame
        under the old stamp (Himawari Target collapsed onto 10-min slots)."""
        lowered = {"x-scan-time": "2026-06-12T20:16:56+00:00",
                   "content-type": "image/webp"}
        self.assertEqual(mp.header_get(lowered, "X-Scan-Time"),
                         "2026-06-12T20:16:56+00:00")
        self.assertIsNone(mp.header_get(lowered, "X-Timestamp"))
        import floater_poller as fp
        self.assertEqual(fp.header_get(lowered, "X-Scan-Time"),
                         "2026-06-12T20:16:56+00:00")

    def test_floater_frame_key_is_second_precision(self):
        """With true scan stamps, two GOES meso scans can land in the same
        wall-clock minute — a minute-precision key would make the second
        overwrite the first's object while the manifest lists both."""
        import floater_poller as fp
        a = dt.datetime(2026, 6, 12, 19, 47, 2, tzinfo=dt.timezone.utc)
        b = dt.datetime(2026, 6, 12, 19, 47, 59, tzinfo=dt.timezone.utc)
        ka = fp.frame_key("ep03", "ir", a, ".webp")
        kb = fp.frame_key("ep03", "ir", b, ".webp")
        self.assertNotEqual(ka, kb)
        self.assertTrue(ka.endswith("20260612T194702Z.webp"), ka)


class EffectiveExtentTests(unittest.TestCase):
    def _grid(self, lon0, lon1, lat0, lat1, n=21):
        lons, lats = np.meshgrid(np.linspace(lon0, lon1, n),
                                 np.linspace(lat1, lat0, n))
        return lats, lons

    def test_full_valid_keeps_request(self):
        from render import _effective_extent
        bbox = [-85.759, 32.274, -71.996, 45.808]
        lats, lons = self._grid(bbox[0], bbox[2], bbox[1], bbox[3])
        valid = np.ones_like(lats, dtype=bool)
        eff = _effective_extent(lats, lons, valid, bbox, bbox[2] - bbox[0])
        self.assertEqual(eff, (bbox[0], bbox[2], bbox[1], bbox[3]))

    def test_limb_corner_cropped(self):
        from render import _effective_extent
        bbox = [-85.0, 30.0, -71.0, 44.0]
        lats, lons = self._grid(bbox[0], bbox[2], bbox[1], bbox[3])
        # Off-disk: everything west of -78 AND south of 37 invalid.
        valid = (lons >= -78.0) & (lats >= 37.0)
        lo, hi, la, lb = _effective_extent(lats, lons, valid, bbox, 14.0)
        self.assertGreaterEqual(lo, -78.1)
        self.assertGreaterEqual(la, 36.9)
        self.assertEqual((hi, lb), (-71.0, 44.0))

    def test_crossing_unwrap(self):
        from render import _effective_extent
        bbox = [141.5, 46.318, -140.321, 71.477]
        span = (bbox[2] - bbox[0]) % 360.0
        # Grid over the unwrapped range, lons stored wrapped to ±180.
        lats, lons_uw = self._grid(141.5, 141.5 + span, bbox[1], bbox[3])
        lons = ((lons_uw + 180.0) % 360.0) - 180.0
        valid = lons_uw >= 160.0          # west 18.5 deg off-disk
        lo, hi, la, lb = _effective_extent(lats, lons, valid, bbox, span)
        self.assertGreaterEqual(lo, 159.9)
        self.assertAlmostEqual(hi, 141.5 + span, places=6)

    def test_empty_valid_falls_back(self):
        from render import _effective_extent
        bbox = [-85.0, 30.0, -71.0, 44.0]
        lats, lons = self._grid(*[bbox[0], bbox[2], bbox[1], bbox[3]])
        eff = _effective_extent(lats, lons, np.zeros_like(lats, bool),
                                bbox, 14.0)
        self.assertEqual(eff, (-85.0, -71.0, 30.0, 44.0))


class AzimuthUpsampleTests(unittest.TestCase):
    def test_wrap_crossing_does_not_tear(self):
        # 358..2 degrees across the field: naive bilinear would pass
        # through ~180 at the wrap; component interpolation stays near 0/360.
        az = np.array([[358.0, 359.0, 1.0, 2.0]] * 3)
        up = _upsample_azimuth(az, (6, 8))
        dist = np.minimum(up % 360.0, 360.0 - (up % 360.0))
        self.assertTrue((dist <= 4.0).all(), up)


class AhiColCropTests(unittest.TestCase):
    def test_bbox_col_range_window(self):
        seg = types.SimpleNamespace(
            sub_lon=140.7, cfac=20466275, lfac=20466275,
            coff=2750.5, loff=2750.5)
        lo, hi = ahi_loader._bbox_col_range(
            seg, (136.778, 21.496, 147.701, 32.35), 5500)
        self.assertGreater(lo, 0)
        self.assertLess(hi, 5500)
        self.assertGreater(hi - lo, 100)   # plausible window, not degenerate

    def test_off_disk_bbox_falls_back_to_full_width(self):
        seg = types.SimpleNamespace(
            sub_lon=140.7, cfac=20466275, lfac=20466275,
            coff=2750.5, loff=2750.5)
        lo, hi = ahi_loader._bbox_col_range(seg, (-60.0, 10.0, -40.0, 25.0), 5500)
        self.assertEqual((lo, hi), (0, 5500))


if __name__ == "__main__":
    unittest.main()
