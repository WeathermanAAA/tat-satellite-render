#!/usr/bin/env python3
"""Pure-logic tests for s1_slots (no AWS/render deps) -- the never-miss CONTROL
logic that pixel-diff structurally cannot exercise (SATELLITE-REARCH §9.x)."""
import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s1_slots as S  # noqa: E402

UTC = dt.timezone.utc

CMIPM2_C13 = ("ABI-L2-CMIPM/2026/169/21/"
              "OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e20261692101042_c20261692101089.nc")
CMIPM1_C13 = ("ABI-L2-CMIPM/2026/169/21/"
              "OR_ABI-L2-CMIPM1-M6C13_G19_s20261692107272_e20261692107342_c20261692107379.nc")
CMIPM2_C10 = ("ABI-L2-CMIPM/2026/169/21/"
              "OR_ABI-L2-CMIPM2-M6C10_G19_s20261692108572_e20261692109043_c20261692109091.nc")
RADC = ("ABI-L1b-RadC/2026/169/21/"
        "OR_ABI-L1b-RadC-M6C08_G19_s20261692106194_e20261692108567_c20261692109023.nc")
GLM = ("GLM-L2-LCFA/2026/169/21/"
       "OR_GLM-L2-LCFA_G19_s20261692109000_e20261692109200_c20261692109221.nc")


class TestScanStart(unittest.TestCase):
    def test_scan_start_drops_tenths_matches_atcf_arithmetic(self):
        # s20261692100572 -> 2026 doy169 21:00:57 (tenths .2 dropped), like
        # satellites._parse_scan_start.
        d = S.scan_start_from_token("20261692100572")
        self.assertEqual(d, dt.datetime(2026, 6, 18, 21, 0, 57, tzinfo=UTC))

    def test_scan_start_matches_satellites_parser_if_available(self):
        # If the heavy stack is importable, assert byte-for-byte agreement with
        # the SAME parser prod uses (the X-Scan-Time stamp must align).
        try:
            import satellites
        except Exception:
            self.skipTest("satellites (heavy deps) not importable here")
        for key in (CMIPM2_C13, CMIPM1_C13, CMIPM2_C10):
            self.assertEqual(S.parse_goes_key(key).scan_start,
                             satellites._parse_scan_start(key))


class TestParseAndFilter(unittest.TestCase):
    def test_cmipm2_c13_is_s1(self):
        s = S.parse_goes_key(CMIPM2_C13)
        self.assertIsNotNone(s)
        self.assertEqual(s.sector_token, "CMIPM2")
        self.assertEqual(s.band, 13)
        self.assertEqual(s.sat, "19")
        self.assertEqual(s.stamp, "20260618T210057Z")
        self.assertTrue(S.is_s1_slot(s))

    def test_m1_is_not_s1(self):
        self.assertFalse(S.is_s1_slot(S.parse_goes_key(CMIPM1_C13)))

    def test_wrong_band_is_not_s1(self):
        self.assertFalse(S.is_s1_slot(S.parse_goes_key(CMIPM2_C10)))

    def test_non_cmipm_unparseable(self):
        self.assertIsNone(S.parse_goes_key(RADC))
        self.assertIsNone(S.parse_goes_key(GLM))
        self.assertFalse(S.is_s1_slot(S.parse_goes_key(RADC)))

    def test_mode3_token_accepted(self):
        # Historical Mode 3 (M3) must still parse.
        k = CMIPM2_C13.replace("-M6C13", "-M3C13")
        self.assertTrue(S.is_s1_slot(S.parse_goes_key(k)))

    def test_garbage_is_none(self):
        self.assertIsNone(S.parse_goes_key(""))
        self.assertIsNone(S.parse_goes_key("not/a/key.txt"))


class TestEnvelopeUnwrap(unittest.TestCase):
    def test_raw_s3_event(self):
        body = ('{"Records":[{"s3":{"object":{"key":"%s"}}}]}' % CMIPM2_C13)
        self.assertEqual(S.extract_object_key(body), CMIPM2_C13)

    def test_sns_envelope(self):
        inner = '{"Records":[{"s3":{"object":{"key":"%s"}}}]}' % CMIPM2_C13
        body = ('{"Type":"Notification","TopicArn":"x","Message":%s}'
                % __import__("json").dumps(inner))
        self.assertEqual(S.extract_object_key(body), CMIPM2_C13)

    def test_dict_body(self):
        body = {"Records": [{"s3": {"object": {"key": CMIPM2_C13}}}]}
        self.assertEqual(S.extract_object_key(body), CMIPM2_C13)

    def test_garbage(self):
        self.assertIsNone(S.extract_object_key("not json"))
        self.assertIsNone(S.extract_object_key('{"foo":1}'))
        self.assertIsNone(S.extract_object_key('{"Message":"not json"}'))


class TestKeys(unittest.TestCase):
    def test_shadow_key(self):
        self.assertEqual(S.shadow_frame_key("shadow", "20260618T210057Z"),
                         "shadow/sat/goes19/meso2/ir/20260618T210057Z.webp")

    def test_prod_key_matches_meso_layout(self):
        self.assertEqual(S.prod_frame_key("20260618T210057Z"),
                         "meso/goes19-m2/ir/20260618T210057Z.webp")

    def test_latest_times_key(self):
        self.assertEqual(S.latest_times_key("shadow"),
                         "shadow/sat/goes19/meso2/ir/latest_times.json")

    def test_stamp_roundtrip(self):
        k = S.shadow_frame_key("shadow", "20260618T210057Z")
        self.assertEqual(S.stamp_from_frame_key(k), "20260618T210057Z")
        self.assertEqual(S.stamp_from_frame_key(
            "meso/goes19-m2/ir/20260618T210057Z.png"), "20260618T210057Z")
        self.assertIsNone(S.stamp_from_frame_key("shadow/.../latest_times.json"))
        self.assertIsNone(S.stamp_from_frame_key("shadow/.../bogus.webp"))


class TestCompletenessGate(unittest.TestCase):
    def test_one_band_immediate(self):
        g = S.CompletenessGate(S.S1_REQUIRED_BANDS)  # {13}
        self.assertTrue(g.mark("slot", 13))         # first time complete
        self.assertTrue(g.is_complete("slot"))
        self.assertFalse(g.mark("slot", 13))        # idempotent: no re-fire

    def test_multiband_accumulates(self):
        g = S.CompletenessGate({1, 2, 3, 13})       # true-color analogue (S3)
        self.assertFalse(g.mark("tc", 1))
        self.assertEqual(g.missing("tc"), {2, 3, 13})
        self.assertFalse(g.mark("tc", 2))
        self.assertFalse(g.mark("tc", 3))
        self.assertFalse(g.is_complete("tc"))
        self.assertTrue(g.mark("tc", 13))           # 5th/last -> complete
        self.assertTrue(g.is_complete("tc"))

    def test_late_band_after_complete_no_refire(self):
        g = S.CompletenessGate({1, 2})
        self.assertFalse(g.mark("s", 1))
        self.assertTrue(g.mark("s", 2))             # complete
        self.assertFalse(g.mark("s", 1))            # duplicate late band: no re-fire

    def test_mcmip_single_item(self):
        g = S.CompletenessGate({"MCMIP"})           # composite = 1 item
        self.assertTrue(g.mark("slot", "MCMIP"))

    def test_seed_and_forget(self):
        g = S.CompletenessGate({13})
        g.seed_complete("slot")
        self.assertTrue(g.is_complete("slot"))
        self.assertFalse(g.mark("slot", 13))        # already done
        g.forget("slot")
        self.assertFalse(g.is_complete("slot"))


class TestLatestTimes(unittest.TestCase):
    def test_sorted_dedup_latest(self):
        lt = S.build_latest_times(
            ["20260618T210157Z", "20260618T210057Z", "20260618T210157Z"],
            "shadow", dt.datetime(2026, 6, 18, 21, 2, 0, tzinfo=UTC))
        self.assertEqual(lt["times"],
                         ["20260618T210057Z", "20260618T210157Z"])
        self.assertEqual(lt["latest"], "20260618T210157Z")
        self.assertEqual(lt["product"], "sat/goes19/meso2/ir")
        self.assertEqual(lt["path"], "sat/goes19/meso2/ir/{t}.webp")
        self.assertEqual(lt["as_of"], "2026-06-18T21:02:00Z")
        self.assertEqual(lt["count"], 2)

    def test_empty(self):
        lt = S.build_latest_times([], "shadow", dt.datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(lt["times"], [])
        self.assertIsNone(lt["latest"])


if __name__ == "__main__":
    unittest.main()
