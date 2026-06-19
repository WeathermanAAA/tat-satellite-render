#!/usr/bin/env python3
"""S1 multi-satellite source registry (STAGE A): GOES-18 (ABI twin) + Himawari-9
(AHI, segmented). Pure logic -- key parsing, per-source ownership/isolation, the
segment-completeness wiring, NOAA listing prefixes, and the /render body."""
import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import s1_slots as S        # noqa: E402
import s1_sources as SRC    # noqa: E402

UTC = dt.timezone.utc

G19 = ("ABI-L2-CMIPM/2026/169/21/"
       "OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e20261692101042_c20261692101089.nc")
G18 = ("ABI-L2-CMIPM/2026/169/21/"
       "OR_ABI-L2-CMIPM2-M6C13_G18_s20261692100572_e20261692101042_c20261692101089.nc")
G18_M1 = ("ABI-L2-CMIPM/2026/169/21/"
          "OR_ABI-L2-CMIPM1-M6C13_G18_s20261692107272_e20261692107342_c20261692107379.nc")


def ahi(seg, band=13, time="1850"):
    return (f"AHI-L1b-FLDK/2026/06/19/{time}/"
            f"HS_H09_20260619_{time}_B{band:02d}_FLDK_R20_S{seg:02d}10.DAT.bz2")


class RegistryTests(unittest.TestCase):
    def test_sources_present(self):
        self.assertEqual(set(SRC.SOURCES), {"goes19", "goes18", "himawari9"})

    def test_get_source_unknown_raises(self):
        with self.assertRaises(SystemExit):
            SRC.get_source("goes17")

    def test_goes19_matches_baseline_product_path(self):
        self.assertEqual(SRC.SOURCES["goes19"].product_path, S.S1_PRODUCT_PATH)

    def test_distinct_queues_and_topics(self):
        qs = {s.queue_name for s in SRC.SOURCES.values()}
        ts = {s.topic_arn for s in SRC.SOURCES.values()}
        self.assertEqual(len(qs), 3)
        self.assertEqual(len(ts), 3)
        self.assertTrue(SRC.SOURCES["himawari9"].topic_arn.endswith("NewHimawariNineObject"))
        self.assertTrue(SRC.SOURCES["goes18"].topic_arn.endswith("NewGOES18Object"))


class AbiSourceTests(unittest.TestCase):
    def test_goes18_parses_and_owns_g18_cmipm2(self):
        src = SRC.get_source("goes18")
        slot = SRC.parse(src, G18)
        self.assertIsNotNone(slot)
        self.assertEqual((slot.sat, slot.sector_token, slot.band), ("18", "CMIPM2", 13))
        self.assertTrue(SRC.is_ours(src, slot))

    def test_per_sat_isolation(self):
        # goes18 must NOT claim a GOES-19 object, and vice-versa.
        self.assertFalse(SRC.is_ours(SRC.get_source("goes18"), SRC.parse(SRC.get_source("goes18"), G19)))
        self.assertFalse(SRC.is_ours(SRC.get_source("goes19"), SRC.parse(SRC.get_source("goes19"), G18)))

    def test_goes18_rejects_m1(self):
        src = SRC.get_source("goes18")
        self.assertFalse(SRC.is_ours(src, SRC.parse(src, G18_M1)))  # CMIPM1 != CMIPM2

    def test_abi_completeness_single_object(self):
        src = SRC.get_source("goes18")
        self.assertEqual(SRC.gate_required(src), frozenset({13}))
        slot = SRC.parse(src, G18)
        self.assertEqual(SRC.gate_item(src, slot), 13)
        self.assertEqual(set(SRC.complete_scans(src, [G18])), {slot.stamp})

    def test_abi_render_body_has_meso_product(self):
        body = SRC.render_body(SRC.get_source("goes18"), [1, 2, 3, 4], "t")
        self.assertEqual(body["product"], "meso")
        self.assertEqual(body["satellite"], "GOES-West")
        self.assertEqual(body["channel"], "clean_ir")


class AhiSourceTests(unittest.TestCase):
    def setUp(self):
        self.src = SRC.get_source("himawari9")

    def test_parse_segment_and_stamp(self):
        slot = SRC.parse(self.src, ahi(1))
        self.assertEqual((slot.sat, slot.sector_token, slot.band, slot.segment),
                         ("H09", "FLDK", 13, 1))
        self.assertEqual(slot.stamp, "20260619T185000Z")  # 10-min slot, seconds 0

    def test_is_ours_band_filter(self):
        self.assertTrue(SRC.is_ours(self.src, SRC.parse(self.src, ahi(1))))
        self.assertFalse(SRC.is_ours(self.src, SRC.parse(self.src, ahi(1, band=7))))

    def test_segment_completeness(self):
        self.assertEqual(SRC.gate_required(self.src), frozenset(range(1, 11)))
        # 9 of 10 segments -> NOT complete (never a half-scan)
        nine = [ahi(s) for s in range(1, 10)]
        self.assertEqual(SRC.complete_scans(self.src, nine), {})
        # all 10 -> complete
        ten = [ahi(s) for s in range(1, 11)]
        comp = SRC.complete_scans(self.src, ten)
        self.assertEqual(set(comp), {"20260619T185000Z"})

    def test_complete_scans_ignores_other_band_segments(self):
        # B07 segments must not count toward B13 completeness
        keys = [ahi(s) for s in range(1, 11)] + [ahi(s, band=7) for s in range(1, 11)]
        comp = SRC.complete_scans(self.src, keys)
        self.assertEqual(set(comp), {"20260619T185000Z"})

    def test_two_scans_independent(self):
        keys = ([ahi(s, time="1850") for s in range(1, 11)]
                + [ahi(s, time="1840") for s in range(1, 6)])  # 1840 incomplete
        comp = SRC.complete_scans(self.src, keys)
        self.assertEqual(set(comp), {"20260619T185000Z"})       # only the complete one

    def test_ahi_render_body_no_product_hint(self):
        body = SRC.render_body(self.src, [1, 2, 3, 4], "t")
        self.assertNotIn("product", body)                       # picker chooses FLDK
        self.assertEqual(body["satellite"], "Himawari-Pacific")

    def test_noaa_prefixes_are_10min_slots(self):
        now = dt.datetime(2026, 6, 19, 18, 55, tzinfo=UTC)
        pre = SRC.noaa_prefixes(self.src, now, 30)
        self.assertTrue(all(p.startswith("AHI-L1b-FLDK/2026/06/19/") for p in pre))
        self.assertIn("AHI-L1b-FLDK/2026/06/19/1850/", pre)

    def test_slot_label_has_segment(self):
        lbl = SRC.slot_label(self.src, SRC.parse(self.src, ahi(3)))
        self.assertIn("himawari9", lbl)
        self.assertIn("S03", lbl)


if __name__ == "__main__":
    unittest.main()
