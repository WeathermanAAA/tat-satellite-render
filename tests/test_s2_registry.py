#!/usr/bin/env python3
"""Pure-logic tests for s2_registry (no AWS/render deps) -- the Stage-2 product
registry generalization of s1_slots. Covers the §9.x completeness truth table
across all three substrates (cmip per-band / mcmip 1-file / ahi_l1b segments),
key parsing for each shape, product routing, deterministic R2 keys, and -- the
load-bearing safety check -- BYTE-PARITY: the registry's Stage-1 row reproduces
s1_slots' outputs exactly, so folding S1 into the registry changes nothing."""
import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s1_slots as S1  # noqa: E402
import s2_registry as R  # noqa: E402

UTC = dt.timezone.utc

# --- sample NOAA object keys (real shapes from the NODD buckets) -------------
CMIPM2_C13 = ("ABI-L2-CMIPM/2026/169/21/"
              "OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e20261692101042_c20261692101089.nc")
CMIPM2_C02 = ("ABI-L2-CMIPM/2026/169/21/"
              "OR_ABI-L2-CMIPM2-M6C02_G19_s20261692100572_e20261692101042_c20261692101089.nc")
CMIPM2_C10 = ("ABI-L2-CMIPM/2026/169/21/"
              "OR_ABI-L2-CMIPM2-M6C10_G19_s20261692108572_e20261692109043_c20261692109091.nc")
CMIPM1_C13 = ("ABI-L2-CMIPM/2026/169/21/"
              "OR_ABI-L2-CMIPM1-M6C13_G19_s20261692107272_e20261692107342_c20261692107379.nc")
CMIPF_C13 = ("ABI-L2-CMIPF/2026/001/00/"
             "OR_ABI-L2-CMIPF-M6C13_G19_s20260010000226_e20260010009546_c20260010010005.nc")
MCMIPM2 = ("ABI-L2-MCMIPM/2026/001/00/"
           "OR_ABI-L2-MCMIPM2-M6_G19_s20260010000301_e20260010000365_c20260010000427.nc")
MCMIPF = ("ABI-L2-MCMIPF/2026/001/00/"
          "OR_ABI-L2-MCMIPF-M6_G19_s20260010000226_e20260010009546_c20260010010005.nc")
AHI_TGT_B13_R301 = ("AHI-L1b-Target/2026/01/01/0000/"
                    "HS_H09_20260101_0000_B13_R301_R20_S0101.DAT.bz2")
AHI_TGT_B13_R303 = ("AHI-L1b-Target/2026/01/01/0000/"
                    "HS_H09_20260101_0000_B13_R303_R20_S0101.DAT.bz2")
AHI_FLDK_B13_S0110 = ("AHI-L1b-FLDK/2026/01/01/0000/"
                      "HS_H09_20260101_0000_B13_FLDK_R20_S0110.DAT.bz2")
RADC = ("ABI-L1b-RadC/2026/169/21/"
        "OR_ABI-L1b-RadC-M6C08_G19_s20261692106194_e20261692108567_c20261692109023.nc")


class TestParseGOES(unittest.TestCase):
    def test_cmip_meso_band(self):
        s = R.parse_key(CMIPM2_C13)
        self.assertEqual((s.family, s.substrate, s.sat), ("goes", "cmip", "19"))
        self.assertEqual(s.sector_token, "CMIPM2")
        self.assertEqual(s.band, 13)
        self.assertEqual(s.stamp, "20260618T210057Z")
        self.assertIsNone(s.segment)

    def test_cmip_full_disk(self):
        s = R.parse_key(CMIPF_C13)
        self.assertEqual(s.sector_token, "CMIPF")
        self.assertEqual(s.band, 13)
        self.assertEqual(s.stamp, "20260101T000022Z")

    def test_cmip_mode3_still_parses(self):
        s = R.parse_key(CMIPM2_C13.replace("-M6C13", "-M3C13"))
        self.assertIsNotNone(s)
        self.assertEqual(s.band, 13)

    def test_mcmip_meso_no_band(self):
        s = R.parse_key(MCMIPM2)
        self.assertEqual((s.family, s.substrate), ("goes", "mcmip"))
        self.assertEqual(s.sector_token, "MCMIPM2")
        self.assertIsNone(s.band)              # composite: no single band
        self.assertEqual(s.stamp, "20260101T000030Z")

    def test_mcmip_full_disk(self):
        s = R.parse_key(MCMIPF)
        self.assertEqual(s.sector_token, "MCMIPF")
        self.assertIsNone(s.band)
        self.assertEqual(s.stamp, "20260101T000022Z")

    def test_scan_start_agrees_with_s1_parser(self):
        # The C13 stamp must match s1_slots byte-for-byte (X-Scan-Time alignment).
        self.assertEqual(R.parse_key(CMIPM2_C13).stamp,
                         S1.parse_goes_key(CMIPM2_C13).stamp)

    def test_garbage(self):
        self.assertIsNone(R.parse_key(""))
        self.assertIsNone(R.parse_key("not/a/key.txt"))
        self.assertIsNone(R.parse_key(RADC))   # RadC is not a product we ingest


class TestParseAHI(unittest.TestCase):
    def test_target_subscan_offsets(self):
        s1 = R.parse_key(AHI_TGT_B13_R301)
        s3 = R.parse_key(AHI_TGT_B13_R303)
        self.assertEqual((s1.family, s1.substrate, s1.sat),
                         ("himawari", "ahi_l1b", "09"))
        self.assertEqual(s1.band, 13)
        self.assertEqual((s1.segment, s1.total_segments), (1, 1))
        # R301 -> block time; R303 -> +2 sub-scans (2 * 150 s = 5 min).
        self.assertEqual(s1.stamp, "20260101T000000Z")
        self.assertEqual(s3.stamp, "20260101T000500Z")

    def test_fldk_segments(self):
        s = R.parse_key(AHI_FLDK_B13_S0110)
        self.assertEqual(s.sector_token, "FLDK")
        self.assertEqual((s.segment, s.total_segments), (1, 10))
        self.assertEqual(s.stamp, "20260101T000000Z")


class TestRouting(unittest.TestCase):
    def test_c13_feeds_ir_irbd_truecolor(self):
        ids = {e.product_id for e in R.matching_entries(R.parse_key(CMIPM2_C13))}
        self.assertEqual(ids, {"goes19-meso2-ir", "goes19-meso2-irbd",
                               "goes19-meso2-truecolor"})

    def test_c02_feeds_only_truecolor(self):
        ids = {e.product_id for e in R.matching_entries(R.parse_key(CMIPM2_C02))}
        self.assertEqual(ids, {"goes19-meso2-truecolor"})

    def test_c10_feeds_nothing_seeded(self):
        self.assertEqual(R.matching_entries(R.parse_key(CMIPM2_C10)), [])

    def test_m1_not_claimed_by_m2_products(self):
        self.assertEqual(R.matching_entries(R.parse_key(CMIPM1_C13)), [])

    def test_mcmipf_feeds_fd(self):
        ids = {e.product_id for e in R.matching_entries(R.parse_key(MCMIPF))}
        self.assertEqual(ids, {"goes19-fd-mcmip"})

    def test_ahi_target_feeds_himawari(self):
        ids = {e.product_id for e in R.matching_entries(R.parse_key(AHI_TGT_B13_R301))}
        self.assertEqual(ids, {"himawari9-target-ir"})

    def test_no_cross_substrate_leak(self):
        # An MCMIP object must never be claimed by a CMIP product and vice versa.
        for e in R.matching_entries(R.parse_key(MCMIPF)):
            self.assertEqual(e.substrate, "mcmip")


class TestCompletenessTruthTable(unittest.TestCase):
    def _gate_for(self, product_id):
        e = R.REGISTRY_BY_ID[product_id]
        return e, e.new_gate()

    def test_cmip_single_band_immediate(self):
        e, g = self._gate_for("goes19-meso2-ir")
        s = R.parse_key(CMIPM2_C13)
        self.assertTrue(g.mark(e.gate_key(s), e.gate_item(s)))   # complete now
        self.assertFalse(g.mark(e.gate_key(s), e.gate_item(s)))  # idempotent

    def test_cmip_truecolor_accumulates_four_bands(self):
        e, g = self._gate_for("goes19-meso2-truecolor")
        self.assertEqual(e.required_items, frozenset({1, 2, 3, 13}))
        # Same slot time, four band files land one by one.
        stamp = "20260618T210057Z"
        for band, complete in [(2, False), (1, False), (3, False), (13, True)]:
            key = CMIPM2_C13.replace("-M6C13", f"-M6C{band:02d}")
            s = R.parse_key(key)
            self.assertEqual(e.gate_key(s), stamp)
            self.assertEqual(g.mark(e.gate_key(s), e.gate_item(s)), complete)
        self.assertTrue(g.is_complete(stamp))

    def test_cmip_truecolor_missing_band_never_completes(self):
        e, g = self._gate_for("goes19-meso2-truecolor")
        for band in (2, 1, 3):                                   # C13 never lands
            s = R.parse_key(CMIPM2_C13.replace("-M6C13", f"-M6C{band:02d}"))
            g.mark(e.gate_key(s), e.gate_item(s))
        self.assertFalse(g.is_complete("20260618T210057Z"))
        self.assertEqual(g.missing("20260618T210057Z"), {13})

    def test_mcmip_single_file_complete_on_arrival(self):
        e, g = self._gate_for("goes19-fd-mcmip")
        self.assertEqual(e.required_items, frozenset({R.MCMIP_ITEM}))
        s = R.parse_key(MCMIPF)
        self.assertTrue(g.mark(e.gate_key(s), e.gate_item(s)))
        self.assertTrue(g.is_complete(s.stamp))

    def test_ahi_target_one_segment_complete(self):
        e, g = self._gate_for("himawari9-target-ir")
        self.assertEqual(e.required_items, frozenset({(13, 1)}))
        s = R.parse_key(AHI_TGT_B13_R301)
        self.assertTrue(g.mark(e.gate_key(s), e.gate_item(s)))

    def test_ahi_fldk_multisegment_accumulates(self):
        # A synthetic FLDK product (10 segments x band 13) exercises the segment
        # matrix -- the doc §3.2 "AHI FLDK = all segments x all required bands".
        e = R.ProductEntry(
            product_id="test-fldk", family="himawari", substrate="ahi_l1b",
            bucket="noaa-himawari9", sat_num="09", s3_prefix="AHI-L1b-FLDK/",
            sns_filter_prefixes=("AHI-L1b-FLDK/",), accept_sectors=frozenset({"FLDK"}),
            channels=("clean_ir",), bands=(13,), ahi_segments=10,
            sat_key="himawari9", sector_key="fd", band_key="ir",
        )
        self.assertEqual(len(e.required_items), 10)
        g = e.new_gate()
        stamp = "20260101T000000Z"
        for seg in range(1, 11):
            key = AHI_FLDK_B13_S0110.replace("_S0110", f"_S{seg:02d}10")
            s = R.parse_key(key)
            self.assertEqual(e.gate_key(s), stamp)
            fired = g.mark(e.gate_key(s), e.gate_item(s))
            self.assertEqual(fired, seg == 10)                  # complete on 10th
        self.assertTrue(g.is_complete(stamp))

    def test_late_duplicate_never_refires(self):
        e, g = self._gate_for("goes19-meso2-ir")
        s = R.parse_key(CMIPM2_C13)
        self.assertTrue(g.mark(e.gate_key(s), e.gate_item(s)))
        self.assertFalse(g.mark(e.gate_key(s), e.gate_item(s)))  # redelivery = no-op


class TestR2Keys(unittest.TestCase):
    def test_frame_and_manifest_keys(self):
        e = R.REGISTRY_BY_ID["goes19-fd-mcmip"]
        self.assertEqual(e.product_path, "sat/goes19/fd/truecolor")
        self.assertEqual(e.frame_key("shadow", "20260101T000022Z"),
                         "shadow/sat/goes19/fd/truecolor/20260101T000022Z.webp")
        self.assertEqual(e.latest_times_key("shadow"),
                         "shadow/sat/goes19/fd/truecolor/latest_times.json")
        self.assertEqual(e.health_key("shadow"),
                         "shadow/sat/goes19/fd/truecolor/health.json")

    def test_new_product_has_no_prod_baseline(self):
        e = R.REGISTRY_BY_ID["goes19-fd-mcmip"]
        self.assertIsNone(e.prod_frame_key("20260101T000022Z"))

    def test_stamp_roundtrip(self):
        e = R.REGISTRY_BY_ID["goes19-fd-mcmip"]
        k = e.frame_key("shadow", "20260101T000022Z")
        self.assertEqual(e.stamp_from_frame_key(k), "20260101T000022Z")
        self.assertIsNone(e.stamp_from_frame_key("shadow/sat/.../latest_times.json"))

    def test_build_latest_times_shape(self):
        e = R.REGISTRY_BY_ID["himawari9-target-ir"]
        lt = e.build_latest_times(
            ["20260101T000500Z", "20260101T000000Z", "20260101T000500Z"],
            "shadow", dt.datetime(2026, 1, 1, 0, 6, tzinfo=UTC))
        self.assertEqual(lt["times"], ["20260101T000000Z", "20260101T000500Z"])
        self.assertEqual(lt["latest"], "20260101T000500Z")
        self.assertEqual(lt["product"], "sat/himawari9/meso/ir")
        self.assertEqual(lt["path"], "sat/himawari9/meso/ir/{t}.webp")
        self.assertEqual(lt["count"], 2)


class TestRenderBody(unittest.TestCase):
    def test_truecolor_render_dispatch(self):
        e = R.REGISTRY_BY_ID["goes19-meso2-truecolor"]
        body = e.render_body([1, 2, 3, 4], "2026-06-18T21:00:57+00:00")
        self.assertEqual(body["channel"], "true_color")
        self.assertEqual(body["enhancement"], "tat_neon")
        self.assertEqual(body["product"], "meso")
        self.assertEqual(body["satellite"], "GOES-East")
        self.assertEqual(body["format"], "webp")


class TestS1ByteParity(unittest.TestCase):
    """The registry's Stage-1 row must reproduce s1_slots EXACTLY -- otherwise
    folding S1 into the registry (Phase 2) would silently move a live shadow key
    or manifest. This is the safety gate for the generalization."""
    E = R.REGISTRY_BY_ID["goes19-meso2-ir"]
    STAMP = "20260618T210057Z"

    def test_frame_key_parity(self):
        self.assertEqual(self.E.frame_key("shadow", self.STAMP),
                         S1.shadow_frame_key("shadow", self.STAMP))

    def test_latest_times_key_parity(self):
        self.assertEqual(self.E.latest_times_key("shadow"),
                         S1.latest_times_key("shadow"))

    def test_health_key_parity(self):
        self.assertEqual(self.E.health_key("shadow"), S1.health_key("shadow"))

    def test_prod_frame_key_parity(self):
        self.assertEqual(self.E.prod_frame_key(self.STAMP),
                         S1.prod_frame_key(self.STAMP))

    def test_required_items_parity(self):
        self.assertEqual(self.E.required_items, S1.S1_REQUIRED_BANDS)

    def test_build_latest_times_parity(self):
        stamps = ["20260618T210157Z", "20260618T210057Z"]
        as_of = dt.datetime(2026, 6, 18, 21, 2, 0, tzinfo=UTC)
        self.assertEqual(self.E.build_latest_times(stamps, "shadow", as_of),
                         S1.build_latest_times(stamps, "shadow", as_of))

    def test_claims_matches_is_s1_slot(self):
        for key in (CMIPM2_C13, CMIPM1_C13, CMIPM2_C10, RADC):
            s2 = R.parse_key(key)
            s1 = S1.parse_goes_key(key)
            self.assertEqual(self.E.claims(s2), S1.is_s1_slot(s1),
                             msg=f"divergence on {key}")


if __name__ == "__main__":
    unittest.main()
