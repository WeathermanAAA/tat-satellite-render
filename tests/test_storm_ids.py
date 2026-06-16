"""The binding ID-join contract (CYCLOLAB_DESIGN.md §3.3).

The AL case is MANDATORY per the greenlight review: no Atlantic storm has
run the models pipeline this season, so nothing else exercises the
``AL -> "l"`` suffix - a first-letter slice would emit "05a" and silently
404 every Atlantic model frame the day the first hurricane opens its lab.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storm_ids import BASIN_SUFFIX, InvestSidError, parse_sid  # noqa: E402


class TestBasinSuffixMap(unittest.TestCase):

    def test_the_map_is_explicit_and_exact(self):
        # The review-mandated values, asserted literally.
        self.assertEqual(BASIN_SUFFIX,
                         {"AL": "l", "EP": "e", "CP": "c", "WP": "w"})

    def test_al_is_l_not_a(self):
        # THE trap: slicing "AL" gives "a"; ATCF says L = Atlantic.
        self.assertEqual(BASIN_SUFFIX["AL"], "l")
        self.assertNotEqual(BASIN_SUFFIX["AL"], "a")


class TestParseSid(unittest.TestCase):

    def test_mandatory_atlantic_case(self):
        # AL052026 -> atcf al052026 -> hafs 05l -> NHC AL052026
        ids = parse_sid("NHC_AL052026")
        self.assertEqual(ids.atcf_long, "al052026")
        self.assertEqual(ids.hafs_id, "05l")
        self.assertEqual(ids.nhc_id, "AL052026")
        self.assertEqual((ids.agency, ids.basin, ids.number, ids.year),
                         ("NHC", "AL", 5, 2026))

    def test_central_pacific(self):
        ids = parse_sid("NHC_CP012026")
        self.assertEqual(ids.hafs_id, "01c")
        self.assertEqual(ids.atcf_long, "cp012026")

    def test_east_pacific_amanda(self):
        # Tonight's real storm.
        ids = parse_sid("NHC_EP012026")
        self.assertEqual(ids.hafs_id, "01e")
        self.assertEqual(ids.atcf_long, "ep012026")
        self.assertEqual(ids.nhc_id, "EP012026")

    def test_west_pacific_jangmi(self):
        ids = parse_sid("JTWC_WP062026")
        self.assertEqual(ids.hafs_id, "06w")
        self.assertEqual(ids.agency, "JTWC")

    def test_invests_parse_as_invest_subset(self):
        # Stage C: invests (90-99) now PARSE - is_invest True, NO hafs_id
        # (they never run the HAFS pipeline), atcf/nhc ids still derived.
        for sid, atcf, nhc in (
                ("NHC_EP932026", "ep932026", "EP932026"),
                ("NHC_AL902026", "al902026", "AL902026"),
                ("JTWC_WP912026", "wp912026", "WP912026")):
            with self.subTest(sid=sid):
                ids = parse_sid(sid)
                self.assertTrue(ids.is_invest)
                self.assertEqual(ids.hafs_id, "")
                self.assertEqual(ids.atcf_long, atcf)
                self.assertEqual(ids.nhc_id, nhc)

    def test_designated_is_not_invest(self):
        self.assertFalse(parse_sid("NHC_EP012026").is_invest)

    def test_atcf_gap_50_to_89_rejected(self):
        for sid in ("NHC_EP502026", "NHC_AL892026"):
            with self.subTest(sid=sid):
                with self.assertRaises(ValueError):
                    parse_sid(sid)

    def test_unmapped_basin_fails_loud(self):
        with self.assertRaises(KeyError):
            parse_sid("JTWC_SH052026")

    def test_malformed_sids(self):
        for sid in ("EP012026", "NHC_EP01", "NHC_", "", "NHC_EPxx2026"):
            with self.subTest(sid=sid):
                with self.assertRaises(ValueError):
                    parse_sid(sid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
