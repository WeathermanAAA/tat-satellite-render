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

from storm_ids import (  # noqa: E402
    BASIN_SUFFIX, InvestSidError, is_real_storm_name, parse_sid)


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


class TestIsRealStormName(unittest.TestCase):
    """The designation-vs-real-name classifier both live pollers share to
    demote-guard NHC CurrentStorms against a lagging synoptic-time upgrade
    (bep042026.dat "DOUGLAS" while CurrentStorms still said "Four-E")."""

    def test_genuine_seasonal_names_are_real(self):
        for nm in ("DOUGLAS", "Amanda", "BORIS", "CRISTINA", "ELIDA",
                   "JANGMI", "ROSE"):
            with self.subTest(name=nm):
                self.assertTrue(is_real_storm_name(nm))

    def test_placeholders_are_not_real(self):
        for nm in ("", "   ", None, "INVEST", "UNNAMED", "NAMELESS"):
            with self.subTest(name=nm):
                self.assertFalse(is_real_storm_name(nm))

    def test_spelled_ordinal_designations_are_not_real(self):
        # b-deck bare word AND CurrentStorms basin-suffixed form.
        for nm in ("ONE", "FOUR", "Four-E", "TWENTY-ONE", "TWENTY-ONE-E",
                   "FIFTY-NINE", "THIRTEEN"):
            with self.subTest(name=nm):
                self.assertFalse(is_real_storm_name(nm))

    def test_numeric_designation_fallbacks_are_not_real(self):
        # parse_bdeck's "#NN" and knackwx's "<num><letter>".
        for nm in ("#04", "#4", "04E", "4E", "10W", "01L", "12"):
            with self.subTest(name=nm):
                self.assertFalse(is_real_storm_name(nm))

    def test_hyphen_only_peels_a_basin_letter_suffix(self):
        # A real name can't be truncated: no genuine name is "<word>-<E/C/L/W>".
        self.assertTrue(is_real_storm_name("JEAN-PIERRE"))   # not a basin suffix
        self.assertFalse(is_real_storm_name("FOUR-E"))

    def test_lockstep_with_cyclolab_shell_number_words(self):
        # The comment's contract: the number-word set is in sync with
        # cyclolab_shell._ptc_number_words() (keep BOTH in step).
        import storm_ids
        import cyclolab_shell
        self.assertEqual(storm_ids._build_number_words(),
                         cyclolab_shell._ptc_number_words())


if __name__ == "__main__":
    unittest.main(verbosity=2)
