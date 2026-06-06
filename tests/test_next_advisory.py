"""Tests for kml_advisories.parse_next_advisory against REAL NHC TCP footers.

The advisory-countdown shell shows the NEXT advisory time NHC STATES in the
Public Advisory (TCP) footer - never a cadence guessed from the wall clock.
parse_next_advisory pulls those stated, dateless local-zone times and
resolves them against the advisory's own issuance.

Real fixtures in ``tests/fixtures/cyclolab/`` (the never-invent rule applies
to inputs too):

``tcp_amanda_current.shtml``
    Tropical Storm Amanda (EP012026) Public Advisory #14, the LIVE 6-hourly
    "complete only" case, fetched from
    https://www.nhc.noaa.gov/text/MIATCPEP1.shtml
    Issued ``500 PM HST Fri Jun 05 2026`` (= 2026-06-06T03:00:00Z). Footer
    states only ``Next complete advisory at 1100 PM HST.``
        1100 PM HST = 23:00 HST; HST is UTC-10 -> 09:00 UTC the next day,
        and 11 PM is after the 5 PM issuance so it stays on Jun 05 HST:
        -> 2026-06-06T09:00:00Z.

``tcp_intermediate_sample.shtml``
    Hurricane Milton (AL142024) Public Advisory #11 - the 3-hourly case with
    intermediates active (Florida hurricane/storm-surge warnings in effect).
    Archive URL:
    https://www.nhc.noaa.gov/archive/2024/al14/al142024.public.011.shtml
    Issued ``400 PM CDT Mon Oct 07 2024`` (= 2024-10-07T21:00:00Z). Footer:
        Next intermediate advisory at 700 PM CDT.   (CDT = UTC-5)
        Next complete advisory at 1000 PM CDT.
        700 PM CDT  = 19:00 -> 2024-10-08T00:00:00Z (intermediate)
        1000 PM CDT = 22:00 -> 2024-10-08T03:00:00Z (complete)
    intermediate (00:00Z) is the earlier and wins next_advisory_utc.

``tcp_last_advisory_synthetic.txt``
    SYNTHETIC (labelled in the name) minimal last-advisory product using the
    REAL NHC final-advisory phrasing ("This is the last public advisory
    issued by the National Hurricane Center on this system.") verbatim from
    Milton's true final advisory #23
    (https://www.nhc.noaa.gov/archive/2024/al14/al142024.public.023.shtml).
    No stated time -> graceful no-countdown (stated False).

Runnable via ``python -m pytest tests/test_next_advisory.py -q`` AND
``python -m unittest tests.test_next_advisory``.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kml_advisories import (  # noqa: E402
    AdvisoryParseError,
    parse_next_advisory,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "cyclolab"
AMANDA = FIX / "tcp_amanda_current.shtml"
MILTON = FIX / "tcp_intermediate_sample.shtml"
LAST = FIX / "tcp_last_advisory_synthetic.txt"

# Issuance instants, parsed by hand from each fixture's header line (the
# clock-free source-freshness rule - the live poller passes the advisory's
# own issued_utc, which it already parses upstream).
AMANDA_ISSUED = "2026-06-06T03:00:00Z"   # 500 PM HST Fri Jun 05 2026
MILTON_ISSUED = "2024-10-07T21:00:00Z"   # 400 PM CDT Mon Oct 07 2024


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


class TestAmandaCurrentComplete(unittest.TestCase):
    """Amanda #14: the 6-hourly complete-only footer."""

    def setUp(self):
        self.result = parse_next_advisory(_read(AMANDA), AMANDA_ISSUED)

    def test_next_advisory_is_stated_complete_time(self):
        # 1100 PM HST -> 2026-06-06T09:00:00Z (literal, computed by hand).
        self.assertEqual(self.result["next_advisory_utc"],
                         "2026-06-06T09:00:00Z")

    def test_kind_is_complete(self):
        self.assertEqual(self.result["kind"], "complete")

    def test_complete_mirrors_next_when_only_complete_stated(self):
        self.assertEqual(self.result["next_complete_utc"],
                         self.result["next_advisory_utc"])
        self.assertEqual(self.result["next_complete_utc"],
                         "2026-06-06T09:00:00Z")

    def test_stated_true(self):
        self.assertTrue(self.result["stated"])

    def test_fixture_really_is_complete_only(self):
        # Guard the fixture: it must NOT contain an intermediate line, or the
        # "complete only" assertions above would be testing the wrong thing.
        body = _read(AMANDA)
        self.assertIn("Next complete advisory at 1100 PM HST", body)
        self.assertNotIn("Next intermediate advisory", body)


class TestMiltonIntermediate(unittest.TestCase):
    """Milton #11: both intermediate and complete stated (3-hourly)."""

    def setUp(self):
        self.result = parse_next_advisory(_read(MILTON), MILTON_ISSUED)

    def test_kind_is_intermediate(self):
        self.assertEqual(self.result["kind"], "intermediate")

    def test_intermediate_time_pinned(self):
        # 700 PM CDT -> 2024-10-08T00:00:00Z.
        self.assertEqual(self.result["next_advisory_utc"],
                         "2024-10-08T00:00:00Z")

    def test_complete_time_pinned(self):
        # 1000 PM CDT -> 2024-10-08T03:00:00Z.
        self.assertEqual(self.result["next_complete_utc"],
                         "2024-10-08T03:00:00Z")

    def test_intermediate_strictly_before_complete(self):
        self.assertLess(self.result["next_advisory_utc"],
                        self.result["next_complete_utc"])

    def test_stated_true(self):
        self.assertTrue(self.result["stated"])

    def test_fixture_really_states_both(self):
        body = _read(MILTON)
        self.assertIn("Next intermediate advisory at 700 PM CDT", body)
        self.assertIn("Next complete advisory at 1000 PM CDT", body)


class TestDayRollover(unittest.TestCase):
    """Stated footer time earlier than issuance resolves to the NEXT day.

    Uses the REAL Amanda text but a SYNTHESIZED issued_utc late enough in the
    HST day that "1100 PM HST" has already passed at issuance, so it must roll
    forward one calendar day (NHC never states a time already in the past).
    """

    def test_rolls_forward_one_day(self):
        # issued 2026-06-07T09:30:00Z = 23:30 HST Jun 06. Stated 11 PM HST
        # (23:00 HST Jun 06) is already past, so it rolls to 23:00 HST Jun 07
        # = 2026-06-08T09:00:00Z.
        result = parse_next_advisory(_read(AMANDA), "2026-06-07T09:30:00Z")
        self.assertEqual(result["next_advisory_utc"],
                         "2026-06-08T09:00:00Z")
        self.assertEqual(result["kind"], "complete")
        self.assertTrue(result["stated"])

    def test_no_rollover_when_time_is_still_ahead(self):
        # Sanity counterpart: at the real issuance, 11 PM HST is still ahead,
        # so NO rollover (stays on the issuance HST date).
        result = parse_next_advisory(_read(AMANDA), AMANDA_ISSUED)
        self.assertEqual(result["next_advisory_utc"],
                         "2026-06-06T09:00:00Z")


class TestLastAdvisory(unittest.TestCase):
    """A final advisory has no countdown - stated False / all None."""

    def setUp(self):
        # Any plausible issuance; the parser must short-circuit on the
        # last-advisory phrasing regardless of issued_utc.
        self.result = parse_next_advisory(_read(LAST), "2026-06-08T03:00:00Z")

    def test_stated_false(self):
        self.assertFalse(self.result["stated"])

    def test_all_fields_none(self):
        self.assertIsNone(self.result["next_advisory_utc"])
        self.assertIsNone(self.result["kind"])
        self.assertIsNone(self.result["next_complete_utc"])

    def test_fixture_uses_real_last_advisory_phrasing(self):
        body = _read(LAST)
        self.assertIn(
            "This is the last public advisory issued by the National "
            "Hurricane", body)


class TestShtmlVsBareText(unittest.TestCase):
    """The .shtml page and the bare <pre> product parse identically."""

    def test_amanda_shtml_equals_bare(self):
        shtml = _read(AMANDA)
        m = re.search(r"<pre>(.*?)</pre>", shtml, re.S | re.I)
        self.assertIsNotNone(m, "fixture should carry a <pre> block")
        bare = m.group(1)
        self.assertEqual(
            parse_next_advisory(bare, AMANDA_ISSUED),
            parse_next_advisory(shtml, AMANDA_ISSUED),
        )

    def test_milton_shtml_equals_bare(self):
        shtml = _read(MILTON)
        m = re.search(r"<pre>(.*?)</pre>", shtml, re.S | re.I)
        self.assertIsNotNone(m)
        bare = m.group(1)
        self.assertEqual(
            parse_next_advisory(bare, MILTON_ISSUED),
            parse_next_advisory(shtml, MILTON_ISSUED),
        )


class TestGarbageInput(unittest.TestCase):
    """Garbage input raises; an absent footer does NOT (graceful None)."""

    def test_none_raises(self):
        with self.assertRaises(AdvisoryParseError):
            parse_next_advisory(None, AMANDA_ISSUED)  # type: ignore[arg-type]

    def test_empty_raises(self):
        with self.assertRaises(AdvisoryParseError):
            parse_next_advisory("", AMANDA_ISSUED)

    def test_non_string_raises(self):
        with self.assertRaises(AdvisoryParseError):
            parse_next_advisory(12345, AMANDA_ISSUED)  # type: ignore[arg-type]

    def test_bad_issued_utc_raises(self):
        with self.assertRaises(AdvisoryParseError):
            parse_next_advisory(_read(AMANDA), "not-a-timestamp")

    def test_absent_footer_is_graceful_none_not_error(self):
        # A product with text but no NEXT ADVISORY block is the ending-storm
        # graceful case, NOT garbage - must return stated False, never raise.
        no_footer = (
            "BULLETIN\nTropical Storm Foo Advisory Number 1\n"
            "500 PM HST Fri Jun 05 2026\n\nDISCUSSION\n----------\n"
            "Some discussion text with no next-advisory footer.\n$$\n")
        result = parse_next_advisory(no_footer, AMANDA_ISSUED)
        self.assertFalse(result["stated"])
        self.assertIsNone(result["next_advisory_utc"])


if __name__ == "__main__":
    unittest.main()
