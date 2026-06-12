"""Prompt retirement of dissipated NHC storms (retire_dissipated_named).

The lingering-floater bug: a dissipating TD's b-deck just STOPS while its
final row still reads a tropical nature, so the tracks feed's is_active
degrades to its 60 h staleness window and the dead storm's floater stayed on
/satellite/ for ~2.5 days (CRISTINA). NHC's CurrentStorms.json drops a storm
with its final advisory -- the authoritative signal this module now uses.
"""

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import floater_poller as fp  # noqa: E402

NOW = dt.datetime(2026, 6, 12, 22, 0, tzinfo=dt.timezone.utc)


def _storm(slug, basin, name, last_fix):
    return fp.Storm(
        sid=f"{basin.lower()}xx2026", slug=slug, name=name, basin=basin,
        lat=12.0, lon=-89.0, category="TD", intensity_kt=30.0,
        last_fix=last_fix,
    )


class RetireDissipatedTests(unittest.TestCase):
    def test_cristina_case_retired(self):
        # EP storm, absent from CurrentStorms, last fix 46 h old -> retired.
        named = {"ep03": _storm("ep03", "EP", "CRISTINA", "2026-06-11T00:00:00")}
        out = fp.retire_dissipated_named(named, {}, True, NOW)
        self.assertEqual(out, {})

    def test_current_storms_listing_keeps_storm(self):
        # Listed in CurrentStorms -> floats regardless of fix age.
        named = {"ep05": _storm("ep05", "EP", "AMANDA", "2026-06-10T00:00:00")}
        cur = {"ep05": named["ep05"]}
        out = fp.retire_dissipated_named(named, cur, True, NOW)
        self.assertIn("ep05", out)

    def test_fresh_fix_survives_listing_hiccup(self):
        # Absent from CurrentStorms but the fix is 3 h old: inside the grace
        # window, so a transient NHC listing hiccup can't drop a live storm.
        named = {"ep06": _storm("ep06", "EP", "SEVEN-E", "2026-06-12T19:00:00")}
        out = fp.retire_dissipated_named(named, {}, True, NOW)
        self.assertIn("ep06", out)

    def test_wp_storms_not_subject_to_nhc_list(self):
        # JTWC basin: CurrentStorms has no coverage; ACTIVE_WINDOW_HOURS
        # governs instead. A WP storm absent from the NHC list must stay.
        named = {"wp06": _storm("wp06", "WP", "JANGMI", "2026-06-11T00:00:00")}
        out = fp.retire_dissipated_named(named, {}, True, NOW)
        self.assertIn("wp06", out)

    def test_fetch_failure_never_mass_retires(self):
        named = {"ep03": _storm("ep03", "EP", "CRISTINA", "2026-06-11T00:00:00")}
        out = fp.retire_dissipated_named(named, {}, False, NOW)
        self.assertIn("ep03", out)

    def test_unparseable_fix_time_retires_when_absent(self):
        named = {"ep09": _storm("ep09", "EP", "GHOST", "")}
        out = fp.retire_dissipated_named(named, {}, True, NOW)
        self.assertEqual(out, {})

    def test_active_window_default_tightened(self):
        # 60 h let dissipated storms linger ~2.5 days; active storms get
        # fixes every ~6 h, so 24 h is four missed cycles.
        self.assertEqual(fp.ACTIVE_WINDOW_HOURS, 24.0)


if __name__ == "__main__":
    unittest.main()
