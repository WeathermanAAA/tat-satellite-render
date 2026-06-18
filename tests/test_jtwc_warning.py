"""Tests for jtwc_warning.parse_jtwc_warning + the cone integration."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import derived_cone as DC  # noqa: E402
import jtwc_warning as J  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "cyclolab")
WEB = open(os.path.join(FIX, "wp0726web.txt")).read()
PROG = open(os.path.join(FIX, "wp0726prog.txt")).read()
RADII = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                    "cyclolab_radii_jtwc_wpac_mean_2015.json")))

# Synthetic FINAL WARNING (no FORECASTS) -> single point.
FINAL = """WTPN31 PGTW 182100
MSGID/GENADMIN/JOINT TYPHOON WRNCEN PEARL HARBOR HI//
SUBJ/TROPICAL DEPRESSION 07W (SEVEN) WARNING NR 015//
REF/A/MSG/.../181351ZJUN2026//
RMKS/
   WARNING POSITION:
   181800Z --- NEAR 20.0N 130.0E
   MAX SUSTAINED WINDS - 025 KT
   THIS IS THE FINAL WARNING ON THIS SYSTEM.
NNNN
"""

# Synthetic Southern/dateline + month-boundary roll (Jan 30 issuance).
SOUTH_DL = """WTPN31 PGTW 302100
SUBJ/TROPICAL STORM 09S (NINE) WARNING NR 003//
REF/A/MSG/.../301351ZJAN2026//
RMKS/
   WARNING POSITION:
   302100Z --- NEAR 15.0S 179.5E
   MAX SUSTAINED WINDS - 045 KT
   FORECASTS:
   12 HRS, VALID AT:
   010900Z --- 16.0S 179.0W
   MAX SUSTAINED WINDS - 050 KT
NNNN
"""


class TestParseReal(unittest.TestCase):
    def test_real_07w(self):
        r = J.parse_jtwc_warning(WEB)
        self.assertEqual(r["warning_number"], 1)
        self.assertEqual(r["issued_utc"], "2026-06-18T21:00:00Z")
        self.assertEqual(r["name"], "SEVEN")
        self.assertEqual(r["sid_hint"], "07W")
        taus = [p["tau_h"] for p in r["points"]]
        self.assertEqual(taus, [0, 12, 24, 36, 48, 60, 72, 96, 120])
        p0 = r["points"][0]
        self.assertEqual((p0["lat"], p0["lon"]), (12.2, 145.1))
        self.assertEqual(p0["valid_utc"], "2026-06-18T18:00:00Z")
        self.assertEqual(p0["wind_kt"], 30.0)
        self.assertEqual(r["points"][-1]["valid_utc"], "2026-06-23T18:00:00Z")
        self.assertEqual(r["points"][1]["wind_kt"], 35.0)

    def test_cone_from_real(self):
        r = J.parse_jtwc_warning(WEB)
        nm = {int(k): float(v) for k, v in RADII["nm_values"].items()}
        cone = DC.derive_cone(r["points"], nm)
        self.assertGreater(len(cone), 10)             # multi-tau swept envelope
        self.assertEqual(cone[0], cone[-1])           # closed ring

    def test_cross_product_warning_number(self):
        # web + prog must agree (prog SUBJ wraps "WARNING \nNR 001").
        self.assertEqual(J.parse_warning_number(WEB), 1)
        self.assertEqual(J.parse_warning_number(PROG), 1)


class TestSigns(unittest.TestCase):
    def test_south_and_west_negative_and_dateline(self):
        r = J.parse_jtwc_warning(SOUTH_DL)
        p0, p1 = r["points"]
        self.assertEqual((p0["lat"], p0["lon"]), (-15.0, 179.5))   # S neg, E pos
        self.assertEqual((p1["lat"], p1["lon"]), (-16.0, -179.0))  # W neg (dateline)

    def test_month_year_roll(self):
        r = J.parse_jtwc_warning(SOUTH_DL)
        self.assertEqual(r["issued_utc"], "2026-01-30T21:00:00Z")
        self.assertEqual(r["points"][1]["valid_utc"], "2026-02-01T09:00:00Z")

    def test_dateline_cone_runs(self):
        r = J.parse_jtwc_warning(SOUTH_DL)
        nm = {int(k): float(v) for k, v in RADII["nm_values"].items()}
        cone = DC.derive_cone(r["points"], nm)   # antimeridian-safe, must not raise
        self.assertGreater(len(cone), 2)


class TestFinalWarning(unittest.TestCase):
    def test_single_point_full_circle(self):
        r = J.parse_jtwc_warning(FINAL)
        self.assertEqual(r["warning_number"], 15)
        self.assertEqual(len(r["points"]), 1)        # FINAL -> tau0 only
        nm = {int(k): float(v) for k, v in RADII["nm_values"].items()}
        cone = DC.derive_cone(r["points"], nm)
        self.assertEqual(len(cone), DC.CIRCLE_SAMPLES + 1)   # full closed circle


class TestMalformed(unittest.TestCase):
    def test_no_warning_nr(self):
        with self.assertRaises(J.JtwcParseError):
            J.parse_jtwc_warning("WTPN31 PGTW 182100\nSUBJ/SOMETHING//\n")

    def test_empty(self):
        with self.assertRaises(J.JtwcParseError):
            J.parse_jtwc_warning("")

    def test_parse_warning_number_none_on_garbage(self):
        self.assertIsNone(J.parse_warning_number("an error page"))


if __name__ == "__main__":
    unittest.main()
