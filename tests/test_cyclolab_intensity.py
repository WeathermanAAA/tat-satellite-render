"""INTENSITY CONE registry + envelope tests (CYCLOLAB_DESIGN §8.6).

The registry literals are pinned VALUE BY VALUE against the design's
sourced table (the never-invent contract - every number was text-
verified against the downloaded source document, incl. the WP 36 h
value 12.9 that an adversarial pass rescued from a 13.9 transposition).
A drift in any number is a data regression, not a tweak.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cyclolab_intensity as ci  # noqa: E402


class TestRegistryLiterals(unittest.TestCase):
    def test_al_mae_and_bias_pinned(self):
        e = ci.basin_entry("AL")
        self.assertEqual(e["method_version"], "nhc-ofcl-5yr-2020-2024")
        self.assertEqual(e["mae_kt"], {"12": 5.1, "24": 7.3, "36": 8.6,
                                       "48": 10.0, "60": 10.5, "72": 10.9,
                                       "96": 12.4, "120": 13.6})
        self.assertEqual(e["bias_kt"], {"12": 0.5, "24": 0.5, "36": 0.3,
                                        "48": 0.3, "60": 0.3, "72": 0.5,
                                        "96": 0.1, "120": -1.6})
        self.assertTrue(e["asymmetric_supported"])
        self.assertEqual(e["source_md5"],
                         "b48e4759ea3153f74314467d0da72567")

    def test_ep_mae_and_bias_pinned(self):
        e = ci.basin_entry("EP")
        self.assertEqual(e["mae_kt"], {"12": 5.7, "24": 8.9, "36": 10.8,
                                       "48": 12.9, "60": 14.4, "72": 15.5,
                                       "96": 17.0, "120": 18.6})
        self.assertEqual(e["bias_kt"]["120"], -5.6)   # the long-range
        self.assertEqual(e["bias_kt"]["96"], -1.8)    # under-forecast
        self.assertTrue(e["asymmetric_supported"])

    def test_cp_stale_window_no_60h_no_bias(self):
        e = ci.basin_entry("CP")
        self.assertEqual(e["method_version"], "cphc-5yr-2015-2019")
        self.assertEqual(e["mae_kt"], {"12": 5.9, "24": 9.0, "36": 11.3,
                                       "48": 12.6, "72": 14.5, "96": 16.5,
                                       "120": 17.4})
        self.assertNotIn("60", e["mae_kt"])           # honestly absent
        self.assertIsNone(e["bias_kt"])
        self.assertFalse(e["asymmetric_supported"])
        self.assertIn("2015-2019", e["staleness_note"])

    def test_wp_atcr2020_pinned_including_the_corrected_36h(self):
        e = ci.basin_entry("WP")
        self.assertEqual(e["method_version"], "jtwc-wpac-atcr2020-5yr")
        self.assertEqual(e["mae_kt"]["36"], 12.9)     # NOT 13.9
        self.assertEqual(e["mae_kt"], {"12": 7.7, "24": 10.7, "36": 12.9,
                                       "48": 15.0, "72": 17.7, "96": 18.9,
                                       "120": 20.3})
        self.assertIsNone(e["bias_kt"])
        self.assertFalse(e["asymmetric_supported"])
        self.assertIn("vintage", " ".join(e.keys()))  # vintage_note present

    def test_honesty_guard_unknown_basin_is_none(self):
        self.assertIsNone(ci.basin_entry("IO"))
        self.assertIsNone(ci.basin_entry(""))
        self.assertIsNone(ci.basin_entry(None))


class TestEnvelopeMath(unittest.TestCase):
    def test_published_taus_pass_through(self):
        e = ci.basin_entry("AL")
        self.assertEqual(ci.mae_at(e, 48), 10.0)
        self.assertEqual(ci.mae_at(e, 120), 13.6)

    def test_tau_zero_anchors_at_zero(self):
        self.assertEqual(ci.mae_at(ci.basin_entry("EP"), 0), 0.0)

    def test_missing_60h_interpolates_linearly(self):
        # CP has no 60 h column: (12.6 + 14.5) / 2 = 13.55
        self.assertAlmostEqual(ci.mae_at(ci.basin_entry("CP"), 60), 13.55)
        # WP likewise: (15.0 + 17.7) / 2 = 13.35... no - midpoint 16.35
        self.assertAlmostEqual(ci.mae_at(ci.basin_entry("WP"), 60), 16.35)

    def test_sub_12h_interpolates_from_zero_anchor(self):
        # 6 h on AL: halfway from the 0-anchor to 5.1 = 2.55
        self.assertAlmostEqual(ci.mae_at(ci.basin_entry("AL"), 6), 2.55)

    def test_beyond_last_tau_clamps(self):
        self.assertEqual(ci.mae_at(ci.basin_entry("AL"), 144), 13.6)

    def test_envelope_rows_and_floor_clamp(self):
        pts = [{"tau_h": 0, "intensity_kt": 30},
               {"tau_h": 48, "intensity_kt": 35},
               {"tau_h": 120, "intensity_kt": 10},
               {"tau_h": 72, "intensity_kt": None}]
        rows = ci.envelope(pts, ci.basin_entry("EP"))
        self.assertEqual([r["tau_h"] for r in rows], [0, 48, 120])
        self.assertEqual(rows[0]["upper"], 30)         # MAE 0 at analysis
        self.assertEqual(rows[0]["lower"], 30)
        self.assertEqual(rows[1]["upper"], 35 + 12.9)
        self.assertEqual(rows[1]["lower"], 35 - 12.9)
        self.assertEqual(rows[2]["lower"], 0.0)        # 10 - 18.6 clamps
        self.assertEqual(rows[2]["upper"], 10 + 18.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
